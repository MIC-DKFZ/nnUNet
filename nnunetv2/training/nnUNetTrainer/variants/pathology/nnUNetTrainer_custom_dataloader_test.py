import os
import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from batchgenerators.utilities.file_and_folder_operations import join, load_json

# Whole slide data dataloader
if os.name == 'nt':
    os.add_dll_directory(r"C:\Program Files\openslide\bin") # windows
from wholeslidedata.iterators import create_batch_iterator
from wholeslidedata.iterators.batchiterator import BatchIterator
from copy import deepcopy

# for network building
from torch import nn
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet, PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_batchnorm
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0, InitWeights_He
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


# for def on_train_start:
from batchgenerators.utilities.file_and_folder_operations import join, save_json, maybe_mkdir_p, isfile
from torch import distributed as dist
from nnunetv2.training.dataloading.utils import unpack_dataset
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.helpers import empty_cache

#for dummy init
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
# for super init
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from datetime import datetime
from torch.cuda.amp import GradScaler
import inspect
from torch.cuda import device_count
from nnunetv2.paths import nnUNet_results, nnUNet_preprocessed

### TODOS
# DONE: rgb_to_0_1 via dataset.json
# DONE: BN by rewriting build_network_architecture function on bottom of this file
# TODO: Data augmentation
# TODO: Batch norm minimum 8 instead of 2
# TODO: Test with simple dataloader instead of dumy data
# TODO: Use plans file for dataloader initialization

class nnUNetTrainer_custom_dataloader_test(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        # super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
### ORIGINAL SUPER INIT (but removing the preprocessed_dataset_folder_base parts)
        self.is_ddp = dist.is_available() and dist.is_initialized()
        self.local_rank = 0 if not self.is_ddp else dist.get_rank()

        self.device = device

        # print what device we are using
        if self.is_ddp:  # implicitly it's clear that we use cuda in this case
            print(f"I am local rank {self.local_rank}. {device_count()} GPUs are available. The world size is "
                  f"{dist.get_world_size()}."
                  f"Setting device to {self.device}")
            self.device = torch.device(type='cuda', index=self.local_rank)
        else:
            if self.device.type == 'cuda':
                # we might want to let the user pick this but for now please pick the correct GPU with CUDA_VISIBLE_DEVICES=X
                self.device = torch.device(type='cuda', index=0)
            print(f"Using device: {self.device}")

        # loading and saving this class for continuing from checkpoint should not happen based on pickling. This
        # would also pickle the network etc. Bad, bad. Instead we just reinstantiate and then load the checkpoint we
        # need. So let's save the init args
        self.my_init_kwargs = {}
        for k in inspect.signature(self.__init__).parameters.keys():
            self.my_init_kwargs[k] = locals()[k]

        ###  Saving all the init args into class variables for later access
        self.plans_manager = PlansManager(plans)
        self.configuration_manager = self.plans_manager.get_configuration(configuration)
        self.configuration_name = configuration
        self.dataset_json = dataset_json
        self.fold = fold
        self.unpack_dataset = unpack_dataset

        ### Setting all the folder names. We need to make sure things don't crash in case we are just running
        # inference and some of the folders may not be defined!
        # self.preprocessed_dataset_folder_base = join(nnUNet_preprocessed, self.plans_manager.dataset_name) \
        #     if nnUNet_preprocessed is not None else None
        self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name,
                                       self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
            if nnUNet_results is not None else None
        self.output_folder = join(self.output_folder_base, f'fold_{fold}')

        # self.preprocessed_dataset_folder = join(self.preprocessed_dataset_folder_base,
        #                                         self.configuration_manager.data_identifier)
        # unlike the previous nnunet folder_with_segs_from_previous_stage is now part of the plans. For now it has to
        # be a different configuration in the same plans
        # IMPORTANT! the mapping must be bijective, so lowres must point to fullres and vice versa (using
        # "previous_stage" and "next_stage"). Otherwise it won't work!
        self.is_cascaded = self.configuration_manager.previous_stage_name is not None
        self.folder_with_segs_from_previous_stage = \
            join(nnUNet_results, self.plans_manager.dataset_name,
                 self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" +
                 self.configuration_manager.previous_stage_name, 'predicted_next_stage', self.configuration_name) \
                if self.is_cascaded else None

        ### Some hyperparameters for you to fiddle with
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.33
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.num_epochs = 1000
        self.current_epoch = 0

        ### Dealing with labels/regions
        self.label_manager = self.plans_manager.get_label_manager(dataset_json)
        # labels can either be a list of int (regular training) or a list of tuples of int (region-based training)
        # needed for predictions. We do sigmoid in case of (overlapping) regions

        self.num_input_channels = None  # -> self.initialize()
        self.network = None  # -> self._get_network()
        self.optimizer = self.lr_scheduler = None  # -> self.initialize
        self.grad_scaler = GradScaler() if self.device.type == 'cuda' else None
        self.loss = None  # -> self.initialize

        ### Simple logging. Don't take that away from me!
        # initialize log file. This is just our log for the print statements etc. Not to be confused with lightning
        # logging
        timestamp = datetime.now()
        maybe_mkdir_p(self.output_folder)
        self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                             (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                              timestamp.second))
        self.logger = nnUNetLogger()

        ### placeholders
        self.dataloader_train = self.dataloader_val = None  # see on_train_start

        ### initializing stuff for remembering things and such
        self._best_ema = None

        ### inference things
        self.inference_allowed_mirroring_axes = None  # this variable is set in
        # self.configure_rotation_dummyDA_mirroring_and_inital_patch_size and will be saved in checkpoints

        ### checkpoint saving stuff
        self.save_every = 50
        self.disable_checkpointing = False

        ## DDP batch size and oversampling can differ between workers and needs adaptation
        # we need to change the batch size in DDP because we don't use any of those distributed samplers
        self._set_batch_size_and_oversample()

        self.was_initialized = False

        self.print_to_log_file("\n#######################################################################\n"
                               "Please cite the following paper when using nnU-Net:\n"
                               "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
                               "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
                               "Nature methods, 18(2), 203-211.\n"
                               "#######################################################################\n",
                               also_print_to_console=True, add_timestamp=False)

### END ORIGINAL SUPER INIT

### INIT - DUMMY BATCH, TEST FOR 2 EPOCHS 
        self.num_epochs = 2

        ### DUMMY BATCH - Copied this from benchmark trainer without dataloading
        self._set_batch_size_and_oversample()
        num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                          self.dataset_json)
        patch_size = self.configuration_manager.patch_size
        
        print('Making dummy data')
        print('\tpatch size:', patch_size)
        print('\tself.label_manager.all_labels:', self.label_manager.all_labels)
        print('\tself._get_deep_supervision_scales()', self._get_deep_supervision_scales())
        # dummy_data = torch.rand((self.batch_size, num_input_channels, *patch_size), device=self.device)
        # dummy_target = [
        #     torch.round(
        #         torch.rand((self.batch_size, 1, *[int(i * j) for i, j in zip(patch_size, k)]), device=self.device) *
        #         max(self.label_manager.all_labels)
        dummy_data = torch.rand((self.configuration_manager.batch_size, num_input_channels, *patch_size), device=self.device)
        dummy_target = [
            torch.round(
                torch.rand((self.configuration_manager.batch_size, 1, *[int(i * j) for i, j in zip(patch_size, k)]), device=self.device) *
                max(self.label_manager.all_labels)
            ) for k in self._get_deep_supervision_scales()]
        self.dummy_batch = {'data': dummy_data, 'target': dummy_target} # only thing we need!

        


### TODO: fix this        
        # self.yaml_source = "C:\Users\joeyspronck\Documents\Github\nnUNet_v2\test_code\example_files.json"
        # self.spacing
###


    def do_split(self):
        if isfile(join(nnUNet_preprocessed, self.plans_manager.dataset_name, 'splits_final.json')):
            print('Found splits_final.json')
        else:
            pass


### GET DATALOADERS - as generator objects
    def get_dataloaders(self):
        
        self.do_split()
        
        
        # return None, None
        print('[Getting WSD dataloaders]')
        dataloader_template = load_json(join(nnUNet_preprocessed, self.plans_manager.dataset_name, 'wsd_dataloader_template.json'))
        files = load_json(join(nnUNet_preprocessed, self.plans_manager.dataset_name, 'files.json')) 
        copy_path = 'C:\\Users\\joeyspronck\\Documents\\Github\\nnUNet_v2\\data\\nnUNet_wsd'
        labels = self.dataset_json['labels']
        label_sample_weights = {
            'invasive tumor': 0.5,
            'tumor-associated stroma': 0.5
        }
        spacing = 0.5
        patch_size = self.configuration_manager.patch_size
        patch_shape = patch_size + [len(self.configuration_manager.normalization_schemes)]
        batch_size = self.configuration_manager.batch_size
        ds_scales = self._get_deep_supervision_scales()
        ds_shapes = [list(np.round([int(i * j) for i, j in zip(patch_size, k)])) for k in ds_scales]
        extra_ds_sizes = [ds_shape for ds_shape in ds_shapes[1:]]
        extra_ds_shapes = tuple([tuple([batch_size]+ds_shape) for ds_shape in ds_shapes[1:]])
        device = self.device

        fill_template = dataloader_template['wholeslidedata']['default']

        fill_template['yaml_source'] = files
        fill_template['labels'] = labels
        fill_template['batch_shape']['batch_size'] = batch_size
        fill_template['batch_shape']['spacing'] = spacing
        fill_template['batch_shape']['shape'] = patch_shape
        fill_template['label_sampler']['labels'] = label_sample_weights
        fill_template['batch_callbacks'][-1]['sizes'] = extra_ds_sizes
        fill_template['dataset']['copy_path'] = copy_path

        # TODO: not here
        class WholeSlidePlainnnUnetBatchIterator(BatchIterator):
            def __next__(self):
                x_batch, y_batch, *extras, _ = super().__next__()
                data = torch.FloatTensor(x_batch.transpose(0,3,1,2) /255.).to(device)
                target = [torch.FloatTensor(np.expand_dims(y_batch, 1)).to(device)] + [
                    torch.FloatTensor(np.expand_dims(extra, 1)).to(device) 
                    for extra in extras]         
                return {'data': data, 'target': target}
            
        cpus = 4
        print('[Creating batch iterators]')
        tiger_batch_iterator = create_batch_iterator(mode="training", 
                                                    user_config= deepcopy(dataloader_template), 
                                                    cpus=cpus, 
                                                    buffer_dtype='uint8',
                                                    context='spawn' if os.name == 'nt' else 'fork',
                                                    extras_shapes = extra_ds_shapes,
                                                    iterator_class=WholeSlidePlainnnUnetBatchIterator
                                                    )

        print('[Returing batch iterators]')
        return tiger_batch_iterator, tiger_batch_iterator


### build_network_architecture changing to BATCH NORM ###
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        num_stages = len(configuration_manager.conv_kernel_sizes)

        dim = len(configuration_manager.conv_kernel_sizes[0])
        conv_op = convert_dim_to_conv_op(dim)

        label_manager = plans_manager.get_label_manager(dataset_json)

        segmentation_network_class_name = configuration_manager.UNet_class_name
        mapping = {
            'PlainConvUNet': PlainConvUNet,
            'ResidualEncoderUNet': ResidualEncoderUNet
        }
        kwargs = {
            'PlainConvUNet': {
                'conv_bias': True,
                'norm_op': get_matching_batchnorm(conv_op),
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None, 'dropout_op_kwargs': None,
                'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
            },
            'ResidualEncoderUNet': {
                'conv_bias': True,
                'norm_op': get_matching_batchnorm(conv_op),
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None, 'dropout_op_kwargs': None,
                'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
            }
        }
        assert segmentation_network_class_name in mapping.keys(), 'The network architecture specified by the plans file ' \
                                                                  'is non-standard (maybe your own?). Yo\'ll have to dive ' \
                                                                  'into either this ' \
                                                                  'function (get_network_from_plans) or ' \
                                                                  'the init of your nnUNetModule to accomodate that.'
        network_class = mapping[segmentation_network_class_name]

        conv_or_blocks_per_stage = {
            'n_conv_per_stage'
            if network_class != ResidualEncoderUNet else 'n_blocks_per_stage': configuration_manager.n_conv_per_stage_encoder,
            'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
        }
        # network class name!!
        model = network_class(
            input_channels=num_input_channels,
            n_stages=num_stages,
            features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                    configuration_manager.unet_max_num_features) for i in range(num_stages)],
            conv_op=conv_op,
            kernel_sizes=configuration_manager.conv_kernel_sizes,
            strides=configuration_manager.pool_op_kernel_sizes,
            num_classes=label_manager.num_segmentation_heads,
            deep_supervision=enable_deep_supervision,
            **conv_or_blocks_per_stage,
            **kwargs[segmentation_network_class_name]
        )
        model.apply(InitWeights_He(1e-2))
        if network_class == ResidualEncoderUNet:
            model.apply(init_last_bn_before_add_to_0)
        return model
    

### on_train_start outcomment fingerprint ###
    def on_train_start(self):
        if not self.was_initialized:
            self.initialize()

        maybe_mkdir_p(self.output_folder)

        # make sure deep supervision is on in the network
        self.set_deep_supervision_enabled(True)

        self.print_plans()
        empty_cache(self.device)

        # maybe unpack
        # if self.unpack_dataset and self.local_rank == 0:
        #     self.print_to_log_file('unpacking dataset...')
        #     unpack_dataset(self.preprocessed_dataset_folder, unpack_segmentation=True, overwrite_existing=False,
        #                    num_processes=max(1, round(get_allowed_n_proc_DA() // 2)))
        #     self.print_to_log_file('unpacking done...')
        print('No unpacking etc. WholeSlideData will copy data locally if dataset copy path is specified in dataloader config')

        if self.is_ddp:
            dist.barrier()

        # dataloaders must be instantiated here because they need access to the training data which may not be present
        # when doing inference
        self.dataloader_train, self.dataloader_val = self.get_dataloaders()

        # copy plans and dataset.json so that they can be used for restoring everything we need for inference
        save_json(self.plans_manager.plans, join(self.output_folder_base, 'plans.json'), sort_keys=False)
        save_json(self.dataset_json, join(self.output_folder_base, 'dataset.json'), sort_keys=False)

        # we don't really need the fingerprint but its still handy to have it with the others
        # shutil.copy(join(self.preprocessed_dataset_folder_base, 'dataset_fingerprint.json'),
        #             join(self.output_folder_base, 'dataset_fingerprint.json'))

        # produces a pdf in output folder
        self.plot_network_architecture()

        self._save_debug_information()

        # print(f"batch size: {self.batch_size}")
        # print(f"oversample: {self.oversample_foreground_percent}")

### RUN TRAINING        
    def run_training(self):
        try:
            self.on_train_start()

            for epoch in range(self.current_epoch, self.num_epochs):
                self.on_epoch_start()

                self.on_train_epoch_start()
                train_outputs = []
                # for batch_id in range(self.num_iterations_per_epoch):
                for batch_id in range(4):
                    train_outputs.append(self.train_step(next(self.dataloader_train))) ### REPLACE self.dummy_batch with next(self.dataloader_train)
                    print('done batch')
                self.on_train_epoch_end(train_outputs)

                with torch.no_grad():
                    self.on_validation_epoch_start()
                    val_outputs = []
                    for batch_id in range(self.num_val_iterations_per_epoch):
                        val_outputs.append(self.validation_step(next(self.dataloader_val))) ### REPLACE self.dummy_batch with next(self.dataloader_val)
                    self.on_validation_epoch_end(val_outputs)

                self.on_epoch_end()
                print('done epoch')
            self.on_train_end()
            print('done training')
        except RuntimeError as e:
            print(e)
            self.crashed_with_runtime_error = True
