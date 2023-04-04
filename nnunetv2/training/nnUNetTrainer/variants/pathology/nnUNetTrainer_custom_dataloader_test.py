import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet, PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_batchnorm
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0, InitWeights_He
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn

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
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        ### TEST FOR 1 EPOCH - remove for default 
        self.num_epochs = 1

        ### DUMMY BATCH - Copied this from benchmark trainer without dataloading
        self._set_batch_size_and_oversample()
        num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                          self.dataset_json)
        patch_size = self.configuration_manager.patch_size
        dummy_data = torch.rand((self.batch_size, num_input_channels, *patch_size), device=self.device)
        dummy_target = [
            torch.round(
                torch.rand((self.batch_size, 1, *[int(i * j) for i, j in zip(patch_size, k)]), device=self.device) *
                max(self.label_manager.all_labels)
            ) for k in self._get_deep_supervision_scales()]
        self.dummy_batch = {'data': dummy_data, 'target': dummy_target} # only thing we need!

    ### GET DATALOADERS - as generator objects
    def get_dataloaders(self):
        return None, None

        
    def run_training(self):
        try:
            self.on_train_start()

            for epoch in range(self.current_epoch, self.num_epochs):
                self.on_epoch_start()

                self.on_train_epoch_start()
                train_outputs = []
                for batch_id in range(self.num_iterations_per_epoch):
                    train_outputs.append(self.train_step(self.dummy_batch)) ### REPLACE self.dummy_batch with next(self.dataloader_train)
                self.on_train_epoch_end(train_outputs)

                with torch.no_grad():
                    self.on_validation_epoch_start()
                    val_outputs = []
                    for batch_id in range(self.num_val_iterations_per_epoch):
                        val_outputs.append(self.validation_step(self.dummy_batch)) ### REPLACE self.dummy_batch with next(self.dataloader_val)
                    self.on_validation_epoch_end(val_outputs)

                self.on_epoch_end()

            self.on_train_end()
        except RuntimeError:
            self.crashed_with_runtime_error = True

    ### BATCH NORM ###
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