import torch
from batchgenerators.dataloading import MultiThreadedAugmenter, SingleThreadedAugmenter
from batchgenerators.transforms import Compose, RenameTransform, GammaTransform, SpatialTransform
from batchgenerators.transforms import DataChannelSelectionTransform, SegChannelSelectionTransform
from batchgenerators.transforms import MirrorTransform, NumpyToTensor
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, AppendChannelsTransform
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.data_augmentation.custom_transforms import Convert3DTo2DTransform, Convert2DTo3DTransform, \
    MaskTransform
from nnunet.training.data_augmentation.default_data_augmentation import default_3D_augmentation_params
from nnunet.training.data_augmentation.pyramid_augmentations import MoveSegAsOneHotToData, \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform, ApplyRandomBinaryOperatorTransform
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer

try:
    from apex import amp
except ImportError:
    amp = None


def get_default_augmentation_withEDT(dataloader_train, dataloader_val, patch_size, idx_of_edts,
                                     params=default_3D_augmentation_params, border_val_seg=-1, pin_memory=True,
                                     seeds_train=None, seeds_val=None):
    tr_transforms = []

    if params.get("selected_data_channels") is not None:
        tr_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))

    if params.get("selected_seg_channels") is not None:
        tr_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

    # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
    if params.get("dummy_2D") is not None and params.get("dummy_2D"):
        tr_transforms.append(Convert3DTo2DTransform())

    tr_transforms.append(SpatialTransform(
        patch_size, patch_center_dist_from_border=None, do_elastic_deform=params.get("do_elastic"),
        alpha=params.get("elastic_deform_alpha"), sigma=params.get("elastic_deform_sigma"),
        do_rotation=params.get("do_rotation"), angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
        angle_z=params.get("rotation_z"), do_scale=params.get("do_scaling"), scale=params.get("scale_range"),
        border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=3, border_mode_seg="constant",
        border_cval_seg=border_val_seg,
        order_seg=1, random_crop=params.get("random_crop"), p_el_per_sample=params.get("p_eldef"),
        p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot")
    ))
    if params.get("dummy_2D") is not None and params.get("dummy_2D"):
        tr_transforms.append(Convert2DTo3DTransform())

    """
    ##############################################################
    ##############################################################
    Here we insert moving the EDT to a different key so that it does not get intensity transformed
    ##############################################################
    ##############################################################
    """
    tr_transforms.append(AppendChannelsTransform("data", "bound", idx_of_edts, remove_from_input=True))

    if params.get("do_gamma"):
        tr_transforms.append(
            GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"),
                           p_per_sample=params["p_gamma"]))

    tr_transforms.append(MirrorTransform(params.get("mirror_axes")))

    if params.get("mask_was_used_for_normalization") is not None:
        mask_was_used_for_normalization = params.get("mask_was_used_for_normalization")
        tr_transforms.append(MaskTransform(mask_was_used_for_normalization, mask_idx_in_seg=0, set_outside_to=0))

    tr_transforms.append(RemoveLabelTransform(-1, 0))

    if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
        tr_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))
        if params.get("advanced_pyramid_augmentations") and not None and params.get("advanced_pyramid_augmentations"):
            tr_transforms.append(ApplyRandomBinaryOperatorTransform(
                channel_idx=list(range(-len(params.get("all_segmentation_labels")), 0)),
                p_per_sample=0.4,
                key="data",
                strel_size=(1, 8)))
            tr_transforms.append(RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                channel_idx=list(range(-len(params.get("all_segmentation_labels")), 0)),
                key="data",
                p_per_sample=0.2,
                fill_with_other_class_p=0.0,
                dont_do_if_covers_more_than_X_percent=0.15))

    tr_transforms.append(RenameTransform('seg', 'target', True))
    tr_transforms.append(NumpyToTensor(['data', 'target', 'bound'], 'float'))
    tr_transforms = Compose(tr_transforms)

    batchgenerator_train = MultiThreadedAugmenter(dataloader_train, tr_transforms, params.get('num_threads'),
                                                  params.get("num_cached_per_thread"), seeds=seeds_train,
                                                  pin_memory=pin_memory)

    val_transforms = []
    val_transforms.append(RemoveLabelTransform(-1, 0))
    if params.get("selected_data_channels") is not None:
        val_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))
    if params.get("selected_seg_channels") is not None:
        val_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

    """
    ##############################################################
    ##############################################################
    Here we insert moving the EDT to a different key
    ##############################################################
    ##############################################################
    """
    val_transforms.append(AppendChannelsTransform("data", "bound", idx_of_edts, remove_from_input=True))

    if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
        val_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))

    val_transforms.append(RenameTransform('seg', 'target', True))
    val_transforms.append(NumpyToTensor(['data', 'target', 'bound'], 'float'))
    val_transforms = Compose(val_transforms)

    batchgenerator_val = MultiThreadedAugmenter(dataloader_val, val_transforms, max(params.get('num_threads') // 2, 1),
                                                params.get("num_cached_per_thread"), seeds=seeds_val,
                                                pin_memory=pin_memory)
    return batchgenerator_train, batchgenerator_val


class nnUNetTrainerEDT(nnUNetTrainer):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.loss = None # insert your loss here

    def process_plans(self, plans):
        super().process_plans(plans)
        self.num_modalities_with_edt = self.num_input_channels + self.num_classes

    def initialize(self, training=True, force_load_plans=False):
        """
        we need to swap out get_default_augmentation so that we can move the EDTs around (get_default_augmentation_withEDT)
        """

        maybe_mkdir_p(self.output_folder)

        if force_load_plans or (self.plans is None):
            self.load_plans_file()

        self.process_plans(self.plans)

        self.setup_DA_params()

        self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                  "_stage%d" % self.stage)
        if training:
            self.dl_tr, self.dl_val = self.get_basic_generators()
            if self.unpack_data:
                self.print_to_log_file("unpacking dataset")
                unpack_dataset(self.folder_with_preprocessed_data)
                self.print_to_log_file("done")
            else:
                self.print_to_log_file(
                    "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                    "will wait all winter for your model to finish!")
            idx_of_edts = list(range(self.num_input_channels, self.num_modalities_with_edt))
            self.tr_gen, self.val_gen = get_default_augmentation_withEDT(self.dl_tr, self.dl_val,
                                                                 self.data_aug_params[
                                                                     'patch_size_for_spatialtransform'],
                                                                 idx_of_edts,
                                                                 self.data_aug_params)
            self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                   also_print_to_console=False)
            self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                   also_print_to_console=False)
        else:
            pass
        self.initialize_network_optimizer_and_scheduler()
        # assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        self.was_initialized = True

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        here we add the boundary to the loss function

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        import IPython;IPython.embed()
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']
        bound = data_dict['bound']

        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data).float()
        if not isinstance(target, torch.Tensor):
            target = torch.from_numpy(target).float()
        if not isinstance(bound, torch.Tensor):
            bound = torch.from_numpy(bound).float()

        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        bound = bound.cuda(non_blocking=True)

        self.optimizer.zero_grad()

        output = self.network(data)
        l = self.loss(output, target, bound)

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        if do_backprop:
            if not self.fp16 or amp is None:
                l.backward()
            else:
                with amp.scale_loss(l, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            self.optimizer.step()

        return l
