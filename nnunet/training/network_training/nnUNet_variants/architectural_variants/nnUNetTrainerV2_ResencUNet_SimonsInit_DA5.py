from typing import List

import numpy as np
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.channel_selection_transforms import SegChannelSelectionTransform
from batchgenerators.transforms.color_transforms import BrightnessTransform, ContrastAugmentationTransform, \
    GammaTransform
from batchgenerators.transforms.local_transforms import BrightnessGradientAdditiveTransform, LocalGammaTransform
from batchgenerators.transforms.noise_transforms import BlankRectangleTransform, MedianFilterTransform, \
    SharpeningTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import Rot90Transform, TransposeAxesTransform, MirrorTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor, \
    OneOfTransform
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.custom_transforms import Convert3DTo2DTransform, Convert2DTo3DTransform, \
    MaskTransform, ConvertSegmentationToRegionsTransform
from nnunet.training.data_augmentation.default_data_augmentation import get_patch_size
from nnunet.training.data_augmentation.downsampling import DownsampleSegForDSTransform2
from nnunet.training.data_augmentation.pyramid_augmentations import MoveSegAsOneHotToData, \
    ApplyRandomBinaryOperatorTransform, RemoveRandomConnectedComponentFromOneHotEncodingTransform
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_ResencUNet_SimonsInit import \
    nnUNetTrainerV2_ResencUNet_SimonsInit
from nnunet.utilities.set_n_proc_DA import get_allowed_n_proc_DA
from torch import nn


class nnUNetTrainerV2_ResencUNet_SimonsInit_DA5(nnUNetTrainerV2_ResencUNet_SimonsInit):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.do_mirroring = True
        self.mirror_axes = None
        proc = get_allowed_n_proc_DA()
        self.num_proc_DA = proc if proc is not None else 12
        self.num_cached = 4
        self.regions_class_order = self.regions = None

    def setup_DA_params(self):
        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes[1:]), axis=0))[:-1]

        self.data_aug_params = dict()
        self.data_aug_params['scale_range'] = (0.7, 1.43)

        # we need this because this is adapted in the cascade
        self.data_aug_params['selected_seg_channels'] = None
        self.data_aug_params["move_last_seg_chanel_to_data"] = False

        if self.threeD:
            if self.do_mirroring:
                self.mirror_axes = (0, 1, 2)
                self.data_aug_params['do_mirror'] = True  # needed for inference
                self.data_aug_params['mirror_axes'] = (0, 1, 2)  # needed for inference
            else:
                self.data_aug_params['mirror_axes'] = tuple()
                self.data_aug_params['do_mirror'] = False

            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)

            if self.do_dummy_2D_aug:
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["dummy_2D"] = True
                self.data_aug_params["rotation_x"] = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
        else:
            if self.do_mirroring:
                self.mirror_axes = (0, 1)
                self.data_aug_params['mirror_axes'] = (0, 1)  # needed for inference
                self.data_aug_params['do_mirror'] = True  # needed for inference
            else:
                self.data_aug_params['mirror_axes'] = tuple()
                self.data_aug_params['do_mirror'] = False  # needed for inference

            self.do_dummy_2D_aug = False

            self.data_aug_params['rotation_x'] = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-0. / 360 * 2. * np.pi, 0. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-0. / 360 * 2. * np.pi, 0. / 360 * 2. * np.pi)

        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])

    def get_train_transforms(self) -> List[AbstractTransform]:
        # used for transpost and rot90
        matching_axes = np.array([sum([i == j for j in self.patch_size]) for i in self.patch_size])
        valid_axes = list(np.where(matching_axes == np.max(matching_axes))[0])

        tr_transforms = []

        if self.data_aug_params['selected_seg_channels'] is not None:
            tr_transforms.append(SegChannelSelectionTransform(self.data_aug_params['selected_seg_channels']))

        if self.do_dummy_2D_aug:
            ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = self.patch_size[1:]
        else:
            patch_size_spatial = self.patch_size
            ignore_axes = None

        tr_transforms.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=None,
                do_elastic_deform=False,
                do_rotation=True,
                angle_x=self.data_aug_params["rotation_x"],
                angle_y=self.data_aug_params["rotation_y"],
                angle_z=self.data_aug_params["rotation_z"],
                p_rot_per_axis=0.5,
                do_scale=True,
                scale=self.data_aug_params['scale_range'],
                border_mode_data="constant",
                border_cval_data=0,
                order_data=3,
                border_mode_seg="constant",
                border_cval_seg=-1,
                order_seg=1,
                random_crop=False,
                p_el_per_sample=0.2,
                p_scale_per_sample=0.2,
                p_rot_per_sample=0.4,
                independent_scale_for_each_axis=True,
            )
        )

        if self.do_dummy_2D_aug:
            tr_transforms.append(Convert2DTo3DTransform())

        if np.any(matching_axes > 1):
            tr_transforms.append(
                Rot90Transform(
                    (0, 1, 2, 3), axes=valid_axes, data_key='data', label_key='seg', p_per_sample=0.5
                ),
            )

        if np.any(matching_axes > 1):
            tr_transforms.append(
                TransposeAxesTransform(valid_axes, data_key='data', label_key='seg', p_per_sample=0.5)
            )

        tr_transforms.append(OneOfTransform([
            MedianFilterTransform(
                (2, 8),
                same_for_each_channel=False,
                p_per_sample=0.2,
                p_per_channel=0.5
            ),
            GaussianBlurTransform((0.3, 1.5),
                                  different_sigma_per_channel=True,
                                  p_per_sample=0.2,
                                  p_per_channel=0.5)
        ]))

        tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))

        tr_transforms.append(BrightnessTransform(0,
                                                 0.5,
                                                 per_channel=True,
                                                 p_per_sample=0.1,
                                                 p_per_channel=0.5
                                                 )
                             )

        tr_transforms.append(OneOfTransform(
            [
                ContrastAugmentationTransform(
                    contrast_range=(0.5, 2),
                    preserve_range=True,
                    per_channel=True,
                    data_key='data',
                    p_per_sample=0.2,
                    p_per_channel=0.5
                ),
                ContrastAugmentationTransform(
                    contrast_range=(0.5, 2),
                    preserve_range=False,
                    per_channel=True,
                    data_key='data',
                    p_per_sample=0.2,
                    p_per_channel=0.5
                ),
            ]
        ))

        tr_transforms.append(
            SimulateLowResolutionTransform(zoom_range=(0.25, 1),
                                           per_channel=True,
                                           p_per_channel=0.5,
                                           order_downsample=0,
                                           order_upsample=3,
                                           p_per_sample=0.15,
                                           ignore_axes=ignore_axes
                                           )
        )

        tr_transforms.append(
            GammaTransform((0.7, 1.5), invert_image=True, per_channel=True, retain_stats=True, p_per_sample=0.1))
        tr_transforms.append(
            GammaTransform((0.7, 1.5), invert_image=True, per_channel=True, retain_stats=True, p_per_sample=0.1))

        if self.do_mirroring:
            tr_transforms.append(MirrorTransform(self.mirror_axes))

        tr_transforms.append(
            BlankRectangleTransform([[max(1, p // 10), p // 3] for p in self.patch_size],
                                    rectangle_value=np.mean,
                                    num_rectangles=(1, 5),
                                    force_square=False,
                                    p_per_sample=0.4,
                                    p_per_channel=0.5
                                    )
        )

        tr_transforms.append(
            BrightnessGradientAdditiveTransform(
                lambda x, y: np.exp(np.random.uniform(np.log(x[y] // 6), np.log(x[y]))),
                (-0.5, 1.5),
                max_strength=lambda x, y: np.random.uniform(-5, -1) if np.random.uniform() < 0.5 else np.random.uniform(
                    1, 5),
                mean_centered=False,
                same_for_all_channels=False,
                p_per_sample=0.3,
                p_per_channel=0.5
            )
        )

        tr_transforms.append(
            LocalGammaTransform(
                lambda x, y: np.exp(np.random.uniform(np.log(x[y] // 6), np.log(x[y]))),
                (-0.5, 1.5),
                lambda: np.random.uniform(0.01, 0.8) if np.random.uniform() < 0.5 else np.random.uniform(1.5, 4),
                same_for_all_channels=False,
                p_per_sample=0.3,
                p_per_channel=0.5
            )
        )

        tr_transforms.append(
            SharpeningTransform(
                strength=(0.1, 1),
                same_for_each_channel=False,
                p_per_sample=0.2,
                p_per_channel=0.5
            )
        )

        if any(self.use_mask_for_norm.values()):
            tr_transforms.append(MaskTransform(self.use_mask_for_norm, mask_idx_in_seg=0, set_outside_to=0))

        tr_transforms.append(RemoveLabelTransform(-1, 0))

        if self.data_aug_params["move_last_seg_chanel_to_data"]:
            all_class_labels = np.arange(1, self.num_classes)
            tr_transforms.append(MoveSegAsOneHotToData(1, all_class_labels, 'seg', 'data'))
            if self.data_aug_params["cascade_do_cascade_augmentations"]:
                tr_transforms.append(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(all_class_labels), 0)),
                        p_per_sample=0.4,
                        key="data",
                        strel_size=(1, 8),
                        p_per_label=1
                    )
                )

                tr_transforms.append(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(all_class_labels), 0)),
                        key="data",
                        p_per_sample=0.2,
                        fill_with_other_class_p=0.15,
                        dont_do_if_covers_more_than_X_percent=0
                    )
                )

        tr_transforms.append(RenameTransform('seg', 'target', True))

        if self.regions is not None:
            tr_transforms.append(ConvertSegmentationToRegionsTransform(self.regions, 'target', 'target'))

        if self.deep_supervision_scales is not None:
            tr_transforms.append(
                DownsampleSegForDSTransform2(self.deep_supervision_scales, 0, input_key='target',
                                             output_key='target')
            )

        tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
        return tr_transforms

    def get_val_transforms(self) -> List[AbstractTransform]:
        val_transforms = list()
        val_transforms.append(RemoveLabelTransform(-1, 0))

        if self.data_aug_params['selected_seg_channels'] is not None:
            val_transforms.append(SegChannelSelectionTransform(self.data_aug_params['selected_seg_channels']))

        if self.data_aug_params["move_last_seg_chanel_to_data"]:
            all_class_labels = np.arange(1, self.num_classes)
            val_transforms.append(MoveSegAsOneHotToData(1, all_class_labels, 'seg', 'data'))
        val_transforms.append(RenameTransform('seg', 'target', True))

        if self.regions is not None:
            val_transforms.append(ConvertSegmentationToRegionsTransform(self.regions, 'target', 'target'))

        if self.deep_supervision_scales is not None:
            val_transforms.append(
                DownsampleSegForDSTransform2(
                    self.deep_supervision_scales, 0, input_key='target',
                    output_key='target')
            )

        val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
        return val_transforms

    def wrap_transforms(self, dataloader_train, dataloader_val, train_transforms, val_transforms):
        tr_gen = NonDetMultiThreadedAugmenter(dataloader_train,
                                              Compose(train_transforms),
                                              self.num_proc_DA,
                                              self.num_cached,
                                              seeds=None,
                                              pin_memory=self.pin_memory)
        val_gen = NonDetMultiThreadedAugmenter(dataloader_val,
                                               Compose(val_transforms),
                                               self.num_proc_DA // 2,
                                               self.num_cached,
                                               seeds=None,
                                               pin_memory=self.pin_memory)
        return tr_gen, val_gen

    def initialize(self, training=True, force_load_plans=False):
        """
        replace DA
        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                tr_transforms = self.get_train_transforms()
                val_transforms = self.get_val_transforms()
                self.tr_gen, self.val_gen = self.wrap_transforms(self.dl_tr, self.dl_val, tr_transforms, val_transforms)

                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()
            self.was_initialized = True
            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
