from typing import Union, Tuple, List

import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, BrightnessTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.local_transforms import BrightnessGradientAdditiveTransform, LocalContrastTransform, \
    LocalGammaTransform, LocalSmoothingTransform
from batchgenerators.transforms.noise_transforms import SharpeningTransform, GaussianNoiseTransform, \
    GaussianBlurTransform, MedianFilterTransform, BlankRectangleTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, Rot90Transform, TransposeAxesTransform, \
    MirrorTransform
from batchgenerators.transforms.utility_transforms import OneOfTransformPerSample, RemoveLabelTransform, \
    RenameTransform, NumpyToTensor

from nnunetv2.training.data_augmentation.custom_transforms.alegra import InhomogeneousSliceIlluminationTransform, \
    LowContrastTransform
from nnunetv2.training.data_augmentation.custom_transforms.cascade_transforms import MoveSegAsOneHotToData, \
    ApplyRandomBinaryOperatorTransform, RemoveRandomConnectedComponentFromOneHotEncodingTransform
from nnunetv2.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import \
    DownsampleSegForDSTransform2
from nnunetv2.training.data_augmentation.custom_transforms.masking import MaskTransform
from nnunetv2.training.data_augmentation.custom_transforms.region_based_training import \
    ConvertSegmentationToRegionsTransform
from nnunetv2.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import Convert3DTo2DTransform, \
    Convert2DTo3DTransform
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
from nnunetv2.training.nnUNetTrainer.variants.sparse_labels.nnUNetTrainer_betterIgnoreSampling import \
    nnUNetDataLoader2DBetterIgnSampling, nnUNetDataLoader3DBetterIgnSampling


class nnUNetTrainer_airwayAug_new(nnUNetTrainerDA5):
    """
    We inherit from nnUNetTrainerDA5 now because that trainer still uses the old DA and we can piggyback on the
    compatibility that it has built in
    """
    @staticmethod
    def get_training_transforms(patch_size: Union[np.ndarray, Tuple[int]],
                                rotation_for_DA: dict,
                                deep_supervision_scales: Union[List, Tuple],
                                mirror_axes: Tuple[int, ...],
                                do_dummy_2d_data_aug: bool,
                                order_resampling_data: int = 3,
                                order_resampling_seg: int = 1,
                                border_val_seg: int = -1,
                                use_mask_for_norm: List[bool] = None,
                                is_cascaded: bool = False,
                                foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                ignore_label: int = None) -> AbstractTransform:
        if is_cascaded and regions is not None:
            raise NotImplementedError('Region based training is not yet implemented for the cascade!')

        tr_transforms = []
        tr_transforms.append(RenameTransform('target', 'seg', True))

        # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
        if do_dummy_2d_data_aug: # todo
            ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        tr_transforms.append(SpatialTransform(
            patch_size_spatial, patch_center_dist_from_border=None,
            do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
            do_rotation=True, angle_x=rotation_for_DA['x'], angle_y=rotation_for_DA['y'], angle_z=rotation_for_DA['z'],
            p_rot_per_axis=1,  # todo experiment with this
            do_scale=True, scale=(0.7, 1.4),
            border_mode_data="constant", border_cval_data=0, order_data=order_resampling_data,
            border_mode_seg="constant", border_cval_seg=border_val_seg, order_seg=order_resampling_seg,
            random_crop=False,  # random cropping is part of our dataloaders
            p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
            independent_scale_for_each_axis=False  # todo experiment with this
        ))

        if do_dummy_2d_data_aug:
            tr_transforms.append(Convert2DTo3DTransform())

        tr_transforms.append(
            InhomogeneousSliceIlluminationTransform(
                (1, 5),
                (2, 8),
                lambda: np.random.uniform(0.2, 0.6) if np.random.uniform() < 0.8 else np.random.uniform(0.7, 1.2),
                (0, 0.3),
                (0.25, 2),
                0.3,
                False,
                1,
                'data'
            )
        )

        ############# transpose and rot90 ############
        tmp = []
        matching_axes = np.array([sum([i == j for j in patch_size]) for i in patch_size])
        if np.any(matching_axes > 1):
            valid_axes = list(np.where(matching_axes == np.max(matching_axes))[0])
            tmp.append(
                Rot90Transform(
                    (0, 1, 2, 3), axes=valid_axes, data_key='data', label_key='seg', p_per_sample=0.25
                ),
            )

        matching_axes = np.array([sum([i == j for j in patch_size]) for i in patch_size])
        if np.any(matching_axes > 1):
            valid_axes = list(np.where(matching_axes == np.max(matching_axes))[0])
            tmp.append(
                TransposeAxesTransform(valid_axes, data_key='data', label_key='seg', p_per_sample=0.25)
            )

        tr_transforms.append(OneOfTransformPerSample(tmp, ['data', 'seg']))

        ######### Noise or sharpening #######################
        # the don't like each other, so separate them
        tmp = []
        tmp.append(
            SharpeningTransform(
                strength=(0.1, 1),
                same_for_each_channel=False,
                p_per_sample=0.2,
                p_per_channel=0.5
            )
        )
        tmp.append(GaussianNoiseTransform(p_per_sample=0.3, noise_variance=(0.1, 2.5)))
        tr_transforms.append(OneOfTransformPerSample(tmp, ['data', 'seg']))

        ######### Brightness #################
        tmp = []
        tmp.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.25))
        tmp.append(BrightnessTransform(0.0,
                                       0.1,
                                       True, p_per_sample=0.25,
                                       p_per_channel=0.1))
        tmp.append(
            BrightnessGradientAdditiveTransform(
                lambda x, y: np.random.uniform(x[y] // 8, x[y] // 2),
                (-0.5, 1.5),
                max_strength=lambda x, y: np.random.uniform(-3, -1) if np.random.uniform() < 0.5 else np.random.uniform(
                    1,
                    3),
                same_for_all_channels=False,
                p_per_sample=0.25,
                p_per_channel=1
            )
        )
        tr_transforms.append(OneOfTransformPerSample(tmp, ['data', 'seg']))

        ######## Contrast ####################
        tmp = []
        tmp.append(
            LowContrastTransform(lambda: np.random.uniform(1e-2, 1.5) ** 2, True, 0.25, 1, 'data')
        )
        tmp.append(ContrastAugmentationTransform(p_per_sample=0.25, contrast_range=(1e-2, 1.5)))
        tmp.append(LocalContrastTransform(
            lambda x, y: np.random.uniform(x[y] // 8, x[y] // 2),
            (-0.5, 1.5),
            new_contrast=(1e-5, 2),
            same_for_all_channels=False,
            p_per_sample=0.25,
            p_per_channel=1
        ))
        tr_transforms.append(OneOfTransformPerSample(tmp, ['data', 'seg']))

        ###### Gamma #################
        tmp = []
        tmp.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
        tmp.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))
        tmp.append(
            LocalGammaTransform(
                lambda x, y: np.random.uniform(x[y] // 8, x[y] // 2),
                lambda x, y: np.random.uniform(-0.5, 0.5) if np.random.uniform() < 0.5 else np.random.uniform(0.5, 1.5),
                lambda: np.random.uniform(0.01, 0.8) if np.random.uniform() < 0.5 else np.random.uniform(1.5, 4),
                same_for_all_channels=False,
                p_per_sample=0.15,
                p_per_channel=1
            )
        )
        tr_transforms.append(OneOfTransformPerSample(tmp, ['data', 'seg']))

        ###### Blur / low res ########
        tmp = []
        tmp.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.25,
                                         p_per_channel=0.5))
        tmp.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                  p_per_channel=0.5,
                                                  order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                  ignore_axes=ignore_axes))
        tmp.append(
            MedianFilterTransform(
                (1, 5),
                same_for_each_channel=False,
                p_per_sample=0.15,
                p_per_channel=0.5
            )
        )
        tmp.append(LocalSmoothingTransform(
            lambda x, y: np.random.uniform(x[y] // 8, x[y] // 2),
            (-0.5, 1.5),
            smoothing_strength=(0.5, 1),
            kernel_size=(0.3, 5),
            same_for_all_channels=True,
            p_per_sample=0.25,
            p_per_channel=1
        ))
        tr_transforms.append(OneOfTransformPerSample(tmp, ['data', 'seg']))

        tr_transforms.append(MirrorTransform(mirror_axes))

        tr_transforms.append(
            BlankRectangleTransform([[max(1, p // 10), p // 3] for p in patch_size],
                                    rectangle_value=np.mean,
                                    num_rectangles=(1, 5),
                                    force_square=False,
                                    p_per_sample=0.3,
                                    p_per_channel=0.5)
        )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            tr_transforms.append(MaskTransform([i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                                               mask_idx_in_seg=0, set_outside_to=0))

        tr_transforms.append(RemoveLabelTransform(-1, 0))

        if is_cascaded:
            if ignore_label is not None:
                raise NotImplementedError('ignore label not yet supported in cascade')
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            use_labels = [i for i in foreground_labels if i != 0]
            tr_transforms.append(MoveSegAsOneHotToData(1, use_labels, 'seg', 'data'))
            tr_transforms.append(ApplyRandomBinaryOperatorTransform(
                channel_idx=list(range(-len(use_labels), 0)),
                p_per_sample=0.4,
                key="data",
                strel_size=(1, 8),
                p_per_label=1))
            tr_transforms.append(
                RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                    channel_idx=list(range(-len(use_labels), 0)),
                    key="data",
                    p_per_sample=0.2,
                    fill_with_other_class_p=0,
                    dont_do_if_covers_more_than_x_percent=0.15))

        tr_transforms.append(RenameTransform('seg', 'target', True))

        if regions is not None:
            # the ignore label must also be converted
            tr_transforms.append(ConvertSegmentationToRegionsTransform(list(regions) + [ignore_label]
                                                                       if ignore_label is not None else regions,
                                                                       'target', 'target'))

        if deep_supervision_scales is not None:
            tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                              output_key='target'))
        tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms


class nnUNetTrainer_airwayAug_new_noSmooth(nnUNetTrainer_airwayAug_new):
    def _build_loss(self):
        # set smooth to 0
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label,
                                  dice_class=MemoryEfficientSoftDiceLoss)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainer_airwayAug_new_noSmooth_betterIgnSampling(nnUNetTrainer_airwayAug_new_noSmooth):
    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        if dim == 2:
            dl_tr = nnUNetDataLoader2DBetterIgnSampling(dataset_tr,
                                                        self.batch_size,
                                                        initial_patch_size,
                                                        self.configuration_manager.patch_size,
                                                        self.label_manager,
                                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                                        sampling_probabilities=None, pad_sides=None)
            dl_val = nnUNetDataLoader2DBetterIgnSampling(dataset_val,
                                                         self.batch_size,
                                                         self.configuration_manager.patch_size,
                                                         self.configuration_manager.patch_size,
                                                         self.label_manager,
                                                         oversample_foreground_percent=self.oversample_foreground_percent,
                                                         sampling_probabilities=None, pad_sides=None)
        else:
            dl_tr = nnUNetDataLoader3DBetterIgnSampling(dataset_tr,
                                                        self.batch_size,
                                                        initial_patch_size,
                                                        self.configuration_manager.patch_size,
                                                        self.label_manager,
                                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                                        sampling_probabilities=None, pad_sides=None)
            dl_val = nnUNetDataLoader3DBetterIgnSampling(dataset_val,
                                                         self.batch_size,
                                                         self.configuration_manager.patch_size,
                                                         self.configuration_manager.patch_size,
                                                         self.label_manager,
                                                         oversample_foreground_percent=self.oversample_foreground_percent,
                                                         sampling_probabilities=None, pad_sides=None)
        return dl_tr, dl_val