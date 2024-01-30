from copy import deepcopy
from typing import Union, List, Tuple

import numpy as np
from dynamic_network_architectures.architectures.residual_unet import ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm
from dynamic_network_architectures.building_blocks.residual import BottleneckD
from torch import nn

from nnunetv2.experiment_planning.experiment_planners.network_topology import get_pool_and_conv_props
from nnunetv2.experiment_planning.experiment_planners.residual_unets.ResEncUNet_planner import ResEncUNetPlanner


class ResEncUNetBottleneckPlanner(ResEncUNetPlanner):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResBottleneckEncUNetPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)

    def get_plans_for_configuration(self,
                                    spacing: Union[np.ndarray, Tuple[float, ...], List[float]],
                                    median_shape: Union[np.ndarray, Tuple[int, ...]],
                                    data_identifier: str,
                                    approximate_n_voxels_dataset: float,
                                    _cache: dict) -> dict:
        def _features_per_stage(num_stages, max_num_features) -> Tuple[int, ...]:
            return tuple([min(max_num_features, self.UNet_reference_com_nfeatures * 2 ** i) for
                          i in range(num_stages)])

        def _keygen(patch_size, strides):
            return str(patch_size) + '_' + str(strides)

        assert all([i > 0 for i in spacing]), f"Spacing must be > 0! Spacing: {spacing}"
        num_input_channels = len(self.dataset_json['channel_names'].keys()
                                 if 'channel_names' in self.dataset_json.keys()
                                 else self.dataset_json['modality'].keys())
        max_num_features = self.UNet_max_features_2d if len(spacing) == 2 else self.UNet_max_features_3d
        unet_conv_op = convert_dim_to_conv_op(len(spacing))

        # print(spacing, median_shape, approximate_n_voxels_dataset)
        # find an initial patch size
        # we first use the spacing to get an aspect ratio
        tmp = 1 / np.array(spacing)

        # we then upscale it so that it initially is certainly larger than what we need (rescale to have the same
        # volume as a patch of size 256 ** 3)
        # this may need to be adapted when using absurdly large GPU memory targets. Increasing this now would not be
        # ideal because large initial patch sizes increase computation time because more iterations in the while loop
        # further down may be required.
        if len(spacing) == 3:
            initial_patch_size = [round(i) for i in tmp * (256 ** 3 / np.prod(tmp)) ** (1 / 3)]
        elif len(spacing) == 2:
            initial_patch_size = [round(i) for i in tmp * (2048 ** 2 / np.prod(tmp)) ** (1 / 2)]
        else:
            raise RuntimeError()

        # clip initial patch size to median_shape. It makes little sense to have it be larger than that. Note that
        # this is different from how nnU-Net v1 does it!
        # todo patch size can still get too large because we pad the patch size to a multiple of 2**n
        initial_patch_size = np.array([min(i, j) for i, j in zip(initial_patch_size, median_shape[:len(spacing)])])

        # use that to get the network topology. Note that this changes the patch_size depending on the number of
        # pooling operations (must be divisible by 2**num_pool in each axis)
        network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, \
        shape_must_be_divisible_by = get_pool_and_conv_props(spacing, initial_patch_size,
                                                             self.UNet_featuremap_min_edge_length,
                                                             999999)
        num_stages = len(pool_op_kernel_sizes)

        norm = get_matching_instancenorm(unet_conv_op)
        architecture_kwargs = {
            'network_class_name': self.UNet_class.__module__ + '.' + self.UNet_class.__name__,
            'arch_kwargs': {
                'n_stages': num_stages,
                'features_per_stage': _features_per_stage(num_stages, max_num_features),
                'conv_op': unet_conv_op.__module__ + '.' + unet_conv_op.__name__,
                'kernel_sizes': conv_kernel_sizes,
                'strides': pool_op_kernel_sizes,
                'n_blocks_per_stage': self.UNet_blocks_per_stage_encoder[:num_stages],
                'n_conv_per_stage_decoder': self.UNet_blocks_per_stage_decoder[:num_stages - 1],
                'conv_bias': True,
                'norm_op': norm.__module__ + '.' + norm.__name__,
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None,
                'dropout_op_kwargs': None,
                'nonlin': 'torch.nn.LeakyReLU',
                'nonlin_kwargs': {'inplace': True},
                'block': BottleneckD.__module__ + '.' + BottleneckD.__name__,
                'bottleneck_channels': [i // 4 for i in _features_per_stage(num_stages, max_num_features)]
            },
            '_kw_requires_import': ('conv_op', 'norm_op', 'dropout_op', 'nonlin', 'block'),
        }

        # now estimate vram consumption
        if _keygen(patch_size, pool_op_kernel_sizes) in _cache.keys():
            estimate = _cache[_keygen(patch_size, pool_op_kernel_sizes)]
        else:
            estimate = self.static_estimate_VRAM_usage(patch_size,
                                                       num_input_channels,
                                                       len(self.dataset_json['labels'].keys()),
                                                       architecture_kwargs['network_class_name'],
                                                       architecture_kwargs['arch_kwargs'],
                                                       architecture_kwargs['_kw_requires_import'],
                                                       )
            _cache[_keygen(patch_size, pool_op_kernel_sizes)] = estimate

        # how large is the reference for us here (batch size etc)?
        # adapt for our vram target
        reference = (self.UNet_reference_val_2d if len(spacing) == 2 else self.UNet_reference_val_3d) * \
                    (self.UNet_vram_target_GB / self.UNet_reference_val_corresp_GB)

        while estimate > reference:
            # print(patch_size)
            # patch size seems to be too large, so we need to reduce it. Reduce the axis that currently violates the
            # aspect ratio the most (that is the largest relative to median shape)
            axis_to_be_reduced = np.argsort([i / j for i, j in zip(patch_size, median_shape[:len(spacing)])])[-1]

            # we cannot simply reduce that axis by shape_must_be_divisible_by[axis_to_be_reduced] because this
            # may cause us to skip some valid sizes, for example shape_must_be_divisible_by is 64 for a shape of 256.
            # If we subtracted that we would end up with 192, skipping 224 which is also a valid patch size
            # (224 / 2**5 = 7; 7 < 2 * self.UNet_featuremap_min_edge_length(4) so it's valid). So we need to first
            # subtract shape_must_be_divisible_by, then recompute it and then subtract the
            # recomputed shape_must_be_divisible_by. Annoying.
            patch_size = list(patch_size)
            tmp = deepcopy(patch_size)
            tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
            _, _, _, _, shape_must_be_divisible_by = \
                get_pool_and_conv_props(spacing, tmp,
                                        self.UNet_featuremap_min_edge_length,
                                        999999)
            patch_size[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]

            # now recompute topology
            network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, \
            shape_must_be_divisible_by = get_pool_and_conv_props(spacing, patch_size,
                                                                 self.UNet_featuremap_min_edge_length,
                                                                 999999)

            num_stages = len(pool_op_kernel_sizes)
            architecture_kwargs['arch_kwargs'].update({
                'n_stages': num_stages,
                'kernel_sizes': conv_kernel_sizes,
                'strides': pool_op_kernel_sizes,
                'features_per_stage': _features_per_stage(num_stages, max_num_features),
                'n_blocks_per_stage': self.UNet_blocks_per_stage_encoder[:num_stages],
                'n_conv_per_stage_decoder': self.UNet_blocks_per_stage_decoder[:num_stages - 1],
                'bottleneck_channels': [i // 4 for i in _features_per_stage(num_stages, max_num_features)]
            })
            if _keygen(patch_size, pool_op_kernel_sizes) in _cache.keys():
                estimate = _cache[_keygen(patch_size, pool_op_kernel_sizes)]
            else:
                estimate = self.static_estimate_VRAM_usage(
                    patch_size,
                    num_input_channels,
                    len(self.dataset_json['labels'].keys()),
                    architecture_kwargs['network_class_name'],
                    architecture_kwargs['arch_kwargs'],
                    architecture_kwargs['_kw_requires_import'],
                )
                _cache[_keygen(patch_size, pool_op_kernel_sizes)] = estimate

        # alright now let's determine the batch size. This will give self.UNet_min_batch_size if the while loop was
        # executed. If not, additional vram headroom is used to increase batch size
        ref_bs = self.UNet_reference_val_corresp_bs_2d if len(spacing) == 2 else self.UNet_reference_val_corresp_bs_3d
        batch_size = round((reference / estimate) * ref_bs)

        # we need to cap the batch size to cover at most 5% of the entire dataset. Overfitting precaution. We cannot
        # go smaller than self.UNet_min_batch_size though
        bs_corresponding_to_5_percent = round(
            approximate_n_voxels_dataset * 0.05 / np.prod(patch_size, dtype=np.float64))
        batch_size = max(min(batch_size, bs_corresponding_to_5_percent), self.UNet_min_batch_size)

        resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs = self.determine_resampling()
        resampling_softmax, resampling_softmax_kwargs = self.determine_segmentation_softmax_export_fn()

        normalization_schemes, mask_is_used_for_norm = \
            self.determine_normalization_scheme_and_whether_mask_is_used_for_norm()

        plan = {
            'data_identifier': data_identifier,
            'preprocessor_name': self.preprocessor_name,
            'batch_size': batch_size,
            'patch_size': patch_size,
            'median_image_size_in_voxels': median_shape,
            'spacing': spacing,
            'normalization_schemes': normalization_schemes,
            'use_mask_for_norm': mask_is_used_for_norm,
            'resampling_fn_data': resampling_data.__name__,
            'resampling_fn_seg': resampling_seg.__name__,
            'resampling_fn_data_kwargs': resampling_data_kwargs,
            'resampling_fn_seg_kwargs': resampling_seg_kwargs,
            'resampling_fn_probabilities': resampling_softmax.__name__,
            'resampling_fn_probabilities_kwargs': resampling_softmax_kwargs,
            'architecture': architecture_kwargs
        }
        return plan

class ResEncUNetBottleneckDeeperPlanner(ResEncUNetBottleneckPlanner):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetDeeperResBottleneckEncUNetPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_blocks_per_stage_encoder = (2, 3, 6, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9)
        self.UNet_blocks_per_stage_decoder = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)


if __name__ == '__main__':
    # we know both of these networks run with batch size 2 and 12 on ~8-10GB, respectively
    net = ResidualEncoderUNet(input_channels=1, n_stages=6, features_per_stage=(32, 64, 128, 256, 320, 320),
                              conv_op=nn.Conv3d, kernel_sizes=3, strides=(1, 2, 2, 2, 2, 2),
                              n_blocks_per_stage=(1, 3, 4, 6, 6, 6), num_classes=3,
                              n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
                              conv_bias=True, norm_op=nn.InstanceNorm3d, norm_op_kwargs={}, dropout_op=None,
                              nonlin=nn.LeakyReLU, nonlin_kwargs={'inplace': True}, deep_supervision=True)
    print(net.compute_conv_feature_map_size((128, 128, 128)))  # -> 558319104. The value you see above was finetuned
    # from this one to match the regular nnunetplans more closely

    net = ResidualEncoderUNet(input_channels=1, n_stages=7, features_per_stage=(32, 64, 128, 256, 512, 512, 512),
                              conv_op=nn.Conv2d, kernel_sizes=3, strides=(1, 2, 2, 2, 2, 2, 2),
                              n_blocks_per_stage=(1, 3, 4, 6, 6, 6, 6), num_classes=3,
                              n_conv_per_stage_decoder=(1, 1, 1, 1, 1, 1),
                              conv_bias=True, norm_op=nn.InstanceNorm2d, norm_op_kwargs={}, dropout_op=None,
                              nonlin=nn.LeakyReLU, nonlin_kwargs={'inplace': True}, deep_supervision=True)
    print(net.compute_conv_feature_map_size((512, 512)))  # -> 129793792
