from copy import deepcopy

import numpy as np
from nnunet.experiment_planning.common_utils import get_pool_and_conv_props_poolLateV2
from nnunet.experiment_planning.experiment_planner_baseline_3DUNet import ExperimentPlanner
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.paths import *


class ExperimentPlanner3D_IsoPatchesInVoxels(ExperimentPlanner):
    """
    patches that are isotropic in the number of voxels (not mm), such as 128x128x128 allow more voxels to be processed
    at once because we don't have to do annoying pooling stuff

    CAREFUL!
    this one does not support transpose_forward and transpose_backward
    """
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlanner3D_IsoPatchesInVoxels, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "nnUNetData_isoPatchesInVoxels"
        self.plans_fname = join(self.preprocessed_output_folder, "nnUNetPlans" + "fixedisoPatchesInVoxels_plans_3D.pkl")

    def get_properties_for_stage(self, current_spacing, original_spacing, original_shape, num_cases,
                                     num_modalities, num_classes):
        """
        """
        new_median_shape = np.round(original_spacing / current_spacing * original_shape).astype(int)
        dataset_num_voxels = np.prod(new_median_shape) * num_cases

        input_patch_size = new_median_shape

        network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_shp, \
        shape_must_be_divisible_by = get_pool_and_conv_props_poolLateV2(input_patch_size,
                                                                        self.unet_featuremap_min_edge_length,
                                                                        self.unet_max_numpool,
                                                                        current_spacing)

        ref = Generic_UNet.use_this_for_batch_size_computation_3D
        here = Generic_UNet.compute_approx_vram_consumption(new_shp, network_num_pool_per_axis,
                                                            self.unet_base_num_features,
                                                            self.unet_max_num_filters, num_modalities,
                                                            num_classes,
                                                            pool_op_kernel_sizes)
        while here > ref:
            # find the largest axis. If patch is isotropic, pick the axis with the largest spacing
            if len(np.unique(new_shp)) == 1:
                axis_to_be_reduced = np.argsort(current_spacing)[-1]
            else:
                axis_to_be_reduced = np.argsort(new_shp)[-1]

            tmp = deepcopy(new_shp)
            tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
            _, _, _, _, shape_must_be_divisible_by_new = \
                get_pool_and_conv_props_poolLateV2(tmp,
                                                   self.unet_featuremap_min_edge_length,
                                                   self.unet_max_numpool,
                                                   current_spacing)
            new_shp[axis_to_be_reduced] -= shape_must_be_divisible_by_new[axis_to_be_reduced]

            # we have to recompute numpool now:
            network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_shp, \
            shape_must_be_divisible_by = get_pool_and_conv_props_poolLateV2(new_shp,
                                                                            self.unet_featuremap_min_edge_length,
                                                                            self.unet_max_numpool,
                                                                            current_spacing)

            here = Generic_UNet.compute_approx_vram_consumption(new_shp, network_num_pool_per_axis,
                                                                self.unet_base_num_features,
                                                                self.unet_max_num_filters, num_modalities,
                                                                num_classes, pool_op_kernel_sizes)
            print(new_shp)

        input_patch_size = new_shp

        batch_size = Generic_UNet.DEFAULT_BATCH_SIZE_3D  # This is what works with 128**3
        batch_size = int(np.floor(max(ref / here, 1) * batch_size))

        # check if batch size is too large
        max_batch_size = np.round(self.batch_size_covers_max_percent_of_dataset * dataset_num_voxels /
                                  np.prod(input_patch_size, dtype=np.int64)).astype(int)
        max_batch_size = max(max_batch_size, self.unet_min_batch_size)
        batch_size = min(batch_size, max_batch_size)

        do_dummy_2D_data_aug = (max(input_patch_size) / input_patch_size[
            0]) > self.anisotropy_threshold

        plan = {
            'batch_size': batch_size,
            'num_pool_per_axis': network_num_pool_per_axis,
            'patch_size': input_patch_size,
            'median_patient_size_in_voxels': new_median_shape,
            'current_spacing': current_spacing,
            'original_spacing': original_spacing,
            'do_dummy_2D_data_aug': do_dummy_2D_data_aug,
            'pool_op_kernel_sizes': pool_op_kernel_sizes,
            'conv_kernel_sizes': conv_kernel_sizes,
        }
        return plan

