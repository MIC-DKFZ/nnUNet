#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from copy import deepcopy

import numpy as np
from nnunet.experiment_planning.experiment_planner_baseline_3DUNet_v21 import \
    ExperimentPlanner3D_v21
from nnunet.experiment_planning.common_utils import get_pool_and_conv_props
from nnunet.paths import *
from nnunet.network_architecture.generic_modular_residual_UNet import FabiansUNet


class ExperimentPlanner3DFabiansResUNet_v21(ExperimentPlanner3D_v21):
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlanner3DFabiansResUNet_v21, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "nnUNetData_plans_v2.1"# "nnUNetData_FabiansResUNet_v2.1"
        self.plans_fname = join(self.preprocessed_output_folder,
                                "nnUNetPlans_FabiansResUNet_v2.1_plans_3D.pkl")

    def get_properties_for_stage(self, current_spacing, original_spacing, original_shape, num_cases,
                                 num_modalities, num_classes):
        """
        We use FabiansUNet instead of Generic_UNet
        """
        new_median_shape = np.round(original_spacing / current_spacing * original_shape).astype(int)
        dataset_num_voxels = np.prod(new_median_shape) * num_cases

        # the next line is what we had before as a default. The patch size had the same aspect ratio as the median shape of a patient. We swapped t
        # input_patch_size = new_median_shape

        # compute how many voxels are one mm
        input_patch_size = 1 / np.array(current_spacing)

        # normalize voxels per mm
        input_patch_size /= input_patch_size.mean()

        # create an isotropic patch of size 512x512x512mm
        input_patch_size *= 1 / min(input_patch_size) * 512  # to get a starting value
        input_patch_size = np.round(input_patch_size).astype(int)

        # clip it to the median shape of the dataset because patches larger then that make not much sense
        input_patch_size = [min(i, j) for i, j in zip(input_patch_size, new_median_shape)]

        network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_shp, \
        shape_must_be_divisible_by = get_pool_and_conv_props(current_spacing, input_patch_size,
                                                             self.unet_featuremap_min_edge_length,
                                                             self.unet_max_numpool)
        pool_op_kernel_sizes = [[1, 1, 1]] + pool_op_kernel_sizes
        blocks_per_stage_encoder = FabiansUNet.default_blocks_per_stage_encoder[:len(pool_op_kernel_sizes)]
        blocks_per_stage_decoder = FabiansUNet.default_blocks_per_stage_decoder[:len(pool_op_kernel_sizes) - 1]

        ref = FabiansUNet.use_this_for_3D_configuration
        here = FabiansUNet.compute_approx_vram_consumption(input_patch_size, self.unet_base_num_features,
                                                           self.unet_max_num_filters, num_modalities, num_classes,
                                                           pool_op_kernel_sizes, blocks_per_stage_encoder,
                                                           blocks_per_stage_decoder, 2, self.unet_min_batch_size,)
        while here > ref:
            axis_to_be_reduced = np.argsort(new_shp / new_median_shape)[-1]

            tmp = deepcopy(new_shp)
            tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
            _, _, _, _, shape_must_be_divisible_by_new = \
                get_pool_and_conv_props(current_spacing, tmp,
                                        self.unet_featuremap_min_edge_length,
                                        self.unet_max_numpool,
                                        )
            new_shp[axis_to_be_reduced] -= shape_must_be_divisible_by_new[axis_to_be_reduced]

            # we have to recompute numpool now:
            network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_shp, \
            shape_must_be_divisible_by = get_pool_and_conv_props(current_spacing, new_shp,
                                                                 self.unet_featuremap_min_edge_length,
                                                                 self.unet_max_numpool,
                                                                 )
            pool_op_kernel_sizes = [[1, 1, 1]] + pool_op_kernel_sizes
            blocks_per_stage_encoder = FabiansUNet.default_blocks_per_stage_encoder[:len(pool_op_kernel_sizes)]
            blocks_per_stage_decoder = FabiansUNet.default_blocks_per_stage_decoder[:len(pool_op_kernel_sizes) - 1]
            here = FabiansUNet.compute_approx_vram_consumption(new_shp, self.unet_base_num_features,
                                                               self.unet_max_num_filters, num_modalities, num_classes,
                                                               pool_op_kernel_sizes, blocks_per_stage_encoder,
                                                               blocks_per_stage_decoder, 2, self.unet_min_batch_size)
        input_patch_size = new_shp

        batch_size = FabiansUNet.default_min_batch_size
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
            'num_blocks_encoder': blocks_per_stage_encoder,
            'num_blocks_decoder': blocks_per_stage_decoder
        }
        return plan

    def run_preprocessing(self, num_threads):
        """
        On all datasets except 3d fullres on spleen the preprocessed data would look identical to
        ExperimentPlanner3D_v21 (I tested decathlon data only). Therefore we just reuse the preprocessed data of
        that other planner
        :param num_threads:
        :return:
        """
        pass
