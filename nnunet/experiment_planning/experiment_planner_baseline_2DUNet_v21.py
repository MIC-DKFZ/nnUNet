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

from nnunet.experiment_planning.common_utils import get_pool_and_conv_props
from nnunet.experiment_planning.experiment_planner_baseline_2DUNet import ExperimentPlanner2D
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.paths import *
import numpy as np


class ExperimentPlanner2D_v21(ExperimentPlanner2D):
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlanner2D_v21, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "nnUNetData_plans_v2.1_2D"
        # self.plans_fname = join(self.preprocessed_output_folder,
        #                         "nnUNetPlansv2.1_plans_2D.pkl")
        self.plans_fname = os.path.join(self.preprocessed_output_folder,
                                "nnUNetPlansv2.1_plans_2D.pkl")
        self.unet_base_num_features = 32

    def get_properties_for_stage(self, current_spacing, original_spacing, original_shape, num_cases,
                                 num_modalities, num_classes):

        new_median_shape = np.round(original_spacing / current_spacing * original_shape).astype(int)

        dataset_num_voxels = np.prod(new_median_shape, dtype=np.int64) * num_cases
        input_patch_size = new_median_shape[1:]

        network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_shp, \
        shape_must_be_divisible_by = get_pool_and_conv_props(current_spacing[1:], input_patch_size,
                                                             self.unet_featuremap_min_edge_length,
                                                             self.unet_max_numpool)

        # we pretend to use 30 feature maps. This will yield the same configuration as in V1. The larger memory
        # footpring of 32 vs 30 is mor ethan offset by the fp16 training. We make fp16 training default
        # Reason for 32 vs 30 feature maps is that 32 is faster in fp16 training (because multiple of 8)
        ref = Generic_UNet.use_this_for_batch_size_computation_2D * Generic_UNet.DEFAULT_BATCH_SIZE_2D / 2  # for batch size 2
        here = Generic_UNet.compute_approx_vram_consumption(new_shp,
                                                            network_num_pool_per_axis,
                                                            30,
                                                            self.unet_max_num_filters,
                                                            num_modalities, num_classes,
                                                            pool_op_kernel_sizes,
                                                            conv_per_stage=self.conv_per_stage)
        while here > ref:
            axis_to_be_reduced = np.argsort(new_shp / new_median_shape[1:])[-1]

            tmp = deepcopy(new_shp)
            tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
            _, _, _, _, shape_must_be_divisible_by_new = \
                get_pool_and_conv_props(current_spacing[1:], tmp, self.unet_featuremap_min_edge_length,
                                        self.unet_max_numpool)
            new_shp[axis_to_be_reduced] -= shape_must_be_divisible_by_new[axis_to_be_reduced]

            # we have to recompute numpool now:
            network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_shp, \
            shape_must_be_divisible_by = get_pool_and_conv_props(current_spacing[1:], new_shp,
                                                                 self.unet_featuremap_min_edge_length,
                                                                 self.unet_max_numpool)

            here = Generic_UNet.compute_approx_vram_consumption(new_shp, network_num_pool_per_axis,
                                                                self.unet_base_num_features,
                                                                self.unet_max_num_filters, num_modalities,
                                                                num_classes, pool_op_kernel_sizes,
                                                                conv_per_stage=self.conv_per_stage)
            # print(new_shp)

        batch_size = int(np.floor(ref / here) * 2)
        input_patch_size = new_shp

        if batch_size < self.unet_min_batch_size:
            raise RuntimeError("This should not happen")

        # check if batch size is too large (more than 5 % of dataset)
        max_batch_size = np.round(self.batch_size_covers_max_percent_of_dataset * dataset_num_voxels /
                                  np.prod(input_patch_size, dtype=np.int64)).astype(int)
        batch_size = max(1, min(batch_size, max_batch_size))

        plan = {
            'batch_size': batch_size,
            'num_pool_per_axis': network_num_pool_per_axis,
            'patch_size': input_patch_size,
            'median_patient_size_in_voxels': new_median_shape,
            'current_spacing': current_spacing,
            'original_spacing': original_spacing,
            'pool_op_kernel_sizes': pool_op_kernel_sizes,
            'conv_kernel_sizes': conv_kernel_sizes,
            'do_dummy_2D_data_aug': False
        }
        return plan
