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


import shutil

import nnunet
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_pickle, subfiles
from multiprocessing.pool import Pool
from nnunet.configuration import default_num_threads
from nnunet.experiment_planning.common_utils import get_pool_and_conv_props
from nnunet.experiment_planning.experiment_planner_baseline_3DUNet import ExperimentPlanner
from nnunet.experiment_planning.utils import add_classes_in_slice_info
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.paths import *
from nnunet.preprocessing.preprocessing import PreprocessorFor2D
from nnunet.training.model_restore import recursive_find_python_class


class ExperimentPlanner2D(ExperimentPlanner):
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlanner2D, self).__init__(folder_with_cropped_data,
                                                  preprocessed_output_folder)
        self.data_identifier = default_data_identifier + "_2D"
        self.plans_fname = os.path.join(self.preprocessed_output_folder, "nnUNetPlans" + "_plans_2D.pkl")

        self.unet_base_num_features = 30
        self.unet_max_num_filters = 512
        self.unet_max_numpool = 999

        self.preprocessor_name = "PreprocessorFor2D"

    def get_properties_for_stage(self, current_spacing, original_spacing, original_shape, num_cases,
                                 num_modalities, num_classes):

        new_median_shape = np.round(original_spacing / current_spacing * original_shape).astype(int)

        dataset_num_voxels = np.prod(new_median_shape, dtype=np.int64) * num_cases
        input_patch_size = new_median_shape[1:]

        network_numpool, net_pool_kernel_sizes, net_conv_kernel_sizes, input_patch_size, \
        shape_must_be_divisible_by = get_pool_and_conv_props(current_spacing[1:], input_patch_size,
                                                             self.unet_featuremap_min_edge_length,
                                                             self.unet_max_numpool)

        estimated_gpu_ram_consumption = Generic_UNet.compute_approx_vram_consumption(input_patch_size,
                                                                                     network_numpool,
                                                                                     self.unet_base_num_features,
                                                                                     self.unet_max_num_filters,
                                                                                     num_modalities, num_classes,
                                                                                     net_pool_kernel_sizes,
                                                                                     conv_per_stage=self.conv_per_stage)

        batch_size = int(np.floor(Generic_UNet.use_this_for_batch_size_computation_2D /
                                  estimated_gpu_ram_consumption * Generic_UNet.DEFAULT_BATCH_SIZE_2D))
        if batch_size < self.unet_min_batch_size:
            raise RuntimeError("This framework is not made to process patches this large. We will add patch-based "
                               "2D networks later. Sorry for the inconvenience")

        # check if batch size is too large (more than 5 % of dataset)
        max_batch_size = np.round(self.batch_size_covers_max_percent_of_dataset * dataset_num_voxels /
                                  np.prod(input_patch_size, dtype=np.int64)).astype(int)
        batch_size = max(1, min(batch_size, max_batch_size))

        plan = {
            'batch_size': batch_size,
            'num_pool_per_axis': network_numpool,
            'patch_size': input_patch_size,
            'median_patient_size_in_voxels': new_median_shape,
            'current_spacing': current_spacing,
            'original_spacing': original_spacing,
            'pool_op_kernel_sizes': net_pool_kernel_sizes,
            'conv_kernel_sizes': net_conv_kernel_sizes,
            'do_dummy_2D_data_aug': False
        }
        return plan

    def plan_experiment(self):
        use_nonzero_mask_for_normalization = self.determine_whether_to_use_mask_for_norm()
        print("Are we using the nonzero maks for normalizaion?", use_nonzero_mask_for_normalization)

        spacings = self.dataset_properties['all_spacings']
        sizes = self.dataset_properties['all_sizes']
        all_classes = self.dataset_properties['all_classes']
        modalities = self.dataset_properties['modalities']
        num_modalities = len(list(modalities.keys()))

        target_spacing = self.get_target_spacing()
        new_shapes = np.array([np.array(i) / target_spacing * np.array(j) for i, j in zip(spacings, sizes)])

        max_spacing_axis = np.argmax(target_spacing)
        remaining_axes = [i for i in list(range(3)) if i != max_spacing_axis]
        self.transpose_forward = [max_spacing_axis] + remaining_axes
        self.transpose_backward = [np.argwhere(np.array(self.transpose_forward) == i)[0][0] for i in range(3)]

        # we base our calculations on the median shape of the datasets
        median_shape = np.median(np.vstack(new_shapes), 0)
        print("the median shape of the dataset is ", median_shape)

        max_shape = np.max(np.vstack(new_shapes), 0)
        print("the max shape in the dataset is ", max_shape)
        min_shape = np.min(np.vstack(new_shapes), 0)
        print("the min shape in the dataset is ", min_shape)

        print("we don't want feature maps smaller than ", self.unet_featuremap_min_edge_length, " in the bottleneck")

        # how many stages will the image pyramid have?
        self.plans_per_stage = []

        target_spacing_transposed = np.array(target_spacing)[self.transpose_forward]
        median_shape_transposed = np.array(median_shape)[self.transpose_forward]
        print("the transposed median shape of the dataset is ", median_shape_transposed)

        self.plans_per_stage.append(
            self.get_properties_for_stage(target_spacing_transposed, target_spacing_transposed, median_shape_transposed,
                                          num_cases=len(self.list_of_cropped_npz_files),
                                          num_modalities=num_modalities,
                                          num_classes=len(all_classes) + 1),
            )

        print(self.plans_per_stage)

        self.plans_per_stage = self.plans_per_stage[::-1]
        self.plans_per_stage = {i: self.plans_per_stage[i] for i in range(len(self.plans_per_stage))}  # convert to dict

        normalization_schemes = self.determine_normalization_scheme()
        # deprecated
        only_keep_largest_connected_component, min_size_per_class, min_region_size_per_class = None, None, None

        # these are independent of the stage
        plans = {'num_stages': len(list(self.plans_per_stage.keys())), 'num_modalities': num_modalities,
                 'modalities': modalities, 'normalization_schemes': normalization_schemes,
                 'dataset_properties': self.dataset_properties, 'list_of_npz_files': self.list_of_cropped_npz_files,
                 'original_spacings': spacings, 'original_sizes': sizes,
                 'preprocessed_data_folder': self.preprocessed_output_folder, 'num_classes': len(all_classes),
                 'all_classes': all_classes, 'base_num_features': self.unet_base_num_features,
                 'use_mask_for_norm': use_nonzero_mask_for_normalization,
                 'keep_only_largest_region': only_keep_largest_connected_component,
                 'min_region_size_per_class': min_region_size_per_class, 'min_size_per_class': min_size_per_class,
                 'transpose_forward': self.transpose_forward, 'transpose_backward': self.transpose_backward,
                 'data_identifier': self.data_identifier, 'plans_per_stage': self.plans_per_stage,
                 'preprocessor_name': self.preprocessor_name,
                 }

        self.plans = plans
        self.save_my_plans()
