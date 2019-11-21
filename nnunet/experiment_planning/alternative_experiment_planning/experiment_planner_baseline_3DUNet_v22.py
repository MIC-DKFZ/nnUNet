#    Copyright 2019 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
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
from copy import deepcopy

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from nnunet.experiment_planning.alternative_experiment_planning.experiment_planner_baseline_3DUNet_v21 import \
    ExperimentPlanner3D_v21
from nnunet.experiment_planning.common_utils import get_pool_and_conv_props
from nnunet.experiment_planning.experiment_planner_baseline_3DUNet import ExperimentPlanner
from nnunet.experiment_planning.plan_and_preprocess_task import create_lists_from_splitted_dataset, split_4d, crop
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.paths import *


class ExperimentPlanner3D_v22(ExperimentPlanner3D_v21):
    """
    """
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super().__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "nnUNetData_plans_v2.2"
        self.plans_fname = join(self.preprocessed_output_folder,
                                default_plans_identifier + "v2.2_plans_3D.pkl")

    def get_target_spacing(self):
        """
        """
        spacings = self.dataset_properties['all_spacings']
        sizes = self.dataset_properties['all_sizes']

        target = np.percentile(np.vstack(spacings), self.target_spacing_percentile, 0)
        target_size = np.percentile(np.vstack(sizes), self.target_spacing_percentile, 0)
        target_size_mm = np.array(target) * np.array(target_size)
        # we need to identify datasets for which a different target spacing could be beneficial. These datasets have
        # the following properties:
        # - one axis which much lower resolution than the others
        # - the lowres axis has much less voxels than the others
        # - (the size in mm of the lowres axis is also reduced)
        worst_spacing_axis = np.argmax(target)
        other_axes = [i for i in range(len(target)) if i != worst_spacing_axis]
        other_spacings = [target[i] for i in other_axes]
        other_sizes = [target_size[i] for i in other_axes]

        has_aniso_spacing = target[worst_spacing_axis] > (self.anisotropy_threshold * min(other_spacings))
        has_aniso_voxels = target_size[worst_spacing_axis] * self.anisotropy_threshold < min(other_sizes)
        # we don't use the last one for now
        #median_size_in_mm = target[target_size_mm] * RESAMPLING_SEPARATE_Z_ANISOTROPY_THRESHOLD < max(target_size_mm)

        if has_aniso_spacing and has_aniso_voxels:
            spacings_of_that_axis = np.vstack(spacings)[:, worst_spacing_axis]
            target_spacing_of_that_axis = np.percentile(spacings_of_that_axis, 10)
            # don't let the spacing of that axis get higher than self.anisotropy_thresholdxthe_other_axes
            target_spacing_of_that_axis = max(min(other_spacings) * self.anisotropy_threshold, target_spacing_of_that_axis)
            target[worst_spacing_axis] = target_spacing_of_that_axis
        return target

