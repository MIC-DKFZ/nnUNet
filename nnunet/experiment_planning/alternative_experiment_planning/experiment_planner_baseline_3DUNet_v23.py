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
from nnunet.configuration import default_num_threads
from nnunet.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from nnunet.experiment_planning.alternative_experiment_planning.experiment_planner_baseline_3DUNet_v21 import \
    ExperimentPlanner3D_v21
from nnunet.experiment_planning.common_utils import get_pool_and_conv_props
from nnunet.experiment_planning.experiment_planner_baseline_3DUNet import ExperimentPlanner
from nnunet.experiment_planning.plan_and_preprocess_task import create_lists_from_splitted_dataset, split_4d, crop
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.paths import *


class ExperimentPlanner3D_v23(ExperimentPlanner3D_v21):
    """
    Combines ExperimentPlannerPoolBasedOnSpacing and ExperimentPlannerTargetSpacingForAnisoAxis

    We also increase the base_num_features to 32. This is solely because mixed precision training with 3D convs and
    Nvidia apex/amp is A LOT faster if the number of filters is divisible by 8
    """
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlanner3D_v23, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "nnUNetData_plans_v2.3"
        self.plans_fname = join(self.preprocessed_output_folder,
                                default_plans_identifier + "v2.3_plans_3D.pkl")
        self.preprocessor_name = "Preprocessor3DDifferentResampling"
