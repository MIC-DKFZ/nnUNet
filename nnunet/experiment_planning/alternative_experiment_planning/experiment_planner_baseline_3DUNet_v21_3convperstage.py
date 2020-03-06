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

from copy import deepcopy

import numpy as np
from nnunet.experiment_planning.common_utils import get_pool_and_conv_props
from nnunet.experiment_planning.experiment_planner_baseline_3DUNet import ExperimentPlanner
from nnunet.experiment_planning.experiment_planner_baseline_3DUNet_v21 import ExperimentPlanner3D_v21
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.paths import *


class ExperimentPlanner3D_v21_3cps(ExperimentPlanner3D_v21):
    """
    have 3x conv-in-lrelu per resolution instead of 2 while remaining in the same memory budget

    This only works with 3d fullres because we use the same data as ExperimentPlanner3D_v21. Lowres would require to
    rerun preprocesing (different patch size = different 3d lowres target spacing)
    """
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlanner3D_v21_3cps, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.plans_fname = join(self.preprocessed_output_folder,
                                "nnUNetPlansv2.1_3cps_plans_3D.pkl")
        self.unet_base_num_features = 32
        self.conv_per_stage = 3

    def run_preprocessing(self, num_threads):
        pass
