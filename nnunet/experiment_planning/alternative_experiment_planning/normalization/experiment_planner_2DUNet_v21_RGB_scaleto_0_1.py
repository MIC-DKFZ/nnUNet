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


from nnunet.experiment_planning.experiment_planner_baseline_2DUNet_v21 import ExperimentPlanner2D_v21
from nnunet.paths import *


class ExperimentPlanner2D_v21_RGB_scaleTo_0_1(ExperimentPlanner2D_v21):
    """
    used by tutorial nnunet.tutorials.custom_preprocessing
    """
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super().__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "nnUNet_RGB_scaleTo_0_1"
        self.plans_fname = join(self.preprocessed_output_folder, "nnUNet_RGB_scaleTo_0_1" + "_plans_2D.pkl")

        # The custom preprocessor class we intend to use is GenericPreprocessor_scale_uint8_to_0_1. It must be located
        # in nnunet.preprocessing (any file and submodule) and will be found by its name. Make sure to always define
        # unique names!
        self.preprocessor_name = 'GenericPreprocessor_scale_uint8_to_0_1'
