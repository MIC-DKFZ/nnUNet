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

from batchgenerators.utilities.file_and_folder_operations import load_pickle
from nnunet.utilities.file_and_folder_operations_winos import * # Join path by slash on windows system.
from nnunet.experiment_planning.experiment_planner_baseline_3DUNet_v21 import ExperimentPlanner3D_v21
from nnunet.paths import *


class ExperimentPlanner3D_v21_Pretrained(ExperimentPlanner3D_v21):
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder, pretrained_model_plans_file: str,
                 pretrained_name: str):
        super().__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.pretrained_model_plans_file = pretrained_model_plans_file
        self.pretrained_name = pretrained_name
        self.data_identifier = "nnUNetData_pretrained_" + pretrained_name
        self.plans_fname = join(self.preprocessed_output_folder, "nnUNetPlans_pretrained_%s_plans_3D.pkl" % pretrained_name)

    def load_pretrained_plans(self):
        classes = self.plans['num_classes']
        self.plans = load_pickle(self.pretrained_model_plans_file)
        self.plans['num_classes'] = classes
        self.transpose_forward = self.plans['transpose_forward']
        self.preprocessor_name = self.plans['preprocessor_name']
        self.plans_per_stage = self.plans['plans_per_stage']
        self.plans['data_identifier'] = self.data_identifier
        self.save_my_plans()
        print(self.plans['plans_per_stage'])

    def run_preprocessing(self, num_threads):
        self.load_pretrained_plans()
        super().run_preprocessing(num_threads)
