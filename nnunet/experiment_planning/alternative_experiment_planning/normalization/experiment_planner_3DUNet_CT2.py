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


from collections import OrderedDict

from nnunet.experiment_planning.experiment_planner_baseline_3DUNet import ExperimentPlanner
from nnunet.paths import *


class ExperimentPlannerCT2(ExperimentPlanner):
    """
    preprocesses CT data with the "CT2" normalization.

    (clip range comes from training set and is the 0.5 and 99.5 percentile of intensities in foreground)
    CT = clip to range, then normalize with global mn and sd (computed on foreground in training set)
    CT2 = clip to range, normalize each case separately with its own mn and std (computed within the area that was in clip_range)
    """
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlannerCT2, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "nnUNet_CT2"
        self.plans_fname = join(self.preprocessed_output_folder, "nnUNetPlans" + "CT2_plans_3D.pkl")

    def determine_normalization_scheme(self):
        schemes = OrderedDict()
        modalities = self.dataset_properties['modalities']
        num_modalities = len(list(modalities.keys()))

        for i in range(num_modalities):
            if modalities[i] == "CT":
                schemes[i] = "CT2"
            else:
                schemes[i] = "nonCT"
        return schemes
