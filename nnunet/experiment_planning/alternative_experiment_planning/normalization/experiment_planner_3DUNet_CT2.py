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
