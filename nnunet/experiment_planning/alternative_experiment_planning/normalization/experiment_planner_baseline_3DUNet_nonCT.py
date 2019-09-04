from nnunet.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from nnunet.experiment_planning.experiment_planner_baseline_3DUNet import ExperimentPlanner
from nnunet.experiment_planning.plan_and_preprocess_task import create_lists_from_splitted_dataset
import shutil
from nnunet.paths import *
from collections import OrderedDict


class ExperimentPlannernonCT(ExperimentPlanner):
    """
    Preprocesses all data in nonCT mode (this is what we use for MRI per default, but here it is applied to CT images
    as well)
    """
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlannernonCT, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "nnUNet_nonCT"
        self.plans_fname = join(self.preprocessed_output_folder, default_plans_identifier + "nonCT_plans_3D.pkl")

    def determine_normalization_scheme(self):
        schemes = OrderedDict()
        modalities = self.dataset_properties['modalities']
        num_modalities = len(list(modalities.keys()))

        for i in range(num_modalities):
            if modalities[i] == "CT":
                schemes[i] = "nonCT"
            else:
                schemes[i] = "nonCT"
        return schemes


if __name__ == "__main__":
    tasks = ['Task03_Liver',
             'Task06_Lung',
             'Task07_Pancreas',
             'Task08_HepaticVessel',
             'Task09_Spleen',
             'Task10_Colon',

             ]

    for t in tasks:
        print("\n\n\n", t)
        cropped_out_dir = os.path.join(cropped_output_dir, t)
        preprocessing_output_dir_this_task = os.path.join(preprocessing_output_dir, t)
        splitted_4d_output_dir_task = os.path.join(splitted_4d_output_dir, t)
        lists, modalities = create_lists_from_splitted_dataset(splitted_4d_output_dir_task)

        dataset_analyzer = DatasetAnalyzer(cropped_out_dir, overwrite=False)
        _ = dataset_analyzer.analyze_dataset() # this will write output files that will be used by the ExperimentPlanner

        maybe_mkdir_p(preprocessing_output_dir_this_task)
        shutil.copy(join(cropped_out_dir, "dataset_properties.pkl"), preprocessing_output_dir_this_task)
        shutil.copy(join(splitted_4d_output_dir, t, "dataset.json"), preprocessing_output_dir_this_task)

        threads = (32, 32)

        print("number of threads: ", threads, "\n")

        exp_planner = ExperimentPlannernonCT(cropped_out_dir, preprocessing_output_dir_this_task)
        exp_planner.plan_experiment()
        exp_planner.run_preprocessing(threads)
