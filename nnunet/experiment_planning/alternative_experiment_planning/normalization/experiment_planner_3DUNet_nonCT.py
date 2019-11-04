from batchgenerators.utilities.file_and_folder_operations import subdirs
from nnunet.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from nnunet.experiment_planning.experiment_planner_baseline_3DUNet import ExperimentPlanner
from nnunet.experiment_planning.plan_and_preprocess_task import create_lists_from_splitted_dataset, split_4d, crop
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_ids", nargs="+", help="list of int")
    parser.add_argument("-p", action="store_true", help="set this if you actually want to run the preprocessing. If "
                                                        "this is not set then this script will only create the plans file")
    parser.add_argument("-tl", type=int, required=False, default=8, help="num_threads_lowres")
    parser.add_argument("-tf", type=int, required=False, default=8, help="num_threads_fullres")

    args = parser.parse_args()
    task_ids = args.task_ids
    run_preprocessing = args.p
    tl = args.tl
    tf = args.tf

    # we need splitted and cropped data. This could be improved in the future TODO
    tasks = []
    for i in task_ids:
        i = int(i)

        splitted_taskString_candidates = subdirs(splitted_4d_output_dir, prefix="Task%02.0d" % i, join=False)
        raw_taskString_candidates = subdirs(raw_dataset_dir, prefix="Task%02.0d" % i, join=False)
        cropped_taskString_candidates = subdirs(cropped_output_dir, prefix="Task%02.0d" % i, join=False)

        # is splitted data there?
        if len(splitted_taskString_candidates) == 0:
            # splitted not there
            assert len(raw_taskString_candidates) > 0, \
                "splitted data is not present and so is raw data (Task %d)" % i
            assert len(raw_taskString_candidates) == 1, "ambiguous task string (raw Task %d)" % i
            # split raw data into splitted
            split_4d(raw_taskString_candidates[0])
        elif len(splitted_taskString_candidates) > 1:
            raise RuntimeError("ambiguous task string (raw Task %d)" % i)
        else:
            pass

        if len(cropped_taskString_candidates) > 1:
            raise RuntimeError("ambiguous task string (raw Task %d)" % i)
        else:
            crop(splitted_taskString_candidates[0], False, tf)

        tasks.append(cropped_taskString_candidates[0])

    for t in tasks:
        try:
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

            threads = (tl, tf)

            print("number of threads: ", threads, "\n")

            exp_planner = ExperimentPlannernonCT(cropped_out_dir, preprocessing_output_dir_this_task)
            exp_planner.plan_experiment()
            if run_preprocessing:
                exp_planner.run_preprocessing(threads)
        except Exception as e:
            print(e)


