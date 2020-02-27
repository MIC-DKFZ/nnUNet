import nnunet
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from nnunet.experiment_planning.utils import create_lists_from_splitted_dataset, crop
from nnunet.paths import *
import shutil
from nnunet.training.model_restore import recursive_find_python_class


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_ids", nargs="+", help="List of integers belonging to the task ids you wish to run"
                                                            " experiment planning and preprocessing for. Each of these "
                                                            "ids must, have a matching folder 'TaskXXX_' in the raw "
                                                            "data folder")
    parser.add_argument("-pl3d", "--planner3d", type=str, default="ExperimentPlanner3D_v21",
                        help="Name of the ExperimentPlanner class for the full resolution 3D U-Net and U-Net cascade. "
                             "Default is ExperimentPlanner3D_v21. Can be 'None', in which case these U-Nets will not be "
                             "configured")
    parser.add_argument("-pl2d", "--planner2d", type=str, default="ExperimentPlanner2D_v21",
                        help="Name of the ExperimentPlanner class for the 2D U-Net. Default is ExperimentPlanner2D_v21. "
                             "Can be 'None', in which case this U-Net will not be configured")
    parser.add_argument("-no_pp", action="store_true",
                        help="Set this flag if you dont want to run the preprocessing. If this is set then this script "
                             "will only run the experiment planning and create the plans file")
    parser.add_argument("-tl", type=int, required=False, default=8,
                        help="Number of processes used for preprocessing the low resolution data for the 3D low "
                             "resolution U-Net. This can be larger than -tf. Don't overdo it or you will run out of "
                             "RAM")
    parser.add_argument("-tf", type=int, required=False, default=8,
                        help="Number of processes used for preprocessing the full resolution data of the 2D U-Net and "
                             "3D U-Net. Don't overdo it or you will run out of RAM")

    args = parser.parse_args()
    task_ids = args.task_ids
    dont_run_preprocessing = args.no_pp
    tl = args.tl
    tf = args.tf
    planner_name3d = args.planner3d
    planner_name2d = args.planner2d

    if planner_name3d == "None":
        planner_name3d = None
    if planner_name2d == "None":
        planner_name2d = None

    # we need raw data
    tasks = []
    for i in task_ids:
        i = int(i)

        taskString_candidates = subdirs(nnUNet_raw_data, prefix="Task%02.0d" % i, join=False)
        cropped_taskString_candidates = subdirs(nnUNet_cropped_data, prefix="Task%02.0d" % i, join=False)

        # we always call crop because it will only crop files that are not yet present
        if len(cropped_taskString_candidates) > 1:
            print(cropped_taskString_candidates)
            raise RuntimeError("ambiguous task string (raw Task %d)" % i)
        else:
            crop(taskString_candidates[0], False, tf)
            cropped_taskString_candidates = subdirs(nnUNet_cropped_data, prefix="Task%02.0d" % i, join=False)

        tasks.append(cropped_taskString_candidates[0])

    search_in = join(nnunet.__path__[0], "experiment_planning")

    if planner_name3d is not None:
        planner_3d = recursive_find_python_class([search_in], planner_name3d, current_module="nnunet.experiment_planning")
        if planner_3d is None:
            raise RuntimeError("Could not find the Planner class %s. Make sure it is located somewhere in "
                               "nnunet.experiment_planning" % planner_name3d)
    else:
        planner_3d = None

    if planner_name2d is not None:
        planner_2d = recursive_find_python_class([search_in], planner_name2d, current_module="nnunet.experiment_planning")
        if planner_2d is None:
            raise RuntimeError("Could not find the Planner class %s. Make sure it is located somewhere in "
                               "nnunet.experiment_planning" % planner_name2d)
    else:
        planner_2d = None

    for t in tasks:
        print("\n\n\n", t)
        cropped_out_dir = os.path.join(nnUNet_cropped_data, t)
        preprocessing_output_dir_this_task = os.path.join(preprocessing_output_dir, t)
        splitted_4d_output_dir_task = os.path.join(nnUNet_raw_data, t)
        lists, modalities = create_lists_from_splitted_dataset(splitted_4d_output_dir_task)

        dataset_analyzer = DatasetAnalyzer(cropped_out_dir, overwrite=False)  # this class creates the fingerprint
        _ = dataset_analyzer.analyze_dataset()  # this will write output files that will be used by the ExperimentPlanner

        maybe_mkdir_p(preprocessing_output_dir_this_task)
        shutil.copy(join(cropped_out_dir, "dataset_properties.pkl"), preprocessing_output_dir_this_task)
        shutil.copy(join(nnUNet_raw_data, t, "dataset.json"), preprocessing_output_dir_this_task)

        threads = (tl, tf)

        print("number of threads: ", threads, "\n")

        if planner_3d is not None:
            exp_planner = planner_3d(cropped_out_dir, preprocessing_output_dir_this_task)
            exp_planner.plan_experiment()
            if not dont_run_preprocessing:  # double negative, yooo
                exp_planner.run_preprocessing(threads)
        if planner_2d is not None:
            exp_planner = planner_2d(cropped_out_dir, preprocessing_output_dir_this_task)
            exp_planner.plan_experiment()
            if not dont_run_preprocessing:  # double negative, yooo
                exp_planner.run_preprocessing(threads)


