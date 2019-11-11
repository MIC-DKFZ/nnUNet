import nnunet
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from nnunet.experiment_planning.plan_and_preprocess_task import split_4d, crop, create_lists_from_splitted_dataset
from nnunet.paths import *
import shutil
from nnunet.training.model_restore import recursive_find_trainer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-pl", "--planner", type=str, default="ExperimentPlanner3D", help="Name of the ExperimentPlanner class")
    parser.add_argument("-t", "--task_ids", nargs="+", help="list of int")
    parser.add_argument("-no_pp", action="store_true", help="set this if you dont want to run the preprocessing. If "
                                                        "this is set then this script will only create the plans file")
    parser.add_argument("-tl", type=int, required=False, default=8, help="num_threads_lowres")
    parser.add_argument("-tf", type=int, required=False, default=8, help="num_threads_fullres")

    args = parser.parse_args()
    task_ids = args.task_ids
    dont_run_preprocessing = args.no_pp
    tl = args.tl
    tf = args.tf
    planner_name = args.planner

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
            print(splitted_taskString_candidates)
            raise RuntimeError("ambiguous task string (raw Task %d)" % i)
        else:
            pass

        if len(cropped_taskString_candidates) > 1:
            print(cropped_taskString_candidates)
            raise RuntimeError("ambiguous task string (raw Task %d)" % i)
        else:
            crop(splitted_taskString_candidates[0], False, tf)

        tasks.append(cropped_taskString_candidates[0])

    search_in = join(nnunet.__path__[0], "experiment_planning")

    planner = recursive_find_trainer([search_in], planner_name, current_module="nnunet.experiment_planning")
    if planner is None:
        raise RuntimeError("Could not find the Planner class %s. Make sure it is located somewhere in "
                           "nnunet.experiment_planning" % planner_name)

    for t in tasks:
        print("\n\n\n", t)
        cropped_out_dir = os.path.join(cropped_output_dir, t)
        preprocessing_output_dir_this_task = os.path.join(preprocessing_output_dir, t)
        splitted_4d_output_dir_task = os.path.join(splitted_4d_output_dir, t)
        lists, modalities = create_lists_from_splitted_dataset(splitted_4d_output_dir_task)

        dataset_analyzer = DatasetAnalyzer(cropped_out_dir, overwrite=False)
        _ = dataset_analyzer.analyze_dataset()  # this will write output files that will be used by the ExperimentPlanner

        maybe_mkdir_p(preprocessing_output_dir_this_task)
        shutil.copy(join(cropped_out_dir, "dataset_properties.pkl"), preprocessing_output_dir_this_task)
        shutil.copy(join(splitted_4d_output_dir, t, "dataset.json"), preprocessing_output_dir_this_task)

        threads = (tl, tf)

        print("number of threads: ", threads, "\n")

        exp_planner = planner(cropped_out_dir, preprocessing_output_dir_this_task)
        exp_planner.plan_experiment()
        if not dont_run_preprocessing:  # double negative, yooo
            exp_planner.run_preprocessing(threads)

