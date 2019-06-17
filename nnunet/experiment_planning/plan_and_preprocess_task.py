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
from nnunet.configuration import default_num_threads
from nnunet.experiment_planning.find_classes_in_slice import add_classes_in_slice_info
from nnunet.preprocessing.cropping import ImageCropper
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import splitted_4d_output_dir, cropped_output_dir, preprocessing_output_dir, raw_dataset_dir
import numpy as np
import pickle
from nnunet.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
import os
from multiprocessing import Pool
import json
import shutil
from nnunet.experiment_planning.common_utils import split_4d_nifti


def split_4d(task_string):
    base_folder = join(raw_dataset_dir, task_string)
    output_folder = join(splitted_4d_output_dir, task_string)

    if isdir(output_folder):
        shutil.rmtree(output_folder)

    files = []
    output_dirs = []

    maybe_mkdir_p(output_folder)
    for subdir in ["imagesTr", "imagesTs"]:
        curr_out_dir = join(output_folder, subdir)
        if not isdir(curr_out_dir):
            os.mkdir(curr_out_dir)
        curr_dir = join(base_folder, subdir)
        nii_files = [join(curr_dir, i) for i in os.listdir(curr_dir) if i.endswith(".nii.gz")]
        nii_files.sort()
        for n in nii_files:
            files.append(n)
            output_dirs.append(curr_out_dir)

    shutil.copytree(join(base_folder, "labelsTr"), join(output_folder, "labelsTr"))

    p = Pool(default_num_threads)
    p.starmap(split_4d_nifti, zip(files, output_dirs))
    p.close()
    p.join()
    shutil.copy(join(base_folder, "dataset.json"), output_folder)


def create_lists_from_splitted_dataset(base_folder_splitted):
    lists = []

    json_file = join(base_folder_splitted, "dataset.json")
    with open(json_file) as jsn:
        d = json.load(jsn)
        training_files = d['training']
    num_modalities = len(d['modality'].keys())
    for tr in training_files:
        cur_pat = []
        for mod in range(num_modalities):
            cur_pat.append(join(base_folder_splitted, "imagesTr", tr['image'].split("/")[-1][:-7] +
                                "_%04.0d.nii.gz" % mod))
        cur_pat.append(join(base_folder_splitted, "labelsTr", tr['label'].split("/")[-1]))
        lists.append(cur_pat)
    return lists, {int(i): d['modality'][str(i)] for i in d['modality'].keys()}


def create_lists_from_splitted_dataset_folder(folder):
    """
    does not rely on dataset.json
    :param folder:
    :return:
    """
    caseIDs = get_caseIDs_from_splitted_dataset_folder(folder)
    list_of_lists = []
    for f in caseIDs:
        list_of_lists.append(subfiles(folder, prefix=f, suffix=".nii.gz", join=True, sort=True))
    return list_of_lists


def get_caseIDs_from_splitted_dataset_folder(folder):
    files = subfiles(folder, suffix=".nii.gz", join=False)
    # all files must be .nii.gz and have 4 digit modality index
    files = [i[:-12] for i in files]
    # only unique patient ids
    files = np.unique(files)
    return files


def crop(task_string, override=False, num_threads=default_num_threads):
    cropped_out_dir = join(cropped_output_dir, task_string)
    maybe_mkdir_p(cropped_out_dir)

    if override and isdir(cropped_out_dir):
        shutil.rmtree(cropped_out_dir)
        maybe_mkdir_p(cropped_out_dir)

    splitted_4d_output_dir_task = join(splitted_4d_output_dir, task_string)
    lists, _ = create_lists_from_splitted_dataset(splitted_4d_output_dir_task)

    imgcrop = ImageCropper(num_threads, cropped_out_dir)
    imgcrop.run_cropping(lists, overwrite_existing=override)
    shutil.copy(join(splitted_4d_output_dir, task_string, "dataset.json"), cropped_out_dir)


def analyze_dataset(task_string, override=False, collect_intensityproperties=True, num_processes=default_num_threads):
    cropped_out_dir = join(cropped_output_dir, task_string)
    dataset_analyzer = DatasetAnalyzer(cropped_out_dir, overwrite=override, num_processes=num_processes)
    _ = dataset_analyzer.analyze_dataset(collect_intensityproperties)


def plan_and_preprocess(task_string, num_threads=default_num_threads, no_preprocessing=False):
    from nnunet.experiment_planning.experiment_planner_baseline_2DUNet import ExperimentPlanner2D
    from nnunet.experiment_planning.experiment_planner_baseline_3DUNet import ExperimentPlanner

    preprocessing_output_dir_this_task_train = join(preprocessing_output_dir, task_string)
    cropped_out_dir = join(cropped_output_dir, task_string)
    maybe_mkdir_p(preprocessing_output_dir_this_task_train)

    shutil.copy(join(cropped_out_dir, "dataset_properties.pkl"), preprocessing_output_dir_this_task_train)
    shutil.copy(join(splitted_4d_output_dir, task_string, "dataset.json"), preprocessing_output_dir_this_task_train)

    exp_planner = ExperimentPlanner(cropped_out_dir, preprocessing_output_dir_this_task_train)
    exp_planner.plan_experiment()
    if not no_preprocessing:
        exp_planner.run_preprocessing(num_threads)

    exp_planner = ExperimentPlanner2D(cropped_out_dir, preprocessing_output_dir_this_task_train)
    exp_planner.plan_experiment()
    if not no_preprocessing:
        exp_planner.run_preprocessing(num_threads)

    # write which class is in which slice to all training cases (required to speed up 2D Dataloader)
    # This is done for all data so that if we wanted to use them with 2D we could do so

    if not no_preprocessing:
        p = Pool(default_num_threads)

        # if there is more than one my_data_identifier (different brnaches) then this code will run for all of them if
        # they start with the same string. not problematic, but not pretty
        stages = [i for i in subdirs(preprocessing_output_dir_this_task_train, join=True, sort=True)
                  if i.split("/")[-1].find("stage") != -1]
        for s in stages:
            print(s.split("/")[-1])
            list_of_npz_files = subfiles(s, True, None, ".npz", True)
            list_of_pkl_files = [i[:-4]+".pkl" for i in list_of_npz_files]
            all_classes = []
            for pk in list_of_pkl_files:
                with open(pk, 'rb') as f:
                    props = pickle.load(f)
                all_classes_tmp = np.array(props['classes'])
                all_classes.append(all_classes_tmp[all_classes_tmp >= 0])
            p.map(add_classes_in_slice_info, zip(list_of_npz_files, list_of_pkl_files, all_classes))
        p.close()
        p.join()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, help="task name. There must be a matching folder in "
                                                       "raw_dataset_dir", required=True)
    parser.add_argument('-p', '--processes', type=int, default=3, help='number of processes to run preprocessing '
                                                                       'of full resolution data. If you run out '
                                                                       'of memory, reduce these. Default: 3', required=False)
    parser.add_argument('-o', '--override', type=int, default=0, help="set this to 1 if you want to override "
                                                                      "cropped data and intensityproperties. Default: 0",
                        required=False)
    parser.add_argument('-s', '--use_splitted', type=int, default=1, help='1 = use splitted data if already present ('
                                                                          'skip split_4d). 0 = do splitting again. '
                                                                          'It is save to set this to 1 at all times '
                                                                          'unless the dataset was updated in the '
                                                                          'meantime. Default: 1', required=False)
    parser.add_argument('-no_preprocessing', type=int, default=0, help='debug only. If set to 1 this will run only'
                                                                       'experiment planning and not run the '
                                                                       'preprocessing')

    args = parser.parse_args()
    task = args.task
    processes = args.processes
    override = args.override
    use_splitted = args.use_splitted
    no_preprocessing = args.no_preprocessing

    if override == 0:
        override = False
    elif override == 1:
        override = True
    else:
        raise ValueError("only 0 or 1 allowed for override")

    if no_preprocessing == 0:
        no_preprocessing = False
    elif no_preprocessing == 1:
        no_preprocessing = True
    else:
        raise ValueError("only 0 or 1 allowed for override")

    if use_splitted == 0:
        use_splitted = False
    elif use_splitted == 1:
        use_splitted = True
    else:
        raise ValueError("only 0 or 1 allowed for use_splitted")

    if task == "all":
        all_tasks_that_need_splitting = subdirs(raw_dataset_dir, prefix="Task", join=False)

        for t in all_tasks_that_need_splitting:
            if not use_splitted or not isdir(join(splitted_4d_output_dir, t)):
                print("splitting task ", t)
                split_4d(t)

        all_splitted_tasks = subdirs(splitted_4d_output_dir, prefix="Task", join=False)
        for t in all_splitted_tasks:
            crop(t, override=override, num_threads=processes)
            analyze_dataset(t, override=override, collect_intensityproperties=True, num_processes=processes)
            plan_and_preprocess(t, processes, no_preprocessing)
    else:
        if not use_splitted or not isdir(join(splitted_4d_output_dir, task)):
            print("splitting task ", task)
            split_4d(task)

        crop(task, override=override, num_threads=processes)
        analyze_dataset(task, override, collect_intensityproperties=True, num_processes=processes)
        plan_and_preprocess(task, processes, no_preprocessing)
