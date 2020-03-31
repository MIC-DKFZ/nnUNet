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

import json
import os
import pickle
import shutil
from collections import OrderedDict
from multiprocessing import Pool

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, isdir, maybe_mkdir_p, subfiles, subdirs, isfile
from nnunet.configuration import default_num_threads
from nnunet.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from nnunet.experiment_planning.common_utils import split_4d_nifti
from nnunet.paths import nnUNet_raw_data, nnUNet_cropped_data, preprocessing_output_dir
from nnunet.preprocessing.cropping import ImageCropper


def split_4d(input_folder, num_processes=default_num_threads, overwrite_task_output_id=None):
    assert isdir(join(input_folder, "imagesTr")) and isdir(join(input_folder, "labelsTr")) and \
           isfile(join(input_folder, "dataset.json")), \
        "The input folder must be a valid Task folder from the Medical Segmentation Decathlon with at least the " \
        "imagesTr and labelsTr subfolders and the dataset.json file"

    while input_folder.endswith("/"):
        input_folder = input_folder[:-1]

    full_task_name = input_folder.split("/")[-1]

    assert full_task_name.startswith("Task"), "The input folder must point to a folder that starts with TaskXX_"

    first_underscore = full_task_name.find("_")
    assert first_underscore == 6, "Input folder start with TaskXX with XX being a 3-digit id: 00, 01, 02 etc"

    input_task_id = int(full_task_name[4:6])
    if overwrite_task_output_id is None:
        overwrite_task_output_id = input_task_id

    task_name = full_task_name[7:]

    output_folder = join(nnUNet_raw_data, "Task%03.0d_" % overwrite_task_output_id + task_name)

    if isdir(output_folder):
        shutil.rmtree(output_folder)

    files = []
    output_dirs = []

    maybe_mkdir_p(output_folder)
    for subdir in ["imagesTr", "imagesTs"]:
        curr_out_dir = join(output_folder, subdir)
        if not isdir(curr_out_dir):
            os.mkdir(curr_out_dir)
        curr_dir = join(input_folder, subdir)
        nii_files = [join(curr_dir, i) for i in os.listdir(curr_dir) if i.endswith(".nii.gz")]
        nii_files.sort()
        for n in nii_files:
            files.append(n)
            output_dirs.append(curr_out_dir)

    shutil.copytree(join(input_folder, "labelsTr"), join(output_folder, "labelsTr"))

    p = Pool(num_processes)
    p.starmap(split_4d_nifti, zip(files, output_dirs))
    p.close()
    p.join()
    shutil.copy(join(input_folder, "dataset.json"), output_folder)


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
    cropped_out_dir = join(nnUNet_cropped_data, task_string)
    maybe_mkdir_p(cropped_out_dir)

    if override and isdir(cropped_out_dir):
        shutil.rmtree(cropped_out_dir)
        maybe_mkdir_p(cropped_out_dir)

    splitted_4d_output_dir_task = join(nnUNet_raw_data, task_string)
    lists, _ = create_lists_from_splitted_dataset(splitted_4d_output_dir_task)

    imgcrop = ImageCropper(num_threads, cropped_out_dir)
    imgcrop.run_cropping(lists, overwrite_existing=override)
    shutil.copy(join(nnUNet_raw_data, task_string, "dataset.json"), cropped_out_dir)


def analyze_dataset(task_string, override=False, collect_intensityproperties=True, num_processes=default_num_threads):
    cropped_out_dir = join(nnUNet_cropped_data, task_string)
    dataset_analyzer = DatasetAnalyzer(cropped_out_dir, overwrite=override, num_processes=num_processes)
    _ = dataset_analyzer.analyze_dataset(collect_intensityproperties)


def plan_and_preprocess(task_string, processes_lowres=default_num_threads, processes_fullres=3, no_preprocessing=False):
    from nnunet.experiment_planning.experiment_planner_baseline_2DUNet import ExperimentPlanner2D
    from nnunet.experiment_planning.experiment_planner_baseline_3DUNet import ExperimentPlanner

    preprocessing_output_dir_this_task_train = join(preprocessing_output_dir, task_string)
    cropped_out_dir = join(nnUNet_cropped_data, task_string)
    maybe_mkdir_p(preprocessing_output_dir_this_task_train)

    shutil.copy(join(cropped_out_dir, "dataset_properties.pkl"), preprocessing_output_dir_this_task_train)
    shutil.copy(join(nnUNet_raw_data, task_string, "dataset.json"), preprocessing_output_dir_this_task_train)

    exp_planner = ExperimentPlanner(cropped_out_dir, preprocessing_output_dir_this_task_train)
    exp_planner.plan_experiment()
    if not no_preprocessing:
        exp_planner.run_preprocessing((processes_lowres, processes_fullres))

    exp_planner = ExperimentPlanner2D(cropped_out_dir, preprocessing_output_dir_this_task_train)
    exp_planner.plan_experiment()
    if not no_preprocessing:
        exp_planner.run_preprocessing(processes_fullres)

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


def add_classes_in_slice_info(args):
    """
    We need this for 2D dataloader with oversampling. As of now it will detect slices that contain specific classes
    at run time, meaning it needs to iterate over an entire patient just to extract one slice. That is obviously bad,
    so we are doing this once beforehand and just give the dataloader the info it needs in the patients pkl file.

    """
    npz_file, pkl_file, all_classes = args
    seg_map = np.load(npz_file)['data'][-1]
    with open(pkl_file, 'rb') as f:
        props = pickle.load(f)
    #if props.get('classes_in_slice_per_axis') is not None:
    print(pkl_file)
    # this will be a dict of dict where the first dict encodes the axis along which a slice is extracted in its keys.
    # The second dict (value of first dict) will have all classes as key and as values a list of all slice ids that
    # contain this class
    classes_in_slice = OrderedDict()
    for axis in range(3):
        other_axes = tuple([i for i in range(3) if i != axis])
        classes_in_slice[axis] = OrderedDict()
        for c in all_classes:
            valid_slices = np.where(np.sum(seg_map == c, axis=other_axes) > 0)[0]
            classes_in_slice[axis][c] = valid_slices

    number_of_voxels_per_class = OrderedDict()
    for c in all_classes:
        number_of_voxels_per_class[c] = np.sum(seg_map == c)

    props['classes_in_slice_per_axis'] = classes_in_slice
    props['number_of_voxels_per_class'] = number_of_voxels_per_class

    with open(pkl_file, 'wb') as f:
        pickle.dump(props, f)
