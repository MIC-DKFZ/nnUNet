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
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import numpy as np
from numpy.random.mtrand import RandomState
import subprocess
from multiprocessing import pool
import pandas as pd
from nnunet.experiment_planning.common_utils import split_4d_nifti


def get_mnms_data(data_root):
    files_raw = []
    files_gt = []
    for r, dirs, files in os.walk(data_root):
        for f in files:
            if f.endswith('nii.gz'):
                file_path = os.path.join(r, f)
                if '_gt' in f:
                    files_gt.append(file_path)
                else:
                    files_raw.append(file_path)
    return files_raw, files_gt


def generate_filename_for_nnunet(pat_id, ts, pat_folder=None, add_zeros=False, vendor=None, centre=None, mode='mnms',
                                 data_format='nii.gz'):
    if not vendor or not centre:
        if add_zeros:
            filename = "{}_{}_0000.{}".format(pat_id, str(ts).zfill(4), data_format)
        else:
            filename = "{}_{}.{}".format(pat_id, str(ts).zfill(4), data_format)
    else:
        if mode == 'mnms':
            if add_zeros:
                filename = "{}_{}_{}_{}_0000.{}".format(pat_id, str(ts).zfill(4), vendor, centre, data_format)
            else:
                filename = "{}_{}_{}_{}.{}".format(pat_id, str(ts).zfill(4), vendor, centre, data_format)
        else:
            if add_zeros:
                filename = "{}_{}_{}_{}_0000.{}".format(vendor, centre, pat_id, str(ts).zfill(4), data_format)
            else:
                filename = "{}_{}_{}_{}.{}".format(vendor, centre, pat_id, str(ts).zfill(4), data_format)

    if pat_folder:
        filename = os.path.join(pat_folder, filename)
    return filename


def select_annotated_frames_mms(data_folder, out_folder, add_zeros=False, mode='mnms',
                                df_path="/media/full/tera2/data/challenges/mms/Training-corrected_original/M&Ms Dataset Information.xlsx"):
    table = pd.read_excel(df_path, index_col='External code')

    raw_image_paths, raw_gt_paths = get_mnms_data(data_folder)
    combine_paths = raw_image_paths + raw_gt_paths # not so nice but only one of the lists will have entreis

    for idx in table.index:
        ed = table.loc[idx, 'ED']
        es = table.loc[idx, 'ES']
        vendor = table.loc[idx, 'Vendor']
        centre = table.loc[idx, 'Centre']

        if vendor != "C": # vendor C is for test data

            # generate old filename (w/o vendor and centre)
            filename_ed_original = combine_paths[
                [(idx in x and str(ed).zfill(4) in x) for x in raw_image_paths] == True]
            filename_es_original = combine_paths[
                [(idx in x and str(es).zfill(4) in x) for x in raw_image_paths] == True]

            # generate new filename with vendor and centre
            filename_ed = generate_filename_for_nnunet(pat_id=idx, ts=ed, pat_folder=out_folder,
                                                       vendor=vendor, centre=centre, add_zeros=add_zeros, mode=mode)
            filename_es = generate_filename_for_nnunet(pat_id=idx, ts=es, pat_folder=out_folder,
                                                       vendor=vendor, centre=centre, add_zeros=add_zeros, mode=mode)

            shutil.copy(filename_ed_original, filename_ed)
            shutil.copy(filename_es_original, filename_es)


def create_custom_splits_for_experiments(task_path):
    data_keys = [i[:-4] for i in
                 subfiles(os.path.join(task_path, "nnUNetData_plans_v2.1_2D_stage0"),
                          join=False, suffix='npz')]
    existing_splits = os.path.join(task_path, "splits_final.pkl")

    splits = load_pickle(existing_splits)
    splits = splits[:5]  # discard old changes

    unique_a_only = np.unique([i.split('_')[0] for i in data_keys if i.find('_A_') != -1])
    unique_b_only = np.unique([i.split('_')[0] for i in data_keys if i.find('_B_') != -1])

    num_train_a = int(np.round(0.8 * len(unique_a_only)))
    num_train_b = int(np.round(0.8 * len(unique_b_only)))

    p = RandomState(1234)
    idx_a_train = p.choice(len(unique_a_only), num_train_a, replace=False)
    idx_b_train = p.choice(len(unique_b_only), num_train_b, replace=False)

    identifiers_a_train = [unique_a_only[i] for i in idx_a_train]
    identifiers_b_train = [unique_b_only[i] for i in idx_b_train]

    identifiers_a_val = [i for i in unique_a_only if i not in identifiers_a_train]
    identifiers_b_val = [i for i in unique_b_only if i not in identifiers_b_train]

    # fold 5 will be train on a and eval on val sets of a and b
    splits.append({'train': [i for i in data_keys if i.split("_")[0] in identifiers_a_train],
                   'val': [i for i in data_keys if i.split("_")[0] in identifiers_a_val] + [i for i in data_keys if
                                                                                            i.split("_")[
                                                                                                0] in identifiers_b_val]})

    # fold 6 will be train on b and eval on val sets of a and b
    splits.append({'train': [i for i in data_keys if i.split("_")[0] in identifiers_b_train],
                   'val': [i for i in data_keys if i.split("_")[0] in identifiers_a_val] + [i for i in data_keys if
                                                                                            i.split("_")[
                                                                                                0] in identifiers_b_val]})

    # fold 7 train on both, eval on both
    splits.append({'train': [i for i in data_keys if i.split("_")[0] in identifiers_b_train] + [i for i in data_keys if i.split("_")[0] in identifiers_a_train],
                   'val': [i for i in data_keys if i.split("_")[0] in identifiers_a_val] + [i for i in data_keys if
                                                                                            i.split("_")[
                                                                                                0] in identifiers_b_val]})
    save_pickle(splits, existing_splits)


if __name__ == "__main__":
    # this script will split 4d data from the M&Ms data set into 3d images for both, raw images and gt annotations.
    # after this script you will be able to start a training on the M&Ms data.
    # use this script as insipration in case other data than M&Ms data is use for training.
    #
    # check also the comments at the END of the script for instructions on how to run the actual training after this
    # script
    #

    # define a task ID for your experiment (I have choosen 114)
    task_name = "Task114_heart_mnms"
    # this is where the downloaded data from the M&Ms challenge shall be placed
    raw_data_dir = "/media/full/tera2/data"
    # don't make changes here
    folder_imagesTr = "imagesTr"
    train_dir = os.path.join(raw_data_dir, task_name, folder_imagesTr)

    # this is where our your splitted files WITH annotation will be stored. Dont make changes here. Otherwise nnUNet
    # might have problems finding the training data later during the training process
    out_dir = os.path.join(os.environ.get('nnUNet_raw_data_base'), 'nnUNet_raw_data', task_name)

    files_raw, files_gt = get_mnms_data(data_root=train_dir)

    filesTs, _ = get_mnms_data(data_root=train_dir)

    split_path_raw_all_ts = os.path.join(raw_data_dir, task_name, "splitted_all_timesteps", folder_imagesTr,
                                         "split_raw_images")
    split_path_gt_all_ts = os.path.join(raw_data_dir, task_name, "splitted_all_timesteps", folder_imagesTr,
                                        "split_annotation")
    maybe_mkdir_p(split_path_raw_all_ts)
    maybe_mkdir_p(split_path_gt_all_ts)

    # for fast splitting of many patients use the following lines
    # however keep in mind that these lines cause problems for some users.
    # If problems occur use the code for loops below
    # print("splitting raw 4d images into 3d images")
    # split_4d_for_all_pat(files_raw, split_path_raw)
    # print("splitting ground truth 4d into 3d files")
    # split_4d_for_all_pat(files_gt, split_path_gt_all_ts)

    print("splitting raw 4d images into 3d images")
    for f in files_raw:
        print("splitting {}".format(f))
        split_4d_nifti(f, split_path_raw_all_ts)
    print("splitting ground truth 4d into 3d files")
    for gt in files_gt:
        split_4d_nifti(gt, split_path_gt_all_ts)
        print("splitting {}".format(gt))

    print("prepared data will be saved at: {}".format(out_dir))
    maybe_mkdir_p(join(out_dir, "imagesTr"))
    maybe_mkdir_p(join(out_dir, "labelsTr"))

    imagesTr_path = os.path.join(out_dir, "imagesTr")
    labelsTr_path = os.path.join(out_dir, "labelsTr")
    # only a small fraction of all timestep in the cardiac cycle possess gt annotation. These timestep will now be
    # selected
    select_annotated_frames_mms(split_path_raw_all_ts, imagesTr_path, add_zeros=True)
    select_annotated_frames_mms(split_path_gt_all_ts, labelsTr_path, add_zeros=False)

    labelsTr = subfiles(labelsTr_path)

    # create a json file that will be needed by nnUNet to initiate the preprocessing process
    json_dict = OrderedDict()
    json_dict['name'] = "M&Ms"
    json_dict['description'] = "short axis cardiac cine MRI segmentation"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "Campello, Víctor M. et al.: Multi-Centre, Multi-Vendor & Multi-Disease Cardiac Image " \
                             "Segmentation. In preparation."
    json_dict['licence'] = "see M&Ms challenge"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "MRI",
    }
    # labels differ for ACDC challenge
    json_dict['labels'] = {
        "0": "background",
        "1": "LVBP",
        "2": "LVM",
        "3": "RV"
    }
    json_dict['numTraining'] = len(labelsTr)
    json_dict['numTest'] = 0
    json_dict['training'] = [{'image': "./imagesTr/%s" % i.split("/")[-1],
                              "label": "./labelsTr/%s" % i.split("/")[-1]} for i in labelsTr]
    json_dict['test'] = []

    save_json(json_dict, os.path.join(out_dir, "dataset.json"))

    #
    # now the data is ready to be preprocessed by the nnUNet
    # the following steps are only needed if you want to reproduce the exact results from the MMS challenge
    #


    # then preprocess data and plan training.
    # run in terminal
    # nnUNet_plan_and_preprocess -t 114 --verify_dataset_integrity # for 2d
    # nnUNet_plan_and_preprocess -t 114 --verify_dataset_integrity -pl3d ExperimentPlannerTargetSpacingForAnisoAxis # for 3d

    # start training and stop it immediately to get a split.pkl file
    # nnUNet_train 2d nnUNetTrainerV2_MMS 114 0

    #
    # then create custom splits as used for the final M&Ms submission
    #

    # in this file comment everything except for the following line
    # create_custom_splits_for_experiments(out_dir)

    # then start training with
    #
    # nnUNet_train 3d_fullres nnUNetTrainerV2_MMS Task114_heart_mnms -p nnUNetPlanstargetSpacingForAnisoAxis 0 # for 3d and fold 0
    # and
    # nnUNet_train 2d nnUNetTrainerV2_MMS Task114_heart_mnms 0 # for 2d and fold 0


