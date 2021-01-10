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


def select_annotated_frames_mms(data_folder, out_folder, add_zeros=False, mode='mnms', df_path="/media/full/tera2/data/challenges/mms/Training-corrected_original/M&Ms Dataset Information.xlsx"):
    table = pd.read_excel(df_path, index_col='External code')

    for idx in table.index:
        ed = table.loc[idx, 'ED']
        es = table.loc[idx, 'ES']
        vendor = table.loc[idx, 'Vendor']
        centre = table.loc[idx, 'Centre']

        if vendor != "C":

            # generate old filename (w/o vendor and centre)
            filename_ed_original = generate_filename_for_nnunet(pat_id=idx, ts=ed, pat_folder=data_folder,
                                                       vendor=None, centre=None, add_zeros=False)
            filename_es_original = generate_filename_for_nnunet(pat_id=idx, ts=es, pat_folder=data_folder,
                                                       vendor=None, centre=None, add_zeros=False)

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

def split_4d_nii(nii_path, split_folder, pat_name=None, add_zeros=False):

    # create temporary folder in which the 3d+t file will be split into many 3d files
    temp_base = os.path.dirname(nii_path)
    temp_location = os.path.join(temp_base, 'tmp')
    if not os.path.isdir(temp_location):
        os.mkdir(temp_location)
    os.chdir(temp_location)

    if not os.path.isdir(split_folder):
        os.mkdir(split_folder)
    _ = subprocess.call(['fslsplit', nii_path])

    # rename files so that the patient's ID is in the filename
    file_list = [f for f in os.listdir(temp_location) if os.path.isfile(f)]
    file_list = sorted(file_list)

    if not pat_name:
        pat_name = os.path.basename(os.path.dirname(nii_path))

    for ts, temp_file in enumerate(file_list):
        # get time
        time_step = temp_file.split('.')[0][3:]
        # make sure the time step is a number. Otherwise trust in pythons sort algorithm
        try:
            int(time_step)
        except:
            time_step = ts

        # change filename AND location -> move files
        if add_zeros:
            new_file_name = '{}_{}_0000.nii.gz'.format(pat_name, time_step)
        else:
            new_file_name = '{}_{}.nii.gz'.format(pat_name, time_step)
        os.rename(os.path.join(temp_location, temp_file),
                  os.path.join(split_folder, new_file_name))

    os.rmdir(temp_location)

def split_4d_parallel(args):
    nii_path, split_folder, pat_name = args
    split_4d_nii(nii_path, split_folder, pat_name)


def split_4d_for_all_pat(files_paths, split_folder):
    p = pool.Pool(8)
    p.map(split_4d_parallel,
          zip(files_paths, [split_folder] * len(files_paths), [None] * len(files_paths)))

if __name__ == "__main__":
    task_name = "Task114_heart_MNMs"
    train_dir = "/media/full/97d8d6e1-1aa1-4761-9dd1-fc6a62cf6264/nnUnet_raw/nnUNet_raw_data/{}/imagesTr".format(task_name)
    test_dir = "/media/full/97d8d6e1-1aa1-4761-9dd1-fc6a62cf6264/nnUnet_raw/nnUNet_raw_data/{}/imagesTs".format(task_name)
    #out_dir='/media/full/tera2/output_nnUNet/preprocessed_data/Task114_heart_mnms'
    out_dir='/media/full/97d8d6e1-1aa1-4761-9dd1-fc6a62cf6264/tmp'

    # train
    all_train_files = [os.path.join(train_dir, x) for x in os.listdir(train_dir)]
    # test
    all_test_files = [os.path.join(test_dir, x) for x in os.listdir(test_dir)]

    data_root = '/media/full/97d8d6e1-1aa1-4761-9dd1-fc6a62cf6264/data/challenges/mms/Training-corrected_original/Labeled'
    files_raw, files_gt = get_mnms_data(data_root=data_root)
    split_path_raw ='/media/full/97d8d6e1-1aa1-4761-9dd1-fc6a62cf6264/data/challenges/mms/temp_split_raw'
    split_path_gt ='/media/full/97d8d6e1-1aa1-4761-9dd1-fc6a62cf6264/data/challenges/mms/temp_split_gt'
    maybe_mkdir_p(split_path_raw)
    maybe_mkdir_p(split_path_gt)

    split_4d_for_all_pat(files_raw, split_path_raw)
    split_4d_for_all_pat(files_gt, split_path_gt)

    out_dir = '/media/full/97d8d6e1-1aa1-4761-9dd1-fc6a62cf6264/nnUnet_raw/nnUNet_raw_data/{}/'.format(task_name)

    maybe_mkdir_p(join(out_dir, "imagesTr"))
    maybe_mkdir_p(join(out_dir, "imagesTs"))
    maybe_mkdir_p(join(out_dir, "labelsTr"))

    imagesTr_path = os.path.join(out_dir, "imagesTr")
    labelsTr_path = os.path.join(out_dir, "labelsTr")
    select_annotated_frames_mms(split_path_raw, imagesTr_path, add_zeros=True)
    select_annotated_frames_mms(split_path_gt, labelsTr_path, add_zeros=False)

    labelsTr = subfiles(labelsTr_path)


    json_dict = OrderedDict()
    json_dict['name'] = "M&Ms"
    json_dict['description'] = "short axis cardiac cine MRI segmentation"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "Campello, Víctor M. et al.: Multi-Centre, Multi-Vendor & Multi-Disease Cardiac Image Segmentation. In preparation."
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
    json_dict['training'] = [{'image': "./imagesTr/%s" % i.split("/")[-1], "label": "./labelsTr/%s" % i.split("/")[-1]} for i in
                             labelsTr]
    json_dict['test'] = []

    save_json(json_dict, os.path.join(out_dir, "dataset.json"))

    # then preprocess data and plan training.
    # run in terminal
    # > nnUNet_plan_and_preprocess -t <TaskID> --verify_dataset_integrity

    # start training and stop it immediately to get a split.pkl file
    # > nnUNet_train 2d nnUNetTrainerV2_MMS <TaskID> 0

    #
    # then create custom splits as used for the final M&Ms submission
    #

    split_file_path = '/media/full/97d8d6e1-1aa1-4761-9dd1-fc6a62cf6264/output_nnUNet/preprocessed_data/{}/'.format(task_name)

    create_custom_splits_for_experiments(split_file_path)

