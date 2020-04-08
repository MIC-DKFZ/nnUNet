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


import numpy as np
from collections import OrderedDict

from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.dataset_conversion.Task043_BraTS_2019 import copy_BraTS_segmentation_and_convert_labels
from nnunet.paths import nnUNet_raw_data
import SimpleITK as sitk
import shutil


def convert_labels_back_to_BraTS(seg: np.ndarray):
    new_seg = np.zeros_like(seg)
    new_seg[seg == 1] = 2
    new_seg[seg == 3] = 4
    new_seg[seg == 2] = 1
    return new_seg


def convert_labels_back_to_BraTS_2018_2019_convention(input_folder: str, output_folder: str):
    """
    reads all prediction files (nifti) in the input folder, converts the labels back to BraTS convention and saves the
    result in output_folder
    :param input_folder:
    :param output_folder:
    :return:
    """
    maybe_mkdir_p(output_folder)
    nii = subfiles(input_folder, suffix='.nii.gz', join=False)
    for n in nii:
        a = sitk.ReadImage(join(input_folder, n))
        b = sitk.GetArrayFromImage(a)
        c = convert_labels_back_to_BraTS(b)
        d = sitk.GetImageFromArray(c)
        d.CopyInformation(a)
        sitk.WriteImage(d, join(output_folder, n))


if __name__ == "__main__":
    """
    REMEMBER TO CONVERT LABELS BACK TO BRATS CONVENTION AFTER PREDICTION!
    """

    task_name = "Task032_BraTS2018"
    downloaded_data_dir = "/home/fabian/Downloads/BraTS2018_train_val_test_data/MICCAI_BraTS_2018_Data_Training"

    target_base = join(nnUNet_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_imagesVal = join(target_base, "imagesVal")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTr = join(target_base, "labelsTr")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_imagesVal)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)

    patient_names = []
    for tpe in ["HGG", "LGG"]:
        cur = join(downloaded_data_dir, tpe)
        for p in subdirs(cur, join=False):
            patdir = join(cur, p)
            patient_name = tpe + "__" + p
            patient_names.append(patient_name)
            t1 = join(patdir, p + "_t1.nii.gz")
            t1c = join(patdir, p + "_t1ce.nii.gz")
            t2 = join(patdir, p + "_t2.nii.gz")
            flair = join(patdir, p + "_flair.nii.gz")
            seg = join(patdir, p + "_seg.nii.gz")

            assert all([
                isfile(t1),
                isfile(t1c),
                isfile(t2),
                isfile(flair),
                isfile(seg)
            ]), "%s" % patient_name

            shutil.copy(t1, join(target_imagesTr, patient_name + "_0000.nii.gz"))
            shutil.copy(t1c, join(target_imagesTr, patient_name + "_0001.nii.gz"))
            shutil.copy(t2, join(target_imagesTr, patient_name + "_0002.nii.gz"))
            shutil.copy(flair, join(target_imagesTr, patient_name + "_0003.nii.gz"))

            copy_BraTS_segmentation_and_convert_labels(seg, join(target_labelsTr, patient_name + ".nii.gz"))

    json_dict = OrderedDict()
    json_dict['name'] = "BraTS2018"
    json_dict['description'] = "nothing"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see BraTS2018"
    json_dict['licence'] = "see BraTS2019 license"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "T1",
        "1": "T1ce",
        "2": "T2",
        "3": "FLAIR"
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "edema",
        "2": "non-enhancing",
        "3": "enhancing",
    }
    json_dict['numTraining'] = len(patient_names)
    json_dict['numTest'] = 0
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                             patient_names]
    json_dict['test'] = []

    save_json(json_dict, join(target_base, "dataset.json"))

    del tpe, cur
    downloaded_data_dir = "/home/fabian/Downloads/BraTS2018_train_val_test_data/MICCAI_BraTS_2018_Data_Validation"

    for p in subdirs(downloaded_data_dir, join=False):
        patdir = join(downloaded_data_dir, p)
        patient_name = p
        t1 = join(patdir, p + "_t1.nii.gz")
        t1c = join(patdir, p + "_t1ce.nii.gz")
        t2 = join(patdir, p + "_t2.nii.gz")
        flair = join(patdir, p + "_flair.nii.gz")

        assert all([
            isfile(t1),
            isfile(t1c),
            isfile(t2),
            isfile(flair),
        ]), "%s" % patient_name

        shutil.copy(t1, join(target_imagesVal, patient_name + "_0000.nii.gz"))
        shutil.copy(t1c, join(target_imagesVal, patient_name + "_0001.nii.gz"))
        shutil.copy(t2, join(target_imagesVal, patient_name + "_0002.nii.gz"))
        shutil.copy(flair, join(target_imagesVal, patient_name + "_0003.nii.gz"))

    downloaded_data_dir = "/home/fabian/Downloads/BraTS2018_train_val_test_data/MICCAI_BraTS_2018_Data_Testing_FIsensee"

    for p in subdirs(downloaded_data_dir, join=False):
        patdir = join(downloaded_data_dir, p)
        patient_name = p
        t1 = join(patdir, p + "_t1.nii.gz")
        t1c = join(patdir, p + "_t1ce.nii.gz")
        t2 = join(patdir, p + "_t2.nii.gz")
        flair = join(patdir, p + "_flair.nii.gz")

        assert all([
            isfile(t1),
            isfile(t1c),
            isfile(t2),
            isfile(flair),
        ]), "%s" % patient_name

        shutil.copy(t1, join(target_imagesTs, patient_name + "_0000.nii.gz"))
        shutil.copy(t1c, join(target_imagesTs, patient_name + "_0001.nii.gz"))
        shutil.copy(t2, join(target_imagesTs, patient_name + "_0002.nii.gz"))
        shutil.copy(flair, join(target_imagesTs, patient_name + "_0003.nii.gz"))
