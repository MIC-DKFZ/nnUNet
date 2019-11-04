import shutil
from collections import OrderedDict

import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isfile, join, maybe_mkdir_p, save_json, subdirs

from nnunet.paths import splitted_4d_output_dir


def copy_nifti_modify_labels(in_file, out_file):
    # use this for segmentation only!!!
    # nnUNet wants the labels to be continuous. BraTS is 0, 1, 2, 4 -> we make that into 0, 1, 2, 3
    # no sanity checks -> #Yolo
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)
    seg_new = np.zeros_like(img_npy)
    seg_new[img_npy == 4] = 3
    seg_new[img_npy == 2] = 1
    seg_new[img_npy == 1] = 2
    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)


if __name__ == "__main__":
    """
    REMEMBER TO CONVERT LABELS BACK TO BRATS CONVENTION AFTER PREDICTION!
    """

    task_name = "Task43_BraTS2019"
    downloaded_data_dir = "/home/fabian/drives/E132-Projekte/Move_to_E132-Rohdaten/BraTS_2019/MICCAI_BraTS_2019_Data_Training"

    target_base = join(splitted_4d_output_dir, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_labelsTr = join(target_base, "labelsTr")

    maybe_mkdir_p(target_imagesTr)
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

            copy_nifti_modify_labels(seg, join(target_labelsTr, patient_name + ".nii.gz"))


    json_dict = OrderedDict()
    json_dict['name'] = "BraTS2019"
    json_dict['description'] = "nothing"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see BraTS2019"
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
