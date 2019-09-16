import shutil
from collections import OrderedDict

import dicom2nifti
import numpy as np
from batchgenerators.utilities.data_splitting import get_split_deterministic
from batchgenerators.utilities.file_and_folder_operations import *
from PIL import Image
import SimpleITK as sitk
from nnunet.paths import preprocessing_output_dir, splitted_4d_output_dir
from nnunet.utilities.sitk_stuff import copy_geometry


def load_png_stack(folder):
    pngs = subfiles(folder, suffix="png")
    pngs.sort()
    loaded = []
    for p in pngs:
        loaded.append(np.array(Image.open(p)))
    loaded = np.stack(loaded, 0)[::-1]
    return loaded


def convert_CT_seg(loaded_png):
    return loaded_png.astype(np.uint16)


def convert_MR_seg(loaded_png):
    result = np.zeros(loaded_png.shape)
    result[(loaded_png > 55) & (loaded_png <= 70)] = 1
    result[(loaded_png > 110) & (loaded_png <= 135)] = 2
    result[(loaded_png > 175) & (loaded_png <= 200)] = 3
    result[(loaded_png > 240) & (loaded_png <= 255)] = 4
    return result


if __name__ == "__main__":
    """
    This script only prepares data to participate in Task 5 and Task 5. I don't like the CT task because 
    1) there are 
    no abdominal organs in the ground truth. In the case of CT we are supposed to train only liver while on MRI we are 
    supposed to train all organs. This would require manual modification of nnU-net to deal with this dataset. This is 
    not what nnU-net is about.
    2) CT Liver or multiorgan segmentation is too easy to get external data for. Therefore the challenges comes down 
    to who gets the b est external data, not who has the best algorithm. Not super interesting.
    
    Task 3 is a subtask of Task 5 so we need to prepare the data only once.
    Difficulty: We need to process both T1 and T2, but T1 has 2 'modalities' (phases). nnU-Net cannot handly varying 
    number of input channels. We need to be creative.
    We deal with this by preparing 2 Variants:
    1) pretend we have 2 modalities for T2 as well by simply stacking a copy of the data
    2) treat all MRI sequences independently, so we now have 3*20 training data instead of 2*20. In inference we then 
    ensemble the results for the two t1 modalities. 
    
    Careful: We need to split manually here to ensure we stratify by patient
    """

    root = "/media/fabian/My Book/datasets/CHAOS_challenge/Train_Sets"
    out_base = splitted_4d_output_dir
    # CT
    # we ignore CT because

    ##############################################################
    # Variant 1
    ##############################################################
    patient_ids = []

    output_folder = join(out_base, "Task37_CHAOS_Task_3_5_Variant1")
    output_images = join(output_folder, "imagesTr")
    output_labels = join(output_folder, "labelsTr")
    maybe_mkdir_p(output_images)
    maybe_mkdir_p(output_labels)

    d = join(root, "MR")

    # Process T1
    patients = subdirs(d, join=False)
    for p in patients:
        patient_name = "T1_" + p
        gt_dir = join(d, p, "T1DUAL", "Ground")
        seg = convert_MR_seg(load_png_stack(gt_dir)[::-1])

        img_dir = join(d, p, "T1DUAL", "DICOM_anon", "InPhase")
        img_outfile = join(output_images, patient_name + "_0000.nii.gz")
        _ = dicom2nifti.convert_dicom.dicom_series_to_nifti(img_dir, img_outfile, reorient_nifti=False)

        img_dir = join(d, p, "T1DUAL", "DICOM_anon", "OutPhase")
        img_outfile = join(output_images, patient_name + "_0001.nii.gz")
        _ = dicom2nifti.convert_dicom.dicom_series_to_nifti(img_dir, img_outfile, reorient_nifti=False)

        img_sitk = sitk.ReadImage(img_outfile)
        img_sitk_npy = sitk.GetArrayFromImage(img_sitk)
        seg_itk = sitk.GetImageFromArray(seg.astype(np.uint8))
        seg_itk = copy_geometry(seg_itk, img_sitk)
        sitk.WriteImage(seg_itk, join(output_labels, patient_name + ".nii.gz"))
        patient_ids.append(patient_name)

    # Process T2
    patients = subdirs(d, join=False)
    for p in patients:
        patient_name = "T2_" + p

        gt_dir = join(d, p, "T2SPIR", "Ground")
        seg = convert_MR_seg(load_png_stack(gt_dir)[::-1])

        img_dir = join(d, p, "T2SPIR", "DICOM_anon")
        img_outfile = join(output_images, patient_name + "_0000.nii.gz")
        _ = dicom2nifti.convert_dicom.dicom_series_to_nifti(img_dir, img_outfile, reorient_nifti=False)
        shutil.copy(join(output_images, patient_name + "_0000.nii.gz"), join(output_images, patient_name + "_0001.nii.gz"))

        img_sitk = sitk.ReadImage(img_outfile)
        img_sitk_npy = sitk.GetArrayFromImage(img_sitk)
        seg_itk = sitk.GetImageFromArray(seg.astype(np.uint8))
        seg_itk = copy_geometry(seg_itk, img_sitk)
        sitk.WriteImage(seg_itk, join(output_labels, patient_name + ".nii.gz"))
        patient_ids.append(patient_name)

    json_dict = OrderedDict()
    json_dict['name'] = "Chaos Challenge Task3/5 Variant 1"
    json_dict['description'] = "nothing"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "https://chaos.grand-challenge.org/Data/"
    json_dict['licence'] = "see https://chaos.grand-challenge.org/Data/"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "MRI",
        "1": "MRI",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "liver",
        "2": "right kidney",
        "3": "left kidney",
        "4": "spleen",
    }
    json_dict['numTraining'] = len(patient_ids)
    json_dict['numTest'] = 0
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                             patient_ids]
    json_dict['test'] = []

    save_json(json_dict, join(output_folder, "dataset.json"))

    ##############################################################
    # Variant 2
    ##############################################################

    patient_ids = []

    output_folder = join(out_base, "Task38_CHAOS_Task_3_5_Variant2")
    output_images = join(output_folder, "imagesTr")
    output_labels = join(output_folder, "labelsTr")
    maybe_mkdir_p(output_images)
    maybe_mkdir_p(output_labels)

    # Process T1
    patients = subdirs(d, join=False)
    for p in patients:
        patient_name_in = "T1_in_" + p
        patient_name_out = "T1_out_" + p
        gt_dir = join(d, p, "T1DUAL", "Ground")
        seg = convert_MR_seg(load_png_stack(gt_dir)[::-1])

        img_dir = join(d, p, "T1DUAL", "DICOM_anon", "InPhase")
        img_outfile = join(output_images, patient_name_in + "_0000.nii.gz")
        _ = dicom2nifti.convert_dicom.dicom_series_to_nifti(img_dir, img_outfile, reorient_nifti=False)

        img_dir = join(d, p, "T1DUAL", "DICOM_anon", "OutPhase")
        img_outfile = join(output_images, patient_name_out + "_0000.nii.gz")
        _ = dicom2nifti.convert_dicom.dicom_series_to_nifti(img_dir, img_outfile, reorient_nifti=False)

        img_sitk = sitk.ReadImage(img_outfile)
        img_sitk_npy = sitk.GetArrayFromImage(img_sitk)
        seg_itk = sitk.GetImageFromArray(seg.astype(np.uint8))
        seg_itk = copy_geometry(seg_itk, img_sitk)
        sitk.WriteImage(seg_itk, join(output_labels, patient_name_in + ".nii.gz"))
        sitk.WriteImage(seg_itk, join(output_labels, patient_name_out + ".nii.gz"))
        patient_ids.append(patient_name_out)
        patient_ids.append(patient_name_in)

    # Process T2
    patients = subdirs(d, join=False)
    for p in patients:
        patient_name = "T2_" + p

        gt_dir = join(d, p, "T2SPIR", "Ground")
        seg = convert_MR_seg(load_png_stack(gt_dir)[::-1])

        img_dir = join(d, p, "T2SPIR", "DICOM_anon")
        img_outfile = join(output_images, patient_name + "_0000.nii.gz")
        _ = dicom2nifti.convert_dicom.dicom_series_to_nifti(img_dir, img_outfile, reorient_nifti=False)

        img_sitk = sitk.ReadImage(img_outfile)
        img_sitk_npy = sitk.GetArrayFromImage(img_sitk)
        seg_itk = sitk.GetImageFromArray(seg.astype(np.uint8))
        seg_itk = copy_geometry(seg_itk, img_sitk)
        sitk.WriteImage(seg_itk, join(output_labels, patient_name + ".nii.gz"))
        patient_ids.append(patient_name)

    json_dict = OrderedDict()
    json_dict['name'] = "Chaos Challenge Task3/5 Variant 2"
    json_dict['description'] = "nothing"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "https://chaos.grand-challenge.org/Data/"
    json_dict['licence'] = "see https://chaos.grand-challenge.org/Data/"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "MRI",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "liver",
        "2": "right kidney",
        "3": "left kidney",
        "4": "spleen",
    }
    json_dict['numTraining'] = len(patient_ids)
    json_dict['numTest'] = 0
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                             patient_ids]
    json_dict['test'] = []

    save_json(json_dict, join(output_folder, "dataset.json"))

    #################################################
    # custom split
    #################################################
    patients = subdirs(d, join=False)
    task_name_variant1 = "Task37_CHAOS_Task_3_5_Variant1"
    task_name_variant2 = "Task38_CHAOS_Task_3_5_Variant2"

    output_preprocessed_v1 = join(preprocessing_output_dir, task_name_variant1)
    maybe_mkdir_p(output_preprocessed_v1)

    output_preprocessed_v2 = join(preprocessing_output_dir, task_name_variant2)
    maybe_mkdir_p(output_preprocessed_v2)

    splits = []
    for fold in range(5):
        tr, val = get_split_deterministic(patients, fold, 5, 12345)
        train = ["T2_" + i for i in tr] + ["T1_" + i for i in tr]
        validation = ["T2_" + i for i in val] + ["T1_" + i for i in val]
        splits.append({
            'train': train,
            'val': validation
        })
    save_pickle(splits, join(output_preprocessed_v1, "splits_final.pkl"))

    splits = []
    for fold in range(5):
        tr, val = get_split_deterministic(patients, fold, 5, 12345)
        train = ["T2_" + i for i in tr] + ["T1_in_" + i for i in tr] + ["T1_out_" + i for i in tr]
        validation = ["T2_" + i for i in val] + ["T1_in_" + i for i in val] + ["T1_out_" + i for i in val]
        splits.append({
            'train': train,
            'val': validation
        })
    save_pickle(splits, join(output_preprocessed_v2, "splits_final.pkl"))

