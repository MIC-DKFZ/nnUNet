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


import SimpleITK as sitk
import dicom2nifti
import numpy as np
from PIL import Image
from batchgenerators.utilities.data_splitting import get_split_deterministic
from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed


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
    result[(loaded_png > 55) & (loaded_png <= 70)] = 1 # liver
    result[(loaded_png > 110) & (loaded_png <= 135)] = 2 # right kidney
    result[(loaded_png > 175) & (loaded_png <= 200)] = 3 # left kidney
    result[(loaded_png > 240) & (loaded_png <= 255)] = 4 # spleen
    return result


def convert_seg_to_intensity_task5(seg):
    seg_new = np.zeros(seg.shape, dtype=np.uint8)
    seg_new[seg == 1] = 63
    seg_new[seg == 2] = 126
    seg_new[seg == 3] = 189
    seg_new[seg == 4] = 252
    return seg_new


def convert_seg_to_intensity_task3(seg):
    seg_new = np.zeros(seg.shape, dtype=np.uint8)
    seg_new[seg == 1] = 63
    return seg_new


def write_pngs_from_nifti(nifti, output_folder, converter=convert_seg_to_intensity_task3):
    npy = sitk.GetArrayFromImage(sitk.ReadImage(nifti))
    seg_new = converter(npy)
    for z in range(len(npy)):
        Image.fromarray(seg_new[z]).save(join(output_folder, "img%03.0d.png" % z))



if __name__ == "__main__":
    """
    This script only prepares data to participate in Task 3 and Task 5. I don't like the CT task because 
    1) there are 
    no abdominal organs in the ground truth. In the case of CT we are supposed to train only liver while on MRI we are 
    supposed to train all organs. This would require manual modification of nnU-net to deal with this dataset. This is 
    not what nnU-net is about.
    2) CT Liver or multiorgan segmentation is too easy to get external data for. Therefore the challenges comes down 
    to who gets the best external data, not who has the best algorithm. Not super interesting.
    
    Task 3 is a subtask of Task 5 so we need to prepare the data only once.
    Difficulty: We need to process both T1 and T2, but T1 has 2 'modalities' (phases). nnU-Net cannot handly varying 
    number of input channels. We need to be creative.
    We deal with this by treating all MRI sequences independently, so we now have 3*20 training data instead of 2*20. 
    In inference we then ensemble the results for the two t1 modalities. 
    
    Careful: We need to split manually here to ensure we stratify by patient
    """

    root = "/media/isensee/Volume/raw_data/CHAOS/Train_Sets"
    root_test = "/media/isensee/Volume/raw_data/CHAOS/Test_Sets"
    out_base = nnUNet_raw

    patient_ids = []
    patient_ids_test = []

    output_folder = join(out_base, "Dataset047_CHAOS_Task_3_5_Variant2_new")
    output_images = join(output_folder, "imagesTr")
    output_imagesTs = join(output_folder, "imagesTs")
    output_labels = join(output_folder, "labelsTr")
    maybe_mkdir_p(output_images)
    maybe_mkdir_p(output_imagesTs)
    maybe_mkdir_p(output_labels)

    # Process T1 train
    d = join(root, "MR")
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
        seg_itk.SetSpacing(img_sitk.GetSpacing())
        seg_itk.SetOrigin(img_sitk.GetOrigin())
        seg_itk.SetDirection(img_sitk.GetDirection())
        sitk.WriteImage(seg_itk, join(output_labels, patient_name_in + ".nii.gz"))
        sitk.WriteImage(seg_itk, join(output_labels, patient_name_out + ".nii.gz"))
        patient_ids.append(patient_name_out)
        patient_ids.append(patient_name_in)

    # Process T1 test
    d = join(root_test, "MR")
    patients = subdirs(d, join=False)
    for p in patients:
        patient_name_in = "T1_in_" + p
        patient_name_out = "T1_out_" + p
        gt_dir = join(d, p, "T1DUAL", "Ground")

        img_dir = join(d, p, "T1DUAL", "DICOM_anon", "InPhase")
        img_outfile = join(output_imagesTs, patient_name_in + "_0000.nii.gz")
        _ = dicom2nifti.convert_dicom.dicom_series_to_nifti(img_dir, img_outfile, reorient_nifti=False)

        img_dir = join(d, p, "T1DUAL", "DICOM_anon", "OutPhase")
        img_outfile = join(output_imagesTs, patient_name_out + "_0000.nii.gz")
        _ = dicom2nifti.convert_dicom.dicom_series_to_nifti(img_dir, img_outfile, reorient_nifti=False)

        img_sitk = sitk.ReadImage(img_outfile)
        img_sitk_npy = sitk.GetArrayFromImage(img_sitk)
        patient_ids_test.append(patient_name_out)
        patient_ids_test.append(patient_name_in)

    # Process T2 train
    d = join(root, "MR")
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
        seg_itk.SetSpacing(img_sitk.GetSpacing())
        seg_itk.SetOrigin(img_sitk.GetOrigin())
        seg_itk.SetDirection(img_sitk.GetDirection())
        sitk.WriteImage(seg_itk, join(output_labels, patient_name + ".nii.gz"))
        patient_ids.append(patient_name)

    # Process T2 test
    d = join(root_test, "MR")
    patients = subdirs(d, join=False)
    for p in patients:
        patient_name = "T2_" + p

        gt_dir = join(d, p, "T2SPIR", "Ground")

        img_dir = join(d, p, "T2SPIR", "DICOM_anon")
        img_outfile = join(output_imagesTs, patient_name + "_0000.nii.gz")
        _ = dicom2nifti.convert_dicom.dicom_series_to_nifti(img_dir, img_outfile, reorient_nifti=False)

        img_sitk = sitk.ReadImage(img_outfile)
        img_sitk_npy = sitk.GetArrayFromImage(img_sitk)
        patient_ids_test.append(patient_name)

    generate_dataset_json(
        join(nnUNet_raw, 'Dataset047_CHAOS_Task_3_5_Variant2_new'),
        {0: 'MR'},
        {'background': 0, 'liver': 1, 'right kidney': 2, 'left kidney': 3, 'spleen': 4},
        len(patient_ids),
        '.nii.gz',
        dataset_name='Dataset047_CHAOS_Task_3_5_Variant2_new',
        reference='https://chaos.grand-challenge.org/Data/',
        license='see https://chaos.grand-challenge.org/Publications/',
    )

    #################################################
    # custom split
    #################################################
    patients = subdirs(join(root, "MR"), join=False)
    task_name_variant2 = 'Dataset047_CHAOS_Task_3_5_Variant2_new'

    output_preprocessed_v2 = join(nnUNet_preprocessed, task_name_variant2)
    maybe_mkdir_p(output_preprocessed_v2)

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

