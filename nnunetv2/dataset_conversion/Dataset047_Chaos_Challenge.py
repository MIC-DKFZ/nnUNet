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
"""
CHAOS challenge (https://chaos.grand-challenge.org/), MRI part (Task 3: liver, Task 5: liver, kidneys and spleen).

Download CHAOS_Train_Sets.zip and CHAOS_Test_Sets.zip from https://doi.org/10.5281/zenodo.3362844 (CC BY-NC-SA 4.0)
and extract both into the same folder, so that it contains Train_Sets/MR and Test_Sets/MR. Then run
    python Dataset047_Chaos_Challenge.py -i FOLDER

This script only prepares data to participate in Task 3 and Task 5. I don't like the CT task because
1) there are no abdominal organs in the ground truth. In the case of CT we are supposed to train only liver while on MRI
we are supposed to train all organs. This would require manual modification of nnU-net to deal with this dataset. This
is not what nnU-net is about.
2) CT Liver or multiorgan segmentation is too easy to get external data for. Therefore the challenges comes down
to who gets the best external data, not who has the best algorithm. Not super interesting.

Task 3 is a subtask of Task 5 so we need to prepare the data only once.
Difficulty: We need to process both T1 and T2, but T1 has 2 'modalities' (phases). nnU-Net cannot handle varying
number of input channels. We need to be creative.
We deal with this by treating all MRI sequences independently, so we now have 3*20 training data instead of 2*20.
In inference we then ensemble the results for the two t1 modalities.

Careful: T1_in_X, T1_out_X and T2_X are the same patient, so the cross-validation must be split by patient. This script
writes splits_final.json (patient-stratified, same folds as the nnU-Net v1 experiments on Task038) into the raw dataset
folder; nnU-Net copies it to nnUNet_preprocessed during experiment planning.

Ported from the nnU-Net v1 script (Task037_038_Chaos_Challenge.py, variant 2). The v1 version saved the split as
splits_final.pkl in the preprocessed folder, which nnU-Net v2 does not read.
"""
import argparse

import SimpleITK as sitk
import dicom2nifti
import numpy as np
from PIL import Image
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, save_json, subdirs, subfiles

from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.crossval_split import generate_crossval_split


def load_png_stack(folder):
    pngs = subfiles(folder, suffix="png")
    pngs.sort()
    loaded = []
    for p in pngs:
        loaded.append(np.array(Image.open(p)))
    loaded = np.stack(loaded, 0)[::-1]
    return loaded


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
    """
    exports a prediction in the png format of the CHAOS submission template
    """
    npy = sitk.GetArrayFromImage(sitk.ReadImage(nifti))
    seg_new = converter(npy)
    for z in range(len(npy)):
        Image.fromarray(seg_new[z]).save(join(output_folder, "img%03.0d.png" % z))


def write_label(seg, reference_image_file, output_files):
    img_sitk = sitk.ReadImage(reference_image_file)
    seg_itk = sitk.GetImageFromArray(seg.astype(np.uint8))
    seg_itk.SetSpacing(img_sitk.GetSpacing())
    seg_itk.SetOrigin(img_sitk.GetOrigin())
    seg_itk.SetDirection(img_sitk.GetDirection())
    for f in output_files:
        sitk.WriteImage(seg_itk, f)


def generate_patient_stratified_splits(patients, seed=12345):
    """
    5 folds over patients; every fold contains T2_X, T1_in_X and T1_out_X of its patients. Same folds as
    batchgenerators' get_split_deterministic(patients, fold, 5, 12345) that the nnU-Net v1 script used.
    """
    splits = []
    for s in generate_crossval_split(sorted(patients), seed=seed, n_splits=5):
        splits.append({k: [prefix + i for prefix in ("T2_", "T1_in_", "T1_out_") for i in s[k]]
                       for k in ('train', 'val')})
    return splits


def convert_chaos_mr(chaos_folder: str, dataset_id: int = 47):
    root = join(chaos_folder, "Train_Sets", "MR")
    root_test = join(chaos_folder, "Test_Sets", "MR")

    dataset_name = f"Dataset{dataset_id:03d}_CHAOS_Task_3_5_Variant2_new"
    output_folder = join(nnUNet_raw, dataset_name)
    output_images = join(output_folder, "imagesTr")
    output_imagesTs = join(output_folder, "imagesTs")
    output_labels = join(output_folder, "labelsTr")
    maybe_mkdir_p(output_images)
    maybe_mkdir_p(output_imagesTs)
    maybe_mkdir_p(output_labels)

    patients = subdirs(root, join=False)
    patients_test = subdirs(root_test, join=False)

    # T1: in phase and out phase become separate cases that share the label (written with the out phase geometry)
    for p in patients:
        seg = convert_MR_seg(load_png_stack(join(root, p, "T1DUAL", "Ground"))[::-1])
        for phase, case in (("InPhase", "T1_in_" + p), ("OutPhase", "T1_out_" + p)):
            img_outfile = join(output_images, case + "_0000.nii.gz")
            dicom2nifti.convert_dicom.dicom_series_to_nifti(join(root, p, "T1DUAL", "DICOM_anon", phase), img_outfile,
                                                            reorient_nifti=False)
        write_label(seg, img_outfile, [join(output_labels, "T1_in_" + p + ".nii.gz"),
                                       join(output_labels, "T1_out_" + p + ".nii.gz")])
    for p in patients_test:
        for phase, case in (("InPhase", "T1_in_" + p), ("OutPhase", "T1_out_" + p)):
            dicom2nifti.convert_dicom.dicom_series_to_nifti(join(root_test, p, "T1DUAL", "DICOM_anon", phase),
                                                            join(output_imagesTs, case + "_0000.nii.gz"),
                                                            reorient_nifti=False)

    # T2
    for p in patients:
        seg = convert_MR_seg(load_png_stack(join(root, p, "T2SPIR", "Ground"))[::-1])
        img_outfile = join(output_images, "T2_" + p + "_0000.nii.gz")
        dicom2nifti.convert_dicom.dicom_series_to_nifti(join(root, p, "T2SPIR", "DICOM_anon"), img_outfile,
                                                        reorient_nifti=False)
        write_label(seg, img_outfile, [join(output_labels, "T2_" + p + ".nii.gz")])
    for p in patients_test:
        dicom2nifti.convert_dicom.dicom_series_to_nifti(join(root_test, p, "T2SPIR", "DICOM_anon"),
                                                        join(output_imagesTs, "T2_" + p + "_0000.nii.gz"),
                                                        reorient_nifti=False)

    dataset_json = {
        'name': dataset_name,
        'description': 'Abdominal MRI from the CHAOS challenge (Combined Healthy Abdominal Organ Segmentation, Tasks '
                       '3/5: MR liver / multi-organ) with segmentations of liver, right kidney, left kidney and '
                       'spleen, acquired at Dokuz Eylül University Hospital (1.5T Philips). Variant 2: every MR '
                       'sequence is a separate single-channel case, so each of the 20 training patients gives three '
                       'cases (T1-DUAL in-phase, T1-DUAL out-of-phase, T2-SPIR; 60 in total), with the T1 label shared '
                       'by both phases; imagesTs holds the 60 unlabeled test-set images.',
        'channel_names': {'0': 'MR'},
        'labels': {'background': 0, 'liver': 1, 'right kidney': 2, 'left kidney': 3, 'spleen': 4},
        'file_ending': '.nii.gz',
        'numTraining': 3 * len(patients),
        'license': 'CC-BY-NC-SA-4.0',
        'commercial_ok': False,
        'reference': [
            'https://chaos.grand-challenge.org/Data/',
            'https://chaos.grand-challenge.org/Download/',
            'https://doi.org/10.5281/zenodo.3362844',
        ],
        'citation': [
            'Kavur AE, Gezer NS, Barış M, Aslan S, Conze PH, Groza V, Pham DD, Chatterjee S, Ernst P, Özkan S, '
            'Baydar B, Lachinov D, Han S, Pauli J, Isensee F, et al. (2021). CHAOS Challenge - combined (CT-MR) '
            'healthy abdominal organ segmentation. Medical Image Analysis 69, 101950. '
            'https://doi.org/10.1016/j.media.2020.101950',
            'Kavur AE, Selver MA, Dicle O, Barış M, Gezer NS (2019). CHAOS - Combined (CT-MR) Healthy Abdominal Organ '
            'Segmentation Challenge Data (Version v1.03) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.3362844',
            'Kavur AE, Gezer NS, Barış M, Şahin Y, Özkan S, Baydar B, Yüksel U, Kılıkçıer Ç, Olut Ş, Bozdağı Akar G, '
            'Ünal G, Dicle O, Selver MA (2020). Comparison of semi-automatic and deep learning-based automatic methods '
            'for liver segmentation in living liver transplant donors. Diagnostic and Interventional Radiology 26(1), '
            '11-21. https://doi.org/10.5152/dir.2019.19025',
        ],
        'converted_by': ['Fabian Isensee'],
        'note': 'Labels were converted from the CHAOS PNG ground truth by intensity range (liver 55-70, right kidney '
                '110-135, left kidney 175-200, spleen 240-255). T1_in_X, T1_out_X and T2_X are the same patient: '
                'splits_final.json in this folder holds a patient-stratified 5-fold split (nnU-Net copies it to '
                'nnUNet_preprocessed during experiment planning); do not use a random split. CT data of CHAOS is not '
                'included. Test-set ground truth is never released.',
    }
    save_json(dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
    save_json(generate_patient_stratified_splits(patients), join(output_folder, 'splits_final.json'), sort_keys=False)
    return dataset_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', type=str, required=True,
                        help='Folder with the extracted CHAOS_Train_Sets.zip and CHAOS_Test_Sets.zip (contains '
                             'Train_Sets/MR and Test_Sets/MR)')
    parser.add_argument('-d', '--dataset_id', type=int, default=47, help='nnU-Net Dataset ID, default: 47')
    args = parser.parse_args()
    convert_chaos_mr(args.input_folder, args.dataset_id)
