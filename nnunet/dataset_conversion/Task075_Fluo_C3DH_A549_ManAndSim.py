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

from multiprocessing import Pool
import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import nnUNet_raw_data
from nnunet.paths import preprocessing_output_dir
from skimage.io import imread


def load_tiff_convert_to_nifti(img_file, lab_file, img_out_base, anno_out, spacing):
    img = imread(img_file)
    img_itk = sitk.GetImageFromArray(img.astype(np.float32))
    img_itk.SetSpacing(np.array(spacing)[::-1])
    sitk.WriteImage(img_itk, join(img_out_base + "_0000.nii.gz"))

    if lab_file is not None:
        l = imread(lab_file)
        l[l > 0] = 1
        l_itk = sitk.GetImageFromArray(l.astype(np.uint8))
        l_itk.SetSpacing(np.array(spacing)[::-1])
        sitk.WriteImage(l_itk, anno_out)


if __name__ == "__main__":
    source_folder_fluo_c3dh_a549 = '/home/isensee/drives/E132-Rohdaten/CellTrackingChallenge/train/Fluo-C3DH-A549'
    source_folder_fluo_c3dh_a549_sim = '/home/isensee/drives/E132-Rohdaten/CellTrackingChallenge/train/Fluo-C3DH-A549-SIM'
    source_folder_fluo_c3dh_a549_test = '/home/isensee/drives/E132-Rohdaten/CellTrackingChallenge/test/Fluo-C3DH-A549'
    source_folder_fluo_c3dh_a549_sim_test = '/home/isensee/drives/E132-Rohdaten/CellTrackingChallenge/test/Fluo-C3DH-A549-SIM'

    task_id = 75
    task_name = 'Fluo_C3DH_A549_ManAndSim'

    spacing = (1, 0.126, 0.126)
    p = Pool(16)

    foldername = "Task%03.0d_%s" % (task_id, task_name)

    out_base = join(nnUNet_raw_data, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    train_patient_names = []
    test_patient_names = []
    res = []

    for source_folder in [source_folder_fluo_c3dh_a549, source_folder_fluo_c3dh_a549_sim]:
        for train_sequence in ['01', '02']:
            train_cases = subfiles(join(source_folder, train_sequence), suffix=".tif", join=False)
            for t in train_cases:
                casename = os.path.basename(source_folder) + "__" + train_sequence + "__" + t[:-4]
                img_file = join(source_folder, train_sequence, t)
                lab_file = join(source_folder, train_sequence + "_GT", "SEG", "man_seg" + t[1:])
                if not isfile(lab_file):
                    # not all cases are annotated in the manual dataset
                    continue
                img_out_base = join(imagestr, casename)
                anno_out = join(labelstr, casename + ".nii.gz")
                res.append(
                    p.starmap_async(load_tiff_convert_to_nifti, ((img_file, lab_file, img_out_base, anno_out, spacing),)))
                train_patient_names.append(casename)

    for test_folder in [source_folder_fluo_c3dh_a549_test, source_folder_fluo_c3dh_a549_sim_test]:
        for test_sequence in ['01', '02']:
            test_cases = subfiles(join(test_folder, test_sequence), suffix=".tif", join=False)
            for t in test_cases:
                casename = os.path.basename(source_folder) + "__" + test_sequence + "__" + t[:-4]
                img_file = join(test_folder, test_sequence, t)
                lab_file = None
                img_out_base = join(imagests, casename)
                anno_out = None
                res.append(
                    p.starmap_async(load_tiff_convert_to_nifti, ((img_file, lab_file, img_out_base, anno_out, spacing),)))
                test_patient_names.append(casename)

    _ = [i.get() for i in res]

    json_dict = {}
    json_dict['name'] = task_name
    json_dict['description'] = ""
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = ""
    json_dict['licence'] = ""
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "BF",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "cell",
    }

    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numTest'] = len(test_patient_names)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                             train_patient_names]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in test_patient_names]

    save_json(json_dict, os.path.join(out_base, "dataset.json"))
    p.close()
    p.join()


    task_name = "Task075_Fluo_C3DH_A549_ManAndSim"
    labelsTr = join(nnUNet_raw_data, task_name, "labelsTr")
    cases = subfiles(labelsTr, suffix='.nii.gz', join=False)
    splits = []
    splits.append(
        {'train': [i[:-7] for i in cases if i.startswith('Fluo-C3DH-A549__01') or i.startswith('Fluo-C3DH-A549-SIM')],
         'val': [i[:-7] for i in cases if i.startswith('Fluo-C3DH-A549__02')]}
    )
    splits.append(
        {'train': [i[:-7] for i in cases if i.startswith('Fluo-C3DH-A549__02') or i.startswith('Fluo-C3DH-A549-SIM')],
         'val': [i[:-7] for i in cases if i.startswith('Fluo-C3DH-A549__01')]}
    )
    splits.append(
        {'train': [i[:-7] for i in cases if i.startswith('Fluo-C3DH-A549__') or i.startswith('Fluo-C3DH-A549-SIM__01')],
         'val': [i[:-7] for i in cases if i.startswith('Fluo-C3DH-A549-SIM__02')]}
    )
    splits.append(
        {'train': [i[:-7] for i in cases if i.startswith('Fluo-C3DH-A549__') or i.startswith('Fluo-C3DH-A549-SIM__02')],
         'val': [i[:-7] for i in cases if i.startswith('Fluo-C3DH-A549-SIM__01')]}
    )
    maybe_mkdir_p(join(preprocessing_output_dir, task_name))
    save_pickle(splits, join(preprocessing_output_dir, task_name, "splits_final.pkl"))

