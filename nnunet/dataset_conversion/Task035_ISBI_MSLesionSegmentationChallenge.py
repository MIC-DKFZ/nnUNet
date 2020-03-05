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

import shutil
from collections import OrderedDict
import numpy as np
import SimpleITK as sitk
import multiprocessing
from batchgenerators.utilities.file_and_folder_operations import *


def convert_to_nii_gz(filename):
    f = sitk.ReadImage(filename)
    sitk.WriteImage(f, os.path.splitext(filename)[0] + ".nii.gz")
    os.remove(filename)


def convert_for_submission(source_dir, target_dir):
    files = subfiles(source_dir, suffix=".nii.gz", join=False)
    maybe_mkdir_p(target_dir)
    for f in files:
        splitted = f.split("__")
        case_id = int(splitted[1])
        timestep = int(splitted[2][:-7])
        t = join(target_dir, "test%02d_%02d_nnUNet.nii" % (case_id, timestep))
        img = sitk.ReadImage(join(source_dir, f))
        sitk.WriteImage(img, t)


if __name__ == "__main__":
    # convert to nifti.gz
    dirs = ['/media/fabian/My Book/MedicalDecathlon/Task035_ISBILesionSegmentation/imagesTr',
            '/media/fabian/My Book/MedicalDecathlon/Task035_ISBILesionSegmentation/imagesTs',
            '/media/fabian/My Book/MedicalDecathlon/Task035_ISBILesionSegmentation/labelsTr']

    p = multiprocessing.Pool(3)

    for d in dirs:
        nii_files = subfiles(d, suffix='.nii')
        p.map(convert_to_nii_gz, nii_files)

    p.close()
    p.join()


    def rename_files(folder):
        all_files = subfiles(folder, join=False)
        # there are max 14 patients per folder, starting with 1
        for patientid in range(1, 15):
            # there are certainly no more than 10 time steps per patient, starting with 1
            for t in range(1, 10):
                patient_files = [i for i in all_files if i.find("%02.0d_%02.0d_" % (patientid, t)) != -1]
                if not len(patient_files) == 4:
                    continue

                flair_file = [i for i in patient_files if i.endswith("_flair_pp.nii.gz")][0]
                mprage_file = [i for i in patient_files if i.endswith("_mprage_pp.nii.gz")][0]
                pd_file = [i for i in patient_files if i.endswith("_pd_pp.nii.gz")][0]
                t2_file = [i for i in patient_files if i.endswith("_t2_pp.nii.gz")][0]

                os.rename(join(folder, flair_file), join(folder, "case__%02.0d__%02.0d_0000.nii.gz" % (patientid, t)))
                os.rename(join(folder, mprage_file), join(folder, "case__%02.0d__%02.0d_0001.nii.gz" % (patientid, t)))
                os.rename(join(folder, pd_file), join(folder, "case__%02.0d__%02.0d_0002.nii.gz" % (patientid, t)))
                os.rename(join(folder, t2_file), join(folder, "case__%02.0d__%02.0d_0003.nii.gz" % (patientid, t)))


    for d in dirs[:-1]:
        rename_files(d)


    # now we have to deal with the training masks, we do it the quick and dirty way here by just creating copies of the
    # training data

    train_folder = '/media/fabian/My Book/MedicalDecathlon/Task035_ISBILesionSegmentation/imagesTr'

    for patientid in range(1, 6):
        for t in range(1, 6):
            fnames_original = subfiles(train_folder, prefix="case__%02.0d__%02.0d" % (patientid, t), suffix=".nii.gz", sort=True)
            for f in fnames_original:
                for mask in [1, 2]:
                    fname_target = f[:-12] + "__mask%d" % mask + f[-12:]
                    shutil.copy(f, fname_target)
                os.remove(f)


    labels_folder = '/media/fabian/My Book/MedicalDecathlon/Task035_ISBILesionSegmentation/labelsTr'

    for patientid in range(1, 6):
        for t in range(1, 6):
            for mask in [1, 2]:
                f = join(labels_folder, "training%02d_%02d_mask%d.nii.gz" % (patientid, t, mask))
                if isfile(f):
                    os.rename(f, join(labels_folder, "case__%02.0d__%02.0d__mask%d.nii.gz" % (patientid, t, mask)))



    tr_files = []
    for patientid in range(1, 6):
        for t in range(1, 6):
            for mask in [1, 2]:
                if isfile(join(labels_folder, "case__%02.0d__%02.0d__mask%d.nii.gz" % (patientid, t, mask))):
                    tr_files.append("case__%02.0d__%02.0d__mask%d.nii.gz" % (patientid, t, mask))


    ts_files = []
    for patientid in range(1, 20):
        for t in range(1, 20):
            if isfile(join("/media/fabian/My Book/MedicalDecathlon/Task035_ISBILesionSegmentation/imagesTs",
                           "case__%02.0d__%02.0d_0000.nii.gz" % (patientid, t))):
                ts_files.append("case__%02.0d__%02.0d.nii.gz" % (patientid, t))


    out_base = '/media/fabian/My Book/MedicalDecathlon/Task035_ISBILesionSegmentation/'

    json_dict = OrderedDict()
    json_dict['name'] = "ISBI_Lesion_Segmentation_Challenge_2015"
    json_dict['description'] = "nothing"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see challenge website"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "flair",
        "1": "mprage",
        "2": "pd",
        "3": "t2"
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "lesion"
    }
    json_dict['numTraining'] = len(subfiles(labels_folder))
    json_dict['numTest'] = len(subfiles('/media/fabian/My Book/MedicalDecathlon/Task035_ISBILesionSegmentation/imagesTs')) // 4
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i[:-7], "label": "./labelsTr/%s.nii.gz" % i[:-7]} for i in
                             tr_files]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i[:-7] for i in ts_files]

    save_json(json_dict, join(out_base, "dataset.json"))

    case_identifiers = np.unique([i[:-12] for i in subfiles("/media/fabian/My Book/MedicalDecathlon/MedicalDecathlon_raw_splitted/Task035_ISBILesionSegmentation/imagesTr", suffix='.nii.gz', join=False)])

    splits = []
    for f in range(5):
        cases = [i for i in range(1, 6) if i != f+1]
        splits.append(OrderedDict())
        splits[-1]['val'] = np.array([i for i in case_identifiers if i.startswith("case__%02d__" % (f + 1))])
        remaining = [i for i in case_identifiers if i not in splits[-1]['val']]
        splits[-1]['train'] = np.array(remaining)

    maybe_mkdir_p("/media/fabian/nnunet/Task035_ISBILesionSegmentation")
    save_pickle(splits, join("/media/fabian/nnunet/Task035_ISBILesionSegmentation", "splits_final.pkl"))
