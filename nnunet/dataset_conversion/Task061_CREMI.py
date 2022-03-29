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
from nnunet.utilities.file_and_folder_operations_winos import * # Join path by slash on windows system.
import numpy as np
from nnunet.paths import nnUNet_raw_data, preprocessing_output_dir
import shutil
import SimpleITK as sitk

try:
    import h5py
except ImportError:
    h5py = None


def load_sample(filename):
    # we need raw data and seg
    f = h5py.File(filename, 'r')
    data = np.array(f['volumes']['raw'])

    if 'labels' in f['volumes'].keys():
        labels = np.array(f['volumes']['labels']['clefts'])
        # clefts are low values, background is high
        labels = (labels < 100000).astype(np.uint8)
    else:
        labels = None
    return data, labels


def save_as_nifti(arr, filename, spacing):
    itk_img = sitk.GetImageFromArray(arr)
    itk_img.SetSpacing(spacing)
    sitk.WriteImage(itk_img, filename)


def prepare_submission():
    from cremi.io import CremiFile
    from cremi.Volume import Volume

    base = "/home/fabian/drives/datasets/results/nnUNet/test_sets/Task061_CREMI/"
    # a+
    pred = sitk.GetArrayFromImage(sitk.ReadImage(join(base, 'results_3d_fullres', "sample_a+.nii.gz"))).astype(np.uint64)
    pred[pred == 0] = 0xffffffffffffffff
    out_a = CremiFile(join(base, 'sample_A+_20160601.hdf'), 'w')
    clefts = Volume(pred, (40., 4., 4.))
    out_a.write_clefts(clefts)
    out_a.close()

    pred = sitk.GetArrayFromImage(sitk.ReadImage(join(base, 'results_3d_fullres', "sample_b+.nii.gz"))).astype(np.uint64)
    pred[pred == 0] = 0xffffffffffffffff
    out_b = CremiFile(join(base, 'sample_B+_20160601.hdf'), 'w')
    clefts = Volume(pred, (40., 4., 4.))
    out_b.write_clefts(clefts)
    out_b.close()

    pred = sitk.GetArrayFromImage(sitk.ReadImage(join(base, 'results_3d_fullres', "sample_c+.nii.gz"))).astype(np.uint64)
    pred[pred == 0] = 0xffffffffffffffff
    out_c = CremiFile(join(base, 'sample_C+_20160601.hdf'), 'w')
    clefts = Volume(pred, (40., 4., 4.))
    out_c.write_clefts(clefts)
    out_c.close()


if __name__ == "__main__":
    assert h5py is not None, "you need h5py for this. Install with 'pip install h5py'"

    foldername = "Task061_CREMI"
    out_base = join(nnUNet_raw_data, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    base = "/media/fabian/My Book/datasets/CREMI"

    # train
    img, label = load_sample(join(base, "sample_A_20160501.hdf"))
    save_as_nifti(img, join(imagestr, "sample_a_0000.nii.gz"), (4, 4, 40))
    save_as_nifti(label, join(labelstr, "sample_a.nii.gz"), (4, 4, 40))
    img, label = load_sample(join(base, "sample_B_20160501.hdf"))
    save_as_nifti(img, join(imagestr, "sample_b_0000.nii.gz"), (4, 4, 40))
    save_as_nifti(label, join(labelstr, "sample_b.nii.gz"), (4, 4, 40))
    img, label = load_sample(join(base, "sample_C_20160501.hdf"))
    save_as_nifti(img, join(imagestr, "sample_c_0000.nii.gz"), (4, 4, 40))
    save_as_nifti(label, join(labelstr, "sample_c.nii.gz"), (4, 4, 40))

    save_as_nifti(img, join(imagestr, "sample_d_0000.nii.gz"), (4, 4, 40))
    save_as_nifti(label, join(labelstr, "sample_d.nii.gz"), (4, 4, 40))

    save_as_nifti(img, join(imagestr, "sample_e_0000.nii.gz"), (4, 4, 40))
    save_as_nifti(label, join(labelstr, "sample_e.nii.gz"), (4, 4, 40))

    # test
    img, label = load_sample(join(base, "sample_A+_20160601.hdf"))
    save_as_nifti(img, join(imagests, "sample_a+_0000.nii.gz"), (4, 4, 40))
    img, label = load_sample(join(base, "sample_B+_20160601.hdf"))
    save_as_nifti(img, join(imagests, "sample_b+_0000.nii.gz"), (4, 4, 40))
    img, label = load_sample(join(base, "sample_C+_20160601.hdf"))
    save_as_nifti(img, join(imagests, "sample_c+_0000.nii.gz"), (4, 4, 40))

    json_dict = OrderedDict()
    json_dict['name'] = foldername
    json_dict['description'] = foldername
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see challenge website"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "EM",
    }
    json_dict['labels'] = {i: str(i) for i in range(2)}

    json_dict['numTraining'] = 5
    json_dict['numTest'] = 1
    json_dict['training'] = [{'image': "./imagesTr/sample_%s.nii.gz" % i, "label": "./labelsTr/sample_%s.nii.gz" % i} for i in
                             ['a', 'b', 'c', 'd', 'e']]

    json_dict['test'] = ["./imagesTs/sample_a+.nii.gz", "./imagesTs/sample_b+.nii.gz", "./imagesTs/sample_c+.nii.gz"]

    save_json(json_dict, os.path.join(out_base, "dataset.json"))

    out_preprocessed = join(preprocessing_output_dir, foldername)
    maybe_mkdir_p(out_preprocessed)
    # manual splits. we train 5 models on all three datasets
    splits = [{'train': ["sample_a", "sample_b", "sample_c"], 'val': ["sample_a", "sample_b", "sample_c"]},
              {'train': ["sample_a", "sample_b", "sample_c"], 'val': ["sample_a", "sample_b", "sample_c"]},
              {'train': ["sample_a", "sample_b", "sample_c"], 'val': ["sample_a", "sample_b", "sample_c"]},
              {'train': ["sample_a", "sample_b", "sample_c"], 'val': ["sample_a", "sample_b", "sample_c"]},
              {'train': ["sample_a", "sample_b", "sample_c"], 'val': ["sample_a", "sample_b", "sample_c"]}]
    save_pickle(splits, join(out_preprocessed, "splits_final.pkl"))