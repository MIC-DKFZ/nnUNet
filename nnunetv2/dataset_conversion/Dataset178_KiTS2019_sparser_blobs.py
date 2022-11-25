import shutil
from multiprocessing import Pool
from typing import Tuple

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.dataset_conversion.Dataset177_KiTS2019_sparse_blobs import load_simulate_annotated_spheres_save
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
import SimpleITK as sitk


def compute_labeled_fractions_image(original_seg_file: str, sparse_seg_file: str, labels: Tuple[int, ...], ignore_label: int):
    """
    assumes everything that is not ignore label was labeled. so labels needs to cover all classes, incl background
    we could also jsut use ignore_label + np.unique but unique is slow af
    """
    orig = sitk.GetArrayFromImage(sitk.ReadImage(original_seg_file))
    sparse = sitk.GetArrayFromImage(sitk.ReadImage(sparse_seg_file))
    results_per_label = {}
    for l in labels:
        mask_orig = orig == l
        mask_sparse = sparse == l
        results_per_label[l] = np.sum(mask_sparse) / np.sum(mask_orig)
    # foreground
    mask_orig = orig != ignore_label
    mask_sparse = sparse != ignore_label
    results_per_label['fg'] = np.sum(mask_sparse) / np.sum(mask_orig)
    return results_per_label


def compute_labeled_fractions_folder(folder_dense, folder_sparse, labels, ignore_label, num_processes: int = 8):
    p = Pool(num_processes)
    files = nifti_files(folder_dense, join=False)
    r = []
    for f in files:
        r.append(p.starmap_async(compute_labeled_fractions_image,
                                 ((join(folder_dense, f), join(folder_sparse, f), labels, ignore_label),)
                                 ))
    r = [i.get() for i in r]
    all_results = {}
    for k in r[0][0].keys():
        all_results[k] = np.nanmean([i[0][k] for i in r])
    return all_results


if __name__ == '__main__':
    """
    like 177 but with less fg sampling
    """
    source_dataset_name = maybe_convert_to_dataset_name(64)
    dataset_name = 'Dataset178_KiTS2019_sparser_blobs'

    labelstr = join(nnUNet_raw, dataset_name, 'labelsTr')
    maybe_mkdir_p(labelstr)
    # we can just copy the images
    shutil.copytree(join(nnUNet_raw, source_dataset_name, 'imagesTr'), join(nnUNet_raw, dataset_name, 'imagesTr'))

    ignore_label = 3  # 0 bg, 1 kidney, 2 tumor
    num_spheres_random = 15
    num_spheres_fg = 3
    sphere_size = (15, 50)
    np.random.seed(12345)

    source_labels = nifti_files(join(nnUNet_raw, source_dataset_name, 'labelsTr'), join=False)

    p = Pool(16)
    r = []
    for s in source_labels:
        r.append(
            p.starmap_async(load_simulate_annotated_spheres_save,
                            ((
                                 join(nnUNet_raw, source_dataset_name, 'labelsTr', s),
                                 join(nnUNet_raw, dataset_name, 'labelsTr', s),
                                 num_spheres_random,
                                 num_spheres_fg,
                                 sphere_size,
                                 ignore_label,
                             ),))
        )
    _ = [i.get() for i in r]
    p.close()
    p.join()
    generate_dataset_json(join(nnUNet_raw, dataset_name), {0: 'CT'},
                          {'background': 0, 'kidney': 1, 'tumor': 2, 'ignore': ignore_label},
                          210, '.nii.gz')

    # compute class fractions
    print(compute_labeled_fractions_folder(join(nnUNet_raw, maybe_convert_to_dataset_name(64), 'labelsTr'),
                                           join(nnUNet_raw, maybe_convert_to_dataset_name(176), 'labelsTr'),
                                            labels=(0, 1, 2),
                                           ignore_label=3
                                           ))