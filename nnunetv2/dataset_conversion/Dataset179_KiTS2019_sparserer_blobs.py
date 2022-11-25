import shutil
from multiprocessing import Pool
from typing import Tuple

import numpy as np
import pandas as pd
from acvl_utils.morphology.morphology_helper import generate_ball
from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.dataset_conversion.Dataset178_KiTS2019_sparser_blobs import compute_labeled_fractions_folder
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
import SimpleITK as sitk


def simulate_annotated_spheres2(seg, num_spheres_random: int, num_spheres_per_class: int, sphere_size: Tuple[int, int], ignore_label: int):
    labels = [i for i in pd.unique(seg.ravel()) if i != 0 and i > 0]
    locs = DefaultPreprocessor._sample_foreground_locations(seg, labels, seed=1234, verbose=False)
    assert len(locs.keys()) > 0
    final_mask = np.zeros_like(seg, dtype=bool)
    max_allowed_size = min(sphere_size[1], (min(seg.shape) - 1) // 2)
    min_allowed_size = min(min(sphere_size[1], (min(seg.shape) - 1) // 2) - 1, sphere_size[0])
    for n in range(num_spheres_random):
        size = np.random.randint(min_allowed_size, max_allowed_size)
        b = generate_ball([size] * 3).astype(bool)
        x = np.random.randint(0, seg.shape[0] - b.shape[0]) if b.shape[0] != seg.shape[0] else 0
        y = np.random.randint(0, seg.shape[1] - b.shape[1]) if b.shape[1] != seg.shape[1] else 0
        z = np.random.randint(0, seg.shape[2] - b.shape[2]) if b.shape[2] != seg.shape[2] else 0
        final_mask[x:x+b.shape[0], y:y+b.shape[1], z:z+b.shape[2]][b] = True
    keys = [i for i in list(locs.keys()) if len(locs[i]) > 0]
    for c in keys:
        if c != 0:
            for n in range(num_spheres_per_class):
                l = locs[c][np.random.choice(len(locs[c]))]
                size = np.random.randint(min_allowed_size, max_allowed_size)
                b = generate_ball([size] * 3).astype(bool)
                x = max(0, l[0] - b.shape[0] // 2)
                y = max(0, l[1] - b.shape[1] // 2)
                z = max(0, l[2] - b.shape[2] // 2)
                x = min(seg.shape[0] - b.shape[0], x)
                y = min(seg.shape[1] - b.shape[1], y)
                z = min(seg.shape[2] - b.shape[2], z)
                final_mask[x:x+b.shape[0], y:y+b.shape[1], z:z+b.shape[2]][b] = True
    ret = np.ones_like(seg, dtype=np.uint8) * ignore_label
    ret[final_mask] = seg[final_mask]
    return ret


def load_simulate_annotated_spheres_save2(source_img: str, target_img: str, num_spheres_random: int,
                                                    num_spheres_per_class: int, sphere_size: Tuple[int, int],
                                          ignore_label: int):
    seg_itk = sitk.ReadImage(source_img)
    seg = sitk.GetArrayFromImage(seg_itk).astype(np.uint8)
    try:
        seg = simulate_annotated_spheres2(seg, num_spheres_random, num_spheres_per_class, sphere_size, ignore_label)
    except Exception as e:
        print(source_img)
        raise e
    seg = sitk.GetImageFromArray(seg)
    seg.SetSpacing(seg_itk.GetSpacing())
    seg.SetDirection(seg_itk.GetDirection())
    seg.SetOrigin(seg_itk.GetOrigin())
    sitk.WriteImage(seg, target_img)


if __name__ == '__main__':
    """
    like 177 but with less fg sampling
    """
    source_dataset_name = maybe_convert_to_dataset_name(64)
    dataset_name = 'Dataset179_KiTS2019_sparserer_blobs'

    labelstr = join(nnUNet_raw, dataset_name, 'labelsTr')
    maybe_mkdir_p(labelstr)
    # we can just copy the images
    shutil.copytree(join(nnUNet_raw, source_dataset_name, 'imagesTr'), join(nnUNet_raw, dataset_name, 'imagesTr'))

    ignore_label = 3  # 0 bg, 1 kidney, 2 tumor
    num_spheres_random = 30
    num_spheres_per_class = 1
    sphere_size = (5, 15)
    np.random.seed(12345)

    source_labels = nifti_files(join(nnUNet_raw, source_dataset_name, 'labelsTr'), join=False)

    p = Pool(16)
    r = []
    for s in source_labels:
        r.append(
            p.starmap_async(load_simulate_annotated_spheres_save2,
                            ((
                                 join(nnUNet_raw, source_dataset_name, 'labelsTr', s),
                                 join(nnUNet_raw, dataset_name, 'labelsTr', s),
                                 num_spheres_random,
                                 num_spheres_per_class,
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
                                           join(nnUNet_raw, maybe_convert_to_dataset_name(179), 'labelsTr'),
                                            labels=(0, 1, 2),
                                           ignore_label=3
                                           ))