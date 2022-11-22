from math import ceil

from acvl_utils.morphology.morphology_helper import generate_ball
from multiprocessing import Pool
from typing import Tuple

import numpy as np
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import *
import shutil

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.paths import nnUNet_raw
import SimpleITK as sitk
from scipy.ndimage import binary_dilation


def simulate_annotated_spheres(seg, num_spheres_random: int, num_spheres_fg: int, sphere_size: Tuple[int, int], ignore_label: int):
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
    for n in range(num_spheres_fg):
        # pick random class
        c = np.random.choice(keys)
        # pick random location
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


def load_simulate_annotated_spheres_save(source_img: str, target_img: str, num_spheres_random: int,
                                                    num_spheres_fg: int, sphere_size: Tuple[int, int], ignore_label: int):
    seg_itk = sitk.ReadImage(source_img)
    seg = sitk.GetArrayFromImage(seg_itk).astype(np.uint8)
    try:
        seg = simulate_annotated_spheres(seg, num_spheres_random, num_spheres_fg, sphere_size, ignore_label)
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
    This dataset is a dummy to test different strategies on how to deal with ignore label . We need to test whether we 
    need adaptations to our hyperparameters depending on the amount and nature of annotations.
    
    This dataset only has every 10th slice annotated. All others are ignore label.
    See also Dataset 177
    """
    source_dataset_name = maybe_convert_to_dataset_name(64)
    dataset_name = 'Dataset177_KiTS2019_sparse_blobs'

    labelstr = join(nnUNet_raw, dataset_name, 'labelsTr')
    maybe_mkdir_p(labelstr)
    # we can just copy the images
    shutil.copytree(join(nnUNet_raw, source_dataset_name, 'imagesTr'), join(nnUNet_raw, dataset_name, 'imagesTr'))

    ignore_label = 3  # 0 bg, 1 kidney, 2 tumor
    num_spheres_random = 25
    num_spheres_fg = 12
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