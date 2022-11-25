import shutil
from multiprocessing import Pool
from typing import Tuple

import numpy as np
from acvl_utils.morphology.morphology_helper import generate_ball
from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.dataset_conversion.Dataset179_KiTS2019_sparserer_blobs import compute_labeled_fractions_folder
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
import SimpleITK as sitk


def simulate_annotated_spheres(seg, spacing, labeled_fraction: float, sphere_volume: Tuple[float, float], ignore_label: int):
    # we sample spheres until we hit the labeled_fraction
    labeled_pixels = 0
    allowed_labeled_pixels = labeled_fraction * np.prod(seg.shape, dtype=np.int64)

    vol_per_pixel = np.prod(spacing)
    avg_volume = np.mean(sphere_volume)
    avg_volume_pixels = avg_volume / vol_per_pixel
    print(f'shape: {seg.shape}, allowed_labeled_pixels: {allowed_labeled_pixels}, est num spheres {allowed_labeled_pixels / avg_volume_pixels}')

    final_mask = np.zeros_like(seg, dtype=bool)
    num_spheres = 0
    while True:
        vol_here = np.random.uniform(sphere_volume[0], sphere_volume[1])
        sphere_radius = (vol_here * 3 / 4 / np.pi) ** (1 / 3)
        b = generate_ball([sphere_radius] * 3, spacing, dtype=bool)
        # figure out if we can add this ball
        added_pixels = np.sum(b)
        current_percentage = labeled_pixels / allowed_labeled_pixels
        theoretical_next = (labeled_pixels + added_pixels) / allowed_labeled_pixels
        if np.abs(current_percentage - 1) > np.abs(theoretical_next - 1):
            x = np.random.randint(0, seg.shape[0] - b.shape[0]) if b.shape[0] != seg.shape[0] else 0
            y = np.random.randint(0, seg.shape[1] - b.shape[1]) if b.shape[1] != seg.shape[1] else 0
            z = np.random.randint(0, seg.shape[2] - b.shape[2]) if b.shape[2] != seg.shape[2] else 0
            final_mask[x:x+b.shape[0], y:y+b.shape[1], z:z+b.shape[2]][b] = True
            labeled_pixels += added_pixels
            num_spheres += 1
        else:
            break
    print(num_spheres)
    ret = np.ones_like(seg, dtype=np.uint8) * ignore_label
    ret[final_mask] = seg[final_mask]
    return ret


def load_simulate_annotated_spheres_save(source_img: str, target_img: str, labeled_image_fraction: float,
                                         sphere_volume: Tuple[int, int],
                                          ignore_label: int):
    seg_itk = sitk.ReadImage(source_img)
    seg = sitk.GetArrayFromImage(seg_itk).astype(np.uint8)
    try:
        seg = simulate_annotated_spheres(seg, list(seg_itk.GetSpacing())[::-1], labeled_image_fraction, sphere_volume,
                                         ignore_label)
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
    dataset_name = 'Dataset178_KiTS2019_sparse_blobs'

    labelstr = join(nnUNet_raw, dataset_name, 'labelsTr')
    maybe_mkdir_p(labelstr)
    # we can just copy the images
    # shutil.copytree(join(nnUNet_raw, source_dataset_name, 'imagesTr'), join(nnUNet_raw, dataset_name, 'imagesTr'))

    ignore_label = 3  # 0 bg, 1 kidney, 2 tumor
    labeled_fraction = 0.1
    sphere_volume = [50000, 100000]
    np.random.seed(12345)

    source_labels = nifti_files(join(nnUNet_raw, source_dataset_name, 'labelsTr'), join=False)

    p = Pool(24)
    r = []
    for s in source_labels:
        r.append(
            p.starmap_async(load_simulate_annotated_spheres_save,
                            ((
                                 join(nnUNet_raw, source_dataset_name, 'labelsTr', s),
                                 join(nnUNet_raw, dataset_name, 'labelsTr', s),
                                 labeled_fraction,
                                 sphere_volume,
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
                                           join(nnUNet_raw, maybe_convert_to_dataset_name(178), 'labelsTr'),
                                            labels=(0, 1, 2),
                                           ignore_label=3, num_processes=16
                                           ))