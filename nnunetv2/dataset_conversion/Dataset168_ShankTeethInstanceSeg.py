import shutil
from multiprocessing import Pool
from time import time
from typing import Tuple
import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from scipy.ndimage import binary_fill_holes
from acvl_utils.instance_segmentation.instance_as_semantic_seg import convert_instanceseg_to_semantic_patched, CENTER_LABEL, BORDER_LABEL
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed


# ideally we want to create a raw dataset with border-core semantic segmentation and then plan and preprocess that.
# Due to the spacing differences, this would cause a lot of rounding errors in the border width after resampling to a
# homogeneous voxel spacing. This is why we need to run the preprocessing for Dataset 162 first and then covnert the
# preprocessed data to border-core. Not ideal, but that's just how it is.

# nnUNetv2_plan_and_preprocess -d 162 -npfp 6 -np 3 -overwrite_plans_name nnUNetPlans_sp05 -overwrite_target_spacing 0.5 0.5 0.5 -gpu_memory_target 11 -c 3d_fullres

def convert_case(source_folder: str, target_folder: str, identifier: str,
                 current_spacing: Tuple[float, ...],
                 border_thickness_in_mm: int = 0.5):
    data = np.load(join(source_folder, identifier + '.npz'))['data']
    pkl = load_pickle(join(source_folder, identifier + '.pkl'))
    seg = np.load(join(source_folder, identifier + '.npz'))['seg']
    seg[seg <= 4] = 0  # remove jaws and prosthetics
    instances = np.unique(seg)
    # small holes in the reference segmentation are super annoying because the spawn a large ring of order around
    # them. These holes are just annotation errors and these rings will confuse the model. Best to fill those holes.
    for i in instances:
        if i != 0:
            mask = seg == i
            mask_closed = binary_fill_holes(mask)
            seg[mask_closed] = i
    semseg = convert_instanceseg_to_semantic_patched(seg.astype(np.uint8)[0], current_spacing, border_thickness_in_mm)[None]
    # semseg = instance2border_semantic(seg, (1, 1, 1), 1)
    np.savez_compressed(join(target_folder, identifier + '.npz'), data=data, seg=semseg)
    pkl['classes'] = np.array([0, 1, 2])
    pkl['class_locations'] = DefaultPreprocessor._sample_foreground_locations(semseg, [1, 2])
    save_pickle(pkl, join(target_folder, identifier + '.pkl'))


def convert_gt(source_file: str, target_file: str,
               border_thickness_in_mm: int = 0.5):
    source_itk = sitk.ReadImage(source_file)
    source_npy = sitk.GetArrayFromImage(source_itk)
    source_npy[source_npy <= 4] = 0
    instances = np.unique(source_npy)
    for i in instances:
        if i != 0:
            mask = source_npy == i
            mask_closed = binary_fill_holes(mask)
            source_npy[mask_closed] = i
    source_spacing = list(source_itk.GetSpacing())[::-1]
    source_semseg = convert_instanceseg_to_semantic_patched(source_npy.astype(np.uint8), source_spacing,
                                                            border_thickness_in_mm)
    itk_img = sitk.GetImageFromArray(source_semseg.astype(np.uint8))
    itk_img.SetSpacing(source_itk.GetSpacing())
    itk_img.SetDirection(source_itk.GetDirection())
    itk_img.SetOrigin(source_itk.GetOrigin())

    itk_img.CopyInformation(source_itk)
    sitk.WriteImage(itk_img, target_file)


if __name__ == '__main__':
    # this conversion starts with the already preprocessed task 162. Be sure to run this first! There is currently
    start = time()
    output_task_name = 'Dataset168_ShankTeethInstanceSeg'
    source_task_name = maybe_convert_to_dataset_name(162)
    p = Pool(8)
    border_thickness_in_mm = 0.9
    current_spacing = (0.5, 0.5, 0.5)

    source_folder = join(nnUNet_preprocessed, source_task_name, 'nnUNetPlans_sp05_3d_fullres')
    target_folder = join(nnUNet_preprocessed, output_task_name, 'nnUNetPlans_sp05_3d_fullres')
    maybe_mkdir_p(target_folder)

    identifiers = [i[:-4] for i in subfiles(source_folder, suffix='npz', join=False)]
    r = p.starmap_async(convert_case,
                        zip([source_folder] * len(identifiers),
                            [target_folder] * len(identifiers),
                            identifiers,
                            [current_spacing] * len(identifiers),
                            [border_thickness_in_mm] * len(identifiers)))

    daatset_json = load_json(join(nnUNet_preprocessed, source_task_name, 'dataset.json'))
    daatset_json['labels'] = {"background": 0, "center": CENTER_LABEL, 'border': BORDER_LABEL}
    save_json(daatset_json, join(nnUNet_preprocessed, output_task_name, 'dataset.json'))

    plans = load_json(join(nnUNet_preprocessed, source_task_name, 'nnUNetPlans_sp05.json'))
    plans['dataset_name'] = output_task_name
    save_json(plans, join(nnUNet_preprocessed, output_task_name, 'nnUNetPlans_sp05.json'))

    shutil.copy(join(nnUNet_preprocessed, source_task_name, 'dataset_fingerprint.json'),
                join(nnUNet_preprocessed, output_task_name, 'dataset_fingerprint.json'))

    _ = r.get()

    gt_source_dir = join(nnUNet_preprocessed, source_task_name, 'gt_segmentations')
    gt_target_dir = join(nnUNet_preprocessed, output_task_name, 'gt_segmentations')
    maybe_mkdir_p(gt_target_dir)

    nii_files = nifti_files(gt_source_dir, join=False)
    r = p.starmap_async(convert_gt,
                        zip([join(gt_source_dir, i) for i in nii_files],
                            [join(gt_target_dir, i) for i in nii_files],
                            [border_thickness_in_mm] * len(identifiers)))
    _ = r.get()


    end = time()
    print(f'that took {end - start} seconds')

    # nnUNetv2_train 168 3d_fullres 0 -p nnUNetPlans_sp05 --use_compressed