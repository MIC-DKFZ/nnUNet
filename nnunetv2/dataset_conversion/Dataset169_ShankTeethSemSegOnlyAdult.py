import shutil
from multiprocessing import Pool
from time import time
from typing import Tuple
import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from scipy.ndimage import binary_fill_holes
from acvl_utils.instance_segmentation.instance_as_semantic_seg import convert_instanceseg_to_semantic_patched, CENTER_LABEL, BORDER_LABEL
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed


def has_non_adult_teeth(filename: str):
    seg = sitk.GetArrayFromImage(sitk.ReadImage(filename)).astype(np.uint8)
    return np.any(seg > 36)


def convert_case(image_file_in, seg_file_in, image_file_out, seg_file_out, remove_juvenile_cases: bool = True):
    # if remove_juvenile_cases we skip the case if it has kids teeth in it. If False we keep it but remove non-adult teeth
    seg_in = sitk.ReadImage(seg_file_in)
    seg_in_npy = sitk.GetArrayFromImage(seg_in).astype(np.uint8)
    if remove_juvenile_cases and np.any(seg_in_npy > 36):
        print(f'removed case {os.path.basename(seg_file_in)}')
        return
    else:
        seg_in_npy[seg_in_npy > 36] = 0
        seg_out = sitk.GetImageFromArray(seg_in_npy)
        seg_out.SetOrigin(seg_in.GetSpacing())
        seg_out.SetDirection(seg_in.GetDirection())
        seg_out.SetSpacing(seg_in.GetSpacing())
        sitk.WriteImage(seg_out, seg_file_out)
        shutil.copy(image_file_in, image_file_out)


if __name__ == '__main__':
    # find cases with non-adult teeth
    # source_folder = '/home/isensee/drives/E132-Rohdaten/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task162_ShankTeeth/labelsTr'
    # nii_files = nifti_files(source_folder, join=True, sort=True)
    # p = Pool(24)
    # res = p.map(has_non_adult_teeth, nii_files)
    # res_inv = [not i for i in res]
    # print(f'num_remaining {np.sum(res_inv)}')
    # print(f'num_with_non_adult_teeth {np.sum(res)}')
    # print(f'total {len(res)}')
    # p.close()
    # p.join()

    # RESULT
    # num_remaining    935
    # num_with_non_adult_teeth    12
    # total    947

    remove_juvenile_cases = True
    source_folder = '/home/isensee/drives/E132-Rohdaten/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task162_ShankTeeth'
    target_dataset_name = 'Dataset169_ShankTeethSemSegOnlyAdult'
    target_folder = join(nnUNet_raw, target_dataset_name)

    maybe_mkdir_p(join(target_folder, 'imagesTr'))
    maybe_mkdir_p(join(target_folder, 'imagesTs'))
    maybe_mkdir_p(join(target_folder, 'labelsTr'))
    maybe_mkdir_p(join(target_folder, 'labelsTs'))

    p = Pool(16)
    res = []

    caseids = [i[:-7] for i in nifti_files(join(source_folder, 'labelsTr'), join=False)]
    for c in caseids:
        source_image = join(source_folder, 'imagesTr', c + '_0000.nii.gz')
        target_image = join(target_folder, 'imagesTr', c + '_0000.nii.gz')
        source_seg = join(source_folder, 'labelsTr', c + '.nii.gz')
        target_seg = join(target_folder, 'labelsTr', c + '.nii.gz')
        res.append(p.starmap_async(
            convert_case,
            ((source_image, source_seg, target_image, target_seg, remove_juvenile_cases), )
        ))

    caseids = [i[:-7] for i in nifti_files(join(source_folder, 'labelsTs'), join=False)]
    for c in caseids:
        source_image = join(source_folder, 'imagesTs', c + '_0000.nii.gz')
        target_image = join(target_folder, 'imagesTs', c + '_0000.nii.gz')
        source_seg = join(source_folder, 'labelsTs', c + '.nii.gz')
        target_seg = join(target_folder, 'labelsTs', c + '.nii.gz')
        res.append(p.starmap_async(
            convert_case,
            ((source_image, source_seg, target_image, target_seg, remove_juvenile_cases), )
        ))

    _ = [i.get() for i in res]
    p.close()
    p.join()

    generate_dataset_json(join(nnUNet_raw, target_dataset_name), {0: 'CT'}, {'background': 0, **{f'{i}': i for i in range(1, 37)}},
                          935, '.nii.gz', target_dataset_name)

    # nnUNetv2_plan_and_preprocess -d 169 -npfp 12 -np 6 -overwrite_plans_name nnUNetPlans_sp05 -overwrite_target_spacing 0.5 0.5 0.5 -gpu_memory_target 11 -c 3d_fullres
