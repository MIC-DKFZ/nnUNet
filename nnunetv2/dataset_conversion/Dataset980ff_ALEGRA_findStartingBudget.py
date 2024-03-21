from multiprocessing import Pool
from typing import Tuple

import numpy as np
import shutil

from acvl_utils.array_manipulation.resampling import maybe_resample_on_gpu
from acvl_utils.morphology.gpu_binary_morphology import gpu_binary_erosion
import torch
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.utilities.sitk_stuff import copy_geometry
from torch.backends import cudnn
from skimage.morphology import ball

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
import nnunetv2.paths as paths
import SimpleITK as sitk
from torch.nn import functional as F

from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape


def resample_save(source_image: str, source_label: str, target_image: str, target_label: str,
                  target_spacing: Tuple[float, ...] = (0.3, 0.3, 0.3), skip_existing: bool = True,
                  export_pool: Pool = None):
    print(f'{os.path.basename(source_image)}')
    if skip_existing and isfile(target_label) and isfile(target_image):
        return None, None

    seg_source = sitk.GetArrayFromImage(sitk.ReadImage(source_label)).astype(np.uint8)
    im_source = sitk.ReadImage(source_image)

    source_spacing = im_source.GetSpacing()
    source_origin = im_source.GetOrigin()
    source_direction = im_source.GetDirection()

    im_source = sitk.GetArrayFromImage(im_source).astype(np.float32)
    source_shape = im_source.shape

    # resample image
    target_shape = compute_new_shape(source_shape, list(source_spacing)[::-1], target_spacing)

    print(f'source shape: {source_shape}, target shape {target_shape}')

    # one hot generation is slow af. Let's do it this way:
    seg_source = torch.from_numpy(seg_source)
    seg_onehot_target_shape = None
    seg_source_gpu = None
    try:
        torch.cuda.empty_cache()
        device = 'cuda:0'
        # having the target array on device will blow up, so we need to have this on CPU
        with torch.no_grad():
            seg_source_gpu = seg_source.to(device)
            seg_onehot_target_shape = F.interpolate(seg_source_gpu.half()[None, None], tuple(target_shape), mode='trilinear')[0, 0].cpu()
        del seg_source_gpu
    except RuntimeError:
        print('GPU wasnt happy with this resampling. Lets give the CPU a chance to sort it out')
        print(f'source shape {source_shape}, target shape {target_shape}')
        del seg_source_gpu
        device = 'cpu'
        with torch.no_grad():
            seg_onehot_target_shape = F.interpolate(seg_source.to(device).float()[None, None], tuple(target_shape), mode='trilinear')[0, 0].cpu()
    finally:
        torch.cuda.empty_cache()

    seg_onehot_target_shape = (seg_onehot_target_shape > 0.5).numpy().astype(np.uint8)

    seg_target_itk = sitk.GetImageFromArray(seg_onehot_target_shape)
    seg_target_itk.SetSpacing(tuple(list(target_spacing)[::-1]))
    seg_target_itk.SetOrigin(source_origin)
    seg_target_itk.SetDirection(source_direction)

    # now resample images. For simplicity, just make this linear
    im_source = maybe_resample_on_gpu(torch.from_numpy(im_source[None]), tuple(target_shape), return_type=torch.float,
                                      compute_precision=torch.float, fallback_compute_precision=float)[0].cpu().numpy()

    # export image
    im_target = sitk.GetImageFromArray(im_source)
    im_target.SetSpacing(tuple(list(target_spacing)[::-1]))
    im_target.SetOrigin(source_origin)
    im_target.SetDirection(source_direction)

    if export_pool is None:
        sitk.WriteImage(im_target, target_image)
        sitk.WriteImage(seg_target_itk, target_label)
        return None, None
    else:
        r1 = export_pool.starmap_async(sitk.WriteImage, ((im_target, target_image),))
        r2 = export_pool.starmap_async(sitk.WriteImage, ((seg_target_itk, target_label),))
        return r1, r2


def sample_starting_budget_patches(orig_seg: np.ndarray, num_patches: int, patch_size: Tuple[int, int, int],
                          ignore_label: int = 2, allowed_overlap_percent: float = 0) -> np.ndarray:
    def check_if_patch_is_allowed(prospective_slicer, current_new_seg):
        patch_pixels = np.prod(patch_size)
        num_nonignore = np.sum(current_new_seg[prospective_slicer] != ignore_label)
        # print(num_nonignore / patch_pixels, allowed_overlap_percent, num_nonignore / patch_pixels <= allowed_overlap_percent)
        return num_nonignore / patch_pixels <= allowed_overlap_percent

    out_seg = np.ones_like(orig_seg, dtype=np.uint8) * ignore_label
    cudnn.deterministic = False
    cudnn.benchmark = False

    print("computing seg border")
    orig_seg_border = orig_seg - gpu_binary_erosion(orig_seg > 0, ball(1))
    border_locs = np.argwhere(orig_seg_border > 0)
    del orig_seg_border

    # the following line should be adapted in case more classes are present!
    print("computing seg locations")
    seg_locs = np.argwhere(orig_seg > 0)
    print("sampling patches")
    for pi in range(num_patches):
        num_attempts = 0
        # random strat
        rnd = np.random.uniform(0, 1)
        if rnd < 0.33:
            print(pi, 'random patch location')
            # random location
            while True:
                lb = [np.random.randint(0, orig_seg.shape[i] - patch_size[i]) for i in range(3)]
                slicer = tuple([slice(lb[i], lb[i] + patch_size[i]) for i in range(3)])
                if check_if_patch_is_allowed(slicer, out_seg) or num_attempts > 1000:
                    break
                else:
                    num_attempts += 1
            out_seg[slicer] = orig_seg[slicer]

        elif 0.33 <= rnd < 0.67:
            # pick random pixel
            print(pi, 'random pixel is patch center')
            while True:
                random_loc = seg_locs[np.random.choice(seg_locs.shape[0])]
                lb = [max(0, i - j // 2) for i, j in zip(random_loc, patch_size)]
                lb = [min(i - j, k) for i, j, k in zip(orig_seg.shape, patch_size, lb)]
                slicer = tuple([slice(lb[i], lb[i] + patch_size[i]) for i in range(3)])
                if check_if_patch_is_allowed(slicer, out_seg) or num_attempts > 1000:
                    break
                else:
                    num_attempts += 1
            out_seg[slicer] = orig_seg[slicer]

        else:
            # pick random border pixel
            print(pi, 'random border pixel is patch center')
            while True:
                random_loc = border_locs[np.random.choice(border_locs.shape[0])]
                lb = [max(0, i - j // 2) for i, j in zip(random_loc, patch_size)]
                lb = [min(i - j, k) for i, j, k in zip(orig_seg.shape, patch_size, lb)]
                slicer = tuple([slice(lb[i], lb[i] + patch_size[i]) for i in range(3)])
                if check_if_patch_is_allowed(slicer, out_seg) or num_attempts > 1000:
                    break
                else:
                    num_attempts += 1
            out_seg[slicer] = orig_seg[slicer]
    return out_seg


def generate_dataset_with_only_patched(source_dataset_dir, target_dataset_name, num_patches, patch_size, ignore_label: int = 2):
    target_dataset_dir = join(paths.nnUNet_raw, target_dataset_name)
    tr_segs = nifti_files(join(source_dataset_dir, 'labelsTr'), join=False)
    np.random.shuffle(tr_segs)
    patches_per_image = num_patches / len(tr_segs)
    maybe_mkdir_p(join(target_dataset_dir, 'labelsTr'))
    num_patches_taken = 0
    for i, tr_seg in zip(np.arange(patches_per_image, num_patches+1e-8, step=patches_per_image), tr_segs):
        num_patches_here = round(i - num_patches_taken)
        num_patches_taken += num_patches_here
        print('num_patches_here', num_patches_here)
        old_seg = sitk.ReadImage(join(source_dataset_dir, 'labelsTr', tr_seg))
        new_seg = sample_starting_budget_patches(sitk.GetArrayFromImage(old_seg), num_patches_here, patch_size, ignore_label)
        new_seg = sitk.GetImageFromArray(new_seg)
        new_seg = copy_geometry(new_seg, old_seg)
        sitk.WriteImage(new_seg, join(target_dataset_dir, 'labelsTr', tr_seg))
    if isdir(join(target_dataset_dir, 'imagesTr')):
        shutil.rmtree(join(target_dataset_dir, 'imagesTr'))
    shutil.copytree(join(source_dataset_dir, 'imagesTr'), join(target_dataset_dir, 'imagesTr'))


if __name__ == '__main__':
    valset_predictions_dir = '/home/isensee/drives/E132-Projekte/Projects/Helmholtz_Imaging_ACVL/HMGU_2021_MurineAirwaySegmentation/nnUNet_parameters/3d_fullres/Task145_LungAirwaySegmentation/nnUNetTrainerV2_airwayAug_new__AirwaySegPlanner/predictedValSet'
    raw_dataset_old_dir = '/home/isensee/drives/E132-Rohdaten/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task145_LungAirwaySegmentation'
    # need to be consistent with previous experiments
    target_spacing = (20., 10.318, 10.318)
    patch_size = (48, 224, 224)

    ###################### first resample the old dataset to target spacing in a temp folder ######################
    dataset_name = 'Dataset980_ALEGRA_fullyAnnotated'
    imagesTr = join(paths.nnUNet_raw, dataset_name, 'imagesTr')
    maybe_mkdir_p(imagesTr)
    imagesTs = join(paths.nnUNet_raw, dataset_name, 'imagesTs')
    maybe_mkdir_p(imagesTs)
    imagesVal = join(paths.nnUNet_raw, dataset_name, 'imagesVal')
    maybe_mkdir_p(imagesVal)
    labelsTr = join(paths.nnUNet_raw, dataset_name, 'labelsTr')
    maybe_mkdir_p(labelsTr)
    labelsTs = join(paths.nnUNet_raw, dataset_name, 'labelsTs')
    maybe_mkdir_p(labelsTs)
    labelsVal = join(paths.nnUNet_raw, dataset_name, 'labelsVal')
    maybe_mkdir_p(labelsVal)

    p = Pool(8)

    # train set
    train_identifiers = [i[:-7] for i in nifti_files(join(raw_dataset_old_dir, 'labelsTr'), join=False)]
    for tr in train_identifiers:
        resample_save(join(raw_dataset_old_dir, 'imagesTr', tr + '_0000.nii.gz'),
                      join(raw_dataset_old_dir, 'labelsTr', tr + '.nii.gz'),
                      target_image=join(imagesTr, tr + '_0000.nii.gz'),
                      target_label=join(labelsTr, tr + '.nii.gz'),
                      target_spacing=target_spacing, skip_existing=True,
                      export_pool=p)

    # val set. Note that labelsVal will be populated with predictions from our segmentation model! NOT manual GT!
    val_identifiers = [i[:-12] for i in nifti_files(join(raw_dataset_old_dir, 'imagesVal'), join=False)]
    for v in val_identifiers:
        label_image = join(valset_predictions_dir, v + '.nii.gz')
        resample_save(join(raw_dataset_old_dir, 'imagesVal', v + '_0000.nii.gz'),
                      label_image,
                      target_image=join(imagesVal, v + '_0000.nii.gz'),
                      target_label=join(labelsVal, v + '.nii.gz'),
                      target_spacing=target_spacing, skip_existing=True,
                      export_pool=p)

    # test set
    test_identifiers = [i[:-7] for i in nifti_files(join(raw_dataset_old_dir, 'labelsTs_fixed'), join=False)]
    for ts in test_identifiers:
        resample_save(join(raw_dataset_old_dir, 'imagesTs_fixed', ts + '_0000.nii.gz'),
                      join(raw_dataset_old_dir, 'labelsTs_fixed', ts + '.nii.gz'),
                      target_image=join(imagesTs, ts + '_0000.nii.gz'),
                      target_label=join(labelsTs, ts + '.nii.gz'),
                      target_spacing=target_spacing, skip_existing=True,
                      export_pool=p)

    p.close()
    p.join()

    # generate custom splits.pkl to simulate fewer train cases
    np.random.seed(1234)
    pp_out_dir = join(paths.nnUNet_preprocessed, dataset_name)
    maybe_mkdir_p(pp_out_dir)
    splits = []
    # 0, 1, 2 -> 1 train case
    for s in range(3):
        tr_cases = list(np.random.choice(train_identifiers, 1, replace=False))
        val_cases = [i for i in train_identifiers if i not in tr_cases]
        splits.append({'train': tr_cases, 'val': val_cases})
    # 3, 4, 5 -> 3 train cases
    for s in range(3):
        tr_cases = list(np.random.choice(train_identifiers, 3, replace=False))
        val_cases = [i for i in train_identifiers if i not in tr_cases]
        splits.append({'train': tr_cases, 'val': val_cases})
    # 6, 7, 8 -> 5 train cases
    for s in range(3):
        tr_cases = list(np.random.choice(train_identifiers, 5, replace=False))
        val_cases = [i for i in train_identifiers if i not in tr_cases]
        splits.append({'train': tr_cases, 'val': val_cases})
    # 9, 10, 11 -> 10 train cases
    for s in range(3):
        tr_cases = list(np.random.choice(train_identifiers, 10, replace=False))
        val_cases = [i for i in train_identifiers if i not in tr_cases]
        splits.append({'train': tr_cases, 'val': val_cases})
    # 12, 13, 14 -> 15 train cases
    for s in range(3):
        tr_cases = list(np.random.choice(train_identifiers, 15, replace=False))
        val_cases = [i for i in train_identifiers if i not in tr_cases]
        splits.append({'train': tr_cases, 'val': val_cases})
    save_json(splits, join(pp_out_dir, 'splits_final.json'))

    # generate dataset.json
    generate_dataset_json(join(paths.nnUNet_raw, dataset_name), {0: 'LSmicroscopy'}, {'background': 0, 'airway': 1}, 21,
                          '.nii.gz', dataset_name=dataset_name)

    # now patches source datasets
    np.random.seed(1234)
    # less than 21 makes no sense because there are 21 train images
    num_patches = (21, 50, 100, 200)
    for i, n in enumerate(num_patches):
        dataset_name_here = "Dataset%03.0d_ALEGRA_startBudget_%03.0dpatches" % (981 + i, n)
        # generate_dataset_with_only_patched(join(nnUNet_raw, dataset_name),
        #                                    dataset_name_here,
        #                                    n, patch_size, 2)
        generate_dataset_json(join(paths.nnUNet_raw, dataset_name_here),
                              {0: 'LSmicroscopy'},
                              {'background': 0, 'airway': 1, 'ignore': 2}, n,
                              '.nii.gz', dataset_name=dataset_name_here)
