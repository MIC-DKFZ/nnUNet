from natsort import natsorted
import numpy as np
from pathlib import Path
import os
from os.path import join
from nnunet.dataset_conversion.utils import generate_dataset_json
import SimpleITK as sitk
import gc
import multiprocessing as mp
from functools import partial


def preprocess_dataset(ribfrac_load_path, ribseg_load_path, dataset_save_path, pool):
    mask_load_path = join(ribseg_load_path, "labelsTr")

    train_image_save_path = join(dataset_save_path, "imagesTr")
    train_mask_save_path = join(dataset_save_path, "labelsTr")
    test_image_save_path = join(dataset_save_path, "imagesTs")
    test_labels_save_path = join(dataset_save_path, "labelsTs")
    Path(train_image_save_path).mkdir(parents=True, exist_ok=True)
    Path(train_mask_save_path).mkdir(parents=True, exist_ok=True)
    Path(test_image_save_path).mkdir(parents=True, exist_ok=True)
    Path(test_labels_save_path).mkdir(parents=True, exist_ok=True)

    mask_filenames = load_filenames(mask_load_path)
    pool.map(partial(preprocess_single, image_load_path=ribfrac_load_path), mask_filenames)


def preprocess_single(filename, image_load_path):
    name = os.path.basename(filename)
    if "-cl.nii.gz" in name:
        return
    id = int(name.split("-")[0][7:])
    image_set = "imagesTr"
    mask_set = "labelsTr"
    if id > 500:
        image_set = "imagesTs"
        mask_set = "labelsTs"
    image, _, _, _ = load_image(join(image_load_path, image_set, "RibFrac{}-image.nii.gz".format(id)), return_meta=True, is_seg=False)
    mask, spacing, _, _ = load_image(filename, return_meta=True, is_seg=True)
    save_image(join(dataset_save_path, image_set, "RibSeg_" + str(id).zfill(4) + "_0000.nii.gz"), image, spacing=spacing, is_seg=False)
    save_image(join(dataset_save_path, mask_set, "RibSeg_" + str(id).zfill(4) + ".nii.gz"), mask, spacing=spacing, is_seg=True)


def load_filenames(img_dir, extensions=None):
    _img_dir = fix_path(img_dir)
    img_filenames = []

    for file in os.listdir(_img_dir):
        if extensions is None or file.endswith(extensions):
            img_filenames.append(_img_dir + file)
    img_filenames = np.asarray(img_filenames)
    img_filenames = natsorted(img_filenames)

    return img_filenames


def fix_path(path):
    if path[-1] != "/":
        path += "/"
    return path


def load_image(filepath, return_meta=False, is_seg=False):
    image = sitk.ReadImage(filepath)
    image_np = sitk.GetArrayFromImage(image)

    if is_seg:
        image_np = np.rint(image_np)
        image_np = image_np.astype(np.int8)  # In special cases segmentations can contain negative labels, so no np.uint8

    if not return_meta:
        return image_np
    else:
        spacing = image.GetSpacing()
        keys = image.GetMetaDataKeys()
        header = {key:image.GetMetaData(key) for key in keys}
        affine = None  # How do I get the affine transform with SimpleITK? With NiBabel it is just image.affine
        return image_np, spacing, affine, header


def save_image(filename, image, spacing=None, affine=None, header=None, is_seg=False, mp_pool=None, free_mem=False):
    if is_seg:
        image = np.rint(image)
        image = image.astype(np.int8)  # In special cases segmentations can contain negative labels, so no np.uint8

    image = sitk.GetImageFromArray(image)

    if header is not None:
        [image.SetMetaData(key, header[key]) for key in header.keys()]

    if spacing is not None:
        image.SetSpacing(spacing)

    if affine is not None:
        pass  # How do I set the affine transform with SimpleITK? With NiBabel it is just nib.Nifti1Image(img, affine=affine, header=header)

    if mp_pool is None:
        sitk.WriteImage(image, filename)
        if free_mem:
            del image
            gc.collect()
    else:
        mp_pool.apply_async(_save, args=(filename, image, free_mem,))
        if free_mem:
            del image
            gc.collect()


def _save(filename, image, free_mem):
    sitk.WriteImage(image, filename)
    if free_mem:
        del image
        gc.collect()


if __name__ == "__main__":
    # Note: Due to a bug in SimpleITK 2.1.x a version of SimpleITK < 2.1.0 is required for loading images. Further, we can't copy the images and masks, but have to load them and resample both to the same spacing.
    # Conversion instructions:
    # 1. All images from both training and validation set of the RibFrac dataset need to be downloaded from https://ribfrac.grand-challenge.org/dataset/ into a new folder named RibFrac
    # 2. The RibSeg masks need to be downloaded from https://zenodo.org/record/5336592 into a new folder named RibSeg
    # 3. Follow unpacking instruction for the RibFrac dataset as in Task154_RibFrac
    # 4. Unzip RibSeg_490_nii.zip from the RibSeg dataset and rename the folder labelsTr

    ribfrac_load_path = "/home/k539i/Documents/datasets/original/RibFrac/"
    ribseg_load_path = "/home/k539i/Documents/datasets/original/RibSeg/"
    dataset_save_path = "/home/k539i/Documents/datasets/preprocessed/Task155_RibSeg/"

    max_imagesTr_id = 500

    pool = mp.Pool(processes=20)

    preprocess_dataset(ribfrac_load_path, ribseg_load_path, dataset_save_path, pool)

    print("Still saving images in background...")
    pool.close()
    pool.join()
    print("All tasks finished.")

    generate_dataset_json(join(dataset_save_path, 'dataset.json'), join(dataset_save_path, "imagesTr"), None, ('CT',), {0: 'bg', 1: 'rib'}, "Task155_RibSeg")
