import SimpleITK as sitk
from natsort import natsorted
import numpy as np
from pathlib import Path
import pandas as pd
from collections import defaultdict
from shutil import copyfile
import os
from os.path import join
from tqdm import tqdm
import gc
import multiprocessing as mp
from nnunet.dataset_conversion.utils import generate_dataset_json
from functools import partial


def preprocess_dataset(dataset_load_path, dataset_save_path, pool):
    train_image_load_path = join(dataset_load_path, "imagesTr")
    train_mask_load_path = join(dataset_load_path, "labelsTr")
    test_image_load_path = join(dataset_load_path, "imagesTs")

    ribfrac_train_info_1_path = join(dataset_load_path, "ribfrac-train-info-1.csv")
    ribfrac_train_info_2_path = join(dataset_load_path, "ribfrac-train-info-2.csv")
    ribfrac_val_info_path = join(dataset_load_path, "ribfrac-val-info.csv")

    train_image_save_path = join(dataset_save_path, "imagesTr")
    train_mask_save_path = join(dataset_save_path, "labelsTr")
    test_image_save_path = join(dataset_save_path, "imagesTs")
    Path(train_image_save_path).mkdir(parents=True, exist_ok=True)
    Path(train_mask_save_path).mkdir(parents=True, exist_ok=True)
    Path(test_image_save_path).mkdir(parents=True, exist_ok=True)

    meta_data = preprocess_csv(ribfrac_train_info_1_path, ribfrac_train_info_2_path, ribfrac_val_info_path)
    preprocess_train(train_image_load_path, train_mask_load_path, meta_data, dataset_save_path, pool)
    preprocess_test(test_image_load_path, dataset_save_path)


def preprocess_csv(ribfrac_train_info_1_path, ribfrac_train_info_2_path, ribfrac_val_info_path):
    meta_data = defaultdict(list)
    for csv_path in [ribfrac_train_info_1_path, ribfrac_train_info_2_path, ribfrac_val_info_path]:
        df = pd.read_csv(csv_path)
        for index, row in df.iterrows():
            name = row["public_id"]
            instance = row["label_id"]
            class_label = row["label_code"]
            meta_data[name].append({"instance": instance, "class_label": class_label})
    return meta_data


def preprocess_train(image_path, mask_path, meta_data, save_path, pool):
    pool.map(partial(preprocess_train_single, image_path=image_path, mask_path=mask_path, meta_data=meta_data, save_path=save_path), meta_data.keys())


def preprocess_train_single(name, image_path, mask_path, meta_data, save_path):
    id = int(name[7:])
    image, _, _, _ = load_image(join(image_path, name + "-image.nii.gz"), return_meta=True, is_seg=False)
    instance_seg_mask, spacing, _, _ = load_image(join(mask_path, name + "-label.nii.gz"), return_meta=True, is_seg=True)
    semantic_seg_mask = np.zeros_like(instance_seg_mask, dtype=int)
    for entry in meta_data[name]:
        semantic_seg_mask[instance_seg_mask == entry["instance"]] = entry["class_label"]
    semantic_seg_mask[semantic_seg_mask == -1] = 5  # Set ignore label to 5
    save_image(join(save_path, "imagesTr/RibFrac_" + str(id).zfill(4) + "_0000.nii.gz"), image, spacing=spacing, is_seg=False)
    save_image(join(save_path, "labelsTr/RibFrac_" + str(id).zfill(4) + ".nii.gz"), semantic_seg_mask, spacing=spacing, is_seg=True)


def preprocess_test(load_test_image_dir, save_path):
    filenames = load_filenames(load_test_image_dir)
    for filename in tqdm(filenames):
        id = int(os.path.basename(filename)[8:-13])
        copyfile(filename, join(save_path, "imagesTs/RibFrac_" + str(id).zfill(4) + "_0000.nii.gz"))


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
    # 1. All sets, parts and CSVs need to be downloaded from https://ribfrac.grand-challenge.org/dataset/
    # 2. Unzip ribfrac-train-images-1.zip (will be unzipped as Part1) and ribfrac-train-images-2.zip (will be unzipped as Part2), move content from Part2 to Part1 and rename the folder to imagesTr
    # 3. Unzip ribfrac-train-labels-1.zip (will be unzipped as Part1) and ribfrac-train-labels-2.zip (will be unzipped as Part2), move content from Part2 to Part1 and rename the folder to labelsTr
    # 4. Unzip ribfrac-val-images.zip and add content to imagesTr, repeat with ribfrac-val-labels.zip
    # 5. Unzip ribfrac-test-images.zip and rename it to imagesTs

    pool = mp.Pool(processes=20)

    dataset_load_path = "/home/k539i/Documents/datasets/original/RibFrac/"
    dataset_save_path = "/home/k539i/Documents/datasets/preprocessed/Task154_RibFrac/"
    preprocess_dataset(dataset_load_path, dataset_save_path, pool)

    print("Still saving images in background...")
    pool.close()
    pool.join()
    print("All tasks finished.")

    labels = {0: "background", 1: "displaced_rib_fracture", 2: "non_displaced_rib_fracture", 3: "buckle_rib_fracture", 4: "segmental_rib_fracture", 5: "unidentified_rib_fracture"}
    generate_dataset_json(join(dataset_save_path, 'dataset.json'), join(dataset_save_path, "imagesTr"), None, ('CT',), labels, "Task154_RibFrac")
