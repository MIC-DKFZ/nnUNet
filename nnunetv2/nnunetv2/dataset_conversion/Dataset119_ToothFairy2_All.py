from typing import Dict, Any
import os
from os.path import join
import json
import random
import multiprocessing

import SimpleITK as sitk
import numpy as np
from tqdm import tqdm


def mapping_DS119() -> Dict[int, int]:
    """Remove all NA Classes and make Class IDs continuous"""
    mapping = {}
    mapping.update({i: i for i in range(1, 19)})  # [1-10]->[1-10] | [11-18]->[11-18]
    mapping.update({i: i - 2 for i in range(21, 29)})  # [21-28]->[19-26]
    mapping.update({i: i - 4 for i in range(31, 39)})  # [31-38]->[27-34]
    mapping.update({i: i - 6 for i in range(41, 49)})  # [41-48]->[35-42]
    return mapping


def mapping_DS120() -> Dict[int, int]:
    """Remove Only Keep Teeth and Jaw Classes"""
    mapping = {}
    mapping.update({i: i for i in range(1, 3)})  # [0-2] -> [0-2]
    mapping.update({i: i - 8 for i in range(11, 19)})  # [11-18]->[3-10]
    mapping.update({i: i - 10 for i in range(21, 29)})  # [21-28]->[11-18]
    mapping.update({i: i - 12 for i in range(31, 39)})  # [31-38]->[19-26]
    mapping.update({i: i - 14 for i in range(41, 49)})  # [41-48]->[27-34]
    return mapping


def mapping_DS121() -> Dict[int, int]:
    """Remove Only Keep Teeth and Jaw Classes"""
    mapping = {}
    mapping.update({i: i - 10 for i in range(11, 19)})  # [11-18]->[3-8]
    mapping.update({i: i - 12 for i in range(21, 29)})  # [21-28]->[11-16]
    mapping.update({i: i - 14 for i in range(31, 39)})  # [31-38]->[19-24]
    mapping.update({i: i - 16 for i in range(41, 49)})  # [41-48]->[27-32]
    return mapping


def load_json(json_file: str) -> Any:
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def write_json(json_file: str, data: Any, indent: int = 4) -> None:
    with open(json_file, "w") as f:
        json.dump(data, f, indent=indent)


def image_to_nifi(input_path: str, output_path: str) -> None:
    image_sitk = sitk.ReadImage(input_path)
    sitk.WriteImage(image_sitk, output_path)


def label_mapping(input_path: str, output_path: str, mapping: Dict[int, int] = None) -> None:

    label_sitk = sitk.ReadImage(input_path)
    if mapping is not None:
        label_np = sitk.GetArrayFromImage(label_sitk)

        label_np_new = np.zeros_like(label_np, dtype=np.uint8)
        for org_id, new_id in mapping.items():
            label_np_new[label_np == org_id] = new_id

        label_sitk_new = sitk.GetImageFromArray(label_np_new)
        label_sitk_new.CopyInformation(label_sitk)
        sitk.WriteImage(label_sitk_new, output_path)
    else:
        sitk.WriteImage(label_sitk, output_path)


def process_images(files: str, img_dir_in: str, img_dir_out: str, n_processes: int = 12):

    os.makedirs(img_dir_out, exist_ok=True)

    iterable = [
        {
            "input_path": join(img_dir_in, file),
            "output_path": join(img_dir_out, file.replace(".mha", ".nii.gz")),
        }
        for file in files
    ]
    with multiprocessing.Pool(processes=n_processes) as pool:
        jobs = [pool.apply_async(image_to_nifi, kwds={**args}) for args in iterable]
        _ = [job.get() for job in tqdm(jobs, desc="Process Images")]


def process_labels(
    files: str, lbl_dir_in: str, lbl_dir_out: str, mapping: Dict[int, int], n_processes: int = 12
) -> None:

    os.makedirs(lbl_dir_out, exist_ok=True)

    iterable = [
        {
            "input_path": join(lbl_dir_in, file),
            "output_path": join(lbl_dir_out, file.replace(".mha", ".nii.gz")),
            "mapping": mapping,
        }
        for file in files
    ]
    with multiprocessing.Pool(processes=n_processes) as pool:
        jobs = [pool.apply_async(label_mapping, kwds={**args}) for args in iterable]
        _ = [job.get() for job in tqdm(jobs, desc="Process Labels...")]


def process_ds(
    root: str, input_ds: str, output_ds: str, mapping: dict, image_link: str = None
) -> None:
    os.makedirs(join(root, output_ds), exist_ok=True)
    os.makedirs(join(root, output_ds, "labelsTr"), exist_ok=True)
    # --- Handle Labels --- #
    lbl_files = os.listdir(join(root, input_ds, "labelsTr"))
    lbl_dir_in = join(root, input_ds, "labelsTr")
    lbl_dir_out = join(root, output_ds, "labelsTr")

    process_labels(lbl_files, lbl_dir_in, lbl_dir_out, mapping, n_processes=12)

    # --- Handle Images --- #
    img_files = os.listdir(join(root, input_ds, "imagesTr"))
    dataset = {}
    if image_link is None:
        img_dir_in = join(root, input_ds, "imagesTr")
        img_dir_out = join(root, output_ds, "imagesTr")

        process_images(img_files, img_dir_in, img_dir_out, n_processes=12)
    else:
        base_name = [file.replace("_0000.mha", "") for file in img_files]
        for name in base_name:
            dataset[name] = {
                "images": [join("..", image_link, "imagesTr", name + "_0000.nii.gz")],
                "label": join("labelsTr", name + ".nii.gz"),
            }

    # --- Generate dataset.json --- #
    dataset_json = load_json(join(root, input_ds, "dataset.json"))
    dataset_json["file_ending"] = ".nii.gz"
    dataset_json["name"] = output_ds
    dataset_json["numTraining"] = len(lbl_files)
    if dataset != {}:
        dataset_json["dataset"] = dataset

    label_dict = dataset_json["labels"]
    label_dict_new = {"background": 0}
    for k, v in label_dict.items():
        if v in mapping.keys():
            label_dict_new[k] = mapping[v]
    dataset_json["labels"] = label_dict_new
    write_json(join(root, output_ds, "dataset.json"), dataset_json)

    # --- Generate splits_final.json --- #
    img_names = [file.replace("_0000.mha", "") for file in img_files]

    random_seed = 42
    random.seed(random_seed)
    random.shuffle(img_names)

    split_index = int(len(img_names) * 0.7)  # 70:30 split
    train_files = img_names[:split_index]
    val_files = img_names[split_index:]
    train_files.sort()
    val_files.sort()

    split = [{"train": train_files, "val": val_files}]
    write_json(join(root, output_ds, "splits_final.json"), split)


if __name__ == "__main__":
    # Different nnUNet Datasets
    # Dataset 112: Raw
    # Dataset 119: Replace NaN classes
    # Dataset 120: Only Teeth + Jaw Classes
    # Dataset 121: Only Teeth Classes

    root = "/media/l727r/data/Teeth_Data/ToothFairy2_Dataset"

    process_ds(root, "Dataset112_ToothFairy2", "Dataset119_ToothFairy2_All", mapping_DS119(), None)
    # process_ds(
    #     root,
    #     "Dataset112_ToothFairy2",
    #     "Dataset120_ToothFairy2_JawTeeth",
    #     mapping_DS120(),
    #     "Dataset119_ToothFairy2_All",
    # )
    # process_ds(
    #     root,
    #     "Dataset112_ToothFairy2",
    #     "Dataset121_ToothFairy2_Teeth",
    #     mapping_DS121(),
    #     "Dataset119_ToothFairy2_All",
    # )
