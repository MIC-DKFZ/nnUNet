from __future__ import annotations
from nnunet.paths import network_training_output_dir
import traceback
from dataclasses import dataclass
from pathlib import Path

import cc3d
import numpy as np
import SimpleITK as sitk

import shutil
from multiprocessing import Pool
from time import time

from batchgenerators.utilities.file_and_folder_operations import subfiles, join
from nnunet.inference.amos2022.inference_code import predict_cases_amos2022
from nnunet.inference.ensemble_predictions import merge

final_pp_for_organs = [
    {
        "organ_id": "1",
        "dice": 0.9762470053409747,
        "json_name": "1_final_pipeline_rate0.75.json"
    },
    {
        "organ_id": "2",
        "dice": 0.9675394825341617,
        "json_name": "1_final_pipeline_rate0.10.json"
    },
    {
        "organ_id": "3",
        "dice": 0.9627731976102032,
        "json_name": "3_final_pipeline.json"
    },
    {
        "organ_id": "4",
        "dice": 0.8736083894974881,
        "json_name": "1_final_pipeline_rate0.25.json"
    },
    {
        "organ_id": "5",
        "dice": 0.8548114298877836,
        "json_name": "1_final_pipeline_rate0.25.json"
    },
    {
        "organ_id": "6",
        "dice": 0.9794434693055409,
        "json_name": "3_final_pipeline.json"
    },
    {
        "organ_id": "7",
        "dice": 0.9157975574739058,
        "json_name": "1_final_pipeline_rate0.75.json"
    },
    {
        "organ_id": "8",
        "dice": 0.9603680866311437,
        "json_name": "1_final_pipeline_rate0.75.json"
    },
    {
        "organ_id": "9",
        "dice": 0.9175585153932445,
        "json_name": "1_final_pipeline_rate0.10.json"
    },
    {
        "organ_id": "10",
        "dice": 0.8673404180730196,
        "json_name": "raw_metric.json"
    },
    {
        "organ_id": "11",
        "dice": 0.7968925183844469,
        "json_name": "raw_metric.json"
    },
    {
        "organ_id": "12",
        "dice": 0.8100986537273488,
        "json_name": "3_final_pipeline.json"
    },
    {
        "organ_id": "13",
        "dice": 0.8378947041022881,
        "json_name": "raw_metric.json"
    },
    {
        "organ_id": "14",
        "dice": 0.9230446621513407,
        "json_name": "1_final_pipeline_rate0.10.json"
    },
    {
        "organ_id": "15",
        "dice": 0.8761071641695802,
        "json_name": "raw_metric.json"
    }
]

min_gt_volume = {
    "1": 14514.522334710227,
    "2": 32323.283026438632,
    "3": 10084.375,
    "4": 1242.140625,
    "5": 418.857421875,
    "6": 653406.1819358291,
    "7": 30.576200410175325,
    "8": 26481.439208984375,
    "9": 31401.75782125006,
    "10": 15213.30078125,
    "11": 1076.1044686954742,
    "12": 663.6894633476112,
    "13": 19243.98566040845,
    "14": 14665.392605173933,
    "15": 7937.479259275528
}


@dataclass
class OrganMap:
    original_organ_id: int
    organ_voxels: int
    organ_map: np.ndarray
    final_organ_id: int = -1


def kidney_adrenal_left_right_confusion(predicted_label_map: np.ndarray, organ_ids: list[int]):
    """
    Expects predicted_label_map with dimensions: z y x
    """
    shape = predicted_label_map.shape
    mid_id_z, mid_id_y, mid_id_x = shape[0] / 2, shape[1] / 2, shape[2] / 2
    kidney_labels = {"right": 2, "left": 3}
    adrenal_labels = {"right": 11, "left": 12}

    all_bin_maps = []
    all_organ_maps = []
    for organ_label in [kidney_labels, adrenal_labels]:
        bin_organ_map = (predicted_label_map == organ_label["right"]) | (predicted_label_map == organ_label["left"])
        # Kidneys tend to be far apart -> maybe bigger rad?
        labeled_organs = cc3d.connected_components(bin_organ_map, connectivity=26)

        assigned_labels = {"right": [], "left": []}
        comp_id = 1  # Skipping background
        while True:
            bin_map = labeled_organs == comp_id
            if not np.any(bin_map):
                break
            ids = np.argwhere(bin_map)
            avg_id_z, avg_id_y, avg_id_x = list(np.mean(ids, axis=0))
            if avg_id_x < mid_id_x:
                assigned_labels["right"].append(comp_id)
            else:
                assigned_labels["left"].append(comp_id)
            comp_id += 1  # Increment.

        if len(assigned_labels["left"]) != 0:
            left_organ_bin_map = np.sum(
                np.stack([labeled_organs == i for i in assigned_labels["left"]], axis=0), axis=0
            )
        else:
            left_organ_bin_map = np.zeros_like(labeled_organs)

        if len(assigned_labels["right"]) != 0:
            right_organ_bin_map = np.sum(
                np.stack([labeled_organs == i for i in assigned_labels["right"]], axis=0), axis=0
            )
        else:
            right_organ_bin_map = np.zeros_like(labeled_organs)

        left_organ_map = left_organ_bin_map * organ_label["left"]
        right_organ_map = right_organ_bin_map * organ_label["right"]
        all_organ_maps.append(left_organ_map.copy())
        all_organ_maps.append(right_organ_map.copy())
        all_bin_maps.append(bin_organ_map.copy())

    all_bin_maps = np.sum(np.stack(all_bin_maps, axis=0), axis=0)
    all_organ_maps = np.sum(np.stack(all_organ_maps, axis=0), axis=0)
    final_organ_maps = np.where(all_bin_maps, all_organ_maps, predicted_label_map)

    return final_organ_maps


def small_organ_filtering(predicted_label_map: np.ndarray, spacing, organ_ids: list[int], rate: float = 1.0):
    assert 0.0 < rate <= 1.0, "Rate can only be between 0 and 1"
    vol_per_voxel = float(np.prod(spacing))
    organ_maps = get_connected_organ_maps(predicted_label_map, organs_of_interest=organ_ids)
    remaining_organ_maps = [
        om.organ_map
        for om in organ_maps
        if (om.organ_voxels * vol_per_voxel) > (min_gt_volume[str(om.original_organ_id)] * rate)
    ]
    if len(remaining_organ_maps) == 0:
        return np.zeros_like(predicted_label_map)
    else:
        remaining_foreground_bin_map = np.sum(np.stack(remaining_organ_maps, axis=0), axis=0).astype(bool)
        filtered_map = predicted_label_map.copy()
        filtered_map[~remaining_foreground_bin_map] = 0
        return filtered_map


def small_organ_filtering_but_at_least_one_instance(
        predicted_label_map: np.ndarray, spacing, organ_ids: list[int], rate: float = 1.0
):
    assert 0.0 < rate <= 1.0, "Rate can only be between 0 and 1"
    vol_per_voxel = float(np.prod(spacing))
    organ_maps = get_connected_organ_maps(predicted_label_map, organs_of_interest=organ_ids)

    rem_organ_bin_maps = []
    for oid in organ_ids:
        rem_oms: list[OrganMap] = [om for om in organ_maps if om.original_organ_id == oid]
        if len(rem_oms) == 1:
            # If only one no need to check size since its largest
            rem_organ_bin_maps.append(rem_oms[0].organ_map.astype(bool))
        else:
            # Determine the largest
            om_size_of_oms_with_oid = [om.organ_voxels for om in rem_oms]
            max_id = int(np.argmax(om_size_of_oms_with_oid))
            for cnt, om in enumerate(rem_oms):
                if cnt == max_id:
                    # Always add largest
                    rem_organ_bin_maps.append(om.organ_map)
                else:
                    # Maybe add the others if not too small!
                    if (om.organ_voxels * vol_per_voxel) > (min_gt_volume[str(om.original_organ_id)] * rate):
                        rem_organ_bin_maps.append(om.organ_map)
    if len(rem_organ_bin_maps) == 0:
        return np.zeros_like(predicted_label_map)
    else:
        remaining_foreground = np.sum(np.stack(rem_organ_bin_maps, axis=0), axis=0).astype(bool)
        filtered = predicted_label_map.copy()
        filtered[~remaining_foreground] = 0
        return filtered


def only_largest_region_filtering(predicted_label_map: np.ndarray, organ_ids: list[int]):
    organ_maps = get_connected_organ_maps(predicted_label_map, organs_of_interest=organ_ids)
    organ_ids = list(set([om.original_organ_id for om in organ_maps]))
    rem_organ_bin_maps = []
    for oid in organ_ids:
        rem_oms: list[OrganMap] = [om for om in organ_maps if om.original_organ_id == oid]
        om_size_of_oms_with_oid = [om.organ_voxels for om in rem_oms]
        max_id = np.argmax(om_size_of_oms_with_oid).astype(int)
        rem_organ_bin_maps.append(rem_oms[max_id].organ_map.astype(bool))
    if len(rem_organ_bin_maps) == 0:
        return np.zeros_like(predicted_label_map)
    else:
        remaining_foreground = np.sum(np.stack(rem_organ_bin_maps, axis=0), axis=0).astype(bool)
        filtered = predicted_label_map.copy()
        filtered[~remaining_foreground] = 0
    return filtered


def get_connected_organ_maps(predicted_label_map: np.ndarray, organs_of_interest: list[int]):
    """
    Goes through the different organ maps and does a connected component analysis on them.
    All the unconnected regions are saved as an `OrganMap` containing the binary map, the organ value and the name.
    """
    organ_maps: list[OrganMap] = []
    for organ_value in organs_of_interest:
        bin_org_map = predicted_label_map == organ_value
        labeled_organ_map: np.ndarray = cc3d.connected_components(bin_org_map, connectivity=26)

        comp_id = 1
        while True:
            largest_volume_bin_map: np.ndarray = labeled_organ_map == comp_id
            if not np.any(largest_volume_bin_map):
                break
            organ_maps.append(
                OrganMap(
                    organ_map=largest_volume_bin_map.astype(np.uint8),
                    original_organ_id=organ_value,
                    organ_voxels=int(np.sum(largest_volume_bin_map)),
                )
            )
            comp_id += 1
    return organ_maps


def load_image(path_to_niigz: Path):
    im: sitk.Image = sitk.ReadImage(str(path_to_niigz))
    arr = sitk.GetArrayFromImage(im)
    spacing = list(im.GetSpacing())[::-1]
    return arr, spacing


def final_filtering(pd: Path, organ_ids: list[int], rate):
    """
    Left right confusion + Filtering of small organs (independent of it beeing  the only instance)
    """
    pd_arr, pd_spacing = load_image(pd)
    if np.any(np.isin(organ_ids, [2, 3, 11, 12])):
        pd_arr = kidney_adrenal_left_right_confusion(pd_arr, organ_ids)
    pd_arr = small_organ_filtering(pd_arr, pd_spacing, organ_ids, rate)
    return pd_arr


def final_filtering2(pd: Path, organ_ids: list[int], rate):
    """
    Left right confusion + Filtering of small organs (if its not the only instance)
    """
    pd_arr, pd_spacing = load_image(pd)
    if np.any(np.isin(organ_ids, [2, 3, 11, 12])):
        pd_arr = kidney_adrenal_left_right_confusion(pd_arr, organ_ids)
    sm_pd_pp_arr = small_organ_filtering_but_at_least_one_instance(pd_arr, pd_spacing, organ_ids, rate)
    return sm_pd_pp_arr


def final_filtering3(pd: Path, organ_ids: list[int]):
    pd_arr, _ = load_image(pd)
    if np.any(np.isin( organ_ids, [2, 3, 11, 12])):
        pd_arr = kidney_adrenal_left_right_confusion(pd_arr, organ_ids)
    pd_arr = only_largest_region_filtering(pd_arr, organ_ids)
    return pd_arr


@dataclass
class PPInfo:
    func: callable
    kwargs: dict
    ids: list[int]


def only_load(pd, *args, **kwargs):
    im = sitk.ReadImage(str(pd))
    return sitk.GetArrayFromImage(im)


def get_method_from_json_name(json_name) -> tuple[callable, None | dict]:
    if json_name == "3_final_pipeline.json":
        return final_filtering3, {}
    elif json_name.startswith("1_final_pipeline_rate"):
        rate = float(json_name[21:25])
        return final_filtering, {"rate": rate}
    elif json_name.startswith("2_final_pipeline_rate"):
        rate = float(json_name[21:25])
        return final_filtering2, {"rate": rate}
    elif json_name == "left_right_kidney_adrenal.json":
        return kidney_adrenal_left_right_confusion, {}
    elif json_name == "raw_metric.json":
        return only_load, {}
    else:
        raise NotImplementedError(f"Unknown json found: {json_name}. Register function here!")


def determine_organwise_pp() -> list[PPInfo]:

    unique_jsons = np.unique([r["json_name"] for r in final_pp_for_organs])
    functions_with_organ_pairs: list[PPInfo] = []
    for uq_json in unique_jsons:
        organ_ids_of_func = [int(pp["organ_id"]) for pp in final_pp_for_organs if pp["json_name"] == uq_json]
        func, kwargs = get_method_from_json_name(uq_json)
        functions_with_organ_pairs.append(PPInfo(func=func, kwargs=kwargs, ids=organ_ids_of_func))
    return functions_with_organ_pairs


def post_process_case(case_sitk_im: Path, pp_organ_info: list[PPInfo]) -> sitk.Image:
    # All these functions only take a path and then read it themselves.
    #   If y
    all_pp_maps = [pp.func(case_sitk_im, pp.ids, **pp.kwargs) for pp in pp_organ_info]

    final_maps = []
    for pp_map, pp in zip(all_pp_maps, pp_organ_info):
        pp_maps = pp_map * np.isin(pp_map, [int(i) for i in pp.ids]).astype(
            np.uint8
        )  # Only predicted organs of interest remain
        final_maps.append(pp_maps)

    final_pp_map = np.sum(np.stack(final_maps, axis=0), axis=0).astype(np.uint8)
    case_sitk_im = str(case_sitk_im)
    # print(case_sitk_im)
    pp_image = sitk.GetImageFromArray(final_pp_map)
    pp_image.CopyInformation(sitk.ReadImage(case_sitk_im))
    return pp_image


def postprocess_image(pd_path: Path, pp_o_info: list[PPInfo], of: Path) -> None:
    try:
        pp_image = post_process_case(pd_path, pp_o_info)
        # print(of, type(of))
        sitk.WriteImage(pp_image, str(Path(of) / pd_path.name))
    except:  # Juts catch everything
        print("Encountered an error in pp but continuing with normal prediction.")
        traceback.print_exc()
    return


if __name__ == '__main__':
    delete_temp_files = True

    input_folder = '/home/fabian/temp/216_examples'  #'./input'  #
    output_folder = '/home/fabian/temp/216_examples_pred_amos2022_testBeforePublishing'   #'./output'  #
    parameter_folder = join(network_training_output_dir, '3d_fullres', 'Task216_AMOS2022_task1')
    # parameter_folder = '/home/fabian/amos2022_docker_task1/parameters' #'/parameters'  #
    folds = (0, 1, 2, 3, 4)
    do_tta = True
    mixed_precision = True
    step_size = 0.5
    num_threads_preprocessing = 3
    num_threads_nifti_save = 3
    overwrite_existing = True
    overwrite_existing_ensembling = True
    num_threads_ensembling = 4
    num_threads_postprocessing = 4

    input_files = subfiles(input_folder, suffix='.nii.gz', join=False)
    input_files_fullpath = [join(input_folder, i) for i in input_files]

    models = [
        'nnUNetTrainerV2_ResencUNet_SimonsInit_DA5__nnUNetPlans_bfnnUNet_fabresnet_31_bs5',
        'nnUNetTrainerV2_ResencUNet_SimonsInit__nnUNetPlans_bfnnUNet_fabresnet_31_spnnU_bs6',
        'nnUNetTrainerV2_ResencUNet_SimonsInit__nnUNetPlans_bfnnUNet_fabresnet_bs5'
    ]
    start = time()
    for m in models:
        output_folder_here = join(output_folder, m)
        output_files = [join(output_folder_here, i) for i in input_files]

        predict_cases_amos2022(join(parameter_folder, m), [[i] for i in input_files_fullpath], output_files, folds,
                               save_npz=True, num_threads_preprocessing=num_threads_preprocessing,
                               num_threads_nifti_save=num_threads_nifti_save,
                               segs_from_prev_stage=None, do_tta=do_tta, overwrite_existing=overwrite_existing,
                               step_size=step_size)
    end_inference = time()

    merge([join(output_folder, m) for m in models], output_folder, num_threads_ensembling,
          override=overwrite_existing_ensembling, postprocessing_file=None, store_npz=False)

    end_ensembling = time()

    if delete_temp_files:
        _ = [shutil.rmtree(join(output_folder, m)) for m in models]

    o_folder = Path(output_folder)
    pp_organ_info = determine_organwise_pp()
    all_pd_paths: list[Path] = []
    all_organ_infos: list[list[PPInfo]] = []
    all_output_folders: list[Path] = []

    for c in list(o_folder.iterdir()):
        if c.name.endswith(".nii.gz"):
            all_pd_paths.append(c)
            all_organ_infos.append(pp_organ_info)
            all_output_folders.append(o_folder)

    with Pool(num_threads_postprocessing) as p:
        p.starmap(postprocess_image, zip(all_pd_paths, all_organ_infos, all_output_folders))

    end_pp = time()
    print("Done babyyyyy!")

    print(f'inference time {end_inference - start}, '
          f'ensembling time {end_ensembling - end_inference}, '
          f'postprocessing time {end_pp - end_ensembling}, '
          f'total time {end_ensembling - start}')
