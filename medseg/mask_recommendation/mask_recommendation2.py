from medseg import utils
import numpy as np
from tqdm import tqdm
import os
from skimage import measure
import copy
import argparse
from shutil import copyfile
import time
import subprocess
import signal
from evaluate import evaluate
import pickle
import random
import multiprocessing as mp
from functools import partial
import pandas as pd
from pathlib import Path
import GeodisTK
import shutil
import json
import medseg.mask_recommendation.my_method2 as my_method2
from  medseg.mask_recommendation.my_method2 import comp_uncertainty_and_prediction as comp_uncertainty_and_prediction
import pandas as pd


def recommend_slices_parallel(prediction_path, uncertainty_path, gt_path, save_path):
    prediction_filenames = utils.load_filenames(prediction_path)
    uncertainty_filenames = utils.load_filenames(uncertainty_path)
    gt_filenames = utils.load_filenames(gt_path)

    # start_time = time.time()
    print("Starting slice recommendation...")
    results = pool.map(partial(recommend_slices_single_case,
                                                   prediction_filenames=prediction_filenames, uncertainty_filenames=uncertainty_filenames, gt_filenames=gt_filenames,
                                                   save_path=save_path),
                                           range(len(uncertainty_filenames)))
    print("Finished slice recommendation.")
    return results


def recommend_slices_single_case(i, prediction_filenames, uncertainty_filenames, gt_filenames, save_path, debug=False):
    uncertainty, affine, spacing, header = utils.load_nifty(uncertainty_filenames[i])
    prediction, _, _, _ = utils.load_nifty(prediction_filenames[i])
    gt, _, _, _ = utils.load_nifty(gt_filenames[i])
    # indices_dim_0: Sagittal
    # indices_dim_1: Coronal
    # indices_dim_2: Axial
    filtered_mask, recommended_slices, recommended_patch_area = find_best_slices(gt, uncertainty)
    slices = gt.shape[2]
    gt_slices = comp_infected_slices(gt)
    size = np.prod(gt.shape)
    infection_size = gt_slices * np.prod(gt.shape[:2])
    prediction_slices = comp_infected_slices(prediction)
    if debug:
        print("name: {}, slices: {}, gt inf slices: {}, pred inf slices: {}, rec slices: {}, rec slice ratio: {}, rec slice inf ratio: {}, patch ratio: {}, patch inf ratio: {}".format(
            os.path.basename(uncertainty_filenames[i]),
            slices,
            gt_slices,
            prediction_slices,
            recommended_slices,
            recommended_slices / slices,
            recommended_slices / gt_slices,
            recommended_patch_area / size,
            recommended_patch_area / infection_size))
    utils.save_nifty(save_path + os.path.basename(uncertainty_filenames[i])[:-7] + "_" + str(modality).zfill(4) + ".nii.gz", filtered_mask, affine, spacing, header, is_mask=True)
    results = {}
    results["total_slices"] = slices
    results["gt_infected_slices"] = gt_slices
    results["pred_infected_slices"] = prediction_slices
    results["recommended_slices"] = recommended_slices
    results["size"] = size
    results["infection_size"] = infection_size
    results["recommended_patch_area"] = recommended_patch_area
    return results


def find_best_slices(gt, uncertainty):
    uncertainty_dim_0 = np.sum(-1 * uncertainty, axis=(1, 2))
    uncertainty_dim_1 = np.sum(-1 * uncertainty, axis=(0, 2))
    uncertainty_dim_2 = np.sum(-1 * uncertainty, axis=(0, 1))
    indices_dim_0 = [np.argsort(uncertainty_dim_0)[0]]
    indices_dim_1 = [np.argsort(uncertainty_dim_1)[0]]
    indices_dim_2 = [np.argsort(uncertainty_dim_2)[0]]
    recommended_slices = len(indices_dim_0) + len(indices_dim_1) + len(indices_dim_2)
    filtered_mask = filter_mask(gt, indices_dim_0, indices_dim_1, indices_dim_2)
    recommended_patch_area = np.prod(filtered_mask[:, :, 0].shape)
    return filtered_mask, recommended_slices, recommended_patch_area


def filter_mask(mask, indices_dim_0, indices_dim_1, indices_dim_2):
    slices = np.zeros_like(mask)

    for index in indices_dim_0:
        slices[int(index), :, :] = 1

    for index in indices_dim_1:
        slices[:, int(index), :] = 1

    for index in indices_dim_2:
        slices[:, :, int(index)] = 1

    # filtered_mask = copy.deepcopy(mask)
    # filtered_mask = np.logical_and(filtered_mask, slices)
    if method == "my_method" or method == "DeepIGeos1" or method == "DeepIGeos2":
        filtered_mask = np.zeros_like(mask)
    else:
        filtered_mask = np.ones_like(mask) * -1
    unique = np.unique(mask)
    for label in unique:
        filtered_mask[(slices == 1) & (mask == label)] = label

    return filtered_mask


def comp_infected_slices(mask):
    mask_slices = np.sum(mask, axis=(0, 1))
    mask_slices = np.count_nonzero(mask_slices)
    return mask_slices


def comp_uncertainty(devices, uncertainty_tmp_path, refined_prediction_save_path):
    comp_uncertainty_and_prediction(devices, recommended_masks_path, refinement_inference, uncertainty_tmp_path, prediction_path, model, class_labels, refined_prediction_save_path)


def eval_test_set(save_dir, version, method, params, reuse):
    result = eval_single_hyperparameters(params, reuse)
    with open(save_dir + version + "_" + method + ".pkl", 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return


def eval_single_hyperparameters(params, reuse):
    print("Starting hyperparam evaluation...")
    print(params)
    shutil.rmtree(recommended_masks_path, ignore_errors=True)
    Path(recommended_masks_path).mkdir(parents=True, exist_ok=True)
    shutil.rmtree(refined_prediction_save_path, ignore_errors=True)
    Path(refined_prediction_save_path).mkdir(parents=True, exist_ok=True)

    recommended_slices, recommended_patch_area = None, None
    current_uncertainty_path = uncertainty_path
    for i in range(params["num_slices"]):
        if i >= 1:
            current_uncertainty_path = uncertainty_tmp_path + "uncertainties/"
        if not reuse:
            recommended_result = recommend_slices_parallel(prediction_path, current_uncertainty_path, gt_path, recommended_masks_path)
            if recommended_slices is None:
                recommended_slices, recommended_patch_area = np.zeros(len(recommended_result)), np.zeros(len(recommended_result))
            recommended_slices += np.asarray([x["recommended_slices"]for x in recommended_result])
            recommended_patch_area += np.asarray([x["recommended_patch_area"]for x in recommended_result])
        reuse = False
        # prediction_result = my_method2.compute_predictions(devices, recommended_masks_path, prediction_path, gt_path, refined_prediction_save_path, refinement_inference_tmp, model, class_labels)
        comp_uncertainty(devices, uncertainty_tmp_path, refined_prediction_save_path)
    for i in range(len(recommended_result)):
        recommended_result[i]["recommended_slices"] = recommended_slices
        recommended_result[i]["recommended_patch_area"] = recommended_patch_area
    prediction_result = evaluate(gt_path, refined_prediction_save_path, class_labels)

    # if debug:
    #     print("inf slice ratio: {}, inf patch ratio: {}".format(np.sum([r["recommended_slices"] for r in recommended_result]) / np.sum([r["gt_infected_slices"] for r in recommended_result]),
    #                                                             np.sum([r["recommended_patch_area"] for r in recommended_result]) / np.sum([r["infection_size"] for r in recommended_result])))
    return {"recommended_result": recommended_result, "prediction_result": prediction_result}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", help="Set the task name", required=True)
    parser.add_argument("-m", "--model", help="Set the model name", required=True)
    parser.add_argument("-s", "--set", help="val/test", required=True)
    parser.add_argument("-uq", "--uncertainty_quantification", help="Set the type of uncertainty quantification method to use", required=True)
    parser.add_argument("-um", "--uncertainty_measure", help="Set the type of uncertainty measure to use", required=True)
    parser.add_argument("-modality", help="Set the modality number", required=True)
    parser.add_argument("--reuse", action="store_true", default=False, help="Reuse recommended masks from last run", required=False)
    args = parser.parse_args()
    devices = [2, 3, 4, 5, 6]

    version = "V10"
    uncertainty_quantification = str(args.uncertainty_quantification)
    uncertainty_measure = str(args.uncertainty_measure)

    if uncertainty_quantification == "e":
        uncertainty_quantification = "ensemble"
    elif uncertainty_quantification == "t":
        uncertainty_quantification = "tta"
    elif uncertainty_quantification == "m":
        uncertainty_quantification = "mcdo"
    else:
        raise RuntimeError("uncertainty_quantification unknown")

    if uncertainty_measure == "b":
        uncertainty_measure = "bhattacharyya_coefficient"
    elif uncertainty_measure == "e":
        uncertainty_measure = "predictive_entropy"
    elif uncertainty_measure == "v":
        uncertainty_measure = "predictive_variance"
    else:
        raise RuntimeError("uncertainty_measure unknown")

    task = args.task  # "Task072_allGuided_ggo"
    model = args.model
    set = args.set  # "val"
    method = "my_method2"
    task_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/" + task + "/"
    base_path = task_path + "refinement_" + set + "/"
    image_path = base_path + "/images/"
    prediction_path = base_path + "/basic_predictions/"
    uncertainty_path = base_path + "/uncertainties/" + uncertainty_quantification + "/" + uncertainty_measure + "/"
    uncertainty_tmp_path = base_path + "/uncertainties_tmp/"
    gt_path = base_path + "/labels/"
    recommended_masks_path = base_path + "/recommended_masks/" + version + "/" + method + "/"
    refined_prediction_save_path = base_path + "/refined_predictions/" + method
    grid_search_save_path = base_path + "/GridSearchResults/"
    test_set_save_path = base_path + "/eval_results/raw/"
    refinement_inference = base_path + "/refinement_inference/"
    modality = int(args.modality)
    reuse = args.reuse

    pool = mp.Pool(processes=8)  # 8
    shutil.rmtree(recommended_masks_path, ignore_errors=True)
    Path(recommended_masks_path).mkdir(parents=True, exist_ok=True)
    shutil.rmtree(refined_prediction_save_path, ignore_errors=True)
    Path(refined_prediction_save_path).mkdir(parents=True, exist_ok=True)

    with open(task_path + "dataset.json") as f:
        class_labels = json.load(f)
    class_labels = np.asarray(list(class_labels["labels"].keys())).astype(int)

    # slice_gap = [20]  # [20, 25]
    # num_slices = [12]
    # grid_search(grid_search_save_path, version, slice_gap, num_slices, 1280, devices, args.parallel)

    default_params = {}
    default_params["slice_gap"] = 20  # 20
    default_params["num_slices"] = 12
    default_params["max_slices_based_on_infected_slices"] = 0.5  # 0.5, 0.2
    default_params["min_uncertainty"] = 0.15  # 0.0
    default_params["default_size"] = 1280

    params = {}
    # # params["slice_gap"] = [10, 15, 20, 25, 30, 40, 50, 70, 80, 90, 100, 110, 120, 130]
    # # params["num_slices"] = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # params["max_slices_based_on_infected_slices"] = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    params["min_uncertainty"] = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    test_set_params = {}
    test_set_params["slice_gap"] = 20
    test_set_params["num_slices"] = 12
    test_set_params["max_slices_based_on_infected_slices"] = 0.23
    test_set_params["min_uncertainty"] = 0.10
    test_set_params["default_size"] = 1280

    # eval_all_hyperparameters(grid_search_save_path, version, method, default_params, params, devices, args.parallel)

    eval_test_set(test_set_save_path, version, method, test_set_params, reuse)

    # pkl2csv("/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task072_allGuided_ggo/GridSearchResults/grid_search_results_V6.pkl")
    print("Test")
