from i3Deep import utils
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
import i3Deep.mask_recommendation.my_method as my_method
import i3Deep.mask_recommendation.deep_i_geos as deep_i_geos
import i3Deep.mask_recommendation.graph_cut as graph_cut
import i3Deep.mask_recommendation.random_walker as random_walker
import i3Deep.mask_recommendation.watershed as watershed
from scipy.ndimage.measurements import center_of_mass
import pandas as pd


# def recommend_slices(image_path, prediction_path, uncertainty_path, gt_path, save_path, find_best_slices_func, num_slices, slice_gap, default_size, deepigeos=0):
#     image_filenames = utils.load_filenames(image_path)
#     prediction_filenames = utils.load_filenames(prediction_path)
#     uncertainty_filenames = utils.load_filenames(uncertainty_path)
#     gt_filenames = utils.load_filenames(gt_path)
#     total_recommended_slices = 0
#     total_gt_slices = 0
#
#     for i in tqdm(range(len(uncertainty_filenames))):
#         uncertainty, affine, spacing, header = utils.load_nifty(uncertainty_filenames[i])
#         prediction, _, _, _ = utils.load_nifty(prediction_filenames[i])
#         gt, _, _, _ = utils.load_nifty(gt_filenames[i])
#         if deepigeos != 0:
#             image, _, _, _ = utils.load_nifty(image_filenames[i])
#         adapted_slice_gap = adapt_slice_gap(uncertainty, slice_gap, default_size)
#         # indices_dim_0: Sagittal
#         # indices_dim_1: Coronal
#         # indices_dim_2: Axial
#         indices_dim_0, indices_dim_1, indices_dim_2 = find_best_slices_func(prediction, uncertainty, num_slices, adapted_slice_gap)
#         recommended_slices = len(indices_dim_0) + len(indices_dim_1) + len(indices_dim_2)
#         gt_slices = comp_infected_slices(gt)
#         total_recommended_slices += recommended_slices
#         total_gt_slices += gt_slices
#         print("name: {} recommended slices: {}, gt slices: {}, ratio: {}".format(os.path.basename(uncertainty_filenames[i]), recommended_slices, gt_slices, recommended_slices / gt_slices))
#         # print("indices_dim_0: {}, indices_dim_1: {}, indices_dim_2: {}".format(indices_dim_0, indices_dim_1, indices_dim_2))
#         filtered_mask = filter_mask(gt, indices_dim_0, indices_dim_1, indices_dim_2)
#         if deepigeos != 0:
#             if deepigeos == 1:
#                 geodisc_lambda = 0.99
#             else:
#                 geodisc_lambda = 0.0
#             geodisc_map = filtered_mask.astype(np.uint8)
#             geodisc_map[geodisc_map < 0] = 0
#             filtered_mask = GeodisTK.geodesic3d_raster_scan(image.astype(np.float32).squeeze(0), geodisc_map, spacing.astype(np.float32), geodisc_lambda, 1)
#         utils.save_nifty(save_path + os.path.basename(uncertainty_filenames[i])[:-7] + "_0001.nii.gz", filtered_mask, affine, spacing, header, is_mask=True)
#     total_ratio = total_recommended_slices / total_gt_slices
#     print("total recommended slices: {}, total gt slices: {}, total ratio: {}".format(total_recommended_slices, total_gt_slices, total_ratio))
#     return total_ratio


def recommend_slices_parallel(prediction_path, uncertainty_path, gt_path, save_path, find_best_slices_func, params):
    prediction_filenames = utils.load_filenames(prediction_path)
    uncertainty_filenames = utils.load_filenames(uncertainty_path)
    gt_filenames = utils.load_filenames(gt_path)

    # start_time = time.time()
    print("Starting slice recommendation...")
    results = pool.map(partial(recommend_slices_single_case,
                                                   prediction_filenames=prediction_filenames, uncertainty_filenames=uncertainty_filenames, gt_filenames=gt_filenames,
                                                   save_path=save_path, find_best_slices_func=find_best_slices_func, params=params),
                                           range(len(uncertainty_filenames)))
    print("Finished slice recommendation.")
    # print("Recommend slices elapsed time: ", time.time() - start_time)
    # results = np.asarray(results)
    # total_recommended_slices = results[:, 0]
    # total_gt_slices = results[:, 1]
    # total_recommended_slices = np.sum(total_recommended_slices)
    # total_gt_slices = np.sum(total_gt_slices)
    # total_ratio = total_recommended_slices / total_gt_slices
    # print("total recommended slices: {}, total gt slices: {}, total ratio: {}".format(total_recommended_slices, total_gt_slices, total_ratio))
    return results


def recommend_slices_single_case(i, prediction_filenames, uncertainty_filenames, gt_filenames, save_path, find_best_slices_func, params, debug=False):
    uncertainty, affine, spacing, header = utils.load_nifty(uncertainty_filenames[i])
    prediction, _, _, _ = utils.load_nifty(prediction_filenames[i])
    gt, _, _, _ = utils.load_nifty(gt_filenames[i])
    params["slice_gap"] = adapt_slice_gap(uncertainty, params["slice_gap"], params["default_size"])
    # indices_dim_0: Sagittal
    # indices_dim_1: Coronal
    # indices_dim_2: Axial
    filtered_mask, recommended_slices, recommended_patch_area = find_best_slices_func(prediction, gt, uncertainty, params, os.path.basename(gt_filenames[i]))
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


def adapt_slice_gap(mask, slice_gap, default_size):
    return int((mask.shape[0] / default_size) * slice_gap)


def find_best_slices_baseline_V1(prediction, gt, uncertainty, params):
    "Find random slices in every dimension without a min slice gap."
    _, recommended_slices, _ = find_best_slices_V7(prediction, gt, uncertainty, params)
    num_slices = int(recommended_slices / 3)
    indices_dim_0 = random.sample(range(uncertainty.shape[0]), num_slices)
    indices_dim_1 = random.sample(range(uncertainty.shape[1]), num_slices)
    indices_dim_2 = random.sample(range(uncertainty.shape[2]), num_slices)
    recommended_slices = len(indices_dim_0) + len(indices_dim_1) + len(indices_dim_2)
    filtered_mask = filter_mask(gt, indices_dim_0, indices_dim_1, indices_dim_2)
    recommended_patch_area = np.prod(filtered_mask[:, :, 0])
    return filtered_mask, recommended_slices, recommended_patch_area


def find_best_slices_baseline_V2(prediction, gt, uncertainty, params):
    "Find random slices in every dimension within equidistant distances."
    _, recommended_slices, _ = find_best_slices_V7(prediction, gt, uncertainty, params)
    num_slices = int(recommended_slices / 3)
    def get_indices(axis):
        slice_sectors = np.linspace(0, uncertainty.shape[axis], num_slices, endpoint=True).astype(int)
        indices = [random.randint(slice_sectors[i-1], slice_sectors[i]) for i in range(1, len(slice_sectors)-1)]
        return indices

    indices_dim_0 = get_indices(0)
    indices_dim_1 = get_indices(1)
    indices_dim_2 = get_indices(2)
    recommended_slices = len(indices_dim_0) + len(indices_dim_1) + len(indices_dim_2)
    filtered_mask = filter_mask(gt, indices_dim_0, indices_dim_1, indices_dim_2)
    recommended_patch_area = np.prod(filtered_mask[:, :, 0].shape)
    return filtered_mask, recommended_slices, recommended_patch_area


def find_best_slices_baseline_V3(prediction, gt, uncertainty, params):
    "Find random slices in every dimension within equidistant distances."
    _, recommended_slices, _ = find_best_slices_V7(prediction, gt, uncertainty, params)
    num_slices = int(recommended_slices / 3)

    def get_start_end_index(axis):
        _axis = [0, 1, 2]
        _axis.remove(axis)
        _axis = tuple(_axis)
        prediction_dim = np.sum(prediction, axis=_axis)
        start = (prediction_dim!=0).argmax()
        prediction_dim = np.flip(prediction_dim)
        end = (prediction_dim!=0).argmax()
        end = len(prediction_dim) - end
        return start, end

    def get_indices(axis):
        start, end = get_start_end_index(axis)
        slice_sectors = np.linspace(start, end, num_slices, endpoint=True).astype(int)
        # print("Start: {}, end: {}, slice_sectors: {}".format(start, end, slice_sectors))
        indices = [random.randint(slice_sectors[i-1], slice_sectors[i]) for i in range(1, len(slice_sectors)-1)]
        return indices

    indices_dim_0 = get_indices(0)
    indices_dim_1 = get_indices(1)
    indices_dim_2 = get_indices(2)
    recommended_slices = len(indices_dim_0) + len(indices_dim_1) + len(indices_dim_2)
    filtered_mask = filter_mask(gt, indices_dim_0, indices_dim_1, indices_dim_2)
    recommended_patch_area = np.prod(filtered_mask[:, :, 0].shape)
    return filtered_mask, recommended_slices, recommended_patch_area


def find_best_slices_V1(prediction, gt, uncertainty, params):
    "Find best slices based on maximum 2D plane uncertainty in every dimension without a min slice gap."
    uncertainty_dim_0 = np.sum(-1*uncertainty, axis=(1, 2))
    uncertainty_dim_1 = np.sum(-1*uncertainty, axis=(0, 2))
    uncertainty_dim_2 = np.sum(-1*uncertainty, axis=(0, 1))
    indices_dim_0 = np.argsort(uncertainty_dim_0)[:params["num_slices"]]
    indices_dim_1 = np.argsort(uncertainty_dim_1)[:params["num_slices"]]
    indices_dim_2 = np.argsort(uncertainty_dim_2)[:params["num_slices"]]
    # dim_0 = uncertainty_dim_0[indices_dim_0]*-1
    # dim_1 = uncertainty_dim_1[indices_dim_1]*-1
    # dim_2 = uncertainty_dim_2[indices_dim_2]*-1

    recommended_slices = len(indices_dim_0) + len(indices_dim_1) + len(indices_dim_2)
    filtered_mask = filter_mask(gt, indices_dim_0, indices_dim_1, indices_dim_2)
    recommended_patch_area = np.prod(filtered_mask[:, :, 0].shape)
    return filtered_mask, recommended_slices, recommended_patch_area


def find_best_slices_V2(prediction, gt, uncertainty, params):
    "Find best slices based on maximum 2D plane uncertainty in every dimension with a min slice gap."
    uncertainty_dim_0 = np.sum(-1*uncertainty, axis=(1, 2))
    uncertainty_dim_1 = np.sum(-1*uncertainty, axis=(0, 2))
    uncertainty_dim_2 = np.sum(-1*uncertainty, axis=(0, 1))
    indices_dim_0 = np.argsort(uncertainty_dim_0)
    indices_dim_1 = np.argsort(uncertainty_dim_1)
    indices_dim_2 = np.argsort(uncertainty_dim_2)
    indices_dim_0 = filter_indices(indices_dim_0, num_slices, slice_gap)
    indices_dim_1 = filter_indices(indices_dim_1, num_slices, slice_gap)
    indices_dim_2 = filter_indices(indices_dim_2, num_slices, slice_gap)
    # dim_0 = uncertainty_dim_0[indices_dim_0]*-1
    # dim_1 = uncertainty_dim_1[indices_dim_1]*-1
    # dim_2 = uncertainty_dim_2[indices_dim_2]*-1
    recommended_slices = len(indices_dim_0) + len(indices_dim_1) + len(indices_dim_2)
    filtered_mask = filter_mask(gt, indices_dim_0, indices_dim_1, indices_dim_2)
    recommended_patch_area = np.prod(filtered_mask[:, :, 0])
    return filtered_mask, recommended_slices, recommended_patch_area


def find_best_slices_V3(prediction, gt, uncertainty, params):
    # Threshold uncertainty values
    # Find and partition blobs
    # For each blob:
    #   Find center of mass:
    #       Mean of X
    #       Mean of Y
    #       Mean of Z
    #   Place three planes in center with normal pointing X, Y and Z
    # Done
    # V4 include uncertainties with V2
    indices_dim_0, indices_dim_1, indices_dim_2 = [], [], []
    prediction = utils.normalize(prediction)
    prediction[prediction >= 0.5] = 1
    prediction[prediction < 0.5] = 0
    labeled = measure.label(prediction, background=False, connectivity=2)
    unique, counts = np.unique(labeled, return_counts=True)
    unique = unique[1:]  # Remove zero / background
    counts = counts[1:]  # Remove zero / background
    sorted_indices = np.argsort(counts)[::-1]
    unique = unique[sorted_indices][:5]

    # blob_masks = []
    # for label in unique:
    #     blob_mask = copy.deepcopy(labeled)
    #     blob_mask[blob_mask != label] = 0
    #     blob_mask[blob_mask == label] = 1
    #     # sub_blob_masks = partition_blob(blob_mask)
    #     # blob_masks.extend(sub_blob_masks)
    #     blob_masks.append(blob_mask)

    for label in unique:
        blob_mask = copy.deepcopy(labeled)
        blob_mask[blob_mask != label] = 0
        blob_mask[blob_mask == label] = 1
        def find_center(dim):
            dims = [0, 1, 2]
            dims.remove(dim)
            blob_mask_dim = np.sum(blob_mask, axis=tuple(dims))
            nonzero_indices = np.nonzero(blob_mask_dim)
            min_index = np.min(nonzero_indices)
            max_index = np.max(nonzero_indices)
            center = int(min_index + ((max_index - min_index) / 2))
            return center
        indices_dim_0.append(find_center(0))
        indices_dim_1.append(find_center(1))
        indices_dim_2.append(find_center(2))
    recommended_slices = len(indices_dim_0) + len(indices_dim_1) + len(indices_dim_2)
    filtered_mask = filter_mask(gt, indices_dim_0, indices_dim_1, indices_dim_2)
    recommended_patch_area = np.prod(filtered_mask[:, :, 0])
    return filtered_mask, recommended_slices, recommended_patch_area


def find_best_slices_V4(prediction, gt, uncertainty, params):
    indices_dim_0_V2, indices_dim_1_V2, indices_dim_2_V2 = find_best_slices_V2(prediction, uncertainty, num_slices, slice_gap)
    indices_dim_0_V3, indices_dim_1_V3, indices_dim_2_V3 = find_best_slices_V3(prediction, uncertainty, num_slices, slice_gap)
    indices_dim_0 = np.concatenate((indices_dim_0_V2, indices_dim_0_V3), axis=0)
    indices_dim_1 = np.concatenate((indices_dim_1_V2, indices_dim_1_V3), axis=0)
    indices_dim_2 = np.concatenate((indices_dim_2_V2, indices_dim_2_V3), axis=0)
    recommended_slices = len(indices_dim_0) + len(indices_dim_1) + len(indices_dim_2)
    filtered_mask = filter_mask(gt, indices_dim_0, indices_dim_1, indices_dim_2)
    recommended_patch_area = np.prod(filtered_mask[:, :, 0])
    return filtered_mask, recommended_slices, recommended_patch_area


def find_best_slices_V5(prediction, gt, uncertainty, params):
    "Find best slices based on maximum 2D plane uncertainty in every dimension with a min slice gap. Take merge indices from every dim and take only the best ignoring the dims."
    uncertainty_dim_0 = np.sum(-1*uncertainty, axis=(1, 2))
    uncertainty_dim_1 = np.sum(-1*uncertainty, axis=(0, 2))
    uncertainty_dim_2 = np.sum(-1*uncertainty, axis=(0, 1))
    indices_dim_0 = np.argsort(uncertainty_dim_0)
    indices_dim_1 = np.argsort(uncertainty_dim_1)
    indices_dim_2 = np.argsort(uncertainty_dim_2)
    indices_dim_0 = filter_indices(indices_dim_0, num_slices*100, slice_gap)
    indices_dim_1 = filter_indices(indices_dim_1, num_slices*100, slice_gap)
    indices_dim_2 = filter_indices(indices_dim_2, num_slices*100, slice_gap)
    sum_dim_0 = uncertainty_dim_0[indices_dim_0]
    sum_dim_1 = uncertainty_dim_0[indices_dim_1]
    sum_dim_2 = uncertainty_dim_0[indices_dim_2]
    dim_0_indices = np.ones(len(sum_dim_0)) * 0
    dim_1_indices = np.ones(len(sum_dim_1)) * 1
    dim_2_indices = np.ones(len(sum_dim_2)) * 2
    sum = np.concatenate([sum_dim_0, sum_dim_1, sum_dim_2], axis=0)
    dim_indices = np.concatenate([dim_0_indices, dim_1_indices, dim_2_indices], axis=0)
    indices_dim = np.concatenate([indices_dim_0, indices_dim_1, indices_dim_2], axis=0)
    indices = np.argsort(sum)[:num_slices]
    dim_indices = dim_indices[indices]
    indices_dim = indices_dim[indices]
    indices_dim_0, indices_dim_1, indices_dim_2 = [], [], []
    for i in range(len(indices)):
        if dim_indices[i] == 0:
            indices_dim_0.append(indices_dim[i])
        elif dim_indices[i] == 1:
            indices_dim_1.append(indices_dim[i])
        else:
            indices_dim_2.append(indices_dim[i])
    recommended_slices = len(indices_dim_0) + len(indices_dim_1) + len(indices_dim_2)
    filtered_mask = filter_mask(gt, indices_dim_0, indices_dim_1, indices_dim_2)
    recommended_patch_area = np.prod(filtered_mask[:, :, 0])
    return filtered_mask, recommended_slices, recommended_patch_area


def find_best_slices_V6(prediction, gt, uncertainty, params):
    "Find best slices based on maximum 2D plane uncertainty in every dimension with a min slice gap. Weight uncertainty based on slice prediction sum."
    uncertainty_dim_0 = np.sum(-1*uncertainty, axis=(1, 2))
    uncertainty_dim_1 = np.sum(-1*uncertainty, axis=(0, 2))
    uncertainty_dim_2 = np.sum(-1*uncertainty, axis=(0, 1))
    prediction_dim_0 = np.sum(prediction, axis=(1, 2))
    prediction_dim_1 = np.sum(prediction, axis=(0, 2))
    prediction_dim_2 = np.sum(prediction, axis=(0, 1))
    prediction_dim_0 = utils.normalize(prediction_dim_0)
    prediction_dim_1 = utils.normalize(prediction_dim_1)
    prediction_dim_2 = utils.normalize(prediction_dim_2)
    uncertainty_dim_0 = uncertainty_dim_0 * prediction_dim_0
    uncertainty_dim_1 = uncertainty_dim_1 * prediction_dim_1
    uncertainty_dim_2 = uncertainty_dim_2 * prediction_dim_2
    indices_dim_0 = np.argsort(uncertainty_dim_0)
    indices_dim_1 = np.argsort(uncertainty_dim_1)
    indices_dim_2 = np.argsort(uncertainty_dim_2)
    indices_dim_0 = filter_indices(indices_dim_0, num_slices, slice_gap)
    indices_dim_1 = filter_indices(indices_dim_1, num_slices, slice_gap)
    indices_dim_2 = filter_indices(indices_dim_2, num_slices, slice_gap)
    recommended_slices = len(indices_dim_0) + len(indices_dim_1) + len(indices_dim_2)
    filtered_mask = filter_mask(gt, indices_dim_0, indices_dim_1, indices_dim_2)
    recommended_patch_area = np.prod(filtered_mask[:, :, 0])
    return filtered_mask, recommended_slices, recommended_patch_area


def find_best_slices_V7(prediction, gt, uncertainty, params, name):
    "Like V2, but filters out all slices with less than 40% of summed uncertainty than that of the max slice"
    uncertainty_dim_0 = np.sum(-1*uncertainty, axis=(1, 2))
    uncertainty_dim_1 = np.sum(-1*uncertainty, axis=(0, 2))
    uncertainty_dim_2 = np.sum(-1*uncertainty, axis=(0, 1))
    indices_dim_0 = np.argsort(uncertainty_dim_0)
    indices_dim_1 = np.argsort(uncertainty_dim_1)
    indices_dim_2 = np.argsort(uncertainty_dim_2)
    indices_dim_0 = filter_indices(indices_dim_0, params["num_slices"], params["slice_gap"])
    indices_dim_1 = filter_indices(indices_dim_1, params["num_slices"], params["slice_gap"])
    indices_dim_2 = filter_indices(indices_dim_2, params["num_slices"], params["slice_gap"])
    uncertainty_dim_0 *= -1
    uncertainty_dim_1 *= -1
    uncertainty_dim_2 *= -1

    def filter_by_required_uncertainty(uncertainty_dim, indices_dim):
        min_required_uncertainty = uncertainty_dim[indices_dim[0]] * params["min_uncertainty"]
        indices_dim = [index_dim for index_dim in indices_dim if uncertainty_dim[index_dim] >= min_required_uncertainty]
        return indices_dim

    indices_dim_0 = filter_by_required_uncertainty(uncertainty_dim_0, indices_dim_0)
    indices_dim_1 = filter_by_required_uncertainty(uncertainty_dim_1, indices_dim_1)
    indices_dim_2 = filter_by_required_uncertainty(uncertainty_dim_2, indices_dim_2)

    num_infected_slices = comp_infected_slices(prediction)
    num_infected_slices = int((num_infected_slices * params["max_slices_based_on_infected_slices"]) / 3)
    if num_infected_slices == 0:
        num_infected_slices = 1
    indices_dim_0 = indices_dim_0[:num_infected_slices]
    indices_dim_1 = indices_dim_1[:num_infected_slices]
    indices_dim_2 = indices_dim_2[:num_infected_slices]

    recommended_slices = len(indices_dim_0) + len(indices_dim_1) + len(indices_dim_2)
    filtered_mask = filter_mask(gt, indices_dim_0, indices_dim_1, indices_dim_2)
    recommended_patch_area = np.prod(filtered_mask[:, :, 0].shape)

    # indices_dim_0: Sagittal
    # indices_dim_1: Coronal
    # indices_dim_2: Axial
    chosen_slices = {}
    chosen_slices["Subject"] = name
    chosen_slices["Sagittal"] = [int(index) for index in indices_dim_0]
    chosen_slices["Coronal"] = [int(index) for index in indices_dim_1]
    chosen_slices["Axial"] = [int(index) for index in indices_dim_2]
    with open(choosen_slices_export_path + name[:-7] + '.json', 'w', encoding='utf-8') as f:
        json.dump(chosen_slices, f, ensure_ascii=False, indent=4)

    return filtered_mask, recommended_slices, recommended_patch_area


def find_best_slices_V8(prediction, gt, uncertainty, params, min_uncertainty=0.15, max_slices_based_on_infected_slices=0.20, roi_uncertainty=0.95):
    "Like V2, but filters out all slices with less than 40% of summed uncertainty than that of the max slice, and computes ROIs that capture 95% of uncertainty"
    uncertainty_dim_0 = np.sum(-1*uncertainty, axis=(1, 2))
    uncertainty_dim_1 = np.sum(-1*uncertainty, axis=(0, 2))
    uncertainty_dim_2 = np.sum(-1*uncertainty, axis=(0, 1))
    indices_dim_0 = np.argsort(uncertainty_dim_0)
    indices_dim_1 = np.argsort(uncertainty_dim_1)
    indices_dim_2 = np.argsort(uncertainty_dim_2)
    indices_dim_0 = filter_indices(indices_dim_0, params["num_slices"], params["slice_gap"])
    indices_dim_1 = filter_indices(indices_dim_1, params["num_slices"], params["slice_gap"])
    indices_dim_2 = filter_indices(indices_dim_2, params["num_slices"], params["slice_gap"])
    uncertainty_dim_0 *= -1
    uncertainty_dim_1 *= -1
    uncertainty_dim_2 *= -1

    def filter_by_required_uncertainty(uncertainty_dim, indices_dim):
        min_required_uncertainty = uncertainty_dim[indices_dim[0]] * min_uncertainty
        indices_dim = [index_dim for index_dim in indices_dim if uncertainty_dim[index_dim] >= min_required_uncertainty]
        return indices_dim

    indices_dim_0 = filter_by_required_uncertainty(uncertainty_dim_0, indices_dim_0)
    indices_dim_1 = filter_by_required_uncertainty(uncertainty_dim_1, indices_dim_1)
    indices_dim_2 = filter_by_required_uncertainty(uncertainty_dim_2, indices_dim_2)

    num_infected_slices = comp_infected_slices(prediction)
    num_infected_slices = int((num_infected_slices * max_slices_based_on_infected_slices) / 3)
    # num_infected_slices = int(num_infected_slices * max_slices_based_on_infected_slices)
    if num_infected_slices == 0:
        num_infected_slices = 1
    indices_dim_0 = indices_dim_0[:num_infected_slices]
    indices_dim_1 = indices_dim_1[:num_infected_slices]
    indices_dim_2 = indices_dim_2[:num_infected_slices]

    recommended_slices = len(indices_dim_0) + len(indices_dim_1) + len(indices_dim_2)
    filtered_mask = filter_mask(gt, indices_dim_0, indices_dim_1, indices_dim_2)

    def find_patches(filtered_mask, uncertainty, indices_dim_0, indices_dim_1, indices_dim_2):
        # Find objects based on uncertainty
        # Crop objects
        # for each object: patch_mask[object] = filtered_mask[object]
        if method == "my_method" or method == "DeepIGeos1" or method == "DeepIGeos2":
            patch_mask = np.zeros_like(filtered_mask)
        else:
            patch_mask = np.ones_like(filtered_mask) * -1
        recommended_patch_area = 0
        indices_dim_all = [indices_dim_0, indices_dim_1, indices_dim_2]

        for axis in range(3):
            for index in indices_dim_all[axis]:
                # center = np.rint(center_of_mass(uncertainty[index, :, :])).astype(int)
                center = np.rint(center_of_mass(np.moveaxis(uncertainty, axis, 0)[index, :, :])).astype(int)
                total_mass = np.sum(np.moveaxis(uncertainty, axis, 0)[index, :, :])
                roi_size = 5
                # while total_mass * roi_uncertainty > np.sum(uncertainty[index, center[0]-roi_size:center[0]+roi_size, center[1]-roi_size:center[1]+roi_size]):
                while total_mass * roi_uncertainty > np.sum(np.moveaxis(uncertainty, axis, 0)[index, center[0]-roi_size:center[0]+roi_size, center[1]-roi_size:center[1]+roi_size]):
                    roi_size += 1
                # recommended_patch_area += roi_size*roi_size*4
                patch_mask_tmp = np.moveaxis(patch_mask, axis, 0)
                filtered_mask_tmp = np.moveaxis(filtered_mask, axis, 0)
                filtered_mask_tmp_patch = filtered_mask_tmp[index, center[0] - roi_size:center[0] + roi_size, center[1] - roi_size:center[1] + roi_size]
                recommended_patch_area += filtered_mask_tmp_patch.size
                patch_mask_tmp[index, center[0] - roi_size:center[0] + roi_size, center[1] - roi_size:center[1] + roi_size] = filtered_mask_tmp_patch
                patch_mask = np.moveaxis(patch_mask_tmp, 0, axis)
                filtered_mask = np.moveaxis(filtered_mask_tmp, 0, axis)
                # patch_mask[index, center[0]-roi_size:center[0]+roi_size, center[1]-roi_size:center[1]+roi_size] = filtered_mask[index, center[0]-roi_size:center[0]+roi_size, center[1]-roi_size:center[1]+roi_size]


        # for index in indices_dim_1:
        #     center = np.rint(center_of_mass(uncertainty[:, index, :])).astype(int)
        #     total_mass = np.sum(uncertainty[:, index, :])
        #     roi_size = 1
        #     while total_mass * roi_uncertainty > np.sum(uncertainty[center[0]-roi_size:center[0]+roi_size, index, center[1]-roi_size:center[1]+roi_size]):
        #         roi_size += 1
        #     recommended_patch_area += roi_size * roi_size * 4
        #     patch_mask[center[0]-roi_size:center[0]+roi_size, index, center[1]-roi_size:center[1]+roi_size] = filtered_mask[center[0]-roi_size:center[0]+roi_size, index, center[1]-roi_size:center[1]+roi_size]
        # for index in indices_dim_2:
        #     center = np.rint(center_of_mass(uncertainty[:, :, index])).astype(int)
        #     total_mass = np.sum(uncertainty[:, :, index])
        #     roi_size = 1
        #     while total_mass * roi_uncertainty > np.sum(uncertainty[center[0]-roi_size:center[0]+roi_size, center[1]-roi_size:center[1]+roi_size, index]):
        #         roi_size += 1
        #     recommended_patch_area += roi_size * roi_size * 4
        #     patch_mask[center[0]-roi_size:center[0]+roi_size, center[1]-roi_size:center[1]+roi_size, index] = filtered_mask[center[0]-roi_size:center[0]+roi_size, center[1]-roi_size:center[1]+roi_size, index]
        return patch_mask, recommended_patch_area

    patch_mask, recommended_patch_area = find_patches(filtered_mask, uncertainty, indices_dim_0, indices_dim_1, indices_dim_2)
    # recommended_patch_area = np.prod(filtered_mask[:, :, 0])
    return patch_mask, recommended_slices, recommended_patch_area


def find_best_slices_V9(prediction, uncertainty, slice_gap, params):
    "Idee: V2, aber nur die slices nehmen die noch min 60% so viel uncertainty haben wie die erste slice + für jede slice den Bereich eingrenzen der 70% der Uncertainty umfasst"
    "Neue metriken dafür machen: selektierte anzahl pixel / gesamt anzahl pixel ; selektierte anzahl infected pixel / gesamt anzahl infected pixel ; Absolute anzahl von selektieren patches"
    pass


def find_best_slices_V10(prediction, uncertainty, slice_gap, params):
    """Idee für V5: V2 aber uncertainties 5 Grad drehen, resamplen V2 ausführen, das ganze 360/5=72 mal"""
    pass



# def partition_blob(blob_mask):
#     def find_side_length(blob_mask, dim):
#         dims = [0, 1, 2]
#         dims.remove(dim)
#         blob_mask_dim = np.sum(blob_mask, axis=tuple(dims))
#         nonzero_indices = np.nonzero(blob_mask_dim)
#         min_index = np.min(nonzero_indices)
#         max_index = np.max(nonzero_indices)
#         side_length = max_index - min_index
#         return side_length
#     return [blob_mask]


def filter_indices(indices, num_slices, slice_gap):
    index = 1
    while index < len(indices) and index < num_slices:  # TODO: index < num_slices: Wrong as not every index is accepted, so index should go over num_slices, but not more that len(acquired_slices)
        tmp = indices[:index]
        if np.min(np.abs(tmp - indices[index])) <= slice_gap:  # TODO: Replace indices[index] with just index? indices[index] is a sum.
            indices = np.delete(indices, index)
        else:
            index += 1
    indices = indices[:num_slices]
    return indices


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
    if method == "my_method" or method == "DeepIGeos1" or method == "DeepIGeos2" or method == "P_Net_BrainTumor" or method == "P_Net_Pancreas":
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


def eval_all_hyperparameters(save_dir, version, method, default_params, params, devices, parallel):
    result_params = {}
    key_name = str(list(params.keys()))
    pbar = tqdm(total=sum([len(params[param_key]) for param_key in params.keys()]))
    for param_key in params.keys():
        result_param_values = {}
        for param_value in params[param_key]:
            current_params = copy.deepcopy(default_params)
            current_params[param_key] = param_value
            result = eval_single_hyperparameters(current_params, parallel)
            result_param_values[param_value] = result
            pbar.update(1)
            #print("Results saved.")
            result_params[param_key] = result_param_values
            with open(save_dir + "hyperparam_eval_results_" + version + "_" + method + "_" + key_name + ".pkl", 'wb') as handle:
                pickle.dump(result_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
        result_params[param_key] = result_param_values
        with open(save_dir + "hyperparam_eval_results_" + version + "_" + method + "_" + key_name + ".pkl", 'wb') as handle:
            pickle.dump(result_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pbar.close()
    # print("Saving hyperparam evaluation...")
    # with open(save_dir + "hyperparam_eval_results_" + version + "_" + method + ".pkl", 'wb') as handle:
    #     pickle.dump(result_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Hyperparam evaluation finished.")


def eval_test_set(save_dir, version, method, params, parallel):
    result = eval_single_hyperparameters(params, parallel)
    with open(save_dir + version + "_" + method + "_" + uncertainty_quantification + ".pkl", 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return


def eval_single_hyperparameters(params, parallel, debug=False):
    print("Starting hyperparam evaluation...")
    print(params)
    if not reuse:
        shutil.rmtree(recommended_masks_path, ignore_errors=True)
        Path(recommended_masks_path).mkdir(parents=True, exist_ok=True)
    shutil.rmtree(refined_prediction_save_path, ignore_errors=True)
    Path(refined_prediction_save_path).mkdir(parents=True, exist_ok=True)
    if not parallel and not reuse:
        # total_ratio = recommend_slices(image_path, prediction_path, uncertainty_path, gt_path, save_path, find_best_slices_func, num_slices, slice_gap, default_size)
        recommended_result = None
    elif not reuse:
        recommended_result = recommend_slices_parallel(prediction_path, uncertainty_path, gt_path, recommended_masks_path, find_best_slices_func, params)
    else:
        recommended_result = None
    prediction_result = compute_predictions()
    # prediction_result = None
    if debug:
        print("inf slice ratio: {}, inf patch ratio: {}".format(np.sum([r["recommended_slices"] for r in recommended_result]) / np.sum([r["gt_infected_slices"] for r in recommended_result]),
                                                                np.sum([r["recommended_patch_area"] for r in recommended_result]) / np.sum([r["infection_size"] for r in recommended_result])))
    return {"recommended_result": recommended_result, "prediction_result": prediction_result}


# def grid_search(save_dir, version, slice_gap_list, num_slices_list, default_size, devices, parallel):
#     results = []
#     if os.path.isfile(save_dir + "grid_search_results_" + version + ".pkl"):
#         with open(save_dir + "grid_search_results_" + version + ".pkl", 'rb') as handle:
#             results = pickle.load(handle)
#     print(results)
#     for slice_gap in slice_gap_list:
#         for num_slices in num_slices_list:
#             print("slice_gap: {}, default_size: {}, num_slices: {}".format(slice_gap, default_size, num_slices))
#             if not parallel and not reuse:
#                 # total_ratio = recommend_slices(image_path, prediction_path, uncertainty_path, gt_path, save_path, find_best_slices_func, num_slices, slice_gap, default_size)
#                 pass
#             elif not reuse:
#                 total_ratio = recommend_slices_parallel(prediction_path, uncertainty_path, gt_path, save_path, find_best_slices_func, num_slices, slice_gap, default_size)
#             else:
#                 total_ratio = -1
#             mean_dice_score, median_dice_score = compute_predictions()
#             results.append({"slice_gap": slice_gap, "num_slices": num_slices, "total_ratio": total_ratio, "mean_dice_score": mean_dice_score, "median_dice_score": median_dice_score})
#             with open(save_dir + "grid_search_results_" + version + ".pkl", 'wb') as handle:
#                 pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     print(results)


def compute_predictions():
    if method == "my_method" or method == "P_Net_BrainTumor" or method == "P_Net_Pancreas" or method == "P_Net_Covid":
        recommended_result = my_method.compute_predictions(devices, recommended_masks_path, prediction_path, gt_path, refined_prediction_save_path, refinement_inference_tmp, model, class_labels, method)
    # elif method == "DeepIGeos1":
    #     recommended_result = deep_i_geos.compute_predictions(devices, recommended_masks_path, image_path, prediction_path, gt_path, refined_prediction_save_path, refinement_inference_tmp, model, class_labels, 0.99)
    # elif method == "DeepIGeos2":
    #     recommended_result = deep_i_geos.compute_predictions(devices, recommended_masks_path, image_path, prediction_path, gt_path, refined_prediction_save_path, refinement_inference_tmp, model, class_labels, 0.00)
    elif method == "GraphCut1":
        recommended_result = graph_cut.compute_predictions(image_path, recommended_masks_path, gt_path, refined_prediction_save_path + "/", method, modality, class_labels)
    elif method == "GraphCut2":
        recommended_result = graph_cut.compute_predictions(image_path, recommended_masks_path, gt_path, refined_prediction_save_path + "/", method, modality, class_labels)
    elif method == "GraphCut3":
        recommended_result = graph_cut.compute_predictions(image_path, recommended_masks_path, gt_path, refined_prediction_save_path + "/", method, modality, class_labels)
    elif method == "random_walker":
        recommended_result = random_walker.compute_predictions(image_path, recommended_masks_path, gt_path, refined_prediction_save_path + "/", modality, class_labels)
    elif method == "watershed":
        recommended_result = watershed.compute_predictions(image_path, recommended_masks_path, gt_path, refined_prediction_save_path + "/", modality, class_labels)
    return recommended_result


def pkl2csv(filename):
    with open(filename, 'rb') as handle:
        results = pickle.load(handle)

    index = np.sort(np.unique([result["slice_gap"] for result in results]))
    columns = np.sort(np.unique([result["num_slices"] for result in results]))
    df = pd.DataFrame(index=index, columns=columns)
    # slice_gap, num_slices
    for result in results:
        df.at[result["slice_gap"], result["num_slices"]] = [result["total_ratio"], result["dice_score"]]
    print(df)
    df.to_csv(filename[:-4] + ".csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", help="Set the task name", required=True)
    parser.add_argument("-m", "--model", help="Set the model name", required=True)
    parser.add_argument("-s", "--set", help="val/test", required=True)
    parser.add_argument("-v", "--version", help="Set the version", required=True)
    parser.add_argument("-uq", "--uncertainty_quantification", help="Set the type of uncertainty quantification method to use", required=True)
    parser.add_argument("-um", "--uncertainty_measure", help="Set the type of uncertainty measure to use", required=True)
    parser.add_argument("--parallel", action="store_true", default=False, help="Set the version", required=False)
    parser.add_argument("-method", help="Set the method", required=True)
    parser.add_argument("-modality", help="Set the modality number", required=True)
    parser.add_argument("--reuse", action="store_true", default=False, help="Reuse recommended masks from last run", required=False)
    parser.add_argument("-a", "--apply", help="Apply for inference (infer) or hyperparameter evaluation (eval)", required=True)
    args = parser.parse_args()
    devices = [0, 1, 2]

    version = str(args.version)
    uncertainty_quantification = str(args.uncertainty_quantification)
    uncertainty_measure = str(args.uncertainty_measure)

    if version == "V1":
        find_best_slices_func = find_best_slices_V1
    elif version == "V2":
        find_best_slices_func = find_best_slices_V2
    elif version == "V3":
        find_best_slices_func = find_best_slices_V3
    elif version == "V4":
        find_best_slices_func = find_best_slices_V4
    elif version == "V5":
        find_best_slices_func = find_best_slices_V5
    elif version == "V6":
        find_best_slices_func = find_best_slices_V6
    elif version == "V7":
        find_best_slices_func = find_best_slices_V7
    elif version == "V8":
        find_best_slices_func = find_best_slices_V8
    elif version == "BV1":
        find_best_slices_func = find_best_slices_baseline_V1
    elif version == "BV2":
        find_best_slices_func = find_best_slices_baseline_V2
    elif version == "BV3":
        find_best_slices_func = find_best_slices_baseline_V3
    else:
        raise RuntimeError("find_best_slices_func unknown")

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
    method = args.method
    task_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/" + task + "/"
    base_path = task_path + "refinement_" + set + "/"
    image_path = base_path + "/images/"
    prediction_path = base_path + "/basic_predictions/"
    uncertainty_path = base_path + "/uncertainties/" + uncertainty_quantification + "/" + uncertainty_measure + "/"
    gt_path = base_path + "/labels/"
    recommended_masks_path = base_path + "/recommended_masks/" + version + "/" + method + "/"
    choosen_slices_export_path = base_path + "/choosen_slices_export/" + version + "/" + method + "/"
    refined_prediction_save_path = base_path + "/refined_predictions/" + method
    grid_search_save_path = base_path + "/GridSearchResults/"
    test_set_save_path = base_path + "/eval_results/raw/"
    refinement_inference_tmp = base_path + "/refinement_inference_tmp/part"
    modality = int(args.modality)
    reuse = args.reuse

    pool = mp.Pool(processes=8)  # 8
    if not reuse:
        shutil.rmtree(recommended_masks_path, ignore_errors=True)
        Path(recommended_masks_path).mkdir(parents=True, exist_ok=True)
    shutil.rmtree(refined_prediction_save_path, ignore_errors=True)
    Path(refined_prediction_save_path).mkdir(parents=True, exist_ok=True)
    shutil.rmtree(choosen_slices_export_path, ignore_errors=True)
    Path(choosen_slices_export_path).mkdir(parents=True, exist_ok=True)

    with open(task_path + "dataset.json") as f:
        class_labels = json.load(f)
    class_labels = np.asarray(list(class_labels["labels"].keys())).astype(int)

    # slice_gap = [20]  # [20, 25]
    # num_slices = [12]
    # grid_search(grid_search_save_path, version, slice_gap, num_slices, 1280, devices, args.parallel)

    if args.apply == "eval":

        default_params = {}
        default_params["slice_gap"] = 20  # 20
        default_params["num_slices"] = 12
        default_params["max_slices_based_on_infected_slices"] = 0.28  # 0.5, 0.23
        default_params["min_uncertainty"] = 0.0  # 0.0, 0.15
        default_params["default_size"] = 1280

        params = {}
        # params["slice_gap"] = [10, 15, 20, 25, 30, 40, 50, 70, 80, 90, 100, 110, 120, 130]
        # params["num_slices"] = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        # params["max_slices_based_on_infected_slices"] = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        params["min_uncertainty"] = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        params_list = [params]

        # params_list = [{"slice_gap": [10, 15, 20, 25, 30, 40, 50, 70, 80, 90, 100, 110, 120, 130]},
        #                {"num_slices": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]},
        #                {"max_slices_based_on_infected_slices": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]},
        #                {"min_uncertainty": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]}]

        for params in params_list:
            eval_all_hyperparameters(grid_search_save_path, version, method, default_params, params, devices, args.parallel)

    elif args.apply == "infer":
        test_set_params = {}
        test_set_params["slice_gap"] = 30  # 20
        test_set_params["num_slices"] = 12  # 12
        test_set_params["max_slices_based_on_infected_slices"] = 0.23
        test_set_params["min_uncertainty"] = 0.10
        test_set_params["default_size"] = 1280

        eval_test_set(test_set_save_path, version, method, test_set_params, args.parallel)
    else:
        raise RuntimeError("Apply unknown.")

    pool.close()
    pool.join()
    print("Test")
