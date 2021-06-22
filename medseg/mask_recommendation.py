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


def recommend_slices_parallel(image_path, prediction_path, uncertainty_path, gt_path, save_path, find_best_slices_func, num_slices, slice_gap, default_size, deepigeos=0):
    image_filenames = utils.load_filenames(image_path)
    prediction_filenames = utils.load_filenames(prediction_path)
    uncertainty_filenames = utils.load_filenames(uncertainty_path)
    gt_filenames = utils.load_filenames(gt_path)
    pool = mp.Pool(processes=8)

    start_time = time.time()
    results = pool.map(partial(recommend_slices_single_case,
                                                   image_filenames=image_filenames, prediction_filenames=prediction_filenames, uncertainty_filenames=uncertainty_filenames, gt_filenames=gt_filenames,
                                                   save_path=save_path, find_best_slices_func=find_best_slices_func, num_slices=num_slices, slice_gap=slice_gap, default_size=default_size, deepigeos=deepigeos),
                                           range(len(uncertainty_filenames)))
    print("Recommend slices elapsed time: ", time.time() - start_time)
    results = np.asarray(results)
    total_recommended_slices = results[:, 0]
    total_gt_slices = results[:, 1]
    total_recommended_slices = np.sum(total_recommended_slices)
    total_gt_slices = np.sum(total_gt_slices)

    total_ratio = total_recommended_slices / total_gt_slices
    print("total recommended slices: {}, total gt slices: {}, total ratio: {}".format(total_recommended_slices, total_gt_slices, total_ratio))
    return total_ratio


def recommend_slices_single_case(i, image_filenames, prediction_filenames, uncertainty_filenames, gt_filenames, save_path, find_best_slices_func, num_slices, slice_gap, default_size, deepigeos=0):
    uncertainty, affine, spacing, header = utils.load_nifty(uncertainty_filenames[i])
    prediction, _, _, _ = utils.load_nifty(prediction_filenames[i])
    gt, _, _, _ = utils.load_nifty(gt_filenames[i])
    if deepigeos != 0:
        image, _, _, _ = utils.load_nifty(image_filenames[i])
    adapted_slice_gap = adapt_slice_gap(uncertainty, slice_gap, default_size)
    # indices_dim_0: Sagittal
    # indices_dim_1: Coronal
    # indices_dim_2: Axial
    indices_dim_0, indices_dim_1, indices_dim_2 = find_best_slices_func(prediction, uncertainty, num_slices, adapted_slice_gap)
    recommended_slices = len(indices_dim_0) + len(indices_dim_1) + len(indices_dim_2)
    gt_slices = comp_infected_slices(gt)
    prediction_slices = comp_infected_slices(prediction)
    print("name: {} recommended slices: {}, gt slices: {}, prediction slices: {}, ratio: {}".format(os.path.basename(uncertainty_filenames[i]), recommended_slices, gt_slices, prediction_slices, recommended_slices / gt_slices))
    # total_recommended_slices += recommended_slices
    # total_gt_slices += gt_slices
    # print("{} recommended slices: {}, gt slices: {}, ratio: {}".format(os.path.basename(uncertainty_filenames[i]), recommended_slices, gt_slices, recommended_slices / gt_slices))
    # print("indices_dim_0: {}, indices_dim_1: {}, indices_dim_2: {}".format(indices_dim_0, indices_dim_1, indices_dim_2))
    filtered_mask = filter_mask(gt, indices_dim_0, indices_dim_1, indices_dim_2)
    if deepigeos != 0:
        if deepigeos == 1:
            geodisc_lambda = 0.99
        else:
            geodisc_lambda = 0.0
        geodisc_map = filtered_mask.astype(np.uint8)
        geodisc_map[geodisc_map < 0] = 0
        filtered_mask = GeodisTK.geodesic3d_raster_scan(image.astype(np.float32), geodisc_map, spacing.astype(np.float32), geodisc_lambda, 1)
        # filtered_mask[geodisc_map == 1] = 0
        utils.save_nifty(save_path + os.path.basename(uncertainty_filenames[i])[:-7] + "_0001.nii.gz", filtered_mask, affine, spacing, header, is_mask=False)
    else:
        utils.save_nifty(save_path + os.path.basename(uncertainty_filenames[i])[:-7] + "_0001.nii.gz", filtered_mask, affine, spacing, header, is_mask=True)
    return recommended_slices, gt_slices


def adapt_slice_gap(mask, slice_gap, default_size):
    return int((mask.shape[0] / default_size) * slice_gap)


def find_best_slices_baseline_V1(prediction, uncertainty, num_slices, slice_gap):
    "Find random slices in every dimension without a min slice gap."
    indices_dim_0 = random.sample(range(uncertainty.shape[0]), num_slices)
    indices_dim_1 = random.sample(range(uncertainty.shape[1]), num_slices)
    indices_dim_2 = random.sample(range(uncertainty.shape[2]), num_slices)
    return indices_dim_0, indices_dim_1, indices_dim_2


def find_best_slices_baseline_V2(prediction, uncertainty, num_slices, slice_gap):
    "Find random slices in every dimension within equidistant distances."
    def get_indices(axis):
        slice_sectors = np.linspace(0, uncertainty.shape[axis], num_slices, endpoint=True).astype(int)
        indices = [random.randint(slice_sectors[i-1], slice_sectors[i]) for i in range(1, len(slice_sectors)-1)]
        return indices

    indices_dim_0 = get_indices(0)
    indices_dim_1 = get_indices(1)
    indices_dim_2 = get_indices(2)
    return indices_dim_0, indices_dim_1, indices_dim_2


def find_best_slices_V1(prediction, uncertainty, num_slices, slice_gap):
    "Find best slices based on maximum 2D plane uncertainty in every dimension without a min slice gap."
    uncertainty_dim_0 = np.sum(-1*uncertainty, axis=(1, 2))
    uncertainty_dim_1 = np.sum(-1*uncertainty, axis=(0, 2))
    uncertainty_dim_2 = np.sum(-1*uncertainty, axis=(0, 1))
    indices_dim_0 = np.argsort(uncertainty_dim_0)[:num_slices]
    indices_dim_1 = np.argsort(uncertainty_dim_1)[:num_slices]
    indices_dim_2 = np.argsort(uncertainty_dim_2)[:num_slices]
    # dim_0 = uncertainty_dim_0[indices_dim_0]*-1
    # dim_1 = uncertainty_dim_1[indices_dim_1]*-1
    # dim_2 = uncertainty_dim_2[indices_dim_2]*-1
    return indices_dim_0, indices_dim_1, indices_dim_2


def find_best_slices_V2(prediction, uncertainty, num_slices, slice_gap):
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
    return indices_dim_0, indices_dim_1, indices_dim_2


def find_best_slices_V3(prediction, uncertainty, num_slices, slice_gap):
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
    return indices_dim_0, indices_dim_1, indices_dim_2


def find_best_slices_V4(prediction, uncertainty, num_slices, slice_gap):
    indices_dim_0_V2, indices_dim_1_V2, indices_dim_2_V2 = find_best_slices_V2(prediction, uncertainty, num_slices, slice_gap)
    indices_dim_0_V3, indices_dim_1_V3, indices_dim_2_V3 = find_best_slices_V3(prediction, uncertainty, num_slices, slice_gap)
    indices_dim_0 = np.concatenate((indices_dim_0_V2, indices_dim_0_V3), axis=0)
    indices_dim_1 = np.concatenate((indices_dim_1_V2, indices_dim_1_V3), axis=0)
    indices_dim_2 = np.concatenate((indices_dim_2_V2, indices_dim_2_V3), axis=0)
    return indices_dim_0, indices_dim_1, indices_dim_2


def find_best_slices_V5(prediction, uncertainty, num_slices, slice_gap):
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
    return indices_dim_0, indices_dim_1, indices_dim_2


def find_best_slices_V6(prediction, uncertainty, num_slices, slice_gap):
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
    return indices_dim_0, indices_dim_1, indices_dim_2


def find_best_slices_V7(prediction, uncertainty, num_slices, slice_gap, min_uncertainty=0.15, max_slices_based_on_infected_slices=0.20):
    "Like V2, but filters out all slices with less than 40% of summed uncertainty than that of the max slice"
    uncertainty_dim_0 = np.sum(-1*uncertainty, axis=(1, 2))
    uncertainty_dim_1 = np.sum(-1*uncertainty, axis=(0, 2))
    uncertainty_dim_2 = np.sum(-1*uncertainty, axis=(0, 1))
    indices_dim_0 = np.argsort(uncertainty_dim_0)
    indices_dim_1 = np.argsort(uncertainty_dim_1)
    indices_dim_2 = np.argsort(uncertainty_dim_2)
    indices_dim_0 = filter_indices(indices_dim_0, num_slices, slice_gap)
    indices_dim_1 = filter_indices(indices_dim_1, num_slices, slice_gap)
    indices_dim_2 = filter_indices(indices_dim_2, num_slices, slice_gap)
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
    if num_infected_slices == 0:
        num_infected_slices = 1
    indices_dim_0 = indices_dim_0[:num_infected_slices]
    indices_dim_1 = indices_dim_1[:num_infected_slices]
    indices_dim_2 = indices_dim_2[:num_infected_slices]

    return indices_dim_0, indices_dim_1, indices_dim_2


def find_best_slices_V8(prediction, uncertainty, num_slices, slice_gap):
    "Idee: V2, aber nur die slices nehmen die noch min 60% so viel uncertainty haben wie die erste slice + für jede slice den Bereich eingrenzen der 70% der Uncertainty umfasst"
    "Neue metriken dafür machen: selektierte anzahl pixel / gesamt anzahl pixel ; selektierte anzahl infected pixel / gesamt anzahl infected pixel ; Absolute anzahl von selektieren patches"
    pass


def find_best_slices_V9(prediction, uncertainty, num_slices, slice_gap):
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
    while index < len(indices) and index < num_slices:
        tmp = indices[:index]
        if np.min(np.abs(tmp - indices[index])) <= slice_gap:  # np.abs(indices[index] - indices[index-1]) <= slice_gap
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

    filtered_mask = copy.deepcopy(mask)
    filtered_mask = np.logical_and(filtered_mask, slices)

    return filtered_mask


def comp_infected_slices(mask):
    mask_slices = np.sum(mask, axis=(0, 1))
    mask_slices = np.count_nonzero(mask_slices)
    return mask_slices


def copy_masks_for_inference(load_dir):
    filenames = utils.load_filenames(load_dir)
    quarter = int(len(filenames) / 4)
    filenames0 = filenames[:quarter]
    filenames1 = filenames[quarter:quarter*2]
    filenames2 = filenames[quarter*2:quarter*3]
    filenames3 = filenames[quarter*3:]
    save_dir0 = refinement_inference_tmp + "0/"
    save_dir1 = refinement_inference_tmp + "1/"
    save_dir2 = refinement_inference_tmp + "2/"
    save_dir3 = refinement_inference_tmp + "3/"

    for filename in filenames0:
        copyfile(filename, save_dir0 + os.path.basename(filename))
    for filename in filenames1:
        copyfile(filename, save_dir1 + os.path.basename(filename))
    for filename in filenames2:
        copyfile(filename, save_dir2 + os.path.basename(filename))
    for filename in filenames3:
        copyfile(filename, save_dir3 + os.path.basename(filename))

def inference(available_devices, gt_path):
    start_time = time.time()
    filenames = utils.load_filenames(refined_prediction_save_path, extensions=None)
    print("load_filenames: ", time.time() - start_time)
    start_time = time.time()
    for filename in filenames:
        os.remove(filename)
    parts_to_process = [0, 1, 2, 3]
    waiting = []
    finished = []
    wait_time = 5
    start_inference_time = time.time()

    print("remove: ", time.time() - start_time)
    print("Starting inference...")
    while parts_to_process:
        if available_devices:
            device = available_devices[0]
            available_devices = available_devices[1:]
            part = parts_to_process[0]
            parts_to_process = parts_to_process[1:]
            print("Processing part {} on device {}...".format(part, device))
            command = 'nnUNet_predict -i ' + str(refinement_inference_tmp) + str(
                part) + ' -o ' + str(refined_prediction_save_path) + ' -tr nnUNetTrainerV2Guided3 -t ' + model + ' -m 3d_fullres -f 0 -d ' + str(
                device) + ' -chk model_best --disable_tta --num_threads_preprocessing 1 --num_threads_nifti_save 1'
            p = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, preexec_fn=os.setsid)
            waiting.append([part, device, p, time.time()])
        else:
            for w in waiting:
                if w[2].poll() is not None:
                    print("Finished part {} on device {} after {}s.".format(w[0], w[1], time.time() - w[3]))
                    available_devices.append(w[1])
                    finished.append(w[0])
                    waiting.remove(w)
                    break
            time.sleep(wait_time)
    print("All parts are being processed.")

    def check_all_predictions_exist():
        filenames = utils.load_filenames(refined_prediction_save_path)
        nr_predictions = len(utils.load_filenames(prediction_path))
        counter = 0
        for filename in filenames:
            if ".nii.gz" in filename:
                counter += 1
        return bool(counter == nr_predictions)

    while waiting and len(finished) < 4 and not check_all_predictions_exist():
        time.sleep(wait_time)
    print("All predictions finished.")
    time.sleep(30)
    print("Cleaning up threads")
    # [os.killpg(os.getpgid(p.pid), signal.SIGTERM) for p in finished]
    [os.killpg(os.getpgid(p[2].pid), signal.SIGTERM) for p in waiting]
    os.remove(refined_prediction_save_path + "/plans.pkl")
    print("Total inference time {}s.".format(time.time() - start_inference_time))
    print("All parts finished processing.")
    mean_dice_score, median_dice_score = evaluate(gt_path, refined_prediction_save_path, (0, 1))
    return mean_dice_score, median_dice_score


def grid_search(save_dir, version, slice_gap_list, num_slices_list, default_size, devices, parallel, deepigeos=0):
    results = []
    if os.path.isfile(save_dir + "grid_search_results_" + version + ".pkl"):
        with open(save_dir + "grid_search_results_" + version + ".pkl", 'rb') as handle:
            results = pickle.load(handle)
    print(results)
    for slice_gap in slice_gap_list:
        for num_slices in num_slices_list:
            print("slice_gap: {}, default_size: {}, num_slices: {}".format(slice_gap, default_size, num_slices))
            if not parallel:
                # total_ratio = recommend_slices(image_path, prediction_path, uncertainty_path, gt_path, save_path, find_best_slices_func, num_slices, slice_gap, default_size)
                pass
            else:
                total_ratio = recommend_slices_parallel(image_path, prediction_path, uncertainty_path, gt_path, save_path, find_best_slices_func, num_slices, slice_gap, default_size, deepigeos)
            start_time = time.time()
            copy_masks_for_inference(save_path)
            print("copy_masks_for_inference: ", time.time() - start_time)
            mean_dice_score, median_dice_score = inference(devices, gt_path)
            results.append({"slice_gap": slice_gap, "num_slices": num_slices, "total_ratio": total_ratio, "mean_dice_score": mean_dice_score, "median_dice_score": median_dice_score})
            with open(save_dir + "grid_search_results_" + version + ".pkl", 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(results)

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
    parser.add_argument("-dig", "--deepigeos", help="Set DeepIGeos ID", required=True)
    args = parser.parse_args()
    devices = [0, 5, 6, 7]

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
    elif version == "BV1":
        find_best_slices_func = find_best_slices_baseline_V1
    elif version == "BV2":
        find_best_slices_func = find_best_slices_baseline_V2
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
    base_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/" + task + "/refinement_" + set + "/"
    image_path = base_path + "/images/"
    prediction_path = base_path + "/basic_predictions/"
    uncertainty_path = base_path + "/uncertainties/" + uncertainty_quantification + "/" + uncertainty_measure + "/"
    gt_path = base_path + "/labels/"
    save_path = base_path + "/recommended_masks/" + version + "/"
    refined_prediction_save_path = base_path + "/refined_predictions"
    grid_search_save_path = base_path + "/GridSearchResults/"
    refinement_inference_tmp = base_path + "/refinement_inference_tmp/part"
    deepigeos = int(args.deepigeos)

    Path(save_path).mkdir(parents=True, exist_ok=True)

    slice_gap = [20]  # [20, 25]
    num_slices = [12]
    grid_search(grid_search_save_path, version, slice_gap, num_slices, 1280, devices, args.parallel, deepigeos)

    # pkl2csv("/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task072_allGuided_ggo/GridSearchResults/grid_search_results_V6.pkl")
