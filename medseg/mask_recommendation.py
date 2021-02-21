from medseg import utils
import numpy as np
from tqdm import tqdm
import copy
import os


def recommend_slices(prediction_path, uncertainty_path, gt_path, save_path, find_best_slices_func, num_slices, slice_gap, default_size):
    prediction_filenames = utils.load_filenames(prediction_path)
    uncertainty_filenames = utils.load_filenames(uncertainty_path)
    gt_filenames = utils.load_filenames(gt_path)

    for i in tqdm(range(len(uncertainty_filenames))):
        uncertainty, affine, spacing, header = utils.load_nifty(uncertainty_filenames[i])
        gt, _, _, _ = utils.load_nifty(gt_filenames[i])
        adapted_slice_gap = adapt_slice_gap(uncertainty, slice_gap, default_size)
        indices_dim_0, indices_dim_1, indices_dim_2 = find_best_slices_func(uncertainty, num_slices, adapted_slice_gap)
        recommended_slices = len(indices_dim_0) + len(indices_dim_1) + len(indices_dim_2)
        gt_slices = comp_gt_slices(gt)
        print("{} recommended slices: {}, gt slices: {}, ratio: {}".format(os.path.basename(uncertainty_filenames[i]), recommended_slices, gt_slices, recommended_slices / gt_slices))
        filtered_mask = filter_mask(gt, indices_dim_0, indices_dim_1, indices_dim_2)
        utils.save_nifty(save_path + os.path.basename(uncertainty_filenames[i]), filtered_mask, affine, spacing, header, is_mask=True)


def adapt_slice_gap(mask, slice_gap, default_size):
    return int((mask.shape[0] / default_size) * slice_gap)


def find_best_slices_V1(uncertainty, num_slices, slice_gap):
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


def find_best_slices_V2(uncertainty, num_slices, slice_gap):
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
        slices[index, :, :] = 1

    for index in indices_dim_1:
        slices[:, index, :] = 1

    for index in indices_dim_2:
        slices[:, :, index] = 1

    filtered_mask = copy.deepcopy(mask)
    filtered_mask = np.logical_and(filtered_mask, slices)

    return filtered_mask

def comp_gt_slices(gt):
    gt_slices = np.sum(gt, axis=(0, 1))
    gt_slices = np.count_nonzero(gt_slices)
    return gt_slices


if __name__ == '__main__':
    version = "V2"

    if version == "V1":
        find_best_slices_func = find_best_slices_V1
    elif version == "V2":
        find_best_slices_func = find_best_slices_V2
    else:
        raise RuntimeError("find_best_slices_func unknown")

    prediction_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task076_frankfurt3Guided/predictions_with_ensemble/"
    uncertainty_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task076_frankfurt3Guided/uncertainties_with_ensemble/ggo/"
    gt_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task076_frankfurt3Guided/labelsTs/"
    save_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task076_frankfurt3Guided/recommended_guiding_masks/" + version + "/"

    slice_gap = 75
    default_size = 1280
    num_slices = 10

    recommend_slices(prediction_path, uncertainty_path, gt_path, save_path, find_best_slices_func, num_slices, slice_gap, default_size)
