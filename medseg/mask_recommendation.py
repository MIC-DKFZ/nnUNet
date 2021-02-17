from medseg import utils
import numpy as np
from tqdm import tqdm
import copy
import os


def recommend_slices(prediction_path, uncertainty_path, gt_path, save_path):
    prediction_filenames = utils.load_filenames(prediction_path)
    uncertainty_filenames = utils.load_filenames(uncertainty_path)
    gt_filenames = utils.load_filenames(gt_path)

    for i in tqdm(range(len(uncertainty_filenames))):
        uncertainty, affine, spacing, header = utils.load_nifty(uncertainty_filenames[i])
        gt, _, _, _ = utils.load_nifty(gt_filenames[i])
        indices_dim_0, indices_dim_1, indices_dim_2 = find_best_slices_V1(uncertainty)
        filtered_mask = filter_mask(gt, indices_dim_0, indices_dim_1, indices_dim_2)
        utils.save_nifty(save_path + os.path.basename(uncertainty_filenames[i]), filtered_mask, affine, spacing, header, is_mask=True)


def find_best_slices_V1(uncertainty, num_slices=10):
    uncertainty_dim_0 = np.sum(-1*uncertainty, axis=(1, 2))
    uncertainty_dim_1 = np.sum(-1*uncertainty, axis=(0, 2))
    uncertainty_dim_2 = np.sum(-1*uncertainty, axis=(0, 1))
    indices_dim_0 = np.argsort(uncertainty_dim_0)[:num_slices]
    indices_dim_1 = np.argsort(uncertainty_dim_1)[:num_slices]
    indices_dim_2 = np.argsort(uncertainty_dim_2)[:num_slices]
    return indices_dim_0, indices_dim_1, indices_dim_2


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


if __name__ == '__main__':
    prediction_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task076_frankfurt3Guided/predictions_with_ensemble/"
    uncertainty_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task076_frankfurt3Guided/uncertainties_with_ensemble/"
    gt_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task076_frankfurt3Guided/labelsTs/"
    save_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task076_frankfurt3Guided/recommended_guiding_masks/V1/"

    recommend_slices(prediction_path, uncertainty_path, gt_path, save_path)
