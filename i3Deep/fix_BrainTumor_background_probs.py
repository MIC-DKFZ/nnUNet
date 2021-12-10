from i3Deep import utils
import numpy as np
import os
from tqdm import tqdm
import argparse
import sys


def filter(filenames):
    filtered_filenames = []
    for filename in filenames:
        if os.path.basename(filename)[5] == "0":
            filtered_filenames.append(filename)
    filtered_filenames = np.asarray(filtered_filenames)
    return filtered_filenames


def fix_backgrounds(filenames, save_path, backup_path):
    for filename in tqdm(filenames):
        image, affine, spacing, header = utils.load_nifty(filename)
        utils.save_nifty(backup_path + os.path.basename(filename), image, affine, spacing, header)
        image = fix_background(image)
        utils.save_nifty(save_path + os.path.basename(filename), image, affine, spacing, header)


def fix_background(image):
    fixed_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        if np.sum(image[i, :, :]) == 0:
            fixed_image[i, :, :] = 1
    for i in range(image.shape[1]):
        if np.sum(image[:, i, :]) == 0:
            fixed_image[:, i, :] = 1
    for i in range(image.shape[2]):
        if np.sum(image[:, :, i]) == 0:
            fixed_image[:, :, i] = 1
    image += fixed_image
    return image


if __name__ == '__main__':
    filepath = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task002_BrainTumour_guided/refinement_val/uncertainties/ensemble/probabilities/"
    save_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task002_BrainTumour_guided/refinement_val/uncertainties/ensemble/probabilities_fixed/"
    backup_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task002_BrainTumour_guided/refinement_val/uncertainties/ensemble/probabilities_backup/"

    filenames = utils.load_filenames(filepath)
    filenames = filter(filenames)
    fix_backgrounds(filenames, save_path, backup_path)