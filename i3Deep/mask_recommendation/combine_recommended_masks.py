from i3Deep import utils
import numpy as np
from tqdm import tqdm
import os
import json
import copy
import shutil
from pathlib import Path


def combine(image_path, prediction_path, chosen_slices_path, save_path, depth):
    shutil.rmtree(save_path, ignore_errors=True)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    image_filenames = utils.load_filenames(image_path)

    for image_filename in tqdm(image_filenames):
        name = os.path.basename(image_filename)[:-12]
        image, affine, spacing, header = utils.load_nifty(image_filename)
        prediction, _, _, _ = utils.load_nifty(prediction_path + name + ".nii.gz")

        with open(chosen_slices_path + name + ".json") as f:
            chosen_slices = json.load(f)

        comined_slices_image_dim0, comined_slices_prediction_dim0 = combine_slices(copy.deepcopy(image), copy.deepcopy(prediction), chosen_slices["Sagittal"], 0, depth)
        comined_slices_image_dim1, comined_slices_prediction_dim1 = combine_slices(copy.deepcopy(image), copy.deepcopy(prediction), chosen_slices["Coronal"], 1, depth)
        comined_slices_image_dim2, comined_slices_prediction_dim2 = combine_slices(copy.deepcopy(image), copy.deepcopy(prediction), chosen_slices["Axial"], 2, depth)

        utils.save_nifty(save_path + name + "_sagittal_image.nii.gz", comined_slices_image_dim0, affine, spacing, header)
        utils.save_nifty(save_path + name + "_sagittal_presegmentation.nii.gz", comined_slices_prediction_dim0, affine, spacing, header, is_mask=True)

        utils.save_nifty(save_path + name + "_coronal_image.nii.gz", comined_slices_image_dim1, affine, spacing, header)
        utils.save_nifty(save_path + name + "_coronal_presegmentation.nii.gz", comined_slices_prediction_dim1, affine, spacing, header, is_mask=True)

        utils.save_nifty(save_path + name + "_axial_image.nii.gz", comined_slices_image_dim2, affine, spacing, header)
        utils.save_nifty(save_path + name + "_axial_presegmentation.nii.gz", comined_slices_prediction_dim2, affine, spacing, header, is_mask=True)


def combine_slices(image, prediction, indices, dim, depth):
    # np.moveaxis(uncertainty, axis, 0)[index, :, :]
    image_min_value = image.min()
    image_max_value = image.max()
    prediction_min_value = prediction.min()
    prediction_max_value = prediction.max()
    indices_with_depth, original_indices_mask = [], []
    for index in indices:
        index_with_depth = list(range(index - depth, index + depth + 1))
        original_index_mask = ([0] * (depth*2+1))
        original_index_mask[depth] = 1
        indices_with_depth.extend(index_with_depth)
        original_indices_mask.extend(original_index_mask)

    slices_image = np.moveaxis(image, dim, 0)[indices_with_depth, :, :]
    slices_prediction = np.moveaxis(prediction, dim, 0)[indices_with_depth, :, :]

    for i in range(len(original_indices_mask)):
        if original_indices_mask[i] == 1:
            slices_image[i] = add_checkerboard_marker(slices_image[i], image_min_value, image_max_value)
            slices_prediction[i] = add_checkerboard_marker(slices_prediction[i], prediction_min_value, prediction_max_value) * 2  # "*2" to change the color of the segmentation for that slice

    slices_image = np.moveaxis(slices_image, 0, dim)
    slices_prediction = np.moveaxis(slices_prediction, 0, dim)

    return slices_image, slices_prediction


def add_checkerboard_marker(slice, min_value, max_value, size=8):
    cell_size = (np.asarray(slice.shape) * 0.01).astype(int)

    for i in range(0, size, 2):
        slice[i * cell_size[0]:(i + 1) * cell_size[0], i * cell_size[1]:(i + 1) * cell_size[1]] = min_value
        slice[(i + 1) * cell_size[0]:(i + 2) * cell_size[0], (i + 1) * cell_size[1]:(i + 2) * cell_size[1]] = max_value

    return slice


if __name__ == '__main__':
    depth = 5
    image_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task070_guided_all_public_ggo/refinement_test/images/"
    prediction_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task070_guided_all_public_ggo/refinement_test/basic_predictions/"
    chosen_slices_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task070_guided_all_public_ggo/refinement_test/choosen_slices_export/V7/my_method/"
    save_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task070_guided_all_public_ggo/refinement_test/combined_slices_depth{}/".format(depth)

    combine(image_path, prediction_path, chosen_slices_path, save_path, depth=depth)
