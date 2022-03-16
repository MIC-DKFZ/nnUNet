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


def combine_gt(gt_path, chosen_slices_path, save_path, depth):
    shutil.rmtree(save_path, ignore_errors=True)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    gt_filenames = utils.load_filenames(gt_path)

    for gt_filename in tqdm(gt_filenames):
        name = os.path.basename(gt_filename)[:-7]
        gt, affine, spacing, header = utils.load_nifty(gt_filename)

        with open(chosen_slices_path + name + ".json") as f:
            chosen_slices = json.load(f)

        comined_slices_prediction_dim0 = combine_slices_gt(copy.deepcopy(gt), chosen_slices["Sagittal"], 0, depth)
        comined_slices_prediction_dim1 = combine_slices_gt(copy.deepcopy(gt), chosen_slices["Coronal"], 1, depth)
        comined_slices_prediction_dim2 = combine_slices_gt(copy.deepcopy(gt), chosen_slices["Axial"], 2, depth)

        utils.save_nifty(save_path + name + "_sag_cor.nii.gz", comined_slices_prediction_dim0, affine, spacing, header, is_mask=True)

        utils.save_nifty(save_path + name + "_cor_cor.nii.gz", comined_slices_prediction_dim1, affine, spacing, header, is_mask=True)

        utils.save_nifty(save_path + name + "_axial_cor.nii.gz", comined_slices_prediction_dim2, affine, spacing, header, is_mask=True)


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


def combine_slices_gt(prediction, indices, dim, depth):
    prediction_min_value = prediction.min()
    prediction_max_value = prediction.max()
    indices_with_depth, original_indices_mask = [], []
    for index in indices:
        index_with_depth = list(range(index - depth, index + depth + 1))
        original_index_mask = ([0] * (depth*2+1))
        original_index_mask[depth] = 1
        indices_with_depth.extend(index_with_depth)
        original_indices_mask.extend(original_index_mask)

    slices_prediction = np.moveaxis(prediction, dim, 0)[indices_with_depth, :, :]

    for i in range(len(original_indices_mask)):
        if original_indices_mask[i] == 1:
            slices_prediction[i] = add_checkerboard_marker(slices_prediction[i], prediction_min_value, prediction_max_value) * 2  # "*2" to change the color of the segmentation for that slice

    slices_prediction = np.moveaxis(slices_prediction, 0, dim)

    return slices_prediction


def add_checkerboard_marker(slice, min_value, max_value, size=8):
    cell_size = (np.asarray(slice.shape) * 0.01).astype(int)

    for i in range(0, size, 2):
        slice[i * cell_size[0]:(i + 1) * cell_size[0], i * cell_size[1]:(i + 1) * cell_size[1]] = min_value
        slice[(i + 1) * cell_size[0]:(i + 2) * cell_size[0], (i + 1) * cell_size[1]:(i + 2) * cell_size[1]] = max_value

    return slice


def extract_corrected_slices(correction_path, save_path, depth, size=8):
    shutil.rmtree(save_path, ignore_errors=True)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    correction_filenames = utils.load_filenames(correction_path)

    for correction_filename in tqdm(correction_filenames):
        name = os.path.basename(correction_filename)[:-7]
        correction, affine, spacing, header = utils.load_nifty(correction_filename)
        # print("name: {}, shape: {}".format(name, correction.shape))

        # if "_axial_" in name:
        #     correction = correction.transpose(2, 1, 0)  # 2, 0, 1
        # elif "_cor_" in name:
        #     correction = correction.transpose(1, 2, 0)  # 1, 0, 2

        if "_axial_" in name:
            index = 2
        elif "_cor_" in name:
            index = 1
        else:
            index = 0

        print("name: {}, shape: {}".format(name, correction.shape))

        corrected_slices = []
        next_slice_index = depth
        for i in range(correction.shape[index]):
            if index == 0:
                slice = correction[i, :, :]
            elif index == 1:
                slice = correction[:, i, :]
            else:
                slice = correction[:, :, i]
            if i == next_slice_index:
                cell_size = (np.asarray(slice.shape) * 0.01).astype(int)
                for i in range(0, size, 2):
                    slice[i * cell_size[0]:(i + 1) * cell_size[0], i * cell_size[1]:(i + 1) * cell_size[1]] = 0
                    slice[(i + 1) * cell_size[0]:(i + 2) * cell_size[0], (i + 1) * cell_size[1]:(i + 2) * cell_size[1]] = 0
                next_slice_index += depth*2+1
                # corrected_slices.append(np.rot90(np.flip(slice), k=1))
                corrected_slices.append(slice)
        corrected_slices = np.asarray(corrected_slices)
        if index == 1:
            corrected_slices = corrected_slices.transpose(1, 0, 2)
        elif index == 2:
            corrected_slices = corrected_slices.transpose(1, 2, 0)
        corrected_slices[corrected_slices > 0] = 1
        utils.save_nifty(save_path + name + "_corrected_slices.nii.gz", corrected_slices, affine, spacing, header, is_mask=True)


def corrected_slices2recommended_mask(corrected_slices_path, gt_dir, metadata_dir, recommended_masks_path):
    shutil.rmtree(recommended_masks_path, ignore_errors=True)
    Path(recommended_masks_path).mkdir(parents=True, exist_ok=True)

    corrected_slices_filenames = utils.load_filenames(corrected_slices_path)
    corrected_slices_filenames = np.reshape(corrected_slices_filenames, (-1, 3))

    for slice_set_filenames in tqdm(corrected_slices_filenames):
        name = os.path.basename(slice_set_filenames[0])[:4]
        sagittal = utils.load_nifty(slice_set_filenames[2])[0].astype(np.uint8)
        coronal = utils.load_nifty(slice_set_filenames[1])[0].astype(np.uint8)
        axial = utils.load_nifty(slice_set_filenames[0])[0].astype(np.uint8)
        with open(metadata_dir + name + ".json") as f:
            metadata = json.load(f)
        gt, affine, spacing, header = utils.load_nifty(gt_dir + name + ".nii.gz")
        mask = np.zeros(gt.shape, dtype=np.uint8)
        for i, index in enumerate(metadata["Sagittal"]):
            # mask[index, :, :][sagittal[i, :, :]] = 1
            mask[index, :, :] = mask[index, :, :] | sagittal[i, :, :]
        for i, index in enumerate(metadata["Coronal"]):
            # mask[:, index, :][coronal[:, i, :]] = 1
            mask[:, index, :] = mask[:, index, :] | coronal[:, i, :]
        for i, index in enumerate(metadata["Axial"]):
            # mask[:, :, index][axial[:, :, i]] = 1
            mask[:, :, index] = mask[:, :, index] | axial[:, :, i]
        utils.save_nifty(recommended_masks_path + name + "_0001.nii.gz", mask, affine, spacing, header, is_mask=True)


def binarize(load_dir):
    mask_filenames = utils.load_filenames(load_dir)
    for mask_filename in tqdm(mask_filenames):
        mask, affine, spacing, header = utils.load_nifty(mask_filename)
        mask[mask > 0] = 1
        utils.save_nifty(mask_filename, mask, affine, spacing, header, is_mask=True)


if __name__ == '__main__':
    depth = 5
    image_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task070_guided_all_public_ggo/refinement_test/images/"
    prediction_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task070_guided_all_public_ggo/refinement_test/basic_predictions/"
    gt_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task070_guided_all_public_ggo/refinement_test/labels/"
    chosen_slices_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task070_guided_all_public_ggo/refinement_test/choosen_slices_export/V7/my_method/"
    save_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task070_guided_all_public_ggo/refinement_test/combined_slices_depth{}/".format(depth)
    gt_slice_save_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task070_guided_all_public_ggo/refinement_test/combined_slices_gt2/"
    correction_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task070_guided_all_public_ggo/refinement_test/corrected_slices/"
    corrected_slices_save_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task070_guided_all_public_ggo/refinement_test/extracted_corrected_slices/"
    corrected_gt_slices_save_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task070_guided_all_public_ggo/refinement_test/extracted_gt_slices2/"
    recommended_masks_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task070_guided_all_public_ggo/refinement_test/isabel_recommended_masks/"

    # combine(image_path, prediction_path, chosen_slices_path, save_path, depth=depth)
    # extract_corrected_slices(correction_path, corrected_slices_save_path, depth)

    # combine_gt(gt_path, chosen_slices_path, gt_slice_save_path, depth=depth)
    # extract_corrected_slices(gt_slice_save_path, corrected_gt_slices_save_path, depth)

    corrected_slices2recommended_mask(corrected_slices_save_path, gt_path, chosen_slices_path, recommended_masks_path)

    # binarize("/home/k539i/Documents/datasets/preprocessed/nnUNet/nnUNet_raw_data/nnUNet_raw_data/Task070_guided_all_public_ggo/refinement_test/recommended_masks/V7/my_method/")