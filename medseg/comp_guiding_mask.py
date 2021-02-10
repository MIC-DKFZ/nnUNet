from medseg import utils
import numpy as np
from tqdm import tqdm
import math
import random
import copy
from skimage import measure
import os


def comp_guiding_mask(load_path, save_path, slice_length, default_size):
    filenames = utils.load_filenames(load_path)
    for filename in tqdm(filenames):
        mask, affine, spacing, header = utils.load_nifty(filename)
        adapted_slice_length = adapt_slice_length(mask, slice_length, default_size)
        mask_slices = comp_slices_mask(mask, adapted_slice_length)
        utils.save_nifty(save_path + os.path.basename(filename), mask_slices, affine, spacing, header, is_mask=True)


def adapt_slice_length(mask, slice_length, default_size):
    return int((mask.shape[0] / default_size) * slice_length)


def comp_slices_mask(mask, slice_length, p=None, nnunet=False):
    labels = range(1, int(np.max(mask)) + 1)
    mask_slices = np.zeros_like(mask)
    for label in labels:
        mask_label = copy.deepcopy(mask)
        mask_label[mask_label != label] = 0
        mask_label[mask_label == label] = 1
        objects_slices_label = comp_slices_label(mask_label, slice_length, p, nnunet).astype(int)
        objects_slices_label[objects_slices_label == 1] = label
        mask_slices += objects_slices_label
    if nnunet:
        mask_slices[mask == -1] = -1
    return mask_slices


def comp_slices_label(mask, slice_length, p=None, nnunet=False):
    objects, ids = measure.label(mask, background=False, connectivity=2, return_num=True)
    mask_slices = np.zeros_like(mask)
    for id in range(1, ids + 1):
        object = copy.deepcopy(objects)
        object[object != id] = 0
        object[object == id] = 1
        object_slices = comp_object_slices(object, slice_length, p, nnunet)
        mask_slices = np.logical_or(mask_slices, object_slices)
    return mask_slices


def comp_object_slices(object, slice_length, p=None, nnunet=False):
    object_slices, is_object_small = comp_object_slices_dim(object, 0, slice_length, p)
    if not is_object_small or nnunet:
        object_slices_1, is_object_small = comp_object_slices_dim(object, 1, slice_length, p)
        object_slices_2, is_object_small = comp_object_slices_dim(object, 2, slice_length, p)
        object_slices = np.logical_or(object_slices, object_slices_1)
        object_slices = np.logical_or(object_slices, object_slices_2)
    return object_slices


def comp_object_slices_dim(object, dim, slice_length, p=None):
    dims = [0, 1, 2]
    dims.remove(dim)
    object_flat = np.sum(object, axis=tuple(dims))
    nonzero_indices = np.nonzero(object_flat)
    min_index = np.min(nonzero_indices)
    max_index = np.max(nonzero_indices)
    object_length = max_index - min_index
    slice_count = math.ceil(object_length / slice_length)
    is_object_small = slice_count == 1
    slice_sectors = np.linspace(min_index, max_index, slice_count, endpoint=True).astype(int)
    if len(slice_sectors) == 1:
        slice_sectors = np.asarray([slice_sectors[0], max_index])

    slices = np.zeros_like(object)
    for i in range(1, len(slice_sectors)):
        if p is not None and p < random.uniform(0, 1):
            continue

        if slice_count <= 2:
            slice_index = slice_sectors[i-1] + (slice_sectors[i] - slice_sectors[i-1])/2
        else:
            slice_index = random.randint(slice_sectors[i-1], slice_sectors[i])
        slice_index = int(slice_index)

        slice_width = 3
        for offset in range(slice_width):
            offset = int(((slice_width - 1) / 2) + offset)
            if slice_index + offset < min_index or max_index < slice_index + offset:
                continue
            if dim == 0:
                slices[slice_index + offset, :, :] = 1
            elif dim == 1:
                slices[:, slice_index + offset, :] = 1
            else:
                slices[:, :, slice_index + offset] = 1

    object_slices = copy.deepcopy(object)
    object_slices = np.logical_and(object_slices, slices)

    return object_slices, is_object_small


def add_to_images_or_masks(image_path, guiding_mask_path, save_path, is_mask=False):
    image_filenames = utils.load_filenames(image_path)
    guiding_mask_filenames = utils.load_filenames(guiding_mask_path)
    for i in tqdm(range(len(image_filenames))):
        image, affine, spacing, header = utils.load_nifty(image_filenames[i])
        guiding_mask, _, _, _ = utils.load_nifty(guiding_mask_filenames[i])
        image = np.stack([image, guiding_mask], axis=-1)
        utils.save_nifty(save_path + os.path.basename(image_filenames[i]), image, affine, spacing, header, is_mask=is_mask)

def rename_guiding_masks(load_path, save_path):
    filenames = utils.load_filenames(load_path)
    for i, filename in enumerate(filenames):
        basename = str(i+1).zfill(4) + "_0001.nii.gz"
        os.rename(filename, save_path + basename)


if __name__ == '__main__':
    slice_length = 75
    default_size = 1280
    load_path = "/gris/gris-f/homelv/kgotkows/datasets/covid19/medseg/lung_ggo_consolidation/train/masks/"
    save_path = "/gris/gris-f/homelv/kgotkows/datasets/covid19/medseg/lung_ggo_consolidation/train/guiding_masks/"
    comp_guiding_mask(load_path, save_path, slice_length, default_size)

    image_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/Task79_frankfurt3/labelsTr/"
    guiding_mask_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/Task79_frankfurt3/guiding_masks/"
    save_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/Task79_frankfurt3/appended_guiding_masks/"
    # add_to_images_or_masks(image_path, guiding_mask_path, save_path, is_mask=True)

    load_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/Task79_frankfurt3/guiding_masks/"
    save_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/Task79_frankfurt3/guiding_masks_renamed/"
    # rename_guiding_masks(load_path, save_path)
