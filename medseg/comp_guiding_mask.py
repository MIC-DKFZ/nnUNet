from medseg import utils
import numpy as np
from tqdm import tqdm
import math
import random
import copy
from skimage import measure
import os
import GeodisTK


def comp_guiding_mask(load_path, save_path, slice_gap, default_size, slice_depth=1):
    filenames = utils.load_filenames(load_path)
    for filename in tqdm(filenames):
        mask, affine, spacing, header = utils.load_nifty(filename)
        adapted_slice_gap = adapt_slice_gap(mask, slice_gap, default_size)
        mask_slices = comp_slices_mask_validation(mask, adapted_slice_gap, slice_depth=slice_depth)
        utils.save_nifty(save_path + os.path.basename(filename), mask_slices, affine, spacing, header, is_mask=True)


def adapt_slice_gap(mask, slice_gap, default_size):
    return int((mask.shape[0] / default_size) * slice_gap)


# def comp_slices_mask_validation(mask, slice_gap, p=None, nnunet=False, slice_depth=3):
#     labels = range(1, int(np.max(mask)) + 1)
#     mask_slices = np.zeros_like(mask)
#     for label in labels:
#         mask_label = copy.deepcopy(mask)
#         mask_label[mask_label != label] = 0
#         mask_label[mask_label == label] = 1
#         objects_slices_label = comp_slices_label(mask_label, slice_gap, p, nnunet, slice_depth).astype(int)
#         objects_slices_label[objects_slices_label == 1] = label
#         mask_slices += objects_slices_label
#     if nnunet:
#         mask_slices[mask == -1] = -1
#     return mask_slices


def comp_slices_mask_validation(mask, slice_gap, p=None, nnunet=False, slice_depth=1):
    binarized_mask = copy.deepcopy(mask)
    binarized_mask[binarized_mask > 0] = 1
    slices = comp_slices_label(binarized_mask, slice_gap, p, nnunet, slice_depth).astype(int)
    unique = np.unique(mask)
    # unique = unique[unique != -1]
    mask_slices = np.zeros_like(mask)
    for label in unique:
        mask_slices[(slices == 1) & (mask == label)] = label
    mask_slices = np.rint(mask_slices)
    if nnunet:
        mask_slices[mask == -1] = -1
    return mask_slices


def comp_slices_label(mask, slice_gap, p=None, nnunet=False, slice_depth=3):
    objects, ids = measure.label(mask, background=False, connectivity=2, return_num=True)
    mask_slices = np.zeros_like(mask)
    for id in range(1, ids + 1):
        object = copy.deepcopy(objects)
        object[object != id] = 0
        object[object == id] = 1
        object_slices = comp_object_slices(object, slice_gap, p, nnunet, slice_depth)
        mask_slices = np.logical_or(mask_slices, object_slices)
    return mask_slices


def comp_object_slices(object, slice_gap, p=None, nnunet=False, slice_depth=3):
    object_slices, is_object_small = comp_object_slices_dim(object, 0, slice_gap, p, slice_depth)
    if not is_object_small or nnunet:
        object_slices_1, is_object_small = comp_object_slices_dim(object, 1, slice_gap, p, slice_depth)
        object_slices_2, is_object_small = comp_object_slices_dim(object, 2, slice_gap, p, slice_depth)
        object_slices = np.logical_or(object_slices, object_slices_1)
        object_slices = np.logical_or(object_slices, object_slices_2)
    return object_slices


def comp_object_slices_dim(object, dim, slice_gap, p=None, slice_depth=3):
    dims = [0, 1, 2]
    dims.remove(dim)
    object_flat = np.sum(object, axis=tuple(dims))
    nonzero_indices = np.nonzero(object_flat)
    min_index = np.min(nonzero_indices)
    max_index = np.max(nonzero_indices)
    object_length = max_index - min_index
    slice_count = math.ceil(object_length / slice_gap)
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

        for offset in range(slice_depth):
            offset = int(((slice_depth - 1) / 2) + offset)
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


def comp_slices_mask_training(mask):
    max_slices = int(mask.shape[2] / 3)
    num_slices = random.randint(0, max_slices)

    slices = np.zeros_like(mask)
    for i in range(num_slices):
        dim = random.randint(0, 2)
        slice_index = random.randint(0, mask.shape[dim]-1)
        if dim == 0:
            slices[slice_index, :, :] = 1
        elif dim == 1:
            slices[:, slice_index, :] = 1
        else:
            slices[:, :, slice_index] = 1

    # mask_slices = copy.deepcopy(mask)
    # mask_slices = np.logical_and(mask_slices, slices).astype(np.float32)
    # mask_slices = np.logical_and(mask, slices).astype(np.float32)
    mask_slices = np.zeros_like(mask)
    mask2 = mask + 1
    unique = np.unique(mask2)
    # unique = unique[unique != -1]
    for label in unique:
        mask_slices[(slices == 1) & (mask2 == label)] = label
    # mask_slices[mask == -1] = -1
    mask_slices = np.rint(mask_slices)
    # name = random.randint(0, 1000)
    # utils.save_nifty("/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/tmp/{}_slices.nii.gz".format(name), slices)
    # utils.save_nifty("/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/tmp/{}_mask.nii.gz".format(name), mask)
    # utils.save_nifty("/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/tmp/{}_guiding_mask.nii.gz".format(name), mask_slices)
    return mask_slices


def add_to_images_or_masks(image_path, guiding_mask_path, save_path, is_mask=False):
    image_filenames = utils.load_filenames(image_path)
    guiding_mask_filenames = utils.load_filenames(guiding_mask_path)
    for i in tqdm(range(len(image_filenames))):
        image, affine, spacing, header = utils.load_nifty(image_filenames[i])
        guiding_mask, _, _, _ = utils.load_nifty(guiding_mask_filenames[i])
        image = np.stack([image, guiding_mask], axis=-1)
        utils.save_nifty(save_path + os.path.basename(image_filenames[i]), image, affine, spacing, header, is_mask=is_mask)


def rename_guiding_masks(data_path, modality):
    filenames = utils.load_filenames(data_path)
    for i, filename in enumerate(filenames):
        # basename = str(i+1).zfill(4) + "_0001.nii.gz"
        basename = os.path.basename(filename)[:-7] + "_" + str(modality).zfill(4) + ".nii.gz"
        os.rename(filename, data_path + basename)


def guiding2geodistk(data_path, lamb=0.99, iterations=4, modality=1):
    filenames = utils.load_filenames(data_path)
    for i in tqdm(range(0, len(filenames), modality+1)):
        image, affine, spacing, header = utils.load_nifty(filenames[i])
        guiding_mask, _, _, _ = utils.load_nifty(filenames[i+modality])

        geodesic_distance_map = GeodisTK.geodesic3d_raster_scan(image.astype(np.float32), guiding_mask.astype(np.uint8), spacing, lamb, iterations)
        # geodesic_distance_map = utils.normalize(geodesic_distance_map)
        # geodesic_distance_map = GeodisTK.geodesic3d_fast_marching(image.astype(np.float32), guiding_mask.astype(np.uint8), lamb)
        utils.save_nifty(filenames[i+modality], geodesic_distance_map, affine, spacing, header, is_mask=False)


if __name__ == '__main__':
    slice_gap = 70  # COVID-19: 70, BrainTumor: 70
    default_size = 1280
    slice_depth = 1
    base_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task002_BrainTumour_guided/"
    load_path = base_path + "labelsTr/"
    save_path = base_path + "guiding_masks/"
    # comp_guiding_mask(load_path, save_path, slice_gap, default_size, slice_depth)
    rename_guiding_masks(save_path, 4)
    # 0.0, 0.99
    # guiding2geodistk("/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task003_BrainTumour_guided_DeepIGeos1/imagesTr/", lamb=0.99, iterations=1, modality=4)

    # image_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/Task79_frankfurt3/labelsTr/"
    # guiding_mask_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/Task79_frankfurt3/guiding_masks/"
    # save_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/Task79_frankfurt3/appended_guiding_masks/"
    # add_to_images_or_masks(image_path, guiding_mask_path, save_path, is_mask=True)
