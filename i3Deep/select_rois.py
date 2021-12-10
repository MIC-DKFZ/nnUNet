import numpy as np
import os
from tqdm import tqdm
from i3Deep import utils
from skimage.util import view_as_windows
from PIL import Image
from pathlib import Path
import cv2

def select_rois(img_dir, uncertainty_mask_dir, save_dir, window_size_percentage=0.02, window_per_border=3, max_rois=5, min_z_distance_percentage=0.1, max_iou=0.1):
    imgs_filenames = utils.load_filenames(img_dir)
    uncertainty_masks_filenames = utils.load_filenames(uncertainty_mask_dir)
    uncertainty_masks = [utils.load_nifty(uncertainty_mask_filename)[0] for uncertainty_mask_filename in uncertainty_masks_filenames]
    uncertainty_masks = [utils.normalize(uncertainty_mask) for uncertainty_mask in uncertainty_masks]
    uncertainty_masks_size_mean = comp_uncertainty_masks_mean(uncertainty_masks)
    window_shapes = comp_window_shapes(uncertainty_masks_size_mean, window_size_percentage, window_per_border)

    for i in tqdm(range(len(imgs_filenames))):
        img, affine, spacing, header = utils.load_nifty(imgs_filenames[i])
        if len(img.shape) == 4:  # TODO: Remove modality in the case of prostate dataset, remove in final version
            img = img[..., 0]
        img_reoriented = utils.reorient(img, affine)  # TODO: Reorient ist hardcoded
        uncertainty_mask_reoriented = utils.reorient(uncertainty_masks[i], affine)
        rois = []  # Each entry is [roi_sum, x, y, z, width, length]
        for window_shape in tqdm(window_shapes):
            window_shape_rois = comp_rois_single_window_shape(uncertainty_mask_reoriented, window_shape)
            rois.extend(window_shape_rois)
        rois = np.asarray(rois)
        rois = filter_rois(rois, max_rois, uncertainty_mask_reoriented.shape, min_z_distance_percentage, max_iou)
        rois = extract_rois(img_reoriented, uncertainty_mask_reoriented, rois)
        save_rois(save_dir, os.path.basename(uncertainty_masks_filenames[i][:-7]) + "/", rois, img, uncertainty_masks[i], affine, spacing, header)
        # TODO: Save 3D img without affine transform for viewing in frontend and show roi coords or affine transform coords?
        # TODO: Save prediction (without uncertainty) as roi as well


def comp_rois_single_window_shape(uncertainty_mask, window_shape):
    uncertainty_mask_tmp = pad(uncertainty_mask, window_shape)
    uncertainty_mask_tmp = np.transpose(uncertainty_mask_tmp, (2, 0, 1))
    rois = []
    for i in range(len(uncertainty_mask_tmp)):
        rois.extend(comp_rois_slice_single_window_shape(uncertainty_mask_tmp[i], window_shape, i))
    return rois


def comp_rois_slice_single_window_shape(uncertainty_mask_slice, window_shape, z):
    step = (5, 5)  # (int(window_shape[0]/3), int(window_shape[1]/3))
    windows = view_as_windows(uncertainty_mask_slice, window_shape, step=step)
    windows = np.sum(windows, axis=(2,3))
    rois = []
    for x in range(windows.shape[0]):
        for y in range(windows.shape[1]):
            rois.append([windows[x][y], x * step[0], y * step[1], z, window_shape[0], window_shape[1]])
    return rois


def filter_rois(rois, max_rois, mask_shape, min_z_distance_percentage, max_iou):
    min_z_distance = int(mask_shape[2] * min_z_distance_percentage)
    rois = rois[np.argsort(-rois[:, 0])]  # Each entry is [roi_sum, x, y, z, width, length]
    accepted_rois = []

    print("Rois len: ", len(rois))
    index = 1
    pbar = tqdm(total=max_rois)
    pbar.update(1)
    while len(accepted_rois) < max_rois and index < len(rois):
        roi1 = rois[index]
        accept = True
        for roi2 in rois[:index]:
            bb1 = {"x1": roi1[1], "x2": roi1[1] + roi1[4], "y1": roi1[2], "y2": roi1[2] + roi1[5]}
            bb2 = {"x1": roi2[1], "x2": roi2[1] + roi2[4], "y1": roi2[2], "y2": roi2[2] + roi2[5]}
            iou = intersection_over_union(bb1, bb2)
            if iou > max_iou and abs(roi1[3] - roi2[3]) < min_z_distance:
                accept = False
                break
        if accept:
            accepted_rois.append(roi1)
            index += 1
            pbar.update(1)
        else:
            rois = np.delete(rois, index, axis=0)
    pbar.close()

    return accepted_rois


def extract_rois(img, mask, rois):
    # Each roi is [roi_sum, x, y, z, width, length]
    # Return [[img_roi, mask_roi, coords, img_with_bb], ...]
    extracted_rois = []
    for roi in rois:
        x = int(roi[1])
        y = int(roi[2])
        z = int(roi[3])
        width = int(roi[4])
        length = int(roi[5])
        img_tmp = pad(img, (width, length))
        mask_tmp = pad(mask, (width, length))
        img_roi = img_tmp[x:x + width, y:y + length, z]
        mask_roi = mask_tmp[x:x + width, y:y + length, z]
        img_with_bb = img_tmp[:, :, z]
        mask_with_bb = mask_tmp[:, :, z]
        img_roi = (utils.normalize(img_roi) * 255).astype(np.uint8)
        #mask_roi = (utils.normalize(mask_roi) * 255).astype(np.uint8)
        mask_roi = (mask_roi * 255).astype(np.uint8)
        img_with_bb = (utils.normalize(img_with_bb) * 255).astype(np.uint8)
        mask_with_bb = (mask_with_bb * 255).astype(np.uint8)
        cv2.rectangle(img_with_bb, (y, x), (y + length, x + width), 255, 1)
        cv2.rectangle(mask_with_bb, (y, x), (y + length, x + width), 255, 1)
        extracted_rois.append([img_roi, mask_roi, img_with_bb, mask_with_bb, x, y, z, width, length])
    return extracted_rois


def save_rois(save_dir, roi_dir, rois, img, uncertainty_mask, affine, spacing, header):
    # rois = [[img_roi, mask_roi, coords, img_with_bb], ...]
    Path(save_dir + roi_dir).mkdir(parents=True, exist_ok=True)
    for i, (img_roi, mask_roi, img_with_bb, mask_with_bb, x, y, z, width, length) in enumerate(rois):
        img_roi = Image.fromarray(img_roi).convert('L')
        mask_roi = Image.fromarray(mask_roi).convert('L')
        img_with_bb = Image.fromarray(img_with_bb).convert('L')
        mask_with_bb = Image.fromarray(mask_with_bb).convert('L')
        img_roi.save(save_dir + roi_dir + str(i).zfill(3) + "_img_x_" + str(x+1) + "_y_" + str(y+1) + "_z_" + str(z+1) + ".png")
        mask_roi.save(save_dir + roi_dir + str(i).zfill(3) + "_uncertainty_x_" + str(x+1) + "_y_" + str(y+1) + "_z_" + str(z+1) + ".png")
        img_with_bb.save(save_dir + roi_dir + str(i).zfill(3) + "_img_with_bb_x_" + str(x + 1) + "_y_" + str(y + 1) + "_z_" + str(z + 1) + ".png")
        mask_with_bb.save(save_dir + roi_dir + str(i).zfill(3) + "_mask_with_bb_x_" + str(x + 1) + "_y_" + str(y + 1) + "_z_" + str(z + 1) + ".png")
    utils.save_nifty(save_dir + roi_dir + "img.nii.gz", img, affine, spacing, header)
    utils.save_nifty(save_dir + roi_dir + "uncertainty_mask.nii.gz", uncertainty_mask, affine, spacing, header)


def comp_uncertainty_masks_mean(uncertainty_masks):
    uncertainty_masks_mean = np.asarray([uncertainty_mask.shape[:2] for uncertainty_mask in uncertainty_masks])
    uncertainty_masks_mean = [np.mean(uncertainty_masks_mean[:, 0]), np.mean(uncertainty_masks_mean[:, 1])]
    uncertainty_masks_mean = np.mean(uncertainty_masks_mean)  # TODO: Currently computes only one global mean (single int)
    return uncertainty_masks_mean


def comp_window_shapes(uncertainty_masks_mean, window_size_percentage, window_per_border):
    max_window_size = np.sqrt((uncertainty_masks_mean ** 2) * window_size_percentage)
    window_sizes = np.linspace(max_window_size * 0.75, max_window_size, window_per_border, endpoint=True, dtype=int)
    window_shapes = []
    for x in window_sizes:
        y = (max_window_size ** 2) / x
        window_shapes.append([x, y])
        window_shapes.append([y, x])

    window_shapes = np.asarray(window_shapes, dtype=int)
    window_shapes = window_shapes[:-1]
    return window_shapes


def pad(data, window_shape):
    width, length = window_shape
    width = int(width - (data.shape[0] % width))
    length = int(length - (data.shape[1] % length))
    data = np.pad(data, ((0, width), (0, length), (0, 0)), 'constant')
    return data


def intersection_over_union(bb1, bb2):
    """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters
        ----------
        bb1 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner

        Returns
        -------
        float
            in [0, 1]
        """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


if __name__ == '__main__':
    img_dir = "/gris/gris-f/homelv/kgotkows/datasets/prostate/Task05_Prostate/imagesTs/"
    uncertainty_mask_dir = "/gris/gris-f/homelv/kgotkows/datasets/prostate/Task05_Prostate/predictions_with_tta_label2/"
    save_dir = "/gris/gris-f/homelv/kgotkows/datasets/prostate/Task05_Prostate/rois/"
    select_rois(img_dir, uncertainty_mask_dir, save_dir)
