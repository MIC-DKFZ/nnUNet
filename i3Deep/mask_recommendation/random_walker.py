from i3Deep import utils
import os
from evaluate import evaluate
import numpy as np
from skimage.segmentation.random_walker_segmentation import random_walker
from tqdm import tqdm
import torchio
import torch

def compute_predictions(image_path, mask_path, gt_path, save_path, nr_modalities, class_labels, resize=True, beta=10):
    image_filenames = utils.load_filenames(image_path)[::nr_modalities]
    mask_filenames = utils.load_filenames(mask_path)
    target_shape = (256, 256, 200)  # (256, 256, 100)

    is_resized = False
    for i in tqdm(range(len(image_filenames))):
        image, affine, spacing, header = utils.load_nifty(image_filenames[i])
        mask, _, _, _ = utils.load_nifty(mask_filenames[i])
        if resize and image.size > np.prod(target_shape):
            is_resized = True
            print("Resized: ", os.path.basename(image_filenames[i]))
            original_shape = image.shape
            image = utils.interpolate(image, (target_shape[0], target_shape[1], original_shape[2]))
            mask = utils.interpolate(mask, (target_shape[0], target_shape[1], original_shape[2]), mask=True)
        image = utils.normalize(image)
        labels = np.unique(mask)
        # labels = labels[labels > 0]
        for label in np.flip(labels):
            mask[mask == label] = label + 1
        mask = mask.astype(np.uint8)
        mask = random_walker(data=image, labels=mask, beta=beta, mode='cg_mg')
        for label in labels:
            mask[mask == label + 1] = label
        if is_resized:
            mask = utils.interpolate(mask, original_shape, mask=True)
        utils.save_nifty(save_path + os.path.basename(mask_filenames[i][:-12] + ".nii.gz"), mask, affine, spacing, header, is_mask=True)
    results = evaluate(gt_path, save_path, class_labels)
    return results


# def compute_predictions(image_path, mask_path, gt_path, save_path):
#     image_filenames = utils.load_filenames(image_path)
#     mask_filenames = utils.load_filenames(mask_path)
#
#     for i in tqdm(range(len(image_filenames))):
#         _, affine, spacing, header = utils.load_nifty(mask_filenames[i])
#         subject = torchio.Subject(image=torchio.ScalarImage(image_filenames[i]), mask=torchio.LabelMap(mask_filenames[i]))
#         sampler = torchio.inference.GridSampler(subject, patch_size=(20, 20, 10), padding_mode='edge')
#         aggregator = torchio.inference.GridAggregator(sampler)
#         for patch in sampler:
#             image = patch["image"][torchio.DATA].numpy()[0]
#             image = utils.normalize(image)
#             mask = patch["mask"][torchio.DATA].numpy()[0]
#             location = torch.tensor(patch[torchio.LOCATION]).unsqueeze(0)
#             if not(image.max() <= 0 or mask.max() == 0):
#                 # image[image < 0] = 0
#                 mask = mask.astype(np.int32)
#                 mask = random_walker(data=image, labels=mask, mode='cg_j')
#             mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0)
#             aggregator.add_batch(mask, location)
#         mask = aggregator.get_output_tensor()
#         utils.save_nifty(save_path + os.path.basename(mask_filenames[i]), mask, affine, spacing, header, is_mask=True)
#     mean_dice_score, median_dice_score = evaluate(gt_path, save_path)
#     return mean_dice_score, median_dice_score