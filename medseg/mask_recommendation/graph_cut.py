from medseg import utils
import os
from evaluate import evaluate
import numpy as np
# from imcut.pycut import ImageGraphCut
from imcut.pycut import ImageGraphCut
from tqdm import tqdm
import copy

def compute_predictions(image_path, mask_path, gt_path, save_path, version, nr_modalities, class_labels, resize=True):
    image_filenames = utils.load_filenames(image_path)[::nr_modalities]
    mask_filenames = utils.load_filenames(mask_path)
    target_shape = (512, 512, 200)  # (256, 256, 100)

    # for i in tqdm(range(len(image_filenames))):
    #     multi_label_mask, _, _, _ = utils.load_nifty(mask_filenames[i])
    #     print("shape: ", multi_label_mask.shape)

    segparams = {
        'use_boundary_penalties': False,
        'boundary_dilatation_distance': 1,
        'boundary_penalties_weight': 1,
        'block_size': 8,  # 8
        'tile_zoom_constant': 1
    }

    is_resized = False
    for i in tqdm(range(len(image_filenames))):
        image, affine, spacing, header = utils.load_nifty(image_filenames[i])
        multi_label_mask, _, _, _ = utils.load_nifty(mask_filenames[i])
        if resize and image.size > np.prod(target_shape):
            print("Resized: ", os.path.basename(image_filenames[i]))
            is_resized = True
            original_shape = image.shape
            image = utils.interpolate(image, (target_shape[0], target_shape[1], original_shape[2]))
            multi_label_mask = utils.interpolate(multi_label_mask, (target_shape[0], target_shape[1], original_shape[2]), mask=True)
        target_multi_label_mask = np.zeros_like(multi_label_mask)
        labels = np.unique(multi_label_mask)
        labels = labels[labels > 0].astype(int)
        # print("labels: ", labels)
        for label in labels:
            # print("label: ", label)
            mask = copy.deepcopy(multi_label_mask)
            mask[mask == label] = -2  # Save foreground
            mask[mask >= 0] = 2  # Background
            mask[mask == -2] = 1  # Restore foreground
            mask[mask == -1] = 0  # Unknown
            # utils.save_nifty(save_path + os.path.basename(mask_filenames[i][:-12] + "_tmp1.nii.gz"), mask, affine, spacing, header, is_mask=True)
            mask = mask.astype(np.uint8)
            if version == "GraphCut1":
                segparams.update({"method": "graphcut"})
                gc = ImageGraphCut(image, segparams=segparams)
            elif version == "GraphCut2":
                segparams.update({"method": "lo2hi"})
                gc = ImageGraphCut(image, segparams=segparams)
            elif version == "GraphCut3":
                segparams.update({"method": "hi2lo"})
                gc = ImageGraphCut(image, segparams=segparams)
            gc.set_seeds(mask)
            gc.run()
            mask = gc.segmentation.squeeze()
            # mask[mask == 0] = -1  # Save foreground
            # mask[mask == 1] = 0  # Background
            # mask[mask == -1] = label  # Restore foreground
            # print(save_path + os.path.basename(mask_filenames[i][:-12] + "_tmp2.nii.gz"))
            # utils.save_nifty(save_path + os.path.basename(mask_filenames[i][:-12] + "_tmp2.nii.gz"), mask, affine, spacing, header, is_mask=True)
            target_multi_label_mask[mask == 0] = label  # 0 is foreground
        if is_resized:
            is_resized = False
            target_multi_label_mask = utils.interpolate(target_multi_label_mask, original_shape, mask=True)
        utils.save_nifty(save_path + os.path.basename(mask_filenames[i][:-12] + ".nii.gz"), target_multi_label_mask, affine, spacing, header, is_mask=True)
    results = evaluate(gt_path, save_path, class_labels)
    return results