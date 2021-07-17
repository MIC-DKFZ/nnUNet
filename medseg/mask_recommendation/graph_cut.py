from medseg import utils
import os
from evaluate import evaluate
import numpy as np
from imcut.pycut import ImageGraphCut
from tqdm import tqdm
import copy

def compute_predictions(image_path, mask_path, gt_path, save_path, version):
    image_filenames = utils.load_filenames(image_path)
    mask_filenames = utils.load_filenames(mask_path)

    segparams = {
        'use_boundary_penalties': False,
        'boundary_dilatation_distance': 1,
        'boundary_penalties_weight': 1,
        'block_size': 8,
        'tile_zoom_constant': 1
    }

    for i in tqdm(range(len(image_filenames))):
        image, affine, spacing, header = utils.load_nifty(image_filenames[i])
        multi_label_mask, _, _, _ = utils.load_nifty(mask_filenames[i])
        target_multi_label_mask = np.zeros_like(multi_label_mask)
        labels = np.unique(multi_label_mask)
        for label in range(1, len(labels) - 1):
            mask = copy.deepcopy(multi_label_mask)
            mask[mask == label] = -2  # Save foreground

            mask[mask >= 0] = 2  # Background
            mask[mask == -2] = 1  # Restore foreground
            mask[mask == -1] = 0  # Unknown
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
            target_multi_label_mask[mask == 0] = label  # 0 is foreground
        utils.save_nifty(save_path + os.path.basename(mask_filenames[i]), target_multi_label_mask, affine, spacing, header, is_mask=True)
    mean_dice_score, median_dice_score = evaluate(gt_path, save_path)
    return mean_dice_score, median_dice_score