import numpy as np
import os
from tqdm import tqdm
from i3Deep import utils

with_tta = "/gris/gris-f/homelv/kgotkows/datasets/prostate/Task05_Prostate/predictions_with_tta/"
without_tta = "/gris/gris-f/homelv/kgotkows/datasets/prostate/Task05_Prostate/predicitons_without_tta/"
diff_dir = "/gris/gris-f/homelv/kgotkows/datasets/prostate/Task05_Prostate/prediciton_diffs/"


def comp_diffs(imgs_with_tta, imgs_without_tta, diff_dir):
    imgs_with_tta = utils.load_filenames(imgs_with_tta)
    imgs_without_tta = utils.load_filenames(imgs_without_tta)

    for i in tqdm(range(len(imgs_with_tta))):
        img_with_tta, affine, spacing, header = utils.load_nifty(imgs_with_tta[i])
        img_without_tta, _, _, _ = utils.load_nifty(imgs_without_tta[i])
        diff = comp_diff(img_with_tta, img_without_tta)
        utils.save_nifty(diff_dir + os.path.basename(imgs_with_tta[i]), diff, affine, spacing, header)


def comp_diff(img_with_tta, img_without_tta):
    diff = img_without_tta - img_with_tta
    diff = np.abs(diff)
    return diff


if __name__ == '__main__':
    comp_diffs(with_tta, without_tta, diff_dir)
