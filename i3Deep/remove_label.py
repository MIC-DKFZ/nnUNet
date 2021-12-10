import numpy as np
from i3Deep import utils
from tqdm import tqdm
import os

def remap_label(load_path, save_path, remap_labels, target_label):
    save_path = utils.fix_path(save_path)
    load_path = utils.fix_path(load_path)
    filenames = utils.load_filenames(load_path)

    for filename in tqdm(filenames):
        basename = os.path.basename(filename)
        mask, affine, spacing, header = utils.load_nifty(filename)
        for label in remap_labels:
            mask[mask == label] = target_label
        mask = np.rint(mask)
        mask = mask.astype(int)
        utils.save_nifty(save_path + basename, mask, affine, spacing, header)


if __name__ == '__main__':
    remap_label("/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task001_BrainTumour/labelsTr_tmp/",
                 "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task001_BrainTumour/labelsTr_tmp/", remap_labels=[2, 3], target_label=1)
