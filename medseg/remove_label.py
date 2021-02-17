import numpy as np
from medseg import utils
from tqdm import tqdm
import os

def remove_label(load_path, save_path, labels_to_remove):
    save_path = utils.fix_path(save_path)
    load_path = utils.fix_path(load_path)
    filenames = utils.load_filenames(load_path)

    for filename in tqdm(filenames):
        basename = os.path.basename(filename)
        mask, affine, spacing, header = utils.load_nifty(filename)
        for label in labels_to_remove:
            mask[mask == label] = 0
        mask = np.rint(mask)
        mask = mask.astype(int)
        utils.save_nifty(save_path + basename, mask, affine, spacing, header)


if __name__ == '__main__':
    remove_label("/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task075_frankfurt3_ggo/labelsTr/",
                 "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task075_frankfurt3_ggo/labelsTr/", labels_to_remove=(2, 3))
