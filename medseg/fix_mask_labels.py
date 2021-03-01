import numpy as np
from medseg import utils
from tqdm import tqdm
import argparse
import os


def round_masks(load_path, save_path):
    filenames = utils.load_filenames(load_path)

    for filename in tqdm(filenames):
        round_mask(filename, save_path)


def round_mask(filename, save_path):
    mask, affine, spacing, header = utils.load_nifty(filename)
    mask = np.rint(mask)
    mask = mask.astype(np.uint8)
    utils.save_nifty(save_path + os.path.basename(filename), mask, affine, spacing, None, is_mask=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input folder", required=True)
    parser.add_argument("-o", "--output", help="Output folder", required=True)
    args = parser.parse_args()

    round_masks(args.input, args.output)
