from i3Deep import utils
import numpy as np
from tqdm import tqdm
import os
import time

def nifty2npy(load_path, save_path):
    load_path = utils.fix_path(load_path)
    save_path = utils.fix_path(save_path)
    filenames = utils.load_filenames(load_path)
    for filename in tqdm(filenames):
        img, affine, spacing, header = utils.load_nifty(filename)
        utils.save_npy(save_path + os.path.basename(filename)[:-7], img, affine, spacing, header)

def load_speed_comparison(load_path_nifty, load_path_npy):
    load_path_nifty = utils.fix_path(load_path_nifty)
    load_path_npy = utils.fix_path(load_path_npy)
    filenames = utils.load_filenames(load_path_nifty)
    start_time_nifty = time.time()
    for filename in tqdm(filenames):
        img, affine, spacing, header = utils.load_nifty(filename)
    end_time_nifty = time.time()
    time_nifty = end_time_nifty - start_time_nifty
    filenames = utils.load_filenames(load_path_npy)
    start_time_npy = time.time()
    for filename in tqdm(filenames):
        img, affine, spacing, header = utils.load_npy(filename)
    end_time_npy = time.time()
    time_npy = end_time_npy - start_time_npy
    print("Nifty: {}, npy: {}, npy-nifty: {}".format(time_nifty, time_npy, time_npy - time_nifty))


if __name__ == '__main__':
    load_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task070_guided_all_public_ggo/refinement_val/uncertainties/ensemble/bhattacharyya_coefficient/"
    save_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task070_guided_all_public_ggo/refinement_val/uncertainties/ensemble/bhattacharyya_coefficient_npy/"
    # nifty2npy(load_path, save_path)
    load_speed_comparison(load_path, save_path)