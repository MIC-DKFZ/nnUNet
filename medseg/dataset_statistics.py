from medseg import utils
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

val_gt_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task070_guided_all_public_ggo/refinement_val/labels/"
test_gt_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task070_guided_all_public_ggo/refinement_test/labels/"

val_gt_filenames = utils.load_filenames(val_gt_path)
test_gt_filenames = utils.load_filenames(test_gt_path)

val_ggo, test_ggo = [], []

for filename in tqdm(val_gt_filenames):
    gt = utils.load_nifty(filename)[0]
    val_ggo.append(np.sum(gt))

for filename in tqdm(test_gt_filenames):
    gt = utils.load_nifty(filename)[0]
    test_ggo.append(np.sum(gt))

val_ggo = np.sort(val_ggo)
test_ggo = np.sort(test_ggo)

plt.bar(range(len(test_ggo)), test_ggo)
plt.bar(range(len(val_ggo)), val_ggo)
#plt.show()

plt.show()