from i3Deep import utils
import numpy as np
from tqdm import tqdm
import os

gt_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task070_guided_all_public_ggo/refinement_test/labels/"
gt_filenames = utils.load_filenames(gt_path)

names, severity_scores, normed_severity_scores = [], [], []

for gt_filename in tqdm(gt_filenames):
    name = os.path.basename(gt_filename)[:-7]
    gt, affine, spacing, header = utils.load_nifty(gt_filename)
    names.append(name)
    severity = np.sum(gt)
    severity_scores.append(severity)
    normed_severity_scores.append(severity / gt.size)

names = np.asarray(names)
severity_scores = np.asarray(severity_scores)
normed_severity_scores = np.asarray(normed_severity_scores)

ordering = np.argsort(-1 * severity_scores)
severity_scores = severity_scores[ordering]
names = names[ordering]
for i in range(len(severity_scores)):
    print("{}: {}".format(names[i], severity_scores[i]))

print("--------------------")

ordering = np.argsort(-1 * normed_severity_scores)
normed_severity_scores = normed_severity_scores[ordering]
names = names[ordering]
for i in range(len(normed_severity_scores)):
    print("{}: {}".format(names[i], normed_severity_scores[i]))