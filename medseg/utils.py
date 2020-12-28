import numpy as np
import os
from natsort import natsorted
import nibabel as nib
from scipy.ndimage import affine_transform
import transforms3d as t3d
import sys
import matplotlib.pyplot as plt

def load_filenames(img_dir, extensions=('.nii.gz')):
    img_filenames, mask_files = [], []

    for file in os.listdir(img_dir):
        if file.endswith(extensions):
            img_filenames.append(img_dir + file)
    img_filenames = np.asarray(img_filenames)
    img_filenames = natsorted(img_filenames)

    return img_filenames

def load_nifty(filepath):
    img = nib.load(filepath)
    affine = img.affine
    img_np = img.get_fdata()
    spacing = img.header["pixdim"][1:4]
    header = img.header
    return img_np, affine, spacing, header

def save_nifty(filepath, img, affine=None, spacing=None, header=None):
    img = nib.Nifti1Image(img, affine=affine, header=header)
    if spacing is not None:
        img.header["pixdim"][1:4] = spacing
    nib.save(img, filepath)

def reorient(img, affine):
    reoriented = np.rot90(img, k=1)
    reoriented = np.fliplr(reoriented)
    # plt.imshow(normalize(img[:, :, 0]))
    # plt.savefig("/gris/gris-f/homelv/kgotkows/datasets/prostate/Task05_Prostate/001.png")
    # plt.imshow(normalize(reoriented[:, :, 0]))
    # plt.savefig("/gris/gris-f/homelv/kgotkows/datasets/prostate/Task05_Prostate/002.png")
    # sys.exit(0)
    return reoriented

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def normalize_list(x):
    min_value = np.min(x)
    max_value = np.min(x)
    return (x - min_value) / (max_value - min_value)
