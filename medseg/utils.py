import numpy as np
import os
from natsort import natsorted
import nibabel as nib

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

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def normalize_list(x):
    min_value = np.min(x)
    max_value = np.min(x)
    return (x - min_value) / (max_value - min_value)
