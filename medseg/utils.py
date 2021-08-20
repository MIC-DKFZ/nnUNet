import numpy as np
import os
from natsort import natsorted
import nibabel as nib
from nilearn.image import resample_img
import torch
from torch.nn import functional as F
from scipy.ndimage import affine_transform
# import transforms3d as t3d
import sys
import matplotlib.pyplot as plt

def fix_path(path):
    if path[-1] != "/":
        path += "/"
    return path

def load_filenames(img_dir, extensions=None):  # '.nii.gz'
    _img_dir = fix_path(img_dir)
    img_filenames = []

    for file in os.listdir(_img_dir):
        if extensions is None or file.endswith(extensions):
            img_filenames.append(_img_dir + file)
    img_filenames = np.asarray(img_filenames)
    img_filenames = natsorted(img_filenames)

    return img_filenames

def load_npy(filepath):
    img = np.load(filepath, allow_pickle=True)
    return img["img"], img["affine"], img["spacing"], img["header"]

def save_npy(filepath, img, affine=None, spacing=None, header=None, is_mask=False):
    if is_mask:
        img = np.rint(img)
        img = img.astype(np.int)
    # img = {"img": img, "affine": affine, "spacing": spacing, "header": header}
    np.savez_compressed(filepath, img=img, affine=affine, spacing=spacing, header=header)

def load_nifty(filepath):
    img = nib.load(filepath)
    # if shape is not None:
    #     if not mask:
    #         img = resample_img(img, target_shape=shape, target_affine=np.eye(4))
    #     else:
    #         img = resample_img(img, target_shape=shape, target_affine=np.eye(4), interpolation='nearest')
    affine = img.affine
    img_np = img.get_fdata()
    spacing = img.header["pixdim"][1:4]
    header = img.header
    return img_np, affine, spacing, header

def save_nifty(filepath, img, affine=None, spacing=None, header=None, is_mask=False):
    if is_mask:
        img = np.rint(img)
        img = img.astype(np.int)
    img = nib.Nifti1Image(img, affine=affine, header=header)
    if spacing is not None:
        img.header["pixdim"][1:4] = spacing
    nib.save(img, filepath)

def reorient(img, affine=None):
    reoriented = np.rot90(img, k=1)
    reoriented = np.fliplr(reoriented)
    # plt.imshow(normalize(img[:, :, 0]))
    # plt.savefig("/gris/gris-f/homelv/kgotkows/datasets/prostate/Task05_Prostate/001.png")
    # plt.imshow(normalize(reoriented[:, :, 0]))
    # plt.savefig("/gris/gris-f/homelv/kgotkows/datasets/prostate/Task05_Prostate/002.png")
    # sys.exit(0)
    return reoriented

def normalize(x, x_min=None, x_max=None):
    if x_min is None:
        x_min = x.min()

    if x_max is None:
        x_max = x.max()

    if x_min == x_max:
        return x * 0
    else:
        return (x - x.min()) / (x.max() - x.min())

def normalize_list(x):
    min_value = np.min(x)
    max_value = np.min(x)
    return (x - min_value) / (max_value - min_value)

def interpolate(data, shape, mask=False):
    data = torch.FloatTensor(data)
    data = data.unsqueeze(0).unsqueeze(0)
    if not mask:
        data = F.interpolate(data, shape, mode="trilinear", align_corners=False)
    else:
        data = F.interpolate(data, shape, mode="nearest")
    data = data.squeeze(0).squeeze(0)
    data = data.numpy()
    return data