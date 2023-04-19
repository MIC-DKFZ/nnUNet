import cv2
import math
import random

import numpy as np
import nibabel as nib

from pathlib import Path
from typing import List
from torch.utils.data import Dataset

from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms import \
    BrightnessMultiplicativeTransform, GammaTransform, \
    BrightnessTransform, ContrastAugmentationTransform
from batchgenerators.transforms.noise_transforms import \
    GaussianNoiseTransform, GaussianBlurTransform


class Dataset3D(Dataset):
    def __init__(
        self,
        root: Path,
        list_path: str,
        global_crop_size: tuple = (16, 128, 128),
        local_crop_size: tuple = (16, 64, 64),
        local_crops_number: int = 0
    ):
        self.root = root
        self.list_path = root / list_path
        self.img_ids = [i_id.strip().split() for i_id in open(self.list_path)]
        self.local_crops_number = local_crops_number
        self.rng = np.random.default_rng(420)
        self.files = []
        for item in self.img_ids:
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print(f'SSL: {len(self.img_ids)} images are loaded!')

        self.global_crop3D_d, self.global_crop3D_h, self.global_crop3D_w = global_crop_size
        self.local_crop3D_d, self.local_crop3D_h, self.local_crop3D_w = local_crop_size

        self.tr_transforms3D_global0 = get_train_transform3D(
             gblur_p=0.8, brightm_p=0.8, bright_p=0.8, gamma_p=0.8)
        self.tr_transforms3D_global1 = get_train_transform3D(
             gblur_p=0.8, brightm_p=0.8, bright_p=0.8, gamma_p=0.8)
        self.tr_transforms3D_local = get_train_transform3D(
             gblur_p=0.5, brightm_p=1.0, bright_p=0.5, gamma_p=1.0)

    def __len__(self):
        return len(self.files)

    def crop_scale(self, image, scale_range: tuple, wise_crop: bool = False, bkgd_val: float = None):
        
        assert wise_crop and (bkgd_val is not None), \
            'If wise cropping needs to be done the background value yo must pass'
        has_desired_ratio = not wise_crop

        _, img_d, img_h, img_w = image.shape

        # Define a random crop size that fits inside the image
        # on height and width, depth is kept fixed
        while not has_desired_ratio:
            scaler_d = 1.
            scaler_h = np.random.uniform(scale_range[0], scale_range[1])
            if (int(self.global_crop3D_h * scaler_h) >= img_h):
                scaler_h = 1.
            scaler_w = np.random.uniform(scale_range[0], scale_range[1])
            if (int(self.global_crop3D_w * scaler_w) >= img_w):
                scaler_w = 1.
            scale_d = int(self.global_crop3D_d * scaler_d)
            scale_h = int(self.global_crop3D_h * scaler_h)
            scale_w = int(self.global_crop3D_w * scaler_w)

            # sample a random origin for the crop
            d0 = random.randint(0, img_d - scale_d)
            h0 = random.randint(0, img_h - scale_h)
            w0 = random.randint(0, img_w - scale_w)
            # define the limits of the crop
            d1 = d0 + scale_d
            h1 = h0 + scale_h
            w1 = w0 + scale_w

            # crop image
            image_crop = image[:, d0: d1, h0: h1, w0: w1]
            bkgd_ratio = (image_crop == bkgd_val).sum() / np.prod(image_crop.shape)
            if bkgd_ratio > 0.4:
                sample = self.rng.uniform(low=0.0, high=1.0)
                has_desired_ratio = False if sample < bkgd_ratio else True
            else:
                has_desired_ratio = True
        return image_crop

    def crop_scale_mirror(self, image: np.ndarray, axes=(0, 1, 2), local: bool = False):
        
        if not local:
            scale_range = (0.8, 1.2)
        else:
            scale_range = (1.5, 2.0)

        image_crop = self.crop_scale(image, scale_range)
        # apply random mirroring on the three axes
        if 2 in axes and np.random.uniform() < 0.8:
            image_crop = image_crop[:, :, :, ::-1]
        if 1 in axes and np.random.uniform() < 0.8:
            image_crop = image_crop[:, :, ::-1]
        if 0 in axes and np.random.uniform() < 0.8:
            image_crop = image_crop[:, ::-1]

        # resize the crop to the original image size and add batch axis
        if (image_crop.shape != image.shape):
            image_crop = cv2.resize(
                image_crop[0].transpose(1, 2, 0),
                (self.global_crop3D_h, self.global_crop3D_w),
                interpolation=cv2.INTER_LINEAR
            )
            image_crop = image_crop[np.newaxis, :].transpose(0, 3, 1, 2)

        return image_crop


    def pad_image(self, img):
        """Pad an image up to the target size."""
        # Define the cols/rows needed in the padding
        rows_missing = math.ceil(self.global_crop3D_w - img.shape[0])
        cols_missing = math.ceil(self.global_crop3D_h - img.shape[1])
        dept_missing = math.ceil(self.global_crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        # pad the image
        depth_padding = (dept_missing // 2, dept_missing - dept_missing // 2)
        rows_padding = (rows_missing // 2, rows_missing - rows_missing // 2)
        cols_padding = (cols_missing // 2, cols_missing - cols_missing // 2)
        padded_img = np.pad(img, (rows_padding, cols_padding, depth_padding), 'constant')
        return padded_img

    def __getitem__(self, index):
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(datafiles["img"])
        image = imageNII.get_fdata()
        name = datafiles["name"]
        # You pad the image in the case is smaller to the patch size
        image = self.pad_image(image)
        image = image[np.newaxis, :]
        image = image.transpose((0, 3, 1, 2))

        img = []
        image_crop_ori = self.crop_scale(image, (1.4, 1.8), wise_crop=False)

        # Global patches, mirror and scale augmentations
        image_crop1 = self.crop_scale_mirror(image_crop_ori, axes=(0, 1, 2), local=False)
        image_crop2 = self.crop_scale_mirror(image_crop_ori, axes=(0, 1, 2), local=False)
        data_dict1 = {'image': image_crop1.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict2 = {'image': image_crop2.astype(np.float32).copy(), 'label': None, 'name': name}

        # Data augmentation (appart from mirroring) NO ROTATION APPLIED
        data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        data_dict2 = self.tr_transforms3D_global1(**data_dict2)

        img.append(data_dict1['image'])
        img.append(data_dict2['image'])

        # Local patches
        for _ in range(self.local_crops_number):
            image_crop_local = self.crop_scale_mirror(image_crop_ori, axes=(0, 1, 2), local=True)
            data_dict_local = {
                'image': image_crop_local.astype(np.float32).copy(),
                'label': None,
                'name': name
            }
            data_dict_local = self.tr_transforms3D_local(**data_dict_local)
            img.append(data_dict_local['image'])

        return img


def get_train_transform3D(gblur_p: float, brightm_p: float, bright_p: float, gamma_p: float):
    tr_transforms = []
    tr_transforms.append(GaussianNoiseTransform(data_key="image"))
    tr_transforms.append(
        GaussianBlurTransform(blur_sigma=(0.1, 2.), different_sigma_per_channel=True,
                              p_per_channel=gblur_p, p_per_sample=0.8, data_key="image"))
    tr_transforms.append(
        BrightnessMultiplicativeTransform((0.75, 1.25), p_per_sample=brightm_p, data_key="image"))
    tr_transforms.append(
        BrightnessTransform(0.0, 0.4, True, p_per_sample=bright_p, p_per_channel=0.8, data_key="image"))
    tr_transforms.append(ContrastAugmentationTransform(data_key="image"))
    tr_transforms.append(
        GammaTransform(gamma_range=(0.7, 1.5), invert_image=False, per_channel=True,
                       retain_stats=True, p_per_sample=gamma_p, data_key="image"))

    # now we compose these transforms together
    tr_transforms = Compose(tr_transforms)
    return tr_transforms
