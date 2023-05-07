import cv2
import json
import math

import numpy as np
import pandas as pd
import nibabel as nib
import multiprocessing as mp

from pathlib import Path
from typing import List, Dict
from torch.utils.data import Dataset

from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms import \
    BrightnessMultiplicativeTransform, GammaTransform, \
    BrightnessTransform, ContrastAugmentationTransform
from batchgenerators.transforms.noise_transforms import \
    GaussianNoiseTransform, GaussianBlurTransform
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.training.dataloading.utils import _convert_to_npy

DEFAULT_TRANSFORMS = {
    'global_scale': [0.8, 1.2], 'local_scale': [1.5, 2.0],
    'mirror': 0.8, 'symmetry': False,
    # sample, channel
    'i_global': {'gamma': [0.8, 1], 'g_noise': [1, 1], 'g_blur': [0.8, 0.8],
                 'mult_bright': [0.8, 1], 'brightness': [0.8, 0.8],
                 'contrast_augm': [1, 1], 'per_channel': True},
    'i_local': {'gamma': [1.0, 1], 'g_noise': [1, 1], 'g_blur': [0,8, 0.5],
                'mult_bright': [1.0, 1], 'brightness': [0.8, 0.5],
                'contrast_augm': [1, 1], 'per_channel': True}
}


class Dataset3D(Dataset):
    def __init__(
        self,
        dataset: str,
        train: bool,
        cfg: Dict
    ):
        self.dataset = dataset
        self.cfg = cfg
        self.rng = np.random.default_rng(420)

        self.global_crop_size = (16, 128, 128)
        self.local_crop_size = (16, 64, 64)
        self.local_crops_number = 0
        self.split_path = None
        self.fixed_slabs = True
        self.n_iters = None
        self.bboxes_path = None
        self.slab_thickness = None
        self.transformations_cfg = DEFAULT_TRANSFORMS
        self.wise_crop = False
        self.multichannel_input = False
        self.dataframe = pd.read_csv(self.cfg['all_datasets_csv_path'], index_col=0)

        fields = ['global_crop_size', 'local_crop_size', 'local_crops_number', 'split_path',
                  'fixed_slabs', 'n_iters', 'bboxes_path', 'slab_thickness', 'transformations_cfg',
                  'wise_crop', 'multichannel_input']
        for field in fields:
            if (field in self.cfg.keys()) and (self.cfg[field] is not None):
                setattr(self, field, self.cfg[field])
        
        self.n_iters = self.cfg['n_iters'] if train else self.cfg['n_val_iters']
        self.tr_cfg = self.transformations_cfg
        print('\ntr cfg:\n',self.tr_cfg)

        # Define whether symmetrical augmentation is going to be performed
        self.apply_sym_tr = False
        if ('symmetry' in self.tr_cfg.keys()) and (self.tr_cfg['symmetry'] is not None):
            self.apply_sym_tr = True

        print('\nsymmetry\n', self.apply_sym_tr)

        # get the splits and the data fingeprint fron nnUNet files
        preproc_path = Path(nnUNet_preprocessed)/dataset
        if self.split_path is None:
            self.split_path = preproc_path/'splits_final.json' 
        with open(self.split_path, 'r') as jfile:
            partition = 'train' if train else 'val'
            self.split_set = json.load(jfile)[0][partition]
        dataset_figerprint_path = preproc_path/'dataset_fingerprint.json'

        # based on the fingerprint, infer the background value
        self.bkgd_value = get_bkgd_value(dataset_figerprint_path)

        # get the list of files to use in ssl weird but is from previous code
        if self.fixed_slabs:
            self.list_path = preproc_path/'ssl_dataset.txt'
            self.img_paths = [Path(i_id.strip().split()[0]) for i_id in open(self.list_path)]
            self.img_paths = [
                ipath for ipath in self.img_paths if ipath.name.split('_')[0] in self.split_set]
            self.files = [{'img': item, 'name': item} for item in self.img_paths]
        else:
            # Get the image files from the nnUNet preprocessed files
            imgs_path = preproc_path / f'{self.cfg["exp_planner"]}_{self.cfg["configuration"]}'
            self.img_paths = [imgs_path/f'{i_id}.npy' for i_id in self.split_set]
            self.npz_paths = [imgs_path/f'{i_id}.npz' for i_id in self.split_set]
            if not all([ip.exists() for ip in self.img_paths]):
                if not all([ip.exists() for ip in self.npz_paths]):
                    missing = [ip for ip in self.npz_paths if (not ip.exists())]
                    raise Exception(f'Missing files in preprocessed data: {missing}')
                else:
                    with mp.Pool(mp.cpu_count()) as pool:
                        for _ in pool.imap(_convert_to_npy, [str(i) for i in self.npz_paths]):
                            pass
            if not all([ip.exists() for ip in self.img_paths]):
                missing = [ip for ip in self.img_paths if (not ip.exists())]
                raise Exception(f'Missing files in preprocessed data: {missing}')

            # Random sample the volumes to get as many images in the list as iterations
            if len(self.img_paths) < self.n_iters:
                missing = self.n_iters - len(self.img_paths)
                idxs = self.rng.integers(0, len(self.img_paths)-1, missing).tolist()
                self.img_paths = self.img_paths + [self.img_paths[idx] for idx in idxs]

            # If the brain images are not already cropped ones, read the bboxes_info file
            if not cfg['already_cropped']:
                with open(self.bboxes_path, 'r') as jfile:
                    self.bboxes = json.load(jfile)
            self.files = [{'img': item, 'name': item.name.replace('.npy', '')} for item in self.img_paths]

        # Define the transformations
        self.global_crop3D_d, self.global_crop3D_h, self.global_crop3D_w = self.global_crop_size
        self.local_crop3D_d, self.local_crop3D_h, self.local_crop3D_w = self.local_crop_size

        self.tr_transforms3D_global0 = get_train_transform3D(**self.tr_cfg['i_global'])
        self.tr_transforms3D_global1 = get_train_transform3D(**self.tr_cfg['i_global'])
        self.tr_transforms3D_local = get_train_transform3D(**self.tr_cfg['i_local'])
        
        print(f'SSL: {len(self.img_paths)} images are loaded!')

    def crop_scale(self, image, scale_range: tuple):
        _, img_d, img_h, img_w = image.shape

        # Define a random crop size that fits inside the image
        # on height and width, depth is kept fixed
        scaler_d = 1.
        scaler_h = self.rng.uniform(scale_range[0], scale_range[1])
        if (int(self.global_crop3D_h * scaler_h) >= img_h):
            scaler_h = 1.
        scaler_w = self.rng.uniform(scale_range[0], scale_range[1])
        if (int(self.global_crop3D_w * scaler_w) >= img_w):
            scaler_w = 1.
        scale_d = int(self.global_crop3D_d * scaler_d)
        scale_h = int(self.global_crop3D_h * scaler_h)
        scale_w = int(self.global_crop3D_w * scaler_w)

        # sample a random origin for the crop
        high = [img_d - scale_d, img_h - scale_h, img_w - scale_w]
        d0, h0, w0 = [self.rng.integers(0, h+1) for h in high]

        # define the limits of the crop
        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w

        # crop image
        image_crop = image[:, d0: d1, h0: h1, w0: w1]

        return image_crop, [d0, d1, h0, h1, w0, w1]

    def crop_scale_resize(self, image: np.ndarray, image_bis: np.ndarray, scale_range: tuple):
        image_crop, bbox = self.crop_scale(image, scale_range)
        image_crop_bis = None
        if image_bis is not None:
            image_crop_bis = image_bis[:, bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]

        # resize the crop to the original image size and add batch axis
        if (image_crop.shape != image.shape):
            image_crop = cv2.resize(
                image_crop[0].transpose(1, 2, 0),
                (self.global_crop3D_h, self.global_crop3D_w),
                interpolation=cv2.INTER_LINEAR
            )
            image_crop = image_crop[np.newaxis, :].transpose(0, 3, 1, 2)
            if image_bis is not None:
                image_crop_bis = cv2.resize(
                    image_crop_bis[0].transpose(1, 2, 0),
                    (self.global_crop3D_h, self.global_crop3D_w),
                    interpolation=cv2.INTER_LINEAR
                )
                image_crop_bis = image_crop_bis[np.newaxis, :].transpose(0, 3, 1, 2)
        return image_crop, image_crop_bis

    def mirror(self, image_crop: np.ndarray, image_crop_bis: np.ndarray,
               axes: tuple = (0, 1, 2), p: float = 0.8):
        # apply random mirroring on the three axes
        if 2 in axes and self.rng.uniform() < p:
            image_crop = image_crop[:, :, :, ::-1]
            if image_crop_bis is not None:
                image_crop_bis = image_crop_bis[:, :, :, ::-1]
        if 1 in axes and self.rng.uniform() < p:
            image_crop = image_crop[:, :, ::-1]
            if image_crop_bis is not None:
                image_crop_bis = image_crop_bis[:, :, ::-1]
        if 0 in axes and self.rng.uniform() < p:
            image_crop = image_crop[:, ::-1]
            if image_crop_bis is not None:
                image_crop_bis = image_crop_bis[:, ::-1]
        return image_crop, image_crop_bis

    def pad_image(self, img: np.ndarray, padd_val: int = 0) -> np.ndarray:
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
        pad_dims = (rows_padding, cols_padding, depth_padding)
        padded_img = np.pad(img, pad_dims, 'constant', constant_values=padd_val)
        return padded_img

    def crop_random_zslab_brain_vol(self, image: np.ndarray, image2: np.ndarray = None,
                                    wise_crop: bool = False) -> np.ndarray:
        if image2 is not None:
            assert image.shape == image2.shape, \
                'In crop zslab, images do not have the same shape: ' \
                f'{image.shape} != {image2.shape}'
        if wise_crop and (self.bkgd_value is None):
            raise Exception('If wise cropping needs to be done the background value yo must pass')
        
        z_shape = image.shape[2]
        top_hlf_slab = self.slab_thickness // 2
        down_hlf_slab = self.slab_thickness - top_hlf_slab
        count = 0
        while True:
            crop_center = self.rng.integers(down_hlf_slab, z_shape-1-top_hlf_slab)
            crop_end = crop_center + top_hlf_slab
            crop_start = crop_center - down_hlf_slab
            crop = image[:, :, crop_start:crop_end]
            if wise_crop:
                bkgd_ratio = (crop == self.bkgd_value).sum() / np.prod(crop.shape)
                # If more than half of the image is background
                if bkgd_ratio > 0.6 and (count < 10):
                    # resample if with a probability directly proportional to the 
                    # bkgd ratio, the more bkgd higher the chances to resample
                    sample = self.rng.uniform(low=0.0, high=1.0)
                    if (sample < bkgd_ratio):
                        count += 1
                    else:
                        break
                else:
                    break
            else:
                break
        image = crop
        if image2 is not None:
            image2 = image2[:, :, crop_start:crop_end]
            return image, image2
        return image

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        name = datafiles["name"]
        dataset_name = self.dataframe.loc[self.dataframe.subject == name, 'dataset_name'].values[0]
        ais = self.dataframe.loc[self.dataframe.subject == name, 'ais'].values[0]

        if self.apply_sym_tr and (self.rng.uniform() < self.tr_cfg['symmetry']):
            self.apply_sym = True
        else:
            self.apply_sym = False
        # read nii file
        if self.fixed_slabs:
            imageNII = nib.load(datafiles["img"])
            image = imageNII.get_fdata()
        else:
            if self.apply_sym:
                image = np.load(datafiles["img"])
                image2 = image[1].transpose(2, 1, 0)
                image = image[0].transpose(2, 1, 0)
            elif self.multichannel_input:
                image = np.load(datafiles["img"])
                image2 = image[1].transpose(2, 1, 0)
                image = image[0].transpose(2, 1, 0)
            else:
                image = np.load(datafiles["img"])[0]
                image = image.transpose(2, 1, 0)
            # if necessary crop brain volume:
            if not self.cfg['already_cropped']:
                bbox = self.bboxes[name]
                ox, oy, oz = bbox['origin_x'], bbox['origin_y'], bbox['origin_z']
                ex, ey, ez = bbox['end_x'], bbox['end_y'], bbox['end_z']
                image = image[oz:ez, oy:ey, ox:ex]
                if self.apply_sym or self.multichannel_input:
                    image2 = image2[oz:ez, oy:ey, ox:ex]
            # Crop a random slab from the 3d volume
            if self.apply_sym or self.multichannel_input:
                image, image2 = self.crop_random_zslab_brain_vol(image, image2,
                                                                 wise_crop=self.cfg['wise_crop'])
            else:
                image = self.crop_random_zslab_brain_vol(image, wise_crop=self.cfg['wise_crop'])

        # bm = if image, image2 = self.crop_random_zslab_brain_vol(image, image2)
        # You pad the image in the case is smaller to the patch size
        image = self.pad_image(image, self.bkgd_value)
        image = image[np.newaxis, :]
        image = image.transpose((0, 3, 1, 2))
        if self.apply_sym or self.multichannel_input:
            image2 = self.pad_image(image2)
            image2 = image2[np.newaxis, :]
            image2 = image2.transpose((0, 3, 1, 2))

        img, labels, dataset_names = [], [], []
        # Get transformations:
        
        # Sample the global crops from the same image or one from the original and one from the 
        # 'contralateral' that should be provided as another channel
        image_crop_ori1_bis, image_crop_ori2_bis = None, None
        if self.apply_sym:
            if self.rng.uniform() < 0.5:
                image_crop_ori1, bbox = self.crop_scale(image, (1.4, 1.8))
                image_crop_ori2 = image2[:, bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
            else:
                image_crop_ori1, bbox = self.crop_scale(image2, (1.4, 1.8))
                image_crop_ori2 = image[:, bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
        elif self.multichannel_input:
            image_crop_ori1, bbox = self.crop_scale(image, (1.4, 1.8))
            image_crop_ori1_bis = image2[:, bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
            image_crop_ori2 = image_crop_ori1.copy()
            image_crop_ori2_bis = image_crop_ori1_bis.copy()
        else:
            image_crop_ori1, _ = self.crop_scale(image, (1.4, 1.8))
            image_crop_ori2 = image_crop_ori1.copy()
        
        # Global patches scaling and mirror augmentations
        if ('global_scale' in self.tr_cfg.keys()) and (self.tr_cfg['global_scale'] is not None):
            image_crop1, image_crop1_bis = self.crop_scale_resize(image_crop_ori1,
                                                                  image_crop_ori1_bis,
                                                                  self.tr_cfg['global_scale'])
            image_crop2, image_crop2_bis = self.crop_scale_resize(image_crop_ori2,
                                                                  image_crop_ori2_bis,
                                                                  self.tr_cfg['global_scale'])
        else:
            image_crop1 = image_crop_ori1.copy()
            image_crop2 = image_crop_ori2.copy()
            image_crop1_bis = image_crop_ori1_bis.copy() if image_crop_ori1_bis is not None else None
            image_crop2_bis = image_crop_ori2_bis.copy() if image_crop_ori2_bis is not None else None

        if ('mirror' in self.tr_cfg.keys()) and (self.tr_cfg['mirror'] is not None):
            p = self.tr_cfg['mirror']
            image_crop1, image_crop1_bis = self.mirror(image_crop=image_crop1,
                                                       image_crop_bis=image_crop1_bis,
                                                       axes=(0, 1, 2), p=p)
            image_crop2, image_crop2_bis = self.mirror(image_crop=image_crop2,
                                                       image_crop_bis=image_crop2_bis,
                                                       axes=(0, 1, 2), p=p)
        if self.multichannel_input:
            image_crop1 = np.stack([image_crop1, image_crop1_bis], axis=0)
            image_crop2 = np.stack([image_crop2, image_crop1_bis], axis=0)
        data_dict1 = {'image': image_crop1.astype(np.float32).copy(),
                      'dataset': dataset_name, 'ais': ais, 'name': name}
        data_dict2 = {'image': image_crop2.astype(np.float32).copy(),
                      'dataset': dataset_name, 'ais': ais, 'name': name}

        # Data augmentation (appart from mirroring) NO ROTATION APPLIED
        data_dict1 = self.tr_transforms3D_global0(**data_dict1)
        data_dict2 = self.tr_transforms3D_global1(**data_dict2)
        
        img.append(data_dict1['image'])
        labels.append(data_dict1['ais'])
        dataset_names.append(data_dict1['dataset'])
        
        img.append(data_dict2['image'])
        labels.append(data_dict2['ais'])
        dataset_names.append(data_dict2['dataset'])

        # Local patches
        for _ in range(self.local_crops_number):
            image_crop_local, image_crop1_bis = self.crop_scale_resize(image_crop_ori1,
                                                                       image_crop_ori1_bis,
                                                                       self.tr_cfg['local_scale'])
            if ('mirror' in self.tr_cfg.keys()) and (self.tr_cfg['mirror'] is not None):
                p = self.tr_cfg['mirror']
                image_crop_local, image_crop_local_bis = self.mirror(image_crop=image_crop_local,
                                                                     image_crop_bis=image_crop_local_bis,
                                                                     axes=(0, 1, 2), p=p)
            if self.multichannel_input:
                image_crop1 = np.stack([image_crop_local, image_crop_local_bis], axis=0)
            data_dict_local = {
                'image': image_crop_local.astype(np.float32).copy(),
                'dataset': dataset_name,
                'ais': ais,
                'name': name
            }
            data_dict_local = self.tr_transforms3D_local(**data_dict_local)
            img.append(data_dict_local['image'])
            labels.append(data_dict_local['ais'])
            dataset_names.append(data_dict_local['dataset'])

        return img, labels, dataset_names


def get_train_transform3D(g_noise: List, g_blur: List, mult_bright: List, per_channel: bool,
                          brightness: List, gamma: List, contrast_augm: List):
    tr_transforms = []
    if g_noise is not None:
        tr_transforms.append(
            GaussianNoiseTransform(noise_variance=(0, 0.1), p_per_sample=g_noise[0],
                                   p_per_channel=g_noise[1], per_channel=per_channel,
                                   data_key="image"))
    if g_blur is not None:
        tr_transforms.append(
            GaussianBlurTransform(blur_sigma=(0.1, 2.), p_per_sample=g_blur[0],
                                  different_sigma_per_axis=False, p_isotropic=0,
                                  different_sigma_per_channel=per_channel,
                                  p_per_channel=g_blur[1], data_key="image"))
    if mult_bright is not None:
        tr_transforms.append(
            BrightnessMultiplicativeTransform((0.75, 1.25), per_channel=per_channel,
                                              p_per_sample=mult_bright[0], data_key="image"))
    if brightness is not None:
        tr_transforms.append(
            BrightnessTransform(mu=0.0, sigma=0.4, per_channel=per_channel,
                                p_per_sample=brightness[1], p_per_channel=brightness[0],
                                data_key="image"))
    if contrast_augm is not None:
        tr_transforms.append(
            ContrastAugmentationTransform(contrast_range=(0.75, 1.25), preserve_range=True,
                                          per_channel=per_channel, p_per_sample=contrast_augm[0],
                                          p_per_channel=contrast_augm[1], data_key="image"))
    if gamma is not None:
        tr_transforms.append(
            GammaTransform(gamma_range=(0.7, 1.5), invert_image=False, per_channel=per_channel,
                           retain_stats=True, p_per_sample=gamma[0], data_key="image"))

    # now we compose these transforms together
    tr_transforms = Compose(tr_transforms)
    return tr_transforms


def get_bkgd_value(dataset_fgrprint_path: Path) -> float:
    with open(dataset_fgrprint_path, 'r') as jfile:
        fgd_intensities = json.load(jfile)["foreground_intensity_properties_per_channel"]['0']
    
    bknd_intensity = fgd_intensities['min']
    mean_intensity = fgd_intensities['mean']
    std_intensity = fgd_intensities['std']
    lower_bound = fgd_intensities['percentile_00_5']
    bknd_intensity = lower_bound if bknd_intensity < lower_bound else bknd_intensity
    bknd_intensity = (bknd_intensity - mean_intensity) / max(std_intensity, 1e-8)
    return bknd_intensity