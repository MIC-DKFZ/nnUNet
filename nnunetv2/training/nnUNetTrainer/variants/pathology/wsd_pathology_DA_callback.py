import numpy as np
from wholeslidedata.samplers.callbacks import BatchCallback
import time

from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform

from nnunetv2.training.data_augmentation.custom_transforms.pathology_transforms import HedTransform, HsvTransform, Clip01

class nnUnetBatchCallback(BatchCallback):
    
    # patch_size_spatial (width/height)
    def __init__(self, patch_size_spatial):
        tr_transforms = []
        rotation_for_DA= {'x': (-np.pi, np.pi), 'y': (0, 0), 'z': (0, 0)}

        tr_transforms.append(SpatialTransform(
            patch_size_spatial, 
            patch_center_dist_from_border=None,
            do_elastic_deform=False,
            alpha=(0, 0),
            sigma=(0, 0),
            do_rotation=True,
            angle_x=rotation_for_DA['x'],
            angle_y=rotation_for_DA['y'],
            angle_z=rotation_for_DA['z'],
            p_rot_per_axis=1,  # todo experiment with this
            do_scale=True,
            scale=(0.7, 1.4),
            border_mode_data="constant",
            border_cval_data=0,
            order_data=3,
            border_mode_seg="constant",
            border_cval_seg=-1,
            order_seg=1,
            random_crop=False,  # random cropping is part of our dataloaders
            p_el_per_sample=0,
            p_scale_per_sample=0.2,
            p_rot_per_sample=0.2,
            independent_scale_for_each_axis=False  # todo experiment with this
        ))

        ####
        if True: #do_hed:
            tr_transforms.append(HedTransform(factor=0.05))
        # if True: #do_hsv:
        #     tr_transforms.append(HsvTransform(h_lim=0.10, s_lim=0.10, v_lim=0.10))
        ####
        
        tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))

        tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                                   p_per_channel=0.5))

        tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))

        tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))

        tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                            p_per_channel=0.5,
                                                            order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                            ignore_axes=None))
 
        tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))

        tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))

        tr_transforms.append(Clip01())
        
        tr_transforms.append(MirrorTransform((0,1)))

        tr_transforms.append(RenameTransform('seg', 'target', True))
        
        
        self._transforms = Compose(tr_transforms)
    
    def __call__(self, x_batch, y_batch):
        # format to nnUNet
        x_batch = np.stack([x/255  for x in x_batch]).transpose((0, 3, 1, 2)).astype('float32')
        y_batch = np.expand_dims(np.stack(y_batch).astype('int8'), 1)
        
        # transform
        start_time = time.time()
        batch = self._transforms(**{'data': x_batch, 'seg': y_batch})
        line_time = time.time() - start_time
        print("Time taken for AUG (callback, multi thread):\t\t\t\t\t\t\t", line_time)
        
        # format back to wsd
        x_batch, y_batch = batch['data'], batch['target']
        x_batch = np.multiply(x_batch, 255).astype(np.uint8)
        return x_batch.transpose((0, 2, 3, 1)), y_batch.squeeze()
    

