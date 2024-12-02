import inspect
import multiprocessing
import os
import shutil
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Tuple, Union, List

import abc
import numpy as np
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from torch import autocast, nn
from torch import distributed as dist
from torch._dynamo import OptimizedModule
from torch.cuda import device_count
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.functional import interpolate

class Convert3DTo2DDistTransform(Convert3DTo2DTransform):
    def apply(self, data_dict, **params):
        if 'dist_map' in data_dict.keys():
            data_dict['nchannels_dist_map'] = deepcopy(data_dict['dist_map']).shape[0]
        return super().apply(data_dict, **params)

    def _apply_to_dist_map(self, dist_map: torch.Tensor, **params) -> torch.Tensor:
        return self._apply_to_image(dist_map, **params)

class Convert2DTo3DDistTransform(Convert2DTo3DTransform):
    def get_parameters(self, **data_dict) -> dict:
        return {i: data_dict[i] for i in
                ['nchannels_img', 'nchannels_seg', 'nchannels_regr_trg', 'nchannels_dist_map']
                if i in data_dict.keys()}

    def apply(self, data_dict, **params):
        if 'nchannels_dist_map' in data_dict.keys():
            del data_dict['nchannels_dist_map']
        return super().apply(data_dict, **params)

class SpatialDistTransform(SpatialTransform):
    def _apply_to_dist_map(self, dist_map, **params) -> torch.Tensor:
        return self._apply_to_image(dist_map, **params)

class MirrorDistTransform(MirrorTransform):
    def _apply_to_dist_map(self, dist_map: torch.Tensor, **params) -> torch.Tensor:
        if len(params['axes']) == 0:
            return dist_map
        axes = [i + 1 for i in params['axes']]
        return torch.flip(dist_map, axes)

class DownsampleSegForDSDistTransform(DownsampleSegForDSTransform):
    def _apply_to_dist_map(self, dist_map: torch.Tensor, **params) -> List[torch.Tensor]:
        results = []
        for s in self.ds_scales:
            if not isinstance(s, (tuple, list)):
                s = [s] * (dist_map.ndim - 1)
            else:
                assert len(s) == dist_map.ndim - 1

            if all([i == 1 for i in s]):
                results.append(dist_map)
            else:
                new_shape = [round(i * j) for i, j in zip(dist_map.shape[1:], s)]
                dtype = dist_map.dtype
                # interpolate is not defined for short etc
                results.append(interpolate(dist_map[None].float(), new_shape, mode='bilinear')[0].to(dtype))
        return results

if __name__ == '__main__':
    pass