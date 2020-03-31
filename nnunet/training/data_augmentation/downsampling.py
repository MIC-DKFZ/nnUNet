#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import torch
from batchgenerators.augmentations.utils import convert_seg_image_to_one_hot_encoding_batched, resize_segmentation
from batchgenerators.transforms import AbstractTransform
from torch.nn.functional import avg_pool2d, avg_pool3d
import numpy as np


class DownsampleSegForDSTransform3(AbstractTransform):
    '''
    returns one hot encodings of the segmentation maps if downsampling has occured (no one hot for highest resolution)
    downsampled segmentations are smooth, not 0/1

    returns torch tensors, not numpy arrays!

    always uses seg channel 0!!

    you should always give classes! Otherwise weird stuff may happen
    '''
    def __init__(self, ds_scales=(1, 0.5, 0.25), input_key="seg", output_key="seg", classes=None):
        self.classes = classes
        self.output_key = output_key
        self.input_key = input_key
        self.ds_scales = ds_scales

    def __call__(self, **data_dict):
        data_dict[self.output_key] = downsample_seg_for_ds_transform3(data_dict[self.input_key][:, 0], self.ds_scales, self.classes)
        return data_dict


def downsample_seg_for_ds_transform3(seg, ds_scales=((1, 1, 1), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)), classes=None):
    output = []
    one_hot = torch.from_numpy(convert_seg_image_to_one_hot_encoding_batched(seg, classes)) # b, c,

    for s in ds_scales:
        if all([i == 1 for i in s]):
            output.append(torch.from_numpy(seg))
        else:
            kernel_size = tuple(int(1 / i) for i in s)
            stride = kernel_size
            pad = tuple((i-1) // 2 for i in kernel_size)

            if len(s) == 2:
                pool_op = avg_pool2d
            elif len(s) == 3:
                pool_op = avg_pool3d
            else:
                raise RuntimeError()

            pooled = pool_op(one_hot, kernel_size, stride, pad, count_include_pad=False, ceil_mode=False)

            output.append(pooled)
    return output


class DownsampleSegForDSTransform2(AbstractTransform):
    '''
    data_dict['output_key'] will be a list of segmentations scaled according to ds_scales
    '''
    def __init__(self, ds_scales=(1, 0.5, 0.25), order=0, cval=0, input_key="seg", output_key="seg", axes=None):
        self.axes = axes
        self.output_key = output_key
        self.input_key = input_key
        self.cval = cval
        self.order = order
        self.ds_scales = ds_scales

    def __call__(self, **data_dict):
        data_dict[self.output_key] = downsample_seg_for_ds_transform2(data_dict[self.input_key], self.ds_scales, self.order,
                                                           self.cval, self.axes)
        return data_dict


def downsample_seg_for_ds_transform2(seg, ds_scales=((1, 1, 1), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)), order=0, cval=0, axes=None):
    if axes is None:
        axes = list(range(2, len(seg.shape)))
    output = []
    for s in ds_scales:
        if all([i == 1 for i in s]):
            output.append(seg)
        else:
            new_shape = np.array(seg.shape).astype(float)
            for i, a in enumerate(axes):
                new_shape[a] *= s[i]
            new_shape = np.round(new_shape).astype(int)
            out_seg = np.zeros(new_shape, dtype=seg.dtype)
            for b in range(seg.shape[0]):
                for c in range(seg.shape[1]):
                    out_seg[b, c] = resize_segmentation(seg[b, c], new_shape[2:], order, cval)
            output.append(out_seg)
    return output
