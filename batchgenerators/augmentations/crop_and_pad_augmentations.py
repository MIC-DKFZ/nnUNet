# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from builtins import range
import numpy as np
from batchgenerators.augmentations.utils import pad_nd_image


def center_crop(data, crop_size, seg=None):
    return crop(data, seg, crop_size, 0, 'center')


def get_lbs_for_random_crop(crop_size, data_shape, margins):
    """

    :param crop_size:
    :param data_shape: (b,c,x,y(,z)) must be the whole thing!
    :param margins:
    :return:
    """
    lbs = []
    for i in range(len(data_shape) - 2):
        if data_shape[i+2] - crop_size[i] - margins[i] > margins[i]:
            lbs.append(np.random.randint(margins[i], data_shape[i+2] - crop_size[i] - margins[i]))
        else:
            lbs.append((data_shape[i+2] - crop_size[i]) // 2)
    return lbs


def get_lbs_for_center_crop(crop_size, data_shape):
    """
    :param crop_size:
    :param data_shape: (b,c,x,y(,z)) must be the whole thing!
    :return:
    """
    lbs = []
    for i in range(len(data_shape) - 2):
        lbs.append((data_shape[i + 2] - crop_size[i]) // 2)
    return lbs


def crop(data, seg=None, crop_size=128, margins=(0, 0, 0), crop_type="center",
         pad_mode='constant', pad_kwargs={'constant_values': 0},
         pad_mode_seg='constant', pad_kwargs_seg={'constant_values': 0}):
    """
    crops data and seg (seg may be None) to crop_size. Whether this will be achieved via center or random crop is
    determined by crop_type. Margin will be respected only for random_crop and will prevent the crops form being closer
    than margin to the respective image border. crop_size can be larger than data_shape - margin -> data/seg will be
    padded with zeros in that case. margins can be negative -> results in padding of data/seg followed by cropping with
    margin=0 for the appropriate axes

    :param data: b, c, x, y(, z)
    :param seg:
    :param crop_size:
    :param margins: distance from each border, can be int or list/tuple of ints (one element for each dimension).
    Can be negative (data/seg will be padded if needed)
    :param crop_type: random or center
    :return:
    """
    if not isinstance(data, (list, tuple, np.ndarray)):
        raise TypeError("data has to be either a numpy array or a list")

    data_shape = tuple([len(data)] + list(data[0].shape))
    data_dtype = data[0].dtype
    dim = len(data_shape) - 2

    if seg is not None:
        seg_shape = tuple([len(seg)] + list(seg[0].shape))
        seg_dtype = seg[0].dtype

        if not isinstance(seg, (list, tuple, np.ndarray)):
            raise TypeError("data has to be either a numpy array or a list")

        assert all([i == j for i, j in zip(seg_shape[2:], data_shape[2:])]), "data and seg must have the same spatial " \
                                                                             "dimensions. Data: %s, seg: %s" % \
                                                                             (str(data_shape), str(seg_shape))

    if type(crop_size) not in (tuple, list, np.ndarray):
        crop_size = [crop_size] * dim
    else:
        assert len(crop_size) == len(
            data_shape) - 2, "If you provide a list/tuple as center crop make sure it has the same dimension as your " \
                             "data (2d/3d)"

    if not isinstance(margins, (np.ndarray, tuple, list)):
        margins = [margins] * dim

    data_return = np.zeros([data_shape[0], data_shape[1]] + list(crop_size), dtype=data_dtype)
    if seg is not None:
        seg_return = np.zeros([seg_shape[0], seg_shape[1]] + list(crop_size), dtype=seg_dtype)
    else:
        seg_return = None

    for b in range(data_shape[0]):
        data_shape_here = [data_shape[0]] + list(data[b].shape)
        if seg is not None:
            seg_shape_here = [seg_shape[0]] + list(seg[b].shape)

        if crop_type == "center":
            lbs = get_lbs_for_center_crop(crop_size, data_shape_here)
        elif crop_type == "random":
            lbs = get_lbs_for_random_crop(crop_size, data_shape_here, margins)
        else:
            raise NotImplementedError("crop_type must be either center or random")

        need_to_pad = [[0, 0]] + [[abs(min(0, lbs[d])),
                                   abs(min(0, data_shape_here[d + 2] - (lbs[d] + crop_size[d])))]
                                  for d in range(dim)]

        # we should crop first, then pad -> reduces i/o for memmaps, reduces RAM usage and improves speed
        ubs = [min(lbs[d] + crop_size[d], data_shape_here[d+2]) for d in range(dim)]
        lbs = [max(0, lbs[d]) for d in range(dim)]

        slicer_data = [slice(0, data_shape_here[1])] + [slice(lbs[d], ubs[d]) for d in range(dim)]
        data_cropped = data[b][tuple(slicer_data)]

        if seg_return is not None:
            slicer_seg = [slice(0, seg_shape_here[1])] + [slice(lbs[d], ubs[d]) for d in range(dim)]
            seg_cropped = seg[b][tuple(slicer_seg)]

        if any([i > 0 for j in need_to_pad for i in j]):
            data_return[b] = np.pad(data_cropped, need_to_pad, pad_mode, **pad_kwargs)
            if seg_return is not None:
                seg_return[b] = np.pad(seg_cropped, need_to_pad, pad_mode_seg, **pad_kwargs_seg)
        else:
            data_return[b] = data_cropped
            if seg_return is not None:
                seg_return[b] = seg_cropped

    return data_return, seg_return


def random_crop(data, seg=None, crop_size=128, margins=[0, 0, 0]):
    return crop(data, seg, crop_size, margins, 'random')


def pad_nd_image_and_seg(data, seg, new_shape=None, must_be_divisible_by=None, pad_mode_data='constant',
                         np_pad_kwargs_data=None, pad_mode_seg='constant', np_pad_kwargs_seg=None):
    """
    Pads data and seg to new_shape. new_shape is thereby understood as min_shape (if data/seg is already larger then
    new_shape the shape stays the same for the dimensions this applies)
    :param data:
    :param seg:
    :param new_shape: if none then only must_be_divisible_by is applied
    :param must_be_divisible_by: UNet like architectures sometimes require the input to be divisibly by some number. This
    will modify new_shape if new_shape is not divisibly by this (by increasing it accordingly).
    must_be_divisible_by should be a list of int (one for each spatial dimension) and this list must have the same
    length as new_shape
    :param pad_mode_data: see np.pad
    :param np_pad_kwargs_data:see np.pad
    :param pad_mode_seg:see np.pad
    :param np_pad_kwargs_seg:see np.pad
    :return:
    """
    sample_data = pad_nd_image(data, new_shape, mode=pad_mode_data, kwargs=np_pad_kwargs_data,
                               return_slicer=False, shape_must_be_divisible_by=must_be_divisible_by)
    if seg is not None:
        sample_seg = pad_nd_image(seg, new_shape, mode=pad_mode_seg, kwargs=np_pad_kwargs_seg,
                                  return_slicer=False, shape_must_be_divisible_by=must_be_divisible_by)
    else:
        sample_seg = None
    return sample_data, sample_seg
