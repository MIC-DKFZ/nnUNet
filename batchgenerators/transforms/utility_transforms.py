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

import copy
from typing import List, Type

import numpy as np

from batchgenerators.augmentations.utils import convert_seg_image_to_one_hot_encoding, \
    convert_seg_to_bounding_box_coordinates, transpose_channels
from batchgenerators.transforms.abstract_transforms import AbstractTransform


class NumpyToTensor(AbstractTransform):
    def __init__(self, keys=None, cast_to=None):
        """Utility function for pytorch. Converts data (and seg) numpy ndarrays to pytorch tensors
        :param keys: specify keys to be converted to tensors. If None then all keys will be converted
        (if value id np.ndarray). Can be a key (typically string) or a list/tuple of keys
        :param cast_to: if not None then the values will be cast to what is specified here. Currently only half, float
        and long supported (use string)
        """
        if keys is not None and not isinstance(keys, (list, tuple)):
            keys = [keys]
        self.keys = keys
        self.cast_to = cast_to

    def cast(self, tensor):
        if self.cast_to is not None:
            if self.cast_to == 'half':
                tensor = tensor.half()
            elif self.cast_to == 'float':
                tensor = tensor.float()
            elif self.cast_to == 'long':
                tensor = tensor.long()
            else:
                raise ValueError('Unknown value for cast_to: %s' % self.cast_to)
        return tensor

    def __call__(self, **data_dict):
        import torch

        if self.keys is None:
            for key, val in data_dict.items():
                if isinstance(val, np.ndarray):
                    data_dict[key] = self.cast(torch.from_numpy(val)).contiguous()
                elif isinstance(val, (list, tuple)) and all([isinstance(i, np.ndarray) for i in val]):
                    data_dict[key] = [self.cast(torch.from_numpy(i)).contiguous() for i in val]
        else:
            for key in self.keys:
                if isinstance(data_dict[key], np.ndarray):
                    data_dict[key] = self.cast(torch.from_numpy(data_dict[key])).contiguous()
                elif isinstance(data_dict[key], (list, tuple)) and all([isinstance(i, np.ndarray) for i in data_dict[key]]):
                    data_dict[key] = [self.cast(torch.from_numpy(i)).contiguous() for i in data_dict[key]]

        return data_dict



class ListToNumpy(AbstractTransform):
    """Utility function for pytorch. Converts data (and seg) numpy ndarrays to pytorch tensors
    """

    def __call__(self, **data_dict):

        for key, val in data_dict.items():
            if isinstance(val, (list, tuple)):
                data_dict[key] = np.asarray(val)

        return data_dict


class ListToTensor(AbstractTransform):
    """Utility function for pytorch. Converts data (and seg) numpy ndarrays to pytorch tensors
    """

    def __call__(self, **data_dict):
        import torch

        for key, val in data_dict.items():
            if isinstance(val, (list, tuple)):
                data_dict[key] = [torch.from_numpy(smpl) for smpl in val]

        return data_dict


class ConvertSegToOnehotTransform(AbstractTransform):
    """Creates a one hot encoding of one of the seg channels. Important when using our soft dice loss.

    Args:
        classes (tuple of int): All the class labels that are in the dataset

        seg_channel (int): channel of seg to convert to onehot

        output_key (string): key to use for output of the one hot encoding. Default is 'seg' but that will override any
        other existing seg channels. Therefore you have the option to change that. BEWARE: Any non-'seg' segmentations
        will not be augmented anymore. Use this only at the very end of your pipeline!
    """

    def __init__(self, classes, seg_channel=0, output_key="seg"):
        self.output_key = output_key
        self.seg_channel = seg_channel
        self.classes = classes

    def __call__(self, **data_dict):
        seg = data_dict.get("seg")
        if seg is not None:
            new_seg = np.zeros([seg.shape[0], len(self.classes)] + list(seg.shape[2:]), dtype=seg.dtype)
            for b in range(seg.shape[0]):
                new_seg[b] = convert_seg_image_to_one_hot_encoding(seg[b, self.seg_channel], self.classes)
            data_dict[self.output_key] = new_seg
        else:
            from warnings import warn
            warn("calling ConvertSegToOnehotTransform but there is no segmentation")

        return data_dict


class ConvertMultiSegToOnehotTransform(AbstractTransform):
    """Regular onehot conversion, but for each channel in the input seg."""

    def __init__(self, classes):
        self.classes = classes

    def __call__(self, **data_dict):
        seg = data_dict.get("seg")
        if seg is not None:
            new_seg = np.zeros([seg.shape[0], len(self.classes) * seg.shape[1]] + list(seg.shape[2:]), dtype=seg.dtype)
            for b in range(seg.shape[0]):
                for c in range(seg.shape[1]):
                    new_seg[b, c*len(self.classes):(c+1)*len(self.classes)] = convert_seg_image_to_one_hot_encoding(seg[b, c], self.classes)
            data_dict["seg"] = new_seg
        else:
            from warnings import warn
            warn("calling ConvertMultiSegToOnehotTransform but there is no segmentation")

        return data_dict


class ConvertSegToArgmaxTransform(AbstractTransform):
    """Apply argmax to segmentation. Intended to be used with onehot segmentations.

    Args:
        labels (list or tuple for int): Label values corresponding to onehot indices. Assumed to be sorted.
        keepdim (bool): Whether to keep the reduced axis with size 1
    """

    def __init__(self, labels=None, keepdim=True):
        self.keepdim = keepdim
        self.labels = labels

    def __call__(self, **data_dict):
        seg = data_dict.get("seg")
        if seg is not None:
            n_labels = seg.shape[1]
            seg = np.argmax(seg, 1)
            if self.keepdim:
                seg = np.expand_dims(seg, 1)
            if self.labels is not None:
                if list(self.labels) != list(range(n_labels)):
                    for index, value in enumerate(reversed(self.labels)):
                        index = n_labels - index - 1
                        seg[seg == index] = value
            data_dict["seg"] = seg
        else:
            from warnings import warn
            warn("Calling ConvertSegToArgmaxTransform but there is no segmentation")

        return data_dict


class ConvertMultiSegToArgmaxTransform(AbstractTransform):
    """Apply argmax to segmentation. This is designed to reduce a onehot seg to one with multiple channels.

    Args:
        output_channels (int): Output segmentation will have this many channels.
            It is required that output_channels evenly divides the number of channels in the input.
        labels (list or tuple for int): Label values corresponding to onehot indices. Assumed to be sorted.
    """

    def __init__(self, output_channels=1, labels=None):
        self.output_channels = output_channels
        self.labels = labels

    def __call__(self, **data_dict):
        seg = data_dict.get("seg")
        if seg is not None:
            if not seg.shape[1] % self.output_channels == 0:
                from warnings import warn
                warn("Calling ConvertMultiSegToArgmaxTransform but number of input channels {} cannot be divided into {} output channels.".format(seg.shape[1], self.output_channels))
            n_labels = seg.shape[1] // self.output_channels
            target_size = list(seg.shape)
            target_size[1] = self.output_channels
            output = np.zeros(target_size, dtype=seg.dtype)
            for i in range(self.output_channels):
                output[:, i] = np.argmax(seg[:, i*n_labels:(i+1)*n_labels], 1)
            if self.labels is not None:
                if list(self.labels) != list(range(n_labels)):
                    for index, value in enumerate(reversed(self.labels)):
                        index = n_labels - index - 1
                        output[output == index] = value
            data_dict["seg"] = output
        else:
            from warnings import warn
            warn("Calling ConvertMultiSegToArgmaxTransform but there is no segmentation")

        return data_dict


class ConvertSegToBoundingBoxCoordinates(AbstractTransform):
    """ Converts segmentation masks into bounding box coordinates.
    """

    def __init__(self, dim, get_rois_from_seg_flag=False, class_specific_seg_flag=False):
        self.dim = dim
        self.get_rois_from_seg_flag = get_rois_from_seg_flag
        self.class_specific_seg_flag = class_specific_seg_flag

    def __call__(self, **data_dict):
        data_dict = convert_seg_to_bounding_box_coordinates(data_dict, self.dim, self.get_rois_from_seg_flag, class_specific_seg_flag=self.class_specific_seg_flag)
        return data_dict


class MoveSegToDataChannel(AbstractTransform):
    """
    concatenates data_dict['seg'] to data_dict['data']
    """
    def __call__(self, **data_dict):
        data_dict['data'] = np.concatenate((data_dict['data'], data_dict['seg']), axis=1)
        return data_dict


class ColorChannelToLastAxisTransform(AbstractTransform):
    """
    moves the color channel to the last axis
    For example:
    shape (b, c, x, y, z) -> shape (b, x, y, z, c)
    """

    def __call__(self, **data_dict):
        data_dict['data'] = transpose_channels(data_dict['data'])
        data_dict['seg'] = transpose_channels(data_dict['seg'])

        return data_dict


class RemoveLabelTransform(AbstractTransform):
    '''
    Replaces all pixels in data_dict[input_key] that have value remove_label with replace_with and saves the result to
    data_dict[output_key]
    '''

    def __init__(self, remove_label, replace_with=0, input_key="seg", output_key="seg"):
        self.output_key = output_key
        self.input_key = input_key
        self.replace_with = replace_with
        self.remove_label = remove_label

    def __call__(self, **data_dict):
        seg = data_dict[self.input_key]
        seg[seg == self.remove_label] = self.replace_with
        data_dict[self.output_key] = seg
        return data_dict


class RenameTransform(AbstractTransform):
    '''
    Saves the value of data_dict[in_key] to data_dict[out_key]. Optionally removes data_dict[in_key] from the dict.
    '''

    def __init__(self, in_key, out_key, delete_old=False):
        self.delete_old = delete_old
        self.out_key = out_key
        self.in_key = in_key

    def __call__(self, **data_dict):
        data_dict[self.out_key] = data_dict[self.in_key]
        if self.delete_old:
            del data_dict[self.in_key]
        return data_dict


class CopyTransform(AbstractTransform):
    """Renames some attributes of the data_dict (e. g. transformations can be applied on different dict items).

    Args:
        re_dict: Dict with the key=origin name, val=new name.
        copy: Copy (and not move (cp vs mv)) to new target val and leave the old ones in place

    Example:
        >>> transforms.CopyTransform({"data": "data2", "seg": "seg2"})
    """

    def __init__(self, re_dict, copy=False):
        self.re_dict = re_dict
        self.copy = copy

    def __call__(self, **data_dict):
        new_dict = {}
        for key, val in data_dict.items():
            if key in self.re_dict:
                n_key = self.re_dict[key]
                if isinstance(n_key, (tuple, list)):
                    for k in n_key:
                        if self.copy:
                            new_dict[k] = copy.deepcopy(val)
                        else:
                            new_dict[k] = val
                else:
                    if self.copy:
                        new_dict[n_key] = copy.deepcopy(val)
                    else:
                        new_dict[n_key] = val
            if key not in self.re_dict:
                new_dict[key] = val

            if self.copy:
                new_dict[key] = copy.deepcopy(val)

        return new_dict

    def __repr__(self):
        return str(type(self).__name__) + " ( " + repr(self.transforms) + " )"


class ReshapeTransform(AbstractTransform):

    def __init__(self, new_shape, key="data"):
        self.key = key
        self.new_shape = new_shape

    def __call__(self, **data_dict):

        data_arr = data_dict[self.key]
        data_shape = data_arr.shape
        c, h, w = data_shape[-3:]

        target_shape = []
        for val in self.new_shape:
            if val == "c":
                target_shape.append(c)
            elif val == "h":
                target_shape.append(h)
            elif val == "w":
                target_shape.append(w)
            else:
                target_shape.append(val)

        data_dict[self.key] = np.reshape(data_dict[self.key], target_shape)

        return data_dict


class AddToDictTransform(AbstractTransform):
    '''
    Add a value of data_dict[key].
    '''

    def __init__(self, in_key, in_val, strict=False):
        self.strict = strict
        self.in_val = in_val
        self.in_key = in_key

    def __call__(self, **data_dict):
        if self.in_key not in data_dict or self.strict:
            data_dict[self.in_key] = self.in_val
        return data_dict


class AppendChannelsTransform(AbstractTransform):
    def __init__(self, input_key, output_key, channel_indexes, remove_from_input=True):
        """
        Moves channels specified by channel_indexes from input_key in data_dict to output_key (by appending in the
        order specified in channel_indexes). The channels will be removed from input if remove_from_input is True
        :param input_key:
        :param output_key:
        :param channel_indexes: must be tuple or list
        :param remove_from_input:
        """
        self.remove_from_input = remove_from_input
        self.channel_indexes = channel_indexes
        self.output_key = output_key
        self.input_key = input_key
        assert isinstance(self.channel_indexes, (tuple, list)), "channel_indexes must be either tuple or list of int"

    def __call__(self, **data_dict):
        inp = data_dict.get(self.input_key)
        outp = data_dict.get(self.output_key)

        assert inp is not None, "input_key %s is not present in data_dict" % self.input_key

        selected_channels = inp[:, self.channel_indexes]

        if outp is None:
            #warn("output key %s is not present in dict, it will be created" % self.output_key)
            outp = selected_channels
            data_dict[self.output_key] = outp
        else:
            outp = np.concatenate((outp, selected_channels), axis=1)
            data_dict[self.output_key] = outp

        if self.remove_from_input:
            remaining = [i for i in range(inp.shape[1]) if i not in self.channel_indexes]
            inp = inp[:, remaining]
            data_dict[self.input_key] = inp

        return data_dict


class ConvertToChannelLastTransform(AbstractTransform):
    def __init__(self, input_keys):
        """
        converts all keys listed in input_keys from (b, c, x, y(, z)) to (b, x, y(, z), c).
        """
        self.input_keys = input_keys

    def __call__(self, **data_dict):
        for k in self.input_keys:
            data = data_dict.get(k)
            if data is None:
                print("WARNING in ConvertToChannelLastTransform: data_dict has no key named", k)
            else:
                if len(data.shape) == 4:
                    new_ordering = (0, 2, 3, 1)
                elif len(data.shape) == 5:
                    new_ordering = (0, 2, 3, 4, 1)
                else:
                    raise RuntimeError("unsupported dimensionality for ConvertToChannelLastTransform:",
                                       len(data.shape),
                                       ". Only 2d (b, c, x, y) and 3d (b, c, x, y, z) are supported for now.")
                assert isinstance(data, np.ndarray), "data_dict[k] must be a numpy array"
                data = data.transpose(new_ordering)
                data_dict[k] = data
        return data_dict


class OneOfTransform(AbstractTransform):
    def __init__(self, list_of_transforms: List):
        """
        Randomly selects one of the transforms given in list_of_transforms and applies it with each call. Remember that
        probabilities of the individual transforms for being applied still exist and apply!
        :param list_of_transforms:
        """
        self.list_of_transforms = list_of_transforms

    def __call__(self, **data_dict):
        i = np.random.choice(len(self.list_of_transforms))
        return self.list_of_transforms[i](**data_dict)
