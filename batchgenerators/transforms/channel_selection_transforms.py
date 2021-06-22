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

import numpy as np
from warnings import warn
from batchgenerators.transforms.abstract_transforms import AbstractTransform


class DataChannelSelectionTransform(AbstractTransform):
    """Selects color channels from the batch and discards the others.

    Args:
        channels (list of int): List of channels to be kept.

    """

    def __init__(self, channels, data_key="data"):
        self.data_key = data_key
        self.channels = channels

    def __call__(self, **data_dict):
        data_dict[self.data_key] = data_dict[self.data_key][:, self.channels]
        return data_dict


class SegChannelSelectionTransform(AbstractTransform):
    """Segmentations may have more than one channel. This transform selects segmentation channels

    Args:
        channels (list of int): List of channels to be kept.

    """

    def __init__(self, channels, keep_discarded_seg=False, label_key="seg"):
        self.label_key = label_key
        self.channels = channels
        self.keep_discarded = keep_discarded_seg

    def __call__(self, **data_dict):
        seg = data_dict.get(self.label_key)

        if seg is None:
            warn("You used SegChannelSelectionTransform but there is no 'seg' key in your data_dict, returning "
                 "data_dict unmodified", Warning)
        else:
            if self.keep_discarded:
                discarded_seg_idx = [i for i in range(len(seg[0])) if i not in self.channels]
                data_dict['discarded_seg'] = seg[:, discarded_seg_idx]
            data_dict[self.label_key] = seg[:, self.channels]
        return data_dict


class SegChannelMergeTransform(AbstractTransform):
    """Merge selected channels of a onehot segmentation. Will merge into lowest index.

    Args:
        channels (list of int): List of channels to be merged.

    """

    def __init__(self, channels, keep_discarded_seg=False, label_key="seg", fill_value=1):
        self.label_key = label_key
        self.channels = sorted(channels)
        self.keep_discarded = keep_discarded_seg
        self.fill_value = fill_value

    def __call__(self, **data_dict):
        seg = data_dict.get(self.label_key)

        if seg is None:
            warn("You used SegChannelSelectionTransform but there is no 'seg' key in your data_dict, returning data_dict unmodified", Warning)
        else:
            if self.keep_discarded:
                data_dict['discarded_seg'] = seg[:, self.channels[1:]]
            all_channels = list(range(seg.shape[1]))
            for i in self.channels[1:]:
                seg[:, self.channels[0]][seg[:, i] != 0] = self.fill_value
                all_channels.remove(i)
            data_dict[self.label_key] = seg[:, all_channels]
        return data_dict


class SegChannelRandomSwapTransform(AbstractTransform):
    """Randomly swap two segmentation channels.

    Args:
        axis1 (int): First axis for swap
        axis2 (int): Second axis for swap
        swap_probability (float): Probability for swap

    """

    def __init__(self, axis1, axis2, swap_probability=0.5, label_key="seg"):
        self.axis1 = axis1
        self.axis2 = axis2
        self.swap_probability = swap_probability
        self.label_key = label_key

    def __call__(self, **data_dict):
        seg = data_dict.get(self.label_key)

        if seg is None:
            warn("You used SegChannelSelectionTransform but there is no 'seg' key in your data_dict, returning "
                 "data_dict unmodified", Warning)
        else:
            random_number = np.random.rand()
            if random_number < self.swap_probability:
                seg[:, [self.axis1, self.axis2]] = seg[:, [self.axis2, self.axis1]]
            data_dict[self.label_key] = seg
        return data_dict


class SegChannelRandomDuplicateTransform(AbstractTransform):
    """Creates an additional seg channel full of zeros and randomly swaps it with the base channel.

    Args:
        axis (int): Axis to be duplicated
        swap_probability (float): Probability for swap

    """

    def __init__(self, axis, swap_probability=0.5, label_key="seg"):
        self.axis = axis
        self.swap_probability = swap_probability
        self.label_key = label_key

    def __call__(self, **data_dict):
        seg = data_dict.get(self.label_key)

        if seg is None:
            warn("You used SegChannelSelectionTransform but there is no 'seg' key in your data_dict, returning "
                 "data_dict unmodified", Warning)
        else:
            seg_shape = list(seg.shape)
            seg_shape[1] = 1
            seg = np.concatenate([seg, np.zeros(seg_shape, dtype=seg.dtype)], 1)
            random_number = np.random.rand()
            if random_number < self.swap_probability:
                seg[:, [self.axis, -1]] = seg[:, [-1, self.axis]]
            data_dict[self.label_key] = seg
        return data_dict


class SegLabelSelectionBinarizeTransform(AbstractTransform):
    """Will create a binary segmentation, with the selected labels in the foreground.

    Args:
        label (int, list of int): Foreground label(s)

    """

    def __init__(self, label, label_key="seg"):
        self.label_key = label_key
        if isinstance(label, int):
            self.label = [label]
        else:
            self.label = sorted(label)

    def __call__(self, **data_dict):
        seg = data_dict.get(self.label_key)

        if seg is None:
            warn("You used SegLabelSelectionBinarizeTransform but there is no 'seg' key in your data_dict, returning "
                 "data_dict unmodified", Warning)
        else:
            discard_labels = set(np.unique(seg)) - set(self.label) - set([0])
            for label in discard_labels:
                seg[seg == label] = 0
            for label in self.label:
                seg[seg == label] = 1
            data_dict[self.label_key] = seg
        return data_dict
