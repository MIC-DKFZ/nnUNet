from typing import Union, List, Tuple, Callable

import numpy as np
from acvl_utils.morphology.morphology_helper import label_with_component_sizes
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from skimage.morphology import ball
from skimage.morphology.binary import binary_erosion, binary_dilation, binary_closing, binary_opening


class MoveSegAsOneHotToData(AbstractTransform):
    def __init__(self, index_in_origin: int, all_labels: Union[Tuple[int, ...], List[int]],
                 key_origin="seg", key_target="data", remove_from_origin=True):
        """
        Takes data_dict[seg][:, index_in_origin], converts it to one hot encoding and appends it to
        data_dict[key_target]. Optionally removes index_in_origin from data_dict[seg].
        """
        self.remove_from_origin = remove_from_origin
        self.all_labels = all_labels
        self.key_target = key_target
        self.key_origin = key_origin
        self.index_in_origin = index_in_origin

    def __call__(self, **data_dict):
        seg = data_dict[self.key_origin][:, self.index_in_origin:self.index_in_origin+1]

        seg_onehot = np.zeros((seg.shape[0], len(self.all_labels), *seg.shape[2:]),
                              dtype=data_dict[self.key_target].dtype)
        for i, l in enumerate(self.all_labels):
            seg_onehot[:, i][seg[:, 0] == l] = 1

        data_dict[self.key_target] = np.concatenate((data_dict[self.key_target], seg_onehot), 1)

        if self.remove_from_origin:
            remaining_channels = [i for i in range(data_dict[self.key_origin].shape[1]) if i != self.index_in_origin]
            data_dict[self.key_origin] = data_dict[self.key_origin][:, remaining_channels]

        return data_dict


class RemoveRandomConnectedComponentFromOneHotEncodingTransform(AbstractTransform):
    def __init__(self, channel_idx: Union[int, List[int]], key: str = "data", p_per_sample: float = 0.2,
                 fill_with_other_class_p: float = 0.25,
                 dont_do_if_covers_more_than_x_percent: float = 0.25, p_per_label: float = 1):
        """
        Randomly removes connected components in the specified channel_idx of data_dict[key]. Only considers components
        smaller than dont_do_if_covers_more_than_X_percent of the sample. Also has the option of simulating
        misclassification as another class (fill_with_other_class_p)
        """
        self.p_per_label = p_per_label
        self.dont_do_if_covers_more_than_x_percent = dont_do_if_covers_more_than_x_percent
        self.fill_with_other_class_p = fill_with_other_class_p
        self.p_per_sample = p_per_sample
        self.key = key
        if not isinstance(channel_idx, (list, tuple)):
            channel_idx = [channel_idx]
        self.channel_idx = channel_idx

    def __call__(self, **data_dict):
        data = data_dict.get(self.key)
        for b in range(data.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                for c in self.channel_idx:
                    if np.random.uniform() < self.p_per_label:
                        # print(np.unique(data[b, c])) ## should be [0, 1]
                        workon = data[b, c].astype(bool)
                        if not np.any(workon):
                            continue
                        num_voxels = np.prod(workon.shape, dtype=np.uint64)
                        lab, component_sizes = label_with_component_sizes(workon.astype(bool))
                        if len(component_sizes) > 0:
                            valid_component_ids = [i for i, j in component_sizes.items() if j <
                                                   num_voxels*self.dont_do_if_covers_more_than_x_percent]
                            # print('RemoveRandomConnectedComponentFromOneHotEncodingTransform', c,
                            # np.unique(data[b, c]), len(component_sizes), valid_component_ids,
                            # len(valid_component_ids))
                            if len(valid_component_ids) > 0:
                                random_component = np.random.choice(valid_component_ids)
                                data[b, c][lab == random_component] = 0
                                if np.random.uniform() < self.fill_with_other_class_p:
                                    other_ch = [i for i in self.channel_idx if i != c]
                                    if len(other_ch) > 0:
                                        other_class = np.random.choice(other_ch)
                                        data[b, other_class][lab == random_component] = 1
        data_dict[self.key] = data
        return data_dict


class ApplyRandomBinaryOperatorTransform(AbstractTransform):
    def __init__(self,
                 channel_idx: Union[int, List[int], Tuple[int, ...]],
                 p_per_sample: float = 0.3,
                 any_of_these: Tuple[Callable] = (binary_dilation, binary_erosion, binary_closing, binary_opening),
                 key: str = "data",
                 strel_size: Tuple[int, int] = (1, 10),
                 p_per_label: float = 1):
        """
        Applies random binary operations (specified by any_of_these) with random ball size (radius is uniformly sampled
        from interval strel_size) to specified channels. Expects the channel_idx to correspond to a hone hot encoded
        segmentation (see for example MoveSegAsOneHotToData)
        """
        self.p_per_label = p_per_label
        self.strel_size = strel_size
        self.key = key
        self.any_of_these = any_of_these
        self.p_per_sample = p_per_sample

        if not isinstance(channel_idx, (list, tuple)):
            channel_idx = [channel_idx]
        self.channel_idx = channel_idx

    def __call__(self, **data_dict):
        for b in range(data_dict[self.key].shape[0]):
            if np.random.uniform() < self.p_per_sample:
                # this needs to be applied in random order to the channels
                np.random.shuffle(self.channel_idx)
                for c in self.channel_idx:
                    if np.random.uniform() < self.p_per_label:
                        operation = np.random.choice(self.any_of_these)
                        selem = ball(np.random.uniform(*self.strel_size))
                        workon = data_dict[self.key][b, c].astype(bool)
                        if not np.any(workon):
                            continue
                        # print(np.unique(workon))
                        res = operation(workon, selem).astype(data_dict[self.key].dtype)
                        # print('ApplyRandomBinaryOperatorTransform', c, operation, np.sum(workon), np.sum(res))
                        data_dict[self.key][b, c] = res

                        # if class was added, we need to remove it in ALL other channels to keep one hot encoding
                        # properties
                        other_ch = [i for i in self.channel_idx if i != c]
                        if len(other_ch) > 0:
                            was_added_mask = (res - workon) > 0
                            for oc in other_ch:
                                data_dict[self.key][b, oc][was_added_mask] = 0
                            # if class was removed, leave it at background
        return data_dict
