from time import time
from typing import Union, List

import numpy as np
import torch


class LabelManager(object):
    def __init__(self, dataset_json: dict, force_use_labels: bool = False):
        self.dataset_json = dataset_json
        self._force_use_labels = force_use_labels

        if force_use_labels:
            self._has_regions = False
        else:
            self._has_regions: bool = any([isinstance(i, (tuple, list)) and len(i) > 1 for i in self.dataset_json['labels'].values()])

        self._ignore_label: Union[None, int] = self._determine_ignore_label()
        self._all_labels: List[int] = self._get_all_labels()

        self._regions: Union[None, List[Union[int, tuple[int, ...]]]] = self._get_regions()

    def _get_all_labels(self) -> List[int]:
        all_labels = []
        for k, r in self.dataset_json['labels'].items():
            # ignore label is not going to be used, hence the name. Duh.
            if k == 'ignore':
                continue
            if isinstance(r, (tuple, list)):
                for ri in r:
                    all_labels.append(int(ri))
            else:
                all_labels.append(int(r))
        all_labels = list(np.unique(all_labels))
        all_labels.sort()
        return all_labels

    def _get_regions(self) -> Union[None, List[Union[int, tuple[int, ...]]]]:
        if not self._has_regions or self._force_use_labels:
            return None
        else:
            assert 'regions_class_order' in self.dataset_json.keys(), 'if region-based training is requested via ' \
                                                                 'dataset.json then you need to define ' \
                                                                 'regions_class_order as well, ' \
                                                                 'see documentation!'  # TODO add this
            regions = []
            for k, r in self.dataset_json['labels'].items():
                # ignore ignore label
                if k == 'ignore':
                    continue
                # ignore regions that are background
                if (isinstance(r, int) and r == 0) \
                        or \
                        (isinstance(r, (tuple, list)) and len(np.unique(r)) == 1 and np.unique(r)[0] == 0):
                    continue
                if isinstance(r, list):
                    r = tuple(r)
                regions.append(r)
            assert len(self.dataset_json['regions_class_order']) == len(regions), 'regions_class_order must have as ' \
                                                                             'many entries as there are ' \
                                                                             'regions'
            return regions

    def _determine_ignore_label(self) -> Union[None, int]:
        ignore_label = self.dataset_json['labels'].get('ignore')
        if ignore_label is not None:
            assert isinstance(ignore_label, int), f'Ignore label has to be an integer. It cannot be a region ' \
                                                  f'(list/tuple). Got {type(ignore_label)}.'
        return ignore_label

    @property
    def has_regions(self) -> bool:
        return self._has_regions

    @property
    def all_regions(self) -> Union[None, List[Union[int, tuple[int, ...]]]]:
        return self._regions

    @property
    def all_labels(self) -> List[int]:
        return self._all_labels

    @property
    def ignore_label(self) -> Union[None, int]:
        return self._ignore_label

    def convert_logits_to_segmentation(self, predicted_probabilities: Union[np.ndarray, torch.Tensor]) -> \
            Union[np.ndarray, torch.Tensor]:
        """
        assumes that inference_nonlinearity was already applied!

        predicted_probabilities has to have shape (c, x, y(, z)) where c is the number of classes/regions
        """
        if not isinstance(predicted_probabilities, (np.ndarray, torch.Tensor)):
            raise RuntimeError(f"Unexpected input type. Expected np.ndarray or torch.Tensor,"
                               f" got {type(predicted_probabilities)}")

        # check correct number of outputs
        assert predicted_probabilities.shape[0] == self.num_segmentation_heads, \
            f'unexpected number of channels in predicted_probabilities. Expected {self.num_segmentation_heads}, ' \
            f'got {predicted_probabilities.shape[0]}. Remeber that predicted_probabilities should have shape ' \
            f'(c, x, y(, z)).'

        if self.has_regions:
            regions_class_order = self.dataset_json['regions_class_order']
            if isinstance(predicted_probabilities, np.ndarray):
                segmentation = np.zeros(predicted_probabilities.shape[1:], dtype=np.uint8)
            else:
                segmentation = torch.zeros(predicted_probabilities.shape[1:], dtype=torch.uint8,
                                           device=predicted_probabilities.device)
            for i, c in enumerate(regions_class_order):
                segmentation[predicted_probabilities[i] > 0.5] = c
        else:
            segmentation = predicted_probabilities.argmax(0)

        return segmentation

    @staticmethod
    def filter_background(classes_or_regions: Union[List[int], List[Union[int, tuple[int, ...]]]]):
        # heck yeah
        # This is definitely taking list comprehension too far. Enjoy.
        return [i for i in classes_or_regions if
                ((not isinstance(i, (tuple, list))) and i != 0)
                or
                (isinstance(i, (tuple, list)) and not (
                        len(np.unique(i)) == 1 and np.unique(i)[0] != 0))]

    @property
    def foreground_regions(self):
        return self.filter_background(self.all_regions)

    @property
    def foreground_labels(self):
        return self.filter_background(self.all_labels)

    @property
    def num_segmentation_heads(self):
        if self.has_regions:
            return len(self.foreground_regions)
        else:
            return len(self.all_labels)


def convert_labelmap_to_one_hot(segmentation: Union[np.ndarray, torch.Tensor],
                                all_labels: Union[List, torch.Tensor, np.ndarray, tuple],
                                output_dtype=None) -> Union[np.ndarray, torch.Tensor]:
    """
    if output_dtype is None then we use np.uint8/torch.uint8
    if input is torch.Tensor then output will be on the same device

    np.ndarray is faster than torch.Tensor

    if segmentation is torch.Tensor, this function will be faster if it is LongTensor. If it is somethine else we have
    to cast which takes time.

    IMPORTANT: This function only works properly if your labels are consecutive integers, so something like 0, 1, 2, 3, ...
    DO NOT use it with 0, 32, 123, 255, ... or whatever (fix your labels, yo)
    """
    if isinstance(segmentation, torch.Tensor):
        result = torch.zeros((len(all_labels), *segmentation.shape),
                             dtype=output_dtype if output_dtype is not None else torch.uint8,
                             device=segmentation.device)
        # variant 1, 2x faster than 2
        result.scatter_(0, segmentation[None].long(), 1)  # why does this have to be long!?
        # variant 2, slower than 1
        # for i, l in enumerate(all_labels):
        #     result[i] = segmentation == l
    else:
        result = np.zeros((len(all_labels), *segmentation.shape),
                          dtype=output_dtype if output_dtype is not None else np.uint8)
        # variant 1, fastest in my testing
        for i, l in enumerate(all_labels):
            result[i] = segmentation == l
        # variant 2. Takes about twice as long so nah
        # result = np.eye(len(all_labels))[segmentation].transpose((3, 0, 1, 2))
    return result


def determine_num_input_channels(plans: dict, configuration: str, dataset_json: dict,
                                 label_manager: LabelManager = None) -> int:
    """
    if label_manager is None we create one from dataset_json. Not recommended.
    """
    if label_manager is None:
        label_manager = LabelManager(dataset_json)

    # cascade has different number of input channels
    if 'previous_stage' in plans['configurations'][configuration].keys():
        if label_manager.has_regions:
            raise NotImplemented('Cascade not yet implemented region-based training')
        num_label_inputs = len(label_manager.foreground_labels)
        num_input_channels = len(dataset_json["modality"]) + num_label_inputs
    else:
        num_input_channels = len(dataset_json["modality"])
    return num_input_channels


if __name__ == '__main__':
    # this code used to be able to differentiate variant 1 and 2 to measure time.
    num_labels = 7
    seg = np.random.randint(0, num_labels, size=(256, 256, 256), dtype=np.uint8)
    seg_torch = torch.from_numpy(seg)
    st = time()
    onehot_npy = convert_labelmap_to_one_hot(seg, np.arange(num_labels))
    time_1 = time()
    onehot_npy2 = convert_labelmap_to_one_hot(seg, np.arange(num_labels))
    time_2 = time()
    onehot_torch = convert_labelmap_to_one_hot(seg_torch, np.arange(num_labels))
    time_torch = time()
    onehot_torch2 = convert_labelmap_to_one_hot(seg_torch, np.arange(num_labels))
    time_torch2 = time()
    print(
        f'np: {time_1 - st}, np2: {time_2 - time_1}, torch: {time_torch - time_2}, torch2: {time_torch2 - time_torch}')
    onehot_torch = onehot_torch.numpy()
    onehot_torch2 = onehot_torch2.numpy()
    print(np.all(onehot_torch == onehot_npy))
    print(np.all(onehot_torch2 == onehot_npy))
