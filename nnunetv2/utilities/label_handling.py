from time import time
from typing import Tuple, Union, List

import numpy as np
import torch
from nnunetv2.utilities.utils import extract_unique_classes_from_dataset_json_labels


def handle_labels(dataset_json) -> Tuple[List, Union[List, None], Union[int, None]]:
    # first we need to check if we have to run region-based training
    region_needed = any([isinstance(i, tuple) and len(i) > 1 for i in dataset_json['labels'].values()])
    if region_needed:
        assert 'regions_class_order' in dataset_json.keys(), 'if region-based training is requested via ' \
                                                                  'dataset.json then you need to define ' \
                                                                  'regions_class_order as well, ' \
                                                                  'see documentation!'  # TODO add this
        regions = list(dataset_json['labels'].values())
        assert len(dataset_json['regions_class_order']) == len(regions), 'regions_class_order must have ans ' \
                                                                              'many entries as there are ' \
                                                                              'regions'
    else:
        regions = None
    all_labels = extract_unique_classes_from_dataset_json_labels(dataset_json['labels'])

    ignore_label = dataset_json['labels'].get('ignore')
    if ignore_label is not None:
        assert isinstance(ignore_label, int), f'Ignore label has to be an integer. It cannot be a region ' \
                                              f'(list/tuple). Got {type(ignore_label)}.'
    return all_labels, regions, ignore_label


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
    print(f'np: {time_1-st}, np2: {time_2-time_1}, torch: {time_torch-time_2}, torch2: {time_torch2 - time_torch}')
    onehot_torch = onehot_torch.numpy()
    onehot_torch2 = onehot_torch2.numpy()
    print(np.all(onehot_torch == onehot_npy))
    print(np.all(onehot_torch2 == onehot_npy))
