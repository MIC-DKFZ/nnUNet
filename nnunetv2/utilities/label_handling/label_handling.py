from __future__ import annotations
from time import time
from typing import Union, List, Tuple, Type

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from batchgenerators.utilities.file_and_folder_operations import join

import nnunetv2
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import softmax_helper_dim0

from typing import TYPE_CHECKING

# see https://adamj.eu/tech/2021/05/13/python-type-hints-how-to-fix-circular-imports/
if TYPE_CHECKING:
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


class LabelManager(object):
    def __init__(self, label_dict: dict, regions_class_order: Union[List[int], None], force_use_labels: bool = False,
                 inference_nonlin=None):
        self._sanity_check(label_dict)
        self.label_dict = label_dict
        self.regions_class_order = regions_class_order
        self._force_use_labels = force_use_labels

        if force_use_labels:
            self._has_regions = False
        else:
            self._has_regions: bool = any(
                [isinstance(i, (tuple, list)) and len(i) > 1 for i in self.label_dict.values()])

        self._ignore_label: Union[None, int] = self._determine_ignore_label()
        self._all_labels: List[int] = self._get_all_labels()

        self._regions: Union[None, List[Union[int, Tuple[int, ...]]]] = self._get_regions()

        if self.has_ignore_label:
            assert self.ignore_label == max(
                self.all_labels) + 1, 'If you use the ignore label it must have the highest ' \
                                      'label value! It cannot be 0 or in between other labels. ' \
                                      'Sorry bro.'

        if inference_nonlin is None:
            self.inference_nonlin = torch.sigmoid if self.has_regions else softmax_helper_dim0
        else:
            self.inference_nonlin = inference_nonlin

    def _sanity_check(self, label_dict: dict):
        if not 'background' in label_dict.keys():
            raise RuntimeError('Background label not declared (remember that this should be label 0!)')
        bg_label = label_dict['background']
        if isinstance(bg_label, (tuple, list)):
            raise RuntimeError(f"Background label must be 0. Not a list. Not a tuple. Your background label: {bg_label}")
        assert int(bg_label) == 0, f"Background label must be 0. Your background label: {bg_label}"
        # not sure if we want to allow regions that contain background. I don't immediately see how this could cause
        # problems so we allow it for now. That doesn't mean that this is explicitly supported. It could be that this
        # just crashes.

    def _get_all_labels(self) -> List[int]:
        all_labels = []
        for k, r in self.label_dict.items():
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

    def _get_regions(self) -> Union[None, List[Union[int, Tuple[int, ...]]]]:
        if not self._has_regions or self._force_use_labels:
            return None
        else:
            assert self.regions_class_order is not None, 'if region-based training is requested then you need to ' \
                                                         'define regions_class_order!'
            regions = []
            for k, r in self.label_dict.items():
                # ignore ignore label
                if k == 'ignore':
                    continue
                # ignore regions that are background
                if (np.isscalar(r) and r == 0) \
                        or \
                        (isinstance(r, (tuple, list)) and len(np.unique(r)) == 1 and np.unique(r)[0] == 0):
                    continue
                if isinstance(r, list):
                    r = tuple(r)
                regions.append(r)
            assert len(self.regions_class_order) == len(regions), 'regions_class_order must have as ' \
                                                                  'many entries as there are ' \
                                                                  'regions'
            return regions

    def _determine_ignore_label(self) -> Union[None, int]:
        ignore_label = self.label_dict.get('ignore')
        if ignore_label is not None:
            assert isinstance(ignore_label, int), f'Ignore label has to be an integer. It cannot be a region ' \
                                                  f'(list/tuple). Got {type(ignore_label)}.'
        return ignore_label

    @property
    def has_regions(self) -> bool:
        return self._has_regions

    @property
    def has_ignore_label(self) -> bool:
        return self.ignore_label is not None

    @property
    def all_regions(self) -> Union[None, List[Union[int, Tuple[int, ...]]]]:
        return self._regions

    @property
    def all_labels(self) -> List[int]:
        return self._all_labels

    @property
    def ignore_label(self) -> Union[None, int]:
        return self._ignore_label

    def apply_inference_nonlin(self, logits: Union[np.ndarray, torch.Tensor]) -> \
            Union[np.ndarray, torch.Tensor]:
        """
        logits has to have shape (c, x, y(, z)) where c is the number of classes/regions
        """
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)

        with torch.no_grad():
            # softmax etc is not implemented for half
            logits = logits.float()
            probabilities = self.inference_nonlin(logits)

        return probabilities

    def convert_probabilities_to_segmentation(self, predicted_probabilities: Union[np.ndarray, torch.Tensor]) -> \
            Union[np.ndarray, torch.Tensor]:
        """
        assumes that inference_nonlinearity was already applied!

        predicted_probabilities has to have shape (c, x, y(, z)) where c is the number of classes/regions
        """
        if not isinstance(predicted_probabilities, (np.ndarray, torch.Tensor)):
            raise RuntimeError(f"Unexpected input type. Expected np.ndarray or torch.Tensor,"
                               f" got {type(predicted_probabilities)}")

        if self.has_regions:
            assert self.regions_class_order is not None, 'if region-based training is requested then you need to ' \
                                                         'define regions_class_order!'
            # check correct number of outputs
        assert predicted_probabilities.shape[0] == self.num_segmentation_heads, \
            f'unexpected number of channels in predicted_probabilities. Expected {self.num_segmentation_heads}, ' \
            f'got {predicted_probabilities.shape[0]}. Remember that predicted_probabilities should have shape ' \
            f'(c, x, y(, z)).'

        if self.has_regions:
            if isinstance(predicted_probabilities, np.ndarray):
                segmentation = np.zeros(predicted_probabilities.shape[1:], dtype=np.uint16)
            else:
                # no uint16 in torch
                segmentation = torch.zeros(predicted_probabilities.shape[1:], dtype=torch.int16,
                                           device=predicted_probabilities.device)
            for i, c in enumerate(self.regions_class_order):
                segmentation[predicted_probabilities[i] > 0.5] = c
        else:
            segmentation = predicted_probabilities.argmax(0)

        return segmentation

    def convert_logits_to_segmentation(self, predicted_logits: Union[np.ndarray, torch.Tensor]) -> \
            Union[np.ndarray, torch.Tensor]:
        input_is_numpy = isinstance(predicted_logits, np.ndarray)
        probabilities = self.apply_inference_nonlin(predicted_logits)
        if input_is_numpy and isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.cpu().numpy()
        return self.convert_probabilities_to_segmentation(probabilities)

    def revert_cropping_on_probabilities(self, predicted_probabilities: Union[torch.Tensor, np.ndarray],
                                         bbox: List[List[int]],
                                         original_shape: Union[List[int], Tuple[int, ...]]):
        """
        ONLY USE THIS WITH PROBABILITIES, DO NOT USE LOGITS AND DO NOT USE FOR SEGMENTATION MAPS!!!

        predicted_probabilities must be (c, x, y(, z))

        Why do we do this here? Well if we pad probabilities we need to make sure that convert_logits_to_segmentation
        correctly returns background in the padded areas. Also we want to ba able to look at the padded probabilities
        and not have strange artifacts.
        Only LabelManager knows how this needs to be done. So let's let him/her do it, ok?
        """
        # revert cropping
        probs_reverted_cropping = np.zeros((predicted_probabilities.shape[0], *original_shape),
                                           dtype=predicted_probabilities.dtype) \
            if isinstance(predicted_probabilities, np.ndarray) else \
            torch.zeros((predicted_probabilities.shape[0], *original_shape), dtype=predicted_probabilities.dtype)

        if not self.has_regions:
            probs_reverted_cropping[0] = 1

        slicer = bounding_box_to_slice(bbox)
        probs_reverted_cropping[tuple([slice(None)] + list(slicer))] = predicted_probabilities
        return probs_reverted_cropping

    @staticmethod
    def filter_background(classes_or_regions: Union[List[int], List[Union[int, Tuple[int, ...]]]]):
        # heck yeah
        # This is definitely taking list comprehension too far. Enjoy.
        return [i for i in classes_or_regions if
                ((not isinstance(i, (tuple, list))) and i != 0)
                or
                (isinstance(i, (tuple, list)) and not (
                        len(np.unique(i)) == 1 and np.unique(i)[0] == 0))]

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


def get_labelmanager_class_from_plans(plans: dict) -> Type[LabelManager]:
    if 'label_manager' not in plans.keys():
        print('No label manager specified in plans. Using default: LabelManager')
        return LabelManager
    else:
        labelmanager_class = recursive_find_python_class(join(nnunetv2.__path__[0], "utilities", "label_handling"),
                                                         plans['label_manager'],
                                                         current_module="nnunetv2.utilities.label_handling")
        return labelmanager_class


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


def determine_num_input_channels(plans_manager: PlansManager,
                                 configuration_or_config_manager: Union[str, ConfigurationManager],
                                 dataset_json: dict) -> int:
    if isinstance(configuration_or_config_manager, str):
        config_manager = plans_manager.get_configuration(configuration_or_config_manager)
    else:
        config_manager = configuration_or_config_manager

    label_manager = plans_manager.get_label_manager(dataset_json)
    num_modalities = len(dataset_json['modality']) if 'modality' in dataset_json.keys() else len(dataset_json['channel_names'])

    # cascade has different number of input channels
    if config_manager.previous_stage_name is not None:
        num_label_inputs = len(label_manager.foreground_labels)
        num_input_channels = num_modalities + num_label_inputs
    else:
        num_input_channels = num_modalities
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
