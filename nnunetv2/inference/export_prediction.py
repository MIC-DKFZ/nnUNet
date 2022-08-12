from nnunetv2.imageio.reader_writer_registry import recursive_find_reader_writer_by_name
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
import os
from copy import deepcopy
from typing import Union, List

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_json, isfile, save_pickle

from nnunetv2.preprocessing.resampling.utils import recursive_find_resampling_fn_by_name
from nnunetv2.utilities.label_handling.label_handling import get_labelmanager


def export_prediction(predicted_array_or_file: Union[np.ndarray, str], properties_dict: dict,
                      configuration_name: str,
                      plans_dict_or_file: Union[dict, str],
                      dataset_json_dict_or_file: Union[dict, str], output_file_truncated: str,
                      save_probabilities: bool = False):

    if isinstance(predicted_array_or_file, str):
        tmp = deepcopy(predicted_array_or_file)
        if predicted_array_or_file.endswith('.npy'):
            predicted_array_or_file = np.load(predicted_array_or_file)
        elif predicted_array_or_file.endswith('.npz'):
            predicted_array_or_file = np.load(predicted_array_or_file)['softmax']
        os.remove(tmp)

    if isinstance(plans_dict_or_file, str):
        plans_dict_or_file = load_json(plans_dict_or_file)
    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    # resample to original shape
    resampling_fn = recursive_find_resampling_fn_by_name(
        plans_dict_or_file['configurations'][configuration_name]["resampling_fn_probabilities"]
    )
    current_spacing = plans_dict_or_file['configurations'][configuration_name]["spacing"] if \
        len(plans_dict_or_file['configurations'][configuration_name]["spacing"]) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [properties_dict['spacing'][0], *plans_dict_or_file['configurations'][configuration_name]["spacing"]]
    predicted_array_or_file = resampling_fn(predicted_array_or_file,
                                            properties_dict['shape_after_cropping_and_before_resampling'],
                                            current_spacing,
                                            properties_dict['spacing'],
                                            **plans_dict_or_file['configurations'][configuration_name]["resampling_fn_probabilities_kwargs"])
    label_manager = get_labelmanager(plans_dict_or_file, dataset_json_dict_or_file)
    segmentation = label_manager.convert_logits_to_segmentation(predicted_array_or_file)

    # put result in bbox (revert cropping)
    segmentation_reverted_cropping = np.zeros(properties_dict['shape_before_cropping'], dtype=np.uint8)
    slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
    segmentation_reverted_cropping[slicer] = segmentation
    del segmentation

    # revert transpose
    segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(plans_dict_or_file['transpose_backward'])

    # save
    if save_probabilities:
        # probabilities are already resampled

        # apply nonlinearity
        predicted_array_or_file = label_manager.apply_inference_nonlin(predicted_array_or_file)

        # revert cropping
        probs_reverted_cropping = label_manager.revert_cropping(predicted_array_or_file,
                                                                properties_dict['bbox_used_for_cropping'],
                                                                properties_dict['shape_before_cropping'])
        # $revert transpose
        probs_reverted_cropping = probs_reverted_cropping.transpose([0] + [i + 1 for i in
                                                                           plans_dict_or_file['transpose_backward']])
        np.savez_compressed(output_file_truncated + '.npz', probabilities=probs_reverted_cropping)
        save_pickle(properties_dict, output_file_truncated + '.pkl')
        del probs_reverted_cropping
    del predicted_array_or_file

    rw = recursive_find_reader_writer_by_name(plans_dict_or_file["image_reader_writer"])()
    rw.write_seg(segmentation_reverted_cropping, output_file_truncated + dataset_json_dict_or_file['file_ending'], properties_dict)


def resample_and_save(predicted: Union[str, np.ndarray], target_shape: List[int], output_file: str,
                      plans_dict_or_file: Union[dict, str], configuration_name: str, properties_dict: dict,
                      dataset_json_dict_or_file: Union[dict, str], next_configuration: str) -> None:
    if isinstance(predicted, str):
        assert isfile(predicted), "If isinstance(segmentation_softmax, str) then " \
                                  "isfile(segmentation_softmax) must be True"
        del_file = deepcopy(predicted)
        predicted = np.load(predicted)
        os.remove(del_file)

    if isinstance(plans_dict_or_file, str):
        plans_dict_or_file = load_json(plans_dict_or_file)
    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    # resample to original shape
    resampling_fn = recursive_find_resampling_fn_by_name(
        plans_dict_or_file['configurations'][configuration_name]["resampling_fn_probabilities"]
    )
    current_spacing = plans_dict_or_file['configurations'][configuration_name]["spacing"] if \
        len(plans_dict_or_file['configurations'][configuration_name]["spacing"]) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [properties_dict['spacing'][0], *plans_dict_or_file['configurations'][configuration_name]["spacing"]]
    target_spacing = plans_dict_or_file['configurations'][next_configuration]["spacing"] if \
        len(plans_dict_or_file['configurations'][next_configuration]["spacing"]) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [properties_dict['spacing'][0], *plans_dict_or_file['configurations'][next_configuration]["spacing"]]
    predicted_array_or_file = resampling_fn(predicted,
                                            target_shape,
                                            current_spacing,
                                            target_spacing,
                                            **plans_dict_or_file['configurations'][configuration_name]["resampling_fn_probabilities_kwargs"])

    # create segmentation (argmax, regions, etc)
    label_manager = get_labelmanager(plans_dict_or_file, dataset_json_dict_or_file)
    segmentation = label_manager.convert_logits_to_segmentation(predicted_array_or_file)

    np.savez_compressed(output_file, seg=segmentation.astype(np.uint8))
