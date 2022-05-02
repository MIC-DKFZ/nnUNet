from nnunetv2.imageio.reader_writer_registry import recursive_find_reader_writer_by_name
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
import os
from copy import deepcopy
from typing import Union

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_json

from nnunetv2.preprocessing.resampling.utils import recursive_find_resampling_fn_by_name


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

    # revert transpose
    predicted_array_or_file = predicted_array_or_file.transpose([0] + [i + 1 for i in
                                                                       plans_dict_or_file['transpose_backward']])

    # resample to original shape
    resampling_fn = recursive_find_resampling_fn_by_name(
        plans_dict_or_file['configurations'][configuration_name]["resampling_fn_softmax"]
    )
    current_spacing = plans_dict_or_file['configurations'][configuration_name]["spacing"] if \
        len(plans_dict_or_file['configurations'][configuration_name]["spacing"]) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [properties_dict['spacing'][0], *plans_dict_or_file['configurations'][configuration_name]["spacing"]]
    predicted_array_or_file = resampling_fn(predicted_array_or_file,
                                            properties_dict['shape_after_cropping_and_before_resampling'],
                                            current_spacing,
                                            properties_dict['spacing'],
                                            **plans_dict_or_file['configurations'][configuration_name]["resampling_fn_softmax_kwargs"])

    # create segmentation (argmax, regions, etc)
    use_regions = any([isinstance(i, tuple) and len(i) > 1 for i in dataset_json_dict_or_file['labels'].values()])
    if use_regions:
        regions_class_order = dataset_json_dict_or_file['regions_class_order']
        segmentation = np.zeros(predicted_array_or_file.shape[1:], dtype=np.uint8)
        for i, c in enumerate(regions_class_order):
            segmentation[predicted_array_or_file[i] > 0.5] = c
    else:
        segmentation = predicted_array_or_file.argmax(0)

    # put result in bbox (revert cropping)
    segmentation_reverted_cropping = np.zeros(properties_dict['shape_before_cropping'], dtype=np.uint8)
    slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
    segmentation_reverted_cropping[slicer] = segmentation

    # save
    if save_probabilities:
        probs_reverted_cropping = np.zeros((predicted_array_or_file.shape[0], *properties_dict['shape_before_cropping']), dtype=np.float16)
        slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
        probs_reverted_cropping[tuple([slice(None)] + list(slicer))] = predicted_array_or_file
        np.savez_compressed(output_file_truncated + '.npz', probabilities=probs_reverted_cropping)
        del probs_reverted_cropping
    del predicted_array_or_file

    rw = recursive_find_reader_writer_by_name(plans_dict_or_file["image_reader_writer"])()
    rw.write_seg(segmentation_reverted_cropping, output_file_truncated + dataset_json_dict_or_file['file_ending'], properties_dict)


