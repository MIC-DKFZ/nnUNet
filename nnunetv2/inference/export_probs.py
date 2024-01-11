from typing import Union

import numpy as np
import torch

from nnunetv2.configuration import default_num_processes
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)


def convert_predicted_logits_to_probs_with_correct_shape(
    predicted_logits: Union[torch.Tensor, np.ndarray],
    plans_manager: PlansManager,
    configuration_manager: ConfigurationManager,
    label_manager: LabelManager,
    properties_dict: dict,
    num_threads_torch: int = default_num_processes,
):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    # resample to original shape
    current_spacing = (
        configuration_manager.spacing
        if len(configuration_manager.spacing)
        == len(properties_dict["shape_after_cropping_and_before_resampling"])
        else [properties_dict["spacing"][0], *configuration_manager.spacing]
    )
    predicted_logits = configuration_manager.resampling_fn_probabilities(
        predicted_logits,
        properties_dict["shape_after_cropping_and_before_resampling"],
        current_spacing,
        properties_dict["spacing"],
    )
    # return value of resampling_fn_probabilities can be ndarray or Tensor but that does not matter because
    # apply_inference_nonlin will convert to torch

    predicted_probabilities = label_manager.apply_inference_nonlin(predicted_logits)
    del predicted_logits

    # revert cropping
    predicted_probabilities = label_manager.revert_cropping_on_probabilities(
        predicted_probabilities,
        properties_dict["bbox_used_for_cropping"],
        properties_dict["shape_before_cropping"],
    )
    predicted_probabilities = predicted_probabilities.cpu().numpy()
    # revert transpose
    predicted_probabilities = predicted_probabilities.transpose(
        [0] + [i + 1 for i in plans_manager.transpose_backward]
    )
    torch.set_num_threads(old_threads)
    return predicted_probabilities
