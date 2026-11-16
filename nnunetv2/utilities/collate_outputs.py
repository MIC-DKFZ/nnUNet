from typing import List

import numpy as np
import torch


def outputs_to_host(outputs: List[dict]) -> List[dict]:
    """
    train_step / validation_step outputs with every torch tensor replaced by a numpy array, as if each step had returned
    tensor.cpu().numpy(). The steps leave their outputs on the device so that the host does not wait for the GPU in every
    iteration; this moves all of them at once, with one transfer per key. Outputs without tensors are returned as they are.

    we expect outputs to be a list of dictionaries where each of the dict has the same set of keys
    """
    keys = [k for k, v in outputs[0].items() if isinstance(v, torch.Tensor)] if len(outputs) > 0 else []
    if len(keys) == 0:
        return outputs
    on_host = {k: torch.stack([o[k].detach() for o in outputs]).cpu().numpy() for k in keys}
    # [i, ...] and not [i]: a 0-d output (the loss) stays a 0-d array instead of becoming a numpy scalar
    return [{**o, **{k: on_host[k][i, ...] for k in keys}} for i, o in enumerate(outputs)]


def collate_outputs(outputs: List[dict]):
    """
    used to collate default train_step and validation_step outputs. If you want something different then you gotta
    extend this

    we expect outputs to be a list of dictionaries where each of the dict has the same set of keys
    """
    outputs = outputs_to_host(outputs)
    collated = {}
    for k in outputs[0].keys():
        if np.isscalar(outputs[0][k]):
            collated[k] = [o[k] for o in outputs]
        elif isinstance(outputs[0][k], np.ndarray):
            collated[k] = np.vstack([o[k][None] for o in outputs])
        elif isinstance(outputs[0][k], list):
            collated[k] = [item for o in outputs for item in o[k]]
        else:
            raise ValueError(f'Cannot collate input of type {type(outputs[0][k])}. '
                             f'Modify collate_outputs to add this functionality')
    return collated
