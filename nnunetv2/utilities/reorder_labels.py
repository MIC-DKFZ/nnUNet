"""
Give a trained model new label ids without retraining.

The output channels of the network's segmentation layers (the 1x1(x1) convolutions decoder.seg_layers.*, the
deep-supervision heads included) are permuted and dataset.json is rewritten: channel j of the new model is the channel
of the label that has id j in the new numbering. The argmax picks the same label name in every voxel, so the
segmentations are the same up to the relabelling. (Only an exact tie between two channels could resolve differently,
since the argmax takes the first maximum.) Typical use: a tool built on nnU-Net numbers its labels its own way and wants
the model to speak its numbering natively instead of mapping ids after every prediction.

Writes a new, inference-only model folder: plans.json, the rewritten dataset.json and, per fold, a .safetensors
checkpoint with what inference reads (network weights, trainer name, configuration, mirror axes). Training state
(optimizer, logs) is not carried over: the per-class logs would refer to the old ids. Region-based models are refused.

    nnUNetv2_reorder_labels -i MODEL_FOLDER -o NEW_MODEL_FOLDER -l labels.json [-f 0 1 ...]

labels.json holds the new {label name: id} mapping, either as is or under a "labels" key (a dataset.json works).
"""
import argparse
import re
import shutil
from typing import Dict, List, Optional, Sequence, Union

import torch
from batchgenerators.utilities.file_and_folder_operations import isdir, join, load_json, maybe_mkdir_p, save_json, \
    subdirs

from nnunetv2.utilities.checkpoints import find_checkpoint_file, load_checkpoint_file, safetensors_filename, \
    save_checkpoint_file

_SEG_LAYER = re.compile(r'(^|\.)seg_layers\.\d+\.(weight|bias)$')


def label_permutation(old_labels: Dict[str, int], new_labels: Dict[str, int]) -> List[int]:
    """perm with new channel j = old channel perm[j]. Both mappings must name the same labels with ids 0 .. C-1."""
    for name, labels in (('old', old_labels), ('new', new_labels)):
        if any(not isinstance(v, int) or isinstance(v, bool) for v in labels.values()):
            raise ValueError(f'The {name} labels are not plain integer ids (region-based models are not supported): '
                             f'{labels}')
        if sorted(labels.values()) != list(range(len(labels))):
            raise ValueError(f'The {name} label ids must be 0 .. {len(labels) - 1}, each once: {sorted(labels.values())}')
    if set(old_labels) != set(new_labels):
        raise ValueError(f'The label names differ. Only in the model: {sorted(set(old_labels) - set(new_labels))}; '
                         f'only in the new labels: {sorted(set(new_labels) - set(old_labels))}')
    return [old_labels[name] for name, _ in sorted(new_labels.items(), key=lambda kv: kv[1])]


def permute_seg_layers(network_weights: dict, perm: Sequence[int]) -> dict:
    """A copy of network_weights with the output channels of every seg_layers entry permuted."""
    out, found = {}, 0
    index = torch.as_tensor(list(perm), dtype=torch.long)
    for k, v in network_weights.items():
        if _SEG_LAYER.search(k):
            if v.shape[0] != len(perm):
                raise ValueError(f'{k} has {v.shape[0]} output channels, the labels {len(perm)}')
            v = v.index_select(0, index).clone()
            found += 1
        out[k] = v
    if found == 0:
        raise ValueError('No segmentation layers (decoder.seg_layers.*) found in the network weights')
    return out


def reorder_model_labels(model_folder: str, output_folder: str, new_labels: Dict[str, int],
                         folds: Optional[Sequence[Union[int, str]]] = None,
                         checkpoint_name: str = 'checkpoint_final.safetensors') -> List[int]:
    """Write model_folder with new_labels' ids to output_folder (see the module docstring). Returns the permutation."""
    dataset_json = load_json(join(model_folder, 'dataset.json'))
    if 'regions_class_order' in dataset_json:
        raise ValueError('Region-based models are not supported')
    perm = label_permutation(dataset_json['labels'], new_labels)
    if folds is None:
        folds = [d[len('fold_'):] for d in subdirs(model_folder, prefix='fold_', join=False)
                 if find_checkpoint_file(join(model_folder, d, checkpoint_name)) is not None]
    if len(folds) == 0:
        raise FileNotFoundError(f'No fold_* folder with {checkpoint_name} in {model_folder}')
    if isdir(output_folder) and len(subdirs(output_folder, prefix='fold_')) > 0:
        raise FileExistsError(f'{output_folder} already holds a model')

    maybe_mkdir_p(output_folder)
    for f in folds:
        checkpoint = load_checkpoint_file(join(model_folder, f'fold_{f}', checkpoint_name), map_location='cpu',
                                          keys=('network_weights', 'trainer_name', 'init_args',
                                                'inference_allowed_mirroring_axes'))
        maybe_mkdir_p(join(output_folder, f'fold_{f}'))
        save_checkpoint_file({
            'network_weights': permute_seg_layers(checkpoint['network_weights'], perm),
            'trainer_name': checkpoint['trainer_name'],
            'init_args': {'configuration': checkpoint['init_args']['configuration'],
                          'fold': checkpoint['init_args'].get('fold')},
            'inference_allowed_mirroring_axes': checkpoint.get('inference_allowed_mirroring_axes'),
        }, join(output_folder, f'fold_{f}', safetensors_filename(checkpoint_name)))
    dataset_json['labels'] = dict(sorted(new_labels.items(), key=lambda kv: kv[1]))
    save_json(dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
    shutil.copyfile(join(model_folder, 'plans.json'), join(output_folder, 'plans.json'))
    return perm


def reorder_labels_entry_point():
    parser = argparse.ArgumentParser(description='Give a trained model new label ids without retraining (permutes the '
                                                 'output channels of its segmentation layers, rewrites dataset.json). '
                                                 'Writes a new, inference-only model folder.')
    parser.add_argument('-i', required=True, help='trained model folder (the one holding plans.json and fold_*)')
    parser.add_argument('-o', required=True, help='output model folder (must not hold a model yet)')
    parser.add_argument('-l', required=True, help='json file with the new {label name: id} mapping, as is or under a '
                                                  '"labels" key (a dataset.json works)')
    parser.add_argument('-f', nargs='+', default=None, help='folds (default: all with a checkpoint)')
    parser.add_argument('-chk', default='checkpoint_final.safetensors',
                        help='checkpoint name (.pth checkpoints are found as well); written as .safetensors')
    args = parser.parse_args()
    labels = load_json(args.l)
    labels = labels.get('labels', labels)
    perm = reorder_model_labels(args.i, args.o, labels, args.f, args.chk)
    print(f'Wrote {args.o}: new channel j = old channel {perm}')


if __name__ == '__main__':
    reorder_labels_entry_point()
