"""
Standalone pass that builds the foreground sampling location store for a preprocessed
configuration folder.

This is deliberately separate from preprocessing:
  * it only reads the stored ``_seg.b2nd`` files, never the image data. On TotalSegmentator v2 that
    is 0.24 GB rather than 45 GB, so a full re-extraction takes minutes.
  * it can therefore be re-run at any time -- after changing the sampling logic, or to migrate a
    dataset that was preprocessed by an older nnU-Net -- without re-running preprocessing.

The sampling itself is unchanged: it calls ``DefaultPreprocessor._sample_foreground_locations``
with the same parameters preprocessing used to use.
"""
import os
from typing import List, Optional, Sequence, Tuple, Union

import blosc2
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isfile, join, load_json, load_pickle

from nnunetv2.configuration import default_num_processes
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.training.dataloading.foreground_locations import (
    MMAP_KWARGS, ForegroundLocationsWriter, has_foreground_locations, normalize_class_key,
    ravel_coords)
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.pool_utils import imap_unordered_with_progress

DEFAULT_SEED = 1234
DEFAULT_MIN_NUM_SAMPLES = 10000
DEFAULT_MIN_PERCENT_COVERAGE = 0.01


def _requested_labels(classes_or_regions: Sequence) -> set:
    labels = set()
    for c in classes_or_regions:
        if isinstance(c, (tuple, list)):
            labels.update(int(i) for i in c)
        else:
            labels.add(int(c))
    return labels


def present_labels_from_class_locations(class_locations: dict, classes_or_regions: Sequence) -> List[int]:
    """
    Which labels may occur in the segmentation, read off a legacy ``class_locations`` dict.

    _sample_foreground_locations leaves an entry empty exactly when none of that class's/region's labels
    are present (a present label always yields at least one sampled coordinate), so a label can be ruled
    out only if it appears in an empty entry and in no non-empty one. Labels the legacy dict says nothing
    about - because dataset.json changed since it was written - are kept: the hint may overstate what is
    present, but it must never miss a label that is.
    """
    in_empty, in_nonempty = set(), set()
    for key, v in class_locations.items():
        labels = key if isinstance(key, (tuple, list)) else (key,)
        (in_nonempty if len(v) > 0 else in_empty).update(int(i) for i in labels)
    return sorted(_requested_labels(classes_or_regions) - (in_empty - in_nonempty))


def extract_case(seg_file: str, pkl_file: Optional[str], classes_or_regions: Sequence,
                 seed: int = DEFAULT_SEED, min_num_samples: int = DEFAULT_MIN_NUM_SAMPLES,
                 min_percent_coverage: float = DEFAULT_MIN_PERCENT_COVERAGE,
                 verbose: bool = False) -> Tuple[Tuple[int, int, int], dict]:
    """
    Runs in a worker process. Returns the case's spatial shape and a dict of
    class key -> sorted 1D uint64 linear indices.
    """
    # imported here so that worker processes started with 'spawn' do not pay for it at import time
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor

    blosc2.set_nthreads(1)
    seg = blosc2.open(urlpath=seg_file, mode='r', dparams={'nthreads': 1}, **MMAP_KWARGS)[:]
    spatial_shape = tuple(int(i) for i in seg.shape[1:])

    # Labels that are not in this segmentation cannot contribute anything, so telling the sampler about
    # them lets it shrink its np.isin. Both sources of that hint are already on disk - preprocessing
    # recorded it, and on a legacy dataset the old sampling result implies it - so there is no reason to
    # scan the segmentation for it here.
    present_labels = None
    if pkl_file is not None and isfile(pkl_file):
        properties = load_pickle(pkl_file)
        if 'present_labels' in properties:
            present_labels = properties['present_labels']
        elif 'class_locations' in properties:
            present_labels = present_labels_from_class_locations(properties['class_locations'], classes_or_regions)

    class_locations = DefaultPreprocessor._sample_foreground_locations(
        seg, classes_or_regions, seed=seed, verbose=verbose,
        min_num_samples=min_num_samples, min_percent_coverage=min_percent_coverage,
        present_labels=present_labels)

    out = {}
    for k, v in class_locations.items():
        if len(v) == 0:
            continue
        out[normalize_class_key(k)] = ravel_coords(np.asarray(v), spatial_shape)
    return spatial_shape, out


def extract_sampling_locations_for_folder(folder: str, classes_or_regions: Sequence,
                                          identifiers: Optional[List[str]] = None,
                                          num_processes: int = default_num_processes,
                                          seed: int = DEFAULT_SEED,
                                          min_num_samples: int = DEFAULT_MIN_NUM_SAMPLES,
                                          min_percent_coverage: float = DEFAULT_MIN_PERCENT_COVERAGE,
                                          verbose: bool = False,
                                          show_progress_bar: bool = True) -> None:
    """
    Build the store for one preprocessed configuration folder.

    Results are written to disk as they arrive, so the parent never holds more than a buffer of
    coordinates regardless of dataset size. Cases may complete in any order.
    """
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

    if identifiers is None:
        identifiers = infer_dataset_class(folder).get_identifiers(folder)
    identifiers = sorted(identifiers)
    if len(identifiers) == 0:
        raise RuntimeError(f'No preprocessed cases found in {folder}')

    class_keys = [normalize_class_key(k) for k in classes_or_regions]
    sampling_parameters = {'seed': seed, 'min_num_samples': min_num_samples,
                           'min_percent_coverage': min_percent_coverage}

    writer = ForegroundLocationsWriter(folder, class_keys, sampling_parameters=sampling_parameters)
    args = [(join(folder, i + '_seg.b2nd'), join(folder, i + '.pkl'), class_keys, seed, min_num_samples,
             min_percent_coverage, verbose) for i in identifiers]
    for i, (shape, locations) in imap_unordered_with_progress(
            extract_case, args, num_processes, desc='Extracting foreground sampling locations',
            show_progress_bar=show_progress_bar):
        writer.add(identifiers[i], shape, locations)
    writer.finalize()


def extract_sampling_locations_dataset(dataset_id_or_name: Union[int, str],
                                       plans_identifier: str = 'nnUNetPlans',
                                       configurations: Union[Tuple[str, ...], List[str]] = (
                                               '2d', '3d_fullres', '3d_lowres'),
                                       num_processes: int = default_num_processes,
                                       overwrite: bool = True,
                                       verbose: bool = False,
                                       show_progress_bar: bool = True) -> None:
    dataset_name = maybe_convert_to_dataset_name(dataset_id_or_name)
    plans_file = join(nnUNet_preprocessed, dataset_name, plans_identifier + '.json')
    if not isfile(plans_file):
        raise RuntimeError(f'Expected plans file ({plans_file}) not found. Run nnUNetv2_plan_experiment first.')
    plans_manager = PlansManager(plans_file)
    dataset_json = load_json(join(nnUNet_preprocessed, dataset_name, 'dataset.json'))
    label_manager = plans_manager.get_label_manager(dataset_json)

    for c in configurations:
        if c not in plans_manager.available_configurations:
            print(f'INFO: Configuration {c} not found in plans file {plans_identifier}.json of dataset '
                  f'{dataset_name}. Skipping.')
            continue
        folder = join(nnUNet_preprocessed, dataset_name, plans_manager.get_configuration(c).data_identifier)
        if not os.path.isdir(folder):
            print(f'INFO: Configuration {c} of dataset {dataset_name} has not been preprocessed '
                  f'({folder} does not exist). Skipping.')
            continue
        if has_foreground_locations(folder) and not overwrite:
            print(f'INFO: Configuration {c} already has a foreground sampling location store. Skipping '
                  f'(use --overwrite to rebuild it).')
            continue
        print(f'Configuration: {c}...')
        extract_sampling_locations_for_folder(
            folder, label_manager.classes_or_regions_for_sampling, num_processes=num_processes,
            verbose=verbose, show_progress_bar=show_progress_bar)
