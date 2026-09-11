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
import multiprocessing
import os
from time import sleep
from typing import List, Optional, Sequence, Tuple, Union

import blosc2
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isfile, join, load_json, load_pickle
from tqdm import tqdm

from nnunetv2.configuration import default_num_processes
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.training.dataloading.foreground_locations import (
    ForegroundLocationsWriter, has_foreground_locations, ravel_coords)
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

DEFAULT_SEED = 1234
DEFAULT_MIN_NUM_SAMPLES = 10000
DEFAULT_MIN_PERCENT_COVERAGE = 0.01


def get_classes_or_regions_to_collect(label_manager) -> List[Union[int, List[int]]]:
    """
    Which classes/regions get sampling locations. Mirrors what DefaultPreprocessor.run_case_npy
    used to do, but without mutating the list owned by the LabelManager.
    """
    collect_for_this = list(label_manager.foreground_regions if label_manager.has_regions
                            else label_manager.foreground_labels)
    if label_manager.has_ignore_label:
        # with an ignore label we also want to be able to sample uniformly from all *annotated*
        # voxels (background included), so that patches without foreground are still annotated
        collect_for_this.append([-1] + label_manager.all_labels)
    return collect_for_this


def _normalize_key(k):
    return tuple(int(i) for i in k) if isinstance(k, (tuple, list)) else int(k)


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
    seg = blosc2.open(urlpath=seg_file, mode='r', dparams={'nthreads': 1})[:]
    spatial_shape = tuple(int(i) for i in seg.shape[1:])

    # preprocessing records which labels a case contains, so we do not have to rediscover them
    present_labels = None
    if pkl_file is not None and isfile(pkl_file):
        present_labels = load_pickle(pkl_file).get('present_labels', None)

    class_locations = DefaultPreprocessor._sample_foreground_locations(
        seg, classes_or_regions, seed=seed, verbose=verbose,
        min_num_samples=min_num_samples, min_percent_coverage=min_percent_coverage,
        present_labels=present_labels)

    out = {}
    for k, v in class_locations.items():
        if len(v) == 0:
            continue
        out[_normalize_key(k)] = ravel_coords(np.asarray(v), spatial_shape)
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

    class_keys = [_normalize_key(k) for k in classes_or_regions]
    sampling_parameters = {'seed': seed, 'min_num_samples': min_num_samples,
                           'min_percent_coverage': min_percent_coverage}

    writer = ForegroundLocationsWriter(folder, class_keys, sampling_parameters=sampling_parameters)

    with multiprocessing.get_context('spawn').Pool(num_processes) as p:
        workers = [j for j in p._pool]
        r = []
        for identifier in identifiers:
            r.append(p.starmap_async(extract_case, ((join(folder, identifier + '_seg.b2nd'),
                                                     join(folder, identifier + '.pkl'),
                                                     class_keys, seed, min_num_samples,
                                                     min_percent_coverage, verbose),)))
        remaining = list(range(len(identifiers)))
        with tqdm(desc='Extracting foreground sampling locations', total=len(identifiers),
                  disable=not show_progress_bar) as pbar:
            while len(remaining) > 0:
                if not all([j.is_alive() for j in workers]):
                    raise RuntimeError('Some background worker is 6 feet under. Yuck. \n'
                                       'One of your background processes is missing. This could be because of '
                                       'an error (look for an error message) or because it was killed '
                                       'by your OS due to running out of RAM. If you don\'t see '
                                       'an error message, out of RAM is likely the problem. In that case '
                                       'reducing the number of workers might help')
                done = [i for i in remaining if r[i].ready()]
                for i in done:
                    shape, locations = r[i].get()[0]
                    writer.add(identifiers[i], shape, locations)
                    r[i] = None  # free the result as soon as it is on disk
                    pbar.update()
                remaining = [i for i in remaining if i not in done]
                if len(done) == 0:
                    sleep(0.1)
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
    classes_or_regions = get_classes_or_regions_to_collect(label_manager)

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
            folder, classes_or_regions, num_processes=num_processes, verbose=verbose,
            show_progress_bar=show_progress_bar)
