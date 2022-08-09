from copy import deepcopy
from multiprocessing import Pool
from typing import List, Union, Tuple

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_pickle, save_pickle, load_json, join, subfiles, \
    maybe_mkdir_p, isdir

from nnunetv2.configuration import default_num_processes
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.imageio.reader_writer_registry import recursive_find_reader_writer_by_name
from nnunetv2.utilities.label_handling.label_handling import LabelManager


def average_probabilities(list_of_files: List[str]) -> np.ndarray:
    assert len(list_of_files), 'At least one file must be given in list_of_files'
    avg = None
    for f in list_of_files:
        if avg is None:
            avg = np.load(f)['probabilities']
            # maybe increase precision to prevent rounding errors
            if avg.dtype != np.float32:
                avg = avg.astype(np.float32)
        else:
            avg += np.load(f)['probabilities']
    avg /= len(list_of_files)
    return avg


def merge_files(list_of_files,
                output_filename_truncated: str,
                output_file_ending: str,
                image_reader_writer: BaseReaderWriter,
                label_manager: LabelManager,
                save_probabilities: bool = False):
    raise NotImplementedError('check that we are getting a labelmanager')
    # load the pkl file associated with the first file in list_of_files
    properties = load_pickle(list_of_files[0][:-4] + '.pkl')
    # load and average predictions
    probabilities = average_probabilities(list_of_files)
    segmentation = label_manager.convert_logits_to_segmentation(probabilities)
    image_reader_writer.write_seg(segmentation, output_filename_truncated + output_file_ending, properties)
    if save_probabilities:
        np.savez_compressed(output_filename_truncated + '.npz', probabilities=probabilities)
        save_pickle(probabilities, output_filename_truncated + '.pkl')


def ensemble_folders(list_of_input_folders: List[str],
                     output_folder: str,
                     save_merged_probabilities: bool = False,
                     num_processes: int = default_num_processes):
    """we need too much shit for this function. Problem is that we now have to support region-based training plus
    multiple input/output formats so there isn't really a way around this.
    We assume each of the folders has a plans.json and dataset.json in it. These are usually copied into those folders
    by nnU-Net during prediction.
    We just pick the dataset.json and plans.json from the first of the folders and we DONT check whether the 5
    folders contain the same files! This can be a feature if results from different datasets are to be merged (only
    works if label dict in dataset.json is the same between these datasets!!!)"""
    dataset_json = load_json(join(list_of_input_folders[0], 'dataset.json'))
    plans = load_json(join(list_of_input_folders[0], 'plans.json'))

    # now collect the files in each of the folders and enforce that all files are present in all folders
    files_per_folder = [set(subfiles(i, suffix='.npz', join=False)) for i in list_of_input_folders]
    # first build a set with all files
    s = deepcopy(files_per_folder[0])
    for f in files_per_folder[1:]:
        s.update(f)
    for f in files_per_folder:
        assert len(s.difference(f)) == 0, "Not all folders contain the same files for ensembling. Please only " \
                                          "provide folders that contain the predictions"
    lists_of_lists_of_files = [[join(fl, fi) for fl in list_of_input_folders] for fi in s]
    output_files_truncated = [join(output_folder, fi[:-4]) for fi in s]

    image_reader_writer = recursive_find_reader_writer_by_name(plans["image_reader_writer"])()

    maybe_mkdir_p(output_folder)

    pool = Pool(num_processes)
    num_preds = len(s)
    _ = pool.starmap(
        merge_files,
        zip(
            lists_of_lists_of_files, output_files_truncated, [dataset_json['file_ending']] * num_preds,
            [image_reader_writer] * num_preds, [dataset_json['labels']] * num_preds,
            [dataset_json.get('regions_class_order')] * num_preds,
            [save_merged_probabilities] * num_preds
        )
    )
    pool.close()
    pool.join()


def ensemble_crossvalidations(list_of_trained_model_folders: List[str],
                              output_folder: str,
                              folds: Union[Tuple[int, ...], List[int]] = (0, 1, 2, 3, 4),
                              num_processes: int = default_num_processes) -> None:
    """
    Feature: different configurations can now have different splits
    """
    dataset_json = load_json(join(list_of_trained_model_folders[0], 'dataset.json'))
    plans = load_json(join(list_of_trained_model_folders[0], 'plans.json'))

    # first collect all unique filenames
    files_per_folder = {}
    unique_filenames = set()
    for tr in list_of_trained_model_folders:
        files_per_folder[tr] = {}
        for f in folds:
            if not isdir(join(tr, f'fold_{f}', 'validation')):
                raise RuntimeError(f'Expected model output directory does not exist. You must train all requested '
                                   f'folds of the speficied model.\nModel: {tr}\nFold: {f}')
            files_here = subfiles(join(tr, f'fold_{f}', 'validation'), suffix='.npz', join=False)
            if len(files_here) == 0:
                raise RuntimeError(f"No .npz files found in folder {join(tr, f'fold_{f}', 'validation')}. Rerun your "
                                   f"validation with the --npz flag. Use nnUNetv2_train [...] --val --npz.")
            files_per_folder[tr][f] = subfiles(join(tr, f'fold_{f}', 'validation'), suffix='.npz', join=False)
            unique_filenames.update(files_per_folder[tr][f])

    # verify that all trained_model_folders have all predictions
    ok = True
    for tr, fi in files_per_folder.items():
        all_files_here = set()
        for f in folds:
            all_files_here.update(fi[f])
        diff = unique_filenames.difference(all_files_here)
        if len(diff) > 0:
            ok = False
            print(f'model {tr} does not seem to contain all predictions. Missing: {diff}')
        if not ok:
            raise RuntimeError('There were missing files, see print statements above this one')

    # now we need to collect where these files are
    file_mapping = []
    for tr in list_of_trained_model_folders:
        file_mapping.append({})
        for f in folds:
            for fi in files_per_folder[tr][f]:
                # check for duplicates
                assert fi not in file_mapping[-1].keys(), f"Duplicate detected. Case {fi} is present in more than " \
                                                          f"one fold of model {tr}."
                file_mapping[-1][fi] = join(tr, f'fold_{f}', 'validation')

    lists_of_lists_of_files = [[fm[i] for i in unique_filenames] for fm in file_mapping]
    output_files_truncated = [join(output_folder, fi[:-4]) for fi in unique_filenames]

    image_reader_writer = recursive_find_reader_writer_by_name(plans["image_reader_writer"])()
    maybe_mkdir_p(output_folder)

    pool = Pool(num_processes)
    num_preds = len(unique_filenames)
    _ = pool.starmap(
        merge_files,
        zip(
            lists_of_lists_of_files,
            output_files_truncated,
            [dataset_json['file_ending']] * num_preds,
            [image_reader_writer] * num_preds,
            [dataset_json['labels']] * num_preds,
            [dataset_json.get('regions_class_order')] * num_preds,
            [False] * num_preds
        )
    )
    pool.close()
    pool.join()
