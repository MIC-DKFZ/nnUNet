import shutil
from multiprocessing import Pool
from typing import Union, Tuple, List, Callable

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_json, subfiles, maybe_mkdir_p, join, isfile, isdir

from nnunetv2.evaluation.evaluate_predictions import region_or_label_to_mask, compute_metrics_on_folder, \
    load_summary_json
from acvl_utils.morphology.morphology_helper import remove_all_but_largest_component
from nnunetv2.configuration import default_num_processes
from nnunetv2.evaluation.find_best_configuration import folds_tuple_to_string, accumulate_cv_results
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.imageio.reader_writer_registry import recursive_find_reader_writer_by_name
from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.label_handling.label_handling import get_labelmanager


def remove_all_but_largest_component_from_segmentation(segmentation: np.ndarray,
                                                      labels_or_regions: Union[int, Tuple[int, ...],
                                                                               List[Union[int, Tuple[int, ...]]]],
                                                      background_label: int = 0) -> np.ndarray:
    mask = np.zeros_like(segmentation, dtype=bool)
    if not isinstance(labels_or_regions, list):
        labels_or_regions = [labels_or_regions]
    for l_or_r in labels_or_regions:
        mask |= region_or_label_to_mask(segmentation, l_or_r)
    mask_keep = remove_all_but_largest_component(mask)
    ret = np.copy(segmentation)  # do not modify the input!
    ret[mask & ~mask_keep] = background_label
    return ret


def apply_postprocessing(segmentation: np.ndarray, pp_fns: List[Callable], pp_fn_kwargs: List[dict]):
    for fn, kwargs in zip(pp_fns, pp_fn_kwargs):
        segmentation = fn(segmentation, **kwargs)
    return segmentation


def load_postprocess_save(segmentation_file: str,
                          output_fname: str,
                          image_reader_writer: BaseReaderWriter,
                          pp_fns: List[Callable],
                          pp_fn_kwargs: List[dict]):
    seg, props = image_reader_writer.read_seg(segmentation_file)
    seg = apply_postprocessing(seg[0], pp_fns, pp_fn_kwargs)
    image_reader_writer.write_seg(seg, output_fname, props)


def determine_postprocessing(folder_predictions: str,
                             folder_ref: str,
                             output_folder: str,
                             plans_file_or_dict: Union[str, dict],
                             dataset_json_file_or_dict: Union[str, dict],
                             num_processes: int = default_num_processes):
    if not isinstance(plans_file_or_dict, dict):
        plans = load_json(plans_file_or_dict)
    else:
        plans = plans_file_or_dict
    if not isinstance(dataset_json_file_or_dict, dict):
        dataset_json = load_json(dataset_json_file_or_dict)
    else:
        dataset_json = dataset_json_file_or_dict

    rw = recursive_find_reader_writer_by_name(plans["image_reader_writer"])()
    label_manager = get_labelmanager(plans, dataset_json)
    labels_or_regions = label_manager.foreground_regions if label_manager.has_regions else label_manager.foreground_labels

    predicted_files = subfiles(folder_predictions, suffix=dataset_json['file_ending'], join=False)
    ref_files = subfiles(folder_ref, suffix=dataset_json['file_ending'], join=False)
    # we should print a warning if not all files from folder_ref are present in folder_predictions
    if not all([i in predicted_files for i in ref_files]):
        print(f'WARNING: Not all files in folder_ref were found in folder_predictions. Determining postprocessing '
              f'should always be done on the entire dataset!')

    # before we start we should evaluate the imaegs in the source folder
    if not isfile(join(folder_predictions, 'summary.json')):
        compute_metrics_on_folder(folder_ref,
                                  folder_predictions,
                                  join(folder_predictions, 'summary.json'),
                                  rw,
                                  dataset_json['file_ending'],
                                  labels_or_regions,
                                  label_manager.ignore_label,
                                  num_processes)

    # we save the postprocessing functions in here
    pp_fns = []
    pp_fn_kwargs = []

    # pool party!
    pool = Pool(num_processes)

    # now let's see whether removing all but the largest foreground region improves the scores
    output_here = join(output_folder, 'temp', 'keep_largest_fg')
    maybe_mkdir_p(output_here)
    pp_fn = remove_all_but_largest_component_from_segmentation
    kwargs = {
        'labels_or_regions': label_manager.foreground_labels,
    }

    pool.starmap(
        load_postprocess_save,
        zip(
            [join(folder_predictions, i) for i in predicted_files],
            [join(output_here, i) for i in predicted_files],
            [rw] * len(predicted_files),
            [[pp_fn]] * len(predicted_files),
            [[kwargs]] * len(predicted_files)
        )
    )
    compute_metrics_on_folder(folder_ref,
                              output_here,
                              join(output_here, 'summary.json'),
                              rw,
                              dataset_json['file_ending'],
                              labels_or_regions,
                              label_manager.ignore_label,
                              num_processes)
    # now we need to figure out if doing this improved the dice scores. We will implement that defensively in so far
    # that if a single class got worse as a result we won't do this. We can change this in the future but right now I
    # prefer to do it this way
    baseline_results = load_summary_json(join(folder_predictions, 'summary.json'))
    pp_results = load_summary_json(join(output_here, 'summary.json'))
    do_this = pp_results['foreground_mean']['Dice'] > baseline_results['foreground_mean']['Dice']
    if do_this:
        for class_id in pp_results['mean'].keys():
            if pp_results['mean'][class_id]['Dice'] < baseline_results['mean'][class_id]['Dice']:
                do_this = False
                break
    if do_this:
        print(f'Results were improved by removing all but the largest foreground region. '
              f'Mean dice before: {round(baseline_results["foreground_mean"]["Dice"], 5)} '
              f'after: {round(pp_results["foreground_mean"]["Dice"], 5)}')
        source = output_here
        pp_fns.append(pp_fn)
        pp_fn_kwargs.append(kwargs)
    else:
        print(f'Removing all but the largest foreground region did not improve results!')
        source = folder_predictions

    # in the old nnU-Net we could just apply all-but-largest component removal to all classes at the same time and
    # then evaluate for each class whether this improved results. This is no longer possible because we now support
    # region-based predictions and regions can overlap, causing interactions
    # in principle the order with which the postprocessing is applied to the regions matter as well and should be
    # investigated, but due to some things that I am too lazy to explain right now it's going to be alright (I think)
    # to stick to the order in which they are declared in dataset.json (if you want to think about it then think about
    # region_class_order)
    if len(labels_or_regions) > 1:
        for label_or_region in labels_or_regions:
            pp_fn = remove_all_but_largest_component_from_segmentation
            kwargs = {
                'labels_or_regions': label_or_region,
            }

            output_here = join(output_folder, 'temp', 'keep_largest_perClassOrRegion')
            maybe_mkdir_p(output_here)

            pool.starmap(
                load_postprocess_save,
                zip(
                    [join(source, i) for i in predicted_files],
                    [join(output_here, i) for i in predicted_files],
                    [rw] * len(predicted_files),
                    [[pp_fn]] * len(predicted_files),
                    [[kwargs]] * len(predicted_files)
                )
            )
            compute_metrics_on_folder(folder_ref,
                                      output_here,
                                      join(output_here, 'summary.json'),
                                      rw,
                                      dataset_json['file_ending'],
                                      labels_or_regions,
                                      label_manager.ignore_label,
                                      num_processes)
            baseline_results = load_summary_json(join(source, 'summary.json'))
            pp_results = load_summary_json(join(output_here, 'summary.json'))
            do_this = pp_results['mean'][label_or_region]['Dice'] > baseline_results['mean'][label_or_region]['Dice']
            if do_this:
                print(f'Results were improved by removing all but the largest component for {label_or_region}. '
                      f'Mean dice before: {round(baseline_results["mean"][label_or_region]["Dice"], 5)} '
                      f'after: {round(pp_results["mean"][label_or_region]["Dice"], 5)}')
                if isdir(join(output_folder, 'temp', 'keep_largest_perClassOrRegion_currentBest')):
                    shutil.rmtree(join(output_folder, 'temp', 'keep_largest_perClassOrRegion_currentBest'))
                shutil.move(output_here, join(output_folder, 'temp', 'keep_largest_perClassOrRegion_currentBest'), )
                source = join(output_folder, 'temp', 'keep_largest_perClassOrRegion_currentBest')
                pp_fns.append(pp_fn)
                pp_fn_kwargs.append(kwargs)
            else:
                print(f'Removing all but the largest component for {label_or_region} did not improve results! '                      
                      f'Mean dice before: {round(baseline_results["mean"][label_or_region]["Dice"], 5)} '
                      f'after: {round(pp_results["mean"][label_or_region]["Dice"], 5)}')
    [shutil.move(join(source, i), join(output_folder, i)) for i in subfiles(source, join=False)]
    shutil.rmtree(join(output_folder, 'temp'))
    return pp_fns, pp_fn_kwargs


if __name__ == '__main__':
    trained_model_folder = '/home/fabian/results/nnUNet_remake/Dataset004_Hippocampus/nnUNetTrainer__nnUNetPlans__3d_fullres'
    folds = (0, 1, 2, 3, 4)
    merged_output_folder = join(trained_model_folder, f'crossval_results_folds_{folds_tuple_to_string(folds)}')
    accumulate_cv_results(trained_model_folder, merged_output_folder, folds, 8, True)

    fns, kwargs = determine_postprocessing(merged_output_folder, join(nnUNet_raw, 'Dataset004_Hippocampus', 'labelsTr'),
                             join(trained_model_folder, 'postprocessed'), join(trained_model_folder, 'plans.json'),
                             join(trained_model_folder, 'dataset.json'), 8)