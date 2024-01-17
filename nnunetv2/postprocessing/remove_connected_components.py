import argparse
import multiprocessing
import shutil
from multiprocessing import Pool
from typing import Union, Tuple, List, Callable

import numpy as np
from acvl_utils.morphology.morphology_helper import remove_all_but_largest_component
from batchgenerators.utilities.file_and_folder_operations import load_json, subfiles, maybe_mkdir_p, join, isfile, \
    isdir, save_pickle, load_pickle, save_json
from nnunetv2.configuration import default_num_processes
from nnunetv2.evaluation.accumulate_cv_results import accumulate_cv_results
from nnunetv2.evaluation.evaluate_predictions import region_or_label_to_mask, compute_metrics_on_folder, \
    load_summary_json, label_or_region_to_key
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.file_path_utilities import folds_tuple_to_string
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


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
                             plans_file_or_dict: Union[str, dict],
                             dataset_json_file_or_dict: Union[str, dict],
                             num_processes: int = default_num_processes,
                             keep_postprocessed_files: bool = True):
    """
    Determines nnUNet postprocessing. Its output is a postprocessing.pkl file in folder_predictions which can be
    used with apply_postprocessing_to_folder.

    Postprocessed files are saved in folder_predictions/postprocessed. Set
    keep_postprocessed_files=False to delete these files after this function is done (temp files will eb created
    and deleted regardless).

    If plans_file_or_dict or dataset_json_file_or_dict are None, we will look for them in input_folder
    """
    output_folder = join(folder_predictions, 'postprocessed')

    if plans_file_or_dict is None:
        expected_plans_file = join(folder_predictions, 'plans.json')
        if not isfile(expected_plans_file):
            raise RuntimeError(f"Expected plans file missing: {expected_plans_file}. The plans files should have been "
                               f"created while running nnUNetv2_predict. Sadge.")
        plans_file_or_dict = load_json(expected_plans_file)
    plans_manager = PlansManager(plans_file_or_dict)

    if dataset_json_file_or_dict is None:
        expected_dataset_json_file = join(folder_predictions, 'dataset.json')
        if not isfile(expected_dataset_json_file):
            raise RuntimeError(
                f"Expected plans file missing: {expected_dataset_json_file}. The plans files should have been "
                f"created while running nnUNetv2_predict. Sadge.")
        dataset_json_file_or_dict = load_json(expected_dataset_json_file)

    if not isinstance(dataset_json_file_or_dict, dict):
        dataset_json = load_json(dataset_json_file_or_dict)
    else:
        dataset_json = dataset_json_file_or_dict

    rw = plans_manager.image_reader_writer_class()
    label_manager = plans_manager.get_label_manager(dataset_json)
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
    with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
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
        # 2023_02_06: I hate myself for the comment above. Thanks past me
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
                          f'Dice before: {round(baseline_results["mean"][label_or_region]["Dice"], 5)} '
                          f'after: {round(pp_results["mean"][label_or_region]["Dice"], 5)}')
                    if isdir(join(output_folder, 'temp', 'keep_largest_perClassOrRegion_currentBest')):
                        shutil.rmtree(join(output_folder, 'temp', 'keep_largest_perClassOrRegion_currentBest'))
                    shutil.move(output_here, join(output_folder, 'temp', 'keep_largest_perClassOrRegion_currentBest'), )
                    source = join(output_folder, 'temp', 'keep_largest_perClassOrRegion_currentBest')
                    pp_fns.append(pp_fn)
                    pp_fn_kwargs.append(kwargs)
                else:
                    print(f'Removing all but the largest component for {label_or_region} did not improve results! '
                          f'Dice before: {round(baseline_results["mean"][label_or_region]["Dice"], 5)} '
                          f'after: {round(pp_results["mean"][label_or_region]["Dice"], 5)}')
    [shutil.copy(join(source, i), join(output_folder, i)) for i in subfiles(source, join=False)]
    save_pickle((pp_fns, pp_fn_kwargs), join(folder_predictions, 'postprocessing.pkl'))

    baseline_results = load_summary_json(join(folder_predictions, 'summary.json'))
    final_results = load_summary_json(join(output_folder, 'summary.json'))
    tmp = {
        'input_folder': {i: baseline_results[i] for i in ['foreground_mean', 'mean']},
        'postprocessed': {i: final_results[i] for i in ['foreground_mean', 'mean']},
        'postprocessing_fns': [i.__name__ for i in pp_fns],
        'postprocessing_kwargs': pp_fn_kwargs,
    }
    # json is a very annoying little bi###. Can't handle tuples as dict keys.
    tmp['input_folder']['mean'] = {label_or_region_to_key(k): tmp['input_folder']['mean'][k] for k in
                                   tmp['input_folder']['mean'].keys()}
    tmp['postprocessed']['mean'] = {label_or_region_to_key(k): tmp['postprocessed']['mean'][k] for k in
                                    tmp['postprocessed']['mean'].keys()}
    # did I already say that I hate json? "TypeError: Object of type int64 is not JSON serializable" You retarded bro?
    recursive_fix_for_json_export(tmp)
    save_json(tmp, join(folder_predictions, 'postprocessing.json'))

    shutil.rmtree(join(output_folder, 'temp'))

    if not keep_postprocessed_files:
        shutil.rmtree(output_folder)
    return pp_fns, pp_fn_kwargs


def apply_postprocessing_to_folder(input_folder: str,
                                   output_folder: str,
                                   pp_fns: List[Callable],
                                   pp_fn_kwargs: List[dict],
                                   plans_file_or_dict: Union[str, dict] = None,
                                   dataset_json_file_or_dict: Union[str, dict] = None,
                                   num_processes=8) -> None:
    """
    If plans_file_or_dict or dataset_json_file_or_dict are None, we will look for them in input_folder
    """
    if plans_file_or_dict is None:
        expected_plans_file = join(input_folder, 'plans.json')
        if not isfile(expected_plans_file):
            raise RuntimeError(f"Expected plans file missing: {expected_plans_file}. The plans file should have been "
                               f"created while running nnUNetv2_predict. Sadge. If the folder you want to apply "
                               f"postprocessing to was create from an ensemble then just specify one of the "
                               f"plans files of the ensemble members in plans_file_or_dict")
        plans_file_or_dict = load_json(expected_plans_file)
    plans_manager = PlansManager(plans_file_or_dict)

    if dataset_json_file_or_dict is None:
        expected_dataset_json_file = join(input_folder, 'dataset.json')
        if not isfile(expected_dataset_json_file):
            raise RuntimeError(
                f"Expected plans file missing: {expected_dataset_json_file}. The dataset.json should have been "
                f"copied while running nnUNetv2_predict/nnUNetv2_ensemble. Sadge.")
        dataset_json_file_or_dict = load_json(expected_dataset_json_file)

    if not isinstance(dataset_json_file_or_dict, dict):
        dataset_json = load_json(dataset_json_file_or_dict)
    else:
        dataset_json = dataset_json_file_or_dict

    rw = plans_manager.image_reader_writer_class()

    maybe_mkdir_p(output_folder)
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        files = subfiles(input_folder, suffix=dataset_json['file_ending'], join=False)

        _ = p.starmap(load_postprocess_save,
                      zip(
                          [join(input_folder, i) for i in files],
                          [join(output_folder, i) for i in files],
                          [rw] * len(files),
                          [pp_fns] * len(files),
                          [pp_fn_kwargs] * len(files)
                      )
                      )


def entry_point_determine_postprocessing_folder():
    parser = argparse.ArgumentParser('Writes postprocessing.pkl and postprocessing.json in input_folder.')
    parser.add_argument('-i', type=str, required=True, help='Input folder')
    parser.add_argument('-ref', type=str, required=True, help='Folder with gt labels')
    parser.add_argument('-plans_json', type=str, required=False, default=None,
                        help="plans file to use. If not specified we will look for the plans.json file in the "
                             "input folder (input_folder/plans.json)")
    parser.add_argument('-dataset_json', type=str, required=False, default=None,
                        help="dataset.json file to use. If not specified we will look for the dataset.json file in the "
                             "input folder (input_folder/dataset.json)")
    parser.add_argument('-np', type=int, required=False, default=default_num_processes,
                        help=f"number of processes to use. Default: {default_num_processes}")
    parser.add_argument('--remove_postprocessed', action='store_true', required=False,
                        help='set this is you don\'t want to keep the postprocessed files')

    args = parser.parse_args()
    determine_postprocessing(args.i, args.ref, args.plans_json, args.dataset_json, args.np,
                             not args.remove_postprocessed)


def entry_point_apply_postprocessing():
    parser = argparse.ArgumentParser('Apples postprocessing specified in pp_pkl_file to input folder.')
    parser.add_argument('-i', type=str, required=True, help='Input folder')
    parser.add_argument('-o', type=str, required=True, help='Output folder')
    parser.add_argument('-pp_pkl_file', type=str, required=True, help='postprocessing.pkl file')
    parser.add_argument('-np', type=int, required=False, default=default_num_processes,
                        help=f"number of processes to use. Default: {default_num_processes}")
    parser.add_argument('-plans_json', type=str, required=False, default=None,
                        help="plans file to use. If not specified we will look for the plans.json file in the "
                             "input folder (input_folder/plans.json)")
    parser.add_argument('-dataset_json', type=str, required=False, default=None,
                        help="dataset.json file to use. If not specified we will look for the dataset.json file in the "
                             "input folder (input_folder/dataset.json)")
    args = parser.parse_args()
    pp_fns, pp_fn_kwargs = load_pickle(args.pp_pkl_file)
    apply_postprocessing_to_folder(args.i, args.o, pp_fns, pp_fn_kwargs, args.plans_json, args.dataset_json, args.np)


if __name__ == '__main__':
    trained_model_folder = '/home/fabian/results/nnUNet_remake/Dataset004_Hippocampus/nnUNetTrainer__nnUNetPlans__3d_fullres'
    labelstr = join(nnUNet_raw, 'Dataset004_Hippocampus', 'labelsTr')
    plans_manager = PlansManager(join(trained_model_folder, 'plans.json'))
    dataset_json = load_json(join(trained_model_folder, 'dataset.json'))
    folds = (0, 1, 2, 3, 4)
    label_manager = plans_manager.get_label_manager(dataset_json)

    merged_output_folder = join(trained_model_folder, f'crossval_results_folds_{folds_tuple_to_string(folds)}')
    accumulate_cv_results(trained_model_folder, merged_output_folder, folds, 8, False)

    fns, kwargs = determine_postprocessing(merged_output_folder, labelstr, plans_manager.plans,
                                           dataset_json, 8, keep_postprocessed_files=True)
    save_pickle((fns, kwargs), join(trained_model_folder, 'postprocessing.pkl'))
    fns, kwargs = load_pickle(join(trained_model_folder, 'postprocessing.pkl'))

    apply_postprocessing_to_folder(merged_output_folder, merged_output_folder + '_pp', fns, kwargs,
                                   plans_manager.plans, dataset_json,
                                   8)
    compute_metrics_on_folder(labelstr,
                              merged_output_folder + '_pp',
                              join(merged_output_folder + '_pp', 'summary.json'),
                              plans_manager.image_reader_writer_class(),
                              dataset_json['file_ending'],
                              label_manager.foreground_regions if label_manager.has_regions else label_manager.foreground_labels,
                              label_manager.ignore_label,
                              8)
