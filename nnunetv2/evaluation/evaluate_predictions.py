import multiprocessing
import os
from copy import deepcopy
from typing import Tuple, List, Union

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, save_json, load_json, \
    isfile
from nnunetv2.configuration import default_num_processes
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json, \
    determine_reader_writer_from_file_ending
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
# the Evaluator class of the previous nnU-Net was great and all but man was it overengineered. Keep it simple
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


def label_or_region_to_key(label_or_region: Union[int, Tuple[int]]):
    return str(label_or_region)


def key_to_label_or_region(key: str):
    try:
        return int(key)
    except ValueError:
        key = key.replace('(', '')
        key = key.replace(')', '')
        split = key.split(',')
        return tuple([int(i) for i in split if len(i) > 0])


def save_summary_json(results: dict, output_file: str):
    """
    json does not support tuples as keys (why does it have to be so shitty) so we need to convert that shit
    ourselves
    """
    results_converted = deepcopy(results)
    # convert keys in mean metrics
    results_converted['mean'] = {label_or_region_to_key(k): results['mean'][k] for k in results['mean'].keys()}
    # convert metric_per_case
    for i in range(len(results_converted["metric_per_case"])):
        results_converted["metric_per_case"][i]['metrics'] = \
            {label_or_region_to_key(k): results["metric_per_case"][i]['metrics'][k]
             for k in results["metric_per_case"][i]['metrics'].keys()}
    # sort_keys=True will make foreground_mean the first entry and thus easy to spot
    save_json(results_converted, output_file, sort_keys=True)


def load_summary_json(filename: str):
    results = load_json(filename)
    # convert keys in mean metrics
    results['mean'] = {key_to_label_or_region(k): results['mean'][k] for k in results['mean'].keys()}
    # convert metric_per_case
    for i in range(len(results["metric_per_case"])):
        results["metric_per_case"][i]['metrics'] = \
            {key_to_label_or_region(k): results["metric_per_case"][i]['metrics'][k]
             for k in results["metric_per_case"][i]['metrics'].keys()}
    return results


def labels_to_list_of_regions(labels: List[int]):
    return [(i,) for i in labels]


def region_or_label_to_mask(segmentation: np.ndarray, region_or_label: Union[int, Tuple[int, ...]]) -> np.ndarray:
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    else:
        mask = np.zeros_like(segmentation, dtype=bool)
        for r in region_or_label:
            mask[segmentation == r] = True
    return mask


def compute_tp_fp_fn_tn(mask_ref: np.ndarray, mask_pred: np.ndarray, ignore_mask: np.ndarray = None):
    if ignore_mask is None:
        use_mask = np.ones_like(mask_ref, dtype=bool)
    else:
        use_mask = ~ignore_mask
    tp = np.sum((mask_ref & mask_pred) & use_mask)
    fp = np.sum(((~mask_ref) & mask_pred) & use_mask)
    fn = np.sum((mask_ref & (~mask_pred)) & use_mask)
    tn = np.sum(((~mask_ref) & (~mask_pred)) & use_mask)
    return tp, fp, fn, tn


def _labels_to_bin_indices(segmentation_chunk: np.ndarray, lut: np.ndarray, lut_offset: int) -> np.ndarray:
    """
    Maps the values of segmentation_chunk to confusion matrix bins through lut. lut covers the value range
    [lut_offset + 1, lut_offset + len(lut) - 2], its first and last entries catch everything below and above.
    """
    chunk_int = segmentation_chunk.astype(np.intp)
    if not np.issubdtype(segmentation_chunk.dtype, np.integer) and not np.array_equal(chunk_int, segmentation_chunk):
        # float to int casting would silently merge e.g. 1.5 into label 1
        raise ValueError('Segmentations must only contain integer values')
    if lut_offset != 0:
        chunk_int -= lut_offset
    np.clip(chunk_int, 0, len(lut) - 1, out=chunk_int)
    return lut[chunk_int]


def compute_confusion_matrix(seg_ref: np.ndarray, seg_pred: np.ndarray, labels: Union[List[int], Tuple[int, ...]],
                             ignore_label: int = None, chunk_size: int = 2 ** 22) -> np.ndarray:
    """
    Returns the confusion matrix of seg_ref (rows) and seg_pred (columns), shape (len(labels) + 1, len(labels) + 1).
    Index i < len(labels) stands for labels[i], the last index collects all values that are not in labels. Voxels
    where seg_ref == ignore_label are not counted at all.

    labels must be unique. The volume is processed in chunks of chunk_size voxels so that the temporary integer
    arrays stay small, no matter how large the images are.
    """
    if seg_ref.shape != seg_pred.shape:
        raise ValueError(f'Shape mismatch between reference {seg_ref.shape} and prediction {seg_pred.shape}')
    labels = [int(i) for i in labels]
    assert len(set(labels)) == len(labels), f'labels must be unique, got {labels}'
    num_labels = len(labels)
    other_bin = num_labels
    ignored_bin = num_labels + 1
    num_bins = num_labels + 2

    # lut entry i corresponds to value i + lut_offset. We only need to cover the values we care about, everything
    # else is clipped onto the first or last entry, which both point to other_bin
    special_values = labels + ([ignore_label] if ignore_label is not None else [])
    if len(special_values) == 0:
        # nothing to distinguish, everything goes into other_bin
        special_values = [0]
    lut_offset = min(special_values) - 1
    lut_len = max(special_values) - lut_offset + 2
    lut_pred = np.full(lut_len, other_bin, dtype=np.intp)
    lut_pred[np.array(labels, dtype=np.intp) - lut_offset] = np.arange(num_labels)
    # the ignore label takes precedence in the reference, even if it is also in labels. Predictions of the ignore
    # label are not special, they are treated like any other value
    lut_ref = lut_pred.copy()
    if ignore_label is not None:
        lut_ref[ignore_label - lut_offset] = ignored_bin

    ref_flat = seg_ref.ravel()
    pred_flat = seg_pred.ravel()
    confusion_matrix = np.zeros(num_bins * num_bins, dtype=np.int64)
    for start in range(0, ref_flat.size, chunk_size):
        bin_idx = _labels_to_bin_indices(ref_flat[start:start + chunk_size], lut_ref, lut_offset)
        bin_idx *= num_bins
        bin_idx += _labels_to_bin_indices(pred_flat[start:start + chunk_size], lut_pred, lut_offset)
        confusion_matrix += np.bincount(bin_idx, minlength=num_bins * num_bins)
    # predictions never land in ignored_bin, so dropping its row removes the ignored voxels entirely
    return confusion_matrix.reshape(num_bins, num_bins)[:ignored_bin, :ignored_bin]


def compute_tp_fp_fn_tn_for_labels_or_regions(seg_ref: np.ndarray, seg_pred: np.ndarray,
                                              labels_or_regions: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                                              ignore_label: int = None) -> dict:
    """
    Same result as running region_or_label_to_mask + compute_tp_fp_fn_tn for each label or region, but derived
    from a single confusion matrix. That makes the cost almost independent of the number of labels/regions.
    Returns {label_or_region: (tp, fp, fn, tn)}
    """
    members = {r: (r,) if np.isscalar(r) else tuple(r) for r in labels_or_regions}
    labels = sorted({int(l) for m in members.values() for l in m})
    label_to_bin = {l: i for i, l in enumerate(labels)}
    confusion_matrix = compute_confusion_matrix(seg_ref, seg_pred, labels, ignore_label)
    num_voxels = confusion_matrix.sum()

    counts = {}
    for r, m in members.items():
        # set: a region listing a label twice must not count its voxels twice
        bins = sorted({label_to_bin[int(l)] for l in m})
        tp = confusion_matrix[np.ix_(bins, bins)].sum()
        n_ref = confusion_matrix[bins, :].sum()
        n_pred = confusion_matrix[:, bins].sum()
        counts[r] = (tp, n_pred - tp, n_ref - tp, num_voxels - n_ref - n_pred + tp)
    return counts


def compute_metrics(reference_file: str, prediction_file: str, image_reader_writer: BaseReaderWriter,
                    labels_or_regions: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                    ignore_label: int = None) -> dict:
    # load images
    seg_ref, seg_ref_dict = image_reader_writer.read_seg(reference_file)
    seg_pred, seg_pred_dict = image_reader_writer.read_seg(prediction_file)

    counts = compute_tp_fp_fn_tn_for_labels_or_regions(seg_ref, seg_pred, labels_or_regions, ignore_label)

    results = {}
    results['reference_file'] = reference_file
    results['prediction_file'] = prediction_file
    results['metrics'] = {}
    for r in labels_or_regions:
        results['metrics'][r] = {}
        tp, fp, fn, tn = counts[r]
        if tp + fp + fn == 0:
            results['metrics'][r]['Dice'] = np.nan
            results['metrics'][r]['IoU'] = np.nan
        else:
            results['metrics'][r]['Dice'] = 2 * tp / (2 * tp + fp + fn)
            results['metrics'][r]['IoU'] = tp / (tp + fp + fn)
        results['metrics'][r]['FP'] = fp
        results['metrics'][r]['TP'] = tp
        results['metrics'][r]['FN'] = fn
        results['metrics'][r]['TN'] = tn
        results['metrics'][r]['n_pred'] = fp + tp
        results['metrics'][r]['n_ref'] = fn + tp
    return results


def compute_metrics_on_folder(folder_ref: str, folder_pred: str, output_file: str,
                              image_reader_writer: BaseReaderWriter,
                              file_ending: str,
                              regions_or_labels: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                              ignore_label: int = None,
                              num_processes: int = default_num_processes,
                              chill: bool = True) -> dict:
    """
    output_file must end with .json; can be None
    """
    if output_file is not None:
        assert output_file.endswith('.json'), 'output_file should end with .json'
    files_pred = subfiles(folder_pred, suffix=file_ending, join=False)
    files_ref = subfiles(folder_ref, suffix=file_ending, join=False)
    if not chill:
        present = [isfile(join(folder_pred, i)) for i in files_ref]
        assert all(present), "Not all files in folder_ref exist in folder_pred"
    files_ref = [join(folder_ref, i) for i in files_pred]
    files_pred = [join(folder_pred, i) for i in files_pred]
    with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
        # for i in list(zip(files_ref, files_pred, [image_reader_writer] * len(files_pred), [regions_or_labels] * len(files_pred), [ignore_label] * len(files_pred))):
        #     compute_metrics(*i)
        results = pool.starmap(
            compute_metrics,
            list(zip(files_ref, files_pred, [image_reader_writer] * len(files_pred), [regions_or_labels] * len(files_pred),
                     [ignore_label] * len(files_pred)))
        )

    # mean metric per class
    metric_list = list(results[0]['metrics'][regions_or_labels[0]].keys())
    means = {}
    for r in regions_or_labels:
        means[r] = {}
        for m in metric_list:
            means[r][m] = np.nanmean([i['metrics'][r][m] for i in results])

    # foreground mean
    foreground_mean = {}
    for m in metric_list:
        values = []
        for k in means.keys():
            if k == 0 or k == '0':
                continue
            values.append(means[k][m])
        foreground_mean[m] = np.mean(values)

    [recursive_fix_for_json_export(i) for i in results]
    recursive_fix_for_json_export(means)
    recursive_fix_for_json_export(foreground_mean)
    result = {'metric_per_case': results, 'mean': means, 'foreground_mean': foreground_mean}
    if output_file is not None:
        save_summary_json(result, output_file)
    return result
    # print('DONE')


def compute_metrics_on_folder2(folder_ref: str, folder_pred: str, dataset_json_file: str, plans_file: str,
                               output_file: str = None,
                               num_processes: int = default_num_processes,
                               chill: bool = False):
    dataset_json = load_json(dataset_json_file)
    # get file ending
    file_ending = dataset_json['file_ending']

    # get reader writer class
    example_file = subfiles(folder_ref, suffix=file_ending, join=True)[0]
    rw = determine_reader_writer_from_dataset_json(dataset_json, example_file)()

    # maybe auto set output file
    if output_file is None:
        output_file = join(folder_pred, 'summary.json')

    lm = PlansManager(plans_file).get_label_manager(dataset_json)
    compute_metrics_on_folder(folder_ref, folder_pred, output_file, rw, file_ending,
                              lm.foreground_regions if lm.has_regions else lm.foreground_labels, lm.ignore_label,
                              num_processes, chill=chill)


def compute_metrics_on_folder_simple(folder_ref: str, folder_pred: str, labels: Union[Tuple[int, ...], List[int]],
                                     output_file: str = None,
                                     num_processes: int = default_num_processes,
                                     ignore_label: int = None,
                                     chill: bool = False):
    example_file = subfiles(folder_ref, join=True)[0]
    file_ending = os.path.splitext(example_file)[-1]
    rw = determine_reader_writer_from_file_ending(file_ending, example_file, allow_nonmatching_filename=True,
                                                  verbose=False)()
    # maybe auto set output file
    if output_file is None:
        output_file = join(folder_pred, 'summary.json')
    compute_metrics_on_folder(folder_ref, folder_pred, output_file, rw, file_ending,
                              labels, ignore_label=ignore_label, num_processes=num_processes, chill=chill)


def evaluate_folder_entry_point():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_folder', type=str, help='folder with gt segmentations')
    parser.add_argument('pred_folder', type=str, help='folder with predicted segmentations')
    parser.add_argument('-djfile', type=str, required=True,
                        help='dataset.json file')
    parser.add_argument('-pfile', type=str, required=True,
                        help='plans.json file')
    parser.add_argument('-o', type=str, required=False, default=None,
                        help='Output file. Optional. Default: pred_folder/summary.json')
    parser.add_argument('-np', type=int, required=False, default=default_num_processes,
                        help=f'number of processes used. Optional. Default: {default_num_processes}')
    parser.add_argument('--chill', action='store_true', help='dont crash if folder_pred does not have all files that are present in folder_gt')
    args = parser.parse_args()
    compute_metrics_on_folder2(args.gt_folder, args.pred_folder, args.djfile, args.pfile, args.o, args.np, chill=args.chill)


def evaluate_simple_entry_point():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_folder', type=str, help='folder with gt segmentations')
    parser.add_argument('pred_folder', type=str, help='folder with predicted segmentations')
    parser.add_argument('-l', type=int, nargs='+', required=True,
                        help='list of labels')
    parser.add_argument('-il', type=int, required=False, default=None,
                        help='ignore label')
    parser.add_argument('-o', type=str, required=False, default=None,
                        help='Output file. Optional. Default: pred_folder/summary.json')
    parser.add_argument('-np', type=int, required=False, default=default_num_processes,
                        help=f'number of processes used. Optional. Default: {default_num_processes}')
    parser.add_argument('--chill', action='store_true', help='dont crash if folder_pred does not have all files that are present in folder_gt')

    args = parser.parse_args()
    compute_metrics_on_folder_simple(args.gt_folder, args.pred_folder, args.l, args.o, args.np, args.il, chill=args.chill)


if __name__ == '__main__':
    folder_ref = '/media/fabian/data/nnUNet_raw/Dataset004_Hippocampus/labelsTr'
    folder_pred = '/home/fabian/results/nnUNet_remake/Dataset004_Hippocampus/nnUNetModule__nnUNetPlans__3d_fullres/fold_0/validation'
    output_file = '/home/fabian/results/nnUNet_remake/Dataset004_Hippocampus/nnUNetModule__nnUNetPlans__3d_fullres/fold_0/validation/summary.json'
    image_reader_writer = SimpleITKIO()
    file_ending = '.nii.gz'
    regions = labels_to_list_of_regions([1, 2])
    ignore_label = None
    num_processes = 12
    compute_metrics_on_folder(folder_ref, folder_pred, output_file, image_reader_writer, file_ending, regions, ignore_label,
                              num_processes)
