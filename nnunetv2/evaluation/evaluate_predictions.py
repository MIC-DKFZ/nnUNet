from multiprocessing import Pool
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, write_json
from nnunetv2.configuration import default_num_processes
from typing import Tuple, List

import numpy as np
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
# the Evaluator class of the previous nnU-Net was great and all but man was it overengineered. Keep it simple
from nnunetv2.utilities.json_export import recursive_fix_for_json_export


def labels_to_list_of_regions(labels: List[int]):
    return [(i, ) for i in labels]


def region_to_mask(segmentation: np.ndarray, region: Tuple[int, ...]) -> np.ndarray:
    mask = np.zeros_like(segmentation, dtype=bool)
    for r in region:
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


def compute_metrics(reference_file: str, prediction_file: str, image_reader_writer: BaseReaderWriter,
                    regions: List[Tuple[int, ...]], ignore_label: int = None) -> dict:
    # load images
    seg_ref, seg_ref_dict = image_reader_writer.read_seg(reference_file)
    seg_pred, seg_pred_dict = image_reader_writer.read_seg(prediction_file)
    # spacing = seg_ref_dict['spacing']

    ignore_mask = reference_file == ignore_label if ignore_label is not None else None

    results = {}
    results['reference_file'] = reference_file
    results['prediction_file'] = prediction_file
    results['metrics'] = {}
    for r in regions:
        k = r[0] if len(r) == 1 else r
        results['metrics'][k] = {}
        mask_ref = region_to_mask(seg_ref, r)
        mask_pred = region_to_mask(seg_pred, r)
        tp, fp, fn, tn = compute_tp_fp_fn_tn(mask_ref, mask_pred, ignore_mask)
        results['metrics'][k]['Dice'] = 2 * tp / (2 * tp + fp + fn)
        results['metrics'][k]['IoU'] = tp / (tp + fp + fn)
        results['metrics'][k]['FP'] = fp
        results['metrics'][k]['TP'] = tp
        results['metrics'][k]['FN'] = fn
        results['metrics'][k]['TN'] = tn
        results['metrics'][k]['n_pred'] = fp + tp
        results['metrics'][k]['n_ref'] = fn + tp
    return results


def compute_metrics_on_folder(folder_ref: str, folder_pred: str, output_file: str, image_reader_writer: BaseReaderWriter,
                              suffix: str,
                              regions: List[Tuple[int, ...]], ignore_label: int = None,
                              num_processes: int = default_num_processes) -> None:
    """
    output_file must end with .json
    """
    assert output_file.endswith('.json'), 'output_file should end with .json'
    files_pred = subfiles(folder_pred, suffix=suffix, join=False)
    files_ref = subfiles(folder_ref, suffix=suffix, join=False)
    assert all([i in files_ref for i in files_pred]), "Not all files in folder_pred exist in folder_ref"
    files_ref = [join(folder_ref, i) for i in files_pred]
    files_pred = [join(folder_pred, i) for i in files_pred]

    pool = Pool(num_processes)
    results = pool.starmap(
        compute_metrics,
        list(zip(files_ref, files_pred, [image_reader_writer] * len(files_pred), [regions] * len(files_pred), [ignore_label] * len(files_pred)))
    )
    k = regions[0][0] if len(regions[0]) == 1 else regions[0]
    metric_list = list(results[0]['metrics'][k].keys())
    means = {}
    for r in regions:
        k = r[0] if len(r) == 1 else r
        means[k] = {}
        for m in metric_list:
            means[k][m] = np.nanmean([i['metrics'][k][m] for i in results])
    pool.close()
    pool.join()
    [recursive_fix_for_json_export(i) for i in results]
    recursive_fix_for_json_export(means)
    write_json({'metric_per_case': results, 'mean': means}, output_file)


if __name__ == '__main__':
    folder_ref = '/media/fabian/data/nnUNet_raw/Dataset004_Hippocampus/labelsTr'
    folder_pred = '/home/fabian/results/nnUNet_remake/Dataset004_Hippocampus/nnUNetModule__nnUNetPlans__3d_fullres/fold_0/validation'
    output_file = '/home/fabian/results/nnUNet_remake/Dataset004_Hippocampus/nnUNetModule__nnUNetPlans__3d_fullres/fold_0/validation/summary.json'
    image_reader_writer = SimpleITKIO()
    suffix = '.nii.gz'
    regions = labels_to_list_of_regions([1, 2])
    ignore_label = None
    num_processes = 12
    compute_metrics_on_folder(folder_ref, folder_pred, output_file, image_reader_writer, suffix, regions, ignore_label, num_processes)