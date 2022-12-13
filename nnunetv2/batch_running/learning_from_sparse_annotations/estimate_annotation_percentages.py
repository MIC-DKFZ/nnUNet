from multiprocessing import Pool

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.evaluation.evaluate_predictions import region_or_label_to_mask
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.label_handling.label_handling import get_labelmanager
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


def load_get_percent_annotated_per_class(npz_file, ref_file, labels_or_regions, ignore_label: int):
    seg = np.load(npz_file)['seg']
    seg_ref = np.load(ref_file)['seg']
    ret = {}
    for lr in labels_or_regions:
        mask = region_or_label_to_mask(seg, lr)
        mask_ref = region_or_label_to_mask(seg_ref, lr)
        n_ref = np.sum(mask_ref)
        if n_ref > 0:
            ret[lr] = np.sum(mask) / np.sum(mask_ref)
        else:
            ret[lr] = np.nan
    mask = region_or_label_to_mask(seg, ignore_label)
    ret['mean'] = np.sum(~mask) / np.prod(seg.shape, dtype=np.int64)
    return ret


def run_on_folder(folder, ref_folder, labels_or_regions, num_processes, ignore_label):
    p = Pool(num_processes)
    files = subfiles(folder, suffix='npz', join=False)
    ref_files = [join(ref_folder, f) for f in files]
    files = [join(folder, f) for f in files]
    res = p.starmap(load_get_percent_annotated_per_class, zip(files, ref_files, [labels_or_regions] * len(files), [ignore_label] * len(files)))
    aggr = {l: np.nanmean([i[l] for i in res]) for l in res[0].keys()}
    # average over foreground
    fg = np.nanmean([aggr[i] for i in labels_or_regions if i != 0])
    aggr['fg'] = fg
    return aggr


def run_on_all_subfolders(base, n_processes: int = 8, prefix='nnUNetPlans'):
    dataset_json = load_json(join(base, 'dataset.json'))
    plans_manager = PlansManager(join(base, 'nnUNetPlans.json'))
    lm = plans_manager.get_label_manager(dataset_json)
    labels_or_regions = lm.all_labels
    subfolders = subdirs(base, prefix=prefix)
    for s in subfolders:
        if os.path.basename(s).find('3d_lowres') != -1:
            ref_folder = join(base, 'nnUNetPlans_3d_lowres')
        elif os.path.basename(s).find('3d_fullres') != -1:
            ref_folder = join(base, 'nnUNetPlans_3d_fullres')
        else:
            continue
        print(s)
        r = run_on_folder(s, ref_folder, labels_or_regions, n_processes, ignore_label=lm.ignore_label)
        recursive_fix_for_json_export(r)
        save_json(r, join(base, os.path.basename(s) + '.json'), sort_keys=False)


if __name__ == '__main__':
    # this codes makes some hard assumptions.
    datasets = (994, 216)
    for d in datasets:
        print(d)
        base = join(nnUNet_preprocessed, maybe_convert_to_dataset_name(d))
        run_on_all_subfolders(base, 32)
    # folder = join(base, 'nnUNetPlans_3d_fullres_sparse_randblobs')
    # dataset_json = load_json(join(base, 'dataset.json'))
    # plans_json = load_json(join(base, 'nnUNetPlans.json'))
    # plans_manager = PlansManager(plans_json)
    # lm = plans_manager.get_label_manager(dataset_json)
    # labels_or_regions = lm.all_labels
    # print(run_on_folder(folder, labels_or_regions, 8))