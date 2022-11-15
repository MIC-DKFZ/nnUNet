#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import shutil
from collections import OrderedDict
from copy import deepcopy
from multiprocessing.pool import Pool
from typing import Tuple

import SimpleITK as sitk
import numpy as np
import scipy.stats as ss
from batchgenerators.utilities.file_and_folder_operations import *
from medpy.metric import dc, hd95
from nnunet.dataset_conversion.Task032_BraTS_2018 import convert_labels_back_to_BraTS_2018_2019_convention
from nnunet.dataset_conversion.Task043_BraTS_2019 import copy_BraTS_segmentation_and_convert_labels
from nnunet.evaluation.region_based_evaluation import get_brats_regions, evaluate_regions
from nnunet.paths import nnUNet_raw_data
from nnunet.postprocessing.consolidate_postprocessing import collect_cv_niftis


def apply_brats_threshold(fname, out_dir, threshold, replace_with):
    img_itk = sitk.ReadImage(fname)
    img_npy = sitk.GetArrayFromImage(img_itk)
    s = np.sum(img_npy == 3)
    if s < threshold:
        # print(s, fname)
        img_npy[img_npy == 3] = replace_with
    img_itk_postprocessed = sitk.GetImageFromArray(img_npy)
    img_itk_postprocessed.CopyInformation(img_itk)
    sitk.WriteImage(img_itk_postprocessed, join(out_dir, fname.split("/")[-1]))


def load_niftis_threshold_compute_dice(gt_file, pred_file, thresholds: Tuple[list, tuple]):
    gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_file))
    pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_file))
    mask_pred = pred == 3
    mask_gt = gt == 3
    num_pred = np.sum(mask_pred)

    num_gt = np.sum(mask_gt)
    dice = dc(mask_pred, mask_gt)

    res_dice = {}
    res_was_smaller = {}

    for t in thresholds:
        was_smaller = False

        if num_pred < t:
            was_smaller = True
            if num_gt == 0:
                dice_here = 1.
            else:
                dice_here = 0.
        else:
            dice_here = deepcopy(dice)

        res_dice[t] = dice_here
        res_was_smaller[t] = was_smaller

    return res_was_smaller, res_dice


def apply_threshold_to_folder(folder_in, folder_out, threshold, replace_with, processes=24):
    maybe_mkdir_p(folder_out)
    niftis = subfiles(folder_in, suffix='.nii.gz', join=True)

    p = Pool(processes)
    p.starmap(apply_brats_threshold, zip(niftis, [folder_out]*len(niftis), [threshold]*len(niftis), [replace_with] * len(niftis)))

    p.close()
    p.join()


def determine_brats_postprocessing(folder_with_preds, folder_with_gt, postprocessed_output_dir, processes=8,
        thresholds=(0, 10, 50, 100, 200, 500, 750, 1000, 1500, 2500, 10000), replace_with=2):
    # find pairs
    nifti_gt = subfiles(folder_with_gt, suffix=".nii.gz", sort=True)

    p = Pool(processes)

    nifti_pred = subfiles(folder_with_preds, suffix='.nii.gz', sort=True)

    results = p.starmap_async(load_niftis_threshold_compute_dice, zip(nifti_gt, nifti_pred, [thresholds] * len(nifti_pred)))
    results = results.get()

    all_dc_per_threshold = {}
    for t in thresholds:
        all_dc_per_threshold[t] = np.array([i[1][t] for i in results])
        print(t, np.mean(all_dc_per_threshold[t]))

    means = [np.mean(all_dc_per_threshold[t]) for t in thresholds]
    best_threshold = thresholds[np.argmax(means)]
    print('best', best_threshold, means[np.argmax(means)])

    maybe_mkdir_p(postprocessed_output_dir)

    p.starmap(apply_brats_threshold, zip(nifti_pred, [postprocessed_output_dir]*len(nifti_pred), [best_threshold]*len(nifti_pred), [replace_with] * len(nifti_pred)))

    p.close()
    p.join()

    save_pickle((thresholds, means, best_threshold, all_dc_per_threshold), join(postprocessed_output_dir, "threshold.pkl"))


def collect_and_prepare(base_dir, num_processes = 12, clean=False):
    """
    collect all cv_niftis, compute brats metrics, compute enh tumor thresholds and summarize in csv
    :param base_dir:
    :return:
    """
    out = join(base_dir, 'cv_results')
    out_pp = join(base_dir, 'cv_results_pp')
    experiments = subfolders(base_dir, join=False, prefix='nnUNetTrainer')
    regions = get_brats_regions()
    gt_dir = join(base_dir, 'gt_niftis')
    replace_with = 2

    failed = []
    successful = []
    for e in experiments:
        print(e)
        try:
            o = join(out, e)
            o_p = join(out_pp, e)
            maybe_mkdir_p(o)
            maybe_mkdir_p(o_p)
            collect_cv_niftis(join(base_dir, e), o)
            if clean or not isfile(join(o, 'summary.csv')):
                evaluate_regions(o, gt_dir, regions, num_processes)
            if clean or not isfile(join(o_p, 'threshold.pkl')):
                determine_brats_postprocessing(o, gt_dir, o_p, num_processes, thresholds=list(np.arange(0, 760, 10)), replace_with=replace_with)
            if clean or not isfile(join(o_p, 'summary.csv')):
                evaluate_regions(o_p, gt_dir, regions, num_processes)
            successful.append(e)
        except Exception as ex:
            print("\nERROR\n", e, ex, "\n")
            failed.append(e)

    # we are interested in the mean (nan is 1) column
    with open(join(base_dir, 'cv_summary.csv'), 'w') as f:
        f.write('name,whole,core,enh,mean\n')
        for e in successful:
            expected_nopp = join(out, e, 'summary.csv')
            expected_pp = join(out, out_pp, e, 'summary.csv')
            if isfile(expected_nopp):
                res = np.loadtxt(expected_nopp, dtype=str, skiprows=0, delimiter=',')[-2]
                as_numeric = [float(i) for i in res[1:]]
                f.write(e + '_noPP,')
                f.write("%0.4f," % as_numeric[0])
                f.write("%0.4f," % as_numeric[1])
                f.write("%0.4f," % as_numeric[2])
                f.write("%0.4f\n" % np.mean(as_numeric))
            if isfile(expected_pp):
                res = np.loadtxt(expected_pp, dtype=str, skiprows=0, delimiter=',')[-2]
                as_numeric = [float(i) for i in res[1:]]
                f.write(e + '_PP,')
                f.write("%0.4f," % as_numeric[0])
                f.write("%0.4f," % as_numeric[1])
                f.write("%0.4f," % as_numeric[2])
                f.write("%0.4f\n" % np.mean(as_numeric))

    # this just crawls the folders and evaluates what it finds
    with open(join(base_dir, 'cv_summary2.csv'), 'w') as f:
        for folder in ['cv_results', 'cv_results_pp']:
            for ex in subdirs(join(base_dir, folder), join=False):
                print(folder, ex)
                expected = join(base_dir, folder, ex, 'summary.csv')
                if clean or not isfile(expected):
                    evaluate_regions(join(base_dir, folder, ex), gt_dir, regions, num_processes)
                if isfile(expected):
                    res = np.loadtxt(expected, dtype=str, skiprows=0, delimiter=',')[-2]
                    as_numeric = [float(i) for i in res[1:]]
                    f.write('%s__%s,' % (folder, ex))
                    f.write("%0.4f," % as_numeric[0])
                    f.write("%0.4f," % as_numeric[1])
                    f.write("%0.4f," % as_numeric[2])
                    f.write("%0.4f\n" % np.mean(as_numeric))

        f.write('name,whole,core,enh,mean\n')
        for e in successful:
            expected_nopp = join(out, e, 'summary.csv')
            expected_pp = join(out, out_pp, e, 'summary.csv')
            if isfile(expected_nopp):
                res = np.loadtxt(expected_nopp, dtype=str, skiprows=0, delimiter=',')[-2]
                as_numeric = [float(i) for i in res[1:]]
                f.write(e + '_noPP,')
                f.write("%0.4f," % as_numeric[0])
                f.write("%0.4f," % as_numeric[1])
                f.write("%0.4f," % as_numeric[2])
                f.write("%0.4f\n" % np.mean(as_numeric))
            if isfile(expected_pp):
                res = np.loadtxt(expected_pp, dtype=str, skiprows=0, delimiter=',')[-2]
                as_numeric = [float(i) for i in res[1:]]
                f.write(e + '_PP,')
                f.write("%0.4f," % as_numeric[0])
                f.write("%0.4f," % as_numeric[1])
                f.write("%0.4f," % as_numeric[2])
                f.write("%0.4f\n" % np.mean(as_numeric))

    # apply threshold to val set
    expected_num_cases = 125
    missing_valset = []
    has_val_pred = []
    for e in successful:
        if isdir(join(base_dir, 'predVal', e)):
            currdir = join(base_dir, 'predVal', e)
            files = subfiles(currdir, suffix='.nii.gz', join=False)
            if len(files) != expected_num_cases:
                print(e, 'prediction not done, found %d files, expected %s' % (len(files), expected_num_cases))
                continue
            output_folder = join(base_dir, 'predVal_PP', e)
            maybe_mkdir_p(output_folder)
            threshold = load_pickle(join(out_pp, e, 'threshold.pkl'))[2]
            if threshold > 1000: threshold = 750  # don't make it too big!
            apply_threshold_to_folder(currdir, output_folder, threshold, replace_with, num_processes)
            has_val_pred.append(e)
        else:
            print(e, 'has no valset predictions')
            missing_valset.append(e)

    # 'nnUNetTrainerV2BraTSRegions_DA3_BN__nnUNetPlansv2.1_bs5_15fold' needs special treatment
    e = 'nnUNetTrainerV2BraTSRegions_DA3_BN__nnUNetPlansv2.1_bs5'
    currdir = join(base_dir, 'predVal', 'nnUNetTrainerV2BraTSRegions_DA3_BN__nnUNetPlansv2.1_bs5_15fold')
    output_folder = join(base_dir, 'predVal_PP', 'nnUNetTrainerV2BraTSRegions_DA3_BN__nnUNetPlansv2.1_bs5_15fold')
    maybe_mkdir_p(output_folder)
    threshold = load_pickle(join(out_pp, e, 'threshold.pkl'))[2]
    if threshold > 1000: threshold = 750  # don't make it too big!
    apply_threshold_to_folder(currdir, output_folder, threshold, replace_with, num_processes)

    # 'nnUNetTrainerV2BraTSRegions_DA3_BN__nnUNetPlansv2.1_bs5_15fold' needs special treatment
    e = 'nnUNetTrainerV2BraTSRegions_DA4_BN__nnUNetPlansv2.1_bs5'
    currdir = join(base_dir, 'predVal', 'nnUNetTrainerV2BraTSRegions_DA4_BN__nnUNetPlansv2.1_bs5_15fold')
    output_folder = join(base_dir, 'predVal_PP', 'nnUNetTrainerV2BraTSRegions_DA4_BN__nnUNetPlansv2.1_bs5_15fold')
    maybe_mkdir_p(output_folder)
    threshold = load_pickle(join(out_pp, e, 'threshold.pkl'))[2]
    if threshold > 1000: threshold = 750  # don't make it too big!
    apply_threshold_to_folder(currdir, output_folder, threshold, replace_with, num_processes)

    # convert val set to brats labels for submission
    output_converted = join(base_dir, 'converted_valSet')

    for source in ['predVal', 'predVal_PP']:
        for e in has_val_pred + ['nnUNetTrainerV2BraTSRegions_DA3_BN__nnUNetPlansv2.1_bs5_15fold', 'nnUNetTrainerV2BraTSRegions_DA4_BN__nnUNetPlansv2.1_bs5_15fold']:
            expected_source_folder = join(base_dir, source, e)
            if not isdir(expected_source_folder):
                print(e, 'has no', source)
                raise RuntimeError()
            files = subfiles(expected_source_folder, suffix='.nii.gz', join=False)
            if len(files) != expected_num_cases:
                print(e, 'prediction not done, found %d files, expected %s' % (len(files), expected_num_cases))
                continue
            target_folder = join(output_converted, source, e)
            maybe_mkdir_p(target_folder)
            convert_labels_back_to_BraTS_2018_2019_convention(expected_source_folder, target_folder)

    summarize_validation_set_predictions(output_converted)


def summarize_validation_set_predictions(base):
    with open(join(base, 'summary.csv'), 'w') as f:
        f.write('name,whole,core,enh,mean,whole,core,enh,mean\n')
        for subf in subfolders(base, join=False):
            for e in subfolders(join(base, subf), join=False):
                expected = join(base, subf, e, 'Stats_Validation_final.csv')
                if not isfile(expected):
                    print(subf, e, 'has missing csv')
                    continue
                a = np.loadtxt(expected, delimiter=',', dtype=str)
                assert a.shape[0] == 131, 'did not evaluate all 125 cases!'
                selected_row = a[-5]
                values = [float(i) for i in selected_row[1:4]]
                f.write(e + "_" + subf + ',')
                f.write("%0.4f," % values[1])
                f.write("%0.4f," % values[2])
                f.write("%0.4f," % values[0])
                f.write("%0.4f," % np.mean(values))
                values = [float(i) for i in selected_row[-3:]]
                f.write("%0.4f," % values[1])
                f.write("%0.4f," % values[2])
                f.write("%0.4f," % values[0])
                f.write("%0.4f\n" % np.mean(values))


def compute_BraTS_dice(ref, pred):
    """
    ref and gt are binary integer numpy.ndarray s
    :param ref:
    :param gt:
    :return:
    """
    num_ref = np.sum(ref)
    num_pred = np.sum(pred)

    if num_ref == 0:
        if num_pred == 0:
            return 1
        else:
            return 0
    else:
        return dc(pred, ref)


def convert_all_to_BraTS(input_folder, output_folder, expected_num_cases=125):
    for s in subdirs(input_folder, join=False):
        nii = subfiles(join(input_folder, s), suffix='.nii.gz', join=False)
        if len(nii) != expected_num_cases:
            print(s)
        else:
            target_dir = join(output_folder, s)
            convert_labels_back_to_BraTS_2018_2019_convention(join(input_folder, s), target_dir, num_processes=6)


def compute_BraTS_HD95(ref, pred):
    """
    ref and gt are binary integer numpy.ndarray s
    spacing is assumed to be (1, 1, 1)
    :param ref:
    :param pred:
    :return:
    """
    num_ref = np.sum(ref)
    num_pred = np.sum(pred)

    if num_ref == 0:
        if num_pred == 0:
            return 0
        else:
            return 373.12866
    elif num_pred == 0 and num_ref != 0:
        return 373.12866
    else:
        return hd95(pred, ref, (1, 1, 1))


def evaluate_BraTS_case(arr: np.ndarray, arr_gt: np.ndarray):
    """
    attempting to reimplement the brats evaluation scheme
    assumes edema=1, non_enh=2, enh=3
    :param arr:
    :param arr_gt:
    :return:
    """
    # whole tumor
    mask_gt = (arr_gt != 0).astype(int)
    mask_pred = (arr != 0).astype(int)
    dc_whole = compute_BraTS_dice(mask_gt, mask_pred)
    hd95_whole = compute_BraTS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    # tumor core
    mask_gt = (arr_gt > 1).astype(int)
    mask_pred = (arr > 1).astype(int)
    dc_core = compute_BraTS_dice(mask_gt, mask_pred)
    hd95_core = compute_BraTS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    # enhancing
    mask_gt = (arr_gt == 3).astype(int)
    mask_pred = (arr == 3).astype(int)
    dc_enh = compute_BraTS_dice(mask_gt, mask_pred)
    hd95_enh = compute_BraTS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    return dc_whole, dc_core, dc_enh, hd95_whole, hd95_core, hd95_enh


def load_evaluate(filename_gt: str, filename_pred: str):
    arr_pred = sitk.GetArrayFromImage(sitk.ReadImage(filename_pred))
    arr_gt = sitk.GetArrayFromImage(sitk.ReadImage(filename_gt))
    return evaluate_BraTS_case(arr_pred, arr_gt)


def evaluate_BraTS_folder(folder_pred, folder_gt, num_processes: int = 24, strict=False):
    nii_pred = subfiles(folder_pred, suffix='.nii.gz', join=False)
    if len(nii_pred) == 0:
        return
    nii_gt = subfiles(folder_gt, suffix='.nii.gz', join=False)
    assert all([i in nii_gt for i in nii_pred]), 'not all predicted niftis have a reference file!'
    if strict:
        assert all([i in nii_pred for i in nii_gt]), 'not all gt niftis have a predicted file!'
    p = Pool(num_processes)
    nii_pred_fullpath = [join(folder_pred, i) for i in nii_pred]
    nii_gt_fullpath = [join(folder_gt, i) for i in nii_pred]
    results = p.starmap(load_evaluate, zip(nii_gt_fullpath, nii_pred_fullpath))
    # now write to output file
    with open(join(folder_pred, 'results.csv'), 'w') as f:
        f.write("name,dc_whole,dc_core,dc_enh,hd95_whole,hd95_core,hd95_enh\n")
        for fname, r in zip(nii_pred, results):
            f.write(fname)
            f.write(",%0.4f,%0.4f,%0.4f,%3.3f,%3.3f,%3.3f\n" % r)


def load_csv_for_ranking(csv_file: str):
    res = np.loadtxt(csv_file, dtype='str', delimiter=',')
    scores = res[1:, [1, 2, 3, -3, -2, -1]].astype(float)
    scores[:, -3:] *= -1
    scores[:, -3:] += 373.129
    assert np.all(scores <= 373.129)
    assert np.all(scores >= 0)
    return scores


def rank_algorithms(data:np.ndarray):
    """
    data is (metrics x experiments x cases)
    :param data:
    :return:
    """
    num_metrics, num_experiments, num_cases = data.shape
    ranks = np.zeros((num_metrics, num_experiments))
    for m in range(6):
        r = np.apply_along_axis(ss.rankdata, 0, -data[m], 'min')
        ranks[m] = r.mean(1)
    average_rank = np.mean(ranks, 0)
    final_ranks = ss.rankdata(average_rank, 'min')
    return final_ranks, average_rank, ranks


def score_and_postprocess_model_based_on_rank_then_aggregate():
    """
    Similarly to BraTS 2017 - BraTS 2019, each participant will be ranked for each of the X test cases. Each case
    includes 3 regions of evaluation, and the metrics used to produce the rankings will be the Dice Similarity
    Coefficient and the 95% Hausdorff distance. Thus, for X number of cases included in the BraTS 2020, each
    participant ends up having X*3*2 rankings. The final ranking score is the average of all these rankings normalized
    by the number of teams.
    https://zenodo.org/record/3718904

    -> let's optimize for this.

    Important: the outcome very much depends on the competing models. We need some references. We only got our own,
    so let's hope this still works
    :return:
    """
    base = "/media/fabian/Results/nnUNet/3d_fullres/Task082_BraTS2020"
    replace_with = 2
    num_processes = 24
    expected_num_cases_val = 125

    # use a separate output folder from the previous experiments to ensure we are not messing things up
    output_base_here = join(base, 'use_brats_ranking')
    maybe_mkdir_p(output_base_here)

    # collect cv niftis and compute metrics with evaluate_BraTS_folder to ensure we work with the same metrics as brats
    out = join(output_base_here, 'cv_results')
    experiments = subfolders(base, join=False, prefix='nnUNetTrainer')
    gt_dir = join(base, 'gt_niftis')

    experiments_with_full_cv = []
    for e in experiments:
        print(e)
        o = join(out, e)
        maybe_mkdir_p(o)
        try:
            collect_cv_niftis(join(base, e), o)
            if not isfile(join(o, 'results.csv')):
                evaluate_BraTS_folder(o, gt_dir, num_processes, strict=True)
            experiments_with_full_cv.append(e)
        except Exception as ex:
            print("\nERROR\n", e, ex, "\n")
            if isfile(join(o, 'results.csv')):
                os.remove(join(o, 'results.csv'))

    # rank the non-postprocessed models
    tmp = np.loadtxt(join(out, experiments_with_full_cv[0], 'results.csv'), dtype='str', delimiter=',')
    num_cases = len(tmp) - 1
    data_for_ranking = np.zeros((6, len(experiments_with_full_cv), num_cases))
    for i, e in enumerate(experiments_with_full_cv):
        scores = load_csv_for_ranking(join(out, e, 'results.csv'))
        for metric in range(6):
            data_for_ranking[metric, i] = scores[:, metric]

    final_ranks, average_rank, ranks = rank_algorithms(data_for_ranking)

    for t in np.argsort(final_ranks):
        print(final_ranks[t], average_rank[t], experiments_with_full_cv[t])

    # for each model, create output directories with different thresholds. evaluate ALL OF THEM (might take a while lol)
    thresholds = np.arange(25, 751, 25)
    output_pp_tmp = join(output_base_here, 'cv_determine_pp_thresholds')
    for e in experiments_with_full_cv:
        input_folder = join(out, e)
        for t in thresholds:
            output_directory = join(output_pp_tmp, e, str(t))
            maybe_mkdir_p(output_directory)
            if not isfile(join(output_directory, 'results.csv')):
                apply_threshold_to_folder(input_folder, output_directory, t, replace_with, processes=16)
                evaluate_BraTS_folder(output_directory, gt_dir, num_processes)

    # load ALL the results!
    results = []
    experiment_names = []
    for e in experiments_with_full_cv:
        for t in thresholds:
            output_directory = join(output_pp_tmp, e, str(t))
            expected_file = join(output_directory, 'results.csv')
            if not isfile(expected_file):
                print(e, 'does not have a results file for threshold', t)
                continue
            results.append(load_csv_for_ranking(expected_file))
            experiment_names.append("%s___%d" % (e, t))
    all_results = np.concatenate([i[None] for i in results], 0).transpose((2, 0, 1))

    # concatenate with non postprocessed models
    all_results = np.concatenate((data_for_ranking, all_results), 1)
    experiment_names += experiments_with_full_cv

    final_ranks, average_rank, ranks = rank_algorithms(all_results)

    for t in np.argsort(final_ranks):
        print(final_ranks[t], average_rank[t], experiment_names[t])

    # for each model, print the non postprocessed model as well as the best postprocessed model. If there are
    # validation set predictions, apply the best threshold to the validation set
    pred_val_base = join(base, 'predVal_PP_rank')
    has_val_pred = []
    for e in experiments_with_full_cv:
        rank_nonpp = final_ranks[experiment_names.index(e)]
        avg_rank_nonpp = average_rank[experiment_names.index(e)]
        print(e, avg_rank_nonpp, rank_nonpp)
        predicted_val = join(base, 'predVal', e)

        pp_models = [j for j, i in enumerate(experiment_names) if i.split("___")[0] == e and i != e]
        if len(pp_models) > 0:
            ranks = [final_ranks[i] for i in pp_models]
            best_idx = np.argmin(ranks)
            best = experiment_names[pp_models[best_idx]]
            best_avg_rank = average_rank[pp_models[best_idx]]
            print(best, best_avg_rank, min(ranks))
            print('')
            # apply threshold to validation set
            best_threshold = int(best.split('___')[-1])
            if not isdir(predicted_val):
                print(e, 'has not valset predictions')
            else:
                files = subfiles(predicted_val, suffix='.nii.gz')
                if len(files) != expected_num_cases_val:
                    print(e, 'has missing val cases. found: %d expected: %d' % (len(files), expected_num_cases_val))
                else:
                    apply_threshold_to_folder(predicted_val, join(pred_val_base, e), best_threshold, replace_with, num_processes)
                    has_val_pred.append(e)
        else:
            print(e, 'not found in ranking')

    # apply nnUNetTrainerV2BraTSRegions_DA3_BN__nnUNetPlansv2.1_bs5 to nnUNetTrainerV2BraTSRegions_DA3_BN__nnUNetPlansv2.1_bs5_15fold
    e = 'nnUNetTrainerV2BraTSRegions_DA3_BN__nnUNetPlansv2.1_bs5'
    pp_models = [j for j, i in enumerate(experiment_names) if i.split("___")[0] == e and i != e]
    ranks = [final_ranks[i] for i in pp_models]
    best_idx = np.argmin(ranks)
    best = experiment_names[pp_models[best_idx]]
    best_avg_rank = average_rank[pp_models[best_idx]]
    best_threshold = int(best.split('___')[-1])
    predicted_val = join(base, 'predVal', 'nnUNetTrainerV2BraTSRegions_DA3_BN__nnUNetPlansv2.1_bs5_15fold')
    apply_threshold_to_folder(predicted_val, join(pred_val_base, 'nnUNetTrainerV2BraTSRegions_DA3_BN__nnUNetPlansv2.1_bs5_15fold'), best_threshold, replace_with, num_processes)
    has_val_pred.append('nnUNetTrainerV2BraTSRegions_DA3_BN__nnUNetPlansv2.1_bs5_15fold')

    # apply nnUNetTrainerV2BraTSRegions_DA4_BN__nnUNetPlansv2.1_bs5 to nnUNetTrainerV2BraTSRegions_DA4_BN__nnUNetPlansv2.1_bs5_15fold
    e = 'nnUNetTrainerV2BraTSRegions_DA4_BN__nnUNetPlansv2.1_bs5'
    pp_models = [j for j, i in enumerate(experiment_names) if i.split("___")[0] == e and i != e]
    ranks = [final_ranks[i] for i in pp_models]
    best_idx = np.argmin(ranks)
    best = experiment_names[pp_models[best_idx]]
    best_avg_rank = average_rank[pp_models[best_idx]]
    best_threshold = int(best.split('___')[-1])
    predicted_val = join(base, 'predVal', 'nnUNetTrainerV2BraTSRegions_DA4_BN__nnUNetPlansv2.1_bs5_15fold')
    apply_threshold_to_folder(predicted_val, join(pred_val_base, 'nnUNetTrainerV2BraTSRegions_DA4_BN__nnUNetPlansv2.1_bs5_15fold'), best_threshold, replace_with, num_processes)
    has_val_pred.append('nnUNetTrainerV2BraTSRegions_DA4_BN__nnUNetPlansv2.1_bs5_15fold')

    # convert valsets
    output_converted = join(base, 'converted_valSet')
    for e in has_val_pred:
        expected_source_folder = join(base, 'predVal_PP_rank', e)
        if not isdir(expected_source_folder):
            print(e, 'has no predVal_PP_rank')
            raise RuntimeError()
        files = subfiles(expected_source_folder, suffix='.nii.gz', join=False)
        if len(files) != expected_num_cases_val:
            print(e, 'prediction not done, found %d files, expected %s' % (len(files), expected_num_cases_val))
            continue
        target_folder = join(output_converted, 'predVal_PP_rank', e)
        maybe_mkdir_p(target_folder)
        convert_labels_back_to_BraTS_2018_2019_convention(expected_source_folder, target_folder)

    # now load all the csvs for the validation set (obtained from evaluation platform) and rank our models on the
    # validation set
    flds = subdirs(output_converted, join=False)
    results_valset = []
    names_valset = []
    for f in flds:
        curr = join(output_converted, f)
        experiments = subdirs(curr, join=False)
        for e in experiments:
            currr = join(curr, e)
            expected_file = join(currr, 'Stats_Validation_final.csv')
            if not isfile(expected_file):
                print(f, e, "has not been evaluated yet!")
            else:
                res = load_csv_for_ranking(expected_file)[:-5]
                assert res.shape[0] == expected_num_cases_val
                results_valset.append(res[None])
                names_valset.append("%s___%s" % (f, e))
    results_valset = np.concatenate(results_valset, 0)  # experiments x cases x metrics
    # convert to metrics x experiments x cases
    results_valset = results_valset.transpose((2, 0, 1))
    final_ranks, average_rank, ranks = rank_algorithms(results_valset)
    for t in np.argsort(final_ranks):
        print(final_ranks[t], average_rank[t], names_valset[t])


if __name__ == "__main__":
    """
    THIS CODE IS A MESS. IT IS PROVIDED AS IS WITH NO GUARANTEES. YOU HAVE TO DIG THROUGH IT YOURSELF. GOOD LUCK ;-)

    REMEMBER TO CONVERT LABELS BACK TO BRATS CONVENTION AFTER PREDICTION!
    """

    task_name = "Task082_BraTS2020"
    downloaded_data_dir = "/kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    downloaded_data_dir_val = "/kaggle/input/brats20-dataset-training-validation/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData"

    target_base = join(nnUNet_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_imagesVal = join(target_base, "imagesVal")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTr = join(target_base, "labelsTr")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_imagesVal)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)

    patient_names = []
    cur = join(downloaded_data_dir)
    for p in subdirs(cur, join=False):
        patdir = join(cur, p)
        patient_name = p
        patient_names.append(patient_name)

        t1_tmp = join(patdir, p + "_t1.nii")
        t1_cmd = f"gzip {t1_tmp}"
        os.system(t1_cmd)

        t1ce_tmp = join(patdir, p + "_t1ce.nii")
        t1ce_cmd = f"gzip {t1ce_tmp}"
        os.system(t1ce_cmd)

        t2_tmp = join(patdir, p + "_t2.nii")
        t2_cmd = f"gzip {t2_tmp}"
        os.system(t2_cmd)

        flair_tmp = join(patdir, p + "_flair.nii")
        flair_cmd = f"gzip {flair_tmp}"
        os.system(flair_cmd)

        seg_tmp = join(patdir, p + "_seg.nii")
        seg_cmd = f"gzip {seg_tmp}"
        os.system(seg_cmd)

        t1_tmp = join(patdir, p + "_t1.nii")
        t1_cmd = f"gzip {t1_tmp}"
        os.system(t1_cmd)

        t1ce_tmp = join(patdir, p + "_t1ce.nii")
        t1ce_cmd = f"gzip {t1ce_tmp}"
        os.system(t1ce_cmd)

        t2_tmp = join(patdir, p + "_t2.nii")
        t2_cmd = f"gzip {t2_tmp}"
        os.system(t2_cmd)

        flair_tmp = join(patdir, p + "_flair.nii")
        flair_cmd = f"gzip {flair_tmp}"
        os.system(flair_cmd)

        seg_tmp = join(patdir, p + "_seg.nii")
        seg_cmd = f"gzip {seg_tmp}"
        os.system(seg_cmd)

        t1 = join(patdir, p + "_t1.nii.gz")
        t1c = join(patdir, p + "_t1ce.nii.gz")
        t2 = join(patdir, p + "_t2.nii.gz")
        flair = join(patdir, p + "_flair.nii.gz")
        seg = join(patdir, p + "_seg.nii.gz")

        assert all([
            isfile(t1),
            isfile(t1c),
            isfile(t2),
            isfile(flair),
            isfile(seg)
        ]), "%s" % patient_name

        shutil.copy(t1, join(target_imagesTr, patient_name + "_0000.nii.gz"))
        shutil.copy(t1c, join(target_imagesTr, patient_name + "_0001.nii.gz"))
        shutil.copy(t2, join(target_imagesTr, patient_name + "_0002.nii.gz"))
        shutil.copy(flair, join(target_imagesTr, patient_name + "_0003.nii.gz"))

        copy_BraTS_segmentation_and_convert_labels(seg, join(target_labelsTr, patient_name + ".nii.gz"))


    json_dict = OrderedDict()
    json_dict['name'] = "BraTS2020"
    json_dict['description'] = "nothing"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see BraTS2020"
    json_dict['licence'] = "see BraTS2020 license"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "T1",
        "1": "T1ce",
        "2": "T2",
        "3": "FLAIR"
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "edema",
        "2": "non-enhancing",
        "3": "enhancing",
    }
    json_dict['numTraining'] = len(patient_names)
    json_dict['numTest'] = 0
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                             patient_names]
    json_dict['test'] = []

    save_json(json_dict, join(target_base, "dataset.json"))

    if downloaded_data_dir_val is not None:
        for p in subdirs(downloaded_data_dir_val, join=False):
            patdir = join(downloaded_data_dir_val, p)
            patient_name = p

            t1_tmp = join(patdir, p + "_t1.nii")
            t1_cmd = f"gzip {t1_tmp}"
            os.system(t1_cmd)

            t1ce_tmp = join(patdir, p + "_t1ce.nii")
            t1ce_cmd = f"gzip {t1ce_tmp}"
            os.system(t1ce_cmd)

            t2_tmp = join(patdir, p + "_t2.nii")
            t2_cmd = f"gzip {t2_tmp}"
            os.system(t2_cmd)

            flair_tmp = join(patdir, p + "_flair.nii")
            flair_cmd = f"gzip {flair_tmp}"
            os.system(flair_cmd)

            t1 = join(patdir, p + "_t1.nii.gz")
            t1c = join(patdir, p + "_t1ce.nii.gz")
            t2 = join(patdir, p + "_t2.nii.gz")
            flair = join(patdir, p + "_flair.nii.gz")

            assert all([
                isfile(t1),
                isfile(t1c),
                isfile(t2),
                isfile(flair),
            ]), "%s" % patient_name

            shutil.copy(t1, join(target_imagesVal, patient_name + "_0000.nii.gz"))
            shutil.copy(t1c, join(target_imagesVal, patient_name + "_0001.nii.gz"))
            shutil.copy(t2, join(target_imagesVal, patient_name + "_0002.nii.gz"))
            shutil.copy(flair, join(target_imagesVal, patient_name + "_0003.nii.gz"))


    downloaded_data_dir_test = "/kaggle/input/brats20-dataset-training-validation/BraTS2020_testData/MICCAI_BraTS2020_TestingData"

    if isdir(downloaded_data_dir_test):
        for p in subdirs(downloaded_data_dir_test, join=False):
            patdir = join(downloaded_data_dir_test, p)
            patient_name = p
            t1 = join(patdir, p + "_t1.nii.gz")
            t1c = join(patdir, p + "_t1ce.nii.gz")
            t2 = join(patdir, p + "_t2.nii.gz")
            flair = join(patdir, p + "_flair.nii.gz")

            assert all([
                isfile(t1),
                isfile(t1c),
                isfile(t2),
                isfile(flair),
            ]), "%s" % patient_name

            shutil.copy(t1, join(target_imagesTs, patient_name + "_0000.nii.gz"))
            shutil.copy(t1c, join(target_imagesTs, patient_name + "_0001.nii.gz"))
            shutil.copy(t2, join(target_imagesTs, patient_name + "_0002.nii.gz"))
            shutil.copy(flair, join(target_imagesTs, patient_name + "_0003.nii.gz"))

    # test set
    #  nnUNet_ensemble -f nnUNetTrainerV2BraTSRegions_DA3_BN_BD__nnUNetPlansv2.1_bs5_5fold nnUNetTrainerV2BraTSRegions_DA4_BN_BD__nnUNetPlansv2.1_bs5_5fold nnUNetTrainerV2BraTSRegions_DA4_BN__nnUNetPlansv2.1_bs5_15fold -o ensembled_nnUNetTrainerV2BraTSRegions_DA3_BN_BD__nnUNetPlansv2.1_bs5_5fold__nnUNetTrainerV2BraTSRegions_DA4_BN_BD__nnUNetPlansv2.1_bs5_5fold__nnUNetTrainerV2BraTSRegions_DA4_BN__nnUNetPlansv2.1_bs5_15fold
    # apply_threshold_to_folder('ensembled_nnUNetTrainerV2BraTSRegions_DA3_BN_BD__nnUNetPlansv2.1_bs5_5fold__nnUNetTrainerV2BraTSRegions_DA4_BN_BD__nnUNetPlansv2.1_bs5_5fold__nnUNetTrainerV2BraTSRegions_DA4_BN__nnUNetPlansv2.1_bs5_15fold/', 'ensemble_PP200/', 200, 2)
    # convert_labels_back_to_BraTS_2018_2019_convention('ensemble_PP200/', 'ensemble_PP200_converted')

    # export for publication of weights
    # nnUNet_export_model_to_zip -tr nnUNetTrainerV2BraTSRegions_DA4_BN -pl nnUNetPlansv2.1_bs5 -f 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 -t 82 -o nnUNetTrainerV2BraTSRegions_DA4_BN__nnUNetPlansv2.1_bs5_15fold.zip --disable_strict
    # nnUNet_export_model_to_zip -tr nnUNetTrainerV2BraTSRegions_DA3_BN_BD -pl nnUNetPlansv2.1_bs5 -f 0 1 2 3 4 -t 82 -o nnUNetTrainerV2BraTSRegions_DA3_BN_BD__nnUNetPlansv2.1_bs5_5fold.zip --disable_strict
    # nnUNet_export_model_to_zip -tr nnUNetTrainerV2BraTSRegions_DA4_BN_BD -pl nnUNetPlansv2.1_bs5 -f 0 1 2 3 4 -t 82 -o nnUNetTrainerV2BraTSRegions_DA4_BN_BD__nnUNetPlansv2.1_bs5_5fold.zip --disable_strict
