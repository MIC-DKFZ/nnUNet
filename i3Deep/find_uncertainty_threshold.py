from i3Deep import utils
import numpy as np
import os
import sys
from tqdm import tqdm
from scipy import optimize
import time
import matplotlib.pyplot as plt
from numbers import Number
import copy
import pickle
import multiprocessing
from multiprocessing import Pool
from functools import partial
from i3Deep import uncertainty_metrices as um


def evaluate(data_dir, prediction_dir, ground_truth_dir, uncertainty_dir, labels, end=None, step=None, parallel=False):
    if end is not None:
        thresholds = np.arange(0.0, end, step)
    else:
        thresholds = None
    print("Thresholds: ", thresholds)
    prediction_filenames = utils.load_filenames(prediction_dir)
    ground_truth_filenames = [os.path.join(ground_truth_dir, os.path.basename(prediction_filename)) for prediction_filename in prediction_filenames]
    uncertainty_filenames = []

    for prediction_filename in prediction_filenames:
        basename = os.path.basename(prediction_filename)
        uncertainty_label_filenames = []
        for label in labels:
            filename = os.path.join(uncertainty_dir, '{}_{}.nii.gz'.format(basename[:-7], label))
            uncertainty_label_filenames.append(filename)
        uncertainty_filenames.append(uncertainty_label_filenames)
    uncertainty_filenames = np.asarray(uncertainty_filenames)

    prediction_filenames, ground_truth_filenames, uncertainty_filenames = remove_missing_cases(prediction_filenames, ground_truth_filenames, uncertainty_filenames)
    results = []

    start_time = time.time()
    for i, label in enumerate(tqdm(labels)):
        predictions, ground_truths, uncertainties = load_data(prediction_filenames, ground_truth_filenames, uncertainty_filenames[:, i])
        predictions, ground_truths = binarize_data_by_label(predictions, ground_truths, label)
        if thresholds is None:
            thresholds = find_best_threshold(predictions, ground_truths, uncertainties)
        if isinstance(thresholds, Number):
            thresholds = [thresholds]
        if not parallel:
            for threshold in thresholds:
                result = evaluate_threshold(predictions, ground_truths, uncertainties, threshold)
                # result["label"] = label
                # result["threshold"] = threshold
                results.append(result)
        else:
            with Pool(processes=4) as pool:  # multiprocessing.cpu_count() kills memory
                results = pool.map(partial(evaluate_threshold, predictions=predictions, ground_truths=ground_truths, uncertainties=uncertainties), thresholds)
            results = [{"label": label, "threshold": thresholds[i], "dice_score": results[i][0], "uncertainty_sum": results[i][1]} for i in range(len(results))]  # TODO: Old

        for key in results[0].keys():
            plt.plot(thresholds, [result[key] for result in results], label=key)
        plt.legend(loc="upper left")
        plt.xlim(0, end)
        plt.ylim(0, 2)
        plt.savefig(data_dir + os.path.basename(uncertainty_dir[:-1]) + "_end" + str(end) + "_step" + str(step) + '.png')

    for result in results:
        print(result)

    with open(data_dir + os.path.basename(uncertainty_dir[:-1]) + "_end" + str(end) + "_step" + str(step) + ".pkl", 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Elapsed time (evaluate): ", time.time() - start_time)


def load_data(prediction_filenames, ground_truth_filenames, uncertainty_filenames):
    print("Loading data...")
    predictions, ground_truths, uncertainties = [], [], []
    target_shape = (512, 512, 260)

    for i in tqdm(range(len(prediction_filenames))):
        prediction = utils.load_nifty(prediction_filenames[i])[0].astype(np.float16)
        ground_truth = utils.load_nifty(ground_truth_filenames[i])[0].astype(np.float16)
        uncertainty = utils.load_nifty(uncertainty_filenames[i])[0].astype(np.float16)
        uncertainty = np.nan_to_num(uncertainty)
        prediction = utils.interpolate(prediction, target_shape, mask=True)
        ground_truth = utils.interpolate(ground_truth, target_shape, mask=True)
        uncertainty = utils.interpolate(uncertainty, target_shape, mask=False)
        predictions.append(prediction)
        ground_truths.append(ground_truth)
        uncertainties.append(uncertainty)

    predictions = np.asarray(predictions)
    ground_truths = np.asarray(ground_truths)
    uncertainties = np.asarray(uncertainties)
    print("Finished loading data")
    return predictions, ground_truths, uncertainties


def binarize_data_by_label(predictions, ground_truths, label):
    predictions = np.rint(predictions)
    ground_truths = np.rint(ground_truths)
    predictions = predictions.astype(int)
    ground_truths = ground_truths.astype(int)
    predictions[predictions != label] = 0
    ground_truths[ground_truths != label] = 0
    predictions[predictions == label] = 1
    ground_truths[ground_truths == label] = 1
    return predictions, ground_truths


def find_best_threshold(predictions, ground_truths, uncertainties):

    def _evaluate_threshold(threshold):
        return 1 - evaluate_threshold(predictions, ground_truths, uncertainties, threshold)["uncertainty_filtered_dice"]

    start_time = time.time()
    result = optimize.minimize_scalar(_evaluate_threshold, bounds=(0, 1))
    print("Elapsed time (find_best_threshold): ", time.time() - start_time)
    print("Success: ", result.success)
    print("best_threshold: ", result.x)
    return result.x


def evaluate_threshold(predictions, ground_truths, uncertainties, threshold):
    print("Threshold: ", threshold)
    start_time = time.time()

    thresholded_uncertainties = threshold_uncertainty(uncertainties, threshold)
    # uncertainty_sum = np.sum(thresholded_uncertainties)
    uncertainty_filtered_dice = um.uncertainty_filtered_dice(predictions, ground_truths, thresholded_uncertainties)
    relaxed_uncertainty_dice = um.relaxed_uncertainty_dice(predictions, ground_truths, thresholded_uncertainties)
    certain_missclassification2uncertainty_ratio = um.certain_missclassification2uncertainty_ratio(predictions, ground_truths, thresholded_uncertainties)
    certain_missclassification2gt_ratio = um.certain_missclassification2gt_ratio(predictions, ground_truths, thresholded_uncertainties)
    certain_missclassification2prediction_ratio = um.certain_missclassification2prediction_ratio(predictions, ground_truths, thresholded_uncertainties)
    uncertainty2prediction_ratio = um.uncertainty2prediction_ratio(thresholded_uncertainties, predictions)
    uncertainty2gt_ratio = um.uncertainty2gt_ratio(thresholded_uncertainties, ground_truths)
    certain2prediction_ratio = um.certain2prediction_ratio(thresholded_uncertainties, predictions)
    print("Elapsed time (evaluate_threshold): ", time.time() - start_time)

    return {"uncertainty_filtered_dice": uncertainty_filtered_dice, "relaxed_uncertainty_dice": relaxed_uncertainty_dice,
            "certain_missclassification2uncertainty_ratio": certain_missclassification2uncertainty_ratio, "certain_missclassification2gt_ratio": certain_missclassification2gt_ratio,
            "certain_missclassification2prediction_ratio": certain_missclassification2prediction_ratio,
            "uncertainty2prediction_ratio": uncertainty2prediction_ratio, "uncertainty2gt_ratio": uncertainty2gt_ratio, "certain2prediction_ratio": certain2prediction_ratio}


def threshold_uncertainty(uncertainty, threshold):
    thresholded_uncertainty = copy.deepcopy(uncertainty)
    thresholded_uncertainty[thresholded_uncertainty <= threshold] = 0
    thresholded_uncertainty[thresholded_uncertainty > threshold] = 1
    thresholded_uncertainty = thresholded_uncertainty.astype(int)
    return thresholded_uncertainty


def remove_missing_cases(prediction_filenames, ground_truth_filenames, uncertainty_filenames):
    existing_prediction_filenames = []
    existing_ground_truth_filenames = []
    existing_uncertainty_filenames = []
    for i in range(len(prediction_filenames)):
        exists = True
        if not os.path.isfile(prediction_filenames[i]):
            exists = False
        if not os.path.isfile(ground_truth_filenames[i]):
            exists = False
        for uncertainty_filename_label in uncertainty_filenames[i]:
            if not os.path.isfile(uncertainty_filename_label):
                exists = False
        if exists:
            existing_prediction_filenames.append(prediction_filenames[i])
            existing_ground_truth_filenames.append(ground_truth_filenames[i])
            existing_uncertainty_filenames.append(uncertainty_filenames[i])
    existing_uncertainty_filenames = np.asarray(existing_uncertainty_filenames)
    return existing_prediction_filenames, existing_ground_truth_filenames, existing_uncertainty_filenames


if __name__ == '__main__':
    # prediction_dir = "/gris/gris-f/homelv/kgotkows/datasets/prostate/Task05_Prostate/predictions_tta_Tr/"
    # ground_truth_dir = "/gris/gris-f/homelv/kgotkows/datasets/prostate/Task05_Prostate/labelsTr/"
    # uncertainty_dir = "/gris/gris-f/homelv/kgotkows/datasets/prostate/Task05_Prostate/uncertainties_tta_Tr/"
    # evaluate(prediction_dir, ground_truth_dir, uncertainty_dir, labels=(1, 2))

    data_dir = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task086_frankfurt2/"
    prediction_dir = data_dir + "predictions_with_tta_merged/"
    ground_truth_dir = data_dir + "labelsTr/"
    uncertainty_dir = data_dir + "uncertainties_tta_variance/"
    end = 0.24  # 0.003
    step = 0.02  # 0.0002
    thresholds = np.arange(0.0, end, step)
    plot = False
    if not plot:
        evaluate(data_dir, prediction_dir, ground_truth_dir, uncertainty_dir, labels=(1,), end=end, step=step)
    else:
        with open(data_dir + os.path.basename(uncertainty_dir[:-1]) + "_end" + str(end) + "_step" + str(step) + ".pkl", 'rb') as handle:
            results = pickle.load(handle)

        for result in results:
            print(result)

        for key in results[0].keys():
            plt.plot(np.arange(0.0, end, step), [result[key] for result in results], label=key)
        plt.legend(loc="upper left")
        plt.xlim(0, end)
        plt.ylim(0, 2)
        plt.savefig(data_dir + os.path.basename(uncertainty_dir[:-1]) + "_end" + str(end) + "_step" + str(step) + '.png')



