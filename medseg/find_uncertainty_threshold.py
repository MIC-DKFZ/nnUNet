from medseg import utils
import numpy as np
import os
import sys
from tqdm import tqdm
from scipy import optimize
import time

def evaluate(prediction_dir, ground_truth_dir, uncertainty_dir, labels, thresholds=None):
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

    for i, label in enumerate(tqdm(labels)):
        predictions, ground_truths, uncertainties = load_data(prediction_filenames, ground_truth_filenames, uncertainty_filenames[:, i])
        predictions_label, ground_truths_label = binarize_data_by_label(predictions, ground_truths, label)
        if thresholds is None:
            thresholds = find_best_threshold(predictions_label, ground_truths_label, uncertainties)
        if not isinstance(thresholds, list):
            thresholds = [thresholds]
        for threshold in thresholds:
            dice_score = evaluate_threshold(predictions_label, ground_truths_label, uncertainties, threshold)
            results.append({"label": label, "threshold": threshold, "dice_score": dice_score})

    for i in range(len(results)):
        label = results[i]["label"]
        best_threshold = results[i]["threshold"]
        dice_score = results[i]["dice_score"]
        print("Label: {}, Threshold: {}, Dice Score: {}".format(label, best_threshold, dice_score))


def load_data(prediction_filenames, ground_truth_filenames, uncertainty_filenames):
    print("Loading data...")
    predictions, ground_truths, uncertainties = [], [], []
    target_shape = (512, 512, 260)

    for i in tqdm(range(len(prediction_filenames))):
        prediction = utils.load_nifty(prediction_filenames[i])[0].astype(np.float16)
        ground_truth = utils.load_nifty(ground_truth_filenames[i])[0].astype(np.float16)
        uncertainty = utils.load_nifty(uncertainty_filenames[i])[0].astype(np.float16)
        prediction = utils.interpolate(prediction, target_shape, mask=True)
        ground_truth = utils.interpolate(ground_truth, target_shape, mask=True)
        uncertainty = utils.interpolate(uncertainty, target_shape, mask=False)
        predictions.append(prediction)
        ground_truths.append(ground_truth)
        uncertainties.append(uncertainty)

    print("Finished loading data")
    return predictions, ground_truths, uncertainties


def binarize_data_by_label(predictions, ground_truths, label):
    predictions_label, ground_truths_label = [], []

    for i in range(len(predictions)):
        prediction = np.rint(predictions[i])
        ground_truth = np.rint(ground_truths[i])
        prediction = prediction.astype(int)
        ground_truth = ground_truth.astype(int)
        prediction[prediction != label] = 0
        ground_truth[ground_truth != label] = 0
        prediction[prediction == label] = 1
        ground_truth[ground_truth == label] = 1
        predictions_label.append(prediction)
        ground_truths_label.append(ground_truth)

    return predictions_label, ground_truths_label


def find_best_threshold(predictions, ground_truths, uncertainties):

    def _evaluate_threshold(threshold):
        return 1 - evaluate_threshold(predictions, ground_truths, uncertainties, threshold)

    start_time = time.time()
    result = optimize.minimize_scalar(_evaluate_threshold, bounds=(0, 5))
    print("Elapsed time: ", time.time() - start_time)
    print("Success: ", result.success)
    print("best_threshold: ", result.x)
    return result.x


def evaluate_threshold(predictions, ground_truths, uncertainties, threshold):
    print("Threshold: ", threshold)
    dice_scores = []
    #tp, fp, tn, fn = comp_partial_confusion_matrix(prediction, ground_truth)

    for i in range(len(predictions)):
        thresholded_uncertainty = threshold_uncertainty(uncertainties[i], threshold)
        tp, fp, tn, fn = comp_uncertainty_confusion_matrix(predictions[i], ground_truths[i], thresholded_uncertainty)
        dice_score = comp_dice_score(tp, fp, tn, fn)
        dice_scores.append(dice_score)

    dice_scores = np.asarray(dice_scores)
    dice_scores = np.mean(dice_scores)
    return dice_scores


def threshold_uncertainty(uncertainty, threshold):
    uncertainty[uncertainty <= threshold] = 0
    uncertainty[uncertainty > threshold] = 1
    return uncertainty


def comp_partial_confusion_matrix(prediction, ground_truth):
    """
    Compute normal confusion matrix. Ignore all elements with positive uncertainty.
    :param prediction:
    :param ground_truth:
    :param uncertainty:
    :return:
    """
    tp = ((prediction == 1) & (ground_truth == 1))
    tn = ((prediction == 0) & (ground_truth == 0))
    fp = ((prediction == 1) & (ground_truth == 0))
    fn = ((prediction == 0) & (ground_truth == 1))
    return tp, fp, tn, fn


# def comp_uncertainty_confusion_matrix(tp, fp, tn, fn, uncertainty):
#     """
#     Compute normal confusion matrix. Ignore all elements with positive uncertainty.
#     :param prediction:
#     :param ground_truth:
#     :param uncertainty:
#     :return:
#     """
#     tp = (tp & (uncertainty == 0))
#     tn = (tn & (uncertainty == 0))
#     fp = (fp & (uncertainty == 0))
#     fn = (fn & (uncertainty == 0))
#     tp = np.sum(tp)
#     tn = np.sum(tn)
#     fp = np.sum(fp)
#     fn = np.sum(fn)
#     return tp, fp, tn, fn


def comp_uncertainty_confusion_matrix(prediction, ground_truth, uncertainty):
    """
    Compute normal confusion matrix. Ignore all elements with positive uncertainty.
    :param prediction:
    :param ground_truth:
    :param uncertainty:
    :return:
    """
    tp = ((prediction == 1) & (ground_truth == 1) & (uncertainty == 0))
    tn = ((prediction == 0) & (ground_truth == 0) & (uncertainty == 0))
    fp = ((prediction == 1) & (ground_truth == 0) & (uncertainty == 0))
    fn = ((prediction == 0) & (ground_truth == 1) & (uncertainty == 0))
    tp = np.sum(tp)
    tn = np.sum(tn)
    fp = np.sum(fp)
    fn = np.sum(fn)
    return tp, fp, tn, fn


def comp_dice_score(tp, fp, tn, fn):
    if tp + fp + fn == 0:
        return 1
    else:
        return (2*tp) / (2*tp + fp + fn)


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

    prediction_dir = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task086_frankfurt2/predictions_with_tta/"
    ground_truth_dir = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task086_frankfurt2/labelsTr/"
    uncertainty_dir = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task086_frankfurt2/uncertainties_tta/"
    evaluate(prediction_dir, ground_truth_dir, uncertainty_dir, labels=(1,), thresholds=[2.6180339603380443])
