from i3Deep import utils
import numpy as np
import os
import sys
from tqdm import tqdm
import copy


def evaluate(prediction_dir, ground_truth_dir, uncertainty_dir, labels):
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
        results.append(evaluate_label(prediction_filenames, ground_truth_filenames, uncertainty_filenames[:, i], label))

    for i in range(len(labels)):
        label = results[i]["label"]
        thresholds = results[i]["thresholds"]
        threshold_scores = results[i]["threshold_scores"]
        for i in range(len(thresholds)):
            print("Label: {}, Threshold: {}, Dice Score: {}, Uncertainty Dice Score 1: {}, Uncertainty Dice Score 2: {}, Uncertainty Miss Coverage Ratio: {}, Uncertainty GT Ratio: {}".format(
                label, thresholds[i], round(threshold_scores[i][0], 3), round(threshold_scores[i][1], 3), round(threshold_scores[i][2], 3), round(threshold_scores[i][3], 3), round(threshold_scores[i][4], 3)))
        print("---------------------------------------")


def evaluate_label(prediction_filenames, ground_truth_filenames, uncertainty_filenames, label):
    threshold_scores = []
    thresholds = [0.5]  # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for i in tqdm(range(len(prediction_filenames))):
        prediction = utils.load_nifty(prediction_filenames[i])[0].astype(np.float16)
        ground_truth = utils.load_nifty(ground_truth_filenames[i])[0].astype(np.float16)
        uncertainty = utils.load_nifty(uncertainty_filenames[i])[0].astype(np.float16)
        case_threshold_scores = evaluate_case(prediction, ground_truth, uncertainty, thresholds, label)
        threshold_scores.append(case_threshold_scores)
    threshold_scores = np.asarray(threshold_scores)
    threshold_scores = np.mean(threshold_scores, axis=0)
    return {"label": label, "thresholds": thresholds, "threshold_scores": threshold_scores}


def evaluate_case(prediction, ground_truth, uncertainty, thresholds, label):
    prediction = np.rint(prediction)
    ground_truth = np.rint(ground_truth)
    prediction = prediction.astype(int)
    ground_truth = ground_truth.astype(int)
    prediction[prediction != label] = 0
    ground_truth[ground_truth != label] = 0
    prediction[prediction == label] = 1
    ground_truth[ground_truth == label] = 1
    uncertainty = utils.normalize(uncertainty)
    case_threshold_scores = []
    for threshold in tqdm(thresholds):
        thresholded_uncertainty = threshold_uncertainty(uncertainty, threshold)
        thresholded_uncertainty = thresholded_uncertainty.astype(int)
        metrices = comp_metrices(prediction, ground_truth, thresholded_uncertainty)
        case_threshold_scores.append(metrices)
    return case_threshold_scores


def comp_metrices(prediction, ground_truth, thresholded_uncertainty):
    tp, fp, tn, fn = comp_confusion_matrix(prediction, ground_truth)
    dice_score = comp_dice_score(tp, fp, tn, fn)
    tp, fp, tn, fn = comp_uncertainty_confusion_matrix1(prediction, ground_truth, thresholded_uncertainty)
    uncertainty_dice_score1 = comp_dice_score(tp, fp, tn, fn)
    tp, fp, tn, fn = comp_uncertainty_confusion_matrix2(prediction, ground_truth, thresholded_uncertainty)
    uncertainty_dice_score2 = comp_dice_score(tp, fp, tn, fn)
    uncertainty_miss_coverage_ratio = comp_uncertainty_miss_coverage_ratio(ground_truth, fp, fn)
    uncertainty_gt_ratio = comp_uncertainty_gt_ratio(thresholded_uncertainty, ground_truth)
    return [dice_score, uncertainty_dice_score1, uncertainty_dice_score2, uncertainty_miss_coverage_ratio, uncertainty_gt_ratio]


def threshold_uncertainty(uncertainty, threshold):
    thresholded_uncertainty = copy.deepcopy(uncertainty)
    thresholded_uncertainty[thresholded_uncertainty <= threshold] = 0
    thresholded_uncertainty[thresholded_uncertainty > threshold] = 1
    thresholded_uncertainty = thresholded_uncertainty.astype(int)
    return thresholded_uncertainty


def comp_confusion_matrix(prediction, ground_truth):
    tp = ((prediction == 1) & (ground_truth == 1))
    tn = ((prediction == 0) & (ground_truth == 0))
    fp = ((prediction == 1) & (ground_truth == 0))
    fn = ((prediction == 0) & (ground_truth == 1))
    tp = np.sum(tp)
    tn = np.sum(tn)
    fp = np.sum(fp)
    fn = np.sum(fn)
    return tp, fp, tn, fn


def comp_uncertainty_confusion_matrix1(prediction, ground_truth, uncertainty):
    tp = ((prediction == 1) & (ground_truth == 1) & (uncertainty == 0)) | ((ground_truth == 1) & (uncertainty == 1))
    tn = ((prediction == 0) & (ground_truth == 0) & (uncertainty == 0)) | ((ground_truth == 0) & (uncertainty == 1))
    fp = ((prediction == 1) & (ground_truth == 0) & (uncertainty == 0))
    fn = ((prediction == 0) & (ground_truth == 1) & (uncertainty == 0))
    tp = np.sum(tp)
    tn = np.sum(tn)
    fp = np.sum(fp)
    fn = np.sum(fn)
    return tp, fp, tn, fn


def comp_uncertainty_confusion_matrix2(prediction, ground_truth, uncertainty):
    tp = (uncertainty == 1)
    tn = [0]
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


def comp_uncertainty_miss_coverage_ratio(ground_truth, fp, fn):
    return (fp + fn) / (np.sum(ground_truth) + sys.float_info.epsilon)


def comp_uncertainty_gt_ratio(thresholded_uncertainty, ground_truth):
    return np.sum(thresholded_uncertainty) / (np.sum(ground_truth) + sys.float_info.epsilon)


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
    evaluate(prediction_dir, ground_truth_dir, uncertainty_dir, labels=(1,))
