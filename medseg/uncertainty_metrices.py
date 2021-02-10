import numpy as np
import sys


def comp_confusion_matrix(prediction, ground_truth):
    """
    Compute normal confusion matrix.
    """
    tp = ((prediction == 1) & (ground_truth == 1))
    tn = ((prediction == 0) & (ground_truth == 0))
    fp = ((prediction == 1) & (ground_truth == 0))
    fn = ((prediction == 0) & (ground_truth == 1))
    tp = np.sum(tp)
    tn = np.sum(tn)
    fp = np.sum(fp)
    fn = np.sum(fn)
    return tp, fp, tn, fn


def uncertainty_filtered_dice(prediction, ground_truth, uncertainty):
    """
    Compute confusion matrix and ignore all elements with positive uncertainty.
    """
    tp = ((prediction == 1) & (ground_truth == 1) & (uncertainty == 0))
    tn = ((prediction == 0) & (ground_truth == 0) & (uncertainty == 0))
    fp = ((prediction == 1) & (ground_truth == 0) & (uncertainty == 0))
    fn = ((prediction == 0) & (ground_truth == 1) & (uncertainty == 0))
    tp = np.sum(tp)
    tn = np.sum(tn)
    fp = np.sum(fp)
    fn = np.sum(fn)
    return comp_dice_score(tp, fp, tn, fn)


def relaxed_uncertainty_dice(prediction, ground_truth, uncertainty):
    """
    Compute confusion matrix and and regard all elements with positive uncertainty as TP or TN.
    Intuition: It should not be punished if the prediction for an element is wrong when it is uncertain.
    """
    tp = ((prediction == 1) & (ground_truth == 1) & (uncertainty == 0)) | ((ground_truth == 1) & (uncertainty == 1))
    tn = ((prediction == 0) & (ground_truth == 0) & (uncertainty == 0)) | ((ground_truth == 0) & (uncertainty == 1))
    fp = ((prediction == 1) & (ground_truth == 0) & (uncertainty == 0))
    fn = ((prediction == 0) & (ground_truth == 1) & (uncertainty == 0))
    tp = np.sum(tp)
    tn = np.sum(tn)
    fp = np.sum(fp)
    fn = np.sum(fn)
    return comp_dice_score(tp, fp, tn, fn)


def certain_missclassification2uncertainty_ratio(prediction, ground_truth, uncertainty):
    """
    Compute the ratio of missclassification where the prediction was certain to total uncertainty.
    Intuition: An estimate on how much a model fails to be uncertain when it should be uncertain regarding the total uncertainty.
    """
    tp = (uncertainty == 1)
    tn = [0]
    fp = ((prediction == 1) & (ground_truth == 0) & (uncertainty == 0))
    fn = ((prediction == 0) & (ground_truth == 1) & (uncertainty == 0))
    tp = np.sum(tp)
    tn = np.sum(tn)
    fp = np.sum(fp)
    fn = np.sum(fn)
    return (fp + fn) / (tp + sys.float_info.epsilon)


def certain_missclassification2gt_ratio(prediction, ground_truth, uncertainty):
    """
    Compute the ratio of missclassification where the prediction was certain to total ground truth.
    Intuition: An estimate on how much a model fails to be uncertain when it should be uncertain regarding the total ground truth.
    """
    fp = ((prediction == 1) & (ground_truth == 0) & (uncertainty == 0))
    fn = ((prediction == 0) & (ground_truth == 1) & (uncertainty == 0))
    fp = np.sum(fp)
    fn = np.sum(fn)
    return (fp + fn) / (np.sum(ground_truth) + sys.float_info.epsilon)


def certain_missclassification2prediction_ratio(prediction, ground_truth, uncertainty):
    """
    Compute the ratio of missclassification where the prediction was certain to total ground truth.
    Intuition: An estimate on how much a model fails to be uncertain when it should be uncertain regarding the total ground truth.
    """
    fp = ((prediction == 1) & (ground_truth == 0) & (uncertainty == 0))
    fn = ((prediction == 0) & (ground_truth == 1) & (uncertainty == 0))
    fp = np.sum(fp)
    fn = np.sum(fn)
    return (fp + fn) / (np.sum(prediction) + sys.float_info.epsilon)


def uncertainty2prediction_ratio(uncertainty, prediction):
    """
    Compute the ratio of total uncertainty to total prediction.
    Intuition: An estimate on how uncertain a model is about its predictions.
    """
    return np.sum(uncertainty) / (np.sum(prediction) + sys.float_info.epsilon)


def uncertainty2gt_ratio(uncertainty, ground_truth):
    """
    Compute the ratio of total uncertainty to total ground truth.
    Intuition: An estimate on how uncertain a model is about predicting a class for the ground truth object.
    """
    return np.sum(uncertainty) / (np.sum(ground_truth) + sys.float_info.epsilon)


def certain2prediction_ratio(uncertainty, prediction):
    """
    Compute the ratio of certain positive predictions to total positive predictions.
    Intuition: An estimate on how much of the actual prediction is still left and is not considered uncertain.
    """
    certain_predictions = ((prediction == 1) & (uncertainty == 0))
    total_predictions = (prediction == 1)
    certain_predictions = np.sum(certain_predictions)
    total_predictions = np.sum(total_predictions)
    return certain_predictions / (total_predictions + sys.float_info.epsilon)


def comp_dice_score(tp, fp, tn, fn):
    """
    Binary dice score.
    """
    if tp + fp + fn == 0:
        return 1
    else:
        return (2*tp) / (2*tp + fp + fn)