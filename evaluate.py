from nnunet.evaluation.evaluator import evaluate_folder
import numpy as np
import pickle
from medseg import utils


def evaluate(ground_truths, predictions):
    labels = get_labels(ground_truths)
    result = evaluate_folder(ground_truths, predictions, labels)

    case_dice_scores = []
    for case in result["all"]:
        case_dice_scores.append(case["1"]["Dice"])

    case_dice_scores = np.asarray([float(score) for score in case_dice_scores])

    dice_scores = []
    print("Mean:")
    for label in result["mean"].keys():
        print(str(label) + ": " + str(result["mean"][label]["Dice"]))
    print("Median:")
    for label in result["median"].keys():
        print(str(label) + ": " + str(result["median"][label]["Dice"]))
    for label in result["mean"].keys():
        dice_scores.append(result["mean"][label]["Dice"])

    return result["mean"]["1"]["Dice"], result["median"]["1"]["Dice"]


def get_labels(gt_path):
    labels = []
    gt_filenames = utils.load_filenames(gt_path)
    for gt_filename in gt_filenames:
        gt, _, _, _ = utils.load_nifty(gt_filename)
        unique = np.unique(gt)
        labels.append(unique)
    labels = np.unique(labels)
    return labels


if __name__ == '__main__':
    ground_truths = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task002_BrainTumour_guided/refinement_test/labels/"
    predictions = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task002_BrainTumour_guided/refinement_test/basic_predictions/"

    evaluate(ground_truths, predictions)