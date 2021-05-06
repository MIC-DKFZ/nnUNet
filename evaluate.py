from nnunet.evaluation.evaluator import evaluate_folder
import numpy as np
import pickle


def evaluate(ground_truths, predictions, labels):
    result = evaluate_folder(ground_truths, predictions, labels)

    case_dice_scores = []
    for case in result["all"]:
        case_dice_scores.append(case["1"]["Dice"])

    case_dice_scores = np.asarray([float(score) for score in case_dice_scores])

    dice_scores = []
    for label in result["mean"].keys():
        print(str(label) + ": " + str(result["mean"][label]["Dice"]))
    for label in result["mean"].keys():
        dice_scores.append(result["mean"][label]["Dice"])

    return result["mean"]["1"]["Dice"]


if __name__ == '__main__':
    ground_truths = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task072_allGuided_ggo/labelsTs/"
    predictions = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task072_allGuided_ggo/Task072_allGuided_ggo_predictionsTs/"
    labels = (0, 1)

    evaluate(ground_truths, predictions, labels)