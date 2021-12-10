from nnunet.evaluation.evaluator import evaluate_folder
import numpy as np
import pickle
from i3Deep import utils


def evaluate(ground_truths, predictions, labels):
    # if labels is None:
    #     labels = get_labels(ground_truths)
    # print("Labels: ", labels)
    result = evaluate_folder(ground_truths, predictions, labels)

    # case_dice_scores = []
    # for case in result["all"]:
    #     case_dice_scores.append(case["1"]["Dice"])
    #
    # case_dice_scores = np.asarray([float(score) for score in case_dice_scores])

    # dice_scores = []
    print("Mean:")
    for label in result["mean"].keys():
        print(str(label) + ": " + str(result["mean"][label]["Dice"]))
    print("Median:")
    for label in result["median"].keys():
        print(str(label) + ": " + str(result["median"][label]["Dice"]))
    # for label in result["mean"].keys():
    #     dice_scores.append(result["mean"][label]["Dice"])

    # return result["mean"]["1"]["Dice"], result["median"]["1"]["Dice"]
    # print("RESULT: ", result)
    return result


def get_labels(gt_path):
    labels = []
    gt_filenames = utils.load_filenames(gt_path)
    for gt_filename in gt_filenames:
        gt, _, _, _ = utils.load_nifty(gt_filename)
        unique = np.unique(gt)
        labels.append(unique)
    labels = np.unique(labels.flatten())
    return labels


if __name__ == '__main__':
    ground_truths = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task008_Pancreas_guided/refinement_test/labels2/"
    predictions = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task008_Pancreas_guided/refinement_test/refined_predictions/GraphCut1/"

    result = evaluate(ground_truths, predictions, labels=[0, 1, 2])
    result = {"recommended_result": [], "prediction_result": result}
    with open("/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task008_Pancreas_guided/refinement_test/eval_results/raw/V7_GraphCut1.pkl", 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)