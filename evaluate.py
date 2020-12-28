from nnunet.evaluation.evaluator import evaluate_folder
import numpy as np
import pickle

ground_truths = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task086_frankfurt2/labelsTr/"
predictions = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task086_frankfurt2/predictions_with_tta/"
scores_name = "exp2_nnunet"

labels = (0, 1)
result = evaluate_folder(ground_truths, predictions, labels)

case_dice_scores = []
for case in result["all"]:
    case_dice_scores.append(case["1"]["Dice"])

case_dice_scores = np.asarray([float(score) for score in case_dice_scores])

# with open('/gris/gris-f/homelv/kgotkows/datasets/covid19/scores/' + scores_name + '.pkl', 'wb') as handle:
#     pickle.dump(case_dice_scores, handle, protocol=pickle.DEFAULT_PROTOCOL)

dice_scores = []
for label in result["mean"].keys():
    print(str(label) + ": " + str(result["mean"][label]["Dice"]))
for label in result["mean"].keys():
    dice_scores.append(result["mean"][label]["Dice"])

