import pickle
import pandas as pd
import numpy as np
import json
import os

def basic_predictions2csv(load_filename, save_path):
    with open(load_filename) as json_file:
        basic_predictions = json.load(json_file)

    def get_class_labels():
        return list(basic_predictions["results"]["all"][0].keys())[:-2]

    class_labels_str = get_class_labels()
    class_labels = np.asarray(class_labels_str, dtype=int)
    class_dice_scores = {}
    for i in range(len(class_labels)):
        class_dice_scores[class_labels[i]] = [x[str(class_labels_str[i])]["Dice"] for x in basic_predictions["results"]["all"]]
    df = pd.DataFrame.from_dict(class_dice_scores)
    print(df)
    df.to_csv(save_path + "basic_predictions.csv", index=False)


def test_results2csv(load_filename, save_path):
    basename = os.path.basename(load_filename)
    with open(load_filename, 'rb') as handle:
        results = pickle.load(handle)

    def get_class_labels():
        return list(results["prediction_result"]["results"]["all"][0].keys())[:-2]

    class_labels_str = get_class_labels()
    class_labels = np.asarray(class_labels_str, dtype=int)
    class_dice_scores = {}
    for i in range(len(class_labels)):
        class_dice_scores[class_labels[i]] = [x[str(class_labels_str[i])]["Dice"] for x in results["prediction_result"]["results"]["all"]]
    df = pd.DataFrame.from_dict(class_dice_scores)
    print(df)
    df.to_csv(save_path + basename + "_scores.csv", index=False)

    recommended_slices = [x["recommended_slices"] for x in results["recommended_result"]]
    gt_infected_slices = [x["gt_infected_slices"] for x in results["recommended_result"]]
    annotation_ratio1 = np.sum(recommended_slices) / np.sum(gt_infected_slices)
    annotation_ratio = np.asarray(recommended_slices) / np.asarray(gt_infected_slices)
    annotation_ratio2 = np.mean(annotation_ratio)
    print("annotation_ratio1: ", annotation_ratio1)
    print("annotation_ratio2: ", annotation_ratio2)



if __name__ == '__main__':
    # load_filename = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task070_guided_all_public_ggo/refinement_test/basic_predictions_summary.json"
    # save_filename = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task070_guided_all_public_ggo/refinement_test/eval_results/processed/"
    # basic_predictions2csv(load_filename, save_filename)

    load_filename = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task070_guided_all_public_ggo/refinement_test/basic_predictions_summary.json"
    save_filename = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task070_guided_all_public_ggo/refinement_test/eval_results/processed/"
    test_results2csv(load_filename, save_filename)