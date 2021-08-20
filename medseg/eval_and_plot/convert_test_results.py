import pickle
import pandas as pd
import numpy as np
import json
import os
from medseg import utils

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
    df.to_csv(save_path + "dice/automatic.csv", index=False)
    for key in class_dice_scores.keys():
        print("Class: {}, Mean: {}, Median: {}".format(key, np.mean(class_dice_scores[key]), np.median(class_dice_scores[key])))


def test_results2csv(load_filename, save_path):
    basename = os.path.basename(load_filename)
    with open(load_filename, 'rb') as handle:
        results = pickle.load(handle)

    def get_class_labels():
        return list(results["prediction_result"]["all"][0].keys())[:-2]

    class_labels_str = get_class_labels()
    class_labels = np.asarray(class_labels_str, dtype=int)
    class_dice_scores = {}
    for i in range(len(class_labels)):
        class_dice_scores[class_labels[i]] = [x[str(class_labels_str[i])]["Dice"] for x in results["prediction_result"]["all"]]
    df = pd.DataFrame.from_dict(class_dice_scores)
    df.to_csv(save_path + "dice/" + basename[3:-4] + ".csv", index=False)
    for key in class_dice_scores.keys():
        print("Name: {}, Class: {}, Mean: {}, Median: {}".format(basename[3:-4], key, np.mean(class_dice_scores[key]), np.median(class_dice_scores[key])))

    if "recommended_result" in results:
        recommended_slices = [x["recommended_slices"] for x in results["recommended_result"]]
        gt_infected_slices = [x["gt_infected_slices"] for x in results["recommended_result"]]
        annotation_ratio = np.asarray(recommended_slices) / np.asarray(gt_infected_slices)
        print("Name: {}, Ratio mean: {}".format(basename[3:-4], np.mean(annotation_ratio)))
        annotation_ratio = {"Ratio": annotation_ratio}
        df = pd.DataFrame.from_dict(annotation_ratio)
        df.to_csv(save_path + "ratio/" + basename[3:-4] + ".csv", index=False)



if __name__ == '__main__':
    tasks = ["Task070_guided_all_public_ggo", "Task002_BrainTumour_guided", "Task008_Pancreas_guided"]
    for task in tasks:
        basepath = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/" + task + "/refinement_test/"

        load_filename = basepath + "basic_predictions_summary.json"
        save_filename = basepath + "eval_results/processed/"
        basic_predictions2csv(load_filename, save_filename)

        result_filenames = utils.load_filenames(basepath + "eval_results/raw/", extensions=None)
        save_filename = basepath + "eval_results/processed/"

        for filename in result_filenames:
            test_results2csv(filename, save_filename)