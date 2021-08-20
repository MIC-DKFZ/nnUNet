import matplotlib.pyplot as plt
import numpy as np
import pickle

# x = np.arange(0.0, 1.0, 0.1)
# y_1 = np.asarray([0.03, 0.05, 0.07, 0.11, 0.12, 0.14, 0.18, 0.20, 0.24, 0.28])
# y_2 = np.asarray([0.1, 0.2, 0.23, 0.5, 0.57, 0.63, 0.8, 0.82, 0.84, 0.86])
#
# plt.plot(x, y_1, label="Annotation ratio")
# plt.plot(x, y_2, label="Dice")
# plt.show()

base_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/"
task = "Task002_BrainTumour_guided"
set = "val"
class_label = 1
basenames = ["hyperparam_eval_results_V7_my_method_['slice_gap'].pkl",
             "hyperparam_eval_results_V7_my_method_['num_slices'].pkl",
             "hyperparam_eval_results_V7_my_method_['min_uncertainty'].pkl",
             "hyperparam_eval_results_V7_my_method_['max_slices_based_on_infected_slices'].pkl"]
names = ["Patch Gap", "Max Patches", "Min Relative Uncertainty", "Max Patch-Foreground Ratio"]

for i, basename in enumerate(basenames):
    filename = base_path + task + "/refinement_" + set + "/GridSearchResults/" + basename

    with open(filename, 'rb') as handle:
        results = pickle.load(handle)

    for param_key in results.keys():
        result_x = list(results[param_key].keys())
        result_y_ratio, result_y_dice = [], []
        for param_value in results[param_key].values():
            recommended_slices = [x["recommended_slices"] for x in param_value["recommended_result"]]
            gt_infected_slices = [x["gt_infected_slices"] for x in param_value["recommended_result"]]
            # annotation_ratio = np.sum(recommended_slices) / np.sum(gt_infected_slices)
            annotation_ratio = np.asarray(recommended_slices) / np.asarray(gt_infected_slices)
            # print(annotation_ratio)
            annotation_ratio = np.mean(annotation_ratio)
            result_y_ratio.append(annotation_ratio)
            result_y_dice.append(param_value["prediction_result"]["mean"][str(class_label)]["Dice"])
        print("Dice: ", result_y_dice)
        print("Ratio: ", result_y_ratio)
        fig, ax1 = plt.subplots()
        ax1.set_ylabel('Dice Score', color="g")
        ax1.plot(result_x, result_y_dice, label="Dice", color="g")
        ax1.tick_params(axis='y', labelcolor="g")
        ax2 = ax1.twinx()
        ax2.set_ylabel('Annotation ratio', color="r")
        plt.plot(result_x, result_y_ratio, label="Annotation ratio", color="r")
        ax2.tick_params(axis='y', labelcolor="r")
        plt.title(names[i])
        fig.tight_layout()
        # plt.show()
        plt.savefig(base_path + task + "/refinement_" + set + "/GridSearchResults/" + names[i] + ".png")
        plt.clf()
