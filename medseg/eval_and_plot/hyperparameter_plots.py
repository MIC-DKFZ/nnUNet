import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib.gridspec as gridspec

# x = np.arange(0.0, 1.0, 0.1)
# y_1 = np.asarray([0.03, 0.05, 0.07, 0.11, 0.12, 0.14, 0.18, 0.20, 0.24, 0.28])
# y_2 = np.asarray([0.1, 0.2, 0.23, 0.5, 0.57, 0.63, 0.8, 0.82, 0.84, 0.86])
#
# plt.plot(x, y_1, label="Annotation ratio")
# plt.plot(x, y_2, label="Dice")
# plt.show()

base_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/"
# task = "Task008_Pancreas_guided"  # Task008_Pancreas_guided  Task070_guided_all_public_ggo  Task002_BrainTumour_guided
set = "val"
class_label = 1
basenames = ["hyperparam_eval_results_V7_my_method_['slice_gap'].pkl",
             "hyperparam_eval_results_V7_my_method_['num_slices'].pkl",
             "hyperparam_eval_results_V7_my_method_['min_uncertainty'].pkl",
             "hyperparam_eval_results_V7_my_method_['max_slices_based_on_infected_slices'].pkl"]
names = ["Patch Gap", "Max Patches", "Min Relative Uncertainty", "Max Patch-Foreground Ratio"]
tasks = ["Task008_Pancreas_guided", "Task002_BrainTumour_guided", "Task070_guided_all_public_ggo"]
task_names = ["Pancreas", "Brain Tumor", "COVID-19"]
gridspec_indices = [[slice(0, 2), slice(0, 2)], [slice(0, 2), slice(2, 4)], [slice(2, 4), slice(1, 3)]]

for i, basename in enumerate(basenames):
    # fig, ax = plt.subplots(1, 3, figsize=(25,5))
    fig = plt.figure(constrained_layout=True, figsize=(12,7))
    gs = fig.add_gridspec(4, 4)
    for j, task in enumerate(tasks):
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
            min_value = np.min([np.min(result_y_dice), np.min(result_y_ratio)])
            max_value = np.max([np.max(result_y_dice), np.max(result_y_ratio)])
            diff_dice = np.max(result_y_dice) - np.min(result_y_dice)
            diff_ratio = np.max(result_y_ratio) - np.min(result_y_ratio)
            diff_max = np.max([diff_dice, diff_ratio])
            ax = fig.add_subplot(gs[gridspec_indices[j][0], gridspec_indices[j][1]])
            ax.set_ylabel('Dice Score', color="g")
            ax.plot(result_x, result_y_dice, label="Dice", color="g")
            ax.set_ylim(np.min(result_y_dice), np.min(result_y_dice) + diff_max)
            ax.tick_params(axis='y', labelcolor="g")
            ax2 = ax.twinx()
            ax2.set_ylabel('Annotation Ratio', color="r")
            ax2.plot(result_x, result_y_ratio, label="Annotation Ratio", color="r")
            ax2.set_ylim(np.min(result_y_ratio), np.min(result_y_ratio) + diff_max)
            ax2.tick_params(axis='y', labelcolor="r")
            ax.set_title(task_names[j])
    fig.suptitle(names[i], fontsize=16)
    fig.tight_layout()
    # plt.show()
    plt.savefig(base_path + "Evaluation/Hyperparameter Evaluation/" + names[i] + ".png", dpi=150)
    plt.clf()
