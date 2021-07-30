import matplotlib.pyplot as plt
import numpy as np
import pickle
import json
from pylab import plot, show, savefig, xlim, figure, ylim, legend, boxplot, setp, axes

# x = np.arange(0.0, 1.0, 0.1)
# y_1 = np.asarray([0.03, 0.05, 0.07, 0.11, 0.12, 0.14, 0.18, 0.20, 0.24, 0.28])
# y_2 = np.asarray([0.1, 0.2, 0.23, 0.5, 0.57, 0.63, 0.8, 0.82, 0.84, 0.86])
#
# plt.plot(x, y_1, label="Annotation ratio")
# plt.plot(x, y_2, label="Dice")
# plt.show()

base_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/"
task = ["Task070_guided_all_public_ggo", "Task002_BrainTumour_guided"]
set = "test"
basename = ["hyperparam_eval_results_V7_my_method_['slice_gap'].pkl", "hyperparam_eval_results_V7_my_method_['num_slices'].pkl"]
class_label = [[1], [1, 2, 3]]

dice_scores, basic_dice_scores = [], []
for i in range(len(task)):
    filename = base_path + task[i] + "/refinement_" + set + "/GridSearchResults/" + basename[i]
    basic_predictions_filename = base_path + task[i] + "/refinement_" + set + "/basic_predictions_summary.json"

    with open(filename, 'rb') as handle:
        results = pickle.load(handle)

    with open(basic_predictions_filename) as json_file:
        basic_predictions = json.load(json_file)

    results = results[list(results.keys())[0]]
    results = results[list(results.keys())[0]]

    recommended_slices = [x["recommended_slices"] for x in results["recommended_result"]]
    gt_infected_slices = [x["gt_infected_slices"] for x in results["recommended_result"]]
    # annotation_ratio = np.sum(recommended_slices) / np.sum(gt_infected_slices)
    annotation_ratio = np.asarray(recommended_slices) / np.asarray(gt_infected_slices)
    # print(annotation_ratio)
    annotation_ratio = np.mean(annotation_ratio)
    for j in range(len(class_label[i])):
        dice_scores.append([x[str(class_label[i][j])]["Dice"] for x in results["prediction_result"]["all"]])
        basic_dice_scores.append([x[str(class_label[i][j])]["Dice"] for x in basic_predictions["results"]["all"]])

def setBoxColors(bp):
    setp(bp['boxes'][0], color='blue')
    setp(bp['caps'][0], color='blue')
    setp(bp['caps'][1], color='blue')
    setp(bp['whiskers'][0], color='blue')
    setp(bp['whiskers'][1], color='blue')
    setp(bp['fliers'][0], color='blue')
    setp(bp['fliers'][1], color='blue')
    setp(bp['medians'][0], color='blue')

    setp(bp['boxes'][1], color='red')
    setp(bp['caps'][2], color='red')
    setp(bp['caps'][3], color='red')
    setp(bp['whiskers'][2], color='red')
    setp(bp['whiskers'][3], color='red')
    # setp(bp['fliers'][2], color='red')
    # setp(bp['fliers'][3], color='red')
    setp(bp['medians'][1], color='red')

fig, ax1 = plt.subplots()

bp = ax1.boxplot([dice_scores[0], basic_dice_scores[0]], positions = [1, 2], widths = 0.6)
setBoxColors(bp)

bp = ax1.boxplot([dice_scores[1], basic_dice_scores[1]], positions = [3, 4], widths = 0.6)
setBoxColors(bp)

bp = ax1.boxplot([dice_scores[2], basic_dice_scores[2]], positions = [5, 6], widths = 0.6)
setBoxColors(bp)

bp = ax1.boxplot([dice_scores[3], basic_dice_scores[3]], positions = [7, 8], widths = 0.6)
setBoxColors(bp)

bp = ax1.boxplot([dice_scores[0], basic_dice_scores[0]], positions = [9, 10], widths = 0.6)
setBoxColors(bp)

bp = ax1.boxplot([dice_scores[0], basic_dice_scores[0]], positions = [11, 12], widths = 0.6)
setBoxColors(bp)

# ax1.locator_params(nbins=3)
# ax1.set_xticklabels(['COVID-19', 'BrainTumor', 'Pancreas'])
# ax1.set_xticks([1.5, 4.5, 7.5]) np.arange(1.5, 13.5, 2)
plt.xticks(ticks=np.arange(1.5, 13.5, 2),
           labels=['COVID-19\n(GGO)', 'BrainTumor\n(Edema)', 'BrainTumor\n(Non-enhancing tumor)',
                   'BrainTumor\n(Enhancing tumour)', 'Pancreas\n(Pancreas)', 'Pancreas\n(Cancer)'], rotation=25,fontsize=8)

plt.ylim(0, 1)
# plt.show()
plt.savefig("/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/test_eval/test_eval.png")
