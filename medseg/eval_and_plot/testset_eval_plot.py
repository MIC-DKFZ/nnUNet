import matplotlib.pyplot as plt
import numpy as np
import pickle
import json
from pylab import plot, show, savefig, xlim, figure, ylim, legend, boxplot, setp, axes
from medseg import utils
import pandas as pd
import os
import seaborn as sns


def plot_dice_box_plots():
    for i, task in enumerate(tasks):
        for j, class_name in enumerate(task_classes[i]):
            df_methods = {}
            for k, method in enumerate(methods):
                filename = base_path + task + "/refinement_" + set + "/eval_results/processed/dice/" + method + ".csv"
                if not os.path.isfile(filename):
                    continue
                df = pd.read_csv(filename)
                df = df.iloc[:, j+1].to_numpy()
                df_methods[method_names[k]] = df
            df = pd.DataFrame.from_dict(df_methods, orient='index').T
            # df['income'].fillna((df['income'].mean()), inplace=True)
            sns.set_theme(style="whitegrid")
            ax = sns.boxplot(data=df)
            # ax.xticks(x, labels, rotation='vertical')
            ax.tick_params(axis='x', rotation=20)
            ax.set_ylim(0, 1)
            ax.set_title("{} ({})".format(task_names[i], class_name))
            plt.show()


        # dice_scores, basic_dice_scores = [], []
        #
        # processed_path = base_path + tasks[i] + "/refinement_" + set + "/eval_results/processed/"
        # result_filenames = utils.load_filenames(processed_path, extensions=None)
        #
        # for result_filename in result_filenames:
        #     method = os.path.basename(result_filename)
        #     if "basic_predictions" in method:
        #         method = "Automatic"
        #     elif "scores" in method:
        #         method = method[3:-len("scores")-5]
        #     else:
        #         continue
        #     df = pd.read_csv(result_filename)
        #     df = df.iloc[:, 1:]
        #     print("Task: {}, method: {}, df: {}".format(tasks[i], method, df))
        #     for class_index, column in enumerate(df):
        #         dice_scores = df[column].to_numpy()
        #
        #     fig, ax1 = plt.subplots()
        #
        #     bp = ax1.boxplot([dice_scores[0], basic_dice_scores[0]], positions = [1, 2], widths = 0.6)
        #     setBoxColors(bp)
        #
        #     bp = ax1.boxplot([dice_scores[1], basic_dice_scores[1]], positions = [3, 4], widths = 0.6)
        #     setBoxColors(bp)
        #
        #     bp = ax1.boxplot([dice_scores[2], basic_dice_scores[2]], positions = [5, 6], widths = 0.6)
        #     setBoxColors(bp)
        #
        #     bp = ax1.boxplot([dice_scores[3], basic_dice_scores[3]], positions = [7, 8], widths = 0.6)
        #     setBoxColors(bp)
        #
        #     bp = ax1.boxplot([dice_scores[0], basic_dice_scores[0]], positions = [9, 10], widths = 0.6)
        #     setBoxColors(bp)
        #
        #     bp = ax1.boxplot([dice_scores[0], basic_dice_scores[0]], positions = [11, 12], widths = 0.6)
        #     setBoxColors(bp)
        #
        #     # ax1.locator_params(nbins=3)
        #     # ax1.set_xticklabels(['COVID-19', 'BrainTumor', 'Pancreas'])
        #     # ax1.set_xticks([1.5, 4.5, 7.5]) np.arange(1.5, 13.5, 2)
        #     plt.xticks(ticks=np.arange(1.5, 13.5, 2),
        #                labels=['COVID-19\n(GGO)', 'BrainTumor\n(Edema)', 'BrainTumor\n(Non-enhancing tumor)',
        #                        'BrainTumor\n(Enhancing tumour)', 'Pancreas\n(Pancreas)', 'Pancreas\n(Cancer)'], rotation=25,fontsize=8)
        #
        #     plt.ylim(0, 1)
        #     # plt.show()
        #     plt.savefig("/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/test_eval/test_eval.png")


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


if __name__ == '__main__':
    base_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/"
    tasks = ["Task070_guided_all_public_ggo", "Task002_BrainTumour_guided", "Task008_Pancreas_guided"]
    task_names = ["COVID-19", "Brain Tumor", "Pancreas"]
    task_classes = [["GGO"], ["Edema", "Non-enhancing Tumor", "Enhancing Tumour"], ["Pancreas", "Cancer"]]
    methods = ["automatic", "my_method", "random1", "random2", "P_Net", "watershed", "random_walker", "GraphCut1"]
    method_names = ["Presegmentation", "Ours", "Random Slices", "Random Slices \n (Min. Dist.)", "P-Net", "Watershed", "Random Walker", "Graphcut"]
    set = "test"

    plot_dice_box_plots()

#     filename = base_path + task[i] + "/refinement_" + set + "/GridSearchResults/" + basename[i]
#     basic_predictions_filename = base_path + task[i] + "/refinement_" + set + "/basic_predictions_summary.json"
#
#     with open(filename, 'rb') as handle:
#         results = pickle.load(handle)
#
#     with open(basic_predictions_filename) as json_file:
#         basic_predictions = json.load(json_file)
#
#     results = results[list(results.keys())[0]]
#     results = results[list(results.keys())[0]]
#
#     recommended_slices = [x["recommended_slices"] for x in results["recommended_result"]]
#     gt_infected_slices = [x["gt_infected_slices"] for x in results["recommended_result"]]
#     # annotation_ratio = np.sum(recommended_slices) / np.sum(gt_infected_slices)
#     annotation_ratio = np.asarray(recommended_slices) / np.asarray(gt_infected_slices)
#     # print(annotation_ratio)
#     annotation_ratio = np.mean(annotation_ratio)
#     for j in range(len(class_label[i])):
#         dice_scores.append([x[str(class_label[i][j])]["Dice"] for x in results["prediction_result"]["all"]])
#         basic_dice_scores.append([x[str(class_label[i][j])]["Dice"] for x in basic_predictions["results"]["all"]])
#
# def setBoxColors(bp):
#     setp(bp['boxes'][0], color='blue')
#     setp(bp['caps'][0], color='blue')
#     setp(bp['caps'][1], color='blue')
#     setp(bp['whiskers'][0], color='blue')
#     setp(bp['whiskers'][1], color='blue')
#     setp(bp['fliers'][0], color='blue')
#     setp(bp['fliers'][1], color='blue')
#     setp(bp['medians'][0], color='blue')
#
#     setp(bp['boxes'][1], color='red')
#     setp(bp['caps'][2], color='red')
#     setp(bp['caps'][3], color='red')
#     setp(bp['whiskers'][2], color='red')
#     setp(bp['whiskers'][3], color='red')
#     # setp(bp['fliers'][2], color='red')
#     # setp(bp['fliers'][3], color='red')
#     setp(bp['medians'][1], color='red')
#
# fig, ax1 = plt.subplots()
#
# bp = ax1.boxplot([dice_scores[0], basic_dice_scores[0]], positions = [1, 2], widths = 0.6)
# setBoxColors(bp)
#
# bp = ax1.boxplot([dice_scores[1], basic_dice_scores[1]], positions = [3, 4], widths = 0.6)
# setBoxColors(bp)
#
# bp = ax1.boxplot([dice_scores[2], basic_dice_scores[2]], positions = [5, 6], widths = 0.6)
# setBoxColors(bp)
#
# bp = ax1.boxplot([dice_scores[3], basic_dice_scores[3]], positions = [7, 8], widths = 0.6)
# setBoxColors(bp)
#
# bp = ax1.boxplot([dice_scores[0], basic_dice_scores[0]], positions = [9, 10], widths = 0.6)
# setBoxColors(bp)
#
# bp = ax1.boxplot([dice_scores[0], basic_dice_scores[0]], positions = [11, 12], widths = 0.6)
# setBoxColors(bp)
#
# # ax1.locator_params(nbins=3)
# # ax1.set_xticklabels(['COVID-19', 'BrainTumor', 'Pancreas'])
# # ax1.set_xticks([1.5, 4.5, 7.5]) np.arange(1.5, 13.5, 2)
# plt.xticks(ticks=np.arange(1.5, 13.5, 2),
#            labels=['COVID-19\n(GGO)', 'BrainTumor\n(Edema)', 'BrainTumor\n(Non-enhancing tumor)',
#                    'BrainTumor\n(Enhancing tumour)', 'Pancreas\n(Pancreas)', 'Pancreas\n(Cancer)'], rotation=25,fontsize=8)
#
# plt.ylim(0, 1)
# # plt.show()
# plt.savefig("/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/test_eval/test_eval.png")
