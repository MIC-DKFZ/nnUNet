import matplotlib.pyplot as plt
import numpy as np
import pickle
import json
from pylab import plot, show, savefig, xlim, figure, ylim, legend, boxplot, setp, axes
from i3Deep import utils
import pandas as pd
import os
import seaborn as sns
import matplotlib


def plot_dice_box_plots():
    sns.set_theme(style="whitegrid")
    style = sns.axes_style()
    fig = plt.figure(constrained_layout=True, figsize=(12*1.5, 2.4*1.5))
    gs = fig.add_gridspec(1, width)
    comparison = {"tta": [], "mcdo": [], "ensemble": []}
    for i, task in enumerate(tasks):
        style["axes.facecolor"] = colors[i]
        sns.set_theme(style=style)
        df_uncertainty_methods = []
        for k, uncertainty_method in enumerate(uncertainty_methods):
            filename = base_path + task + "/refinement_" + set + "/eval_results/processed/dice/my_method_" + uncertainty_method + ".csv"
            if not os.path.isfile(filename):
                continue
            df = pd.read_csv(filename)
            df = df.iloc[:, 1:]
            for j, column in enumerate(df):
                scores = df[column].to_numpy()
                scores = scores[~np.isnan(scores)]
                mean = np.mean(scores)
                std = np.std(scores)
                print("Task: {}, Class: {}, Method: {}, Mean: {}, Std: {}".format(task, task_classes[i][j], uncertainty_method.replace('\n', ' '), round(mean, 3), round(std, 3)))
                comparison[uncertainty_method].append(mean)
            df = df.stack().reset_index()
            df = df.iloc[:, 1:]
            df = df.assign(Predictor=uncertainty_names[k])
            df = df.rename(columns={"level_1": "Class", 0: "Dice"})
            df_uncertainty_methods.append(df)
        df = pd.concat(df_uncertainty_methods)
        for j, class_name in enumerate(task_classes[i]):
            # df["Class"] = df["Class"].replace(j+1, class_name)
            df["Class"].replace({str(j+1): class_name}, inplace=True)
        print(df.head())
        axs = fig.add_subplot(gs[gridspec_pos[i][0][0]:gridspec_pos[i][1][0], gridspec_pos[i][0][1]:gridspec_pos[i][1][1]])
        axs = sns.boxplot(x="Class", y="Dice", hue="Predictor", data=df, ax=axs)
        # axs.tick_params(axis='x', rotation=20)
        axs.set_ylim(0.4, 1)
        axs.set_title("{}".format(task_names[i]), fontsize=16)
    plt.savefig(base_path + "Evaluation/Results/" + set + "/uncertainty_ablation/uncertainty_ablation.png", dpi=150, bbox_inches='tight')
    plt.clf()
    for key in comparison.keys():
        mean = np.mean(comparison[key])
        print("{}: {}".format(key, mean))


if __name__ == '__main__':
    base_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/"
    tasks = ["Task002_BrainTumour_guided", "Task008_Pancreas_guided", "Task070_guided_all_public_ggo"]
    task_names = ["Brain Tumor", "Pancreas", "COVID-19"]
    task_classes = [["Edema", "Non-Enh. Tumor", "Enh. Tumor"], ["Pancreas", "Cancer"], ["GGO"]]
    uncertainty_methods = ["tta", "mcdo", "ensemble"]
    uncertainty_names = ["TTA", "MC Dropout", "Deep Ensembles"]
    set = "val"
    width = 6*3
    print("width: ", width)
    task_widths = [int(round(width*0.5)-1), int(round(width*0.666*0.5)), int(round(width*0.333*0.5))]
    print("task_widths: ", task_widths)
    gridspec_pos = [[[0, 0], [1, task_widths[0]-1]], [[0, task_widths[0]], [1, task_widths[0]+task_widths[1]-1]], [[0, task_widths[0]+task_widths[1]], [1, width]]]
    print("gridspec_pos: ", gridspec_pos)
    colors = ["#FFEEEE", "#EEEEFF", "#EEFFF0"]

    plot_dice_box_plots()