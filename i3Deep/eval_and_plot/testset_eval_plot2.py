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
    fig = plt.figure(constrained_layout=True, figsize=(12*1.3, 5*1.35))
    gs = fig.add_gridspec(2, 3)
    for i, task in enumerate(tasks):
        # fig, axs = plt.subplots(1, n_figures[i], squeeze=False, figsize=(figsizes[i][0], figsizes[i][1]))
        # axs = fig.add_subplot(gs[gridspec_pos[i][0], gridspec_pos[i][1]])
        # fig.tight_layout()
        style["axes.facecolor"] = colors[i]
        sns.set_theme(style=style)
        for j, class_name in enumerate(task_classes[i]):
            df_methods = {}
            for k, method in enumerate(methods[i]):
                filename = base_path + task + "/refinement_" + set + "/eval_results/processed/dice/" + method + ".csv"
                if not os.path.isfile(filename):
                    continue
                df = pd.read_csv(filename)
                df = df.iloc[:, j+1].to_numpy()
                df_methods[method_names[i][k]] = df
            df = pd.DataFrame.from_dict(df_methods, orient='index').T
            for method in df:
                scores = df[method].to_numpy()
                scores = scores[~np.isnan(scores)]
                mean = np.mean(scores)
                std = np.std(scores)
                print("Task: {}, Class: {}, Method: {}, Mean: {}, Std: {}".format(task, class_name, method.replace('\n', ' '), round(mean, 3), round(std, 3)))
            axs = fig.add_subplot(gs[gridspec_pos[i][j][0], gridspec_pos[i][j][1]])
            axs = sns.boxplot(data=df, ax=axs)
            axs.tick_params(axis='x', rotation=20)
            axs.set_ylim(0, 1)
            axs.set_title("{}".format(class_name), fontsize=16)
        # fig.set_aspect(3)
        #plt.suptitle("{}".format(task_names[i]))# , fontsize=16)
    plt.savefig(base_path + "Evaluation/Results/results.svg", dpi=150, bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    base_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/"
    tasks = ["Task002_BrainTumour_guided", "Task008_Pancreas_guided", "Task070_guided_all_public_ggo"]
    task_names = ["Brain Tumor", "Pancreas", "COVID-19"]
    task_classes = [["Brain Tumor - Edema", "Brain Tumor - Non-enhancing Tumor", "Brain Tumor - Enhancing Tumor"], ["Pancreas - Pancreas", "Pancreas - Cancer"], ["COVID-19 - GGO"]]
    methods = [
        ["automatic", "my_method", "P_Net", "watershed", "random_walker"],
        ["automatic", "my_method", "P_Net", "watershed", "random_walker"],
        ["automatic", "my_method", "P_Net", "watershed", "GraphCut1"]
    ]  # ["automatic", "my_method", "P_Net", "watershed", "random_walker", "GraphCut1", "random1", "random2", "random3"]
    method_names = [
        ["Preseg.", "nnU-Net", "P-Net", "Watershed", "Random\nWalker"],
        ["Preseg.", "nnU-Net", "P-Net", "Watershed", "Random\nWalker"],
        ["Preseg.", "nnU-Net", "P-Net", "Watershed", "Graph-Cut\n"]
    ]  # ["Preseg.", "nnU-Net", "P-Net", "Watershed", "Random Walker", "Graph-Cut", "RS1", "RS2", "RS3"]
    set = "test"
    # figsizes = [[8, 5], [29, 10], [12, 5]]
    # gridspec_indices = [[[slice(0, 4), slice(0, 4)]],
    #                     [[slice(0, 2), slice(0, 1)], [slice(0, 2), slice(1, 2)], [slice(0, 2), slice(2, 3)]],
    #                     [[slice(0, 4), slice(0, 2)], [slice(0, 4), slice(2, 4)]]]
    n_figures = [3, 2, 1]
    # aspect_ratios = [(1/3)-0.15, 0.185, 1-0.15]
    cm = 1 / 2.54
    width = 50
    height = 10
    figsizes = [[width*cm, height*cm], [(width*(2/3))*cm, height*cm], [(width*(1/3))*cm, height*cm]]
    gridspec_pos = [
        [[0, 0], [0, 1], [0, 2]],
        [[1, 0], [1, 1]],
        [[1, 2]]
    ]
    colors = ["#FFEEEE", "#EEEEFF", "#EEFFF0"]

    plot_dice_box_plots()