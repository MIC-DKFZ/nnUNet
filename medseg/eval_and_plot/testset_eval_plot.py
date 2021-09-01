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
        fig = plt.figure(constrained_layout=True, figsize=(figsizes[i][0], figsizes[i][1]))
        gs = fig.add_gridspec(4, 4)
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
            sns.set_theme(style="whitegrid")
            ax = fig.add_subplot(gs[gridspec_indices[i][j][0], gridspec_indices[i][j][1]])
            ax = sns.boxplot(data=df, ax=ax)
            ax.tick_params(axis='x', rotation=20)
            ax.set_ylim(0, 1)
            ax.set_title("{}".format(class_name))
        plt.suptitle("{}".format(task_names[i]), fontsize=16)
        # plt.show()
        plt.savefig(base_path + "Evaluation/Results/" + task_names[i] + ".png", bbox_inches='tight')
        plt.clf()


if __name__ == '__main__':
    base_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/"
    tasks = ["Task070_guided_all_public_ggo", "Task002_BrainTumour_guided", "Task008_Pancreas_guided"]
    task_names = ["COVID-19", "Brain Tumor", "Pancreas"]
    task_classes = [["GGO"], ["Edema", "Non-enhancing Tumor", "Enhancing Tumour"], ["Pancreas", "Cancer"]]
    methods = ["automatic", "my_method", "P_Net", "watershed", "random_walker", "GraphCut1", "random1", "random2"]
    method_names = ["Preseg.", "Ours", "P-Net", "Watershed", "Random Walker", "Graphcut", "Random Slices", "Random Slices \n (Min. Dist.)"]
    set = "test"
    figsizes = [[8, 5], [12, 8], [12, 5]]
    gridspec_indices = [[[slice(0, 4), slice(0, 4)]],
                        [[slice(0, 2), slice(0, 2)], [slice(0, 2), slice(2, 4)], [slice(2, 4), slice(1, 3)]],
                        [[slice(0, 4), slice(0, 2)], [slice(0, 4), slice(2, 4)]]]

    plot_dice_box_plots()