import matplotlib.pyplot as plt
import numpy as np
import pickle
import json
from pylab import plot, show, savefig, xlim, figure, ylim, legend, boxplot, setp, axes
from i3Deep import utils
import pandas as pd
import os
import seaborn as sns


def plot_annotatio_ratio():
    fig = plt.figure(constrained_layout=True, figsize=(12, 7))
    gs = fig.add_gridspec(4, 4)
    for i, task in enumerate(tasks):
        filename = base_path + task + "/refinement_" + set + "/eval_results/processed/ratio/my_method.csv"
        df = pd.read_csv(filename).rename(columns={'Ratio': 'Annotation Ratio'})
        sns.set_theme(style="whitegrid")
        ax = fig.add_subplot(gs[gridspec_indices[i][0], gridspec_indices[i][1]])
        ax.set_title(task_names[i])
        # sns.displot(df, x="Annotation Ratio", kind="kde", ax=ax)
        ax = sns.kdeplot(data=df, x="Annotation Ratio", ax=ax)
        ax.set_xlim(0, 1)
        print("Task: {}, mAR: {}".format(task, np.mean(df.to_numpy())))
    plt.suptitle("Annotation Ratio", fontsize=16)
    plt.savefig(base_path + "Evaluation/Results/AR.png", bbox_inches='tight')


if __name__ == '__main__':
    base_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/"
    tasks = ["Task002_BrainTumour_guided", "Task008_Pancreas_guided", "Task070_guided_all_public_ggo"]
    task_names = ["Brain Tumor", "Pancreas", "COVID-19"]
    set = "test"
    gridspec_indices = [[slice(0, 2), slice(0, 2)], [slice(0, 2), slice(2, 4)], [slice(2, 4), slice(1, 3)]]

    plot_annotatio_ratio()