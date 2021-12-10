import matplotlib.pyplot as plt
import numpy as np
from i3Deep import utils
import os

def plot():
    for i, task in enumerate(tasks):
        filenames = utils.load_filenames(base_path + task + "/qualitative_results/cropped/")
        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(1, 6)
        # gs.update(wspace=0.0025, hspace=0.0025)  # set the spacing between axes.
        for j, filename in enumerate(filenames):
            image = plt.imread(filename)
            method = os.path.basename(filename)[:-4]
            name = methods[method]
            ax = fig.add_subplot(gs[:, gridspec_indices[method][1]])
            ax.imshow(image)
            ax.set_title(name, fontsize=8)
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            ax.axis('off')
        # plt.tight_layout()
        # plt.margins(0, 0)
        # plt.suptitle(task_names[i], fontsize=8)
        # plt.show()
        plt.savefig(base_path + "qualitative_results/" + task_names[i] + ".png", bbox_inches='tight', dpi=300)
        plt.clf()


if __name__ == '__main__':
    base_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Evaluation/Results/"
    tasks = ["Task002_BrainTumour_guided", "Task008_Pancreas_guided", "Task070_guided_all_public_ggo"]
    task_names = ["Brain Tumor", "Pancreas", "COVID-19"]
    methods = {"automatic": "Preseg.", "gt": "Ground Truth", "my_method": "nnU-Net", "P_Net": "P-Net", "random_walker": "Random Walker", "watershed": "Watershed", "graphcut": "GraphCut"}
    # gridspec_indices = [[slice(0, 1), slice(0, 1)], [slice(0, 1), slice(1, 2)], [slice(0, 1), slice(2, 3)], [slice(1, 2), slice(0, 1)], [slice(1, 2), slice(1, 2)], [slice(1, 2), slice(2, 3)]]
    gridspec_indices = {"gt": [slice(0, 1), slice(0, 1)],
                        "automatic": [slice(0, 1), slice(1, 2)],
                        "my_method": [slice(0, 1), slice(2, 3)],
                        "P_Net": [slice(0, 1), slice(3, 4)],
                        "watershed": [slice(0, 1), slice(4, 5)],
                        "random_walker": [slice(0, 1), slice(5, 6)],
                        "graphcut": [slice(0, 1), slice(5, 6)]}
    plot()
