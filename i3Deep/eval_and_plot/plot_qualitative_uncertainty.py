import matplotlib.pyplot as plt
import numpy as np
from i3Deep import utils
import os

def plot():
    for i, task in enumerate(tasks):
        filenames = [base_path + task + "/" + key + ".png" for key in methods]
        # filenames = utils.load_filenames(base_path + task + "/")
        # fig = plt.figure(constrained_layout=False)
        # gs = fig.add_gridspec(1, 3)
        fig, axs = plt.subplots(1, 3)
        # gs.update(wspace=0.0025, hspace=0.0025)  # set the spacing between axes.
        for j, filename in enumerate(filenames):
            image = plt.imread(filename)
            method = os.path.basename(filename)[:-4]
            name = methods[method]
            # ax = fig.add_subplot(gs[gridspec_indices[method][0], gridspec_indices[method][1]])
            axs[j].imshow(image)
            axs[j].set_title(name)
            axs[j].axes.xaxis.set_visible(False)
            axs[j].axes.yaxis.set_visible(False)
            axs[j].axis('off')
        plt.tight_layout()
        # plt.margins(0, 0)
        # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        #plt.title(task_names[i], fontsize=16)
        # plt.show()
        plt.savefig(base_path + task_names[i] + ".png", bbox_inches='tight')
        plt.clf()


if __name__ == '__main__':
    base_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Evaluation/Uncertainty Evaluation/qualitative_results/"
    tasks = ["Task002_BrainTumour_guided"]
    task_names = ["Brain Tumor"]
    methods = {"tta": "TTA", "mcdo": "MC Dropout", "ensemble": "Deep Ensemble"}
    # gridspec_indices = [[slice(0, 1), slice(0, 1)], [slice(0, 1), slice(1, 2)], [slice(0, 1), slice(2, 3)], [slice(1, 2), slice(0, 1)], [slice(1, 2), slice(1, 2)], [slice(1, 2), slice(2, 3)]]
    gridspec_indices = {"tta": [slice(0, 1), slice(0, 1)],
                        "mcdo": [slice(0, 1), slice(1, 2)],
                        "ensemble": [slice(0, 1), slice(2, 3)]}
    plot()
