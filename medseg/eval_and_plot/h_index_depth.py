from medseg import utils
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import GeodisTK
import collections
from collections import defaultdict


def comp_h_index_depth(uncertainty_path, thresholds):
    uncertainty_filenames = utils.load_filenames(uncertainty_path)[:3]

    dictionaries = defaultdict(list)
    for i in tqdm(range(len(uncertainty_filenames))):
        uncertainty = utils.load_nifty(uncertainty_filenames[i])[0]

        for threshold in thresholds:
            dictionary = h_index_depth_counts(uncertainty, threshold)
            dictionaries[threshold].append(dictionary)

    h_index_depths = defaultdict(list)
    for threshold in thresholds:
        h_index_depth = h_index_depth_result(dictionaries[threshold])
        h_index_depths[threshold] = h_index_depth

    return h_index_depths


# def h_index_depth1(uncertainty):
#     euclidean_distance_map = np.ones_like(uncertainty) - uncertainty
#     euclidean_distance_map = euclidean_distance_map.astype(np.float32)
#     for i in range(len(euclidean_distance_map)):
#         euclidean_distance_map[i] = GeodisTK.geodesic3d_raster_scan(np.zeros_like(uncertainty[i]).astype(np.float32), euclidean_distance_map[i].astype(np.uint8), np.asarray([1.0, 1.0, 1.0], dtype=np.float32), 0.0, 4)
#     unique, count = np.unique(euclidean_distance_map, return_counts=True)
#     unique = np.flip(unique)
#     count = np.flip(count)
#     h_index = 0
#     sum = 0
#     for i in range(len(unique)):
#         sum += count[i]
#         if unique[i] <= sum:
#             h_index = unique[i]
#             break
#     return h_index


def h_index_depth_counts(uncertainty, threshold):
    euclidean_distance_map = (uncertainty > threshold).astype(np.int)
    euclidean_distance_map = np.ones_like(euclidean_distance_map) - euclidean_distance_map
    euclidean_distance_map = euclidean_distance_map.astype(np.float32)
    euclidean_distance_map = GeodisTK.geodesic3d_raster_scan(np.zeros_like(uncertainty).astype(np.float32), euclidean_distance_map.astype(np.uint8), np.asarray([1.0, 1.0, 1.0], dtype=np.float32), 0.0, 4)
    uniques, counts = np.unique(euclidean_distance_map, return_counts=True)
    uniques_rounded = np.rint(uniques)
    uniques = defaultdict(int)
    for i in range(len(uniques_rounded)):
        uniques[uniques_rounded[i]] += counts[i]
    dictionary = dict(zip(uniques, counts))
    dictionary = collections.Counter(dictionary)
    return dictionary


def h_index_depth_result(dictionaries):
    dictionary = collections.Counter()
    for d in dictionaries:
        dictionary += d
    print("dictionary: ", dictionary)
    counts = list(dictionary.values())
    counts = np.flip(counts)
    h_index_depth = 0

    for count in counts:
        if h_index_depth <= count:
            h_index_depth += 1
        else:
            break
    return h_index_depth


if __name__ == '__main__':
    base_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/"
    tasks = ["Task002_BrainTumour_guided", "Task008_Pancreas_guided", "Task070_guided_all_public_ggo"]
    task_names = ["Brain Tumor", "Pancreas", "COVID-19"]
    set = "val"
    uqs = ["ensemble", "mcdo", "tta"]
    uqs_names = ["Ensemble", "MC Dropout", "TTA"]
    um = "predictive_entropy"
    um_name = "Entropy"
    thresholds = np.arange(0.0, 1.0, 0.03)
    gridspec_indices = [[slice(0, 2), slice(0, 2)], [slice(0, 2), slice(2, 4)], [slice(2, 4), slice(1, 3)]]
    n_bins = 20
    load = False

    fig = plt.figure(constrained_layout=True, figsize=(12, 7))
    gs = fig.add_gridspec(4, 4)
    for k, task in enumerate(tasks):
        print("Task: ", task)
        ax = fig.add_subplot(gs[gridspec_indices[k][0], gridspec_indices[k][1]])
        for i, uq in enumerate(uqs):
            if load:
                with open(base_path + task_names[k] + "_" + uq + ".pkl", 'rb') as handle:
                    h_index_depth = pickle.load(handle)
            else:
                uncertainty_path = base_path + task + "/refinement_" + set + "/uncertainties/" + uq + "/" + um + "/"
                h_index_depth = comp_h_index_depth(uncertainty_path, thresholds)
                with open(base_path + task_names[k] + "_" + uq + ".pkl", 'wb') as handle:
                    pickle.dump(h_index_depth, handle, protocol=pickle.HIGHEST_PROTOCOL)
            x = list(thresholds)
            y = list(h_index_depth.values())
            ax.plot(x, y, label=uqs_names[i])
            ax.set_title(task_names[k])
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Thresholds")
        ax.set_ylabel("h-index-depth")
    plt.suptitle("h-index-depth", fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.tight_layout()
    plt.savefig(base_path + "h_index_depth.png", dpi=150)  # bbox_inches='tight'
