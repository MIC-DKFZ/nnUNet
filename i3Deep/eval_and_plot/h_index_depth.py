from i3Deep import utils
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import GeodisTK
import collections
from collections import defaultdict
import multiprocessing as mp
from functools import partial


def comp_h_index_depth(uncertainty_path, thresholds, parallel=True):
    uncertainty_filenames = utils.load_filenames(uncertainty_path)
    dictionaries = defaultdict(list)

    if not parallel:
        for i in tqdm(range(len(uncertainty_filenames))):
            uncertainty = utils.load_nifty(uncertainty_filenames[i])[0]

            for threshold in thresholds:
                dictionary = h_index_depth_counts(uncertainty, threshold)
                dictionaries[threshold].append(dictionary)
    else:
        pool = mp.Pool(processes=10)
        results = pool.map(partial(comp_h_index_depth_single, uncertainty_filenames=uncertainty_filenames, thresholds=thresholds), range(len(uncertainty_filenames)))

        for result in results:
            for threshold in thresholds:
                dictionaries[threshold].extend(result[threshold])

        pool.close()
        pool.join()

    h_index_depths = defaultdict(list)
    for threshold in thresholds:
        h_index_depth = h_index_depth_result(dictionaries[threshold])
        h_index_depths[threshold] = h_index_depth

    return h_index_depths


def comp_h_index_depth_single(i, uncertainty_filenames, thresholds):
    uncertainty = utils.load_nifty(uncertainty_filenames[i])[0]

    dictionaries = defaultdict(list)
    for threshold in thresholds:
        dictionary = h_index_depth_counts(uncertainty, threshold)
        dictionaries[threshold].append(dictionary)
    return dictionaries


def h_index_depth_counts(uncertainty, threshold):
    euclidean_distance_map = (uncertainty > threshold).astype(np.int)
    euclidean_distance_map = np.ones_like(euclidean_distance_map) - euclidean_distance_map
    euclidean_distance_map = euclidean_distance_map.astype(np.float32)
    euclidean_distance_map = GeodisTK.geodesic3d_raster_scan(np.zeros_like(uncertainty).astype(np.float32), euclidean_distance_map.astype(np.uint8), np.asarray([1.0, 1.0, 1.0], dtype=np.float32), 0.0, 4)
    euclidean_distance_map = np.rint(euclidean_distance_map)
    uniques, counts = np.unique(euclidean_distance_map, return_counts=True)
    dictionary = dict(zip(uniques, counts))
    dictionary = collections.Counter(dictionary)
    return dictionary


def h_index_depth_result(dictionaries):
    dictionary = collections.Counter()
    for d in dictionaries:
        dictionary += d
    uniques = np.asarray(list(dictionary.keys()))
    counts = np.asarray(list(dictionary.values()))
    indices = np.argsort(uniques)
    counts = counts[indices]
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
    thresholds = np.arange(0.0, 1.0, 0.1)  # np.arange(0.0, 1.0, 0.03)
    gridspec_indices = [[slice(0, 2), slice(0, 2)], [slice(0, 2), slice(2, 4)], [slice(2, 4), slice(1, 3)]]
    load = False

    fig = plt.figure(constrained_layout=True, figsize=(12, 7))
    gs = fig.add_gridspec(4, 4)
    for k, task in enumerate(tasks):
        print("Task: ", task)
        ax = fig.add_subplot(gs[gridspec_indices[k][0], gridspec_indices[k][1]])
        for i, uq in enumerate(uqs):
            if load:
                with open(base_path + "Evaluation/h-index-depth/" + task_names[k] + "_" + uq + ".pkl", 'rb') as handle:
                    h_index_depth = pickle.load(handle)
            else:
                uncertainty_path = base_path + task + "/refinement_" + set + "/uncertainties/" + uq + "/" + um + "/"
                h_index_depth = comp_h_index_depth(uncertainty_path, thresholds)
                with open(base_path + "Evaluation/h-index-depth/" + task_names[k] + "_" + uq + ".pkl", 'wb') as handle:
                    pickle.dump(h_index_depth, handle, protocol=pickle.HIGHEST_PROTOCOL)
            x = list(thresholds)
            y = list(h_index_depth.values())
            ax.plot(x, y, label=uqs_names[i])
            ax.set_title(task_names[k])
        # ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Thresholds")
        ax.set_ylabel("h-index-depth")
    plt.suptitle("h-index-depth", fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.tight_layout()
    plt.savefig(base_path + "Evaluation/h-index-depth/h_index_depth.png", dpi=150)  # bbox_inches='tight'
