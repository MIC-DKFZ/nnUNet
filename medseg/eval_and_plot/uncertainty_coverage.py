import numpy as np
from collections import defaultdict
from medseg import utils
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import pickle
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt
import argparse
from matplotlib.lines import Line2D


# def comp_uncertainty_coverage(uncertainty_dir, prediction_dir, gt_dir, thresholds, parallel=True, target_shape=(256, 256, 50)):
#     uncertainty_dir = utils.fix_path(uncertainty_dir)
#     prediction_dir = utils.fix_path(prediction_dir)
#     gt_dir = utils.fix_path(gt_dir)
#     uncertainty_filenames = utils.load_filenames(uncertainty_dir)
#     prediction_filenames = utils.load_filenames(prediction_dir)
#     gt_filenames = utils.load_filenames(gt_dir)
#     U_threshold = defaultdict(lambda: defaultdict(int))
#     if parallel:
#         pool = mp.Pool(processes=8)
#
#     for i in tqdm(range(len(uncertainty_filenames)), desc="Case"):
#         uncertainty = utils.load_nifty(uncertainty_filenames[i])[0]
#         prediction = utils.load_nifty(prediction_filenames[i])[0]
#         ground_truth = utils.load_nifty(gt_filenames[i])[0]
#         uncertainty = utils.interpolate(uncertainty, target_shape, mask=False)
#         prediction = utils.interpolate(prediction, target_shape, mask=True)
#         ground_truth = utils.interpolate(ground_truth, target_shape, mask=True)
#         prediction = np.rint(prediction).flatten().astype(int)
#         ground_truth = np.rint(ground_truth).flatten().astype(int)
#         prediction, ground_truth = guard_input(prediction, ground_truth)
#         missclassification = comp_missclassification(prediction, ground_truth)
#         if not parallel:
#             for threshold in tqdm(thresholds, leave=False, desc="Threshold"):
#                 U_t, U_f = comp_uncertainty_true_false(threshold, uncertainty, missclassification)
#                 U_threshold[threshold]["U_t"] += U_t
#                 U_threshold[threshold]["U_f"] += U_f
#                 # print("Threshold: {}, U_t: {}, U_f: {}".format(threshold, U_t, U_f))
#         else:
#             results = pool.map(partial(comp_uncertainty_true_false, _uncertainty=uncertainty, missclassification=missclassification), thresholds)
#             results = np.asarray(results)
#             U_t_threshold = results[:, 0]
#             U_f_threshold = results[:, 1]
#             for i, threshold in enumerate(thresholds):
#                 U_threshold[threshold]["U_t"] += U_t_threshold[i]
#                 U_threshold[threshold]["U_f"] += U_f_threshold[i]
#
#     if parallel:
#         pool.close()
#         pool.join()
#
#     U_t = 0
#     UC_threshold = []
#     for threshold in thresholds:
#         U_t += U_threshold[threshold]["U_t"]
#     for threshold in thresholds:
#         UC_threshold.append((U_threshold[threshold]["U_t"]**2) / (U_threshold[threshold]["U_f"] * U_t))
#     UC = np.sum(UC_threshold)
#     return {"UC": UC, "UC_threshold": UC_threshold, "Thresholds": thresholds}


def comp_uncertainty_coverage(uncertainty_dir, prediction_dir, gt_dir, thresholds, parallel=True, resize=True, target_shape=(256, 256, 50)):
    uncertainty_dir = utils.fix_path(uncertainty_dir)
    prediction_dir = utils.fix_path(prediction_dir)
    gt_dir = utils.fix_path(gt_dir)
    uncertainty_filenames = utils.load_filenames(uncertainty_dir)
    prediction_filenames = utils.load_filenames(prediction_dir)
    gt_filenames = utils.load_filenames(gt_dir)
    U_threshold = defaultdict(lambda: defaultdict(int))
    if parallel:
        pool = mp.Pool(processes=10)

    if not parallel:
        results = []
        for i in tqdm(range(len(uncertainty_filenames)), desc="Case"):
            U_threshold_single = comp_uncertainty_coverage_single(i, uncertainty_filenames, prediction_filenames, gt_filenames, thresholds, resize, target_shape)
            results.append(U_threshold_single)
    else:
        results = pool.map(partial(comp_uncertainty_coverage_single, uncertainty_filenames=uncertainty_filenames, prediction_filenames=prediction_filenames,
                                   gt_filenames=gt_filenames, thresholds=thresholds, resize=resize, target_shape=target_shape), range(len(uncertainty_filenames)))
    results = np.asarray(results)
    U_t_threshold = results[:, 0]
    U_f_threshold = results[:, 1]
    M = results[:, 2, 0]
    M = np.sum(M)
    for i in range(len(uncertainty_filenames)):
        for j, threshold in enumerate(thresholds):
            U_threshold[threshold]["U_t"] += U_t_threshold[i][j]
            U_threshold[threshold]["U_f"] += U_f_threshold[i][j]

    if parallel:
        pool.close()
        pool.join()

    U_t, U_f = 0, 0
    UC_threshold = []
    for threshold in thresholds:
        U_t += U_threshold[threshold]["U_t"]
        U_f += U_threshold[threshold]["U_f"]
    for threshold in thresholds:
        # UC_threshold.append((U_threshold[threshold]["U_t"]**2) / (U_threshold[threshold]["U_f"] * U_t))
        UC_threshold.append((U_threshold[threshold]["U_t"] / U_threshold[threshold]["U_f"]) / (U_threshold[threshold]["U_t"] / M))
    UC = np.sum(UC_threshold)
    return {"UC": UC, "UC_threshold": UC_threshold, "Thresholds": thresholds}


def comp_uncertainty_coverage_single(i, uncertainty_filenames, prediction_filenames, gt_filenames, thresholds, resize, target_shape):
    #print("starting")
    uncertainty = utils.load_nifty(uncertainty_filenames[i])[0]
    prediction = utils.load_nifty(prediction_filenames[i])[0]
    ground_truth = utils.load_nifty(gt_filenames[i])[0]
    if resize:
        uncertainty = utils.interpolate(uncertainty, target_shape, mask=False)
        prediction = utils.interpolate(prediction, target_shape, mask=True)
        ground_truth = utils.interpolate(ground_truth, target_shape, mask=True)
    prediction = np.rint(prediction).flatten().astype(int)
    ground_truth = np.rint(ground_truth).flatten().astype(int)
    prediction, ground_truth = guard_input(prediction, ground_truth)
    missclassification = comp_missclassification(prediction, ground_truth)
    M = np.sum(missclassification)
    # U_threshold = defaultdict(lambda: defaultdict(int))
    U_t_threshold, U_f_threshold = [], []
    for threshold in thresholds:
        U_t, U_f = comp_uncertainty_true_false(threshold, uncertainty, missclassification)
        # U_threshold[threshold]["U_t"] += U_t
        # U_threshold[threshold]["U_f"] += U_f
        U_t_threshold.append(U_t)
        U_f_threshold.append(U_f)
    #print("finished")
    return U_t_threshold, U_f_threshold, [M]*len(thresholds)



def comp_uncertainty_true_false(threshold, _uncertainty, missclassification):
    uncertainty = (_uncertainty > threshold).flatten().astype(int)
    # prediction = np.rint(_prediction).flatten().astype(int)
    # ground_truth = np.rint(_ground_truth).flatten().astype(int)
    # uncertainty, ground_truth = guard_input(uncertainty, ground_truth)
    uncertainty, missclassification = guard_input(uncertainty, missclassification)
    # missclassification = comp_missclassification(prediction, ground_truth)
    _, U_f, _, U_t = confusion_matrix(missclassification, uncertainty).ravel()
    U_t, U_f = float(U_t), float(U_f)
    return U_t, U_f


def comp_missclassification(prediction, ground_truth):
    missclassification = np.zeros_like(prediction)
    missclassification[prediction != ground_truth] = 1
    return missclassification


def guard_input(uncertainty, ground_truth):
    """The metrices F1, MCC and AP all have rare edge cases that result in a NaN output.
    For example in the case that all ground truth labels and binarized uncertainties are postive -> there are no true negatives and no false negatives -> sqrt of zero for MCC score, similar for the others.
    This methods changes the value of 4 pixels in the ground truth and the binarized uncertainty to ensure that there is always at least one TP, TN, FP and FN.
    The influence of this modification on the metric results is neglectable."""
    uncertainty[0] = 1
    uncertainty[1] = 0
    uncertainty[2] = 1
    uncertainty[3] = 0
    ground_truth[0] = 1
    ground_truth[1] = 0
    ground_truth[2] = 0
    ground_truth[3] = 1
    return uncertainty, ground_truth


def plot_results_combined(load_dir):

    def load(uncertainty_quantification, uncertainty_measure):
        with open(load_dir + 'UC_{}_{}.pkl'.format(uncertainty_quantification, uncertainty_measure), 'rb') as handle:
            results = pickle.load(handle)
            return {"UC_threshold": results["UC_threshold"], "Thresholds": results["Thresholds"]}

    fig, ax = plt.subplots()
    results = load("ensemble", "bhattacharyya_coefficient")
    plt.plot(results["Thresholds"], results["UC_threshold"], linestyle="-", color="r", label="UC")
    legend_ensemble = Line2D([0, 1], [0, 1], linestyle='-', color="r")
    results = load("mcdo", "bhattacharyya_coefficient")
    plt.plot(results["Thresholds"], results["UC_threshold"], linestyle="-", color="g", label="UC")
    legend_mcdo = Line2D([0, 1], [0, 1], linestyle='-', color="g")
    results = load("tta", "bhattacharyya_coefficient")
    plt.plot(results["Thresholds"], results["UC_threshold"], linestyle="-", color="b", label="UC")
    legend_tta = Line2D([0, 1], [0, 1], linestyle='-', color="b")
    ax.legend([legend_ensemble, legend_mcdo, legend_tta], ["Ensemble", "MC-Dropout", "TTA"], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylim(0, 0.1)
    plt.title("Uncertainty coverage (Bhattacharyya coefficient)")
    plt.savefig(load_dir + "UC_BC.png", bbox_inches='tight')
    plt.clf()

    fig, ax = plt.subplots()
    results = load("ensemble", "predictive_entropy")
    plt.plot(results["Thresholds"], results["UC_threshold"], linestyle="-", color="r", label="UC")
    legend_ensemble = Line2D([0, 1], [0, 1], linestyle='-', color="r")
    results = load("mcdo", "predictive_entropy")
    plt.plot(results["Thresholds"], results["UC_threshold"], linestyle="-", color="g", label="UC")
    legend_mcdo = Line2D([0, 1], [0, 1], linestyle='-', color="g")
    results = load("tta", "predictive_entropy")
    plt.plot(results["Thresholds"], results["UC_threshold"], linestyle="-", color="b", label="UC")
    legend_tta = Line2D([0, 1], [0, 1], linestyle='-', color="b")
    ax.legend([legend_ensemble, legend_mcdo, legend_tta], ["Ensemble", "MC-Dropout", "TTA"], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylim(0, 0.1)
    plt.title("Uncertainty coverage (Predictive entropy)")
    plt.savefig(load_dir + "UC_E.png", bbox_inches='tight')
    plt.clf()

    fig, ax = plt.subplots()
    results = load("ensemble", "predictive_variance")
    plt.plot(results["Thresholds"], results["UC_threshold"], linestyle="-", color="r", label="UC")
    legend_ensemble = Line2D([0, 1], [0, 1], linestyle='-', color="r")
    results = load("mcdo", "predictive_variance")
    plt.plot(results["Thresholds"], results["UC_threshold"], linestyle="-", color="g", label="UC")
    legend_mcdo = Line2D([0, 1], [0, 1], linestyle='-', color="g")
    results = load("tta", "predictive_variance")
    plt.plot(results["Thresholds"], results["UC_threshold"], linestyle="-", color="b", label="UC")
    legend_tta = Line2D([0, 1], [0, 1], linestyle='-', color="b")
    ax.legend([legend_ensemble, legend_mcdo, legend_tta], ["Ensemble", "MC-Dropout", "TTA"], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylim(0, 0.1)
    plt.title("Uncertainty coverage (Predictive variance)")
    plt.savefig(load_dir + "UC_V.png", bbox_inches='tight')
    plt.clf()

def plot_results_combined2(load_dir):

    def load(uncertainty_quantification, uncertainty_measure):
        with open(load_dir + 'UC_{}_{}.pkl'.format(uncertainty_quantification, uncertainty_measure), 'rb') as handle:
            results = pickle.load(handle)
            return {"UC_threshold": results["UC_threshold"], "Thresholds": results["Thresholds"]}

    fig, ax = plt.subplots()
    results = load("ensemble", "bhattacharyya_coefficient")
    plt.plot(results["Thresholds"], results["UC_threshold"], linestyle="-", color="r", label="UC")
    legend_BC = Line2D([0, 1], [0, 1], linestyle='-', color="r")
    results = load("ensemble", "predictive_entropy")
    plt.plot(results["Thresholds"], results["UC_threshold"], linestyle="-", color="g", label="UC")
    legend_entropy = Line2D([0, 1], [0, 1], linestyle='-', color="g")
    results = load("ensemble", "predictive_variance")
    plt.plot(results["Thresholds"], results["UC_threshold"], linestyle="-", color="b", label="UC")
    legend_variance = Line2D([0, 1], [0, 1], linestyle='-', color="b")
    ax.legend([legend_BC, legend_entropy, legend_variance], ["BC", "Entropy", "Variance"], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #plt.ylim(0, 0.1)
    plt.title("Uncertainty coverage")
    plt.savefig(load_dir + "UC.png", bbox_inches='tight')
    plt.clf()


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-t", "--task", help="Task", required=True)
#     parser.add_argument("-s", "--set", help="val/test", required=True)
#     args = parser.parse_args()
#
#     thresholds = np.arange(0.0, 1.0, 0.03)
#     uqs = ["ensemble", "tta", "mcdo"]
#     ums = ["bhattacharyya_coefficient", "predictive_entropy", "predictive_variance"]
#     task = args.task
#
#     for uq in uqs:
#         for um in ums:
#             base_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/" + task + "/"
#             uncertainty_dir = base_path + "refinement_" + args.set + "/uncertainties/" + uq + "/" + um + "/"
#             prediction_dir = base_path + "refinement_" + args.set + "/basic_predictions/"
#             gt_dir = base_path + "refinement_" + args.set + "/labels/"
#             save_dir = base_path + "refinement_" + args.set + "/"
#             name = save_dir + "uncertainty_evaluation/UC_" + uq + "_" + um
#
#             results = comp_uncertainty_coverage(uncertainty_dir, prediction_dir, gt_dir, thresholds)
#             print("UQ: {}, UM: {}, UC: {}".format(uq, um, results["UC"]))
#
#             with open(name + ".pkl", 'wb') as handle:
#                 pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#     plot_results_combined("/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/" + task + "/refinement_" + args.set + "/uncertainty_evaluation/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", help="Task", required=True)
    parser.add_argument("-s", "--set", help="val/test", required=True)
    args = parser.parse_args()

    thresholds = np.arange(0.0, 1.0, 0.03)
    uqs = ["ensemble"]
    ums = ["bhattacharyya_coefficient", "predictive_entropy", "predictive_variance"]
    task = args.task

    for uq in uqs:
        for um in ums:
            base_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/" + task + "/"
            uncertainty_dir = base_path + "refinement_" + args.set + "/uncertainties/" + uq + "/" + um + "/"
            prediction_dir = base_path + "refinement_" + args.set + "/basic_predictions/"
            gt_dir = base_path + "refinement_" + args.set + "/labels/"
            save_dir = base_path + "refinement_" + args.set + "/"
            name = save_dir + "uncertainty_evaluation/UC_" + uq + "_" + um

            results = comp_uncertainty_coverage(uncertainty_dir, prediction_dir, gt_dir, thresholds)
            print("UQ: {}, UM: {}, UC: {}".format(uq, um, results["UC"]))

            with open(name + ".pkl", 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plot_results_combined2("/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/" + task + "/refinement_" + args.set + "/uncertainty_evaluation/")

