import numpy as np
from collections import defaultdict
from medseg import utils
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import pickle
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt
from numbers import Number
import argparse


def evaluate(uncertainty_dir, prediction_dir, gt_dir, save_dir, normalize, thresholds, parallel, target_shape=(256, 256, 50)):
    metrices_meter = MetricesMeter(threshold=thresholds, parallel=parallel, normalize=normalize)
    uncertainty_dir = utils.fix_path(uncertainty_dir)
    prediction_dir = utils.fix_path(prediction_dir)
    gt_dir = utils.fix_path(gt_dir)
    uncertainty_filenames = utils.load_filenames(uncertainty_dir)
    prediction_filenames = utils.load_filenames(prediction_dir)
    gt_filenames = utils.load_filenames(gt_dir)
    uncertainty_all, prediction_all, ground_truth_all = [], [], []

    for i in tqdm(range(len(uncertainty_filenames))):
        uncertainty = utils.load_nifty(uncertainty_filenames[i])[0]#.astype(np.float16)
        prediction = utils.load_nifty(prediction_filenames[i])[0]#.astype(np.float16)
        ground_truth = utils.load_nifty(gt_filenames[i])[0]#.astype(np.float16)
        # print("Uncertainty min: {}, max: {}".format(np.min(uncertainty), np.max(uncertainty)))
        # print("Ground truth min: {}, max: {}".format(np.min(ground_truth), np.max(ground_truth)))
        # print("Shape: ", uncertainty.shape)
        # uncertainty = utils.interpolate(uncertainty, (target_shape[0], target_shape[1], uncertainty.shape[2]), mask=False)
        # ground_truth = utils.interpolate(ground_truth, (target_shape[0], target_shape[1], ground_truth.shape[2]), mask=True)
        uncertainty = utils.interpolate(uncertainty, target_shape, mask=False)
        prediction = utils.interpolate(prediction, target_shape, mask=True)
        ground_truth = utils.interpolate(ground_truth, target_shape, mask=True)
        uncertainty_all.append(uncertainty)
        prediction_all.append(prediction)
        ground_truth_all.append(ground_truth)
        # print("Resized shape: ", uncertainty.shape)
        # break
    uncertainty_all = np.concatenate(uncertainty_all, axis=0)
    prediction_all = np.concatenate(prediction_all, axis=0)
    ground_truth_all = np.concatenate(ground_truth_all, axis=0)
    print("Memory size (Uncertainty): {} GB".format(round(uncertainty_all.nbytes / 1024 / 1024 / 1024,2)))
    print("Memory size (Prediction): {} GB".format(round(prediction_all.nbytes / 1024 / 1024 / 1024, 2)))
    print("Memory size (Ground Truth): {} GB".format(round(ground_truth_all.nbytes / 1024 / 1024 / 1024, 2)))
    print("Max uncertainty: ", np.max(uncertainty_all))
    metrices_meter.update(uncertainty_all, prediction_all, ground_truth_all)
    # break

    results = metrices_meter.compute()
    print("Computed results")

    return results


def plot_results(results, name):
    for key in results.keys():
        print("{}: {}".format(key, results[key]))
        if key != "Thresholds":
            plt.plot(results["Thresholds"], results[key], label=key)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylim(0, 1)
    plt.savefig(name + ".png", bbox_inches='tight')
    plt.clf()
    print("Finished plot")


class MetricesMeter:
    def __init__(self, threshold, parallel=False, normalize=False):
        self.metrices_results = defaultdict(lambda: defaultdict(list))
        self.parallel = parallel
        self.normalize = normalize
        if self.parallel:
            self.pool = mp.Pool(processes=10)
        if isinstance(threshold, Number):
            self.thresholds = [threshold]
        else:
            self.thresholds = threshold

    def reset(self):
        self.metrices_results = defaultdict(lambda: defaultdict(list))

    def update(self, uncertainty, prediction, ground_truth):
        if self.normalize:
            uncertainty = utils.normalize(uncertainty)
        if not self.parallel:
            for threshold in tqdm(self.thresholds):
                metrices_results_threshold = comp_metrics(threshold, uncertainty, prediction, ground_truth)
                for metric in metrices_results_threshold.keys():
                    self.metrices_results[threshold][metric].append(metrices_results_threshold[metric])
        else:
            metrices_results_thresholds = self.pool.map(partial(comp_metrics, _uncertainty=uncertainty, _prediction=prediction, _ground_truth=ground_truth), self.thresholds)
            for i in range(len(self.thresholds)):
                for key in metrices_results_thresholds[i].keys():
                    self.metrices_results[self.thresholds[i]][key].append(metrices_results_thresholds[i][key])

    def compute(self):
        for threshold in self.thresholds:
            for key in self.metrices_results[threshold].keys():
                self.metrices_results[threshold][key] = np.mean(self.metrices_results[threshold][key])
        metrices = list(self.metrices_results[self.thresholds[0]].keys())
        results = {}
        results["Thresholds"] = self.thresholds
        for metric in metrices:
            results[metric] = [self.metrices_results[threshold][metric] for threshold in self.thresholds]
        return results


def comp_metrics(threshold, _uncertainty, _prediction, _ground_truth):
    print("+++")
    # uncertainty = (_uncertainty.astype('float64') > threshold.astype('float64')).astype('float64').flatten().astype(int)
    uncertainty = (_uncertainty > threshold).flatten().astype(int)
    prediction = np.rint(_prediction).flatten().astype(int)
    ground_truth = np.rint(_ground_truth).flatten().astype(int)
    uncertainty, ground_truth = guard_input(uncertainty, ground_truth)
    uncertainty, prediction = guard_input(uncertainty, prediction)
    # u_gt_tn, u_gt_fp, u_gt_fn, u_gt_tp = confusion_matrix(ground_truth, uncertainty).ravel()
    # u_p_tn, u_p_fp, u_p_fn, u_p_tp = confusion_matrix(ground_truth, uncertainty).ravel()
    # u_gt_tp, u_gt_tn, u_gt_fp, u_gt_fn = float(u_gt_tp), float(u_gt_tn), float(u_gt_fp), float(u_gt_fn)
    # u_p_tp, u_p_tn, u_p_fp, u_p_fn = float(u_p_tp), float(u_p_tn), float(u_p_fp), float(u_p_fn)

    missclassification = comp_missclassification(prediction, ground_truth)
    true_and_confident, true_but_uncertain, false_and_confident, false_and_uncertain = confusion_matrix(missclassification, uncertainty).ravel()
    true_and_confident, true_but_uncertain, false_and_confident, false_and_uncertain = float(true_and_confident), float(true_but_uncertain), float(false_and_confident), float(false_and_uncertain)
    metrices_results = {}

    metrices_results["Missclassification-Uncertainty-Coverage"] = missclassification_uncertainty_coverage(false_and_uncertain, missclassification)
    metrices_results["False-Confidence-To-Missclassification-Ratio"] = false_confidence2missclassification_ratio(false_and_confident, missclassification)
    # metrices_results["False-Confidence-To-MUC-Ratio"] = false_confidence2MUC_ratio(false_and_confident, false_and_uncertain)
    metrices_results["False-Confidence-To-GT-Ratio"] = false_confidence2gt_ratio(false_and_confident, ground_truth)
    metrices_results["Uncertainty-To-GT-Ratio"] = uncertainty2gt_ratio(uncertainty, ground_truth)
    metrices_results["True-Confident-Prediction-To-Correct-Prediction-Ratio"] = true_confident_prediction2correct_prediction_ratio(uncertainty, prediction, ground_truth)
    # metrices_results["Threshold"] = threshold
    print("---")
    return metrices_results


def comp_missclassification(prediction, ground_truth):
    missclassification = np.zeros_like(prediction)
    missclassification[prediction != ground_truth] = 1
    return missclassification


def missclassification_uncertainty_coverage(false_and_uncertain, missclassification): # MUC
    "Ratio of how many missclassified voxels (false predictions) are also assumed (covered) to be uncertain by the model. Higher is better."
    return false_and_uncertain / np.sum(missclassification)


def false_confidence2missclassification_ratio(false_and_confident, missclassification):
    "Measures how much the model is confident in its prediction but is actually wrong. Set in relation to the total missclassification. Smaller is better."
    return false_and_confident / np.sum(missclassification)


# def false_confidence2MUC_ratio(false_and_confident, false_and_uncertain):
#     "Measures how much the model is confident in its prediction but is actually wrong. Set in relation to the total missclassification uncertainty coverage. Smaller is better."
#     return false_and_confident / false_and_uncertain


def false_confidence2gt_ratio(false_and_confident, ground_truth):
    "Measures how much the model is confident in its prediction but is actually wrong. Set in relation to the total ground truth. Smaller is better."
    return false_and_confident / np.sum(ground_truth)


def uncertainty2gt_ratio(uncertainty, ground_truth):
    "Measures how much uncertainty there is compared to ground truth."
    return np.sum(uncertainty) / np.sum(ground_truth)


def true_confident_prediction2correct_prediction_ratio(uncertainty, prediction, ground_truth):
    correct_prediction = np.zeros_like(prediction)
    correct_prediction[(prediction == 1) & (ground_truth == 1)] = 1
    confident_prediction = np.zeros_like(prediction)
    confident_prediction[(correct_prediction == 1) & (uncertainty == 0)] = 1
    return np.sum(confident_prediction) / np.sum(correct_prediction)


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--set", help="val/test", required=True)
    parser.add_argument("-uq", "--uncertainty_quantification", help="Set the type of uncertainty quantification method to use", required=True)
    parser.add_argument("-um", "--uncertainty_measure", help="Set the type of uncertainty measure to use", required=True)
    args = parser.parse_args()

    uncertainty_quantification_args = ["e", "t", "m"]
    uncertainty_measure_args = ["b"]

    for uncertainty_quantification_arg in uncertainty_quantification_args:
        for uncertainty_measure_arg in uncertainty_measure_args:
            args.uncertainty_quantification = uncertainty_quantification_arg
            args.uncertainty_measure = uncertainty_measure_arg

            uncertainty_quantification = str(args.uncertainty_quantification)
            uncertainty_measure = str(args.uncertainty_measure)
            name = "uncertainty_evaluation_ " + uncertainty_quantification + "_" + uncertainty_measure

            if uncertainty_quantification == "e":
                uncertainty_quantification = "ensemble"
            elif uncertainty_quantification == "t":
                uncertainty_quantification = "tta"
            elif uncertainty_quantification == "m":
                uncertainty_quantification = "mcdo"
            else:
                raise RuntimeError("uncertainty_quantification unknown")

            if uncertainty_measure == "b":
                uncertainty_measure = "bhattacharyya_coefficient"
            elif uncertainty_measure == "e":
                uncertainty_measure = "predictive_entropy"
            elif uncertainty_measure == "v":
                uncertainty_measure = "predictive_variance"
            else:
                raise RuntimeError("uncertainty_measure unknown")

            base_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task070_guided_all_public_ggo/"
            uncertainty_dir = base_path + "refinement_" + args.set + "/uncertainties/" + uncertainty_quantification + "/" + uncertainty_measure + "/"
            prediction_dir = base_path + "refinement_" + args.set + "/basic_predictions/"
            gt_dir = base_path + "refinement_" + args.set + "/labels/"
            save_dir = base_path + "refinement_" + args.set + "/"
            name = save_dir + "uncertainty_evaluation_ " + uncertainty_quantification + "_" + uncertainty_measure

            # bhattacharyya_coefficient: normalize = False, thresholds = np.arange(0.0, 0.2, 0.01)
            # predictive_entropy:
            # predictive_variance:

            normalize = False
            parallel = True

            # b
            thresholds1 = np.arange(0.0, 0.2, 0.005)
            thresholds2 = np.arange(0.2, 0.3, 0.01)
            thresholds = np.concatenate([thresholds1, thresholds2], axis=0)

            # # e
            # thresholds1 = np.arange(0.0, 0.02, 0.0005)
            # thresholds2 = np.arange(0.02, 0.05, 0.01)
            # thresholds = np.concatenate([thresholds1, thresholds2], axis=0)

            # # v
            # thresholds1 = np.arange(0.0, 0.02, 0.0005)
            # thresholds2 = np.arange(0.02, 0.1, 0.01)
            # thresholds = np.concatenate([thresholds1, thresholds2], axis=0)

            results = evaluate(uncertainty_dir, prediction_dir, gt_dir, save_dir, normalize, thresholds, parallel)

            with open(name + ".pkl", 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # with open(save_dir + 'uncertainty_evaluation.pkl', 'rb') as handle:
            #     results = pickle.load(handle)
            #
            plot_results(results, name)

