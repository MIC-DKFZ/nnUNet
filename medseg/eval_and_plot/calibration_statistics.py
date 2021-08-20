from medseg import utils
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle


def load_pred_and_gt(prediction_path, gt_path):
    prediction_filenames = utils.load_filenames(prediction_path)
    gt_filenames = utils.load_filenames(gt_path)

    predictions, gts = [], []
    for i in tqdm(range(len(prediction_filenames))):
        prediction = utils.load_nifty(prediction_filenames[i])[0]
        gt = utils.load_nifty(gt_filenames[i])[0]
        predictions.append(prediction.flatten())
        gts.append(gt.flatten())
    predictions = np.concatenate(predictions, axis=0)
    gts = np.concatenate(gts, axis=0)
    labels = np.unique(gts)
    return predictions, gts, labels


def load_uncertainty(uncertainty_path):
    uncertainty_filenames = utils.load_filenames(uncertainty_path)

    uncertainties = []
    for i in tqdm(range(len(uncertainty_filenames))):
        uncertainty = utils.load_nifty(uncertainty_filenames[i])[0]
        uncertainties.append(uncertainty.flatten())
    uncertainties = np.concatenate(uncertainties, axis=0)
    return uncertainties


def reliability_diagram(uncertainty, prediction, gt, n_bins, labels):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidence = 1 - uncertainty

    ece = 0
    mce = []
    accuracy = np.zeros_like(gt)
    for label in labels:
        accuracy = (accuracy == 1) | ((prediction == label) & (gt == label))
    total_samples = np.sum((prediction > 0) | (gt > 0))
    # accuracy = prediction == gt
    # total_samples = confidence.size
    x, y, samples_per_bin = [], [], []
    summed = 0
    for bin_lower, bin_upper in tqdm(zip(bin_lowers, bin_uppers), total=n_bins):
        bin_mask = np.zeros_like(gt)
        for label in labels:
            bin_mask = (bin_mask == 1) | ((bin_lower <= confidence) & (confidence < bin_upper) & ((prediction == label) | (gt == label)))
        # bin_mask = (bin_lower <= confidence) & (confidence < bin_upper) & ((prediction == 1) | (gt == 1))
        bin_samples = np.sum(bin_mask)
        if bin_samples > 0:
            samples_per_bin.append(bin_samples/total_samples)
            summed += bin_samples
            bin_confidence = confidence[bin_mask]
            bin_accuracy = accuracy[bin_mask]
            bin_confidence = np.mean(bin_confidence)
            bin_accuracy = np.mean(bin_accuracy)
            ece += (bin_samples/total_samples) * np.abs(bin_accuracy - bin_confidence)
            # print("ECE: ", ece)
            mce.append(np.abs(bin_accuracy - bin_confidence))
            # print("MCE: ", mce)
            x.append(bin_upper)
            y.append(bin_accuracy)
    mce = np.max(mce)
    return x, y, samples_per_bin, ece, mce


# if __name__ == '__main__':
#     base_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/"
#     tasks = ["Task070_guided_all_public_ggo", "Task002_BrainTumour_guided", "Task008_Pancreas_guided"]
#     task_names = ["COVID-19", "Brain Tumor", "Pancreas"]
#     set = "val"
#     uq = "ensemble"
#     ums = ["confidence", "bhattacharyya_coefficient", "predictive_entropy", "predictive_variance"]
#     um_names = ["Max Class Softmax", "BC", "Entropy", "Variance"]
#     n_bins = 20
#     load = True
#
#
#     for i, task in enumerate(tasks):
#         if load:
#             with open(base_path + task + ".pkl", 'rb') as handle:
#                 um_results = pickle.load(handle)
#         else:
#             prediction_path = base_path + task + "/refinement_" + set + "/basic_predictions/"
#             gt_path = base_path + task + "/refinement_" + set + "/labels/"
#             predictions, gts, labels = load_pred_and_gt(prediction_path, gt_path)
#             um_results = {}
#             for j in range(len(ums)):
#                 uncertainty_path = base_path + task + "/refinement_" + set + "/uncertainties/" + uq + "/" + ums[j] + "/"
#                 uncertainties = load_uncertainty(uncertainty_path)
#                 x, y, samples_per_bin, ece, mce = reliability_diagram(uncertainties, predictions, gts, n_bins, labels)
#                 um_results[ums[j]] = {"x": x, "y": y, "samples_per_bin": samples_per_bin, "ece": ece, "mce": mce}
#             with open(base_path + task + ".pkl", 'wb') as handle:
#                 pickle.dump(um_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
#         for j, key in enumerate(um_results.keys()):
#             x, y, samples_per_bin, ece, mce = um_results[key]["x"], um_results[key]["y"], um_results[key]["samples_per_bin"], um_results[key]["ece"], um_results[key]["mce"]
#             print("Task: {}, um: {}, ECE: {}, MCE: {}".format(task, key, ece, mce))
#             plt.plot(x, y, label=um_names[j])
#         x_y_ideal = np.arange(0.0, 1.1, 0.1)
#         plt.title("Reliability Diagram ({})".format(task_names[i]))
#         plt.plot(x_y_ideal, x_y_ideal, label="Ideal", linestyle="--")
#         plt.ylim(0, 1)
#         plt.xlim(0, 1)
#         plt.xlabel("Confidence")
#         plt.ylabel("Accuracy")
#         plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#         plt.savefig(base_path + task + ".png", bbox_inches='tight')
#         plt.clf()


if __name__ == '__main__':
    base_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/"
    task = "Task070_guided_all_public_ggo"
    task_name = "COVID-19"
    set = "val"
    uqs = ["ensemble", "mcdo", "tta"]
    uqs_names = ["Ensemble", "MC Dropout", "TTA"]
    ums = ["confidence", "bhattacharyya_coefficient", "predictive_entropy", "predictive_variance"]
    um_names = ["Max Class Softmax", "BC", "Entropy", "Variance"]
    n_bins = 20
    load = True


    for i, uq in enumerate(uqs):
        if load:
            with open(base_path + uq + ".pkl", 'rb') as handle:
                uq_results = pickle.load(handle)
        else:
            prediction_path = base_path + task + "/refinement_" + set + "/basic_predictions/"
            gt_path = base_path + task + "/refinement_" + set + "/labels/"
            predictions, gts, labels = load_pred_and_gt(prediction_path, gt_path)
            uq_results = {}
            for j in range(len(ums)):
                uncertainty_path = base_path + task + "/refinement_" + set + "/uncertainties/" + uq + "/" + ums[j] + "/"
                uncertainties = load_uncertainty(uncertainty_path)
                x, y, samples_per_bin, ece, mce = reliability_diagram(uncertainties, predictions, gts, n_bins, labels)
                uq_results[ums[j]] = {"x": x, "y": y, "samples_per_bin": samples_per_bin, "ece": ece, "mce": mce}
            with open(base_path + uq + ".pkl", 'wb') as handle:
                pickle.dump(uq_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        for j, key in enumerate(uq_results.keys()):
            x, y, samples_per_bin, ece, mce = uq_results[key]["x"], uq_results[key]["y"], uq_results[key]["samples_per_bin"], uq_results[key]["ece"], uq_results[key]["mce"]
            print("UQ: {}, um: {}, ECE: {}, MCE: {}".format(uq, key, ece, mce))
            plt.plot(x, y, label=um_names[j])
        x_y_ideal = np.arange(0.0, 1.1, 0.1)
        plt.title("Reliability Diagram ({})".format(uqs_names[i]))
        plt.plot(x_y_ideal, x_y_ideal, label="Ideal", linestyle="--")
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.xlabel("Confidence")
        plt.ylabel("Accuracy")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig(base_path + uq + ".png", bbox_inches='tight')
        plt.clf()
