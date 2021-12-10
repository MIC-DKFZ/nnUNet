from i3Deep import utils
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


def reliability_diagram_bce(uncertainty, prediction, gt, n_bins, labels):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidence = 1 - uncertainty

    bce = 0
    # mce = []
    accuracy = np.zeros_like(gt)
    labels = labels[1:]
    # print("Labels: ", labels)
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
            # ece += (bin_samples/total_samples) * np.abs(bin_accuracy - bin_confidence)
            print("bin_accuracy: {}, bin_confidence: {}".format(bin_accuracy, bin_confidence))
            bce += (1 / n_bins) * np.abs(bin_accuracy - bin_confidence)
            # print("ECE: ", ece)
            # mce.append(np.abs(bin_accuracy - bin_confidence))
            # print("MCE: ", mce)
            x.append(bin_upper)
            y.append(bin_accuracy)
    # mce = np.max(mce)
    return x, y, samples_per_bin, bce


def reliability_diagram_ece(uncertainty, prediction, gt, n_bins, labels):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidence = 1 - uncertainty

    ece = 0
    # mce = []
    accuracy = np.zeros_like(gt)
    #labels = labels[1:]
    # print("Labels: ", labels)
    for label in labels:
        accuracy = (accuracy == 1) | ((prediction == label) & (gt == label))
    total_samples = np.prod(gt.shape)
    # accuracy = prediction == gt
    # total_samples = confidence.size
    x, y, samples_per_bin = [], [], []
    for bin_lower, bin_upper in tqdm(zip(bin_lowers, bin_uppers), total=n_bins):
        bin_mask = (bin_lower <= confidence) & (confidence < bin_upper)
        bin_samples = np.sum(bin_mask)
        if bin_samples > 0:
            samples_per_bin.append(bin_samples/total_samples)
            bin_confidence = confidence[bin_mask]
            bin_accuracy = accuracy[bin_mask]
            bin_confidence = np.mean(bin_confidence)
            bin_accuracy = np.mean(bin_accuracy)
            ece += (bin_samples/total_samples) * np.abs(bin_accuracy - bin_confidence)
            print("bin_accuracy: {}, bin_confidence: {}".format(bin_accuracy, bin_confidence))
            x.append(bin_upper)
            y.append(bin_accuracy)
    return x, y, samples_per_bin, ece


if __name__ == '__main__':
    base_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/"
    tasks = ["Task002_BrainTumour_guided", "Task008_Pancreas_guided", "Task070_guided_all_public_ggo"]
    task_names = ["Brain Tumor", "Pancreas", "COVID-19"]
    set = "val"
    uqs = ["ensemble", "mcdo", "tta"]
    uqs_names = ["Deep Ensemble", "MC Dropout", "TTA"]
    ums = ["confidence", "bhattacharyya_coefficient", "predictive_entropy", "predictive_variance"]
    um_names = ["Simple Confidence", "BC", "Entropy", "Variance"]
    gridspec_indices = [[slice(0, 2), slice(0, 2)], [slice(0, 2), slice(2, 4)], [slice(2, 4), slice(1, 3)]]
    n_bins = 20
    load = True
    comp_bce = False
    if comp_bce:
        folder = "BCE"
    else:
        folder = "ECE"

    for k, task in enumerate(tasks):
        fig = plt.figure(constrained_layout=True, figsize=(12, 7))
        gs = fig.add_gridspec(4, 4)
        print("Task: ", task)
        for i, uq in enumerate(uqs):
            if load:
                with open(base_path + "Evaluation/Uncertainty Evaluation/" + folder + "/" + task_names[k] + "_" + uq + ".pkl", 'rb') as handle:
                    uq_results = pickle.load(handle)
            else:
                prediction_path = base_path + task + "/refinement_" + set + "/basic_predictions/"
                gt_path = base_path + task + "/refinement_" + set + "/labels/"
                predictions, gts, labels = load_pred_and_gt(prediction_path, gt_path)
                uq_results = {}
                for j in range(len(ums)):
                    uncertainty_path = base_path + task + "/refinement_" + set + "/uncertainties/" + uq + "/" + ums[j] + "/"
                    uncertainties = load_uncertainty(uncertainty_path)

                    # import seaborn as sns
                    # uncertainties = uncertainties[predictions != 0]
                    # uncertainties = 1 - uncertainties
                    # # plt.hist(uncertainties)
                    # fig, ax = plt.subplots(1, figsize=(5, 5))
                    # ax = sns.kdeplot(data=uncertainties, color="blue", shade=True, ax=ax)
                    # ax.set_xlim(0, 1)
                    # ax.set_xlabel("Confidence")
                    # ax.set_ylabel("Frequency")
                    # plt.show()

                    if comp_bce:
                        x, y, samples_per_bin, bce = reliability_diagram_bce(uncertainties, predictions, gts, n_bins, labels)
                    else:
                        x, y, samples_per_bin, bce = reliability_diagram_ece(uncertainties, predictions, gts, n_bins, labels)
                    uq_results[ums[j]] = {"x": x, "y": y, "samples_per_bin": samples_per_bin, "bce": bce}
                with open(base_path + "Evaluation/Uncertainty Evaluation/" + folder + "/" + task_names[k] + "_" + uq + ".pkl", 'wb') as handle:
                    pickle.dump(uq_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
            ax = fig.add_subplot(gs[gridspec_indices[i][0], gridspec_indices[i][1]])
            for j, key in enumerate(uq_results.keys()):
                x, y, samples_per_bin, bce = uq_results[key]["x"], uq_results[key]["y"], uq_results[key]["samples_per_bin"], uq_results[key]["bce"]
                print("UQ: {}, um: {}, BCE: {}".format(uq, key, round(bce, 4)))
                # x, y, samples_per_bin = uq_results[key]["x"], uq_results[key]["y"], uq_results[key]["samples_per_bin"]
                ax.plot(x, y, label=um_names[j])
            x_y_ideal = np.arange(0.0, 1.1, 0.1)
            ax.set_title(uqs_names[i])
            ax.plot(x_y_ideal, x_y_ideal, label="Ideal", linestyle="--")
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Accuracy")
        plt.suptitle("Balanced Reliability Curve ({})".format(task_names[k]), fontsize=16)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig(base_path + "Evaluation/Uncertainty Evaluation/" + folder + "/" + task_names[k] + ".png", bbox_inches='tight')
        plt.clf()
