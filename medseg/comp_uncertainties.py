from medseg import utils
import numpy as np
import os
from tqdm import tqdm
import argparse
import sys


def comp_uncertainties(load_dir, save_dir, uncertainty_estimator, type="part"):
    load_dir = utils.fix_path(load_dir)
    save_dir = utils.fix_path(save_dir)
    filenames = utils.load_filenames(load_dir)
    cases, nr_labels, nr_parts = group_data(filenames)
    print("nr_cases: ", len(cases))
    print("nr_labels: ", nr_labels)
    print("nr_parts: ", nr_parts)

    for case in tqdm(cases):
        for label in range(nr_labels+1):
            predictions = []
            for part in range(nr_parts+1):
                name = load_dir + str(case).zfill(4) + "_" + str(label) + "_" + type + "_" + str(part) + ".nii.gz"
                prediction, affine, spacing, header = utils.load_nifty(name)
                predictions.append(prediction.astype(np.float16))
            predictions = np.stack(predictions)
            uncertainty = uncertainty_estimator(predictions)
            name = save_dir + str(case).zfill(4) + "_" + str(label) + ".nii.gz"
            utils.save_nifty(name, uncertainty, affine, spacing, header)


def group_data(filenames):
    cases = []
    nr_labels = 0
    nr_parts = 0

    for filename in filenames:
        filename = os.path.basename(filename)
        case_nr = int(filename[:4])
        cases.append(case_nr)
        label_nr = int(filename[5:6])
        if nr_labels < label_nr:
            nr_labels = label_nr
        part_nr = int(filename[12:13])
        if nr_parts < part_nr:
            nr_parts = part_nr
    # Remove duplicates
    cases = list(dict.fromkeys(cases))
    return cases, nr_labels, nr_parts


def comp_variance_uncertainty(predictions):
    predictive_posterior_mean = np.mean(predictions, axis=0)
    uncertainty = np.zeros_like(predictive_posterior_mean)
    for prediction in predictions:
        uncertainty += (prediction - predictive_posterior_mean) ** 2
    uncertainty /= len(predictions)
    return uncertainty


def comp_entropy_uncertainty(predictions):
    uncertainty = np.zeros_like(predictions)
    for prediction in predictions:
        uncertainty = prediction * np.log(prediction + sys.float_info.epsilon)
    uncertainty = -1 * np.sum(uncertainty, axis=0)
    return uncertainty


def comp_bhattacharyya_uncertainty(predictions):
    uncertainty = np.sqrt(np.prod(predictions, axis=0))
    uncertainty += np.sqrt(np.prod(1 - predictions, axis=0))
    return uncertainty


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input folder", required=True)
    parser.add_argument("-o", "--output", help="Output folder", required=True)
    parser.add_argument("-m", "--method", help="Variance (v), entropy (e) or bhattacharyya (b)", required=True)
    args = parser.parse_args()

    method = str(args.method)

    if method == "v":
        uncertainty_estimator = comp_variance_uncertainty
    elif method == "e":
        uncertainty_estimator = comp_entropy_uncertainty
    elif method == "e":
        uncertainty_estimator = comp_bhattacharyya_uncertainty
    else:
        raise RuntimeError("Unknown uncertainty estimator.")

    comp_uncertainties(args.input, args.output, uncertainty_estimator)
