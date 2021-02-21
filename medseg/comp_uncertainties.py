from medseg import utils
import numpy as np
import os
from tqdm import tqdm
import argparse


def comp_tta_uncertainties(load_dir, save_dir, type="part"):
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
            uncertainty = comp_variance_uncertainty(predictions)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input folder", required=True)
    parser.add_argument("-o", "--output", help="Output folder", required=True)
    args = parser.parse_args()

    comp_tta_uncertainties(args.input, args.output)
