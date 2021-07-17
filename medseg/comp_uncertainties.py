from medseg import utils
import numpy as np
import os
from tqdm import tqdm
import argparse
import sys
from scipy.special import softmax


def comp_uncertainties(load_dir, save_dir, uncertainty_estimator, basename=None, cases=None, nr_labels=None, nr_parts=None, zfill=4, count_start=None, type="part", merge_uncertainties=False, class_dices=None):
    load_dir = utils.fix_path(load_dir)
    save_dir = utils.fix_path(save_dir)
    filenames = utils.load_filenames(load_dir)
    if cases is None:
        cases, nr_labels, nr_parts = group_data(filenames)
    else:
        cases = list(range(count_start, cases+count_start, 1))
    print("nr_cases: ", len(cases))
    print("nr_labels: ", nr_labels)
    print("nr_parts: ", nr_parts)

    for case in tqdm(cases):
        uncertainty_classes = []
        for label in range(nr_labels):
            predictions = []
            for part in range(nr_parts):
                if basename is not None:
                    name = load_dir + basename + "_" + str(case).zfill(zfill) + "_" + str(label) + "_" + type + "_" + str(part) + ".nii.gz"
                else:
                    name = load_dir + str(case).zfill(zfill) + "_" + str(label) + "_" + type + "_" + str(part) + ".nii.gz"
                prediction, affine, spacing, header = utils.load_nifty(name)
                predictions.append(prediction.astype(np.float16))
            predictions = np.stack(predictions)
            uncertainty = uncertainty_estimator(predictions)
            if not merge_uncertainties:
                name = create_name(save_dir, basename, case, label, zfill, False)
                utils.save_nifty(name, uncertainty, affine, spacing, header)
            else:
                uncertainty_classes.append(uncertainty)
        if merge_uncertainties:
            uncertainty_classes = uncertainty_classes[1:]  # Remove background uncertainty
            if not class_dices:
                uncertainty_classes = np.mean(uncertainty_classes, axis=0)
            else:
                weights = [1 - dice for dice in class_dices]
                weights = softmax(weights)
                uncertainty_classes = np.sum(np.asarray(uncertainty_classes) * np.asarray(weights))
            name = create_name(save_dir, basename, case, None, zfill, True)
            utils.save_nifty(name, uncertainty_classes, affine, spacing, header)


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
    return cases, nr_labels+1, nr_parts+1


def comp_variance_uncertainty(predictions):
    mean = np.mean(predictions, axis=0)
    uncertainty = np.zeros_like(mean)
    for prediction in predictions:
        uncertainty += (prediction - mean) ** 2
    uncertainty /= len(predictions)
    return uncertainty


def comp_entropy_uncertainty(predictions):
    predictions = predictions.transpose(1, 2, 3, 0)
    predictions = softmax(predictions, axis=3)
    predictions = predictions.transpose(3, 0, 1, 2)
    uncertainty = np.zeros(predictions.shape[1:])
    for prediction in predictions:
        uncertainty += prediction * np.log2(prediction + sys.float_info.epsilon)
    uncertainty = -1 * uncertainty / np.log2(predictions.shape[0])
    uncertainty = 1 - uncertainty
    return uncertainty


def comp_bhattacharyya_uncertainty(predictions):
    uncertainty = np.power(np.prod(predictions, axis=0), 1.0/float(predictions.shape[0]))
    uncertainty += np.power(np.prod(1 - predictions, axis=0), 1.0/float(predictions.shape[0]))
    uncertainty = 1 - uncertainty
    return uncertainty


def create_name(save_dir, basename, case, label, zfill, merged):
    if not merged:
        if basename is not None:
            name = save_dir + basename + "_" + str(case).zfill(zfill) + "_" + str(label) + ".nii.gz"
        else:
            name = save_dir + str(case).zfill(4) + "_" + str(label) + ".nii.gz"
    else:
        if basename is not None:
            name = save_dir + basename + "_" + str(case).zfill(zfill)  + ".nii.gz"
        else:
            name = save_dir + str(case).zfill(4) + ".nii.gz"
    return name


def preprocess_uncertainty_files(data_dir):
    data_dir = utils.fix_path(data_dir)
    filenames = utils.load_filenames(data_dir)

    for filename in tqdm(filenames):
        if filename[-9:] == "_0.nii.gz":
            os.remove(filename)
        else:
            os.rename(filename, filename[:-9] + ".nii.gz")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input folder", required=True)
    parser.add_argument("-o", "--output", help="Output folder", required=True)
    parser.add_argument("-m", "--method", help="Variance (v), entropy (e) or bhattacharyya (b)", required=True)

    # Stupid arguments
    parser.add_argument("-nr_c", "--nr_cases", default=None, help="Number of cases", required=False)
    parser.add_argument("-nr_l", "--nr_labels", default=None, help="Number of labels", required=False)
    parser.add_argument("-nr_p", "--nr_parts", default=None, help="Number of parts", required=False)
    parser.add_argument("-basename", default=None, help="Basename", required=False)
    parser.add_argument("-zfill", default=4, help="Number of leading zeros for case numeration", required=False)
    parser.add_argument("-count_start", default=None, help="Case counter starting point", required=False)
    args = parser.parse_args()

    method = str(args.method)
    cases = int(args.nr_cases)
    nr_labels = int(args.nr_labels)
    nr_parts = int(args.nr_parts)
    basename = args.basename
    zfill = int(args.zfill)
    count_start = int(args.count_start)

    if method == "v":
        uncertainty_estimator = comp_variance_uncertainty
    elif method == "e":
        uncertainty_estimator = comp_entropy_uncertainty
    elif method == "b":
        uncertainty_estimator = comp_bhattacharyya_uncertainty
    else:
        raise RuntimeError("Unknown uncertainty estimator.")

    comp_uncertainties(args.input, args.output, uncertainty_estimator, basename, cases, nr_labels, nr_parts, zfill, count_start)

    # preprocess_uncertainty_files("/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task004_BrainTumour_guided/uncertainties/ensemble/predictive_entropy")
