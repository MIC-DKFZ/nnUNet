from i3Deep import utils
import numpy as np
import os
from tqdm import tqdm
import argparse
import sys
from scipy.special import softmax
import multiprocessing as mp
from functools import partial
from scipy.stats import entropy


def comp_uncertainties(load_dir, save_dir, uncertainty_estimator, basename=None, cases=None, nr_labels=None, nr_parts=None, zfill=4, count_start=None, type="part", merge_uncertainties=False, class_dices=None, class_uncertainty=False, parallel=False):
    load_dir = utils.fix_path(load_dir)
    save_dir = utils.fix_path(save_dir)
    filenames = utils.load_filenames(load_dir)
    if cases is None:
        cases, nr_labels, nr_parts = group_data(filenames)
    else:
        cases = list(range(count_start, cases+count_start, 1))
    print(cases)
    print("nr_cases: ", len(cases))
    print("nr_labels: ", nr_labels)
    print("nr_parts: ", nr_parts)

    weights = None
    if class_dices is not None:
        weights = [1 - dice for dice in class_dices]
        weights = softmax(weights)
        print("weights: ", weights)

    if not parallel:
        for case in tqdm(cases):
            comp_uncertainties_single_case(case, load_dir, save_dir, weights, uncertainty_estimator, basename, nr_labels, nr_parts, zfill, count_start, type, merge_uncertainties, class_dices, class_uncertainty)
    else:
        pool = mp.Pool(processes=8)  # 8
        pool.map(partial(comp_uncertainties_single_case, load_dir=load_dir, save_dir=save_dir, weights=weights,
                         uncertainty_estimator=uncertainty_estimator, basename=basename, nr_labels=nr_labels,
                         nr_parts=nr_parts, zfill=zfill, count_start=count_start, type=type, merge_uncertainties=merge_uncertainties,
                         class_dices=class_dices, class_uncertainty=class_uncertainty), cases)
        pool.close()
        pool.join()

def comp_uncertainties_single_case(case, load_dir, save_dir, weights, uncertainty_estimator, basename, nr_labels, nr_parts, zfill, count_start, type, merge_uncertainties, class_dices, class_uncertainty):
    uncertainty_classes, prediction_classes = [], []
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
        if not class_uncertainty:
            uncertainty = uncertainty_estimator(predictions)
        if not merge_uncertainties and not class_uncertainty:
            name = create_name(save_dir, basename, case, label, zfill, False)
            utils.save_nifty(name, uncertainty, affine, spacing, header)
        elif not class_uncertainty:
            uncertainty_classes.append(uncertainty)
        # predictions = np.mean(predictions, axis=0)
        prediction_classes.append(predictions)
    if class_uncertainty:
        prediction_classes = np.stack(prediction_classes, axis=0)
        uncertainty = uncertainty_estimator(prediction_classes)
        uncertainty = np.nan_to_num(uncertainty)
        name = create_name(save_dir, basename, case, None, zfill, True)
        utils.save_nifty(name, uncertainty, affine, spacing, header)
    if merge_uncertainties and not class_uncertainty:
        uncertainty_classes = uncertainty_classes[1:]  # Remove background uncertainty
        if class_dices is None:
            uncertainty_classes = np.mean(uncertainty_classes, axis=0)
        else:
            uncertainty_classes = np.asarray(uncertainty_classes)
            uncertainty_classes = [uncertainty_classes[i] * weights[i] for i in range(len(uncertainty_classes))]
            uncertainty_classes = np.sum(uncertainty_classes, axis=0)
        name = create_name(save_dir, basename, case, None, zfill, True)
        utils.save_nifty(name, uncertainty_classes, affine, spacing, header)
    # print("Finished case: ", case)


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
    uncertainty = utils.normalize(uncertainty, x_min=0, x_max=0.25)
    return uncertainty


def comp_class_variance_uncertainty(predictions):
    uncertainty = []
    for prediction in predictions:
        uncertainty.append(comp_variance_uncertainty(prediction))
    uncertainty = np.mean(uncertainty, axis=0)
    return uncertainty


def comp_entropy_uncertainty(predictions):
    # predictions = predictions.transpose(1, 2, 3, 0)
    # predictions = softmax(predictions, axis=3)
    # predictions = predictions.transpose(3, 0, 1, 2)
    # # uncertainty = np.zeros(predictions.shape[1:])
    # # for prediction in predictions:
    # #     uncertainty += prediction * np.log2(prediction + sys.float_info.epsilon)
    # # uncertainty = -1 * uncertainty / np.log2(predictions.shape[0])
    # uncertainty = entropy(predictions, base=2, axis=0) / np.log2(predictions.shape[0])
    predictions = np.mean(predictions, axis=0)
    predictions = np.stack([predictions, 1-predictions], axis=0)
    uncertainty = entropy(predictions, base=2, axis=0) / np.log2(predictions.shape[0])
    # uncertainty = np.nan_to_num(uncertainty)
    # uncertainty = 1 - uncertainty
    return uncertainty

def comp_class_entropy_uncertainty(predictions):
    predictions = np.mean(predictions, axis=1)
    uncertainty = entropy(predictions, base=2, axis=0) / np.log2(predictions.shape[0])
    # uncertainty = np.nan_to_num(uncertainty)
    return uncertainty


def comp_bhattacharyya_uncertainty(predictions):
    # predictions = predictions.transpose(1, 2, 3, 0)
    # predictions = softmax(predictions, axis=3)
    # predictions = predictions.transpose(3, 0, 1, 2)
    uncertainty = np.power(np.prod(predictions, axis=0), 1.0/float(predictions.shape[0]))
    uncertainty += np.power(np.prod(1 - predictions, axis=0), 1.0/float(predictions.shape[0]))
    uncertainty = 1 - uncertainty
    return uncertainty


def comp_class_bhattacharyya_uncertainty(predictions):
    uncertainty = np.zeros(predictions.shape[2:])
    for prediction in predictions:
        uncertainty += np.power(np.prod(prediction, axis=0), 1.0/float(predictions.shape[0]))
    uncertainty = 1 - uncertainty
    return uncertainty


def comp_confidence(predictions):
    predictions = np.mean(predictions, axis=1)
    return 1 - np.max(predictions, axis=0)


def comp_prediction(predictions):
    predictions = np.mean(predictions, axis=1)
    predictions = np.argmax(predictions, axis=0)
    return predictions


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
    # parser.add_argument("-o", "--output", help="Output folder", required=True)
    parser.add_argument("-m", "--method", help="Variance (v), entropy (e) or bhattacharyya (b)", required=True)
    parser.add_argument("--class_dices", action="store_true", default=False, help="Use class dices?", required=False)
    parser.add_argument("--parallel", action="store_true", default=False, help="Set the version", required=False)

    # Stupid arguments
    parser.add_argument("-nr_c", "--nr_cases", default=None, help="Number of cases", required=False)  # 50
    parser.add_argument("-nr_l", "--nr_labels", default=None, help="Number of labels", required=False)  # 4
    parser.add_argument("-nr_p", "--nr_parts", default=None, help="Number of parts", required=False)  # 5
    parser.add_argument("-basename", default=None, help="Basename", required=False)  # BRATS
    parser.add_argument("-zfill", default=4, help="Number of leading zeros for case numeration", required=False)  # 3
    parser.add_argument("-count_start", default=None, help="Case counter starting point", required=False)  # 101
    args = parser.parse_args()

    merge_uncertainties = True
    if args.class_dices:
        # class_dices = (0.83, 0.59, 0.73)
        # class_dices = (0.78, 0.57, 0.79)  # Task002_BrainTumour_guided test
        class_dices = (0.75, 0.28)  # Task008_Pancreas_guided val
    else:
        class_dices = None

    method = str(args.method)
    if args.nr_cases is not None:
        cases = int(args.nr_cases)
        nr_labels = int(args.nr_labels)
        nr_parts = int(args.nr_parts)
        basename = args.basename
        zfill = int(args.zfill)
        count_start = int(args.count_start)
    else:
        cases = None
        nr_labels = None
        nr_parts = None
        basename = None
        zfill = 4
        count_start = None

    input_path = args.input + "/probabilities"
    output_path = args.input

    if method == "v":
        uncertainty_estimator = comp_variance_uncertainty
        class_uncertainty = False
        output_path += "/predictive_variance"
    elif method == "e":
        uncertainty_estimator = comp_entropy_uncertainty
        class_uncertainty = False
        output_path += "/predictive_entropy"
    elif method == "b":
        uncertainty_estimator = comp_bhattacharyya_uncertainty
        class_uncertainty = False
        output_path += "/bhattacharyya_coefficient"
    elif method == "conf":
        uncertainty_estimator = comp_confidence
        class_uncertainty = True
        output_path += "/confidence"
    elif method == "pred":
        uncertainty_estimator = comp_prediction
        class_uncertainty = True
        output_path += "/predictions"
    elif method == "class_e":
        uncertainty_estimator = comp_class_entropy_uncertainty
        class_uncertainty = True
        output_path += "/predictive_entropy"
    elif method == "class_b":
        uncertainty_estimator = comp_class_bhattacharyya_uncertainty
        class_uncertainty = True
        output_path += "/bhattacharyya_coefficient"
    elif method == "class_v":
        uncertainty_estimator = comp_class_variance_uncertainty
        class_uncertainty = True
        output_path += "/predictive_variance"
    else:
        raise RuntimeError("Unknown uncertainty estimator.")

    comp_uncertainties(input_path, output_path, uncertainty_estimator, basename, cases, nr_labels, nr_parts, zfill, count_start, merge_uncertainties=merge_uncertainties, class_dices=class_dices, class_uncertainty=class_uncertainty, parallel=args.parallel)

    # preprocess_uncertainty_files("/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task004_BrainTumour_guided/uncertainties/ensemble/predictive_entropy")
