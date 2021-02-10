from medseg import utils
import numpy as np
import os
from tqdm import tqdm


def comp_tta_uncertainties(load_dir, save_dir, type):
    filenames = utils.load_filenames(load_dir)
    nr_cases, nr_labels, nr_parts = group_data(filenames)
    print("nr_cases: ", nr_cases)
    print("nr_labels: ", nr_labels)
    print("nr_parts: ", nr_parts)

    for case in tqdm(range(nr_cases+1)):
        for label in range(nr_labels+1):
            predictions = []
            for part in range(nr_parts+1):
                name = load_dir + str(case+1).zfill(4) + "_" + str(label) + "_" + type + "_" + str(part) + ".nii.gz"
                prediction, affine, spacing, header = utils.load_nifty(name)
                predictions.append(prediction.astype(np.float16))
            predictions = np.stack(predictions)
            uncertainty = comp_variance_uncertainty(predictions)
            name = save_dir + str(case+1).zfill(4) + "_" + str(label) + ".nii.gz"
            utils.save_nifty(name, uncertainty, affine, spacing, header)


def group_data(filenames):
    nr_cases = 0
    nr_labels = 0
    nr_parts = 0

    for filename in filenames:
        filename = os.path.basename(filename)
        case_nr = int(filename[:4])
        if nr_cases < case_nr:
            nr_cases = case_nr
        label_nr = int(filename[5:6])
        if nr_labels < label_nr:
            nr_labels = label_nr
        part_nr = int(filename[12:13])
        if nr_parts < part_nr:
            nr_parts = part_nr
    return nr_cases, nr_labels, nr_parts


def comp_variance_uncertainty(predictions):
    predictive_posterior_mean = np.mean(predictions, axis=0)
    uncertainty = np.zeros_like(predictive_posterior_mean)
    for prediction in predictions:
        uncertainty += (prediction - predictive_posterior_mean) ** 2
    uncertainty /= len(predictions)
    return uncertainty


if __name__ == '__main__':
    comp_tta_uncertainties("/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task086_frankfurt2/predictions_with_ensemble/",
                           "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task086_frankfurt2/uncertainties_ensemble_variance/", "part")
