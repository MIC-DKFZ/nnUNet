from i3Deep import utils
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import skimage.measure


def eval_all_missed_compontents(gt_path, prediction_path, save_path):
    prediction_filenames = utils.load_filenames(prediction_path)
    gt_filenames = utils.load_filenames(gt_path)

    all_detected_components_size, all_missed_components_size = [], []
    for i in tqdm(range(len(prediction_filenames))):
        prediction = utils.load_nifty(prediction_filenames[i])[0]
        gt = utils.load_nifty(gt_filenames[i])[0]
        detected_components_size, missed_components_size = eval_missed_componets(gt, prediction)
        all_detected_components_size.append(detected_components_size)
        all_missed_components_size.append(missed_components_size)
    with open(save_path + "missed_components.pkl", 'wb') as handle:
        pickle.dump([all_detected_components_size, all_missed_components_size], handle, protocol=pickle.HIGHEST_PROTOCOL)


def eval_missed_componets(gt, prediction):
    labels, unique, count = comp_components(gt)
    detected_components_size, missed_components_size = [], []
    for i, label in enumerate(unique):
        label_mask = labels == label
        is_missed = np.sum(prediction[label_mask]) == 0
        if is_missed:
            missed_components_size.append(count[i])
        else:
            detected_components_size.append(count[i])
    return detected_components_size, missed_components_size


def comp_components(gt):
    labels, num = skimage.measure.label(gt, return_num=True)
    unique, count = np.unique(labels, return_counts=True)
    unique = unique[1:]
    count = count[1:]
    return labels, unique, count


if __name__ == '__main__':
    gt_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task070_guided_all_public_ggo/refinement_test/labels/"
    prediction_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task070_guided_all_public_ggo/refinement_test/refined_predictions/my_method/"
    save_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task070_guided_all_public_ggo/refinement_test/"

    eval_all_missed_compontents(gt_path, prediction_path, save_path)