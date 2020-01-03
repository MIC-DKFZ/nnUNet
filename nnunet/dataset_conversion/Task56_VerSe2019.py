import subprocess
from collections import OrderedDict
import SimpleITK as sitk
from multiprocess.pool import Pool
from nnunet.configuration import default_num_threads
from nnunet.dataset_conversion.Task56_Verse_normalize_orientation import normalize_slice_orientation, read_image, \
    save_image, restore_original_slice_orientation
from nnunet.paths import splitted_4d_output_dir
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from medpy import metric
import numpy as np


def load_corr_save(in_folder: str, out_folder: str, filename: str):
    assert filename.endswith(".nii.gz")
    maybe_mkdir_p(out_folder)
    img, header = read_image(join(in_folder, filename))
    img_corr, header_corr = normalize_slice_orientation(img, header)
    # now we save without restoring original slice orientation. We pickle the header for later
    save_image(img_corr, header_corr, join(out_folder, filename))
    save_pickle(header, join(out_folder, filename[:-7] + ".pkl"))

    # just a test to see if we can reproduce the original image
    # img_corr2, header_corr2 = restore_original_slice_orientation(img_corr, header_corr)
    # save_image(img_corr2, header_corr2, join(out_folder, filename[:-7] + "_re.nii.gz"))
    # seems to work


def evaluate_verse_case(sitk_file_ref:str, sitk_file_test:str):
    """
    Only vertebra that are present in the reference will be evaluated
    :param sitk_file_ref:
    :param sitk_file_test:
    :return:
    """
    gt_npy = sitk.GetArrayFromImage(sitk.ReadImage(sitk_file_ref))
    pred_npy = sitk.GetArrayFromImage(sitk.ReadImage(sitk_file_test))
    dice_scores = []
    for label in range(1, 26):
        mask_gt = gt_npy == label
        if np.sum(mask_gt) > 0:
            mask_pred = pred_npy == label
            dc = metric.dc(mask_pred, mask_gt)
        else:
            dc = np.nan
        dice_scores.append(dc)
    return dice_scores


def evaluate_verse_folder(folder_pred, folder_gt, out_json="/home/fabian/verse.json"):
    p = Pool(default_num_threads)
    files_gt_bare = subfiles(folder_gt, join=False)
    assert all([isfile(join(folder_pred, i)) for i in files_gt_bare]), "some files are missing in the predicted folder"
    files_pred = [join(folder_pred, i) for i in files_gt_bare]
    files_gt = [join(folder_gt, i) for i in files_gt_bare]

    results = p.starmap_async(evaluate_verse_case, zip(files_gt, files_pred))

    results = results.get()

    dct = {i: j for i, j in zip(files_gt_bare, results)}

    results_stacked = np.vstack(results)
    results_mean = np.nanmean(results_stacked, 0)
    overall_mean = np.nanmean(results_mean)

    save_json((dct, list(results_mean), overall_mean), out_json)


if __name__ == "__main__":
    base = "/media/fabian/DeepLearningData/VerSe2019"
    base_corrOrient = "/media/fabian/DeepLearningData/VerSe2019_corrOrient"

    # correct orientation
    train_files_base = subfiles(join(base, "train"), join=False, suffix="_seg.nii.gz")
    train_segs = [i[:-len("_seg.nii.gz")] + "_seg.nii.gz" for i in train_files_base]
    train_data = [i[:-len("_seg.nii.gz")] + ".nii.gz" for i in train_files_base]
    test_files_base = [i[:-len(".nii.gz")] for i in subfiles(join(base, "test"), join=False, suffix=".nii.gz")]
    test_data = [i + ".nii.gz" for i in test_files_base]

    for i in train_segs + train_data:
        load_corr_save(join(base, "train"), join(base_corrOrient, "train"), i)
    for i in test_data:
        load_corr_save(join(base, "test"), join(base_corrOrient, "test"), i)

    train_files_base = subfiles(join(base_corrOrient, "train"), join=True, suffix="_seg.nii.gz")
    train_segs = [i[:-len("_seg.nii.gz")] + "_seg.nii.gz" for i in train_files_base]
    train_data = [i[:-len("_seg.nii.gz")] + ".nii.gz" for i in train_files_base]
    test_files_base = [i[:-len(".nii.gz")] for i in subfiles(join(base_corrOrient, "test"), join=True, suffix=".nii.gz")]
    test_data = [i + ".nii.gz" for i in test_files_base]


    task_id = 56
    task_name = "VerSe"

    foldername = "Task%02.0d_%s" % (task_id, task_name)

    out_base = join(splitted_4d_output_dir, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    train_patient_names = [i[:-len("_seg.nii.gz")] for i in subfiles(join(base_corrOrient, "train"), join=False, suffix="_seg.nii.gz")]
    for p in train_patient_names:
        curr = join(base_corrOrient, "train")
        label_file = join(curr, p + "_seg.nii.gz")
        image_file = join(curr, p + ".nii.gz")
        shutil.copy(image_file, join(imagestr, p + "_0000.nii.gz"))
        shutil.copy(label_file, join(labelstr, p + ".nii.gz"))

    test_patient_names = [i[:-7] for i in subfiles(join(base_corrOrient, "test"), join=False, suffix=".nii.gz")]
    for p in test_patient_names:
        curr = join(base_corrOrient, "test")
        image_file = join(curr, p + ".nii.gz")
        shutil.copy(image_file, join(imagests, p + "_0000.nii.gz"))


    json_dict = OrderedDict()
    json_dict['name'] = "VerSe2019"
    json_dict['description'] = "VerSe2019"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see challenge website"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = {i: str(i) for i in range(26)}

    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numTest'] = len(test_patient_names)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1], "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1]} for i in
                             train_patient_names]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i.split("/")[-1] for i in test_patient_names]

    save_json(json_dict, os.path.join(out_base, "dataset.json"))


    # run this part of the code once training is done
    folder_gt = "/media/fabian/My Book/MedicalDecathlon/nnUNet_raw_splitted/Task56_VerSe/labelsTr"

    folder_pred = "/home/fabian/drives/datasets/results/nnUNet/3d_fullres/Task56_VerSe/nnUNetTrainerV2__nnUNetPlansv2.1/cv_niftis_raw"
    out_json = "/home/fabian/Task56_VerSe_3d_fullres_summary.json"
    evaluate_verse_folder(folder_pred, folder_gt, out_json)

    folder_pred = "/home/fabian/drives/datasets/results/nnUNet/3d_lowres/Task56_VerSe/nnUNetTrainerV2__nnUNetPlansv2.1/cv_niftis_raw"
    out_json = "/home/fabian/Task56_VerSe_3d_lowres_summary.json"
    evaluate_verse_folder(folder_pred, folder_gt, out_json)

    folder_pred = "/home/fabian/drives/datasets/results/nnUNet/3d_cascade_fullres/Task56_VerSe/nnUNetTrainerV2CascadeFullRes__nnUNetPlansv2.1/cv_niftis_raw"
    out_json = "/home/fabian/Task56_VerSe_3d_cascade_fullres_summary.json"
    evaluate_verse_folder(folder_pred, folder_gt, out_json)

