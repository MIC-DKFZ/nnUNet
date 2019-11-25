import shutil
from copy import deepcopy

from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import SimpleITK as sitk
from multiprocessing import Pool
from medpy.metric import dc
import numpy as np
from nnunet.paths import network_training_output_dir


def compute_dice_scores(ref: str, pred: str):
    ref = sitk.GetArrayFromImage(sitk.ReadImage(ref))
    pred = sitk.GetArrayFromImage(sitk.ReadImage(pred))
    kidney_mask_ref = ref > 0
    kidney_mask_pred = pred > 0
    if np.sum(kidney_mask_pred) == 0 and kidney_mask_ref.sum() == 0:
        kidney_dice = np.nan
    else:
        kidney_dice = dc(kidney_mask_pred, kidney_mask_ref)

    tumor_mask_ref = ref == 2
    tumor_mask_pred = pred == 2
    if np.sum(tumor_mask_ref) == 0 and tumor_mask_pred.sum() == 0:
        tumor_dice = np.nan
    else:
        tumor_dice = dc(tumor_mask_ref, tumor_mask_pred)

    geometric_mean = np.mean((kidney_dice, tumor_dice))
    return kidney_dice, tumor_dice, geometric_mean


def evaluate_folder(folder_gt: str, folder_pred: str):
    p = Pool(8)
    niftis = subfiles(folder_gt, suffix=".nii.gz", join=False)
    images_gt = [join(folder_gt, i) for i in niftis]
    images_pred = [join(folder_pred, i) for i in niftis]
    results = p.starmap(compute_dice_scores, zip(images_gt, images_pred))
    p.close()
    p.join()

    with open(join(folder_pred, "results.csv"), 'w') as f:
        for i, ni in enumerate(niftis):
            f.write("%s,%0.4f,%0.4f,%0.4f\n" % (ni, *results[i]))


def copy_npz_fom_valsets():
    '''
    this is preparation for ensembling
    :return:
    '''
    base = join(network_training_output_dir, "3d_lowres/Task48_KiTS_clean")
    folders = ['nnUNetTrainerNewCandidate23_FabiansPreActResNet__nnUNetPlans',
               'nnUNetTrainerNewCandidate23_FabiansResNet__nnUNetPlans',
               'nnUNetTrainerNewCandidate23__nnUNetPlans']
    for f in folders:
        out = join(base, f, 'crossval_npz')
        maybe_mkdir_p(out)
        shutil.copy(join(base, f, 'plans.pkl'), out)
        for fold in range(5):
            cur = join(base, f, 'fold_%d' % fold, 'validation_raw')
            npz_files = subfiles(cur, suffix='.npz', join=False)
            pkl_files = [i[:-3] + 'pkl' for i in npz_files]
            assert all([isfile(join(cur, i)) for i in pkl_files])
            for n in npz_files:
                corresponding_pkl = n[:-3] + 'pkl'
                shutil.copy(join(cur, n), out)
                shutil.copy(join(cur, corresponding_pkl), out)


def ensemble(experiments=('nnUNetTrainerNewCandidate23_FabiansPreActResNet__nnUNetPlans',
               'nnUNetTrainerNewCandidate23_FabiansResNet__nnUNetPlans'), out_dir="/media/fabian/Results/nnUNet/3d_lowres/Task48_KiTS_clean/ensemble_preactres_and_res"):
    from nnunet.inference.ensemble_predictions import merge
    folders = [join(network_training_output_dir, "3d_lowres/Task48_KiTS_clean", i, 'crossval_npz') for i in experiments]
    merge(folders, out_dir, 8)


def prepare_submission(fld='/home/fabian/datasets_fabian/predicted_KiTS_nnUNetTrainerNewCandidate23_FabiansResNet',
                       out='/home/fabian/datasets_fabian/predicted_KiTS_nnUNetTrainerNewCandidate23_FabiansResNet_submitted'):
    nii = subfiles(fld, join=False, suffix='.nii.gz')
    maybe_mkdir_p(out)
    for n in nii:
        outfname = n.replace('case', 'prediction')
        shutil.copy(join(fld, n), join(out, outfname))


def pretent_to_be_nnUNetTrainer(base, folds=(0, 1, 2, 3, 4)):
    """
    changes best checkpoint pickle nnunettrainer class name to nnUNetTrainer
    :param experiments:
    :return:
    """
    for fold in folds:
        cur = join(base, "fold_%d" % fold)
        pkl_file = join(cur, 'model_best.model.pkl')
        a = load_pickle(pkl_file)
        a['name_old'] = deepcopy(a['name'])
        a['name'] = 'nnUNetTrainer'
        save_pickle(a, pkl_file)


def reset_trainerName(base, folds=(0, 1, 2, 3, 4)):
    for fold in folds:
        cur = join(base, "fold_%d" % fold)
        pkl_file = join(cur, 'model_best.model.pkl')
        a = load_pickle(pkl_file)
        a['name'] = a['name_old']
        del a['name_old']
        save_pickle(a, pkl_file)


def nnUNetTrainer_these(experiments=('nnUNetTrainerNewCandidate23_FabiansPreActResNet__nnUNetPlans',
               'nnUNetTrainerNewCandidate23_FabiansResNet__nnUNetPlans',
               'nnUNetTrainerNewCandidate23__nnUNetPlans')):
    """
    changes best checkpoint pickle nnunettrainer class name to nnUNetTrainer
    :param experiments:
    :return:
    """
    base = join(network_training_output_dir, "3d_lowres/Task48_KiTS_clean")
    for exp in experiments:
        cur = join(base, exp)
        pretent_to_be_nnUNetTrainer(cur)


def reset_trainerName_these(experiments=('nnUNetTrainerNewCandidate23_FabiansPreActResNet__nnUNetPlans',
               'nnUNetTrainerNewCandidate23_FabiansResNet__nnUNetPlans',
               'nnUNetTrainerNewCandidate23__nnUNetPlans')):
    """
    changes best checkpoint pickle nnunettrainer class name to nnUNetTrainer
    :param experiments:
    :return:
    """
    base = join(network_training_output_dir, "3d_lowres/Task48_KiTS_clean")
    for exp in experiments:
        cur = join(base, exp)
        reset_trainerName(cur)


if __name__ == "__main__":
    base = "/media/fabian/My Book/datasets/KiTS2019_Challenge/kits19/data"
    out = "/media/fabian/My Book/MedicalDecathlon/nnUNet_raw_splitted/Task40_KiTS"
    cases = subdirs(base, join=False)

    maybe_mkdir_p(out)
    maybe_mkdir_p(join(out, "imagesTr"))
    maybe_mkdir_p(join(out, "imagesTs"))
    maybe_mkdir_p(join(out, "labelsTr"))

    for c in cases:
        case_id = int(c.split("_")[-1])
        if case_id < 210:
            shutil.copy(join(base, c, "imaging.nii.gz"), join(out, "imagesTr", c + "_0000.nii.gz"))
            shutil.copy(join(base, c, "segmentation.nii.gz"), join(out, "labelsTr", c + ".nii.gz"))
        else:
            shutil.copy(join(base, c, "imaging.nii.gz"), join(out, "imagesTs", c + "_0000.nii.gz"))

    json_dict = {}
    json_dict['name'] = "KiTS"
    json_dict['description'] = "kidney and kidney tumor segmentation"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "KiTS data for nnunet"
    json_dict['licence'] = ""
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "Kidney",
        "2": "Tumor"
    }
    json_dict['numTraining'] = len(cases)
    json_dict['numTest'] = 0
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                             cases]
    json_dict['test'] = []

    save_json(json_dict, os.path.join(out, "dataset.json"))

