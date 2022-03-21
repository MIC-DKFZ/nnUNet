#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import shutil
import subprocess

import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.utilities.file_and_folder_operations_winos import * # Join path by slash on windows system.

from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.paths import nnUNet_raw_data
from nnunet.paths import preprocessing_output_dir
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name


def increase_batch_size(plans_file: str, save_as: str, bs_factor: int):
    a = load_pickle(plans_file)
    stages = list(a['plans_per_stage'].keys())
    for s in stages:
        a['plans_per_stage'][s]['batch_size'] *= bs_factor
    save_pickle(a, save_as)


def prepare_submission(folder_in, folder_out):
    nii = subfiles(folder_in, suffix='.gz', join=False)
    maybe_mkdir_p(folder_out)
    for n in nii:
        i = n.split('-')[-1][:-10]
        shutil.copy(join(folder_in, n), join(folder_out, i + '.nii.gz'))


def get_ids_from_folder(folder):
    cts = subfiles(folder, suffix='_ct.nii.gz', join=False)
    ids = []
    for c in cts:
        ids.append(c.split('-')[-1][:-10])
    return ids


def postprocess_submission(folder_ct, folder_pred, folder_postprocessed, bbox_distance_to_seg_in_cm=7.5):
    """
    segment with lung mask, get bbox from that, use bbox to remove predictions in background

    WE EXPERIMENTED WITH THAT ON THE VALIDATION SET AND FOUND THAT IT DOESN'T DO ANYTHING. NOT USED FOR TEST SET
    """
    # pip install git+https://github.com/JoHof/lungmask
    cts = subfiles(folder_ct, suffix='_ct.nii.gz', join=False)
    output_files = [i[:-10] + '_lungmask.nii.gz' for i in cts]

    # run lungmask on everything
    for i, o in zip(cts, output_files):
        if not isfile(join(folder_ct, o)):
            subprocess.call(['lungmask', join(folder_ct, i), join(folder_ct, o), '--modelname', 'R231CovidWeb'])

    if not isdir(folder_postprocessed):
        maybe_mkdir_p(folder_postprocessed)

    ids = get_ids_from_folder(folder_ct)
    for i in ids:
        # find lungmask
        lungmask_file = join(folder_ct, 'volume-covid19-A-' + i + '_lungmask.nii.gz')
        if not isfile(lungmask_file):
            raise RuntimeError('missing lung')
        seg_file = join(folder_pred, 'volume-covid19-A-' + i + '_ct.nii.gz')
        if not isfile(seg_file):
            raise RuntimeError('missing seg')

        lung_mask = sitk.GetArrayFromImage(sitk.ReadImage(lungmask_file))
        seg_itk = sitk.ReadImage(seg_file)
        seg = sitk.GetArrayFromImage(seg_itk)

        where = np.argwhere(lung_mask != 0)
        bbox = [
            [min(where[:, 0]), max(where[:, 0])],
            [min(where[:, 1]), max(where[:, 1])],
            [min(where[:, 2]), max(where[:, 2])],
        ]

        spacing = np.array(seg_itk.GetSpacing())[::-1]
        # print(bbox)
        for dim in range(3):
            sp = spacing[dim]
            voxels_extend = max(int(np.ceil(bbox_distance_to_seg_in_cm / sp)), 1)
            bbox[dim][0] = max(0, bbox[dim][0] - voxels_extend)
            bbox[dim][1] = min(seg.shape[dim], bbox[dim][1] + voxels_extend)
        # print(bbox)

        seg_old = np.copy(seg)
        seg[0:bbox[0][0], :, :] = 0
        seg[bbox[0][1]:, :, :] = 0
        seg[:, 0:bbox[1][0], :] = 0
        seg[:, bbox[1][1]:, :] = 0
        seg[:, :, 0:bbox[2][0]] = 0
        seg[:, :, bbox[2][1]:] = 0
        if np.any(seg_old != seg):
            print('changed seg', i)
            argwhere = np.argwhere(seg != seg_old)
            print(argwhere[np.random.choice(len(argwhere), 10)])

        seg_corr = sitk.GetImageFromArray(seg)
        seg_corr.CopyInformation(seg_itk)
        sitk.WriteImage(seg_corr, join(folder_postprocessed, 'volume-covid19-A-' + i + '_ct.nii.gz'))


def manually_set_configurations():
    """
    ALSO NOT USED!
    :return:
    """
    task115_dir = join(preprocessing_output_dir, convert_id_to_task_name(115))

    ## larger patch size

    # task115 3d_fullres default is:
    """
    {'batch_size': 2, 
    'num_pool_per_axis': [2, 6, 6], 
    'patch_size': array([ 28, 256, 256]), 
    'median_patient_size_in_voxels': array([ 62, 512, 512]), 
    'current_spacing': array([5.        , 0.74199998, 0.74199998]), 
    'original_spacing': array([5.        , 0.74199998, 0.74199998]), 
    'do_dummy_2D_data_aug': True, 
    'pool_op_kernel_sizes': [[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2], [1, 2, 2]], 
    'conv_kernel_sizes': [[1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]}
    """
    plans = load_pickle(join(task115_dir, 'nnUNetPlansv2.1_plans_3D.pkl'))
    fullres_stage = plans['plans_per_stage'][1]
    fullres_stage['patch_size'] = np.array([ 64, 320, 320])
    fullres_stage['num_pool_per_axis'] = [4, 6, 6]
    fullres_stage['pool_op_kernel_sizes'] = [[1, 2, 2],
                                            [1, 2, 2],
                                            [2, 2, 2],
                                            [2, 2, 2],
                                            [2, 2, 2],
                                            [2, 2, 2]]
    fullres_stage['conv_kernel_sizes'] = [[1, 3, 3],
                                        [1, 3, 3],
                                        [3, 3, 3],
                                        [3, 3, 3],
                                        [3, 3, 3],
                                        [3, 3, 3],
                                        [3, 3, 3]]

    save_pickle(plans, join(task115_dir, 'nnUNetPlansv2.1_custom_plans_3D.pkl'))

    ## larger batch size
    # (default for all 3d trainings is batch size 2)
    increase_batch_size(join(task115_dir, 'nnUNetPlansv2.1_plans_3D.pkl'), join(task115_dir, 'nnUNetPlansv2.1_bs3x_plans_3D.pkl'), 3)
    increase_batch_size(join(task115_dir, 'nnUNetPlansv2.1_plans_3D.pkl'), join(task115_dir, 'nnUNetPlansv2.1_bs5x_plans_3D.pkl'), 5)

    # residual unet
    """
    default is:
    Out[7]: 
    {'batch_size': 2,
     'num_pool_per_axis': [2, 6, 5],
     'patch_size': array([ 28, 256, 224]),
     'median_patient_size_in_voxels': array([ 62, 512, 512]),
     'current_spacing': array([5.        , 0.74199998, 0.74199998]),
     'original_spacing': array([5.        , 0.74199998, 0.74199998]),
     'do_dummy_2D_data_aug': True,
     'pool_op_kernel_sizes': [[1, 1, 1],
      [1, 2, 2],
      [1, 2, 2],
      [2, 2, 2],
      [2, 2, 2],
      [1, 2, 2],
      [1, 2, 1]],
     'conv_kernel_sizes': [[1, 3, 3],
      [1, 3, 3],
      [3, 3, 3],
      [3, 3, 3],
      [3, 3, 3],
      [3, 3, 3],
      [3, 3, 3]],
     'num_blocks_encoder': (1, 2, 3, 4, 4, 4, 4),
     'num_blocks_decoder': (1, 1, 1, 1, 1, 1)}
    """
    plans = load_pickle(join(task115_dir, 'nnUNetPlans_FabiansResUNet_v2.1_plans_3D.pkl'))
    fullres_stage = plans['plans_per_stage'][1]
    fullres_stage['patch_size'] = np.array([ 56, 256, 256])
    fullres_stage['num_pool_per_axis'] = [3, 6, 6]
    fullres_stage['pool_op_kernel_sizes'] = [[1, 1, 1],
                                             [1, 2, 2],
                                             [1, 2, 2],
                                            [2, 2, 2],
                                            [2, 2, 2],
                                            [2, 2, 2],
                                            [1, 2, 2]]
    fullres_stage['conv_kernel_sizes'] = [[1, 3, 3],
                                        [1, 3, 3],
                                        [3, 3, 3],
                                        [3, 3, 3],
                                        [3, 3, 3],
                                        [3, 3, 3],
                                        [3, 3, 3]]
    save_pickle(plans, join(task115_dir, 'nnUNetPlans_FabiansResUNet_v2.1_custom_plans_3D.pkl'))


def check_same(img1: str, img2: str):
    """
    checking initial vs corrected dataset
    :param img1:
    :param img2:
    :return:
    """
    img1 = sitk.GetArrayFromImage(sitk.ReadImage(img1))
    img2 = sitk.GetArrayFromImage(sitk.ReadImage(img2))
    if not np.all([i==j for i, j in zip(img1.shape, img2.shape)]):
        print('shape')
        return False
    else:
        same = np.all(img1==img2)
        if same: return True
        else:
            diffs = np.argwhere(img1!=img2)
            print('content in', diffs.shape[0], 'voxels')
            print('random disagreements:')
            print(diffs[np.random.choice(len(diffs), min(3, diffs.shape[0]), replace=False)])
            return False


def check_dataset_same(dataset_old='/home/fabian/Downloads/COVID-19-20/Train',
                       dataset_new='/home/fabian/data/COVID-19-20_officialCorrected/COVID-19-20_v2/Train'):
    """
    :param dataset_old:
    :param dataset_new:
    :return:
    """
    cases = [i[:-10] for i in subfiles(dataset_new, suffix='_ct.nii.gz', join=False)]
    for c in cases:
        data_file = join(dataset_old, c + '_ct_corrDouble.nii.gz')
        corrected_double = False
        if not isfile(data_file):
            data_file = join(dataset_old, c+'_ct.nii.gz')
        else:
            corrected_double = True
        data_file_new = join(dataset_new, c+'_ct.nii.gz')

        same = check_same(data_file, data_file_new)
        if not same: print('data differs in case', c, '\n')

        seg_file = join(dataset_old, c + '_seg_corrDouble_corrected.nii.gz')
        if not isfile(seg_file):
            seg_file = join(dataset_old, c + '_seg_corrected_auto.nii.gz')
            if isfile(seg_file):
                assert ~corrected_double
            else:
                seg_file = join(dataset_old, c + '_seg_corrected.nii.gz')
                if isfile(seg_file):
                    assert ~corrected_double
                else:
                    seg_file = join(dataset_old, c + '_seg_corrDouble.nii.gz')
                    if isfile(seg_file):
                        assert ~corrected_double
                    else:
                        seg_file = join(dataset_old, c + '_seg.nii.gz')
        seg_file_new = join(dataset_new, c + '_seg.nii.gz')
        same = check_same(seg_file, seg_file_new)
        if not same: print('seg differs in case', c, '\n')


if __name__ == '__main__':
    # this is the folder containing the data as downloaded from https://covid-segmentation.grand-challenge.org/COVID-19-20/
    # (zip file was decompressed!)
    downloaded_data_dir = '/home/fabian/data/COVID-19-20_officialCorrected/COVID-19-20_v2/'

    task_name = "Task115_COVIDSegChallenge"

    target_base = join(nnUNet_raw_data, task_name)

    target_imagesTr = join(target_base, "imagesTr")
    target_imagesVal = join(target_base, "imagesVal")
    target_labelsTr = join(target_base, "labelsTr")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_imagesVal)
    maybe_mkdir_p(target_labelsTr)

    train_orig = join(downloaded_data_dir, "Train")

    # convert training set
    cases = [i[:-10] for i in subfiles(train_orig, suffix='_ct.nii.gz', join=False)]
    for c in cases:
        data_file = join(train_orig, c+'_ct.nii.gz')

        # before there was the official corrected dataset we did some corrections of our own. These corrections were
        # dropped when the official dataset was revised.
        seg_file = join(train_orig, c + '_seg_corrected.nii.gz')
        if not isfile(seg_file):
            seg_file = join(train_orig, c + '_seg.nii.gz')

        shutil.copy(data_file, join(target_imagesTr, c + "_0000.nii.gz"))
        shutil.copy(seg_file, join(target_labelsTr, c + '.nii.gz'))

    val_orig = join(downloaded_data_dir, "Validation")
    cases = [i[:-10] for i in subfiles(val_orig, suffix='_ct.nii.gz', join=False)]
    for c in cases:
        data_file = join(val_orig, c + '_ct.nii.gz')

        shutil.copy(data_file, join(target_imagesVal, c + "_0000.nii.gz"))

    generate_dataset_json(
        join(target_base, 'dataset.json'),
        target_imagesTr,
        None,
        ("CT", ),
        {0: 'background', 1: 'covid'},
        task_name,
        dataset_reference='https://covid-segmentation.grand-challenge.org/COVID-19-20/'
    )

    # performance summary (train set 5-fold cross-validation)

    # baselines
    # 3d_fullres nnUNetTrainerV2__nnUNetPlans_v2.1						            0.7441
    # 3d_lowres nnUNetTrainerV2__nnUNetPlans_v2.1						            0.745

    # models used for test set prediction
    # 3d_fullres nnUNetTrainerV2_ResencUNet_DA3__nnUNetPlans_FabiansResUNet_v2.1	0.7543
    # 3d_fullres nnUNetTrainerV2_ResencUNet__nnUNetPlans_FabiansResUNet_v2.1		0.7527
    # 3d_lowres nnUNetTrainerV2_ResencUNet_DA3_BN__nnUNetPlans_FabiansResUNet_v2.1	0.7513
    # 3d_fullres nnUNetTrainerV2_DA3_BN__nnUNetPlans_v2.1					        0.7498
    # 3d_fullres nnUNetTrainerV2_DA3__nnUNetPlans_v2.1					            0.7532

    # Test set prediction
    # nnUNet_predict -i COVID-19-20_TestSet -o covid_testset_predictions/3d_fullres/nnUNetTrainerV2_ResencUNet_DA3__nnUNetPlans_FabiansResUNet_v2.1 -tr nnUNetTrainerV2_ResencUNet_DA3 -p nnUNetPlans_FabiansResUNet_v2.1 -m 3d_fullres -f 0 1 2 3 4 5 6 7 8 9 -t 115 -z
    # nnUNet_predict -i COVID-19-20_TestSet -o covid_testset_predictions/3d_fullres/nnUNetTrainerV2_ResencUNet__nnUNetPlans_FabiansResUNet_v2.1 -tr nnUNetTrainerV2_ResencUNet -p nnUNetPlans_FabiansResUNet_v2.1 -m 3d_fullres -f 0 1 2 3 4 5 6 7 8 9 -t 115 -z
    # nnUNet_predict -i COVID-19-20_TestSet -o covid_testset_predictions/3d_lowres/nnUNetTrainerV2_ResencUNet_DA3_BN__nnUNetPlans_FabiansResUNet_v2.1 -tr nnUNetTrainerV2_ResencUNet_DA3_BN -p nnUNetPlans_FabiansResUNet_v2.1 -m 3d_lowres -f 0 1 2 3 4 5 6 7 8 9 -t 115 -z
    # nnUNet_predict -i COVID-19-20_TestSet -o covid_testset_predictions/3d_fullres/nnUNetTrainerV2_DA3_BN__nnUNetPlans_v2.1 -tr nnUNetTrainerV2_DA3_BN -m 3d_fullres -f 0 1 2 3 4 5 6 7 8 9 -t 115 -z
    # nnUNet_predict -i COVID-19-20_TestSet -o covid_testset_predictions/3d_fullres/nnUNetTrainerV2_DA3__nnUNetPlans_v2.1 -tr nnUNetTrainerV2_DA3 -m 3d_fullres -f 0 1 2 3 4 5 6 7 8 9 -t 115 -z

    # nnUNet_ensemble -f 3d_lowres/nnUNetTrainerV2_ResencUNet_DA3_BN__nnUNetPlans_FabiansResUNet_v2.1/ 3d_fullres/nnUNetTrainerV2_ResencUNet__nnUNetPlans_FabiansResUNet_v2.1/ 3d_fullres/nnUNetTrainerV2_ResencUNet_DA3__nnUNetPlans_FabiansResUNet_v2.1/ 3d_fullres/nnUNetTrainerV2_DA3_BN__nnUNetPlans_v2.1/ 3d_fullres/nnUNetTrainerV2_DA3__nnUNetPlans_v2.1/ -o ensembled
