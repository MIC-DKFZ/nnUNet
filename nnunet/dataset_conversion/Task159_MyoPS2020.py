import SimpleITK
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.utilities.file_and_folder_operations_winos import * # Join path by slash on windows system.
import shutil

import SimpleITK as sitk
from nnunet.paths import nnUNet_raw_data
from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.utilities.sitk_stuff import copy_geometry


def convert_labels_to_nnunet(source_nifti: str, target_nifti: str):
    img = sitk.ReadImage(source_nifti)
    img_npy = sitk.GetArrayFromImage(img)
    nnunet_seg = np.zeros(img_npy.shape, dtype=np.uint8)
    # why are they not using normal labels and instead use random numbers???
    nnunet_seg[img_npy == 500] = 1  # left ventricular (LV) blood pool (500)
    nnunet_seg[img_npy == 600] = 2  # right ventricular blood pool (600)
    nnunet_seg[img_npy == 200] = 3  # LV normal myocardium (200)
    nnunet_seg[img_npy == 1220] = 4  # LV myocardial edema (1220)
    nnunet_seg[img_npy == 2221] = 5  # LV myocardial scars (2221)
    nnunet_seg_itk = sitk.GetImageFromArray(nnunet_seg)
    nnunet_seg_itk = copy_geometry(nnunet_seg_itk, img)
    sitk.WriteImage(nnunet_seg_itk, target_nifti)


def convert_labels_back_to_myops(source_nifti: str, target_nifti: str):
    nnunet_itk = sitk.ReadImage(source_nifti)
    nnunet_npy = sitk.GetArrayFromImage(nnunet_itk)
    myops_seg = np.zeros(nnunet_npy.shape, dtype=np.uint8)
    # why are they not using normal labels and instead use random numbers???
    myops_seg[nnunet_npy == 1] = 500  # left ventricular (LV) blood pool (500)
    myops_seg[nnunet_npy == 2] = 600  # right ventricular blood pool (600)
    myops_seg[nnunet_npy == 3] = 200  # LV normal myocardium (200)
    myops_seg[nnunet_npy == 4] = 1220  # LV myocardial edema (1220)
    myops_seg[nnunet_npy == 5] = 2221  # LV myocardial scars (2221)
    myops_seg_itk = sitk.GetImageFromArray(myops_seg)
    myops_seg_itk = copy_geometry(myops_seg_itk, nnunet_itk)
    sitk.WriteImage(myops_seg_itk, target_nifti)


if __name__ == '__main__':
    # this is where we extracted all the archives. This folder must have the subfolders test20, train25,
    # train25_myops_gd. We do not use test_data_gd because the test GT is encoded and cannot be used as it is
    base = '/home/fabian/Downloads/MyoPS 2020 Dataset'

    # Arbitrary task id. This is just to ensure each dataset ha a unique number. Set this to whatever ([0-999]) you
    # want
    task_id = 159
    task_name = "MyoPS2020"

    foldername = "Task%03.0d_%s" % (task_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw_data, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    imagestr_source = join(base, 'train25')
    imagests_source = join(base, 'test20')
    labelstr_source = join(base, 'train25_myops_gd')

    # convert training set
    nii_files = nifti_files(imagestr_source, join=False)
    # remove their modality identifier. Conveniently it's always 2 characters. np.unique to get the identifiers
    identifiers = np.unique([i[:-len('_C0.nii.gz')] for i in nii_files])
    for i in identifiers:
        shutil.copy(join(imagestr_source, i + "_C0.nii.gz"), join(imagestr, i + '_0000.nii.gz'))
        shutil.copy(join(imagestr_source, i + "_DE.nii.gz"), join(imagestr, i + '_0001.nii.gz'))
        shutil.copy(join(imagestr_source, i + "_T2.nii.gz"), join(imagestr, i + '_0002.nii.gz'))
        convert_labels_to_nnunet(join(labelstr_source, i + '_gd.nii.gz'), join(labelstr, i + '.nii.gz'))

    # test set
    nii_files = nifti_files(imagests_source, join=False)
    # remove their modality identifier. Conveniently it's always 2 characters. np.unique to get the identifiers
    identifiers = np.unique([i[:-len('_C0.nii.gz')] for i in nii_files])
    for i in identifiers:
        shutil.copy(join(imagests_source, i + "_C0.nii.gz"), join(imagests, i + '_0000.nii.gz'))
        shutil.copy(join(imagests_source, i + "_DE.nii.gz"), join(imagests, i + '_0001.nii.gz'))
        shutil.copy(join(imagests_source, i + "_T2.nii.gz"), join(imagests, i + '_0002.nii.gz'))

    generate_dataset_json(join(out_base, 'dataset.json'),
                          imagestr,
                          None,
                          ('C0', 'DE', 'T2'),
                          {
                              0: 'background',
                              1: "left ventricular (LV) blood pool",
                              2: "right ventricular blood pool",
                              3: "LV normal myocardium",
                              4: "LV myocardial edema",
                              5: "LV myocardial scars",
                          },
                          task_name,
                          license='see http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/myops20/index.html',
                          dataset_description='see http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/myops20/index.html',
                          dataset_reference='http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/myops20/index.html',
                          dataset_release='0')

    # REMEMBER THAT TEST SET INFERENCE WILL REQUIRE YOU CONVERT THE LABELS BACK TO THEIR CONVENTION
    # use convert_labels_back_to_myops for that!
    # man I am such a nice person. Love you guys.