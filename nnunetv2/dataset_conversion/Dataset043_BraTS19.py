import multiprocessing
import shutil

import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw


def copy_BraTS_segmentation_and_convert_labels_to_nnUNet(in_file: str, out_file: str) -> None:
    # use this for segmentation only!!!
    # nnUNet wants the labels to be continuous. BraTS is 0, 1, 2, 4 -> we make that into 0, 1, 2, 3
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)

    uniques = np.unique(img_npy)
    for u in uniques:
        if u not in [0, 1, 2, 4]:
            raise RuntimeError('unexpected label')

    seg_new = np.zeros_like(img_npy)
    seg_new[img_npy == 4] = 3
    seg_new[img_npy == 2] = 1
    seg_new[img_npy == 1] = 2
    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)


def convert_labels_back_to_BraTS(seg: np.ndarray):
    new_seg = np.zeros_like(seg)
    new_seg[seg == 1] = 2
    new_seg[seg == 3] = 4
    new_seg[seg == 2] = 1
    return new_seg


def load_convert_labels_back_to_BraTS(filename, input_folder, output_folder):
    a = sitk.ReadImage(join(input_folder, filename))
    b = sitk.GetArrayFromImage(a)
    c = convert_labels_back_to_BraTS(b)
    d = sitk.GetImageFromArray(c)
    d.CopyInformation(a)
    sitk.WriteImage(d, join(output_folder, filename))


def convert_folder_with_preds_back_to_BraTS_labeling_convention(input_folder: str, output_folder: str,
                                                                num_processes: int = 12):
    """
    reads all prediction files (nifti) in the input folder, converts the labels back to BraTS convention and saves the
    """
    maybe_mkdir_p(output_folder)
    nii = subfiles(input_folder, suffix='.nii.gz', join=False)
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        p.starmap(load_convert_labels_back_to_BraTS, zip(nii, [input_folder] * len(nii), [output_folder] * len(nii)))


if __name__ == '__main__':
    brats_data_dir = ...

    task_id = 43
    task_name = "BraTS2019"

    foldername = "Dataset%03.0d_%s" % (task_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    case_ids_hgg = subdirs(join(brats_data_dir, "HGG"), prefix='BraTS', join=False)
    case_ids_lgg = subdirs(join(brats_data_dir, "LGG"), prefix="BraTS", join=False)

    print("copying hggs")
    for c in tqdm(case_ids_hgg):
        shutil.copy(join(brats_data_dir, "HGG", c, c + "_t1.nii"), join(imagestr, c + '_0000.nii'))
        shutil.copy(join(brats_data_dir, "HGG", c, c + "_t1ce.nii"), join(imagestr, c + '_0001.nii'))
        shutil.copy(join(brats_data_dir, "HGG", c, c + "_t2.nii"), join(imagestr, c + '_0002.nii'))
        shutil.copy(join(brats_data_dir, "HGG", c, c + "_flair.nii"), join(imagestr, c + '_0003.nii'))

        copy_BraTS_segmentation_and_convert_labels_to_nnUNet(join(brats_data_dir, "HGG", c, c + "_seg.nii"),
                                                             join(labelstr, c + '.nii'))
    print("copying lggs")
    for c in tqdm(case_ids_lgg):
        shutil.copy(join(brats_data_dir, "LGG", c, c + "_t1.nii"), join(imagestr, c + '_0000.nii'))
        shutil.copy(join(brats_data_dir, "LGG", c, c + "_t1ce.nii"), join(imagestr, c + '_0001.nii'))
        shutil.copy(join(brats_data_dir, "LGG", c, c + "_t2.nii"), join(imagestr, c + '_0002.nii'))
        shutil.copy(join(brats_data_dir, "LGG", c, c + "_flair.nii"), join(imagestr, c + '_0003.nii'))

        copy_BraTS_segmentation_and_convert_labels_to_nnUNet(join(brats_data_dir, "LGG", c, c + "_seg.nii"),
                                                             join(labelstr, c + '.nii'))

    generate_dataset_json(out_base,
                          channel_names={0: 'T1', 1: 'T1ce', 2: 'T2', 3: 'Flair'},
                          labels={
                              'background': 0,
                              'whole tumor': (1, 2, 3),
                              'tumor core': (2, 3),
                              'enhancing tumor': (3,)
                          },
                          num_training_cases=(len(case_ids_hgg) + len(case_ids_lgg)),
                          file_ending='.nii',
                          regions_class_order=(1, 2, 3),
                          license='see https://www.synapse.org/#!Synapse:syn25829067/wiki/610863',
                          reference='see https://www.synapse.org/#!Synapse:syn25829067/wiki/610863',
                          dataset_release='1.0')
