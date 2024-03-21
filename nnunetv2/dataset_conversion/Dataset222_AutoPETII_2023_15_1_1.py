import multiprocessing

import numpy as np
from acvl_utils.cropping_and_padding.bounding_boxes import crop_to_bbox, pad_bbox
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
import nnunetv2.paths as paths
import SimpleITK as sitk

from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape, resample_data_or_seg_to_shape
import nibabel as nib
import nilearn
import nilearn.image


def load_case_resample_save(infile_ct, infile_pet, infile_seg, outfile_ct, outfile_pet, outfile_seg, target_spacing,
                            metadata_file,
                            suv_threshold=0.5):
    """
    we need to do this jointly for all images because we cannot afford rounding errors in target shape to cause
    a shape mismatch
    """
    # load and resample PET image + seg image
    pet_itk = sitk.ReadImage(infile_pet)
    pet_spacing = list(pet_itk.GetSpacing())[::-1]
    pet_npy = sitk.GetArrayFromImage(pet_itk)
    new_shape = compute_new_shape(pet_npy.shape, pet_spacing, target_spacing)
    pet_resampled = resample_data_or_seg_to_shape(pet_npy[None], new_shape, pet_spacing, target_spacing, False, 1, 0,
                                                 force_separate_z=False)[0]
    pet_resampled_itk = sitk.GetImageFromArray(pet_resampled)
    pet_resampled_itk.SetSpacing(list(target_spacing)[::-1])
    pet_resampled_itk.SetOrigin(pet_itk.GetOrigin())
    pet_resampled_itk.SetDirection(pet_itk.GetDirection())
    sitk.WriteImage(pet_resampled_itk, outfile_pet)
    del pet_itk, pet_resampled, pet_resampled_itk

    seg_itk = sitk.ReadImage(infile_seg)
    spacing_real = list(seg_itk.GetSpacing())[::-1]
    seg_resampled = resample_data_or_seg_to_shape(sitk.GetArrayFromImage(seg_itk)[None], new_shape, spacing_real,
                                                  target_spacing, True, 1, 0,
                                                  force_separate_z=False)[0]
    seg_resampled_itk = sitk.GetImageFromArray(seg_resampled)
    seg_resampled_itk.SetSpacing(list(target_spacing)[::-1])
    seg_resampled_itk.SetOrigin(seg_itk.GetOrigin())
    seg_resampled_itk.SetDirection(seg_itk.GetDirection())
    sitk.WriteImage(seg_resampled_itk, outfile_seg)
    del seg_itk, seg_resampled, seg_resampled_itk

    # ct image must be resampled with nilearn.image.resample_to_img because it doesnt have the same geometry as the
    # pet image
    ct = nib.load(infile_ct)
    pet = nib.load(outfile_pet)
    ct_resampled = nilearn.image.resample_to_img(ct, pet, fill_value=-1024)
    nib.save(ct_resampled, outfile_ct)

    # now crop the pet image and apply the same crop to ct
    pet_itk = sitk.ReadImage(outfile_pet)
    pet_npy = sitk.GetArrayFromImage(pet_itk)
    orig_shape = pet_npy.shape
    pet_npy_orig = np.copy(pet_npy)
    pet_npy_orig[pet_npy_orig < 1e-2] = 0
    seg_npy = sitk.GetArrayFromImage(sitk.ReadImage(outfile_seg))
    pet_npy[pet_npy<suv_threshold] = 0
    _, _, bbox = crop_to_nonzero(pet_npy[None], None, nonzero_label=0)
    bbox = pad_bbox(bbox, pad_amount=1, array_shape=pet_npy.shape)
    pet_cropped = crop_to_bbox(pet_npy_orig, bbox)
    seg_cropped = crop_to_bbox(seg_npy, bbox)

    ct = sitk.GetArrayFromImage(sitk.ReadImage(outfile_ct))
    ct = crop_to_bbox(ct, bbox)

    pet_cropped_itk = sitk.GetImageFromArray(pet_cropped)
    pet_cropped_itk.SetSpacing(list(target_spacing)[::-1])
    pet_cropped_itk.SetOrigin(pet_itk.GetOrigin())
    pet_cropped_itk.SetDirection(pet_itk.GetDirection())
    sitk.WriteImage(pet_cropped_itk, outfile_pet)

    seg_cropped_itk = sitk.GetImageFromArray(seg_cropped.astype(np.uint8))
    seg_cropped_itk.SetSpacing(list(target_spacing)[::-1])
    seg_cropped_itk.SetOrigin(pet_itk.GetOrigin())
    seg_cropped_itk.SetDirection(pet_itk.GetDirection())
    sitk.WriteImage(seg_cropped_itk, outfile_seg)

    ct = sitk.GetImageFromArray(ct.astype(np.int16))
    ct.SetSpacing(list(target_spacing)[::-1])
    ct.SetOrigin(pet_itk.GetOrigin())
    ct.SetDirection(pet_itk.GetDirection())
    sitk.WriteImage(ct, outfile_ct)

    save_json({'spacing': [float(i) for i in target_spacing], 'bbox': bbox, 'orig_shape': orig_shape}, metadata_file)



def convert_autopet(autopet_base_dir: str = '/media/isensee/My Book1/AutoPET/nifti/FDG-PET-CT-Lesions',
                    nnunet_dataset_id: int = 222, target_spacing=(1.500, 1.0182, 1.0182),
                    task_name: str = "AutoPETII_2023_resampled_15_1_1"):
    foldername = "Dataset%03.0d_%s" % (nnunet_dataset_id, task_name)

    # setting up nnU-Net folders
    out_base = join(paths.nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    with multiprocessing.get_context("spawn").Pool(5) as p:

        patients = subdirs(autopet_base_dir, prefix='PETCT', join=False)
        n = 0
        identifiers = []
        r = []
        for pat in patients:
            patient_acquisitions = subdirs(join(autopet_base_dir, pat), join=False)
            for pa in patient_acquisitions:
                n += 1
                identifier = f"{pat}_{pa}"
                identifiers.append(identifier)
                infile_ct = join(autopet_base_dir, pat, pa, 'CT.nii.gz')
                infile_pet = join(autopet_base_dir, pat, pa, 'SUV.nii.gz')
                infile_seg = join(autopet_base_dir, pat, pa, 'SEG.nii.gz')
                outfile_ct = join(imagestr, f'{identifier}_0000.nii.gz')
                outfile_pet = join(imagestr, f'{identifier}_0001.nii.gz')
                outfile_seg = join(labelstr, f'{identifier}.nii.gz')
                metadata_file = join(imagestr, f'{identifier}.json')

                r.append(p.starmap_async(load_case_resample_save,
                                         ((
                                              infile_ct, infile_pet, infile_seg, outfile_ct, outfile_pet,
                                              outfile_seg, target_spacing, metadata_file
                                          ),)))
        [i.get() for i in r]

    generate_dataset_json(out_base, {0: "CT", 1: "CT"},
                          labels={
                              "background": 0,
                              "tumor": 1
                          },
                          num_training_cases=n, file_ending='.nii.gz',
                          dataset_name=task_name, reference='https://autopet-ii.grand-challenge.org/',
                          release='release',
                          # overwrite_image_reader_writer='NibabelIOWithReorient',
                          description=task_name)

    # manual split
    splits = []
    for fold in range(5):
        val_patients = patients[fold:: 5]
        splits.append(
            {
                'train': [i for i in identifiers if not any([i.startswith(v) for v in val_patients])],
                'val': [i for i in identifiers if any([i.startswith(v) for v in val_patients])],
            }
        )
    pp_out_dir = join(paths.nnUNet_preprocessed, foldername)
    maybe_mkdir_p(pp_out_dir)
    save_json(splits, join(pp_out_dir, 'splits_final.json'), sort_keys=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', type=str,
                        help="The downloaded and extracted autopet dataset (must have PETCT_XXX subfolders)")
    parser.add_argument('-d', required=False, type=int, default=222, help='nnU-Net Dataset ID, default: 222')
    args = parser.parse_args()
    amos_base = args.input_folder
    convert_autopet(amos_base, args.d)
    # infile_ct = '/home/isensee/temp/autopet/CT.nii.gz'
    # infile_pet = '/home/isensee/temp/autopet/SUV.nii.gz'
    # infile_seg = '/home/isensee/temp/autopet/SEG.nii.gz'
    # outdir = join('/home/isensee/temp/autopet/', 'out')
    # maybe_mkdir_p(outdir)
    # outfile_ct = join(outdir, 'CT.nii.gz')
    # outfile_seg = join(outdir, 'SEG.nii.gz')
    # outfile_pet = join(outdir, 'SUV.nii.gz')
    # target_spacing = (1.500, 1.0182, 1.0182)
