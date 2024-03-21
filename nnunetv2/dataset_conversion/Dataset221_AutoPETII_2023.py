from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
import nnunetv2.paths as paths


def convert_autopet(autopet_base_dir:str = '/media/isensee/My Book1/AutoPET/nifti/FDG-PET-CT-Lesions',
                     nnunet_dataset_id: int = 221):
    task_name = "AutoPETII_2023"

    foldername = "Dataset%03.0d_%s" % (nnunet_dataset_id, task_name)

    # setting up nnU-Net folders
    out_base = join(paths.nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    patients = subdirs(autopet_base_dir, prefix='PETCT', join=False)
    n = 0
    identifiers = []
    for pat in patients:
        patient_acquisitions = subdirs(join(autopet_base_dir, pat), join=False)
        for pa in patient_acquisitions:
            n += 1
            identifier = f"{pat}_{pa}"
            identifiers.append(identifier)
            if not isfile(join(imagestr, f'{identifier}_0000.nii.gz')):
                shutil.copy(join(autopet_base_dir, pat, pa, 'CTres.nii.gz'), join(imagestr, f'{identifier}_0000.nii.gz'))
            if not isfile(join(imagestr, f'{identifier}_0001.nii.gz')):
                shutil.copy(join(autopet_base_dir, pat, pa, 'SUV.nii.gz'), join(imagestr, f'{identifier}_0001.nii.gz'))
            if not isfile(join(imagestr, f'{identifier}.nii.gz')):
                shutil.copy(join(autopet_base_dir, pat, pa, 'SEG.nii.gz'), join(labelstr, f'{identifier}.nii.gz'))

    generate_dataset_json(out_base, {0: "CT", 1:"CT"},
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
        val_patients = patients[fold :: 5]
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
    parser.add_argument('-d', required=False, type=int, default=221, help='nnU-Net Dataset ID, default: 221')
    args = parser.parse_args()
    amos_base = args.input_folder
    convert_autopet(amos_base, args.d)
