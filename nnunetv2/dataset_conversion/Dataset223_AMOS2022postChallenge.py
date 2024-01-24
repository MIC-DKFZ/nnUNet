import shutil

from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.paths import nnUNet_raw
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

if __name__ == '__main__':
    downloaded_amos_dir = '/home/isensee/amos22/amos22' # downloaded and extracted from https://zenodo.org/record/7155725#.Y0OOCOxBztM

    target_dataset_id = 223
    target_dataset_name = f'Dataset{target_dataset_id:3.0f}_AMOS2022postChallenge'

    maybe_mkdir_p(join(nnUNet_raw, target_dataset_name))
    imagesTr = join(nnUNet_raw, target_dataset_name, 'imagesTr')
    imagesTs = join(nnUNet_raw, target_dataset_name, 'imagesTs')
    labelsTr = join(nnUNet_raw, target_dataset_name, 'labelsTr')
    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(imagesTs)
    maybe_mkdir_p(labelsTr)

    train_identifiers = []
    # copy images
    source = join(downloaded_amos_dir, 'imagesTr')
    source_files = nifti_files(source, join=False)
    train_identifiers += source_files
    for s in source_files:
        shutil.copy(join(source, s), join(imagesTr, s[:-7] + '_0000.nii.gz'))

    source = join(downloaded_amos_dir, 'imagesVa')
    source_files = nifti_files(source, join=False)
    train_identifiers += source_files
    for s in source_files:
        shutil.copy(join(source, s), join(imagesTr, s[:-7] + '_0000.nii.gz'))

    source = join(downloaded_amos_dir, 'imagesTs')
    source_files = nifti_files(source, join=False)
    for s in source_files:
        shutil.copy(join(source, s), join(imagesTs, s[:-7] + '_0000.nii.gz'))

    # copy labels
    source = join(downloaded_amos_dir, 'labelsTr')
    source_files = nifti_files(source, join=False)
    for s in source_files:
        shutil.copy(join(source, s), join(labelsTr, s))

    source = join(downloaded_amos_dir, 'labelsVa')
    source_files = nifti_files(source, join=False)
    for s in source_files:
        shutil.copy(join(source, s), join(labelsTr, s))

    old_dataset_json = load_json(join(downloaded_amos_dir, 'dataset.json'))
    new_labels = {v: k for k, v in old_dataset_json['labels'].items()}

    generate_dataset_json(join(nnUNet_raw, target_dataset_name), {0: 'nonCT'}, new_labels,
                          num_training_cases=len(train_identifiers), file_ending='.nii.gz', regions_class_order=None,
                          dataset_name=target_dataset_name, reference='https://zenodo.org/record/7155725#.Y0OOCOxBztM',
                          license=old_dataset_json['licence'],  # typo in OG dataset.json
                          description=old_dataset_json['description'],
                          release=old_dataset_json['release'])
