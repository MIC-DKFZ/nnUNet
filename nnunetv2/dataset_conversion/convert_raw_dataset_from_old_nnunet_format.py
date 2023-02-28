import shutil
from copy import deepcopy

from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, isdir, load_json, save_json
from nnunetv2.paths import nnUNet_raw


def convert(source_folder, target_dataset_name):
    """
    remember that old tasks were called TaskXXX_YYY and new ones are called DatasetXXX_YYY
    source_folder
    """
    if isdir(join(nnUNet_raw, target_dataset_name)):
        raise RuntimeError(f'Target dataset name {target_dataset_name} already exists. Aborting... '
                           f'(we might break something). If you are sure you want to proceed, please manually '
                           f'delete {join(nnUNet_raw, target_dataset_name)}')
    maybe_mkdir_p(join(nnUNet_raw, target_dataset_name))
    shutil.copytree(join(source_folder, 'imagesTr'), join(nnUNet_raw, target_dataset_name, 'imagesTr'))
    shutil.copytree(join(source_folder, 'labelsTr'), join(nnUNet_raw, target_dataset_name, 'labelsTr'))
    if isdir(join(source_folder, 'imagesTs')):
        shutil.copytree(join(source_folder, 'imagesTs'), join(nnUNet_raw, target_dataset_name, 'imagesTs'))
    if isdir(join(source_folder, 'labelsTs')):
        shutil.copytree(join(source_folder, 'labelsTs'), join(nnUNet_raw, target_dataset_name, 'labelsTs'))
    if isdir(join(source_folder, 'imagesVal')):
        shutil.copytree(join(source_folder, 'imagesVal'), join(nnUNet_raw, target_dataset_name, 'imagesVal'))
    if isdir(join(source_folder, 'labelsVal')):
        shutil.copytree(join(source_folder, 'labelsVal'), join(nnUNet_raw, target_dataset_name, 'labelsVal'))
    shutil.copy(join(source_folder, 'dataset.json'), join(nnUNet_raw, target_dataset_name))

    dataset_json = load_json(join(nnUNet_raw, target_dataset_name, 'dataset.json'))
    del dataset_json['tensorImageSize']
    del dataset_json['numTest']
    del dataset_json['training']
    del dataset_json['test']
    dataset_json['channel_names'] = deepcopy(dataset_json['modality'])
    del dataset_json['modality']

    dataset_json['labels'] = {j: int(i) for i, j in dataset_json['labels'].items()}
    dataset_json['file_ending'] = ".nii.gz"
    save_json(dataset_json, join(nnUNet_raw, target_dataset_name, 'dataset.json'))


def convert_entry_point():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", type=str,
                        help='Raw old nnUNet dataset. This must be the folder with imagesTr,labelsTr etc subfolders! '
                             'Please provide the PATH to the old Task, not just the task name. nnU-Net V2 does not '
                             'know where v1 tasks are.')
    parser.add_argument("output_dataset_name", type=str,
                        help='New dataset NAME (not path!). Must follow the DatasetXXX_NAME convention!')
    args = parser.parse_args()
    convert(args.input_folder, args.output_dataset_name)
