import os

from batchgenerators.utilities.file_and_folder_operations import *

import nnunetv2.paths as paths
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets

if __name__ == '__main__':
    # creates a dummy dataset where there are no files in imagestr and labelstr
    source_dataset = 'Dataset004_Hippocampus'

    target_dataset = 'Dataset987_dummyDataset4'
    target_dataset_dir = join(paths.nnUNet_raw, target_dataset)
    maybe_mkdir_p(target_dataset_dir)

    dataset = get_filenames_of_train_images_and_targets(join(paths.nnUNet_raw, source_dataset))

    # the returned dataset will have absolute paths. We should use relative paths so that you can freely copy
    # datasets around between systems. As long as the source dataset is there it will continue working even if
    # nnUNet_raw is in different locations

    # paths must be relative to target_dataset_dir!!!
    for k in dataset.keys():
        dataset[k]['label'] = os.path.relpath(dataset[k]['label'], target_dataset_dir)
        dataset[k]['images'] = [os.path.relpath(i, target_dataset_dir) for i in dataset[k]['images']]

    # load old dataset.json
    dataset_json = load_json(join(paths.nnUNet_raw, source_dataset, 'dataset.json'))
    dataset_json['dataset'] = dataset

    # save
    save_json(dataset_json, join(target_dataset_dir, 'dataset.json'), sort_keys=False)
