import shutil

from batchgenerators.utilities.file_and_folder_operations import isdir, join

import nnunetv2.paths as paths

if __name__ == '__main__':
    # deletes everything!
    dataset_names = [
        'Dataset996_IntegrationTest_Hippocampus_regions_ignore',
        'Dataset997_IntegrationTest_Hippocampus_regions',
        'Dataset998_IntegrationTest_Hippocampus_ignore',
        'Dataset999_IntegrationTest_Hippocampus',
    ]
    for fld in [paths.nnUNet_raw, paths.nnUNet_preprocessed, paths.nnUNet_results]:
        for d in dataset_names:
            if isdir(join(fld, d)):
                shutil.rmtree(join(fld, d))

