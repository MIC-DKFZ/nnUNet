import shutil

from batchgenerators.utilities.file_and_folder_operations import isdir, join

from nnunetv2.paths import nnUNet_raw, nnUNet_results, nnUNet_preprocessed

if __name__ == '__main__':
    # deletes everything!
    dataset_names = [
        'Dataset996_IntegrationTest_Hippocampus_regions_ignore',
        'Dataset997_IntegrationTest_Hippocampus_regions',
        'Dataset998_IntegrationTest_Hippocampus_ignore',
        'Dataset999_IntegrationTest_Hippocampus',
    ]
    for fld in [nnUNet_raw, nnUNet_preprocessed, nnUNet_results]:
        for d in dataset_names:
            if isdir(join(fld, d)):
                shutil.rmtree(join(fld, d))

