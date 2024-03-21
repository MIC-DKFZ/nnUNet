import shutil

from batchgenerators.utilities.file_and_folder_operations import isdir, join, load_json, save_json

from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
import nnunetv2.paths as paths

if __name__ == '__main__':
    dataset_name = 'IntegrationTest_Hippocampus_regions'
    dataset_id = 997
    dataset_name = f"Dataset{dataset_id:03d}_{dataset_name}"

    try:
        existing_dataset_name = maybe_convert_to_dataset_name(dataset_id)
        if existing_dataset_name != dataset_name:
            raise FileExistsError(
                f"A different dataset with id {dataset_id} already exists :-(: {existing_dataset_name}. If "
                f"you intent to delete it, remember to also remove it in nnUNet_preprocessed and "
                f"nnUNet_results!")
    except RuntimeError:
        pass

    if isdir(join(paths.nnUNet_raw, dataset_name)):
        shutil.rmtree(join(paths.nnUNet_raw, dataset_name))

    source_dataset = maybe_convert_to_dataset_name(4)
    shutil.copytree(join(paths.nnUNet_raw, source_dataset), join(paths.nnUNet_raw, dataset_name))

    # additionally optimize entire hippocampus region, remove Posterior
    dj = load_json(join(paths.nnUNet_raw, dataset_name, 'dataset.json'))
    dj['labels'] = {
        'background': 0,
        'hippocampus': (1, 2),
        'anterior': 1
    }
    dj['regions_class_order'] = (2, 1)
    save_json(dj, join(paths.nnUNet_raw, dataset_name, 'dataset.json'), sort_keys=False)
