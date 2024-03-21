from batchgenerators.utilities.file_and_folder_operations import save_json, join, load_json

from nnunetv2.batch_running.learning_from_sparse_annotations.estimate_annotation_percentages import run_on_folder
import nnunetv2.paths as paths
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.json_export import recursive_fix_for_json_export

if __name__ == '__main__':
    datasets = (981, 982, 983, 984)
    ref = 980
    ref_name = maybe_convert_to_dataset_name(ref)
    ref_folder = join(paths.nnUNet_preprocessed, ref_name, 'nnUNetPlans_3d_fullres')
    for d in datasets:
        print(d)
        folder = join(paths.nnUNet_preprocessed, maybe_convert_to_dataset_name(d), 'nnUNetPlans_3d_fullres')
        r = run_on_folder(folder, ref_folder, list(load_json(join(folder, '../', 'dataset.json'))['labels'].values()), 8, ignore_label=2)
        recursive_fix_for_json_export(r)
        save_json(r, join(folder, '../', 'percent_annotated.json'), sort_keys=False)
