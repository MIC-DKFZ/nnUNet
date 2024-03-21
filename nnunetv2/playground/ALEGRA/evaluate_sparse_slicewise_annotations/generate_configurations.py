from nnunetv2.playground.ALEGRA.evaluate_sparse_slicewise_annotations.configuration import *
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
import nnunetv2.paths as paths

if __name__ == '__main__':
    for d in datasets:
        dataset_name = maybe_convert_to_dataset_name(d)
        plans_file = join(paths.nnUNet_preprocessed, dataset_name, 'nnUNetPlans.json')
        plans = load_json(plans_file)
        for nc, pp in new_configurations.items():
            plans['configurations'][nc] = {
                'inherits_from': base_configuration,
                "data_identifier": 'nnUNetPlans_' + pp,
                "preprocessor_name": pp
            }
        write_json(plans, plans_file)
