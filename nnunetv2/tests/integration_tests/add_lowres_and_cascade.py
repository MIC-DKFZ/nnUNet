from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', nargs='+', type=int, help='List of dataset ids')
    args = parser.parse_args()

    for d in args.d:
        dataset_name = maybe_convert_to_dataset_name(d)
        plans = load_json(join(nnUNet_preprocessed, dataset_name, 'nnUNetPlans.json'))
        plans['configurations']['3d_lowres'] = {
            "data_identifier": "nnUNetPlans_3d_lowres",
            'inherits_from': '3d_fullres',
            "patch_size": [20, 28, 20],
            "median_patient_size_in_voxels": [18.0, 25.0, 18.0],
            "spacing": [2.0, 2.0, 2.0],
            "n_conv_per_stage_encoder": [2, 2, 2],
            "n_conv_per_stage_decoder": [2, 2],
            "num_pool_per_axis": [2, 2, 2],
            "pool_op_kernel_sizes": [[1, 1, 1], [2, 2, 2], [2, 2, 2]],
            "conv_kernel_sizes": [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
            "next_stage": "3d_cascade_fullres"
        }
        plans['configurations']['3d_cascade_fullres'] = {
            'inherits_from': '3d_fullres',
            "previous_stage": "3d_lowres"
        }
        save_json(plans, join(nnUNet_preprocessed, dataset_name, 'nnUNetPlans.json'))