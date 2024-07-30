from copy import deepcopy

from batchgenerators.utilities.file_and_folder_operations import *

import nnunetv2.paths as paths
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', nargs='+', type=int, help='List of dataset ids')
    args = parser.parse_args()

    for d in args.d:
        dataset_name = maybe_convert_to_dataset_name(d)
        plans = load_json(join(paths.nnUNet_preprocessed, dataset_name, 'nnUNetPlans.json'))
        plans['configurations']['3d_lowres'] = {
            "data_identifier": "nnUNetPlans_3d_lowres",  # do not be a dumbo and forget this. I was a dumbo. And I paid dearly with ~10 min debugging time
            'inherits_from': '3d_fullres',
            "patch_size": [20, 28, 20],
            "median_image_size_in_voxels": [18.0, 25.0, 18.0],
            "spacing": [2.0, 2.0, 2.0],
            "architecture": deepcopy(plans['configurations']['3d_fullres']["architecture"]),
            "next_stage": "3d_cascade_fullres"
        }
        plans['configurations']['3d_lowres']['architecture']["arch_kwargs"]['n_conv_per_stage'] = [2, 2, 2]
        plans['configurations']['3d_lowres']['architecture']["arch_kwargs"]['n_conv_per_stage_decoder'] = [2, 2]
        plans['configurations']['3d_lowres']['architecture']["arch_kwargs"]['strides'] = [[1, 1, 1], [2, 2, 2], [2, 2, 2]]
        plans['configurations']['3d_lowres']['architecture']["arch_kwargs"]['kernel_sizes'] = [[3, 3, 3], [3, 3, 3], [3, 3, 3]]
        plans['configurations']['3d_lowres']['architecture']["arch_kwargs"]['n_stages'] = 3
        plans['configurations']['3d_lowres']['architecture']["arch_kwargs"]['features_per_stage'] = [
            32,
            64,
            128
        ]

        plans['configurations']['3d_cascade_fullres'] = {
            'inherits_from': '3d_fullres',
            "previous_stage": "3d_lowres"
        }
        save_json(plans, join(paths.nnUNet_preprocessed, dataset_name, 'nnUNetPlans.json'), sort_keys=False)
