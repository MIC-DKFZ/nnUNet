import shutil
from copy import deepcopy
from typing import List, Union, Tuple
import argparse

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import load_json, join, save_json, isfile, maybe_mkdir_p
from nnunetv2.configuration import ANISO_THRESHOLD
from nnunetv2.experiment_planning.experiment_planners.network_topology import get_pool_and_conv_props
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
from nnunetv2.preprocessing.normalization.map_channel_name_to_normalization import get_normalization_scheme
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape, compute_new_shape
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets
from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner

class DistanceTransformExperimentPlanner(ExperimentPlanner):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'DistanceTransformPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name, overwrite_target_spacing, suppress_transpose)
        
        # Custom properties for distance maps
        self.distance_maps_dir = join(nnUNet_preprocessed, self.dataset_name, 'distance_transforms')

    def plan_experiment(self):
        """
        Extend the default plan_experiment function to incorporate distance maps into the plans.
        """
        _tmp = {}

        # Get transpose
        transpose_forward, transpose_backward = self.determine_transpose()

        # Get fullres spacing and transpose it
        fullres_spacing = self.determine_fullres_target_spacing()
        fullres_spacing_transposed = fullres_spacing[transpose_forward]

        # Get transposed new median shape (what we would have after resampling)
        new_shapes = [compute_new_shape(j, i, fullres_spacing) for i, j in
                      zip(self.dataset_fingerprint['spacings'], self.dataset_fingerprint['shapes_after_crop'])]
        new_median_shape = np.median(new_shapes, 0)
        new_median_shape_transposed = new_median_shape[transpose_forward]

        approximate_n_voxels_dataset = float(np.prod(new_median_shape_transposed, dtype=np.float64) *
                                             self.dataset_json['numTraining'])
        # Only run 3d if this is a 3d dataset
        if new_median_shape_transposed[0] != 1:
            plan_3d_fullres = self.get_plans_for_configuration(fullres_spacing_transposed,
                                                               new_median_shape_transposed,
                                                               self.generate_data_identifier('3d_fullres'),
                                                               approximate_n_voxels_dataset, _tmp)
            plan_3d_fullres['distance_maps_dir'] = self.distance_maps_dir  # Add distance maps directory to the plans
            plan_3d_fullres['batch_dice'] = True
        else:
            plan_3d_fullres = None

        # 2D configuration
        plan_2d = self.get_plans_for_configuration(fullres_spacing_transposed[1:],
                                                   new_median_shape_transposed[1:],
                                                   self.generate_data_identifier('2d'), approximate_n_voxels_dataset,
                                                   _tmp)
        plan_2d['distance_maps_dir'] = self.distance_maps_dir  # Add distance maps directory to the plans
        plan_2d['batch_dice'] = True

        print('2D U-Net configuration:')
        print(plan_2d)
        print()

        # Median spacing and shape, just for reference when printing the plans
        median_spacing = np.median(self.dataset_fingerprint['spacings'], 0)[transpose_forward]
        median_shape = np.median(self.dataset_fingerprint['shapes_after_crop'], 0)[transpose_forward]

        # Instead of writing all that into the plans we just copy the original file. More files, but less crowded
        # per file.
        shutil.copy(join(self.raw_dataset_folder, 'dataset.json'),
                    join(nnUNet_preprocessed, self.dataset_name, 'dataset.json'))

        # JSON serialization adjustment
        plans = {
            'dataset_name': self.dataset_name,
            'plans_name': self.plans_identifier,
            'original_median_spacing_after_transp': [float(i) for i in median_spacing],
            'original_median_shape_after_transp': [int(round(i)) for i in median_shape],
            'image_reader_writer': self.determine_reader_writer().__name__,
            'transpose_forward': [int(i) for i in transpose_forward],
            'transpose_backward': [int(i) for i in transpose_backward],
            'configurations': {'2d': plan_2d},
            'experiment_planner_used': self.__class__.__name__,
            'label_manager': 'LabelManager',
            'foreground_intensity_properties_per_channel': self.dataset_fingerprint['foreground_intensity_properties_per_channel']
        }

        if plan_3d_fullres is not None:
            plans['configurations']['3d_fullres'] = plan_3d_fullres
            print('3D fullres U-Net configuration:')
            print(plan_3d_fullres)
            print()

        self.plans = plans
        self.save_plans(plans)
        return plans

    def save_plans(self, plans):
        recursive_fix_for_json_export(plans)

        plans_file = join(nnUNet_preprocessed, self.dataset_name, self.plans_identifier + '.json')

        # Avoid overwriting existing custom configurations every time this is executed.
        if isfile(plans_file):
            old_plans = load_json(plans_file)
            old_configurations = old_plans['configurations']
            for c in plans['configurations'].keys():
                if c in old_configurations.keys():
                    del (old_configurations[c])
            plans['configurations'].update(old_configurations)

        maybe_mkdir_p(join(nnUNet_preprocessed, self.dataset_name))
        save_json(plans, plans_file, sort_keys=False)
        print(f"Plans were saved to {join(nnUNet_preprocessed, self.dataset_name, self.plans_identifier + '.json')}")

def test_distance_transform_experiment_planner(dataset_name_or_id=101):
    # Create an instance of the planner
    planner = DistanceTransformExperimentPlanner(dataset_name_or_id=dataset_name_or_id,
                                                 gpu_memory_target_in_gb=8)
    
    # Run the planner
    try:
        plans = planner.plan_experiment()
        print("Experiment planner executed successfully.")
        
        # Check if the plans have the distance_maps_dir property in their configurations
        assert plans['configurations'] is not None, "Configurations are missing from the plans."
        assert '2d' in plans['configurations'], "2D plan configuration is missing."
        assert 'distance_maps_dir' in plans['configurations']['2d'], "Distance maps directory is missing in 2D configuration."
        
        if '3d_fullres' in plans['configurations']:
            assert 'distance_maps_dir' in plans['configurations']['3d_fullres'], "Distance maps directory is missing in 3D fullres configuration."
        
    except Exception as e:
        print(f"Experiment planner test failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the DistanceTransformExperimentPlanner.")
    parser.add_argument('-d', '--dataset_id', type=str, default="test_dataset", help='Dataset name or ID to be used for testing.')
    args = parser.parse_args()

    # Run the test function with the provided dataset ID
    test_distance_transform_experiment_planner(dataset_name_or_id=args.dataset_id)