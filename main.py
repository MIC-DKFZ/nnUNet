#!/usr/bin/env python
"""
Multi-Task Pancreatic Cancer Segmentation with nnUNetv2
Main script for testing plan_and_preprocess API and future training/inference
"""

import os
import argparse
import shutil
from pathlib import Path
import pandas as pd

# Register custom components first
import src
from src import register_custom_components

# nnUNet imports
from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, plan_experiments, preprocess
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed, nnUNet_results
from batchgenerators.utilities.file_and_folder_operations import join, isdir, maybe_mkdir_p


# def setup_environment():
#     """Setup nnUNet environment variables if not set"""
#     base_path = Path.cwd() / 'nnunet_data'

#     os.environ['nnUNet_raw'] = str(base_path / 'nnUNet_raw')
#     os.environ['nnUNet_preprocessed'] = str(base_path / 'nnUNet_preprocessed')
#     os.environ['nnUNet_results'] = str(base_path / 'nnUNet_results')

#     # Create directories
#     for path_env in ['nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']:
#         path = Path(os.environ[path_env])
#         path.mkdir(parents=True, exist_ok=True)

#     print(f"nnUNet_raw: {os.environ['nnUNet_raw']}")
#     print(f"nnUNet_preprocessed: {os.environ['nnUNet_preprocessed']}")
#     print(f"nnUNet_results: {os.environ['nnUNet_results']}")


def create_labels_csv(raw_dataset_folder: str):
    """
    Create labels.csv from raw dataset folder structure
    Assumes subtype0/, subtype1/, subtype2/ folders exist
    """
    labels_data = []

    for subtype in ['subtype0', 'subtype1', 'subtype2']:
        subtype_folder = join(raw_dataset_folder, 'imagesTr', subtype)
        if os.path.exists(subtype_folder):
            # Extract case IDs from image files
            for file in os.listdir(subtype_folder):
                if file.endswith('_0000.nii.gz'):
                    case_id = file.replace('_0000.nii.gz', '')
                    subtype_id = int(subtype.replace('subtype', ''))
                    labels_data.append({'case_id': case_id, 'subtype': subtype_id})

    # Save to CSV
    df = pd.DataFrame(labels_data)
    labels_csv_path = join(raw_dataset_folder, 'labels.csv')
    df.to_csv(labels_csv_path, index=False)

    print(f"Created labels.csv with {len(labels_data)} cases")
    print(f"Distribution: {df['subtype'].value_counts().to_dict()}")

    return labels_csv_path


def copy_labels_to_preprocessed(dataset_id: int):
    """Copy labels.csv from raw to preprocessed folder"""
    dataset_name = maybe_convert_to_dataset_name(dataset_id)

    raw_labels_path = join(nnUNet_raw, dataset_name, 'labels.csv')
    preprocessed_folder = join(nnUNet_preprocessed, dataset_name)
    preprocessed_labels_path = join(preprocessed_folder, 'labels.csv')

    if os.path.exists(raw_labels_path) and os.path.exists(preprocessed_folder):
        shutil.copy2(raw_labels_path, preprocessed_labels_path)
        print(f"Copied labels.csv to {preprocessed_labels_path}")
    else:
        print(f"Warning: Could not copy labels.csv")
        print(f"Raw path exists: {os.path.exists(raw_labels_path)}")
        print(f"Preprocessed folder exists: {os.path.exists(preprocessed_folder)}")


def test_plan_and_preprocess(dataset_id: int,
                           configurations: list = None,
                           verify_dataset_integrity: bool = True,
                           extract_fingerprints_flag: bool = True,
                           clean: bool = False,
                           verbose: bool = True):
    """
    Test plan_and_preprocess with custom planner using individual API functions
    """
    if configurations is None:
        configurations = ['3d_fullres']

    print(f"\n{'='*50}")
    print(f"Testing plan_and_preprocess for dataset {dataset_id}")
    print(f"Configurations: {configurations}")
    print(f"{'='*50}")

    try:
        # Step 1: Extract fingerprints
        if extract_fingerprints_flag:
            print("Step 1: Extracting fingerprints...")
            extract_fingerprints(
                dataset_ids=[dataset_id],
                fingerprint_extractor_class_name='DatasetFingerprintExtractor',
                num_processes=8,
                check_dataset_integrity=verify_dataset_integrity,
                clean=clean,
                verbose=verbose
            )

        # Step 2: Plan experiments
        print("Step 2: Planning experiments...")
        # Import and use planner directly since recursive_find_python_class can't find our custom class
        from src.experiment_planning.multitask_residual_encoder_planner import MultiTasknnUNetPlannerResEncM
        from nnunetv2.experiment_planning.plan_and_preprocess_api import plan_experiment_dataset

        _, plans_identifier = plan_experiment_dataset(
            dataset_id=dataset_id,
            experiment_planner_class=MultiTasknnUNetPlannerResEncM,  # Our custom planner
            gpu_memory_target_in_gb=24.0,
            preprocess_class_name='DefaultPreprocessor',
            overwrite_target_spacing=None,
            overwrite_plans_name='nnUNetPlans_multitask'
        )

        # Step 3: Preprocess
        print("Step 3: Preprocessing...")
        # Determine number of processes for each config
        default_np = {"2d": 8, "3d_fullres": 4, "3d_lowres": 8}
        num_processes = [default_np.get(c, 4) for c in configurations]

        preprocess(
            dataset_ids=[dataset_id],
            plans_identifier=plans_identifier,
            configurations=configurations,
            num_processes=num_processes,
            verbose=verbose
        )

        print("✓ plan_and_preprocess completed successfully")

        # Copy labels.csv to preprocessed folder
        copy_labels_to_preprocessed(dataset_id)

        return True

    except Exception as e:
        print(f"✗ plan_and_preprocess failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader(dataset_id: int, configuration: str = '3d_fullres'):
    """Test custom dataloader with preprocessed data"""
    from src.training.dataloading.multitask_dataset import MultiTasknnUNetDataset

    dataset_name = maybe_convert_to_dataset_name(dataset_id)
    preprocessed_folder = join(nnUNet_preprocessed, dataset_name, 'MultiTasknnUNetPlannerResEncM__nnUNetPlans_multitask', configuration)

    if not os.path.exists(preprocessed_folder):
        print(f"Preprocessed folder not found: {preprocessed_folder}")
        return False

    try:
        # Get case identifiers
        from nnunetv2.training.dataloading.nnunet_dataset import nnUNetBaseDataset
        case_identifiers = nnUNetBaseDataset.get_identifiers(preprocessed_folder)

        print(f"Found {len(case_identifiers)} cases in {preprocessed_folder}")

        # Test dataloader
        dataset = MultiTasknnUNetDataset(
            folder=preprocessed_folder,
            identifiers=case_identifiers[:5],  # Test first 5 cases
            folder_with_segs_from_previous_stage=None
        )

        print(f"Classification distribution: {dataset._get_label_distribution(dataset.classification_labels)}")

        # Test loading a case
        if len(case_identifiers) > 0:
            test_case = case_identifiers[0]
            data, seg, seg_prev, properties, classification = dataset.load_case(test_case)

            print(f"Test case {test_case}:")
            print(f"  Data shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")
            print(f"  Seg shape: {seg.shape if hasattr(seg, 'shape') else 'N/A'}")
            print(f"  Classification: {classification}")
            print(f"  Properties keys: {list(properties.keys()) if properties else 'N/A'}")

        print("✓ Dataloader test completed successfully")
        return True

    except Exception as e:
        print(f"✗ Dataloader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Multi-Task Pancreas Segmentation')
    parser.add_argument('--dataset_id', type=int, default=1, help='Dataset ID')
    parser.add_argument('--mode', choices=['plan', 'test_data', 'train', 'inference'],
                       default='plan', help='Mode to run')
    parser.add_argument('--configurations', nargs='+', default=['3d_fullres'],
                       help='Configurations to process')
    parser.add_argument('--clean', action='store_true', help='Clean previous preprocessing')
    parser.add_argument('--verbose', action='store_true', default=True, help='Verbose output')

    args = parser.parse_args()

    print("Multi-Task Pancreatic Cancer Segmentation")
    print("==========================================")

    # Setup environment
    # setup_environment() # REMOVE: doesn't work

    # Register components
    print("Registering custom nnUNet components...")
    register_custom_components()

    if args.mode == 'plan':
        print("\nMode: Plan and Preprocess")
        success = test_plan_and_preprocess(
            dataset_id=args.dataset_id,
            configurations=args.configurations,
            clean=args.clean,
            verbose=args.verbose
        )

        if success:
            print("\nTesting dataloader...")
            test_dataloader(args.dataset_id, args.configurations[0])

    elif args.mode == 'test_data':
        print("\nMode: Test Dataloader")
        test_dataloader(args.dataset_id, args.configurations[0])

    elif args.mode == 'train':
        print("\nMode: Training (TODO)")
        # TODO: Implement training with nnUNetTrainerMultiTask
        pass

    elif args.mode == 'inference':
        print("\nMode: Inference (TODO)")
        # TODO: Implement inference pipeline
        pass


if __name__ == '__main__':
    main()