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
import torch

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
                           verbose: bool = False):
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
            verbose=verbose,
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
    preprocessed_folder = join(nnUNet_preprocessed, dataset_name, 'nnUNetPlans_multitask_'+configuration)

    if not os.path.exists(preprocessed_folder):
        print(f"Preprocessed folder not found: {preprocessed_folder}")
        return False

    try:
        # Get case identifiers
        from src.training.dataloading.multitask_dataset import MultiTasknnUNetDataset
        case_identifiers = MultiTasknnUNetDataset.get_identifiers(preprocessed_folder)

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
            data, seg, seg_prev, properties = dataset.load_case(test_case)

            print(f"Test case {test_case}:")
            print(f"  Data shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")
            print(f"  Seg shape: {seg.shape if hasattr(seg, 'shape') else 'N/A'}")
            print(f"  Classification: {properties.get('classification_label')}")
            print(f"  Properties keys: {list(properties.keys()) if properties else 'N/A'}")

        print("✓ Dataloader test completed successfully")
        return True

    except Exception as e:
        print(f"✗ Dataloader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_multitask_model(dataset_id: int,
                         configuration: str = '3d_fullres',
                         fold: int = 0,
                         continue_training: bool = False,
                         only_run_validation: bool = False,
                         num_epochs: int = 200,
                         use_compressed_data: bool = False,
                         export_validation_probabilities: bool = True,
                         val_disable_overwrite: bool = False,
                         disable_checkpointing: bool = False,
                         device: str = 'cuda',
                         custom_stage_epochs: list = None):
    """
    Train multi-task model using custom trainer
    """
    print(f"\n{'='*50}")
    print(f"Training Multi-Task Model")
    print(f"Dataset: {dataset_id}, Config: {configuration}, Fold: {fold}")
    print(f"{'='*50}")

    try:
        # Import custom trainer
        from src.training.multitask_trainer import nnUNetTrainerMultiTask

        dataset_name = maybe_convert_to_dataset_name(dataset_id)
        plans_identifier = 'nnUNetPlans_multitask'

        # Check if preprocessed data exists
        preprocessed_folder = join(nnUNet_preprocessed, dataset_name, f'{plans_identifier}_{configuration}')
        if not os.path.exists(preprocessed_folder):
            raise RuntimeError(f"Preprocessed data not found: {preprocessed_folder}")

        # Load plans and dataset json
        plans_file = join(nnUNet_preprocessed, dataset_name, f'{plans_identifier}.json')
        dataset_json_file = join(nnUNet_preprocessed, dataset_name, 'dataset.json')

        if not os.path.exists(plans_file):
            raise RuntimeError(f"Plans file not found: {plans_file}")
        if not os.path.exists(dataset_json_file):
            raise RuntimeError(f"Dataset json not found: {dataset_json_file}")

        from batchgenerators.utilities.file_and_folder_operations import load_json
        plans = load_json(plans_file)
        dataset_json = load_json(dataset_json_file)

        # Initialize trainer
        trainer = nnUNetTrainerMultiTask(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            device=torch.device(device),
        )
        if custom_stage_epochs is not None:
            trainer.set_custom_stage_epochs(custom_stage_epochs)

        # Set up output folder
        output_folder = join(nnUNet_results, dataset_name, str('nnUNetTrainerMultiTask__' + plans_identifier + '_' + configuration))
        trainer.output_folder = join(output_folder, f'fold_{fold}')
        maybe_mkdir_p(trainer.output_folder)

        # Set training parameters
        trainer.num_epochs = num_epochs
        trainer.save_every = 25

        print(f"Output folder: {trainer.output_folder}")
        print(f"Training epochs: {num_epochs}")

        if only_run_validation:
            print("Running validation only...")
            trainer.load_checkpoint(join(trainer.output_folder, 'checkpoint_final.pth'))
            trainer.run_validation()

        else:
            # Check for existing checkpoint
            if continue_training and os.path.exists(join(trainer.output_folder, 'checkpoint_latest.pth')):
                print("Continuing training from latest checkpoint...")
                trainer.load_checkpoint(join(trainer.output_folder, 'checkpoint_latest.pth'))

            # Run training
            print("Starting training...")
            trainer.run_training()

            print("Training completed successfully!")

        return True

    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_inference(dataset_id: int,
                 configuration: str = '3d_fullres',
                 fold: int = 0,
                 input_folder: str = None,
                 output_folder: str = None,
                 save_probabilities: bool = False,
                 use_folds: str = 'all',
                 step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_device: bool = False,
                 verbose: bool = False,
                 overwrite_existing: bool = True):
    """
    Run inference with trained multi-task model
    """
    print(f"\n{'='*50}")
    print(f"Running Multi-Task Inference")
    print(f"Dataset: {dataset_id}, Config: {configuration}")
    print(f"{'='*50}")
    return
    try:
        from src.training.multitask_trainer import nnUNetTrainerMultiTask
        from nnunetv2.inference.predict import nnUNetPredictor

        dataset_name = maybe_convert_to_dataset_name(dataset_id)
        plans_identifier = 'nnUNetPlans_multitask'

        # Set default paths if not provided
        if input_folder is None:
            input_folder = join(nnUNet_raw, dataset_name, 'imagesTs')
        if output_folder is None:
            output_folder = join(nnUNet_results, dataset_name, 'predictions')

        maybe_mkdir_p(output_folder)

        # Model folder
        model_folder = join(nnUNet_results, dataset_name, 'nnUNetTrainerMultiTask__' + plans_identifier, configuration)

        if not os.path.exists(model_folder):
            raise RuntimeError(f"Model folder not found: {model_folder}")

        # Use nnUNet predictor
        predictor = nnUNetPredictor(
            tile_step_size=step_size,
            use_gaussian=use_gaussian,
            use_mirroring=use_mirroring,
            perform_everything_on_device=perform_everything_on_device,
            verbose=verbose,
            verbose_preprocessing=verbose,
            allow_tqdm=True
        )

        # Initialize predictor
        predictor.initialize_from_trained_model_folder(
            model_folder,
            use_folds=use_folds,
            checkpoint_name='checkpoint_final.pth'
        )

        # Run prediction
        predictor.predict_from_files(
            list_of_lists_or_source_folder=input_folder,
            output_folder_or_list_of_truncated_output_files=output_folder,
            save_probabilities=save_probabilities,
            overwrite=overwrite_existing,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2,
            folder_with_segs_from_previous_stage=None,
            num_parts=1,
            part_id=0
        )

        print(f"✓ Inference completed successfully")
        print(f"Results saved to: {output_folder}")

        return True

    except Exception as e:
        print(f"✗ Inference failed: {e}")
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
    parser.add_argument('--verbose', action='store_true', default=False, help='Verbose output')

    # Training arguments
    parser.add_argument('--fold', type=int, default=0, help='Cross-validation fold')
    parser.add_argument('--continue_training', action='store_true', help='Continue from checkpoint')
    parser.add_argument('--validation_only', action='store_true', help='Run validation only')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--device', default='cuda', help='Training device')

    # Inference arguments
    parser.add_argument('--input_folder', type=str, help='Input folder for inference')
    parser.add_argument('--output_folder', type=str, help='Output folder for inference')
    parser.add_argument('--save_probabilities', action='store_true', help='Save prediction probabilities')
    parser.add_argument('--step_size', type=float, default=0.5, help='Step size for sliding window')

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
        print("\nMode: Training")
        success = train_multitask_model(
            dataset_id=args.dataset_id,
            configuration=args.configurations[0],
            fold=args.fold,
            continue_training=args.continue_training,
            only_run_validation=args.validation_only,
            num_epochs=args.num_epochs,
            device=args.device,
            custom_stage_epochs=[0, 0, 0, 100]  # You can specify custom epochs per stage if needed
        )

        if success:
            print("\n✓ Training completed successfully!")
            print("You can now run inference using --mode inference")

    elif args.mode == 'inference':
        print("\nMode: Inference")
        success = run_inference(
            dataset_id=args.dataset_id,
            configuration=args.configurations[0],
            fold=args.fold,
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            save_probabilities=args.save_probabilities,
            step_size=args.step_size,
            verbose=args.verbose
        )
        if success:
            print("\n✓ Inference completed successfully!")


if __name__ == '__main__':
    main()