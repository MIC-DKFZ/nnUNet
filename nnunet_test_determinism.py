import os
import subprocess
import numpy as np
import SimpleITK as sitk
import torch
from batchgenerators.utilities.file_and_folder_operations import save_json, load_json, join, isfile
import shutil
import sys

import pickle

DUMMY_DATASET_ID = 1
NUM_TRAINING_SAMPLES = 5
NUM_EPOCHS_FOR_TEST = 2

def get_best_device():
    """Automatically selects the best available device."""
    # if torch.cuda.is_available():
    #     print("✅ CUDA is available! Using GPU.")
    #     return torch.device("cuda")
    # # The check for MPS needs to be more robust, especially for older torch versions
    # if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #     print("✅ Apple MPS is available! Using GPU.")
    #     return torch.device("mps")
    # print("⚠️ No GPU detected. Falling back to CPU.")
    return torch.device("cpu")

def setup_environment_and_directories():
    """Creates the necessary directories and sets the environment variables."""
    print("--- Setting up environment and directories ---")
    project_root = os.getcwd()
    if not isfile(os.path.join(project_root, 'setup.py')):
        print("\n❌ Error: This script must be run from the root directory of the nnUNet repository.")
        print(f"Please 'cd' to your nnUNet folder and run it from there.")
        sys.exit()

    sys.path.insert(0, project_root)

    raw_data_dir = os.path.join(project_root, "nnUNet_raw")
    preprocessed_dir = os.path.join(project_root, "nnUNet_preprocessed")
    results_dir = os.path.join(project_root, "nnUNet_results")

    os.makedirs(os.path.join(raw_data_dir, f"Dataset{DUMMY_DATASET_ID:03d}_Dummy", "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(raw_data_dir, f"Dataset{DUMMY_DATASET_ID:03d}_Dummy", "labelsTr"), exist_ok=True)
    os.makedirs(preprocessed_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    os.environ["nnUNet_raw"] = raw_data_dir
    os.environ["nnUNet_preprocessed"] = preprocessed_dir
    os.environ["nnUNet_results"] = results_dir
    print("Environment variables set.")
    print("-" * 50)


def generate_dummy_dataset():
    """Generates a synthetic dataset and the corresponding dataset.json file."""
    print("--- Generating dummy dataset ---")
    dataset_folder = os.path.join(os.environ["nnUNet_raw"], f"Dataset{DUMMY_DATASET_ID:03d}_Dummy")
    images_tr_dir = os.path.join(dataset_folder, "imagesTr")
    labels_tr_dir = os.path.join(dataset_folder, "labelsTr")

    for i in range(NUM_TRAINING_SAMPLES):
        case_identifier = f"dummy_{i:03d}"
        image_filename = os.path.join(images_tr_dir, f"{case_identifier}_0000.nii.gz")
        label_filename = os.path.join(labels_tr_dir, f"{case_identifier}.nii.gz")

        image = np.random.rand(5, 32, 32).astype(np.float32)
        image_sitk = sitk.GetImageFromArray(image)
        image_sitk.SetSpacing([9.7, 1.5, 1.5])
        sitk.WriteImage(image_sitk, image_filename)

        seg = (image > 0.3).astype(np.uint8)
        seg_sitk = sitk.GetImageFromArray(seg)
        seg_sitk.SetSpacing([9.7, 1.5, 1.5])
        sitk.WriteImage(seg_sitk, label_filename)

    print(f"Generated {NUM_TRAINING_SAMPLES} training samples.")

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "foreground": 1},
        "numTraining": NUM_TRAINING_SAMPLES,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
        "dataset_name": "Dataset001_Dummy"
    }
    save_json(dataset_json, os.path.join(dataset_folder, "dataset.json"))
    print("dataset.json created.")
    print("-" * 50)


def run_preprocessing():
    """Executes the nnU-Net preprocessing command using the current python interpreter."""
    print("--- Running nnU-Net preprocessing ---")

    command = [
        sys.executable,
        "-m", "nnunetv2.experiment_planning.plan_and_preprocess_entrypoints",
        "-d", str(DUMMY_DATASET_ID),
        "-c", "2d", "3d_fullres",
    ]

    try:
        subprocess.run(command, check=True, timeout=300)
        print("Preprocessing completed successfully.")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error during preprocessing: {e}")
        print("Please ensure nnU-Net's dependencies are installed in your virtual environment.")
        sys.exit()
    except subprocess.TimeoutExpired:
        print("Error: Preprocessing timed out after 5 minutes.")
        sys.exit()
    print("-" * 50)


def run_training_for_one_config(config, output_folder_name, device):
    """Runs a few epochs of training with deterministic settings for a single configuration."""
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    plans_file = join(os.environ["nnUNet_preprocessed"], f"Dataset{DUMMY_DATASET_ID:03d}_Dummy", "nnUNetPlans.json")
    dataset_json_file = join(os.environ["nnUNet_raw"], f"Dataset{DUMMY_DATASET_ID:03d}_Dummy", "dataset.json")

    plans = load_json(plans_file)
    dataset_json = load_json(dataset_json_file)

    trainer = nnUNetTrainer(plans=plans,
                            configuration=config,
                            fold=0,
                            dataset_json=dataset_json,
                            deterministic=True,
                            device=device)

    trainer.num_epochs = NUM_EPOCHS_FOR_TEST
    trainer.output_folder = join(os.environ["nnUNet_results"], output_folder_name)

    if os.path.isdir(trainer.output_folder):
        shutil.rmtree(trainer.output_folder)
    os.makedirs(trainer.output_folder)

    print(f"Starting training run: {output_folder_name} for config '{config}' on device: {device}")
    trainer.run_training()
    print(f"Finished training run: {output_folder_name}")
    return trainer.output_folder


def compare_checkpoints(ckpt1_path, ckpt2_path):
    """Compares two checkpoints for equality."""
    print("\n--- Comparing checkpoints ---")

    try:
        ckpt1 = torch.load(ckpt1_path, map_location='cpu', weights_only=False)
        ckpt2 = torch.load(ckpt2_path, map_location='cpu', weights_only=False)
    except pickle.UnpicklingError as e:
        print(f"Error loading checkpoint: {e}")
        print("This might happen if the checkpoint file is corrupted.")
        return False


    weights1 = ckpt1['network_weights']
    weights2 = ckpt2['network_weights']
    weights_are_equal = all(torch.equal(weights1[key], weights2[key]) for key in weights1.keys())
    print(f"Model weights are identical: {weights_are_equal}")

    log1 = ckpt1['logging']
    log2 = ckpt2['logging']
    train_loss_equal = np.allclose(log1['train_losses'], log2['train_losses'])
    val_loss_equal = np.allclose(log1['val_losses'], log2['val_losses'])
    print(f"Training losses are identical: {train_loss_equal}")
    print(f"Validation losses are identical: {val_loss_equal}")

    return weights_are_equal and train_loss_equal and val_loss_equal

def run_test_for_config(config_name, device):
    """Runs the full determinism test for a given configuration and returns the result."""
    print(f"\n{'='*10} STARTING DETERMINISM TEST FOR CONFIGURATION: {config_name.upper()} {'='*10}")

    plans_file = join(os.environ["nnUNet_preprocessed"], f"Dataset{DUMMY_DATASET_ID:03d}_Dummy", "nnUNetPlans.json")
    if not isfile(plans_file):
        print(f"Error: nnUNetPlans.json not found. Cannot proceed.")
        return "ERROR"

    plans = load_json(plans_file)
    if config_name not in plans['configurations']:
        print(f"⚪ SKIPPED: Configuration '{config_name}' not found in nnUNetPlans.json.")
        return "SKIPPED"

    output_folder1 = run_training_for_one_config(config_name, f"run1_{config_name}", device)
    output_folder2 = run_training_for_one_config(config_name, f"run2_{config_name}", device)

    checkpoint1_path = join(output_folder1, "checkpoint_final.pth")
    checkpoint2_path = join(output_folder2, "checkpoint_final.pth")

    if not isfile(checkpoint1_path) or not isfile(checkpoint2_path):
        print(f"❌ FAILED: Could not find final checkpoints for {config_name}. Training might have failed.")
        return "FAILED"

    passed = compare_checkpoints(checkpoint1_path, checkpoint2_path)

    if passed:
        print(f"✅ PASSED: The training process for '{config_name}' is deterministic.")
    else:
        print(f"❌ FAILED: The training process for '{config_name}' is not deterministic.")

    print(f"{'='*10} FINISHED TEST FOR CONFIGURATION: {config_name.upper()} {'='*10}")
    return "PASSED" if passed else "FAILED"

def main():
    """Main function to run the entire pipeline."""
    setup_environment_and_directories()

    try:
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    except ImportError:
        print("\n❌ Error: Could not import nnUNetTrainer. ")
        print("Please make sure you are running this script from the root of the nnUNet repository.")
        sys.exit()

    device = get_best_device()

    generate_dummy_dataset()
    run_preprocessing()

    results = {}
    results["2d"] = run_test_for_config("2d", device)
    results["3d_fullres"] = run_test_for_config("3d_fullres", device)
    

    print("\n\n" + "="*25)
    print("  DETERMINISM TEST REPORT")
    print("="*25)
    for config, result in results.items():
        if result == "PASSED":
            icon = "✅"
        elif result == "FAILED":
            icon = "❌"
        elif result == "SKIPPED":
            icon = "⚪"
        else:
            icon = "❗"
        print(f"Configuration '{config}': {icon} {result}")
    print("="*25)


if __name__ == "__main__":
    main()