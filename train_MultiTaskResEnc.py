#!/usr/bin/env python3
"""
train_MultiTaskResEnc.py

Complete training script for MultiTask ResEnc U-Net model.
This script performs two main tasks:
1. Preprocesses and generates plans according to the MultiTask_base_planner
2. Runs training and saves model artifacts with the given configuration and plan

The script uses the simplest multitask model without attention (MultiTaskResEncUNet)
and can be run either with subprocess calls or direct function calls.
"""

import os
import sys
import subprocess
import traceback
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup nnUNet environment variables"""
    env_vars = os.environ.copy()
    env_vars['nnUNet_raw'] = "/mnt/data/gpu-server/nnUNet_modified/nnunet_data/nnUNet_raw"
    env_vars['nnUNet_preprocessed'] = "/mnt/data/gpu-server/nnUNet_modified/nnunet_data/nnUNet_preprocessed"
    env_vars['nnUNet_results'] = "/mnt/data/gpu-server/nnUNet_modified/nnunet_data/nnUNet_results"

    # Ensure directories exist
    for path in [env_vars['nnUNet_raw'], env_vars['nnUNet_preprocessed'], env_vars['nnUNet_results']]:
        Path(path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {path}")

    return env_vars

def run_subprocess_with_logging(cmd, env_vars, stage_name):
    """Run subprocess with proper logging and error handling"""
    logger.info(f"Starting {stage_name}...")
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(
            cmd,
            env=env_vars,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Stream output in real-time
        while True:
            stdout_line = process.stdout.readline() if process.stdout is not None else None
            stderr_line = process.stderr.readline() if process.stderr is not None else None

            if stdout_line:
                logger.info(f"STDOUT: {stdout_line.strip()}")
            if stderr_line:
                logger.info(f"STDERR: {stderr_line.strip()}")

            if process.poll() is not None:
                break

        # Get remaining output
        stdout, stderr = process.communicate()

        if stdout:
            for line in stdout.splitlines():
                logger.info(f"STDOUT: {line}")
        if stderr:
            for line in stderr.splitlines():
                logger.info(f"STDERR: {line}")

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

        logger.info(f"✅ {stage_name} completed successfully")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {stage_name} failed with return code {e.returncode}")
        logger.error(f"Command: {' '.join(cmd)}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error during {stage_name}: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def preprocess_and_plan(env_vars, dataset_id="1", configuration="3d_fullres",
                       planner="MultiTaskResEncUNetPlanner", num_processes=8):
    """
    Step 1: Preprocess and generate plan according to the MultiTask_base_planner

    Args:
        env_vars (dict): Environment variables
        dataset_id (str): Dataset ID (default: "1")
        configuration (str): Configuration name (default: "3d_fullres")
        planner (str): Planner class name (default: "MultiTaskResEncUNetPlanner")
        num_processes (int): Number of processes for preprocessing
    """
    logger.info("="*60)
    logger.info("STEP 1: PREPROCESSING AND PLANNING")
    logger.info("="*60)

    # Add src to Python path for custom modules
    src_path = Path(__file__).parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
        logger.info(f"Added {src_path} to Python path")

    cmd = [
        "uv", "run", "--extra", "cu124",
        "nnUNetv2_plan_and_preprocess",
        "-pl", planner,  # Use our custom MultiTask planner
        "-d", dataset_id,
        "-c", configuration,
        "-npfp", str(num_processes),
        "--verify_dataset_integrity"
    ]

    return run_subprocess_with_logging(cmd, env_vars, "Preprocessing and Planning")

def train_model(env_vars, dataset_id="1", configuration="3d_fullres",
                planner="MultiTaskResEncUNetPlanner", trainer="nnUNetTrainerMultiTask",
                fold="0", continue_training=False):
    """
    Step 2: Run training script and save model artifacts

    Args:
        env_vars (dict): Environment variables
        dataset_id (str): Dataset ID
        configuration (str): Configuration name
        planner (str): Planner class name
        trainer (str): Trainer class name
        fold (str): Fold number for cross-validation
        continue_training (bool): Whether to continue from existing checkpoint
    """
    logger.info("="*60)
    logger.info("STEP 2: TRAINING")
    logger.info("="*60)

    cmd = [
        "uv", "run", "--extra", "cu124",
        "nnUNetv2_train",
        "-d", dataset_id,
        "-c", configuration,
        "-p", planner,    # Our custom planner
        "-tr", trainer,   # Our custom multi-task trainer
        "-f", fold
    ]

    if continue_training:
        cmd.append("--c")
        logger.info("Continue training from existing checkpoint enabled")

    return run_subprocess_with_logging(cmd, env_vars, "Training")

# def direct_training_approach(dataset_id="1", configuration="3d_fullres", fold=0):
#     """
#     Alternative approach: Direct training without subprocess calls

#     This approach directly imports and uses the nnUNet classes for more control
#     and better debugging capabilities.
#     """
#     logger.info("="*60)
#     logger.info("ALTERNATIVE: DIRECT TRAINING APPROACH")
#     logger.info("="*60)

#     try:
#         # Add src to Python path
#         src_path = Path(__file__).parent / "src"
#         if str(src_path) not in sys.path:
#             sys.path.insert(0, str(src_path))

#         # Import required modules
#         from src.planners.multitask_base_planner import MultiTaskResEncUNetPlanner
#         from src.trainers.nnUNetTrainerMultiTask import nnUNetTrainerMultiTask
#         from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

#         logger.info("✅ Successfully imported custom modules")

#         # Step 1: Planning and preprocessing
#         logger.info("Starting planning and preprocessing...")
#         dataset_name = maybe_convert_to_dataset_name(dataset_id)
#         planner = MultiTaskResEncUNetPlanner(dataset_name, 8)  # 8GB GPU memory target

#         logger.info(f"Planning experiment for dataset: {dataset_name}")
#         planner.plan_experiment()
#         logger.info("✅ Planning completed")

#         planner.run_preprocessing()
#         logger.info("✅ Preprocessing completed")

#         # Step 2: Training
#         logger.info("Starting training...")
#         trainer = nnUNetTrainerMultiTask(
#             plans=planner.plans,
#             configuration=configuration,
#             fold=fold,
#             dataset_json=planner.dataset_json,
#             device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         )

#         logger.info("✅ Trainer initialized")

#         # Run training
#         trainer.run_training()
#         logger.info("✅ Training completed successfully")

#         return True

#     except ImportError as e:
#         logger.error(f"❌ Import error: {e}")
#         logger.error("Falling back to subprocess approach...")
#         return False
#     except Exception as e:
#         logger.error(f"❌ Error in direct training: {e}")
#         logger.error(traceback.format_exc())
#         return False

def main():
    """
    Main execution function

    The script will attempt direct training first, and fall back to subprocess
    calls if there are import issues or other problems.
    """
    logger.info("🚀 Starting MultiTask ResEnc U-Net Training")
    logger.info("="*60)

    # Configuration
    DATASET_ID = "1"
    CONFIGURATION = "3d_fullres"
    PLANNER = "MultiTaskResEncUNetPlanner"  # Our custom multi-task planner
    PLAN = "nnUNetMultiTaskResEncUNetPlans"
    TRAINER = "nnUNetTrainerMultiTask"      # Our custom multi-task trainer
    FOLD = "0"
    NUM_PROCESSES = 8

    try:
        # Setup environment
        env_vars = setup_environment()
        logger.info("✅ Environment setup completed")

        # # Method 1: Try direct approach first (better for debugging)
        # logger.info("Attempting direct training approach...")
        # if direct_training_approach(DATASET_ID, CONFIGURATION, int(FOLD)):
        #     logger.info("🎉 Training completed successfully using direct approach!")
        #     return

        # Method 2: Fall back to subprocess approach
        # logger.info("Using subprocess approach...")

        # Step 1: Preprocessing and Planning
        success = preprocess_and_plan(
            env_vars=env_vars,
            dataset_id=DATASET_ID,
            configuration=CONFIGURATION,
            planner=PLANNER,
            num_processes=NUM_PROCESSES
        )

        if not success:
            logger.error("❌ Preprocessing failed")
            return

        # Step 2: Training
        success = train_model(
            env_vars=env_vars,
            dataset_id=DATASET_ID,
            configuration=CONFIGURATION,
            planner=PLANNER,
            trainer=TRAINER,
            fold=FOLD,
            continue_training=False
        )

        if success:
            logger.info("🎉 Training pipeline completed successfully!")
            logger.info(f"Model artifacts saved in: {env_vars['nnUNet_results']}")
        else:
            logger.error("❌ Training failed")

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()