"""
PyTorch Lightning-based training script for nnUNet.

This script provides a Lightning-based alternative to the standard nnUNet training,
leveraging Lightning's built-in features for:
- Multi-GPU training (DDP)
- Mixed precision training (automatic)
- Model checkpointing (ModelCheckpoint callback)
- Logging (WandB, TensorBoard, etc.)
- Fault tolerance and resumption

While maintaining nnUNet's excellent out-of-the-box performance.
"""

import os
import multiprocessing
import argparse
from typing import Union, Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

import nnunetv2
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.training.nnUNetTrainer.nnUNetLightningModule import nnUNetLightningModule
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class


def get_lightning_module_from_args(
    dataset_name_or_id: Union[int, str],
    configuration: str,
    fold: int,
    trainer_name: str = 'nnUNetLightningModule',
    plans_identifier: str = 'nnUNetPlans',
    device: torch.device = torch.device('cuda')
) -> nnUNetLightningModule:
    """
    Create and return a nnUNet Lightning module based on the provided arguments.

    Args:
        dataset_name_or_id: Dataset name (e.g., 'Dataset001_BrainTumour') or ID
        configuration: Configuration name (e.g., '3d_fullres', '2d')
        fold: Cross-validation fold number
        trainer_name: Name of the trainer class to use
        plans_identifier: Plans file identifier
        device: PyTorch device (note: Lightning will override this)

    Returns:
        Initialized nnUNetLightningModule
    """
    # Load the trainer class (LightningModule)
    # First try to find it as a Lightning module
    nnunet_trainer_class = recursive_find_python_class(
        join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        trainer_name,
        'nnunetv2.training.nnUNetTrainer'
    )

    if nnunet_trainer_class is None:
        # If not found, default to nnUNetLightningModule
        print(f"Could not find {trainer_name}, using nnUNetLightningModule as default")
        nnunet_trainer_class = nnUNetLightningModule

    # Verify it's a LightningModule
    if not issubclass(nnunet_trainer_class, pl.LightningModule):
        raise RuntimeError(
            f'The requested trainer class {trainer_name} must inherit from '
            f'pl.LightningModule (PyTorch Lightning). For Lightning-based training, '
            f'use nnUNetLightningModule or a custom LightningModule-based trainer.'
        )

    # Handle dataset input
    if isinstance(dataset_name_or_id, str) and dataset_name_or_id.startswith('Dataset'):
        pass
    else:
        try:
            dataset_name_or_id = int(dataset_name_or_id)
        except ValueError:
            raise ValueError(
                f'dataset_name_or_id must either be an integer or a valid dataset name with the pattern '
                f'DatasetXXX_YYY where XXX are the three(!) task ID digits. Your input: {dataset_name_or_id}'
            )

    # Load plans and dataset info
    preprocessed_dataset_folder_base = join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_name_or_id))
    plans_file = join(preprocessed_dataset_folder_base, plans_identifier + '.json')
    plans = load_json(plans_file)
    dataset_json = load_json(join(preprocessed_dataset_folder_base, 'dataset.json'))

    # Create the Lightning module
    lightning_module = nnunet_trainer_class(
        plans=plans,
        configuration=configuration,
        fold=fold,
        dataset_json=dataset_json,
        device=device
    )

    return lightning_module


def setup_callbacks(
    output_folder: str,
    save_every_n_epochs: int = 50,
    monitor_metric: str = 'val_loss',
) -> list:
    """
    Setup Lightning callbacks for training.

    Args:
        output_folder: Directory to save checkpoints
        save_every_n_epochs: Save checkpoint every N epochs
        monitor_metric: Metric to monitor for best checkpoint

    Returns:
        List of Lightning callbacks
    """
    callbacks = []

    # ModelCheckpoint for best model (based on validation loss)
    checkpoint_callback_best = ModelCheckpoint(
        dirpath=output_folder,
        filename='checkpoint_best',
        monitor=monitor_metric,
        mode='min',
        save_top_k=1,
        save_last=False,
        verbose=True,
    )
    callbacks.append(checkpoint_callback_best)

    # ModelCheckpoint for latest model
    checkpoint_callback_last = ModelCheckpoint(
        dirpath=output_folder,
        filename='checkpoint_latest',
        monitor=None,
        save_top_k=1,
        save_last=True,
        every_n_epochs=save_every_n_epochs,
        verbose=False,
    )
    callbacks.append(checkpoint_callback_last)

    # ModelCheckpoint for final model (saved at the end)
    checkpoint_callback_final = ModelCheckpoint(
        dirpath=output_folder,
        filename='checkpoint_final',
        monitor=None,
        save_top_k=1,
        save_last=True,
        verbose=False,
    )
    callbacks.append(checkpoint_callback_final)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)

    # Progress bar
    progress_bar = RichProgressBar()
    callbacks.append(progress_bar)

    return callbacks


def setup_logger(
    logger_type: str = 'tensorboard',
    output_folder: str = None,
    experiment_name: str = None,
    **kwargs
):
    """
    Setup Lightning logger.

    Args:
        logger_type: Type of logger ('tensorboard', 'wandb', or 'none')
        output_folder: Directory for logs
        experiment_name: Name of the experiment
        **kwargs: Additional logger-specific arguments

    Returns:
        Lightning logger instance or None
    """
    if logger_type == 'none':
        return None
    elif logger_type == 'tensorboard':
        return TensorBoardLogger(
            save_dir=output_folder,
            name=experiment_name,
            version='lightning',
        )
    elif logger_type == 'wandb':
        return WandbLogger(
            project=kwargs.get('wandb_project', 'nnUNet'),
            name=experiment_name,
            save_dir=output_folder,
            log_model=False,  # We handle checkpointing ourselves
        )
    else:
        raise ValueError(f"Unknown logger type: {logger_type}. Choose from 'tensorboard', 'wandb', or 'none'")


def run_training_lightning(
    dataset_name_or_id: Union[str, int],
    configuration: str,
    fold: Union[int, str],
    trainer_class_name: str = 'nnUNetLightningModule',
    plans_identifier: str = 'nnUNetPlans',
    pretrained_weights: Optional[str] = None,
    num_gpus: int = 1,
    num_epochs: int = 1000,
    continue_training: bool = False,
    logger_type: str = 'tensorboard',
    wandb_project: Optional[str] = None,
    precision: str = '16-mixed',
    device: str = 'cuda',
):
    """
    Run nnUNet training using PyTorch Lightning.

    Args:
        dataset_name_or_id: Dataset name or ID
        configuration: Configuration name (e.g., '3d_fullres')
        fold: Cross-validation fold
        trainer_class_name: Lightning module class name
        plans_identifier: Plans identifier
        pretrained_weights: Path to pretrained weights (optional)
        num_gpus: Number of GPUs to use
        num_epochs: Number of training epochs
        continue_training: Whether to continue from checkpoint
        logger_type: Type of logger ('tensorboard', 'wandb', 'none')
        wandb_project: WandB project name (if using wandb)
        precision: Training precision ('32', '16-mixed', 'bf16-mixed')
        device: Device type ('cuda', 'cpu', 'mps')
    """
    if plans_identifier == 'nnUNetPlans':
        print("\n############################\n"
              "INFO: You are using the old nnU-Net default plans. We have updated our recommendations. "
              "Please consider using those instead! "
              "Read more here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md"
              "\n############################\n")

    # Handle fold conversion
    if isinstance(fold, str):
        if fold != 'all':
            try:
                fold = int(fold)
            except ValueError as e:
                print(f'Unable to convert given value for fold to int: {fold}. '
                      f'fold must be either "all" or an integer!')
                raise e

    # Set device
    if device == 'cpu':
        torch.set_num_threads(multiprocessing.cpu_count())
        device_obj = torch.device('cpu')
    elif device == 'cuda':
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device_obj = torch.device('cuda')
    else:
        device_obj = torch.device(device)

    # Create Lightning module
    print(f"Creating nnUNet Lightning module for {dataset_name_or_id}, "
          f"configuration {configuration}, fold {fold}")
    lightning_module = get_lightning_module_from_args(
        dataset_name_or_id=dataset_name_or_id,
        configuration=configuration,
        fold=fold,
        trainer_name=trainer_class_name,
        plans_identifier=plans_identifier,
        device=device_obj
    )

    # Load pretrained weights if provided
    if pretrained_weights is not None and not continue_training:
        print(f"Loading pretrained weights from {pretrained_weights}")
        if not lightning_module.was_initialized:
            lightning_module.initialize()
        from nnunetv2.run.load_pretrained_weights import load_pretrained_weights
        load_pretrained_weights(lightning_module.network, pretrained_weights, verbose=True)

    # Setup callbacks
    output_folder = lightning_module.output_folder
    callbacks = setup_callbacks(
        output_folder=output_folder,
        save_every_n_epochs=lightning_module.save_every,
        monitor_metric='val_loss',
    )

    # Setup logger
    experiment_name = f"{maybe_convert_to_dataset_name(dataset_name_or_id)}_{configuration}_fold{fold}"
    logger = setup_logger(
        logger_type=logger_type,
        output_folder=output_folder,
        experiment_name=experiment_name,
        wandb_project=wandb_project,
    )

    # Setup DDP strategy for multi-GPU training
    if num_gpus > 1:
        strategy = DDPStrategy(
            find_unused_parameters=False,
            static_graph=False,  # Set to True if your model structure doesn't change
        )
    else:
        strategy = 'auto'

    # Determine checkpoint path for resuming
    ckpt_path = None
    if continue_training:
        checkpoint_path = join(output_folder, 'checkpoint_latest.ckpt')
        if not isfile(checkpoint_path):
            checkpoint_path = join(output_folder, 'checkpoint_final.ckpt')
        if not isfile(checkpoint_path):
            checkpoint_path = join(output_folder, 'checkpoint_best.ckpt')
        if isfile(checkpoint_path):
            print(f"Resuming training from {checkpoint_path}")
            ckpt_path = checkpoint_path
        else:
            print("WARNING: Could not find checkpoint to resume from. Starting fresh training.")

    # Create Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator=device,
        devices=num_gpus if device == 'cuda' else 'auto',
        strategy=strategy,
        precision=precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=1,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,  # For performance (set to True for reproducibility)
        benchmark=True,  # Enable cudnn benchmark for performance
        gradient_clip_val=12.0,  # nnUNet uses gradient clipping with norm 12
        gradient_clip_algorithm='norm',
        # Validation settings
        val_check_interval=1.0,  # Validate once per epoch
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,  # Skip sanity validation
        # Other settings
        inference_mode=True,  # Use inference mode for validation (faster)
    )

    # Train the model
    print("\n" + "="*80)
    print("Starting Lightning training")
    print("="*80 + "\n")

    trainer.fit(
        model=lightning_module,
        ckpt_path=ckpt_path,
    )

    print("\n" + "="*80)
    print("Training completed!")
    print("="*80 + "\n")


def run_training_entry():
    """
    Entry point for the Lightning-based training script.
    Handles argument parsing and calls run_training_lightning.
    """
    parser = argparse.ArgumentParser(
        description='nnU-Net Training with PyTorch Lightning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('dataset_name_or_id', type=str,
                        help="Dataset name or ID to train with")
    parser.add_argument('configuration', type=str,
                        help="Configuration that should be trained (e.g., '3d_fullres', '2d')")
    parser.add_argument('fold', type=str,
                        help='Fold of the 5-fold cross-validation. Should be an int between 0 and 4, or "all".')

    # Optional trainer arguments
    parser.add_argument('-tr', '--trainer', type=str, default='nnUNetLightningModule',
                        help='Use this flag to specify a custom Lightning trainer module')
    parser.add_argument('-p', '--plans', type=str, default='nnUNetPlans',
                        help='Plans identifier to use')
    parser.add_argument('-pretrained_weights', type=str, default=None,
                        help='Path to nnU-Net checkpoint file to be used as pretrained model')

    # Training configuration
    parser.add_argument('-num_gpus', '--num_gpus', type=int, default=1,
                        help='Number of GPUs to use for training')
    parser.add_argument('-epochs', '--num_epochs', type=int, default=1000,
                        help='Number of epochs to train')
    parser.add_argument('--c', '--continue', action='store_true', dest='continue_training',
                        help='Continue training from latest checkpoint')

    # Logging
    parser.add_argument('-logger', '--logger', type=str, default='tensorboard',
                        choices=['tensorboard', 'wandb', 'none'],
                        help='Logger to use for experiment tracking')
    parser.add_argument('-wandb_project', '--wandb_project', type=str, default=None,
                        help='WandB project name (only used if logger is wandb)')

    # Performance settings
    parser.add_argument('-precision', '--precision', type=str, default='16-mixed',
                        choices=['32', '16-mixed', 'bf16-mixed', '32-true', '16-true', 'bf16-true'],
                        help='Training precision (mixed precision recommended for speed)')
    parser.add_argument('-device', '--device', type=str, default='cuda',
                        choices=['cuda', 'cpu', 'mps'],
                        help="Device to use for training")

    args = parser.parse_args()

    # Run training
    run_training_lightning(
        dataset_name_or_id=args.dataset_name_or_id,
        configuration=args.configuration,
        fold=args.fold,
        trainer_class_name=args.trainer,
        plans_identifier=args.plans,
        pretrained_weights=args.pretrained_weights,
        num_gpus=args.num_gpus,
        num_epochs=args.num_epochs,
        continue_training=args.continue_training,
        logger_type=args.logger,
        wandb_project=args.wandb_project,
        precision=args.precision,
        device=args.device,
    )


if __name__ == '__main__':
    # Set environment variables for optimal performance
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    # Reduces the number of threads used for compiling
    os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'

    run_training_entry()
