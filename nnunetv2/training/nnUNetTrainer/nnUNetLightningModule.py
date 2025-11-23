"""
PyTorch Lightning wrapper for nnUNetTrainer.

This module provides a Lightning-compatible interface to nnUNet's training logic
while preserving all of nnUNet's out-of-the-box performance and features.
"""

import torch
from typing import Union, List, Optional
import pytorch_lightning as pl
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.collate_outputs import collate_outputs
from torch import distributed as dist
import numpy as np


class nnUNetLightningModule(pl.LightningModule, nnUNetTrainer):
    """
    PyTorch Lightning wrapper for nnUNetTrainer.

    This class inherits from both LightningModule and nnUNetTrainer to provide
    Lightning's features (multi-GPU, mixed precision, logging, checkpointing)
    while maintaining nnUNet's training logic and performance.

    Args:
        plans: nnUNet plans dictionary
        configuration: Configuration name (e.g., '3d_fullres', '2d')
        fold: Cross-validation fold number
        dataset_json: Dataset metadata dictionary
        device: PyTorch device (will be overridden by Lightning)
    """

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device('cuda')):
        # Initialize both parent classes
        # Note: we call LightningModule.__init__ first, then nnUNetTrainer.__init__
        pl.LightningModule.__init__(self)
        nnUNetTrainer.__init__(self, plans, configuration, fold, dataset_json, device)

        # Store whether we're using automatic optimization (we are, but keep track)
        self.automatic_optimization = True

        # We'll let Lightning handle DDP, so we need to override is_ddp detection
        # in some methods. Lightning sets up DDP automatically.

        # Store validation outputs for epoch-level metrics
        self.validation_step_outputs = []
        self.training_step_outputs = []

    def setup(self, stage: Optional[str] = None):
        """
        Lightning hook called at the beginning of fit.
        This is where we initialize the trainer if not already done.
        """
        if stage == 'fit':
            if not self.was_initialized:
                self.initialize()

    def prepare_data(self):
        """
        Lightning hook for data preparation (downloading, etc.).
        Only called on rank 0 in distributed training.

        For nnUNet, we handle dataset unpacking here.
        """
        # This is called only on rank 0
        if hasattr(self, 'dataset_class') and self.dataset_class is not None:
            from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
            self.dataset_class.unpack_dataset(
                self.preprocessed_dataset_folder,
                overwrite_existing=False,
                num_processes=max(1, round(get_allowed_n_proc_DA() // 2)),
                verify=True)

    def train_dataloader(self):
        """
        Lightning hook to get the training dataloader.
        """
        if self.dataloader_train is None:
            self.dataloader_train, self.dataloader_val = self.get_dataloaders()
        return self.dataloader_train

    def val_dataloader(self):
        """
        Lightning hook to get the validation dataloader.
        """
        if self.dataloader_val is None:
            self.dataloader_train, self.dataloader_val = self.get_dataloaders()
        return self.dataloader_val

    def configure_optimizers(self):
        """
        Lightning hook to configure optimizers and learning rate schedulers.
        Uses nnUNet's default optimizer configuration.
        """
        optimizer, lr_scheduler = nnUNetTrainer.configure_optimizers(self)

        # Lightning expects a specific format for lr_schedulers
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'epoch',  # step every epoch
                'frequency': 1,
                'name': 'PolyLR',
            }
        }

    def training_step(self, batch, batch_idx):
        """
        Lightning training step. Wraps nnUNet's train_step.

        Note: Lightning handles the optimizer step automatically when using
        automatic_optimization=True (default), so we need to adapt nnUNet's
        train_step which manually handles optimization.
        """
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Forward pass (autocast is handled by Lightning's precision plugin)
        output = self.network(data)
        loss = self.loss(output, target)

        # Log the loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        output_dict = {'loss': loss.detach().cpu().numpy()}
        self.training_step_outputs.append(output_dict)
        return {'loss': loss}

    def on_train_epoch_start(self):
        """
        Lightning hook called at the start of each training epoch.
        Wraps nnUNet's on_train_epoch_start.
        """
        # Don't call lr_scheduler.step here - Lightning handles it
        self.network.train()
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.trainer.optimizers[0].param_groups[0]['lr'], decimals=5)}")
        self.logger.log('lrs', self.trainer.optimizers[0].param_groups[0]['lr'], self.current_epoch)

    def on_train_epoch_end(self):
        """
        Lightning hook called at the end of each training epoch.
        Logs training metrics to nnUNet logger.
        """
        if len(self.training_step_outputs) == 0:
            return

        outputs = collate_outputs(self.training_step_outputs)

        if self.trainer.world_size > 1:
            import torch.distributed as dist
            losses_tr = [None for _ in range(self.trainer.world_size)]
            dist.all_gather_object(losses_tr, outputs['loss'])
            loss_here = np.vstack(losses_tr).mean()
        else:
            loss_here = np.mean(outputs['loss'])

        self.logger.log('train_losses', loss_here, self.current_epoch)

        # Clear training outputs for next epoch
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        """
        Lightning validation step. Wraps nnUNet's validation_step.
        """
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Forward pass (autocast handled by Lightning)
        output = self.network(data)
        loss = self.loss(output, target)

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        # Log validation loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        output_dict = {'val_loss': loss.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
        self.validation_step_outputs.append(output_dict)
        return output_dict

    def on_validation_epoch_start(self):
        """
        Lightning hook called at the start of validation.
        """
        self.network.eval()

    def on_validation_epoch_end(self):
        """
        Lightning hook called at the end of validation epoch.
        Computes dice metrics from collected validation outputs.
        """
        if len(self.validation_step_outputs) == 0:
            return

        # Collate outputs from all validation steps
        outputs_collated = collate_outputs(self.validation_step_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        # In DDP mode, we need to gather from all processes
        # Lightning handles this automatically through all_gather, but we need to do it manually
        # for our custom metrics
        if self.trainer.world_size > 1:
            # Gather tp, fp, fn from all processes
            import torch.distributed as dist
            world_size = self.trainer.world_size

            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['val_loss'])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated['val_loss'])

        # Compute dice scores per class
        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)

        # Log to nnUNet logger
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)

        # Also log to Lightning logger
        self.log('mean_fg_dice', mean_fg_dice, on_epoch=True, prog_bar=True, sync_dist=False)
        for i, dice in enumerate(global_dc_per_class):
            self.log(f'dice_class_{i}', dice, on_epoch=True, sync_dist=False)

        # Clear validation outputs for next epoch
        self.validation_step_outputs.clear()

    def on_train_start(self):
        """
        Lightning hook called at the very beginning of training.
        """
        # Most of this is already handled by setup() and other Lightning hooks
        if not self.was_initialized:
            self.initialize()

        # Set deep supervision
        self.set_deep_supervision_enabled(self.enable_deep_supervision)

        # Print plans
        self.print_plans()

        # Save configuration files
        from batchgenerators.utilities.file_and_folder_operations import save_json, join, maybe_mkdir_p
        import shutil

        if self.local_rank == 0:
            maybe_mkdir_p(self.output_folder)
            save_json(self.plans_manager.plans, join(self.output_folder_base, 'plans.json'), sort_keys=False)
            save_json(self.dataset_json, join(self.output_folder_base, 'dataset.json'), sort_keys=False)

            # Copy fingerprint
            shutil.copy(join(self.preprocessed_dataset_folder_base, 'dataset_fingerprint.json'),
                       join(self.output_folder_base, 'dataset_fingerprint.json'))

        # Plot network architecture
        self.plot_network_architecture()

        # Save debug information
        self._save_debug_information()

    def on_train_end(self):
        """
        Lightning hook called at the end of training.
        """
        # Clean up dataloaders
        import sys
        import os
        from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
        from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter

        old_stdout = sys.stdout
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            if self.dataloader_train is not None and \
                    isinstance(self.dataloader_train, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
                self.dataloader_train._finish()
            if self.dataloader_val is not None and \
                    isinstance(self.dataloader_val, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
                self.dataloader_val._finish()
            sys.stdout = old_stdout

        from nnunetv2.utilities.helpers import empty_cache
        empty_cache(self.device)
        self.print_to_log_file("Training done.")

    def on_epoch_start(self):
        """
        Lightning hook called at the start of each epoch.
        """
        from time import time
        self.logger.log('epoch_start_timestamps', time(), self.current_epoch)

    def on_epoch_end(self):
        """
        Lightning hook called at the end of each epoch.
        Handles checkpointing and logging.
        """
        from time import time
        from batchgenerators.utilities.file_and_folder_operations import join, isfile
        import os

        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        # Print epoch statistics
        if 'train_losses' in self.logger.my_fantastic_logging and len(self.logger.my_fantastic_logging['train_losses']) > 0:
            self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        if 'val_losses' in self.logger.my_fantastic_logging and len(self.logger.my_fantastic_logging['val_losses']) > 0:
            self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        if 'dice_per_class_or_region' in self.logger.my_fantastic_logging and len(self.logger.my_fantastic_logging['dice_per_class_or_region']) > 0:
            self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                                   self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        if 'epoch_end_timestamps' in self.logger.my_fantastic_logging and 'epoch_start_timestamps' in self.logger.my_fantastic_logging:
            if len(self.logger.my_fantastic_logging['epoch_end_timestamps']) > 0 and len(self.logger.my_fantastic_logging['epoch_start_timestamps']) > 0:
                self.print_to_log_file(
                    f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # Plot progress (Lightning handles most logging via callbacks)
        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        # Increment epoch counter
        self.current_epoch += 1

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        """
        Lightning hook for gradient clipping.
        nnUNet uses gradient clipping with max_norm=12.
        """
        if gradient_clip_val is None:
            gradient_clip_val = 12
        if gradient_clip_algorithm is None:
            gradient_clip_algorithm = 'norm'

        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm
        )

    @property
    def local_rank(self):
        """
        Override local_rank to use Lightning's rank tracking.
        """
        if hasattr(self, 'trainer') and self.trainer is not None:
            return self.trainer.global_rank if self.trainer.global_rank is not None else 0
        return 0

    @property
    def is_ddp(self):
        """
        Override is_ddp to detect when Lightning is using DDP.
        """
        if hasattr(self, 'trainer') and self.trainer is not None:
            return self.trainer.world_size > 1
        return dist.is_available() and dist.is_initialized()
