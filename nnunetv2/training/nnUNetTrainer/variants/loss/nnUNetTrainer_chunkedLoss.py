"""
Trainers whose CE and Dice are chunked over the batch dimension, for datasets with many classes.

PyTorch's nll_loss2d CUDA kernel (cross entropy backward) indexes with int32, so the backward hits an
illegal memory access once batch * num_classes * voxels >= 2**31 - observed on 3d_fullres, 192^3 patches,
118 classes, at >= 3 samples per GPU. Below that limit nothing is chunked and the exact stock computation
runs, so these trainers are safe to use at any size.
"""
import torch

from nnunetv2.training.loss.chunked_losses import ChunkedMemoryEfficientSoftDiceLoss, DC_and_CE_loss_chunked
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class ChunkedLossMixin:
    """
    Swaps the loss classes nnUNetTrainer._build_loss instantiates for their batch-chunked counterparts.
    Implemented as class attributes rather than a _build_loss override so that it composes with any other
    trainer variant without per-combination code.
    """
    dice_class = ChunkedMemoryEfficientSoftDiceLoss
    dc_and_ce_class = DC_and_CE_loss_chunked


class nnUNetTrainer_chunkedLoss(ChunkedLossMixin, nnUNetTrainer):
    """Stock trainer, chunked loss."""


class nnUNetTrainer_chunkedLoss_5epochs(nnUNetTrainer_chunkedLoss):
    """used for debugging plans / checking that a many-class dataset fits and runs at all"""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 5
