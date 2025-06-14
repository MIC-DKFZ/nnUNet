import torch
from torch import nn
import numpy as np
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerFrozenEncoderClsRobust import nnUNetTrainerFrozenEncoderClsRobust

class nnUNetTrainerFrozenEncoderClsWeightedCE(nnUNetTrainerFrozenEncoderClsRobust):
    """
    Robust trainer specifically using Weighted CrossEntropy Loss
    Based on class distribution: 0:71, 1:121, 2:96
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # Override to force weighted CE
        self.loss_type = 'weighted_ce'
        self.optimizer_type = 'adamw'
        self.scheduler_type = 'cosine'

        # Re-initialize loss with the forced type
        self._init_classification_loss()

        print(f"✅ nnUNetTrainerFrozenEncoderClsWeightedCE initialized")
        print(f"✅ Using Weighted CrossEntropy Loss")
        print(f"✅ Class weights based on distribution: 0:71, 1:121, 2:96")
