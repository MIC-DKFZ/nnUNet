import torch
from torch import autocast

from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from torch.nn.parallel import DistributedDataParallel as DDP


class nnUNetTrainerNoDeepSupervision(nnUNetTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_deep_supervision = False
