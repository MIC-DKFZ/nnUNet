from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.loss_functions.TopK_loss import TopKLoss


class nnUNetTrainerV2_Loss_TopK10(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.loss = TopKLoss(k=10)


nnUNetTrainerV2_Loss_TopK10_copy1 = nnUNetTrainerV2_Loss_TopK10
nnUNetTrainerV2_Loss_TopK10_copy2 = nnUNetTrainerV2_Loss_TopK10
nnUNetTrainerV2_Loss_TopK10_copy3 = nnUNetTrainerV2_Loss_TopK10
nnUNetTrainerV2_Loss_TopK10_copy4 = nnUNetTrainerV2_Loss_TopK10

