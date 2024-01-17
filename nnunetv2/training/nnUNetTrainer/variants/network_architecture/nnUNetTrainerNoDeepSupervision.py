from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch


class nnUNetTrainerNoDeepSupervision(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = False
