from typing import Tuple

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5


class anon_nnUNetTrainer1(nnUNetTrainerDA5):
    """
    combines
    - nnUNetTrainerDA5
    - nnUNetTrainerCosAnneal
    - nnUNetTrainer_probabilisticOversampling_033
    Recommendation is to use this with nnUNetResEncUNetPlans
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: str = 'cuda'):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.oversample_foreground_percent = 0.33
        self.print_to_log_file(f"self.oversample_foreground_percent {self.oversample_foreground_percent}")

    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        if dim == 2:
            dl_tr = nnUNetDataLoader2D(dataset_tr, self.plans['configurations'][self.configuration]['batch_size'],
                                       initial_patch_size,
                                       self.plans['configurations'][self.configuration]['patch_size'],
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None, probabilistic_oversampling=True)
            dl_val = nnUNetDataLoader2D(dataset_val, self.plans['configurations'][self.configuration]['batch_size'],
                                        self.plans['configurations'][self.configuration]['patch_size'],
                                        self.plans['configurations'][self.configuration]['patch_size'],
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None, probabilistic_oversampling=True)
        else:
            dl_tr = nnUNetDataLoader3D(dataset_tr, self.plans['configurations'][self.configuration]['batch_size'],
                                       initial_patch_size,
                                       self.plans['configurations'][self.configuration]['patch_size'],
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None, probabilistic_oversampling=True)
            dl_val = nnUNetDataLoader3D(dataset_val, self.plans['configurations'][self.configuration]['batch_size'],
                                        self.plans['configurations'][self.configuration]['patch_size'],
                                        self.plans['configurations'][self.configuration]['patch_size'],
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None, probabilistic_oversampling=True)
        return dl_tr, dl_val

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        return optimizer, lr_scheduler


class anon_nnUNetTrainer2(nnUNetTrainerDA5):
    """
    combines
    - nnUNetTrainerDA5
    - nnUNetTrainer_probabilisticOversampling_033
    Recommendation is to use this with nnUNetResEncUNetPlans
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: str = 'cuda'):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.oversample_foreground_percent = 0.33
        self.print_to_log_file(f"self.oversample_foreground_percent {self.oversample_foreground_percent}")

    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        if dim == 2:
            dl_tr = nnUNetDataLoader2D(dataset_tr, self.plans['configurations'][self.configuration]['batch_size'],
                                       initial_patch_size,
                                       self.plans['configurations'][self.configuration]['patch_size'],
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None, probabilistic_oversampling=True)
            dl_val = nnUNetDataLoader2D(dataset_val, self.plans['configurations'][self.configuration]['batch_size'],
                                        self.plans['configurations'][self.configuration]['patch_size'],
                                        self.plans['configurations'][self.configuration]['patch_size'],
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None, probabilistic_oversampling=True)
        else:
            dl_tr = nnUNetDataLoader3D(dataset_tr, self.plans['configurations'][self.configuration]['batch_size'],
                                       initial_patch_size,
                                       self.plans['configurations'][self.configuration]['patch_size'],
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None, probabilistic_oversampling=True)
            dl_val = nnUNetDataLoader3D(dataset_val, self.plans['configurations'][self.configuration]['batch_size'],
                                        self.plans['configurations'][self.configuration]['patch_size'],
                                        self.plans['configurations'][self.configuration]['patch_size'],
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None, probabilistic_oversampling=True)
        return dl_tr, dl_val

