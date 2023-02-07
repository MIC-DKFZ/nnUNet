from typing import Tuple

from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import numpy as np


class nnUNetTrainer_probabilisticOversampling(nnUNetTrainer):
    """
    sampling of foreground happens randomly and not for the last 33% of samples in a batch
    since most trainings happen with batch size 2 and nnunet guarantees at least one fg sample, effectively this can
    be 50%
    Here we compute the actual oversampling percentage used by nnUNetTrainer in order to be as consistent as possible.
    If we switch to this oversampling then we can keep it at a constant 0.33 or whatever.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: str = 'cuda'):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.oversample_foreground_percent = float(np.mean(
            [not sample_idx < round(self.configuration_manager.batch_size * (1 - self.oversample_foreground_percent))
             for sample_idx in range(self.configuration_manager.batch_size)]))
        self.print_to_log_file(f"self.oversample_foreground_percent {self.oversample_foreground_percent}")

    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        if dim == 2:
            dl_tr = nnUNetDataLoader2D(dataset_tr,
                                       self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None, probabilistic_oversampling=True)
            dl_val = nnUNetDataLoader2D(dataset_val,
                                        self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None, probabilistic_oversampling=True)
        else:
            dl_tr = nnUNetDataLoader3D(dataset_tr,
                                       self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None, probabilistic_oversampling=True)
            dl_val = nnUNetDataLoader3D(dataset_val,
                                        self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None, probabilistic_oversampling=True)
        return dl_tr, dl_val


class nnUNetTrainer_probabilisticOversampling_033(nnUNetTrainer_probabilisticOversampling):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: str = 'cuda'):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.oversample_foreground_percent = 0.33


class nnUNetTrainer_probabilisticOversampling_010(nnUNetTrainer_probabilisticOversampling):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: str = 'cuda'):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.oversample_foreground_percent = 0.1


