from typing import Tuple

from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_probabilisticOversampling(nnUNetTrainer):
    """
    sampling of foreground happens randomly and not for the last 33% of samples in a batch
    since most trainings happen with batch size 2 and nnuent guarantees at least one fg sample, effectively this is 50%
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: str = 'cuda:0'):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.oversample_foreground_percent = 0.5

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