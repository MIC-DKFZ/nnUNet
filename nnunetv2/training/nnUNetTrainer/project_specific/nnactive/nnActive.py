from typing import Tuple, Union

import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.variants.sparse_labels.nnUNetTrainer_betterIgnoreSampling import (
    nnUNetTrainer_betterIgnoreSampling_noSmooth,
)


class nnActiveTrainer_2epochs(nnUNetTrainer_betterIgnoreSampling_noSmooth):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        """Used for training with shorter epoch count."""
        super().__init__(
            plans, configuration, fold, dataset_json, unpack_dataset, device
        )
        self.num_epochs = 2


class nnActiveTrainer_5epochs(nnUNetTrainer_betterIgnoreSampling_noSmooth):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        """Used for training with shorter epoch count."""
        super().__init__(
            plans, configuration, fold, dataset_json, unpack_dataset, device
        )
        self.num_epochs = 5


class nnActiveTrainer_50epochs(nnUNetTrainer_betterIgnoreSampling_noSmooth):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        """Used for training with shorter epoch count."""
        super().__init__(
            plans, configuration, fold, dataset_json, unpack_dataset, device
        )
        self.num_epochs = 50


class nnActiveTrainer_100epochs(nnUNetTrainer_betterIgnoreSampling_noSmooth):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        """Used for training with shorter epoch count."""
        super().__init__(
            plans, configuration, fold, dataset_json, unpack_dataset, device
        )
        self.num_epochs = 100


class nnActiveTrainer_200epochs(nnUNetTrainer_betterIgnoreSampling_noSmooth):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        """Used for training with shorter epoch count."""
        super().__init__(
            plans, configuration, fold, dataset_json, unpack_dataset, device
        )
        self.num_epochs = 200


class nnActiveTrainer_250epochs(nnUNetTrainer_betterIgnoreSampling_noSmooth):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        """Used for training with shorter epoch count."""
        super().__init__(
            plans, configuration, fold, dataset_json, unpack_dataset, device
        )
        self.num_epochs = 250


class nnActiveTrainer_500epochs(nnUNetTrainer_betterIgnoreSampling_noSmooth):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        """Used for training with shorter epoch count."""
        super().__init__(
            plans, configuration, fold, dataset_json, unpack_dataset, device
        )
        self.num_epochs = 500


class nnActiveTrainer_1000epochs(nnUNetTrainer_betterIgnoreSampling_noSmooth):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        """Used for training with shorter epoch count."""
        super().__init__(
            plans, configuration, fold, dataset_json, unpack_dataset, device
        )
        self.num_epochs = 1000
