from typing import Tuple, Union

import numpy as np
import torch
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger

from nnunetv2.training.nnUNetTrainer.variants.sparse_labels.nnUNetTrainer_betterIgnoreSampling import (
    nnUNetTrainer_betterIgnoreSampling_noSmooth,
)


class nnActiveTrainer_2epochs(nnUNetTrainer_betterIgnoreSampling_noSmooth):
    def __post_init__(self):
        super().__post_init__()
        self.num_epochs = 2



class nnActiveTrainer_5epochs(nnUNetTrainer_betterIgnoreSampling_noSmooth):
    def __post_init__(self):
        super().__post_init__()
        self.num_epochs = 5



class nnActiveTrainer_50epochs(nnUNetTrainer_betterIgnoreSampling_noSmooth):
    def __post_init__(self):
        super().__post_init__()
        self.num_epochs = 50



class nnActiveTrainer_100epochs(nnUNetTrainer_betterIgnoreSampling_noSmooth):
    def __post_init__(self):
        super().__post_init__()
        self.num_epochs = 100



class nnActiveTrainer_200epochs(nnUNetTrainer_betterIgnoreSampling_noSmooth):
    def __post_init__(self):
        super().__post_init__()
        self.num_epochs = 200



class nnActiveTrainer_250epochs(nnUNetTrainer_betterIgnoreSampling_noSmooth):
    def __post_init__(self):
        super().__post_init__()
        self.num_epochs = 250



class nnActiveTrainer_500epochs(nnUNetTrainer_betterIgnoreSampling_noSmooth):
    def __post_init__(self):
        super().__post_init__()
        self.num_epochs = 500



class nnActiveTrainer_1000epochs(nnUNetTrainer_betterIgnoreSampling_noSmooth):
    def __post_init__(self):
        super().__post_init__()
        self.num_epochs = 1000
