import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_5epochs(nnUNetTrainer):
    """used for debugging plans etc"""

    def __post_init__(self):
        super().__post_init__()
        self.num_epochs = 5


class nnUNetTrainer_1epoch(nnUNetTrainer):
    """used for debugging plans etc"""

    def __post_init__(self):
        super().__post_init__()
        self.num_epochs = 1

class nnUNetTrainer_2epochs(nnUNetTrainer):
    """used for debugging plans etc"""

    def __post_init__(self):
        super().__post_init__()
        self.num_epochs = 2


class nnUNetTrainer_10epochs(nnUNetTrainer):
    """used for debugging plans etc"""

    def __post_init__(self):
        super().__post_init__()
        self.num_epochs = 10


class nnUNetTrainer_20epochs(nnUNetTrainer):
    """used for debugging plans etc"""

    def __post_init__(self):
        super().__post_init__()
        self.num_epochs = 20

class nnUNetTrainer_50epochs(nnUNetTrainer):
    """used for debugging plans etc"""

    def __post_init__(self):
        super().__post_init__()
        self.num_epochs = 50

class nnUNetTrainer_100epochs(nnUNetTrainer):
    """used for debugging plans etc"""

    def __post_init__(self):
        super().__post_init__()
        self.num_epochs = 100

class nnUNetTrainer_200epochs(nnUNetTrainer):
    """used for debugging plans etc"""

    def __post_init__(self):
        super().__post_init__()
        self.num_epochs = 200


class nnUNetTrainer_betterIgnoreSampling_noSmooth_50epochs(nnUNetTrainer_betterIgnoreSampling_noSmooth):
    """used for debugging plans etc"""

    def __post_init__(self):
        super().__post_init__()
        self.num_epochs = 50


class nnUNetTrainer_250epochs(nnUNetTrainer):
    """used for debugging plans etc"""

    def __post_init__(self):
        super().__post_init__()
        self.num_epochs = 250


class nnUNetTrainer_500epochs(nnUNetTrainer):
    """used for debugging plans etc"""

    def __post_init__(self):
        super().__post_init__()
        self.num_epochs = 500


class nnUNetTrainer_750epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 750


class nnUNetTrainer_2000epochs(nnUNetTrainer):
    """used for debugging plans etc"""

    def __post_init__(self):
        super().__post_init__()
        self.num_epochs = 200

    
class nnUNetTrainer_4000epochs(nnUNetTrainer):
    """used for debugging plans etc"""

    def __post_init__(self):
        super().__post_init__()
        self.num_epochs = 4000


class nnUNetTrainer_8000epochs(nnUNetTrainer):
    """used for debugging plans etc"""

    def __post_init__(self):
        super().__post_init__()
        self.num_epochs = 8000