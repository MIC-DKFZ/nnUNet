from nnunetv2.training.nnUNetTrainer.project_specific.alegra.nnUNetTrainer_airwayAug_new import (
    nnUNetTrainer_airwayAug_new_noSmooth_betterIgnSampling,
)


class nnActiveTrainer_airway_2epochs(
    nnUNetTrainer_airwayAug_new_noSmooth_betterIgnSampling
):
    def __post_init__(self):
        self.num_epochs = 2


class nnActiveTrainer_airway_5epochs(
    nnUNetTrainer_airwayAug_new_noSmooth_betterIgnSampling
):
    def __post_init__(self):
        self.num_epochs = 5


class nnActiveTrainer_airway_50epochs(
    nnUNetTrainer_airwayAug_new_noSmooth_betterIgnSampling
):
    def __post_init__(self):
        self.num_epochs = 50


class nnActiveTrainer_airway_100epochs(
    nnUNetTrainer_airwayAug_new_noSmooth_betterIgnSampling
):
    def __post_init__(self):
        self.num_epochs = 100


class nnActiveTrainer_airway_200epochs(
    nnUNetTrainer_airwayAug_new_noSmooth_betterIgnSampling
):
    def __post_init__(self):
        self.num_epochs = 200


class nnActiveTrainer_airway_250epochs(
    nnUNetTrainer_airwayAug_new_noSmooth_betterIgnSampling
):
    def __post_init__(self):
        self.num_epochs = 250


class nnActiveTrainer_airway_500epochs(
    nnUNetTrainer_airwayAug_new_noSmooth_betterIgnSampling
):
    def __post_init__(self):
        self.num_epochs = 500


class nnActiveTrainer_airway_1000epochs(
    nnUNetTrainer_airwayAug_new_noSmooth_betterIgnSampling
):
    def __post_init__(self):
        self.num_epochs = 1000
