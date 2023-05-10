import os
os.add_dll_directory(r"C:\Program Files\openslide\bin") # windows
from wholeslidedata.iterators import create_batch_iterator
from wholeslidedata.visualization.plotting import plot_batch
# from dicfg.magics import ConfigMagics
from wholeslidedata.samplers.callbacks import BatchCallback

from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor


from multiprocessing import freeze_support

# This will allow loading yaml in an notebook cell
# ConfigMagics.register_magics()

import matplotlib.pyplot as plt

import numpy as np

if __name__ == '__main__':
    freeze_support()
    with create_batch_iterator(user_config='test.yml', 
                               context='spawn' if os.name=='nt' else 'fork',
                                mode='training') as training_batch_generator:
        for i in range(4):
            print(i)
            x_batch, y_batch, info =  next(training_batch_generator)
            plot_batch(x_batch, y_batch)

