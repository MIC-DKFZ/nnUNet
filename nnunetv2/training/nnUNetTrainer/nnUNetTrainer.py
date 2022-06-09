import shutil
import sys
from copy import deepcopy
from datetime import datetime
from multiprocessing import Pool
from time import time, sleep
from typing import Union, Optional, Tuple, List, Dict, Any

import numpy as np
import pytorch_lightning as pl
import torch
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p, \
    split_path
from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder, labels_to_list_of_regions
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer, recursive_find_reader_writer_by_name
from nnunetv2.inference.export_prediction import export_prediction
from nnunetv2.inference.sliding_window_prediction import predict_sliding_window_return_logits, compute_gaussian
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.data_augmentation.custom_transforms.cascade_transforms import MoveSegAsOneHotToData, \
    ApplyRandomBinaryOperatorTransform, RemoveRandomConnectedComponentFromOneHotEncodingTransform
from nnunetv2.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import \
    DownsampleSegForDSTransform2
from nnunetv2.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import \
    LimitedLenWrapper
from nnunetv2.training.data_augmentation.custom_transforms.masking import MaskTransform
from nnunetv2.training.data_augmentation.custom_transforms.region_based_training import \
    ConvertSegmentationToRegionsTransform
from nnunetv2.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import Convert2DTo3DTransform, \
    Convert3DTo2DTransform
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.utils import unpack_dataset, get_case_identifiers
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import softmax_helper_dim0
from nnunetv2.utilities.label_handling import handle_labels
from nnunetv2.utilities.tensor_utilities import sum_tensor
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from sklearn.model_selection import KFold
import inspect


class nnUNetTrainer(object):
    def __init__(self, dataset_name_or_id: Union[int, str], plans_name: str, configuration: str, fold: int,
                 unpack_dataset: bool = True, folder_with_segs_from_previous_stage: str = None):
        # loading and saving this class for continuing from checkpoint should not happen based on pickling. This
        # would also pickle the network etc. Bad, bad. Instead we just reinstantiate and then load the checkpoint we
        # need. So let's save the init args
        self.my_init_kwargs = {}
        for k in inspect.signature(self.__init__).parameters.keys():
            self.my_init_kwargs[k] = locals()[k]

        self.dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)

        self.preprocessed_dataset_folder_base = join(nnUNet_preprocessed, self.dataset_name)
        self.plans_file = join(self.preprocessed_dataset_folder_base, plans_name + '.json')

        self.plans = self.preprocessed_dataset_folder = self.dataset_json = None
        self.labels = self.regions = self.ignore_label = None

        self.inference_nonlinearity = None

        self.configuration = configuration
        self.unpack_dataset = unpack_dataset
        self.folder_with_segs_from_previous_stage = folder_with_segs_from_previous_stage
        self.fold = fold

        self.initial_lr = 1e-2
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.33

        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.num_epochs = 1000

        self.output_folder_base = join(nnUNet_results, self.dataset_name,
                                       self.__class__.__name__ + '__' + plans_name + "__" + configuration)
        self.output_folder = join(self.output_folder_base, f'fold_{fold}')

        # initialize log file. This is just our log for the print statements etc. Not to be confused with lightning
        # logging
        timestamp = datetime.now()
        self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                             (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                              timestamp.second))

        self.loss = None

        # all the logging alternatives are shit. Either too complicated or doesn't to what I want them to do. Sadge
        # yes yes I know it's not da lightning wae but who knows da wae anywae?
        self.my_fantastic_logging = {
            'mean_fg_dice': list(),
            'dice_per_class_or_region': list(),
            'train_losses': list(),
            'val_losses': list(),
            'lrs': list(),
            'epoch_start_timestamps': list(),
            'epoch_end_timestamps': list()
        }
        # shut up, this logging is great

        self._ema_pseudo_dice = None
        self._best_ema = None

        self.inference_allowed_mirroring_axes = None  # this variable is set in
        # self.configure_rotation_dummyDA_mirroring_and_inital_patch_size and will be saved in checkpoints
        self.inference_gaussian = None  # will be set in self.on_predict_start to speed up inference. After
        # prediction it is set back to None in self.on_predict_end
        self.inference_segmentation_export_pool = None
        self.inference_parameters = {
            'tile_step_size': 0.5,
            'use_gaussian': True,
            'use_mirroring': True,
            'perform_everything_on_gpu': False,
            'verbose': True,
            'save_probabilities': False,
            'n_processes_segmentation_export': default_num_processes,
        }

    def initialize(self, train: bool = True):
        self.plans = load_json(self.plans_file)
        self.preprocessed_dataset_folder = join(self.preprocessed_dataset_folder_base,
                                                self.plans['configurations'][configuration]["data_identifier"])
        self.dataset_json = load_json(join(self.preprocessed_dataset_folder_base, 'dataset.json'))

        # labels can either be a list of int (regular training) or a list of tuples of int (region-based training)
        self.labels, self.regions, self.ignore_label = handle_labels(self.dataset_json)

        # needed for predictions
        self.inference_nonlinearity = torch.sigmoid if self.regions is not None else softmax_helper_dim0

        if train:
            maybe_mkdir_p(self.output_folder)

        # if you want to swap out the network architecture you need to change that here. You do not need to use the
        # plans.json file at all if you don't want to, just make sure your architecture is compatible with the patch
        # size dictated by the plans!
        self.network = get_network_from_plans(self.plans, self.dataset_json, configuration)

        if self.regions is None:
            self.loss = DC_and_CE_loss({'batch_dice': self.plans['configurations'][self.configuration]['batch_dice'],
                                        'smooth': 1e-5, 'do_bg': False}, {}, weight_ce=1, weight_dice=1,
                                       ignore_label=self.ignore_label)
        else:
            self.loss = DC_and_BCE_loss({},
                                        {'batch_dice': self.plans['configurations'][self.configuration]['batch_dice'],
                                         'do_bg': True, 'smooth': 1e-5}, ignore_label=self.ignore_label)
        self.wrap_loss_for_deep_supervision()