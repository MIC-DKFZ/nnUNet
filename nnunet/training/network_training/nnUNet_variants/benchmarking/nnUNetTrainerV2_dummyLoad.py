#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from typing import Tuple

import torch

from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss
from nnunet.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_noDeepSupervision import \
    nnUNetTrainerV2_noDeepSupervision
from nnunet.training.network_training.nnUNet_variants.benchmarking.nnUNetTrainerV2_2epochs import nnUNetTrainerV2_5epochs
from torch.cuda.amp import autocast
import numpy as np


class nnUNetTrainerV2_5epochs_dummyLoad(nnUNetTrainerV2_5epochs):
    def initialize(self, training=True, force_load_plans=False):
        super().initialize(training, force_load_plans)
        self.some_batch = torch.rand((self.batch_size, self.num_input_channels, *self.patch_size)).float().cuda()

        self.some_gt = [torch.round(torch.rand((self.batch_size, 1, *[int(i * j) for i, j in zip(self.patch_size, k)])) * (self.num_classes - 1)).float().cuda() for k in self.deep_supervision_scales]

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        data = self.some_batch
        target = self.some_gt

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()


class nnUNetTrainerV2_2epochs_dummyLoad(nnUNetTrainerV2_5epochs_dummyLoad):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, max_num_epochs=1000):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16, max_num_epochs)
        self.max_num_epochs = 2


class nnUNetTrainerV2_5epochs_dummyLoadCEnoDS(nnUNetTrainerV2_noDeepSupervision):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, max_num_epochs=1000):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16, max_num_epochs)
        self.max_num_epochs = 5
        self.loss = RobustCrossEntropyLoss()

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True, step_size: float = 0.5,
                 save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs=None, run_postprocessing_on_folds: bool = True):
        pass

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def save_checkpoint(self, fname, save_optimizer=True):
        pass

    def initialize(self, training=True, force_load_plans=False):
        super().initialize(training, force_load_plans)
        self.some_batch = torch.rand((self.batch_size, self.num_input_channels, *self.patch_size)).float().cuda()

        self.some_gt = torch.round(torch.rand((self.batch_size, *self.patch_size)) * (self.num_classes - 1)).long().cuda()

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        data = self.some_batch
        target = self.some_gt

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()

    def run_online_evaluation(self, output, target):
        pass

    def finish_online_evaluation(self):
        pass
