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

import numpy as np
import torch
from nnunet.network_architecture.generic_modular_residual_UNet import FabiansUNet, get_default_network_config
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.nd_softmax import softmax_helper


class nnUNetTrainerV2_ResencUNet(nnUNetTrainerV2):
    def initialize_network(self):
        if self.threeD:
            cfg = get_default_network_config(3, None, norm_type="in")

        else:
            cfg = get_default_network_config(1, None, norm_type="in")

        stage_plans = self.plans['plans_per_stage'][self.stage]
        conv_kernel_sizes = stage_plans['conv_kernel_sizes']
        blocks_per_stage_encoder = stage_plans['num_blocks_encoder']
        blocks_per_stage_decoder = stage_plans['num_blocks_decoder']
        pool_op_kernel_sizes = stage_plans['pool_op_kernel_sizes']

        self.network = FabiansUNet(self.num_input_channels, self.base_num_features, blocks_per_stage_encoder, 2,
                                   pool_op_kernel_sizes, conv_kernel_sizes, cfg, self.num_classes,
                                   blocks_per_stage_decoder, True, False, 320, InitWeights_He(1e-2))

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def setup_DA_params(self):
        """
        net_num_pool_op_kernel_sizes is different in resunet
        """
        super().setup_DA_params()
        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes[1:]), axis=0))[:-1]

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True, step_size: float = 0.5,
                 save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 force_separate_z: bool = None, interpolation_order: int = 3, interpolation_order_z=0,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        ds = self.network.decoder.deep_supervision
        self.network.decoder.deep_supervision = False
        ret = nnUNetTrainer.validate(self, do_mirroring=do_mirroring, use_sliding_window=use_sliding_window,
                                     step_size=step_size, save_softmax=save_softmax, use_gaussian=use_gaussian,
                                     overwrite=overwrite, validation_folder_name=validation_folder_name,
                                     debug=debug, all_in_gpu=all_in_gpu,
                                     segmentation_export_kwargs=segmentation_export_kwargs,
                                     run_postprocessing_on_folds=run_postprocessing_on_folds)
        self.network.decoder.deep_supervision = ds
        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        ds = self.network.decoder.deep_supervision
        self.network.decoder.deep_supervision = False
        ret = nnUNetTrainer.predict_preprocessed_data_return_seg_and_softmax(self, data, do_mirroring=do_mirroring,
                                                                             mirror_axes=mirror_axes,
                                                                             use_sliding_window=use_sliding_window,
                                                                             step_size=step_size,
                                                                             use_gaussian=use_gaussian,
                                                                             pad_border_mode=pad_border_mode,
                                                                             pad_kwargs=pad_kwargs,
                                                                             all_in_gpu=all_in_gpu,
                                                                             verbose=verbose,
                                                                             mixed_precision=mixed_precision)
        self.network.decoder.deep_supervision = ds
        return ret

    def run_training(self):
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.decoder.deep_supervision
        self.network.decoder.deep_supervision = True
        ret = nnUNetTrainer.run_training(self)
        self.network.decoder.deep_supervision = ds
        return ret


nnUNetTrainerV2_ResencUNet_copy1 = nnUNetTrainerV2_ResencUNet
nnUNetTrainerV2_ResencUNet_copy2 = nnUNetTrainerV2_ResencUNet
nnUNetTrainerV2_ResencUNet_copy3 = nnUNetTrainerV2_ResencUNet
nnUNetTrainerV2_ResencUNet_copy4 = nnUNetTrainerV2_ResencUNet
