import numpy as np
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

        self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def setup_DA_params(self):
        """
        net_num_pool_op_kernel_sizes is different in resunet
        """
        super().setup_DA_params()
        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes[1:]), axis=0))[:-1]

    def validate(self, do_mirroring: bool = True, use_train_mode: bool = False, tiled: bool = True, step: int = 2,
                 save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 force_separate_z: bool = None, interpolation_order: int = 3, interpolation_order_z=0):
        ds = self.network.decoder.deep_supervision
        self.network.deep_supervision = False
        ret = nnUNetTrainer.validate(self, do_mirroring, use_train_mode, tiled, step, save_softmax, use_gaussian,
                               overwrite, validation_folder_name, debug, all_in_gpu,
                               force_separate_z=force_separate_z, interpolation_order=interpolation_order,
                               interpolation_order_z=interpolation_order_z)
        self.network.decoder.deep_supervision = ds
        return ret

    def predict_preprocessed_data_return_softmax(self, data, do_mirroring, num_repeats, use_train_mode, batch_size,
                                                 mirror_axes, tiled, tile_in_z, step, min_size, use_gaussian,
                                                 all_in_gpu=False):
        ds = self.network.decoder.deep_supervision
        self.network.decoder.deep_supervision = False
        ret = nnUNetTrainer.predict_preprocessed_data_return_softmax(self, data, do_mirroring, num_repeats, use_train_mode,
                                                               batch_size,
                                                               mirror_axes, tiled, tile_in_z, step, min_size,
                                                               use_gaussian, all_in_gpu)
        self.network.decoder.deep_supervision = ds
        return ret

    def predict_preprocessed_data_return_softmax_and_seg(self, data, do_mirroring, num_repeats, use_train_mode,
                                                         batch_size,
                                                         mirror_axes, tiled, tile_in_z, step, min_size, use_gaussian,
                                                         all_in_gpu=False):
        ds = self.network.decoder.deep_supervision
        self.network.deep_supervision = False
        ret = nnUNetTrainer.predict_preprocessed_data_return_softmax_and_seg(self, data, do_mirroring, num_repeats, use_train_mode,
                                                                       batch_size,
                                                                       mirror_axes, tiled, tile_in_z, step, min_size,
                                                                       use_gaussian, all_in_gpu)
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
