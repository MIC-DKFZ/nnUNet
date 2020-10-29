import torch
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.network_training.nnUNet_variants.data_augmentation.nnUNetTrainerV2_insaneDA import \
    nnUNetTrainerV2_insaneDA
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn


class nnUNetTrainerV2_MMS(nnUNetTrainerV2_insaneDA):
    def setup_DA_params(self):
        super().setup_DA_params()
        self.data_aug_params["p_rot"] = 0.7
        self.data_aug_params["p_eldef"] = 0.1
        self.data_aug_params["p_scale"] = 0.3

        self.data_aug_params["independent_scale_factor_for_each_axis"] = True
        self.data_aug_params["p_independent_scale_per_axis"] = 0.3

        self.data_aug_params["do_additive_brightness"] = True
        self.data_aug_params["additive_brightness_mu"] = 0
        self.data_aug_params["additive_brightness_sigma"] = 0.2
        self.data_aug_params["additive_brightness_p_per_sample"] = 0.3
        self.data_aug_params["additive_brightness_p_per_channel"] = 1

        self.data_aug_params["elastic_deform_alpha"] = (0., 300.)
        self.data_aug_params["elastic_deform_sigma"] = (9., 15.)

        self.data_aug_params['gamma_range'] = (0.5, 1.6)

    def initialize_network(self):
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.BatchNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.BatchNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    """def run_training(self):
        from batchviewer import view_batch
        a = next(self.tr_gen)
        view_batch(a['data'])
        import IPython;IPython.embed()"""
