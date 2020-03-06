from apex import amp
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
from nnunet.network_architecture.custom_modules.mish import Mish


class nnUNetTrainerV2_Mish(nnUNetTrainerV2):
    def initialize_network(self):
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = Mish
        net_nonlin_kwargs = {}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(0),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def _maybe_init_amp(self):
        """
        In O1 mish will result in super super high memory usage. I believe that may be because amp decides to be save
        and use fp32 for all activation functions. By using O2 we reduce memory comsumption by a lot
        :return:
        """
        # we use fp16 for training only, not inference
        if self.fp16:
            if not self.amp_initialized:
                if amp is not None:
                    self.network, self.optimizer = amp.initialize(self.network, self.optimizer, opt_level="O2")
                    self.amp_initialized = True
                else:
                    self.print_to_log_file("WARNING: FP16 training was requested but nvidia apex is not installed. "
                                           "Install it from https://github.com/NVIDIA/apex")
