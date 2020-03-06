from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He

from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
from torch.nn.functional import gelu


class GeLU(nn.Module):
    def forward(self, x):
        return gelu(x)


class nnUNetTrainerV2_GeLU(nnUNetTrainerV2):
    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - ReLU
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
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
        net_nonlin = GeLU
        net_nonlin_kwargs = {}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    2, 2, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper


nnUNetTrainerV2_GeLU_copy1 = nnUNetTrainerV2_GeLU
nnUNetTrainerV2_GeLU_copy2 = nnUNetTrainerV2_GeLU
nnUNetTrainerV2_GeLU_copy3 = nnUNetTrainerV2_GeLU
nnUNetTrainerV2_GeLU_copy4 = nnUNetTrainerV2_GeLU
