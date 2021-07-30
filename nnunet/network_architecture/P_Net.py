import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.network_architecture.initialization import InitWeights_He

class P_Net(SegmentationNetwork):
    def __init__(self, patch_size, in_channels=2, out_channels=32, num_classes=2, weightInitializer=InitWeights_He(1e-2), deep_supervision=False):  # or out_channels = 16/64
        super(P_Net, self).__init__()

        self.do_ds = False
        self.patch_size = patch_size.tolist()

        self.block1 = nn.Sequential(
          nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0, dilation=1),
          nn.ReLU(),
          nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0, dilation=1), # or kernel_size=[3, 3, 3]
          nn.ReLU(),
        )
        self.block2 = nn.Sequential(
          nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0, dilation=2),
          nn.ReLU(),
          nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0, dilation=2), # or kernel_size=[3, 3, 3]
          nn.ReLU(),
        )
        self.block3 = nn.Sequential(
          nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0, dilation=3), # or kernel_size=[3, 3, 1]
          nn.ReLU(),
          nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0, dilation=3),
          nn.ReLU(),
          nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0, dilation=3),
          nn.ReLU(),
        )
        self.block4 = nn.Sequential(
          nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0, dilation=4), # or kernel_size=[3, 3, 1]
          nn.ReLU(),
          nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0, dilation=4),
          nn.ReLU(),
          nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0, dilation=4),
          nn.ReLU(),
        )
        self.block5 = nn.Sequential(
          nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0, dilation=5), # or kernel_size=[3, 3, 1]
          nn.ReLU(),
          nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0, dilation=5),
          nn.ReLU(),
          nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0, dilation=5),
          nn.ReLU(),
        )
        self.block6 = nn.Sequential(
          nn.Conv3d(in_channels=int(out_channels/4)*5, out_channels=out_channels, kernel_size=3, stride=1, padding=0, dilation=1), # or kernel_size=[3, 3, 1]
          nn.ReLU(),
          nn.Conv3d(in_channels=out_channels, out_channels=num_classes, kernel_size=3, stride=1, padding=0, dilation=1),
          # nn.ReLU(),
        )

        self.compress1 = nn.Sequential(
          nn.Conv3d(in_channels=out_channels, out_channels=int(out_channels/4), kernel_size=1, stride=1, padding=0, dilation=1),
          nn.ReLU(),
        )
        self.compress2 = nn.Sequential(
          nn.Conv3d(in_channels=out_channels, out_channels=int(out_channels/4), kernel_size=1, stride=1, padding=0, dilation=1),
          nn.ReLU(),
        )
        self.compress3 = nn.Sequential(
          nn.Conv3d(in_channels=out_channels, out_channels=int(out_channels/4), kernel_size=1, stride=1, padding=0, dilation=1),
          nn.ReLU(),
        )
        self.compress4 = nn.Sequential(
          nn.Conv3d(in_channels=out_channels, out_channels=int(out_channels/4), kernel_size=1, stride=1, padding=0, dilation=1),
          nn.ReLU(),
        )
        self.compress5 = nn.Sequential(
          nn.Conv3d(in_channels=out_channels, out_channels=int(out_channels/4), kernel_size=1, stride=1, padding=0, dilation=1),
          nn.ReLU(),
        )

        self.upsample1 = nn.Upsample(size=self.patch_size, mode='trilinear', align_corners=False)  # [96, 160, 160]
        self.upsample2 = nn.Upsample(size=self.patch_size, mode='trilinear', align_corners=False)
        self.upsample3 = nn.Upsample(size=self.patch_size, mode='trilinear', align_corners=False)
        self.upsample4 = nn.Upsample(size=self.patch_size, mode='trilinear', align_corners=False)
        self.upsample5 = nn.Upsample(size=self.patch_size, mode='trilinear', align_corners=False)
        self.upsample6 = nn.Upsample(size=self.patch_size, mode='trilinear', align_corners=False)

    def forward(self, x):
        x = self.block1(x)
        compress1 = self.compress1(x)
        x = self.block2(x)
        compress2 = self.compress2(x)
        x = self.block3(x)
        compress3 = self.compress3(x)
        x = self.block4(x)
        compress4 = self.compress4(x)
        x = self.block5(x)
        compress5 = self.compress5(x)
        compress1 = self.upsample1(compress1)
        compress2 = self.upsample2(compress2)
        compress3 = self.upsample3(compress3)
        compress4 = self.upsample4(compress4)
        compress5 = self.upsample5(compress5)
        x = torch.cat((compress1, compress2, compress3, compress4, compress5), dim=1)
        x = self.block6(x)
        x = self.upsample6(x)
        # x = softmax_helper(x)
        return x

    def compute_approx_vram_consumption(self):
        return 715000000


# class P_Net(SegmentationNetwork):
#     def __init__(self, in_channels=2, out_channels=32, num_classes=2, weightInitializer=InitWeights_He(1e-2), deep_supervision=False):  # or out_channels = 16/64
#         super(P_Net, self).__init__()
#
#         self.do_ds = False
#
#         self.block1 = nn.Sequential(
#           nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
#           nn.ReLU(),
#           nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=1), # or kernel_size=[3, 3, 3]
#           nn.ReLU(),
#         )
#         self.block2 = nn.Sequential(
#           nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=2, dilation=2),
#           nn.ReLU(),
#           nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=2, dilation=2), # or kernel_size=[3, 3, 3]
#           nn.ReLU(),
#         )
#         self.block3 = nn.Sequential(
#           nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=3, dilation=3), # or kernel_size=[3, 3, 1]
#           nn.ReLU(),
#           nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=3, dilation=3),
#           nn.ReLU(),
#           nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=3, dilation=3),
#           nn.ReLU(),
#         )
#         self.block4 = nn.Sequential(
#           nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=4, dilation=4), # or kernel_size=[3, 3, 1]
#           nn.ReLU(),
#           nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=4, dilation=4),
#           nn.ReLU(),
#           nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=4, dilation=4),
#           nn.ReLU(),
#         )
#         self.block5 = nn.Sequential(
#           nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=5, dilation=5), # or kernel_size=[3, 3, 1]
#           nn.ReLU(),
#           nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=5, dilation=5),
#           nn.ReLU(),
#           nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=5, dilation=5),
#           nn.ReLU(),
#         )
#         self.block6 = nn.Sequential(
#           nn.Conv3d(in_channels=int(out_channels/4)*5, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=1), # or kernel_size=[3, 3, 1]
#           nn.ReLU(),
#           nn.Conv3d(in_channels=out_channels, out_channels=num_classes, kernel_size=3, stride=1, padding=1, dilation=1),
#           # nn.ReLU(),
#         )
#
#         self.compress1 = nn.Sequential(
#           nn.Conv3d(in_channels=out_channels, out_channels=int(out_channels/4), kernel_size=1, stride=1, padding=0, dilation=1),
#           nn.ReLU(),
#         )
#         self.compress2 = nn.Sequential(
#           nn.Conv3d(in_channels=out_channels, out_channels=int(out_channels/4), kernel_size=1, stride=1, padding=0, dilation=1),
#           nn.ReLU(),
#         )
#         self.compress3 = nn.Sequential(
#           nn.Conv3d(in_channels=out_channels, out_channels=int(out_channels/4), kernel_size=1, stride=1, padding=0, dilation=1),
#           nn.ReLU(),
#         )
#         self.compress4 = nn.Sequential(
#           nn.Conv3d(in_channels=out_channels, out_channels=int(out_channels/4), kernel_size=1, stride=1, padding=0, dilation=1),
#           nn.ReLU(),
#         )
#         self.compress5 = nn.Sequential(
#           nn.Conv3d(in_channels=out_channels, out_channels=int(out_channels/4), kernel_size=1, stride=1, padding=0, dilation=1),
#           nn.ReLU(),
#         )
#
#         self.apply(weightInitializer)
#
#     def forward(self, x):
#         x = self.block1(x)
#         compress1 = self.compress1(x)
#         x = self.block2(x)
#         compress2 = self.compress2(x)
#         x = self.block3(x)
#         compress3 = self.compress3(x)
#         x = self.block4(x)
#         compress4 = self.compress4(x)
#         x = self.block5(x)
#         compress5 = self.compress5(x)
#         x = torch.cat((compress1, compress2, compress3, compress4, compress5), dim=1)
#         x = self.block6(x)
#         return x
#
#     def compute_approx_vram_consumption(self):
#         return 715000000

