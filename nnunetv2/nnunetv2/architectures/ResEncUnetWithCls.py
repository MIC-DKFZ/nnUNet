import argparse
import torch
from torch import nn
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

class ResEncUnetWithCls(nn.Module):
    """
    ResEnc U-Net backbone with a classification head.
    Loads pretrained segmentation weights and freezes them.
    """
    def __init__(self, pretrained_checkpoint: str, input_channels: int, num_classes: int, num_cls_classes: int, **unet_kwargs):
        super().__init__()
        # instantiate backbone
        self.unet = ResidualEncoderUNet(input_channels=input_channels,
                                        num_classes=num_classes,
                                        **unet_kwargs)
        # load pretrained weights
        checkpoint = torch.load(pretrained_checkpoint, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('network_weights', checkpoint)
        # load with non-strict to ignore missing cls layers
        self.unet.load_state_dict(state_dict, strict=False)
        # freeze backbone parameters
        for param in self.unet.parameters():
            param.requires_grad = False

        # Expose decoder for nnU-Net trainer compatibility
        self.decoder = self.unet.decoder
        # Expose other potentially needed attributes
        self.encoder = self.unet.encoder
        # Store constructor parameters since ResidualEncoderUNet doesn't expose them as attributes
        self.input_channels = input_channels
        self.num_classes = num_classes

        # classification head
        feat = unet_kwargs['features_per_stage'][-1]
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(feat, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_cls_classes)
        )

    def forward(self, x):
        # Get encoder features for classification
        encoder_features = self.unet.encoder(x)
        # Use the deepest encoder features (highest level) for classification
        deepest_features = encoder_features[-1]  # Last encoder stage
        cls_out = self.cls_head(deepest_features)

        # Get full segmentation output
        seg_out = self.unet(x)

        return {'segmentation': seg_out, 'classification': cls_out}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to pretrained ResEnc U-Net checkpoint')
    args = parser.parse_args()
    # Example instantiation
    model = ResEncUnetWithCls(
        pretrained_checkpoint=args.checkpoint,
        input_channels=1,
        num_classes=3,
        num_cls_classes=3,
        n_stages=6,
        features_per_stage=(32, 64, 128, 256, 320, 320),
        conv_op=torch.nn.Conv3d,
        kernel_sizes=[
            (1, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3)
        ],
        strides=[
            (1, 1, 1),
            (1, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2)
        ],
        n_blocks_per_stage=(1, 3, 4, 6, 6, 6),
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
        conv_bias=True,
        norm_op=torch.nn.InstanceNorm3d,
        norm_op_kwargs={},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={'inplace': True},
        deep_supervision=True
    )
    print('Pretrained ResEnc U-Net loaded and classification head initialized successfully.')
