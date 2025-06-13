import torch
from torch import nn
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

class ResEncUnetWithClsRobust(nn.Module):
    """
    ResEnc U-Net backbone with a robust classification head.
    Loads pretrained segmentation weights and freezes them.
    Features:
    - Deeper classification head with dropout and batch norm
    - Multi-scale feature fusion
    - Better initialization
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
        # freeze backbone parameters initially
        for param in self.unet.parameters():
            param.requires_grad = False
        self.unet.encoder.eval()
        self.unet.decoder.eval()

        # zero out dropout layers in the encoder
        for module in self.unet.encoder.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.p = 0.0

        # Expose decoder for nnU-Net trainer compatibility
        self.decoder = self.unet.decoder
        # Expose other potentially needed attributes
        self.encoder = self.unet.encoder
        # Store constructor parameters since ResidualEncoderUNet doesn't expose them as attributes
        self.input_channels = input_channels
        self.num_classes = num_classes

        # Get the deepest encoder feature dimension (bottleneck)
        features_per_stage = unet_kwargs['features_per_stage']
        bottleneck_dim = features_per_stage[-1]  # Deepest features (320 in your case)

        # Simple but effective classification head inspired by the decoder structure
        # Uses the same architectural principles as the original nnU-Net
        # Replace InstanceNorm1d with BatchNorm1d for better convergence
        self.cls_head = nn.Sequential(
            # Global spatial pooling to convert spatial features to vector
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),

            # Follow nnU-Net's decoder pattern: conv + norm + activation
            # First layer: reduce dimensionality while preserving information
            nn.Linear(bottleneck_dim, bottleneck_dim // 2),  # 320 -> 160
            nn.BatchNorm1d(bottleneck_dim // 2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(0.2),

            # Second layer: further dimensionality reduction
            nn.Linear(bottleneck_dim // 2, bottleneck_dim // 4),  # 160 -> 80
            nn.BatchNorm1d(bottleneck_dim // 4),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(0.3),

            # Third layer: compress to intermediate representation
            nn.Linear(bottleneck_dim // 4, bottleneck_dim // 8),  # 80 -> 40
            nn.BatchNorm1d(bottleneck_dim // 8),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(0.3),

            # Output layer: final classification
            nn.Linear(bottleneck_dim // 8, num_cls_classes)  # 40 -> 3
        )

        # Initialize classification head with Xavier initialization
        self._init_classification_head()

    def _init_classification_head(self):
        """Initialize classification head weights following nnU-Net conventions"""
        for module in self.cls_head.modules():
            if isinstance(module, nn.Linear):
                # Use He initialization (similar to what nnU-Net uses for convolutions)
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.InstanceNorm1d)):
                # Initialize norm layers like nnU-Net, but check if weights exist
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Get encoder features for classification
        encoder_features = self.unet.encoder(x)

        # Use the deepest encoder features (bottleneck) for classification
        # This contains the most abstract, high-level features suitable for global classification
        bottleneck_features = encoder_features[-1]  # Shape: [batch, 320, depth, height, width]

        # Classification prediction using bottleneck features
        cls_out = self.cls_head(bottleneck_features)

        # Get full segmentation output
        seg_out = self.unet(x)

        return {'segmentation': seg_out, 'classification': cls_out}

    def setup_partial_finetuning(self, base_lr=1e-3, encoder_lr=1e-5, unfreeze_stages=None):
        """
        Setup partial fine-tuning for the multi-task model.

        Args:
            base_lr: Learning rate for classification head
            encoder_lr: Learning rate for encoder layers (much smaller)
            unfreeze_stages: List of stage indices to unfreeze (0-based).
                           If None, unfreezes the last 2 stages by default.

        Returns:
            optimizer: Configured optimizer with different learning rates
        """
        # Keep segmentation head (decoder) completely frozen
        for param in self.unet.decoder.parameters():
            param.requires_grad = False

        # Keep encoder frozen initially
        for param in self.unet.encoder.parameters():
            param.requires_grad = False

        # Default: unfreeze the last 2 stages (deepest layers)
        if unfreeze_stages is None:
            # For a 6-stage encoder, this would be stages 4 and 5 (0-indexed)
            unfreeze_stages = [-2, -1]  # Last 2 stages

        # Get the number of stages in the encoder
        n_stages = len(self.unet.encoder.stages)

        # Convert negative indices to positive
        unfreeze_stages = [stage if stage >= 0 else n_stages + stage for stage in unfreeze_stages]

        # Unfreeze specified encoder stages
        unfrozen_params = []
        for stage_idx in unfreeze_stages:
            if 0 <= stage_idx < n_stages:
                stage = self.unet.encoder.stages[stage_idx]
                for param in stage.parameters():
                    param.requires_grad = True
                unfrozen_params.extend(list(stage.parameters()))
                print(f"Unfrozing encoder stage {stage_idx}")

        # Create optimizer with different learning rates
        param_groups = [
            {
                'params': [p for p in unfrozen_params if p.requires_grad],
                'lr': encoder_lr,
                'name': 'encoder_stages'
            },
            {
                'params': self.cls_head.parameters(),
                'lr': base_lr,
                'name': 'cls_head'
            }
        ]

        # Filter out empty parameter groups
        param_groups = [group for group in param_groups if len(list(group['params'])) > 0]

        optimizer = torch.optim.Adam(param_groups)

        print(f"Setup partial fine-tuning:")
        print(f"  - Unfrozen encoder stages: {unfreeze_stages}")
        print(f"  - Encoder LR: {encoder_lr}")
        print(f"  - Classification head LR: {base_lr}")
        print(f"  - Total unfrozen encoder params: {sum(p.numel() for p in unfrozen_params if p.requires_grad)}")
        print(f"  - Classification head params: {sum(p.numel() for p in self.cls_head.parameters())}")

        return optimizer

    def setup_full_encoder_finetuning(self, base_lr=1e-3, encoder_lr=1e-5):
        """
        Setup full encoder fine-tuning: freeze only segmentation decoder.
        This is more aggressive than partial fine-tuning and allows the entire
        encoder to adapt to classification while preserving segmentation capability.

        Args:
            base_lr: Learning rate for classification head
            encoder_lr: Learning rate for encoder layers (should be smaller)

        Returns:
            optimizer: Configured optimizer with different learning rates
        """
        # Freeze only decoder-specific parameters (not the embedded encoder)
        for name, param in self.unet.decoder.named_parameters():
            if not name.startswith('encoder.'):
                param.requires_grad = False
            else:
                param.requires_grad = True

        # Unfreeze entire encoder
        for param in self.unet.encoder.parameters():
            param.requires_grad = True

        # Classification head is already trainable
        for param in self.cls_head.parameters():
            param.requires_grad = True

        # Create optimizer with different learning rates
        encoder_params = list(self.unet.encoder.parameters())
        cls_head_params = list(self.cls_head.parameters())

        param_groups = [
            {
                'params': encoder_params,
                'lr': encoder_lr,
                'name': 'full_encoder'
            },
            {
                'params': cls_head_params,
                'lr': base_lr,
                'name': 'cls_head'
            }
        ]

        import torch.optim as optim
        optimizer = optim.Adam(param_groups)

        # Print setup info
        encoder_trainable = sum(p.numel() for p in encoder_params if p.requires_grad)
        cls_trainable = sum(p.numel() for p in cls_head_params if p.requires_grad)
        decoder_only_trainable = sum(p.numel() for name, p in self.unet.decoder.named_parameters()
                                    if p.requires_grad and not name.startswith('encoder.'))

        print(f"Setup full encoder fine-tuning:")
        print(f"  - Full encoder params: {encoder_trainable:,}")
        print(f"  - Classification head params: {cls_trainable:,}")
        print(f"  - Decoder-only trainable params: {decoder_only_trainable:,}")
        print(f"  - Encoder LR: {encoder_lr}")
        print(f"  - Classification head LR: {base_lr}")

        return optimizer

    def freeze_all_encoder(self):
        """Freeze all encoder parameters (useful for resetting)"""
        for param in self.unet.encoder.parameters():
            param.requires_grad = False
        print("All encoder parameters frozen")

    def get_trainable_params_info(self):
        """Get information about trainable parameters"""
        encoder_trainable = sum(p.numel() for p in self.unet.encoder.parameters() if p.requires_grad)
        decoder_trainable = sum(p.numel() for p in self.unet.decoder.parameters() if p.requires_grad)
        cls_trainable = sum(p.numel() for p in self.cls_head.parameters() if p.requires_grad)

        total_params = sum(p.numel() for p in self.parameters())
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'encoder_trainable': encoder_trainable,
            'decoder_trainable': decoder_trainable,
            'cls_head_trainable': cls_trainable,
            'total_params': total_params,
            'total_trainable': total_trainable,
            'trainable_percentage': 100.0 * total_trainable / total_params
        }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to pretrained ResEnc U-Net checkpoint')
    args = parser.parse_args()
    # Example instantiation
    model = ResEncUnetWithClsRobust(
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
    print('Enhanced ResEnc U-Net with robust classification head loaded successfully.')
