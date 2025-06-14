import torch
from torch import nn
import torch.nn.functional as F
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder


# Spatial attention multi-scale classification head that builds on successful architecture.
class SpatialAttentionMultiScaleHead(nn.Module):
    """
    Enhanced multi-scale classification head with spatial attention mechanisms.
    Builds on your successful architecture by adding attention before pooling.
    """
    def __init__(self, encoder_channels=[32, 64, 128, 256, 320, 320], bottleneck_dim=320, num_classes=3):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.num_stages = len(encoder_channels)

        # Spatial attention modules for each scale
        self.spatial_attentions = nn.ModuleList([
            SpatialAttentionBlock(ch) for ch in encoder_channels
        ])

        # Weighted pooling instead of simple average
        self.weighted_pools = nn.ModuleList([
            WeightedAdaptivePool3d() for _ in encoder_channels
        ])

        # Project each scale to same dimension (keeping your proven architecture)
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(ch, bottleneck_dim // 4),
                nn.LayerNorm(bottleneck_dim // 4),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Dropout(0.2)
            ) for ch in encoder_channels
        ])

        # Optional: Learn importance of each scale
        self.scale_weights = nn.Parameter(torch.ones(len(encoder_channels)))

        # Keep your successful classifier architecture
        concat_dim = (bottleneck_dim // 4) * len(encoder_channels)
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(bottleneck_dim, bottleneck_dim // 2),
            nn.LayerNorm(bottleneck_dim // 2),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(bottleneck_dim // 2, num_classes)
        )

        self._init_weights()

    def forward(self, encoder_features):
        """
        Args:
            encoder_features: List of features from encoder stages
        """
        if len(encoder_features) != self.num_stages:
            raise ValueError(f"Expected {self.num_stages} encoder features, got {len(encoder_features)}")

        pooled_features = []
        attention_maps = []  # For visualization if needed

        for i, (feat, attn, pool, proj) in enumerate(
            zip(encoder_features, self.spatial_attentions, self.weighted_pools, self.projections)
        ):
            # Generate spatial attention map
            attention_map = attn(feat)
            attention_maps.append(attention_map)

            # Apply attention to features
            attended_feat = feat * attention_map

            # Weighted pooling using attention scores
            pooled = pool(attended_feat, attention_map)

            # Project to common dimension
            projected = proj(pooled)

            # Apply learned scale importance
            scale_weight = F.softmax(self.scale_weights, dim=0)[i]
            pooled_features.append(projected * scale_weight)

        # Concatenate all scale features
        concatenated = torch.cat(pooled_features, dim=1)

        # Final classification
        output = self.classifier(concatenated)

        # Return output and attention maps (useful for interpretability)
        return output, attention_maps

    def _init_weights(self):
        """Initialize weights following nnU-Net conventions"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.LayerNorm, nn.InstanceNorm3d)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')


class SpatialAttentionBlock(nn.Module):
    """
    Generates spatial attention maps highlighting important regions.
    Uses both channel-wise and spatial information.
    """
    def __init__(self, in_channels):
        super().__init__()
        # Channel attention path
        self.channel_attention = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 16, 1),
            nn.BatchNorm3d(in_channels // 16),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 16, in_channels, 1)
        )

        # Spatial attention path
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(in_channels, 1, kernel_size=7, padding=3),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        # Combine both paths
        self.combine = nn.Conv3d(2, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_pool = F.adaptive_avg_pool3d(x, 1)
        max_pool = F.adaptive_max_pool3d(x, 1)
        channel_att = self.channel_attention(avg_pool + max_pool)
        channel_att = torch.sigmoid(channel_att)

        # Apply channel attention
        x_channel = x * channel_att

        # Spatial attention on channel-attended features
        spatial_att = self.spatial_attention(x_channel)

        # Alternative: Combine avg and max for spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)

        # Final spatial attention map
        attention = self.sigmoid(self.combine(combined))

        return attention


class WeightedAdaptivePool3d(nn.Module):
    """
    Performs weighted pooling where weights come from attention maps.
    More sophisticated than simple average pooling.
    """
    def __init__(self):
        super().__init__()

    def forward(self, features, attention_weights):
        """
        Args:
            features: [B, C, D, H, W]
            attention_weights: [B, 1, D, H, W]
        """
        B, C, D, H, W = features.shape

        # Normalize attention weights
        attention_weights = attention_weights.view(B, 1, -1)
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = attention_weights.view(B, 1, D, H, W)

        # Weighted sum
        features_weighted = features * attention_weights
        pooled = features_weighted.view(B, C, -1).sum(dim=-1)

        return pooled


# Simpler alternative if the above is too complex
class SimpleSpatialAttentionHead(nn.Module):
    """
    A simpler version that adds minimal attention to your working architecture.
    Less risky to implement while still providing attention benefits.
    """
    def __init__(self, encoder_channels=[32, 64, 128, 256, 320, 320], bottleneck_dim=320, num_classes=3):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.num_stages = len(encoder_channels)

        # Simple attention: just learn what regions to focus on
        self.attention_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(ch, ch // 8, 1),
                nn.ReLU(inplace=True),
                nn.Conv3d(ch // 8, 1, 1),
                nn.Sigmoid()
            ) for ch in encoder_channels
        ])

        # Keep everything else from your working architecture
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool3d(1) for _ in encoder_channels
        ])

        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(ch, bottleneck_dim // 4),
                nn.LayerNorm(bottleneck_dim // 4),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Dropout(0.2)
            ) for ch in encoder_channels
        ])

        concat_dim = (bottleneck_dim // 4) * len(encoder_channels)
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(bottleneck_dim, bottleneck_dim // 2),
            nn.LayerNorm(bottleneck_dim // 2),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(bottleneck_dim // 2, num_classes)
        )

        self._init_weights()

    def forward(self, encoder_features):
        pooled_features = []

        for feat, attn_conv, pool, proj in zip(
            encoder_features, self.attention_convs, self.pools, self.projections
        ):
            # Generate simple attention map
            attention = attn_conv(feat)

            # Apply attention
            attended_feat = feat * attention

            # Standard pooling (proven to work in your setup)
            pooled = pool(attended_feat).flatten(1)

            # Project
            projected = proj(pooled)
            pooled_features.append(projected)

        concatenated = torch.cat(pooled_features, dim=1)
        return self.classifier(concatenated)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.LayerNorm, nn.InstanceNorm1d, nn.InstanceNorm3d)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')


# Multiscale classification head that combines features from all encoder stages.
class MultiScaleClassificationHead(nn.Module):
    """
    Multi-scale classification head that combines features from all encoder stages.
    Similar to Feature Pyramid Networks (FPN) but for classification.
    """
    def __init__(self, encoder_channels=[32, 64, 128, 256, 320, 320], bottleneck_dim=320, num_classes=3):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.num_stages = len(encoder_channels)

        # Adaptive pooling for each scale to normalize spatial dimensions
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool3d(1) for _ in encoder_channels
        ])

        # Project each scale to same dimension for concatenation
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(ch, bottleneck_dim // 4),
                nn.LayerNorm(bottleneck_dim // 4),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Dropout(0.2)
            ) for ch in encoder_channels
        ])

        # Final classifier that processes concatenated multi-scale features
        concat_dim = (bottleneck_dim // 4) * len(encoder_channels)
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(bottleneck_dim, bottleneck_dim // 2),
            nn.LayerNorm(bottleneck_dim // 2),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(bottleneck_dim // 2, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def forward(self, encoder_features):
        """
        Args:
            encoder_features: List of features from encoder stages
                             [stage0, stage1, stage2, stage3, stage4, stage5]
        """
        if len(encoder_features) != self.num_stages:
            raise ValueError(f"Expected {self.num_stages} encoder features, got {len(encoder_features)}")

        # Process each scale
        pooled_features = []
        for feat, pool, proj in zip(encoder_features, self.pools, self.projections):
            # Global average pooling to get [batch, channels]
            pooled = pool(feat).flatten(1)
            # Project to common dimension
            projected = proj(pooled)
            pooled_features.append(projected)

        # Concatenate all scale features
        concatenated = torch.cat(pooled_features, dim=1)

        # Final classification
        return self.classifier(concatenated)

    def _init_weights(self):
        """Initialize weights following nnU-Net conventions"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.LayerNorm, nn.InstanceNorm1d)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0)


class UNetDecoderClassificationHead(nn.Module):
    """
    Classification head that uses UNet decoder architecture for feature fusion.
    This reuses the proven UNet feature fusion patterns but for classification.
    Leverages hierarchical feature fusion from the UNet decoder design.
    """
    def __init__(self, encoder_channels, num_classes, conv_op=torch.nn.Conv3d,
                 norm_op=torch.nn.InstanceNorm3d, nonlin=torch.nn.LeakyReLU):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.nonlin = nonlin

        # Create custom decoder layers for feature fusion (inspired by UNet decoder)
        # We'll progressively fuse features from deep to shallow
        self.decoder_layers = nn.ModuleList()

        # Start from deepest features and work upward
        in_channels = encoder_channels[-1]  # Start with bottleneck features

        for i in range(len(encoder_channels) - 2, -1, -1):  # Go from second-to-last to first
            skip_channels = encoder_channels[i]
            out_channels = skip_channels  # Output same channels as skip connection

            # Decoder block: upsampling + skip connection + convolution
            decoder_block = nn.Sequential(
                # Upsample and combine with skip connection
                conv_op(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
                norm_op(out_channels),
                nonlin(inplace=True),
                conv_op(out_channels, out_channels, kernel_size=3, padding=1),
                norm_op(out_channels),
                nonlin(inplace=True)
            )
            self.decoder_layers.append(decoder_block)
            in_channels = out_channels

        # Global pooling and final classification layers
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        final_channels = encoder_channels[0]  # Channels after all decoder layers

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_channels, final_channels * 2),
            nn.LayerNorm(final_channels * 2),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(final_channels * 2, final_channels),
            nn.LayerNorm(final_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(final_channels, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following nnU-Net conventions"""
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.InstanceNorm1d, nn.InstanceNorm3d, nn.InstanceNorm2d, nn.InstanceNorm1d)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, encoder_features):
        """
        Args:
            encoder_features: List of features from encoder stages
                             [stage0, stage1, stage2, stage3, stage4, stage5]
        """
        # Start with the deepest features (bottleneck)
        x = encoder_features[-1]

        # Progressive feature fusion using decoder-style architecture
        for i, decoder_layer in enumerate(self.decoder_layers):
            # Get corresponding skip connection (going from deep to shallow)
            skip_idx = len(encoder_features) - 2 - i
            skip_features = encoder_features[skip_idx]

            # Upsample current features to match skip connection spatial size
            target_size = skip_features.shape[2:]  # Get spatial dimensions
            x_upsampled = F.interpolate(x, size=target_size, mode='trilinear', align_corners=False)

            # Concatenate with skip connection and apply decoder block
            x = torch.cat([x_upsampled, skip_features], dim=1)
            x = decoder_layer(x)

        # Global pooling and classification
        pooled = self.global_pool(x)
        return self.classifier(pooled)


class ResEncUnetWithClsImproved(nn.Module):
    """
    ResEnc U-Net backbone with improved classification heads and robust training.
    Loads pretrained segmentation weights and freezes them.
    Features:
    - Deeper classification head with dropout and batch norm
    - Multi-scale feature fusion
    - UNet decoder-based classification
    - Better initialization
    - Flexible training modes (partial/full encoder fine-tuning)
    """
    def __init__(self, pretrained_checkpoint: str, input_channels: int, num_classes: int, num_cls_classes: int,
                 classification_mode='bottleneck', **unet_kwargs):
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

        # Get encoder configuration
        features_per_stage = unet_kwargs['features_per_stage']
        bottleneck_dim = features_per_stage[-1]  # Deepest features (320 in your case)

        # Store classification mode and create appropriate classification head
        self.classification_mode = classification_mode

        if classification_mode == 'bottleneck':
            # Simple but effective classification head using only bottleneck features
            self.cls_head = nn.Sequential(
                # Global spatial pooling to convert spatial features to vector
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),

                # Follow nnU-Net's decoder pattern: conv + norm + activation
                # First layer: reduce dimensionality while preserving information
                nn.Linear(bottleneck_dim, bottleneck_dim // 2),  # 320 -> 160
                nn.LayerNorm(bottleneck_dim // 2),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(0.2),

                # Second layer: further dimensionality reduction
                nn.Linear(bottleneck_dim // 2, bottleneck_dim // 4),  # 160 -> 80
                nn.LayerNorm(bottleneck_dim // 4),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(0.3),

                # Third layer: compress to intermediate representation
                nn.Linear(bottleneck_dim // 4, bottleneck_dim // 8),  # 80 -> 40
                nn.LayerNorm(bottleneck_dim // 8),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(0.3),

                # Output layer: final classification
                nn.Linear(bottleneck_dim // 8, num_cls_classes)  # 40 -> 3
            )
        elif classification_mode == 'multiscale':
            # Multi-scale classification head using all encoder features
            self.cls_head = MultiScaleClassificationHead(
                encoder_channels=features_per_stage,
                bottleneck_dim=bottleneck_dim,
                num_classes=num_cls_classes
            )
        elif classification_mode == 'unet_decoder':
            # UNet decoder-based classification head
            self.cls_head = UNetDecoderClassificationHead(
                encoder_channels=features_per_stage,
                num_classes=num_cls_classes,
                conv_op=unet_kwargs.get('conv_op', torch.nn.Conv3d),
                norm_op=unet_kwargs.get('norm_op', torch.nn.InstanceNorm3d),
                nonlin=unet_kwargs.get('nonlin', torch.nn.LeakyReLU)
            )
        elif classification_mode == 'spatial_attention_multiscale':
            self.cls_head = SpatialAttentionMultiScaleHead(
                encoder_channels=features_per_stage,
                bottleneck_dim=bottleneck_dim,
                num_classes=num_cls_classes
            )
        elif classification_mode == 'simple_spatial_attention':
            self.cls_head = SimpleSpatialAttentionHead(
                encoder_channels=features_per_stage,
                bottleneck_dim=bottleneck_dim,
                num_classes=num_cls_classes
            )
        else:
            raise ValueError(f"Unknown classification_mode: {classification_mode}. "
                           f"Choose from: 'bottleneck', 'multiscale', 'unet_decoder', 'spatial_attention_multiscale', 'simple_spatial_attention'")
        self.classification_mode = classification_mode
        # Initialize classification head with Xavier initialization
        self._init_classification_head()

    def _init_classification_head(self):
        """Initialize classification head weights following nnU-Net conventions"""
        if self.classification_mode == 'bottleneck':
            for module in self.cls_head.modules():
                if isinstance(module, nn.Linear):
                    # Use He initialization (similar to what nnU-Net uses for convolutions)
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, (nn.BatchNorm1d, nn.InstanceNorm1d, nn.LayerNorm)):
                    # Initialize norm layers like nnU-Net, but check if weights exist
                    if hasattr(module, 'weight') and module.weight is not None:
                        nn.init.constant_(module.weight, 1)
                    if hasattr(module, 'bias') and module.bias is not None:
                        nn.init.constant_(module.bias, 0)
        # For multiscale and unet_decoder modes, initialization is handled in their respective classes

    def forward(self, x):
        # Get encoder features for classification
        encoder_features = self.unet.encoder(x)

        # Classification prediction based on mode
        if self.classification_mode == 'bottleneck':
            # Use only the deepest encoder features (bottleneck)
            bottleneck_features = encoder_features[-1]  # Shape: [batch, 320, depth, height, width]
            cls_out = self.cls_head(bottleneck_features)
        else:
            # Use all encoder features (multiscale or unet_decoder)
            if self.classification_mode in ['multiscale', 'unet_decoder', 'simple_spatial_attention']:
                cls_out = self.cls_head(encoder_features)
            elif self.classification_mode == 'spatial_attention_multiscale':
                cls_out, _ = self.cls_head(encoder_features)

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
                print(f"Unfreezing encoder stage {stage_idx}")

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

        print(f"Setup partial fine-tuning ({self.classification_mode} mode):")
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

        optimizer = torch.optim.Adam(param_groups)

        # Print setup info
        encoder_trainable = sum(p.numel() for p in encoder_params if p.requires_grad)
        cls_trainable = sum(p.numel() for p in cls_head_params if p.requires_grad)
        decoder_only_trainable = sum(p.numel() for name, p in self.unet.decoder.named_parameters()
                                    if p.requires_grad and not name.startswith('encoder.'))

        print(f"Setup full encoder fine-tuning ({self.classification_mode} mode):")
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
            'trainable_percentage': 100.0 * total_trainable / total_params,
            'classification_mode': self.classification_mode
        }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to pretrained ResEnc U-Net checkpoint')
    parser.add_argument('--mode', type=str, default='bottleneck',
                        choices=['bottleneck', 'multiscale', 'unet_decoder'],
                        help='Classification head mode')
    args = parser.parse_args()

    # Example instantiation
    model = ResEncUnetWithClsImproved(
        pretrained_checkpoint=args.checkpoint,
        input_channels=1,
        num_classes=3,
        num_cls_classes=3,
        classification_mode=args.mode,
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
    print(f'Enhanced ResEnc U-Net with robust classification head ({args.mode} mode) loaded successfully.')
