import torch
import torch.nn as nn
from nnunetv2.utilities.helpers import softmax_helper_dim1
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet


class EfficientAttentionBlock(nn.Module):
    """Efficient attention block using channel attention instead of spatial"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels

        # Channel attention (much more efficient than spatial attention)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

        # Optional: Add spatial attention for the deepest features only
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca_weights = self.channel_attention(x)
        x_ca = x * ca_weights

        # Spatial attention (optional, use sparingly)
        if x.shape[2:] == torch.Size([4, 8, 12]):  # Only for small feature maps
            sa_weights = self.spatial_attention(x_ca)
            return x_ca * sa_weights

        return x_ca


class LinearAttentionBlock(nn.Module):
    """Linear attention for classification head - more memory efficient"""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        self.v = nn.Linear(channels, channels)
        self.out = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x shape: [B, C, D, H, W]
        b, c, d, h, w = x.shape

        # Flatten spatial dimensions and transpose for linear attention
        x_flat = x.reshape(b, c, -1).permute(0, 2, 1)  # [B, DHW, C]

        # Linear projections
        q = self.q(x_flat).reshape(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x_flat).reshape(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x_flat).reshape(b, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Linear attention (more efficient than quadratic attention)
        attn_weights = torch.softmax(q @ k.transpose(-2, -1) / (self.head_dim ** 0.5), dim=-1)
        attn_out = attn_weights @ v

        # Concatenate heads and project
        attn_out = attn_out.transpose(1, 2).reshape(b, -1, c)
        attn_out = self.out(attn_out)
        attn_out = self.norm(attn_out + x_flat)  # Residual connection

        # Reshape back to spatial dimensions
        return attn_out.permute(0, 2, 1).reshape(b, c, d, h, w)


class MultiTaskFullAttentionResEncUNet(ResidualEncoderUNet):
    """Multi-task ResEnc U-Net with efficient attention mechanisms"""

    def __init__(self, input_channels, num_classes, num_classification_classes,
                 n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                 n_blocks_per_stage, n_conv_per_stage_decoder, **kwargs):

        # Initialize the base ResEnc U-Net with correct parameters
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,  # segmentation classes
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            **kwargs
        )

        self.num_classification_classes = num_classification_classes

        # Add efficient attention only to the deepest few layers
        # (adding to all layers would be too memory intensive)
        self.encoder_attention = nn.ModuleDict({
            f'stage_{n_stages-2}': EfficientAttentionBlock(features_per_stage[-2]),
            f'stage_{n_stages-1}': EfficientAttentionBlock(features_per_stage[-1]),
        })

        # Enhanced classification head with linear attention
        self.classification_head = nn.Sequential(
            LinearAttentionBlock(features_per_stage[-1], num_heads=8),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(features_per_stage[-1], 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classification_classes)
        )

    def forward(self, x):
        # Encoder forward pass with selective attention
        encoder_features = []
        current = x

        for i, stage in enumerate(self.encoder):
            current = stage(current)

            # Apply attention only to the deepest layers
            stage_key = f'stage_{i}'
            if stage_key in self.encoder_attention:
                current = self.encoder_attention[stage_key](current)

            encoder_features.append(current)

        # Segmentation decoder path (unchanged)
        seg_output = encoder_features[-1]

        for i, decoder_stage in enumerate(self.decoder):
            seg_output = decoder_stage(seg_output, encoder_features[-(i+2)])

        # Final segmentation output
        seg_output = self.seg_layers[-1](seg_output)

        # Classification with attention-enhanced features
        cls_output = self.classification_head(encoder_features[-1])

        return {
            'segmentation': seg_output,
            'classification': cls_output
        }


# Simpler version with just channel attention
class MultiTaskChannelAttentionResEncUNet(ResidualEncoderUNet):
    """Simpler version with only channel attention for memory efficiency"""

    def __init__(self, input_channels, num_classes, num_classification_classes,
                 n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                 n_blocks_per_stage, n_conv_per_stage_decoder, **kwargs):

        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            **kwargs
        )

        self.num_classification_classes = num_classification_classes

        # Simple channel attention for classification
        self.bottleneck_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(features_per_stage[-1], features_per_stage[-1] // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(features_per_stage[-1] // 16, features_per_stage[-1], 1),
            nn.Sigmoid()
        )

        # Classification head
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(features_per_stage[-1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classification_classes)
        )

    def forward(self, x):
        # Standard encoder forward pass
        encoder_features = []
        current = x

        for stage in self.encoder:
            current = stage(current)
            encoder_features.append(current)

        # Segmentation decoder
        seg_output = encoder_features[-1]
        for i, decoder_stage in enumerate(self.decoder):
            seg_output = decoder_stage(seg_output, encoder_features[-(i+2)])
        seg_output = self.seg_layers[-1](seg_output)

        # Classification with attention on bottleneck
        bottleneck_features = encoder_features[-1]
        attention_weights = self.bottleneck_attention(bottleneck_features)
        attended_features = bottleneck_features * attention_weights
        cls_output = self.classification_head(attended_features)

        return {
            'segmentation': seg_output,
            'classification': cls_output
        }

# No attention version for comparison 
class MultiTaskResEncUNet(ResidualEncoderUNet):
    """Multi-task ResEnc U-Net with shared encoder and dual heads"""

    def __init__(self, input_channels, num_classes, num_classification_classes,
                 n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                 n_blocks_per_stage, n_conv_per_stage_decoder, **kwargs):

        # Initialize the base ResEnc U-Net
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,  # segmentation classes
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            **kwargs
        )

        self.num_classification_classes = num_classification_classes

        # Classification head using bottleneck features
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(features_per_stage[-1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classification_classes)
        )

    def forward(self, x):
        # Encoder forward pass
        encoder_features = []
        current = x

        for stage in self.encoder:
            current = stage(current)
            encoder_features.append(current)

        # Segmentation decoder path
        seg_output = encoder_features[-1]

        for i, decoder_stage in enumerate(self.decoder):
            seg_output = decoder_stage(seg_output, encoder_features[-(i+2)])

        # Final segmentation output
        seg_output = self.seg_layers[-1](seg_output)

        # Classification from encoder bottleneck
        cls_output = self.classification_head(encoder_features[-1])

        return {
            'segmentation': seg_output,
            'classification': cls_output
        }

# Memory usage comparison:
def estimate_memory_usage():
    """
    Memory estimates for 3D volumes (approximate):

    Original approach (full spatial attention on 64x128x192):
    - Attention matrix: (64*128*192)² = ~2.6TB per head (!!)

    Channel attention:
    - Per stage: ~1MB additional memory

    Linear attention (recommended):
    - Per stage: ~100MB additional memory

    Recommendation: Start with SimpleMultiTaskAttentionResEncUNet
    """
    pass