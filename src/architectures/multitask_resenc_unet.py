import torch
import torch.nn as nn
from typing import Union, Tuple, List
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim


class MultiTaskResEncUNet(ResidualEncoderUNet):
    """
    Multi-task ResidualEncoderUNet with shared encoder and dual decoders
    for segmentation and classification tasks
    """

    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: type,
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, List[int], Tuple[int, ...]] = None,
                 conv_bias: bool = False,
                 norm_op: Union[None, type] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, type] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, type] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 classification_config: dict = None):

        # Store conv_op for later use
        self.conv_op = conv_op

        # Set default for n_conv_per_stage_decoder if None
        if n_conv_per_stage_decoder is None:
            n_conv_per_stage_decoder = [1] * (n_stages - 1)

        # Initialize parent ResidualEncoderUNet for segmentation
        super().__init__(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision
        )

        # Classification configuration
        if classification_config is None:
            classification_config = {
                'head_type': 'mlp',  # 'mlp' or 'spatial_attention'
                'num_classes': 3,
                'dropout_rate': 0.3,
                'latent_dim': 1024,  # Large latent representation for expressiveness
                'mlp_hidden_dims': [512, 256],  # MLP hidden dimensions
                'use_all_features': False,  # Use last encoder stage for MLP
                # Legacy spatial attention config (kept for backward compatibility)
                'hidden_dims': [256, 128]
            }
        self.classification_config = classification_config

        # Build classification decoder
        self._build_classification_decoder()

        # Manual weights from configuration (no learnable parameters)
        self.seg_weight = 1.0  # Default values, will be updated from config
        self.cls_weight = 1.0

        # Training stage control
        self.training_stage = "full"  # full, enc_seg, enc_cls, joint_finetune
        self._apply_training_stage()

    def initialize(self, module):
        """
        nnUNet-compatible initialization method that gets called via model.apply(model.initialize)
        This replaces the standard nnUNet initialization with our custom multi-task initialization
        """
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            self._initialize_conv_layer(module)
        elif isinstance(module, nn.Linear):
            self._initialize_linear_layer(module)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                                nn.LayerNorm)):
            self._initialize_norm_layer(module)

        # print model parameter count
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✓ Initialized {module.__class__.__name__} with {total_params} total parameters, "
              f"{trainable_params} trainable parameters")

    def _initialize_conv_layer(self, module):
        """Initialize convolutional layers with context-aware strategy"""
        module_name = self._get_module_name(module)

        if 'attention' in module_name and 'spatial_attention' in module_name:
            # Spatial attention conv should start small to prevent saturation
            nn.init.xavier_uniform_(module.weight, gain=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif 'attention' in module_name:
            # Other attention convs
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        else:
            # Standard convolutions - use Kaiming for LeakyReLU(0.2)
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _initialize_linear_layer(self, module):
        """Initialize linear layers with context-aware strategy"""
        module_name = self._get_module_name(module)

        if 'channel_attention' in module_name:
            # Channel attention layers
            if self._is_final_attention_layer(module):
                # Final attention layer (before sigmoid) - small init
                nn.init.xavier_uniform_(module.weight, gain=0.1)
            else:
                # Intermediate attention layers
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        elif 'classifier' in module_name:
            # Final classification layer - very small initialization
            nn.init.normal_(module.weight, 0, 0.001)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        elif 'classification_head' in module_name:
            # Classification head layers - Kaiming for LeakyReLU(0.2)
            layer_depth = self._get_layer_depth(module_name)
            base_gain = 1.0 / max(1, layer_depth)  # Reduce gain for deeper layers

            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
            # Apply depth-based scaling
            with torch.no_grad():
                module.weight.data *= base_gain

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        else:
            # Default linear layer initialization
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _initialize_norm_layer(self, module):
        """Initialize normalization layers"""
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.constant_(module.weight, 1)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, 0)

    def _get_module_name(self, module):
        """Get the hierarchical name of a module for context-aware initialization"""
        for name, mod in self.named_modules():
            if mod is module:
                return name
        return ""

    def _is_final_attention_layer(self, module):
        """Check if this is the final layer in an attention module (before sigmoid)"""
        module_name = self._get_module_name(module)
        # Look for patterns that indicate this is the final attention layer
        return (module_name.endswith('.4') or  # Often the last layer in Sequential
                'sigmoid' in module_name.lower() or
                module_name.count('.') >= 3)  # Deep in the hierarchy

    def _get_layer_depth(self, module_name):
        """Get the depth of a layer within the classification head"""
        if 'classification_head' not in module_name:
            return 1
        # Count the layer depth by parsing the name
        parts = module_name.split('.')
        try:
            # Look for numeric indices that indicate layer position
            layer_indices = [int(p) for p in parts if p.isdigit()]
            return max(layer_indices) + 1 if layer_indices else 1
        except:
            return 1

    def set_manual_weights(self, seg_weight: float, cls_weight: float):
        """Set manual weights for multitask loss"""
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight

    def _build_classification_decoder(self):
        """Build classification decoder based on head_type configuration"""
        head_type = self.classification_config.get('head_type', 'mlp')

        if head_type == 'mlp':
            self._build_mlp_classification_decoder()
        elif head_type == 'spatial_attention':
            self._build_spatial_attention_classification_decoder()
        else:
            raise ValueError(f"Unknown head_type: {head_type}. Must be 'mlp' or 'spatial_attention'")

    def _build_mlp_classification_decoder(self):
        """Build MLP-based classification decoder with latent layer"""
        # Get the last encoder stage output channels
        last_stage_channels = self.encoder.stages[-1].output_channels

        # Add latent representation layer
        latent_dim = self.classification_config.get('latent_dim', 1024)
        self.latent_layer = LatentRepresentationLayer(
            input_channels=last_stage_channels,
            latent_dim=latent_dim,
            conv_op=self.conv_op
        )

        # Build MLP classification head
        self.classification_head = MLPClassificationHead(
            input_channels=latent_dim,
            num_classes=self.classification_config['num_classes'],
            hidden_dims=self.classification_config.get('mlp_hidden_dims', [512, 256]),
            dropout_rate=self.classification_config.get('dropout_rate', 0.3),
            conv_op=self.conv_op
        )

        print(f"✓ Built MLP classification head with latent dim: {latent_dim}")

    def _build_spatial_attention_classification_decoder(self):
        """Build spatial attention-based classification decoder (legacy)"""
        conv_dim = convert_conv_op_to_dim(self.conv_op)

        # Determine which features to use and their types
        if self.classification_config['use_all_features']:
            # Use stem + all encoder stages
            feature_channels = [self.encoder.stem.output_channels]
            scale_types = ["early"]

            for i, stage in enumerate(self.encoder.stages):
                feature_channels.append(stage.output_channels)
                if i < 2:
                    scale_types.append("early")
                elif i < 4:
                    scale_types.append("middle")
                else:
                    scale_types.append("late")
        else:
            # Use last 3 stages for better multi-scale representation
            feature_channels = []
            scale_types = []

            num_stages = len(self.encoder.stages)
            for i in range(max(0, num_stages-3), num_stages):
                feature_channels.append(self.encoder.stages[i].output_channels)
                if i < num_stages - 2:
                    scale_types.append("middle")
                else:
                    scale_types.append("late")

        # Determine output spatial size for pooling
        if conv_dim == 3:
            pool_output_size = (2, 2, 2)  # Small but not 1x1x1
            pool_volume = 8
        else:
            pool_output_size = (2, 2)
            pool_volume = 4

        # Build scale-specific processors
        self.scale_processors = nn.ModuleList([
            ScaleSpecificProcessor(
                channels=channels,
                conv_op=self.conv_op,
                scale_type=scale_type,
                output_size=pool_output_size
            )
            for channels, scale_type in zip(feature_channels, scale_types)
        ])

        # Calculate total feature dimension
        # Each processor outputs: channels * pool_volume * 2 (avg + max pooling)
        total_features = sum(channels * pool_volume * 2 for channels in feature_channels)

        # Enhanced classification head
        self.classification_head = EnhancedClassificationHead(
            total_features=total_features,
            num_classes=self.classification_config['num_classes'],
            dropout_rate=self.classification_config['dropout_rate']
        )

        # Store attention maps for visualization (optional)
        self.attention_maps = []

        print("✓ Built spatial attention classification head")

    def forward_classification_part(self, encoder_outputs):
        """
        Forward pass for classification - handles both MLP and spatial attention
        """
        head_type = self.classification_config.get('head_type', 'mlp')

        if head_type == 'mlp':
            return self._forward_mlp_classification(encoder_outputs)
        elif head_type == 'spatial_attention':
            return self._forward_spatial_attention_classification(encoder_outputs)
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def _forward_mlp_classification(self, encoder_outputs):
        """Forward pass for MLP classification head"""
        # Use the last encoder stage output
        last_encoder_features = encoder_outputs[-1]  # [B, C, H, W, D]

        # Apply latent layer
        latent_features = self.latent_layer(last_encoder_features)  # [B, latent_dim, H, W, D]

        # Apply MLP head (includes global pooling)
        classification_output = self.classification_head(latent_features)

        return classification_output

    def _forward_spatial_attention_classification(self, encoder_outputs):
        """Forward pass for spatial attention classification head (legacy)"""
        if self.classification_config['use_all_features']:
            # Use all encoder outputs including stem
            features_to_process = encoder_outputs
            processors_to_use = self.scale_processors
        else:
            # Use last 3 encoder stages
            num_stages = len(encoder_outputs) - 1  # -1 because first is stem
            start_idx = max(1, num_stages - 2)  # Start from stem + (num_stages - 3)
            features_to_process = encoder_outputs[start_idx:]
            processors_to_use = self.scale_processors

        # Process each scale
        scale_features = []
        self.attention_maps = []  # Store for visualization

        for features, processor in zip(features_to_process, processors_to_use):
            processed_features, attention_map = processor(features)
            scale_features.append(processed_features)
            self.attention_maps.append(attention_map)

        # Concatenate all processed features
        if scale_features:
            fused_features = torch.cat(scale_features, dim=1)
            classification_output = self.classification_head(fused_features)
        else:
            # Fallback to original approach
            cls_x = self.global_pools[0](encoder_outputs[-1])
            classification_output = self.classification_head(cls_x.flatten(1))

        return classification_output

    def forward(self, x):
        """Forward pass returning both segmentation and classification outputs"""
        encoder_outputs = []

        # Process stem first
        x = self.encoder.stem(x)
        encoder_outputs.append(x)

        # Then process encoder stages
        for stage in self.encoder.stages:
            x = stage(x)
            encoder_outputs.append(x)

        # Use parent's forward method for segmentation
        # But we need to replicate the encoder part since we already did it
        skips = encoder_outputs[:-1]  # All except the last (deepest) features
        seg_x = encoder_outputs[-1]   # Deepest features

        # Decoder forward pass (following UNetDecoder pattern)
        for i, (decoder_stage, transpconv) in enumerate(zip(self.decoder.stages, self.decoder.transpconvs)):
            seg_x = transpconv(seg_x)
            seg_x = torch.cat([seg_x, skips[-(i+1)]], dim=1)
            seg_x = decoder_stage(seg_x)

        # The decoder output IS the segmentation result
        segmentation_output = seg_x

        # Classification decoder
        classification_output = self.forward_classification_part(encoder_outputs)

        return {
            'segmentation': segmentation_output,
            'classification': classification_output
        }

    def set_training_stage(self, stage: str):
        """Set training stage and freeze/unfreeze parameters accordingly"""
        valid_stages = ['full', 'enc_seg', 'enc_cls', 'joint_finetune']
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage: {stage}. Must be one of {valid_stages}")

        self.training_stage = stage
        self._apply_training_stage()

    def _apply_training_stage(self):
        """Apply parameter freezing based on training stage - DON'T reinitialize parameters!"""
        # First freeze everything
        for param in self.parameters():
            param.requires_grad = False

        if self.training_stage == "full":
            # Unfreeze all parameters
            for param in self.parameters():
                param.requires_grad = True

        elif self.training_stage == "enc_seg":
            # Unfreeze encoder and segmentation decoder
            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.decoder.parameters():
                param.requires_grad = True

        elif self.training_stage == "enc_cls":
            # Unfreeze encoder and classification head
            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.classification_head.parameters():
                param.requires_grad = True
            # Also unfreeze latent layer if using MLP head
            if hasattr(self, 'latent_layer'):
                for param in self.latent_layer.parameters():
                    param.requires_grad = True
            if hasattr(self, 'global_pools'):
                for param in self.global_pools.parameters():
                    param.requires_grad = True

        elif self.training_stage == "joint_finetune":
            # Unfreeze last two encoder stages and both decoders
            for stage_idx in [-2, -1]:  # Last two stages
                if stage_idx < len(self.encoder.stages):
                    for param in self.encoder.stages[stage_idx].parameters():
                        param.requires_grad = True

            # Unfreeze both decoders
            for param in self.decoder.parameters():
                param.requires_grad = True
            for param in self.classification_head.parameters():
                param.requires_grad = True
            if hasattr(self, 'global_pools'):
                for param in self.global_pools.parameters():
                    param.requires_grad = True


    def get_training_stage_info(self):
        """Get information about current training stage and parameter status"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Count parameters by component
        encoder_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        decoder_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        cls_head_params = sum(p.numel() for p in self.classification_head.parameters() if p.requires_grad)

        info = {
            'training_stage': self.training_stage,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'trainable_ratio': trainable_params / total_params,
            'component_params': {
                'encoder': encoder_params,
                'seg_decoder': decoder_params,
                'cls_head': cls_head_params
            },
            'manual_weights': {
                'seg_weight': self.seg_weight,
                'cls_weight': self.cls_weight
            }
        }

        return info

    def compute_multitask_loss(self, outputs, targets, seg_loss_fn, cls_loss_fn):
        """Compute multitask loss with manual weights"""
        seg_pred = outputs['segmentation']
        cls_pred = outputs['classification']
        seg_target = targets['segmentation']
        cls_target = targets['classification']

        # Compute individual losses
        seg_loss = seg_loss_fn(seg_pred, seg_target)
        cls_loss = cls_loss_fn(cls_pred, cls_target)

        # Stage specific loss handling
        if self.training_stage == "enc_seg":
            # Only segmentation loss
            return {
                'total_loss': seg_loss,
                'segmentation_loss': seg_loss,
                'classification_loss': cls_loss,
                'seg_weight': 1.0,
                'cls_weight': 0.0
            }
        elif self.training_stage == "enc_cls":
            # Only classification loss
            return {
                'total_loss': cls_loss,
                'segmentation_loss': seg_loss,
                'classification_loss': cls_loss,
                'seg_weight': 0.0,
                'cls_weight': 1.0
            }
        else:
            # Simple manual weighting
            total_loss = self.seg_weight * seg_loss + self.cls_weight * cls_loss

            return {
                'total_loss': total_loss,
                'segmentation_loss': seg_loss,
                'classification_loss': cls_loss,
                'seg_weight': self.seg_weight,
                'cls_weight': self.cls_weight
            }

    def post_initialization_setup(self):
        """
        Call this after the model has been created and initialized by nnUNet
        This handles the final setup after initialization
        """
        self._apply_training_stage()
        self._log_initialization_info()

    def _log_initialization_info(self):
        """Log information about the initialization"""
        if hasattr(self, 'attention_maps'):
            print("✓ Multi-task architecture initialized with spatial attention")
            print(f"  - Classification config: {self.classification_config}")
            print(f"  - Training stage: {self.training_stage}")
            print(f"  - Manual weights: seg={self.seg_weight:.3f}, cls={self.cls_weight:.3f}")

class SpatialAttentionModule(nn.Module):
    """
    Spatial attention module that learns where to focus for classification
    """
    def __init__(self, channels: int, conv_op: type, reduction_ratio: int = 16):
        super().__init__()
        self.conv_op = conv_op

        # Channel attention branch
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1) if conv_op == nn.Conv3d else nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels),
            nn.Sigmoid()
        )

        # Spatial attention branch
        kernel_size = 7
        padding = kernel_size // 2
        self.spatial_attention = nn.Sequential(
            conv_op(2, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        b, c = x.shape[:2]
        channel_weights = self.channel_attention(x).view(b, c, *([1] * (x.ndim - 2)))
        x_channel = x * channel_weights

        # Spatial attention
        avg_pool = torch.mean(x_channel, dim=1, keepdim=True)
        max_pool, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_weights = self.spatial_attention(spatial_input)

        # Apply both attentions
        attended_features = x_channel * spatial_weights

        return attended_features, spatial_weights


class ScaleSpecificProcessor(nn.Module):
    """
    Process features at different scales with appropriate strategies
    """
    def __init__(self, channels: int, conv_op: type, scale_type: str, output_size: tuple):
        super().__init__()
        self.scale_type = scale_type
        self.conv_op = conv_op

        # Spatial attention
        self.attention = SpatialAttentionModule(channels, conv_op)

        # Scale-specific processing
        if scale_type == "early":  # Focus on texture
            self.feature_enhancer = nn.Sequential(
                conv_op(channels, channels, kernel_size=3, padding=1, groups=channels//4),
                nn.BatchNorm3d(channels) if conv_op == nn.Conv3d else nn.BatchNorm2d(channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        elif scale_type == "middle":  # Focus on structure
            self.feature_enhancer = nn.Sequential(
                conv_op(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(channels) if conv_op == nn.Conv3d else nn.BatchNorm2d(channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:  # "late" - Focus on global context
            self.feature_enhancer = nn.Identity()

        # Adaptive pooling - not to 1x1x1 but small spatial size
        if conv_op == nn.Conv3d:
            self.adaptive_pool_avg = nn.AdaptiveAvgPool3d(output_size)
            self.adaptive_pool_max = nn.AdaptiveMaxPool3d(output_size)
        else:
            self.adaptive_pool_avg = nn.AdaptiveAvgPool2d(output_size)
            self.adaptive_pool_max = nn.AdaptiveMaxPool2d(output_size)

    def forward(self, x):
        # Apply attention
        attended_features, attention_map = self.attention(x)

        # Scale-specific enhancement
        enhanced_features = self.feature_enhancer(attended_features)

        # Dual pooling to preserve different types of information
        avg_pooled = self.adaptive_pool_avg(enhanced_features)
        max_pooled = self.adaptive_pool_max(enhanced_features)

        # Flatten and concatenate
        avg_flat = avg_pooled.flatten(1)
        max_flat = max_pooled.flatten(1)

        combined = torch.cat([avg_flat, max_flat], dim=1)

        return combined, attention_map


class LatentRepresentationLayer(nn.Module):
    """
    Latent representation layer to increase expressiveness of the encoder output
    """
    def __init__(self, input_channels: int, latent_dim: int, conv_op: type):
        super().__init__()
        self.conv_op = conv_op

        # 1x1 convolution to project to latent space
        self.latent_projection = conv_op(input_channels, latent_dim, kernel_size=1, bias=True)

        # Normalization and activation
        if conv_op == nn.Conv3d:
            self.norm = nn.BatchNorm3d(latent_dim)
        elif conv_op == nn.Conv2d:
            self.norm = nn.BatchNorm2d(latent_dim)
        else:
            self.norm = nn.BatchNorm1d(latent_dim)

        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        """
        Args:
            x: Input tensor from encoder [B, C, H, W, D] or [B, C, H, W]
        Returns:
            Latent representation with same spatial dimensions but latent_dim channels
        """
        x = self.latent_projection(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class MLPClassificationHead(nn.Module):
    """
    Simple but effective MLP classification head with global pooling
    """
    def __init__(self, input_channels: int, num_classes: int, hidden_dims: List[int],
                 dropout_rate: float, conv_op: type):
        super().__init__()
        self.conv_op = conv_op

        # Global average pooling
        if conv_op == nn.Conv3d:
            self.global_pool = nn.AdaptiveAvgPool3d(1)
        elif conv_op == nn.Conv2d:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Build MLP layers
        layers = []
        prev_dim = input_channels

        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        # Final classification layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize MLP weights with proper scaling"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, H, W, D] or [B, C, H, W]
        Returns:
            Classification logits [B, num_classes]
        """
        # Global pooling
        x = self.global_pool(x)  # [B, C, 1, 1, 1] or [B, C, 1, 1]
        x = x.flatten(1)  # [B, C]

        # MLP forward
        return self.mlp(x)


class EnhancedClassificationHead(nn.Module):
    """
    Enhanced classification head with proper regularization (Legacy - for spatial attention)
    """
    def __init__(self, total_features: int, num_classes: int = 3, dropout_rate: float = 0.3):
        super().__init__()

        # Progressive dimensionality reduction with residual connections
        self.input_projection = nn.Linear(total_features, 256)

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout_rate * 0.7)
            ),
            nn.Sequential(
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout_rate * 0.5)
            )
        ])

        # Final classification layer
        self.classifier = nn.Linear(64, num_classes)

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_projection(x)

        for layer in self.layers:
            x = layer(x)

        return self.classifier(x)
