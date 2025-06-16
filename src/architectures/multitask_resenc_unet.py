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
        elif head_type == 'latent_spatial':
            self._build_latent_spatial_classification_decoder()
        else:
            raise ValueError(f"Unknown head_type: {head_type}. Must be 'mlp', 'spatial_attention', or 'latent_spatial'")

    def _build_mlp_classification_decoder(self):
        """Build MLP-based classification decoder with latent layer"""
        # Get the last encoder stage output channels
        last_stage_channels = self.encoder.stages[-1].output_channels

        # Check if we have new-style configuration with latent_layer and classification_head configs
        if 'latent_layer' in self.classification_config and 'classification_head' in self.classification_config:
            # New configuration style - use config-based approach
            latent_config = self.classification_config['latent_layer']
            cls_head_config = self.classification_config['classification_head']

            # Build latent layer with config
            self.latent_layer = LatentRepresentationLayer(
                input_channels=last_stage_channels,
                config=latent_config,
                conv_op=self.conv_op
            )

            # Build classification head with config
            self.classification_head = MLPClassificationHead(
                input_channels=last_stage_channels,  # Input channels remain the same due to residual connection
                config=cls_head_config,
                conv_op=self.conv_op
            )

            print(f"✓ Built enhanced MLP classification head with multi-layer latent representation")
        else:
            # Legacy configuration style - backward compatibility
            latent_dim = self.classification_config.get('latent_dim', 1024)

            # Create legacy config for latent layer
            legacy_latent_config = {
                'compression_channels': latent_dim // 2,
                'bottleneck_reduction': 2,
                'use_bottleneck': False,
                'dropout_rate': 0.1,
                'activation': 'torch.nn.LeakyReLU'
            }

            self.latent_layer = LatentRepresentationLayer(
                input_channels=last_stage_channels,
                config=legacy_latent_config,
                conv_op=self.conv_op
            )

            # Build MLP classification head (legacy)
            self.classification_head = MLPClassificationHead(
                input_channels=last_stage_channels,
                num_classes=self.classification_config['num_classes'],
                hidden_dims=self.classification_config.get('mlp_hidden_dims', [512, 256]),
                dropout_rate=self.classification_config.get('dropout_rate', 0.3),
                conv_op=self.conv_op
            )

            print(f"✓ Built legacy MLP classification head with latent dim: {latent_dim}")

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

    def _build_latent_spatial_classification_decoder(self):
        """Build latent spatial attention-based classification decoder"""
        # Get the last encoder stage output channels
        last_stage_channels = self.encoder.stages[-1].output_channels

        # Build latent layer first (same as MLP approach)
        if 'latent_layer' in self.classification_config and 'classification_head' in self.classification_config:
            # New configuration style
            latent_config = self.classification_config['latent_layer']
            cls_head_config = self.classification_config['classification_head']

            # Build latent layer with config
            self.latent_layer = LatentRepresentationLayer(
                input_channels=last_stage_channels,
                config=latent_config,
                conv_op=self.conv_op
            )

            # Build latent spatial attention head
            self.classification_head = LatentSpatialAttentionHead(
                input_channels=last_stage_channels,  # Input channels remain the same due to residual connection
                config=cls_head_config,
                conv_op=self.conv_op
            )

            print(f"✓ Built latent spatial attention classification head with multi-layer latent representation")
        else:
            # Legacy configuration style - backward compatibility
            latent_dim = self.classification_config.get('latent_dim', 1024)

            # Create legacy config for latent layer
            legacy_latent_config = {
                'compression_channels': latent_dim // 2,
                'bottleneck_reduction': 2,
                'use_bottleneck': False,
                'dropout_rate': 0.1,
                'activation': 'torch.nn.LeakyReLU'
            }

            self.latent_layer = LatentRepresentationLayer(
                input_channels=last_stage_channels,
                config=legacy_latent_config,
                conv_op=self.conv_op
            )

            # Build latent spatial attention head (legacy)
            legacy_cls_config = {
                'num_classes': self.classification_config['num_classes'],
                'hidden_dims': self.classification_config.get('mlp_hidden_dims', [512, 256]),
                'dropout_rate': self.classification_config.get('dropout_rate', 0.3),
                'attention_config': {
                    'reduction_ratio': 16,
                    'spatial_kernel_size': 7
                }
            }

            self.classification_head = LatentSpatialAttentionHead(
                input_channels=last_stage_channels,
                config=legacy_cls_config,
                conv_op=self.conv_op
            )

            print(f"✓ Built legacy latent spatial attention classification head with latent dim: {latent_dim}")

    def forward_classification_part(self, encoder_outputs):
        """
        Forward pass for classification - handles both MLP and spatial attention
        """
        head_type = self.classification_config.get('head_type', 'mlp')

        if head_type == 'mlp':
            return self._forward_mlp_classification(encoder_outputs)
        elif head_type == 'spatial_attention':
            return self._forward_spatial_attention_classification(encoder_outputs)
        elif head_type == 'latent_spatial':
            return self._forward_latent_spatial_classification(encoder_outputs)
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

    def _forward_latent_spatial_classification(self, encoder_outputs):
        """Forward pass for latent spatial attention classification head"""
        # Use the last encoder stage output
        last_encoder_features = encoder_outputs[-1]  # [B, C, H, W, D]

        # Apply latent layer first
        latent_features = self.latent_layer(last_encoder_features)  # [B, latent_dim, H, W, D]

        # Apply latent spatial attention head (includes spatial attention + pooling + MLP)
        classification_output = self.classification_head(latent_features)

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
            # Also unfreeze latent layer if using MLP head
            if hasattr(self, 'latent_layer'):
                for param in self.latent_layer.parameters():
                    param.requires_grad = True
            if hasattr(self, 'global_pools'):
                for param in self.global_pools.parameters():
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
            # Also unfreeze latent layer if using MLP head
            if hasattr(self, 'latent_layer'):
                for param in self.latent_layer.parameters():
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
    Multi-layer latent representation with compression and bottleneck design
    Based on paper implementation with multiple convolution layers
    """
    def __init__(self, input_channels: int, config: dict, conv_op: type):
        super().__init__()
        self.conv_op = conv_op
        self.config = config

        # Extract configuration parameters
        compression_channels = config.get('compression_channels', input_channels // 2)
        bottleneck_reduction = config.get('bottleneck_reduction', 2)
        use_bottleneck = config.get('use_bottleneck', True)
        dropout_rate = config.get('dropout_rate', 0.1)

        # Determine normalization layer
        if conv_op == nn.Conv3d:
            norm_layer = nn.InstanceNorm3d
        elif conv_op == nn.Conv2d:
            norm_layer = nn.InstanceNorm2d
        else:
            norm_layer = nn.InstanceNorm1d

        # Get activation function
        activation_class = self._get_activation_class(config.get('activation', 'torch.nn.LeakyReLU'))

        # Build compression pathway: input → compress → bottleneck → expand → output
        layers = []

        # 1. Initial compression layer
        layers.extend([
            conv_op(input_channels, compression_channels, kernel_size=3, padding=1, bias=True),
            norm_layer(compression_channels, eps=1e-5, affine=True),
            activation_class(0.2, inplace=True),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        ])

        # 2. Bottleneck layer (if enabled)
        if use_bottleneck:
            bottleneck_channels = compression_channels // bottleneck_reduction
            layers.extend([
                conv_op(compression_channels, bottleneck_channels, kernel_size=1, bias=True),
                norm_layer(bottleneck_channels, eps=1e-5, affine=True),
                activation_class(0.2, inplace=True),
                conv_op(bottleneck_channels, compression_channels, kernel_size=1, bias=True),
                norm_layer(compression_channels, eps=1e-5, affine=True),
                activation_class(0.2, inplace=True)
            ])

        # 3. Expansion back to input channels
        layers.extend([
            conv_op(compression_channels, input_channels, kernel_size=3, padding=1, bias=True),
            norm_layer(input_channels, eps=1e-5, affine=True),
            activation_class(0.2, inplace=True)
        ])

        self.latent_layers = nn.Sequential(*[layer for layer in layers if not isinstance(layer, nn.Identity)])

        # Residual connection for better gradient flow
        self.use_residual = True

        # Initialize weights
        self._initialize_weights()

    def _get_activation_class(self, activation_str):
        """Get activation class from string"""
        if 'LeakyReLU' in activation_str:
            return nn.LeakyReLU
        elif 'ReLU' in activation_str:
            return nn.ReLU
        else:
            return nn.LeakyReLU  # Default

    def _initialize_weights(self):
        """Initialize weights for latent layers"""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: Input tensor from encoder [B, C, H, W, D] or [B, C, H, W]
        Returns:
            Enhanced latent representation with same spatial dimensions and channels
        """
        identity = x

        # Apply latent transformation
        out = self.latent_layers(x)

        # Add residual connection
        if self.use_residual:
            out = out + identity

        return out


class MLPClassificationHead(nn.Module):
    """
    Enhanced MLP classification head with conv upsampling + FC layers
    Supports both config-based and legacy parameter-based initialization
    """
    def __init__(self, input_channels: int, conv_op: type, config: dict = None,
                 num_classes: int = None, hidden_dims: List[int] = None, dropout_rate: float = None):
        super().__init__()
        self.conv_op = conv_op

        # Handle both new config-based and legacy parameter-based initialization
        if config is not None:
            # New config-based approach
            self._build_from_config(input_channels, config)
        else:
            # Legacy approach - for backward compatibility
            if num_classes is None or hidden_dims is None or dropout_rate is None:
                raise ValueError("For legacy initialization, num_classes, hidden_dims, and dropout_rate must be provided")
            self._build_legacy(input_channels, num_classes, hidden_dims, dropout_rate)

        # Initialize weights
        self._initialize_weights()

    def _build_from_config(self, input_channels: int, config: dict):
        """Build classification head from configuration"""
        self.mlp_only = config.get('mlp_only', False)
        self.num_classes = config['num_classes']
        self.dropout_rate = config.get('dropout_rate', 0.2)

        # Get configuration sections
        initial_conv_config = config.get('initial_conv_config', {})
        conv_layers_config = config.get('conv_layers', [])
        hidden_dims = config.get('hidden_dims', [64, 32])

        if self.mlp_only:
            # Skip convolution layers, directly use MLP
            self.conv_layers = nn.Identity()

            # Global average pooling
            if self.conv_op == nn.Conv3d:
                self.global_pool = nn.AdaptiveAvgPool3d(1)
            elif self.conv_op == nn.Conv2d:
                self.global_pool = nn.AdaptiveAvgPool2d(1)
            else:
                self.global_pool = nn.AdaptiveAvgPool1d(1)

            # Build MLP layers directly from input channels
            mlp_layers = []
            prev_dim = input_channels

            for hidden_dim in hidden_dims:
                mlp_layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(self.dropout_rate)
                ])
                prev_dim = hidden_dim

            # Final classification layer
            mlp_layers.append(nn.Linear(prev_dim, self.num_classes))
            self.mlp = nn.Sequential(*mlp_layers)

        else:
            # Use convolution layers as before
            # Determine normalization layer
            if self.conv_op == nn.Conv3d:
                norm_layer = nn.InstanceNorm3d
            elif self.conv_op == nn.Conv2d:
                norm_layer = nn.InstanceNorm2d
            else:
                norm_layer = nn.InstanceNorm1d

            # Get activation function
            activation_class = self._get_activation_class(initial_conv_config.get('activation', 'torch.nn.LeakyReLU'))

            # Build convolutional layers
            conv_layers = []
            current_channels = input_channels

            # 1. Initial convolution (channel expansion)
            if initial_conv_config:
                output_channels = initial_conv_config.get('output_channels', current_channels * 2)
                kernel_size = initial_conv_config.get('kernel_size', [3, 3, 3] if self.conv_op == nn.Conv3d else [3, 3])
                stride = initial_conv_config.get('stride', [1, 1, 1] if self.conv_op == nn.Conv3d else [1, 1])
                padding = initial_conv_config.get('padding', [1, 1, 1] if self.conv_op == nn.Conv3d else [1, 1])

                conv_layers.extend([
                    self.conv_op(current_channels, output_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=True),
                    norm_layer(output_channels, eps=1e-5, affine=True) if initial_conv_config.get('use_batch_norm', True) else nn.Identity(),
                    activation_class(0.2, inplace=True)
                ])
                current_channels = output_channels

            # 2. Additional conv layers (spatial reduction)
            for layer_config in conv_layers_config:
                out_channels = layer_config.get('out_channels', current_channels // 2)
                kernel_size = layer_config.get('kernel_size', [3, 3, 3] if self.conv_op == nn.Conv3d else [3, 3])
                stride = layer_config.get('stride', [2, 2, 2] if self.conv_op == nn.Conv3d else [2, 2])
                padding = layer_config.get('padding', [1, 1, 1] if self.conv_op == nn.Conv3d else [1, 1])
                layer_dropout = layer_config.get('dropout_rate', 0.0)

                conv_layers.extend([
                    self.conv_op(current_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=True),
                    norm_layer(out_channels, eps=1e-5, affine=True) if layer_config.get('use_batch_norm', True) else nn.Identity(),
                    activation_class(0.2, inplace=True),
                    nn.Dropout(layer_dropout) if layer_dropout > 0 else nn.Identity()
                ])
                current_channels = out_channels

            # Remove Identity layers
            self.conv_layers = nn.Sequential(*[layer for layer in conv_layers if not isinstance(layer, nn.Identity)])

            # 3. Global pooling
            pooling_type = config.get('global_pooling', 'adaptive_avg')
            if pooling_type == 'adaptive_avg':
                if self.conv_op == nn.Conv3d:
                    self.global_pool = nn.AdaptiveAvgPool3d(1)
                elif self.conv_op == nn.Conv2d:
                    self.global_pool = nn.AdaptiveAvgPool2d(1)
                else:
                    self.global_pool = nn.AdaptiveAvgPool1d(1)
            else:
                # Add other pooling types if needed
                if self.conv_op == nn.Conv3d:
                    self.global_pool = nn.AdaptiveAvgPool3d(1)
                elif self.conv_op == nn.Conv2d:
                    self.global_pool = nn.AdaptiveAvgPool2d(1)
                else:
                    self.global_pool = nn.AdaptiveAvgPool1d(1)

            # 4. Build MLP layers
            mlp_layers = []
            prev_dim = current_channels

            for hidden_dim in hidden_dims:
                mlp_layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(self.dropout_rate)
                ])
                prev_dim = hidden_dim

            # Final classification layer
            mlp_layers.append(nn.Linear(prev_dim, self.num_classes))
            self.mlp = nn.Sequential(*mlp_layers)

    def _build_legacy(self, input_channels: int, num_classes: int, hidden_dims: List[int], dropout_rate: float):
        """Build classification head using legacy parameters"""
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # No conv layers in legacy mode - just global pooling + MLP
        self.conv_layers = nn.Identity()

        # Global average pooling
        if self.conv_op == nn.Conv3d:
            self.global_pool = nn.AdaptiveAvgPool3d(1)
        elif self.conv_op == nn.Conv2d:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Build MLP layers
        mlp_layers = []
        prev_dim = input_channels

        for hidden_dim in hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        # Final classification layer
        mlp_layers.append(nn.Linear(prev_dim, num_classes))

        self.mlp = nn.Sequential(*mlp_layers)

    def _get_activation_class(self, activation_str):
        """Get activation class from string"""
        if 'LeakyReLU' in activation_str:
            return nn.LeakyReLU
        elif 'ReLU' in activation_str:
            return nn.ReLU
        else:
            return nn.LeakyReLU  # Default

    def _initialize_weights(self):
        """Initialize weights for all layers"""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d, nn.BatchNorm1d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, H, W, D] or [B, C, H, W]
        Returns:
            Classification logits [B, num_classes]
        """
        # Apply convolutional layers (Identity when mlp_only=True)
        x = self.conv_layers(x)

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


class LatentSpatialAttentionHead(nn.Module):
    """
    Spatial attention applied to latent representation features
    Combines the benefits of latent processing with spatial attention mechanisms
    """
    def __init__(self, input_channels: int, conv_op: type, config: dict):
        super().__init__()
        self.conv_op = conv_op
        self.config = config

        # Extract configuration
        self.num_classes = config['num_classes']
        self.dropout_rate = config.get('dropout_rate', 0.2)
        hidden_dims = config.get('hidden_dims', [256, 128])

        # Attention configuration
        attention_config = config.get('attention_config', {})
        self.reduction_ratio = attention_config.get('reduction_ratio', 16)
        spatial_kernel_size = attention_config.get('spatial_kernel_size', 7)

        # Spatial attention module for latent features
        self.spatial_attention = SpatialAttentionModule(
            channels=input_channels,
            conv_op=conv_op,
            reduction_ratio=self.reduction_ratio
        )

        # Attention-weighted pooling
        if conv_op == nn.Conv3d:
            self.global_pool_avg = nn.AdaptiveAvgPool3d(1)
            self.global_pool_max = nn.AdaptiveMaxPool3d(1)
        elif conv_op == nn.Conv2d:
            self.global_pool_avg = nn.AdaptiveAvgPool2d(1)
            self.global_pool_max = nn.AdaptiveMaxPool2d(1)
        else:
            self.global_pool_avg = nn.AdaptiveAvgPool1d(1)
            self.global_pool_max = nn.AdaptiveMaxPool1d(1)

        # MLP classification head
        mlp_layers = []
        # Input features: input_channels * 2 (avg + max pooling)
        prev_dim = input_channels * 2

        for hidden_dim in hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim

        # Final classification layer
        mlp_layers.append(nn.Linear(prev_dim, self.num_classes))
        self.mlp = nn.Sequential(*mlp_layers)

        # Store attention maps for visualization
        self.attention_maps = []

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for all layers"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: Latent features [B, C, H, W, D] or [B, C, H, W]
        Returns:
            Classification logits [B, num_classes]
        """
        # Apply spatial attention to latent features
        attended_features, attention_map = self.spatial_attention(x)

        # Store attention map for visualization
        self.attention_maps = [attention_map]

        # Attention-weighted pooling (both avg and max)
        avg_pooled = self.global_pool_avg(attended_features)  # [B, C, 1, 1, 1] or [B, C, 1, 1]
        max_pooled = self.global_pool_max(attended_features)  # [B, C, 1, 1, 1] or [B, C, 1, 1]

        # Flatten and concatenate
        avg_flat = avg_pooled.flatten(1)  # [B, C]
        max_flat = max_pooled.flatten(1)  # [B, C]

        # Combine avg and max pooled features
        combined_features = torch.cat([avg_flat, max_flat], dim=1)  # [B, 2*C]

        # Apply MLP for classification
        classification_output = self.mlp(combined_features)

        return classification_output
