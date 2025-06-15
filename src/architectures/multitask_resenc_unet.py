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
                'num_classes': 3,
                'dropout_rate': 0.2,
                'hidden_dims': [256, 128],
                'use_all_features': True
            }
        self.classification_config = classification_config

        # Build classification decoder
        self._build_classification_decoder()

        # Training stage control
        self.training_stage = "full"  # full, enc_seg, enc_cls, joint_finetune
        self._apply_training_stage()

        # Uncertainty weighting parameters (learnable)
        self.log_var_seg = nn.Parameter(torch.zeros(1))
        self.log_var_cls = nn.Parameter(torch.zeros(1))

    def _build_classification_decoder(self):
        """Build multi-scale classification decoder"""
        conv_dim = convert_conv_op_to_dim(self.conv_op)

        if self.classification_config['use_all_features']:
            # Use features from stem + all encoder stages
            # Stem output channels + encoder stage output channels
            feature_channels = [self.encoder.stem.output_channels]  # Stem output
            for stage in self.encoder.stages:
                feature_channels.append(stage.output_channels)
        else:
            # Use only deepest features (last encoder stage)
            feature_channels = [self.encoder.stages[-1].output_channels]

        # Global pooling layers for each feature level
        if conv_dim == 2:
            self.global_pools = nn.ModuleList([
                nn.AdaptiveAvgPool2d(1) for _ in feature_channels
            ])
        else:  # 3D
            self.global_pools = nn.ModuleList([
                nn.AdaptiveAvgPool3d(1) for _ in feature_channels
            ])

        # Calculate total feature dimension
        total_features = sum(feature_channels)

        # Classification head
        hidden_dims = self.classification_config['hidden_dims']
        dropout_rate = self.classification_config['dropout_rate']
        num_classes = self.classification_config['num_classes']

        layers = []
        input_dim = total_features

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim

        # Final classification layer
        layers.append(nn.Linear(input_dim, num_classes))

        self.classification_head = nn.Sequential(*layers)

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
        if self.classification_config['use_all_features']:
            # Extract features from all encoder levels (including stem output)
            cls_features = []
            for i, feature_map in enumerate(encoder_outputs):
                pooled = self.global_pools[i](feature_map)
                cls_features.append(pooled.flatten(1))
            cls_x = torch.cat(cls_features, dim=1)
        else:
            # Use only deepest features
            cls_x = self.global_pools[0](encoder_outputs[-1])
            cls_x = cls_x.flatten(1)

        classification_output = self.classification_head(cls_x)

        return {
            'segmentation': segmentation_output,
            'classification': classification_output,
            'uncertainty_weights': {
                'seg': torch.exp(-self.log_var_seg),
                'cls': torch.exp(-self.log_var_cls)
            }
        }

    def set_training_stage(self, stage: str):
        """Set training stage and freeze/unfreeze parameters accordingly"""
        valid_stages = ['full', 'enc_seg', 'enc_cls', 'joint_finetune']
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage: {stage}. Must be one of {valid_stages}")

        self.training_stage = stage
        self._apply_training_stage()

    def _apply_training_stage(self):
        """Apply parameter freezing based on training stage"""
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
            # Keep uncertainty weights trainable
            self.log_var_seg.requires_grad = True

        elif self.training_stage == "enc_cls":
            # Unfreeze encoder and classification head
            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.classification_head.parameters():
                param.requires_grad = True
            for param in self.global_pools.parameters():
                param.requires_grad = True
            # Keep uncertainty weights trainable
            self.log_var_cls.requires_grad = True

        elif self.training_stage == "joint_finetune":
            # Unfreeze last two encoder stages and both decoders
            for stage_idx in [-2, -1]:  # Last two stages
                for param in self.encoder.stages[stage_idx].parameters():
                    param.requires_grad = True

            # Unfreeze both decoders
            for param in self.decoder.parameters():
                param.requires_grad = True
            for param in self.classification_head.parameters():
                param.requires_grad = True
            for param in self.global_pools.parameters():
                param.requires_grad = True

            # Keep uncertainty weights trainable
            self.log_var_seg.requires_grad = True
            self.log_var_cls.requires_grad = True

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
            'uncertainty_weights': {
                'seg_weight': torch.exp(-self.log_var_seg).item(),
                'cls_weight': torch.exp(-self.log_var_cls).item()
            }
        }

        return info

    def compute_multitask_loss(self, outputs, targets, seg_loss_fn, cls_loss_fn):
        """Compute uncertainty-weighted multitask loss"""
        seg_pred = outputs['segmentation']
        cls_pred = outputs['classification']
        seg_target = targets['segmentation']
        cls_target = targets['classification']

        # Compute individual losses
        seg_loss = seg_loss_fn(seg_pred, seg_target)
        cls_loss = cls_loss_fn(cls_pred, cls_target)

        # stage specific loss handling
        if self.training_stage == "enc_seg":
            # Only segmentation loss, no uncertainty weighting
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
            # Uncertainty weighting
            seg_weight = torch.exp(-self.log_var_seg)
            cls_weight = torch.exp(-self.log_var_cls)

            # Weighted loss with regularization terms
            total_loss = seg_weight * seg_loss + self.log_var_seg + cls_weight * cls_loss + self.log_var_cls

            # L2 regularization on uncertainty weights
            log_var_reg = 0.01 * (self.log_var_seg.pow(2) + self.log_var_cls.pow(2))
            total_loss += log_var_reg

            return {
                'total_loss': total_loss,
                'segmentation_loss': seg_loss,
                'classification_loss': cls_loss,
                'seg_weight': seg_weight.item(),
                'cls_weight': cls_weight.item()
            }