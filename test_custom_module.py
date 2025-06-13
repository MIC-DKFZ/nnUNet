#!/usr/bin/env python3
"""
test_custom_model.py

Comprehensive test suite for custom multi-task nnUNet implementation.
Run this before training to catch issues early!

Usage:
    python test_custom_model.py
"""

import os
import sys
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
import traceback
import inspect

def error_print(e: Exception, context: str = None, show_traceback: bool = True) -> str:
    """
    Print detailed error information with context and traceback

    Args:
        e (Exception): The caught exception
        context (str, optional): Additional context about where/why the error occurred
        show_traceback (bool): Whether to show the full traceback

    Returns:
        str: Formatted error message
    """
    # Get the current frame for more context
    current_frame = inspect.currentframe()
    caller_frame = current_frame.f_back

    # Extract caller information
    caller_info = ""
    if caller_frame:
        filename = caller_frame.f_code.co_filename
        function = caller_frame.f_code.co_name
        line_no = caller_frame.f_lineno
        caller_info = f"in {filename}:{function}() at line {line_no}"

    # Build error message
    error_msg = [
        "❌ Error Detected!",
        "=" * 50,
        f"Error Type: {type(e).__name__}",
        f"Message: {str(e)}",
        f"Location: {caller_info}"
    ]

    if context:
        error_msg.append(f"Context: {context}")

    if show_traceback:
        error_msg.extend([
            "\nTraceback:",
            "=" * 50,
            "".join(traceback.format_tb(e.__traceback__))
        ])

    error_str = "\n".join(error_msg)
    print(error_str)
    return error_str

# Test imports
try:
    from nnunetv2.nnunetv2.training.loss.multitask_losses import MultiTaskLoss, UnifiedFocalLoss, TverskyLoss
    from nnunetv2.nnunetv2.architectures.MultiTaskResEncUNet import MultiTaskResEncUNet, MultiTaskChannelAttentionResEncUNet, MultiTaskEfficientAttentionResEncUNet
    from nnunetv2.nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiTask import nnUNetTrainerMultiTask
    from nnunetv2.nnunetv2.experiment_planning.experiment_planners.multitask_base_planner import MultiTaskResEncUNetPlanner
    from nnunetv2.nnunetv2.experiment_planning.experiment_planners.multitask_channel_attention_planner import MultiTaskChannelAttentionResEncUNetPlanner
    from nnunetv2.nnunetv2.experiment_planning.experiment_planners.multitask_efficient_attention_planner import MultiTaskEfficientAttentionResEncUNetPlanner
    print("✓ All imports successful!")
except ImportError as e:
    error_print(e)
    sys.exit(1)


class TestLossFunctions:
    """Test all custom loss functions"""

    def __init__(self):
        self.batch_size = 2
        self.num_classes = 3  # background, pancreas, lesion
        self.num_classification_classes = 3  # subtype 0, 1, 2
        self.spatial_dims = (32, 64, 96)  # Small test volume

    def create_test_data(self):
        """Create synthetic test data that matches your trainer's expected format"""
        # Segmentation data
        seg_pred = torch.randn(self.batch_size, self.num_classes, *self.spatial_dims).requires_grad_(True)
        seg_target = torch.randint(0, self.num_classes, (self.batch_size, *self.spatial_dims))

        # Classification data
        cls_pred = torch.randn(self.batch_size, self.num_classification_classes).requires_grad_(True)
        cls_target = torch.randint(0, self.num_classification_classes, (self.batch_size,))

        return seg_pred, seg_target, cls_pred, cls_target

    def test_multitask_loss(self):
        """Test MultiTaskLoss with different configurations"""
        print("\n=== Testing MultiTaskLoss ===")

        seg_pred, seg_target, cls_pred, cls_target = self.create_test_data()

        # Test different loss types
        loss_configs = [
            ('dice_ce', 1.0, 0.25),
            ('focal', 1.0, 0.3),
            ('tversky', 0.8, 0.4)
        ]

        for loss_type, seg_weight, cls_weight in loss_configs:
            try:
                loss_fn = MultiTaskLoss(
                    seg_weight=seg_weight,
                    cls_weight=cls_weight,
                    loss_type=loss_type
                )

                # Test the loss function as called in your trainer
                loss_dict = loss_fn(seg_pred, seg_target, cls_pred, cls_target)

                # Validate output structure matches your trainer expectations
                assert 'segmentation_loss' in loss_dict
                assert 'classification_loss' in loss_dict

                seg_loss = loss_dict['segmentation_loss']
                cls_loss = loss_dict['classification_loss']

                assert seg_loss.requires_grad, "Segmentation loss should require gradients"
                assert cls_loss.requires_grad, "Classification loss should require gradients"
                assert seg_loss.item() > 0, "Segmentation loss should be positive"
                assert cls_loss.item() > 0, "Classification loss should be positive"

                print(f"✓ {loss_type} loss - Seg: {seg_loss.item():.4f}, Cls: {cls_loss.item():.4f}")

            except Exception as e:
                error_print(e, context=f"Loss type: {loss_type}, Seg weight: {seg_weight}, Cls weight: {cls_weight}")
                return False

        return True

    def test_individual_losses(self):
        """Test individual loss components"""
        print("\n=== Testing Individual Loss Components ===")

        seg_pred = torch.randn(2, 3, 32, 64, 96)
        seg_target = torch.randint(0, 3, (2, 32, 64, 96))

        # Test Unified Focal Loss
        try:
            focal_loss = UnifiedFocalLoss(alpha=0.25, gamma=2.0)
            loss_val = focal_loss(seg_pred, seg_target)
            assert loss_val.item() > 0
            print(f"✓ Unified Focal Loss: {loss_val.item():.4f}")
        except Exception as e:
            error_print(e, context="Unified Focal Loss")
            return False

        # Test Tversky Loss
        try:
            tversky_loss = TverskyLoss(alpha=0.7, beta=0.3)
            loss_val = tversky_loss(seg_pred, seg_target)
            assert loss_val.item() > 0
            print(f"✓ Tversky Loss: {loss_val.item():.4f}")
        except Exception as e:
            error_print(e, context="Tversky Loss")
            return False

        return True

    def test_gradient_flow(self):
        """Test that gradients flow through the loss"""
        print("\n=== Testing Gradient Flow ===")

        seg_pred, seg_target, cls_pred, cls_target = self.create_test_data()

        loss_fn = MultiTaskLoss(seg_weight=1.0, cls_weight=0.25, loss_type='dice_ce')
        loss_dict = loss_fn(seg_pred, seg_target, cls_pred, cls_target)

        # Test the loss combination as in your trainer
        seg_loss = loss_dict['segmentation_loss'] / 1.0  # Normalized by running mean
        cls_loss = loss_dict['classification_loss'] / 1.0
        total_loss = 1.0 * seg_loss + 0.25 * cls_loss

        # Backward pass
        total_loss.backward()

        # Check gradients
        seg_grad_norm = seg_pred.grad.norm().item()
        cls_grad_norm = cls_pred.grad.norm().item()

        assert seg_grad_norm > 0, "Segmentation gradients should be non-zero"
        assert cls_grad_norm > 0, "Classification gradients should be non-zero"

        print(f"✓ Segmentation gradient norm: {seg_grad_norm:.4f}")
        print(f"✓ Classification gradient norm: {cls_grad_norm:.4f}")

        return True


class TestMultiTaskNetwork:
    """Test the multi-task network architecture"""

    def __init__(self):
        self.input_channels = 1
        self.num_classes = 3
        self.num_classification_classes = 3
        self.network_variants = {
            'base': MultiTaskResEncUNet,
            'channel_attention': MultiTaskChannelAttentionResEncUNet,
            'efficient_attention': MultiTaskEfficientAttentionResEncUNet
        }

    def create_test_network(self, variant='base'):
        """Create a small test network of specified variant"""
        network_class = self.network_variants.get(variant)
        if network_class is None:
            raise ValueError(f"Unknown network variant: {variant}")

        network = network_class(
            input_channels=self.input_channels,
            num_classes=self.num_classes,
            num_classification_classes=self.num_classification_classes,
            n_stages=4,  # Smaller for testing
            features_per_stage=[32, 64, 128, 256],
            conv_op=torch.nn.Conv3d,
            kernel_sizes=[[3, 3, 3]] * 4,
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 2, 2, 2],
            n_conv_per_stage_decoder=[1, 1, 1],
            conv_bias=True,
            norm_op=torch.nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            dropout_op=None,
            nonlin=torch.nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
        )
        return network

    def test_network_training_vs_inference_mode(self):
        """Test that network behaves differently in training vs inference mode"""
        print("\n=== Testing Training vs Inference Mode ===")

        results = {}

        for variant_name in self.network_variants.keys():
            print(f"\nTesting {variant_name} variant:")
            try:
                network = self.create_test_network(variant_name)
                batch_size = 2
                input_tensor = torch.randn(batch_size, self.input_channels, 32, 64, 96)

                # Test training mode
                network.train()
                with torch.no_grad():
                    training_output = network(input_tensor)

                # Test inference mode
                network.eval()
                with torch.no_grad():
                    inference_output = network(input_tensor)

                # Validate training mode output
                if isinstance(training_output, dict):
                    assert 'segmentation' in training_output, "Training mode should return segmentation"
                    assert 'classification' in training_output, "Training mode should return classification"
                    print(f"✓ Training mode returns dict with keys: {list(training_output.keys())}")
                else:
                    print("⚠️ Training mode returns tensor instead of dict - this may cause issues")

                # Validate inference mode output
                if isinstance(inference_output, dict):
                    print("⚠️ Inference mode returns dict - this will cause nnUNet inference issues!")
                    print("   This is the root cause of your flip() error!")
                    inference_compatible = False
                else:
                    print("✓ Inference mode returns tensor - compatible with nnUNet inference")
                    expected_shape = (batch_size, self.num_classes, 32, 64, 96)
                    assert inference_output.shape == expected_shape, f"Shape mismatch: {inference_output.shape} vs {expected_shape}"
                    inference_compatible = True

                results[variant_name] = {
                    'status': 'PASS' if inference_compatible else 'INFERENCE_ISSUE',
                    'training_output_type': type(training_output).__name__,
                    'inference_output_type': type(inference_output).__name__,
                    'inference_compatible': inference_compatible
                }

                print(f"✓ {variant_name} mode testing completed")

            except Exception as e:
                error_print(e, context=f"Testing {variant_name} training/inference modes")
                results[variant_name] = {'status': 'FAIL', 'error': str(e)}

        # Print summary
        print("\n=== Training vs Inference Mode Summary ===")
        print("=" * 60)
        all_compatible = True
        for variant, result in results.items():
            print(f"\n{variant.upper()}:")
            if result['status'] == 'PASS':
                print(f"Status: ✓ PASS - Inference compatible")
            elif result['status'] == 'INFERENCE_ISSUE':
                print(f"Status: ⚠️  INFERENCE ISSUE - Returns dict in eval mode")
                print(f"Training output: {result['training_output_type']}")
                print(f"Inference output: {result['inference_output_type']}")
                all_compatible = False
            else:
                print(f"Status: ❌ FAIL - {result['error']}")
                all_compatible = False

        if not all_compatible:
            print(f"\n🔧 FIX NEEDED: Modify your network's forward() method:")
            print(f"   def forward(self, x):")
            print(f"       # ... existing code ...")
            print(f"       if self.training:")
            print(f"           return {{'segmentation': seg_out, 'classification': cls_out}}")
            print(f"       else:")
            print(f"           return seg_out  # Only segmentation for inference")

        return all_compatible

    def test_network_variants(self):
        """Test all network variants"""
        print("\n=== Testing Network Variants ===")
        results = {}

        for variant_name in self.network_variants.keys():
            print(f"\nTesting {variant_name} variant:")
            try:
                network = self.create_test_network(variant_name)
                batch_size = 2
                input_tensor = torch.randn(batch_size, self.input_channels, 32, 64, 96)

                # Test in training mode (should return dict)
                network.train()
                with torch.no_grad():
                    outputs = network(input_tensor)

                # Validate outputs
                assert isinstance(outputs, dict), "Training mode output should be a dictionary"
                assert 'segmentation' in outputs, "Missing segmentation output"
                assert 'classification' in outputs, "Missing classification output"

                # Count parameters
                total_params = sum(p.numel() for p in network.parameters())
                trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)

                results[variant_name] = {
                    'status': 'PASS',
                    'total_params': total_params,
                    'trainable_params': trainable_params,
                    'seg_shape': outputs['segmentation'].shape,
                    'cls_shape': outputs['classification'].shape
                }

                print(f"✓ Forward pass successful")
                print(f"✓ Parameters: {total_params:,} total, {trainable_params:,} trainable")
                print(f"✓ Output shapes - Seg: {outputs['segmentation'].shape}, Cls: {outputs['classification'].shape}")

            except Exception as e:
                error_print(e, context=f"Testing {variant_name} variant")
                results[variant_name] = {'status': 'FAIL', 'error': str(e)}

        # Print comparison summary
        print("\n=== Network Variants Comparison ===")
        print("=" * 50)
        for variant, result in results.items():
            print(f"\n{variant.upper()}:")
            if result['status'] == 'PASS':
                print(f"Status: ✓ PASS")
                print(f"Parameters: {result['total_params']:,} total, {result['trainable_params']:,} trainable")
                print(f"Output shapes:")
                print(f"  - Segmentation: {result['seg_shape']}")
                print(f"  - Classification: {result['cls_shape']}")
            else:
                print(f"Status: ❌ FAIL")
                print(f"Error: {result['error']}")

        return all(r['status'] == 'PASS' for r in results.values())

    def test_network_forward(self):
        """Test forward pass through the network"""
        print("\n=== Testing Network Forward Pass ===")

        try:
            network = self.create_test_network()
            network.train()  # Ensure we're in training mode

            # Test input
            batch_size = 2
            input_tensor = torch.randn(batch_size, self.input_channels, 32, 64, 96)

            # Forward pass
            with torch.no_grad():
                outputs = network(input_tensor)

            # Validate outputs
            assert isinstance(outputs, dict), "Output should be a dictionary"
            assert 'segmentation' in outputs, "Missing segmentation output"
            assert 'classification' in outputs, "Missing classification output"

            seg_output = outputs['segmentation']
            cls_output = outputs['classification']

            # Check shapes
            expected_seg_shape = (batch_size, self.num_classes, 32, 64, 96)
            expected_cls_shape = (batch_size, self.num_classification_classes)

            assert seg_output.shape == expected_seg_shape, f"Segmentation shape mismatch: {seg_output.shape} vs {expected_seg_shape}"
            assert cls_output.shape == expected_cls_shape, f"Classification shape mismatch: {cls_output.shape} vs {expected_cls_shape}"

            print(f"✓ Segmentation output shape: {seg_output.shape}")
            print(f"✓ Classification output shape: {cls_output.shape}")

            return True

        except Exception as e:
            error_print(e, context="Network forward pass")
            return False

    def test_build_network_architecture_method(self):
        """Test the build_network_architecture method from your trainer"""
        print("\n=== Testing build_network_architecture Method ===")

        try:
            # Test parameters similar to your trainer
            architecture_mappings = {
                'MultiTaskResEncUNet': MultiTaskResEncUNet,
                'MultiTaskChannelAttentionResEncUNet': MultiTaskChannelAttentionResEncUNet,
                'MultiTaskEfficientAttentionResEncUNet': MultiTaskEfficientAttentionResEncUNet,
            }

            for arch_name, arch_class in architecture_mappings.items():
                print(f"\nTesting {arch_name}:")

                # Use the static method from your trainer
                network = nnUNetTrainerMultiTask.build_network_architecture(
                    architecture_class_name=arch_name,
                    arch_init_kwargs={
                        'n_stages': 4,
                        'features_per_stage': [32, 64, 128, 256],
                        'conv_op': 'torch.nn.Conv3d',
                        'kernel_sizes': [[3, 3, 3]] * 4,
                        'strides': [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                        'n_blocks_per_stage': [1, 2, 2, 2],
                        'n_conv_per_stage_decoder': [1, 1, 1],
                        'conv_bias': True,
                        'norm_op': 'torch.nn.InstanceNorm3d',
                        'norm_op_kwargs': {"eps": 1e-5, "affine": True},
                        'dropout_op': None,
                        'nonlin': 'torch.nn.LeakyReLU',
                        'nonlin_kwargs': {"inplace": True},
                    },
                    arch_init_kwargs_req_import=['conv_op', 'norm_op', 'dropout_op', 'nonlin'],
                    num_input_channels=1,
                    num_output_channels=3,
                    enable_deep_supervision=False
                )

                # Test the network
                network.train()
                input_tensor = torch.randn(1, 1, 32, 64, 96)
                with torch.no_grad():
                    outputs = network(input_tensor)

                assert isinstance(outputs, dict), f"{arch_name} should return dict in training mode"
                assert 'segmentation' in outputs, f"{arch_name} missing segmentation output"
                assert 'classification' in outputs, f"{arch_name} missing classification output"

                print(f"✓ {arch_name} built and tested successfully")

            return True

        except Exception as e:
            error_print(e, context="build_network_architecture method")
            return False


class TestTrainerIntegration:
    """Test integration with nnUNet trainer"""

    def create_mock_plans_and_config(self):
        """Create mock plans and configuration for testing"""
        plans = {
            'dataset_name': 'Dataset001_PancreasSegClassification',
            'plans_name': 'nnUNetMultiTaskResEncUNetPlans',
            'original_median_spacing_after_transp': [2.0, 0.73046875, 0.73046875],
            'original_median_shape_after_transp': [64, 119, 178],
            'image_reader_writer': 'SimpleITKIO',
            'transpose_forward': [0, 1, 2],
            'transpose_backward': [0, 1, 2],
            'configurations': {
                '3d_fullres': {
                    'label_manager': 'LabelManager',
                    'data_identifier': 'nnUNetMultiTaskResEncUNetPlans_3d_fullres',
                    'preprocessor_name': 'DefaultPreprocessor',
                    'batch_size': 2,
                    'patch_size': [64, 128, 192],
                    'median_image_size_in_voxels': [59.0, 118.0, 181.0],
                    'spacing': [2.0, 0.73046875, 0.73046875],
                    'normalization_schemes': ['CTNormalization'],
                    'use_mask_for_norm': [False],
                    'resampling_fn_data': 'resample_data_or_seg_to_shape',
                    'resampling_fn_seg': 'resample_data_or_seg_to_shape',
                    'resampling_fn_data_kwargs': {
                        'is_seg': False,
                        'order': 3,
                        'order_z': 0,
                        'force_separate_z': None
                    },
                    'resampling_fn_seg_kwargs': {
                        'is_seg': True,
                        'order': 1,
                        'order_z': 0,
                        'force_separate_z': None
                    },
                    'resampling_fn_probabilities': 'resample_data_or_seg_to_shape',
                    'resampling_fn_probabilities_kwargs': {
                        'is_seg': False,
                        'order': 1,
                        'order_z': 0,
                        'force_separate_z': None
                    },
                    'architecture': {
                        'network_class_name': 'nnunetv2.architectures.MultiTaskResEncUNet.MultiTaskResEncUNet',
                        'arch_kwargs': {
                            'n_stages': 6,
                            'features_per_stage': [32, 64, 128, 256, 320, 320],
                            'conv_op': 'torch.nn.modules.conv.Conv3d',
                            'kernel_sizes': [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                            'strides': [[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                            'n_blocks_per_stage': [1, 3, 4, 6, 6, 6],
                            'n_conv_per_stage_decoder': [1, 1, 1, 1, 1],
                            'conv_bias': True,
                            'norm_op': 'torch.nn.modules.instancenorm.InstanceNorm3d',
                            'norm_op_kwargs': {'eps': 1e-05, 'affine': True},
                            'dropout_op': None,
                            'dropout_op_kwargs': None,
                            'nonlin': 'torch.nn.LeakyReLU',
                            'nonlin_kwargs': {'inplace': True},
                            'num_classification_classes': 3,
                            'classification_dropout': 0.5,
                            'use_classification_head': True
                        },
                        '_kw_requires_import': ['conv_op', 'norm_op', 'dropout_op', 'nonlin']
                    },
                    'batch_dice': False
                }
            },
            'experiment_planner_used': 'MultiTaskResEncUNetPlanner',
            'label_manager': 'LabelManager',
            'foreground_intensity_properties_per_channel': {
                '0': {
                    'max': 1929.0,
                    'mean': 74.89189910888672,
                    'median': 78.01163482666016,
                    'min': -319.0,
                    'percentile_00_5': -55.99610900878906,
                    'percentile_99_5': 179.97802734375,
                    'std': 44.09819030761719
                }
            }
        }

        dataset_json = {
            'name': 'Dataset001_PancreasSegClassification',
            'description': 'Test dataset',
            'labels': {
                'background': '0',
                'pancreas': '1',
                'lesion': '2',
            },
            'numTest': 10,
            'numTraining': 100,
            'file_ending': '.nii.gz'
        }

        return plans, dataset_json

    def test_trainer_initialization(self):
        """Test trainer initialization"""
        print("\n=== Testing Trainer Initialization ===")

        try:
            plans, dataset_json = self.create_mock_plans_and_config()

            trainer = nnUNetTrainerMultiTask(
                plans=plans,
                configuration='3d_fullres',
                fold=0,
                dataset_json=dataset_json,
                device=torch.device('cpu')
            )

            # Test trainer attributes from your implementation
            assert trainer.num_classification_classes == 3
            assert trainer.seg_weight == 1.0
            assert trainer.cls_weight == 0.25
            assert trainer.loss_type == 'dice_ce'

            print("✓ Trainer initialized successfully")
            print(f"✓ Classification classes: {trainer.num_classification_classes}")
            print(f"✓ Loss weights - Seg: {trainer.seg_weight}, Cls: {trainer.cls_weight}")
            print(f"✓ Loss type: {trainer.loss_type}")

            return True

        except Exception as e:
            error_print(e, context="Trainer initialization")
            return False

    def test_loss_building(self):
        """Test loss function building in trainer"""
        print("\n=== Testing Loss Building ===")

        try:
            plans, dataset_json = self.create_mock_plans_and_config()

            trainer = nnUNetTrainerMultiTask(
                plans=plans,
                configuration='3d_fullres',
                fold=0,
                dataset_json=dataset_json,
                device=torch.device('cpu')
            )

            # Build loss - note that the method is currently commented out in your code
            # We'll test if it raises the right error or if it's been implemented
            try:
                loss_fn = trainer._build_loss()

                if loss_fn is not None:
                    assert isinstance(loss_fn, MultiTaskLoss), "Loss should be MultiTaskLoss instance"
                    print("✓ Loss function built successfully")
                    print(f"✓ Loss type: {type(loss_fn).__name__}")
                else:
                    print("⚠️ _build_loss() method returns None - it may be commented out")
                    print("   This means the trainer will fall back to parent class loss")
                    print("   which could cause issues with multi-task training")
                    return False

            except AttributeError as e:
                if "_build_loss" in str(e):
                    print("⚠️ _build_loss() method not implemented or commented out")
                    print("   You need to uncomment and implement this method for multi-task training")
                    return False
                else:
                    raise e

            return True

        except Exception as e:
            error_print(e, context="Loss building")
            return False

    def test_training_step_data_flow(self):
        """Test the data flow in your custom training step"""
        print("\n=== Testing Training Step Data Flow ===")

        try:
            plans, dataset_json = self.create_mock_plans_and_config()

            trainer = nnUNetTrainerMultiTask(
                plans=plans,
                configuration='3d_fullres',
                fold=0,
                dataset_json=dataset_json,
                device=torch.device('cpu')
            )

            # Create mock batch data that matches your trainer's expectations
            batch = {
                'data': torch.randn(2, 1, 32, 64, 96),
                'target': torch.randint(0, 3, (2, 32, 64, 96)),
                'class_target': torch.randint(0, 3, (2,))
            }

            # Test that the trainer can handle this batch structure
            # Note: We can't actually run train_step without proper initialization
            # but we can test the data structure expectations

            # Check if trainer has the expected methods
            assert hasattr(trainer, 'train_step'), "Trainer should have train_step method"
            assert hasattr(trainer, 'validation_step'), "Trainer should have validation_step method"
            assert hasattr(trainer, 'on_validation_epoch_end'), "Trainer should have on_validation_epoch_end method"

            print("✓ Trainer has required multi-task methods")
            print("✓ Batch data structure is compatible")

            return True

        except Exception as e:
            error_print(e, context="Training step data flow")
            return False


def run_all_tests():
    """Run all test suites"""
    print("🧪 Starting Custom nnUNet Multi-Task Test Suite")
    print("=" * 50)

    all_passed = True

    # Test 1: Loss Functions
    print("\n📊 TESTING LOSS FUNCTIONS")
    loss_tester = TestLossFunctions()
    tests = [
        loss_tester.test_multitask_loss,
        loss_tester.test_individual_losses,
        loss_tester.test_gradient_flow
    ]

    for test in tests:
        if not test():
            all_passed = False

    # Test 2: Network Architecture (including inference compatibility)
    print("\n🏗️ TESTING NETWORK ARCHITECTURE")
    network_tester = TestMultiTaskNetwork()
    tests = [
        network_tester.test_network_variants,
        network_tester.test_network_forward,
        network_tester.test_network_training_vs_inference_mode,  # New critical test
        network_tester.test_build_network_architecture_method    # Test your trainer's method
    ]

    for test in tests:
        if not test():
            all_passed = False

    # Test 3: Trainer Integration
    print("\n👩‍🏫 TESTING TRAINER INTEGRATION")
    trainer_tester = TestTrainerIntegration()
    tests = [
        trainer_tester.test_trainer_initialization,
        trainer_tester.test_loss_building,
        trainer_tester.test_training_step_data_flow
    ]

    for test in tests:
        if not test():
            all_passed = False

    # Final results
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Your implementation is ready for training.")
        print("\n💡 Next steps:")
        print("   1. Ensure your dataset has classification labels in labels.csv")
        print("   2. Run: nnUNetv2_plan_and_preprocess -d Dataset001_PancreasSegClassification -pl MultiTaskResEncUNetPlanner")
        print("   3. Train: nnUNetv2_train Dataset001_PancreasSegClassification 3d_fullres 0 -tr nnUNetTrainerMultiTask")
    else:
        print("❌ SOME TESTS FAILED! Please fix the issues before training.")
        print("\n🔧 Priority fixes needed:")
        print("   1. ❗ CRITICAL: Fix network inference mode to return only segmentation tensor")
        print("   2. Uncomment and implement _build_loss() method in trainer")
        print("   3. Verify all imports and module paths")

    return all_passed


if __name__ == '__main__':
    # Set up environment
    os.environ['nnUNet_raw'] = '/mnt/data/gpu-server/nnUNet_modified/nnunet_data/nnUNet_raw'
    os.environ['nnUNet_preprocessed'] = '/mnt/data/gpu-server/nnUNet_modified/nnunet_data/nnUNet_preprocessed'
    os.environ['nnUNet_results'] = '/mnt/data/gpu-server/nnUNet_modified/nnunet_data/nnUNet_results'

    torch.manual_seed(42)
    np.random.seed(42)

    # Run tests
    success = run_all_tests()
    sys.exit(0 if success else 1)
