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
    from src.architectures.MultiTaskResEncUNet import MultiTaskResEncUNet, MultiTaskChannelAttentionResEncUNet, MultiTaskEfficientAttentionResEncUNet
    from nnunetv2.nnunetv2.training.nnUNetTrainer.nnUNetTrainerMultiTask import nnUNetTrainerMultiTask
    from src.planners.multitask_base_planner import MultiTaskResEncUNetPlanner
    from src.planners.multitask_channel_attention_planner import MultiTaskChannelAttentionResEncUNetPlanner
    from src.planners.multitask_efficient_attention_planner import MultiTaskEfficientAttentionResEncUNetPlanner
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
        """Create synthetic test data"""
        # Segmentation data``
        seg_pred = torch.randn(self.batch_size, self.num_classes, *self.spatial_dims).requires_grad_(True)
        seg_target = torch.randint(0, self.num_classes, (self.batch_size, *self.spatial_dims))

        # Classification data
        cls_pred = torch.randn(self.batch_size, self.num_classification_classes).requires_grad_(True)
        cls_target = torch.randint(0, self.num_classification_classes, (self.batch_size,))

        outputs = {
            'segmentation': seg_pred,
            'classification': cls_pred
        }

        targets = {
            'segmentation': seg_target,
            'classification': cls_target
        }

        return outputs, targets

    def test_multitask_loss(self):
        """Test MultiTaskLoss with different configurations"""
        print("\n=== Testing MultiTaskLoss ===")

        outputs, targets = self.create_test_data()

        # Test different loss types
        loss_configs = [
            ('dice_ce', 1.0, 0.5),
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

                loss_dict = loss_fn(outputs, targets)

                # Validate output
                assert 'total_loss' in loss_dict
                assert 'segmentation_loss' in loss_dict
                assert 'classification_loss' in loss_dict

                total_loss = loss_dict['total_loss']
                assert total_loss.requires_grad, "Loss should require gradients"
                assert total_loss.item() > 0, "Loss should be positive"

                print(f"✓ {loss_type} loss: {total_loss.item():.4f}")

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

        outputs, targets = self.create_test_data()
        outputs['segmentation'].requires_grad_(True)
        outputs['classification'].requires_grad_(True)

        loss_fn = MultiTaskLoss(seg_weight=1.0, cls_weight=0.5, loss_type='dice_ce')
        loss_dict = loss_fn(outputs, targets)

        # Backward pass
        loss_dict['total_loss'].backward()

        # Check gradients
        seg_grad_norm = outputs['segmentation'].grad.norm().item()
        cls_grad_norm = outputs['classification'].grad.norm().item()

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

                # Forward pass
                with torch.no_grad():
                    outputs = network(input_tensor)

                # Validate outputs
                assert isinstance(outputs, dict), "Output should be a dictionary"
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

    def test_network_memory(self):
        """Test memory usage of the network"""
        print("\n=== Testing Network Memory Usage ===")

        if not torch.cuda.is_available():
            print("⚠ CUDA not available, skipping memory test")
            return True

        try:
            device = torch.device('cuda')
            network = self.create_test_network().to(device)

            # Clear cache
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated()

            # Forward pass
            input_tensor = torch.randn(1, self.input_channels, 64, 128, 192).to(device)
            with torch.no_grad():
                outputs = network(input_tensor)

            memory_after = torch.cuda.memory_allocated()
            memory_used = (memory_after - memory_before) / 1024**3  # GB

            print(f"✓ Memory used: {memory_used:.2f} GB")

            # Warn if memory usage is too high
            if memory_used > 8.0:
                print(f"⚠ High memory usage detected: {memory_used:.2f} GB")

            return True

        except Exception as e:
            error_print(e, context="Memory test")
            return False

    def test_memory_comparison(self):
        """Compare memory usage across network variants"""
        print("\n=== Comparing Memory Usage Across Variants ===")

        if not torch.cuda.is_available():
            print("⚠ CUDA not available, skipping memory comparison")
            return True

        results = {}
        device = torch.device('cuda')
        input_tensor = None

        try:
            # Generate input tensor first to avoid memory fragmentation
            input_tensor = torch.randn(1, self.input_channels, 64, 128, 192).to(device)

            for variant_name in self.network_variants.keys():
                print(f"\nTesting {variant_name} variant...")

                try:
                    # Clear everything from GPU
                    if 'network' in locals():
                        del network
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                    # Record baseline
                    base_memory = torch.cuda.memory_allocated()

                    # Create network
                    network = self.create_test_network(variant_name).to(device)

                    # Record memory after model creation
                    model_memory = (torch.cuda.memory_allocated() - base_memory) / 1024**3
                    print(f"  Model size: {model_memory:.2f} GB")

                    # Forward pass
                    with torch.no_grad():
                        start_forward = torch.cuda.memory_allocated()
                        outputs = network(input_tensor)
                        torch.cuda.synchronize()  # Ensure forward pass is complete

                    # Calculate memory metrics
                    peak_memory = torch.cuda.max_memory_allocated()
                    forward_memory = (peak_memory - start_forward) / 1024**3
                    total_memory = (peak_memory - base_memory) / 1024**3

                    results[variant_name] = {
                        'status': 'PASS',
                        'model_size': model_memory,
                        'forward_memory': forward_memory,
                        'total_memory': total_memory,
                        'peak_memory': peak_memory / 1024**3
                    }

                    print(f"  Forward pass memory: {forward_memory:.2f} GB")
                    print(f"  Total memory used: {total_memory:.2f} GB")
                    print(f"✓ {variant_name} test completed")

                except Exception as e:
                    error_print(e, context=f"Memory test for {variant_name}")
                    results[variant_name] = {'status': 'FAIL', 'error': str(e)}
                    continue

                finally:
                    # Cleanup after each variant
                    if 'network' in locals():
                        del network
                    torch.cuda.empty_cache()

        except Exception as e:
            error_print(e, context="Memory comparison setup")
            return False

        finally:
            # Final cleanup
            if input_tensor is not None:
                del input_tensor
            torch.cuda.empty_cache()

        # Print comparison table
        print("\nMemory Usage Comparison:")
        print("=" * 80)
        print(f"{'Variant':20s} | {'Model Size':12s} | {'Forward Pass':12s} | {'Total Usage':12s}")
        print("-" * 80)

        for variant, result in results.items():
            if result['status'] == 'PASS':
                print(f"{variant:20s} | {result['model_size']:10.2f} GB | {result['forward_memory']:10.2f} GB | {result['total_memory']:10.2f} GB")
            else:
                print(f"{variant:20s} | {'FAILED':-^38s}")

        print("=" * 80)

        return all(r['status'] == 'PASS' for r in results.values())

    def test_network_parameters(self):
        """Test network parameter count and gradient flow"""
        print("\n=== Testing Network Parameters ===")

        try:
            network = self.create_test_network()

            # Count parameters
            total_params = sum(p.numel() for p in network.parameters())
            trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)

            print(f"✓ Total parameters: {total_params:,}")
            print(f"✓ Trainable parameters: {trainable_params:,}")

            # Test gradient flow
            input_tensor = torch.randn(1, self.input_channels, 32, 64, 96)
            outputs = network(input_tensor)

            # Dummy loss
            seg_loss = outputs['segmentation'].sum()
            cls_loss = outputs['classification'].sum()
            total_loss = seg_loss + cls_loss

            total_loss.backward()

            # Check if gradients exist
            grad_norms = []
            for name, param in network.named_parameters():
                if param.grad is not None:
                    grad_norms.append(param.grad.norm().item())

            assert len(grad_norms) > 0, "No gradients found"
            avg_grad_norm = np.mean(grad_norms)
            print(f"✓ Average gradient norm: {avg_grad_norm:.6f}")

            return True

        except Exception as e:
            error_print(e, context="Parameter test")
            return False


class TestPlannerIntegration:
    """Test multi-task planner functionality"""

    def create_mock_dataset_json(self):
        """Create mock dataset JSON for testing"""
        return {
            'name': 'Dataset001_PancreasSegClassification',
            'description': 'Test dataset for multi-task learning',
            'labels': {
                '0': 'background',
                '1': 'pancreas',
                '2': 'lesion'
            },
            'numTest': 10,
            'numTraining': 100,
            'file_ending': '.nii.gz',
            'classification_labels': {
                '0': 'subtype_0',
                '1': 'subtype_1',
                '2': 'subtype_2'
            }
        }

    def test_planner_initialization(self):
        """Test planner initialization with different variants"""
        print("\n=== Testing Planner Initialization ===")

        planners = {
            'base': MultiTaskResEncUNetPlanner,
            'channel_attention': MultiTaskChannelAttentionResEncUNetPlanner,
            'efficient_attention': MultiTaskEfficientAttentionResEncUNetPlanner
        }

        dataset_json = self.create_mock_dataset_json()

        for planner_name, planner_class in planners.items():
            try:
                print(f"\nTesting {planner_name} planner:")

                # Create planner with minimal settings
                planner = planner_class(
                    dataset_name_or_id='Dataset001_PancreasSegClassification',
                    gpu_memory_target_in_gb=8,
                    preprocessor_name='DefaultPreprocessor'
                )

                # Test basic attributes
                assert hasattr(planner, 'UNet_class'), f"{planner_name} should have UNet_class"
                assert hasattr(planner, 'plans_identifier'), f"{planner_name} should have plans_identifier"

                print(f"✓ {planner_name} planner initialized successfully")
                print(f"✓ UNet class: {planner.UNet_class.__name__}")
                print(f"✓ Plans identifier: {planner.plans_identifier}")

            except Exception as e:
                error_print(e, context=f"Testing {planner_name} planner initialization")
                return False

        return True

    def test_plan_generation(self):
        """Test plan generation for multi-task configurations"""
        print("\n=== Testing Plan Generation ===")

        try:
            # Use base planner for testing
            planner = MultiTaskResEncUNetPlanner(
                dataset_name_or_id='Dataset001_PancreasSegClassification',
                gpu_memory_target_in_gb=8
            )

            # Create dummy parameters for plan generation
            spacing = [2.0, 0.73046875, 0.73046875]
            median_shape = (64, 119, 178)
            data_identifier = 'nnUNetMultiTaskResEncUNetPlans_3d_fullres'
            approximate_n_voxels_dataset = 1000000.0
            cache = {}

            # Generate plan
            plan = planner.get_plans_for_configuration(
                spacing=spacing,
                median_shape=median_shape,
                data_identifier=data_identifier,
                approximate_n_voxels_dataset=approximate_n_voxels_dataset,
                _cache=cache
            )

            # Validate plan structure
            assert isinstance(plan, dict), "Plan should be a dictionary"
            assert 'architecture' in plan, "Plan should contain architecture"
            assert 'network_class_name' in plan['architecture'], "Architecture should specify network class"

            # Check multi-task specific parameters
            arch_kwargs = plan['architecture']['arch_kwargs']
            assert 'num_classification_classes' in arch_kwargs, "Should have classification classes parameter"
            assert 'use_classification_head' in arch_kwargs, "Should have classification head parameter"

            print("✓ Plan generated successfully")
            print(f"✓ Network class: {plan['architecture']['network_class_name']}")
            print(f"✓ Classification classes: {arch_kwargs['num_classification_classes']}")

            return True

        except Exception as e:
            error_print(e, context="Plan generation")
            return False

    def test_memory_estimation(self):
        """Test VRAM estimation for multi-task networks"""
        print("\n=== Testing Memory Estimation ===")

        try:
            planner = MultiTaskResEncUNetPlanner(
                dataset_name_or_id='Dataset001_PancreasSegClassification',
                gpu_memory_target_in_gb=8
            )

            # Test parameters
            patch_size = [64, 128, 192]
            num_input_channels = 1
            num_output_channels = 3
            network_class_name = 'src.architectures.MultiTaskResEncUNet.MultiTaskResEncUNet'
            arch_kwargs = {
                'n_stages': 6,
                'features_per_stage': (32, 64, 128, 256, 320, 320),
                'conv_op': 'torch.nn.modules.conv.Conv3d',
                'kernel_sizes': ((1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)),
                'strides': ((1, 1, 1), (1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
                'n_blocks_per_stage': (1, 3, 4, 6, 6, 6),
                'n_conv_per_stage_decoder': (1, 1, 1, 1, 1),
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
            }
            arch_kwargs_req_import = ('conv_op', 'norm_op', 'dropout_op', 'nonlin')

            # Estimate memory
            memory_estimate = planner.static_estimate_VRAM_usage(
                patch_size=patch_size,
                num_input_channels=num_input_channels,
                num_output_channels=num_output_channels,
                network_class_name=network_class_name,
                arch_kwargs=arch_kwargs,
                arch_kwargs_req_import=arch_kwargs_req_import
            )

            assert memory_estimate > 0, "Memory estimate should be positive"
            memory_gb = memory_estimate / (1024**3)

            print(f"✓ Memory estimation successful: {memory_gb:.2f} GB")

            return True

        except Exception as e:
            error_print(e, context="Memory estimation")
            return False


class TestTrainerIntegration:
    """Test integration with nnUNet trainer"""

    def create_mock_plans_and_config(self):
        """Create mock plans and configuration for testing"""
        plans = {
            'dataset_name': 'Dataset001_PancreasSegClassification',
            'plans_name': 'nnUNetResEncUNetMPlans',
            'original_median_spacing_after_transp': [2.0, 0.73046875, 0.73046875],
            'original_median_shape_after_transp': [64, 119, 178],
            'image_reader_writer': 'SimpleITKIO',
            'transpose_forward': [0, 1, 2],
            'transpose_backward': [0, 1, 2],
            'configurations': {
                '2d': {
                    'label_manager': 'LabelManager',
                    'data_identifier': 'nnUNetPlans_2d',
                    'preprocessor_name': 'DefaultPreprocessor',
                    'batch_size': 134,
                    'patch_size': [128, 192],
                    'median_image_size_in_voxels': [118.0, 181.0],
                    'spacing': [0.73046875, 0.73046875],
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
                        'network_class_name': 'dynamic_network_architectures.architectures.unet.ResidualEncoderUNet',
                        'arch_kwargs': {
                            'n_stages': 6,
                            'features_per_stage': [32, 64, 128, 256, 512, 512],
                            'conv_op': 'torch.nn.modules.conv.Conv2d',
                            'kernel_sizes': [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]],
                            'strides': [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
                            'n_blocks_per_stage': [1, 3, 4, 6, 6, 6],
                            'n_conv_per_stage_decoder': [1, 1, 1, 1, 1],
                            'conv_bias': True,
                            'norm_op': 'torch.nn.modules.instancenorm.InstanceNorm2d',
                            'norm_op_kwargs': {'eps': 1e-05, 'affine': True},
                            'dropout_op': None,
                            'dropout_op_kwargs': None,
                            'nonlin': 'torch.nn.LeakyReLU',
                            'nonlin_kwargs': {'inplace': True}
                        },
                        '_kw_requires_import': ['conv_op', 'norm_op', 'dropout_op', 'nonlin']
                    },
                    'batch_dice': True
                },
                '3d_fullres': {
                    'label_manager': 'LabelManager',
                    'data_identifier': 'nnUNetPlans_3d_fullres',
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
                        'network_class_name': 'dynamic_network_architectures.architectures.unet.ResidualEncoderUNet',
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
                            'nonlin_kwargs': {'inplace': True}
                        },
                        '_kw_requires_import': ['conv_op', 'norm_op', 'dropout_op', 'nonlin']
                    },
                    'batch_dice': False
                }
            },
            'experiment_planner_used': 'nnUNetPlannerResEncM',
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

            # Create temporary directory for outputs
            with tempfile.TemporaryDirectory() as temp_dir:
                trainer = nnUNetTrainerMultiTask(
                    plans=plans,
                    configuration='3d_fullres',
                    fold=0,
                    dataset_json=dataset_json,
                    device=torch.device('cpu')
                )

                # Test trainer attributes
                assert trainer.num_classification_classes == 3
                assert trainer.seg_weight == 1.0
                assert trainer.cls_weight == 0.5

                print("✓ Trainer initialized successfully")
                print(f"✓ Classification classes: {trainer.num_classification_classes}")
                print(f"✓ Loss weights - Seg: {trainer.seg_weight}, Cls: {trainer.cls_weight}")

                return True

        except Exception as e:
            error_print(e, context="Trainer initialization")
            return False

    def test_loss_building(self):
        """Test loss function building in trainer"""
        print("\n=== Testing Loss Building ===")

        try:
            plans, dataset_json = self.create_mock_plans_and_config()

            with tempfile.TemporaryDirectory() as temp_dir:
                trainer = nnUNetTrainerMultiTask(
                    plans=plans,
                    configuration='3d_fullres',
                    fold=0,
                    dataset_json=dataset_json,
                    device=torch.device('cpu')
                )

                # Build loss
                loss_fn = trainer._build_loss()

                assert loss_fn is not None, "Loss function should not be None"
                assert isinstance(loss_fn, MultiTaskLoss), "Loss should be MultiTaskLoss instance"

                print("✓ Loss function built successfully")
                print(f"✓ Loss type: {type(loss_fn).__name__}")

                return True

        except Exception as e:
            error_print(e, context="Loss building")
            return False


def run_all_tests():
    """Run all test suites"""
    print("🧪 Starting Custom nnUNet Test Suite")
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

    # Test 2: Network Architecture
    print("\n🏗️ TESTING NETWORK ARCHITECTURE")
    network_tester = TestMultiTaskNetwork()
    tests = [
        network_tester.test_network_variants,    # New test
        network_tester.test_memory_comparison,   # New test
        # network_tester.test_network_forward,     # Original test
        # network_tester.test_network_memory,      # Original test
        # network_tester.test_network_parameters   # Original test
    ]

    for test in tests:
        if not test():
            all_passed = False

    # Test 3: Planner Integration
    print("\n📋 TESTING PLANNER INTEGRATION")
    planner_tester = TestPlannerIntegration()
    tests = [
        planner_tester.test_planner_initialization,
        planner_tester.test_plan_generation,
        planner_tester.test_memory_estimation
    ]

    for test in tests:
        if not test():
            all_passed = False

    # Test 4: Trainer Integration
    print("\n👩‍🏫 TESTING TRAINER INTEGRATION")
    trainer_tester = TestTrainerIntegration()
    tests = [
        trainer_tester.test_trainer_initialization,
        trainer_tester.test_loss_building
    ]

    for test in tests:
        if not test():
            all_passed = False

    # Final results
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Your implementation is ready for training.")
        print("\n💡 Next steps:")
        print("   1. Prepare your dataset in nnUNet format")
        print("   2. Run: nnUNetv2_plan_and_preprocess -d Dataset001_PancreasSegClassification")
        print("   3. Train: nnUNetv2_train Dataset001_PancreasSegClassification 3d_fullres 0 -tr nnUNetTrainerMultiTask")
    else:
        print("❌ SOME TESTS FAILED! Please fix the issues before training.")
        print("\n🔧 Debug tips:")
        print("   - Check import paths in custom_nnunet modules")
        print("   - Verify network architecture parameters")
        print("   - Test with smaller batch sizes if memory issues")

    return all_passed


if __name__ == '__main__':
    # add nnUNet_raw, preprocessed, and results directories to os.environ
    os.environ['nnUNet_raw'] = '/mnt/data/gpu-server/nnUNet_modified/nnunet_data/nnUNet_raw'
    os.environ['nnUNet_preprocessed'] = '/mnt/data/gpu-server/nnUNet_modified/nnunet_data/nnUNet_preprocessed'
    os.environ['nnUNet_results'] = '/mnt/data/gpu-server/nnUNet_modified/nnunet_data/nnUNet_results'

    # Set up environment
    torch.manual_seed(42)
    np.random.seed(42)

    # Run tests
    success = run_all_tests()
    sys.exit(0 if success else 1)
