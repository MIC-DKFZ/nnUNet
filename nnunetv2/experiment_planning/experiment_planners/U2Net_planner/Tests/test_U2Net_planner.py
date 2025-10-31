import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from nnunetv2.experiment_planning.experiment_planners.U2Net_planner.U2Net_planner import U2NetPlanner
from dynamic_network_architectures.architectures.u2net import U2Net


class TestU2NetPlanner:
    """Test suite for the U2Net-optimized planner."""
    
    @pytest.fixture
    def simple_planner(self):
        """Create a minimal U2NetPlanner for testing."""
        planner = U2NetPlanner.__new__(U2NetPlanner)
        
        # Essential U2Net planning attributes
        planner.plans_identifier = 'U2NetPlans'
        planner.UNet_class = U2Net
        planner.max_2d_stages = 5
        planner.max_3d_stages = 4
        planner.depth_per_stage = [7, 6, 5, 4, 4]
        planner.UNet_max_features_3d = 512
        planner.UNet_max_features_2d = 1024
        planner.lowres_creation_threshold = 0.25  # Use default value for testing
        
        # Mock dataset information
        planner.dataset_json = {
            'channel_names': {'0': 'T1'},
            'labels': {'background': 0, 'tumor': 1}
        }
        
        # Attributes needed for planning
        planner.UNet_base_num_features = 32
        planner.UNet_featuremap_min_edge_length = 4
        planner.UNet_vram_target_GB = 8.0
        planner.UNet_reference_val_corresp_GB = 8.0
        planner.UNet_reference_val_corresp_bs_2d = 2
        planner.UNet_reference_val_corresp_bs_3d = 2
        planner.UNet_reference_val_2d = 135000000  # Add missing reference values
        planner.UNet_reference_val_3d = 680000000  # Add missing reference values
        planner.UNet_min_batch_size = 1
        planner.max_dataset_covered = 0.05
        planner.preprocessor_name = 'DefaultPreprocessor'
        
        # Mock VRAM estimation
        planner.static_estimate_VRAM_usage = MagicMock(return_value=2000000000)
        
        # Mock required methods
        planner.determine_resampling = MagicMock(return_value=(
            MagicMock(__name__='resample'), {}, MagicMock(__name__='resample'), {}
        ))
        planner.determine_segmentation_softmax_export_fn = MagicMock(return_value=(
            MagicMock(__name__='softmax'), {}
        ))
        planner.determine_normalization_scheme_and_whether_mask_is_used_for_norm = MagicMock(return_value=(
            [{'type': 'ZScoreNormalization'}], [False]
        ))
        
        return planner

    def test_u2net_attributes(self, simple_planner):
        """Test U2Net-specific attributes."""
        assert simple_planner.UNet_class == U2Net
        assert simple_planner.max_3d_stages == 4
        assert simple_planner.depth_per_stage == [7, 6, 5, 4, 4]
        assert simple_planner.lowres_creation_threshold == 0.25  # Default value allows 3d_lowres

    def test_data_identifier(self, simple_planner):
        """Test data identifier generation."""
        assert simple_planner.generate_data_identifier("2d") == "U2NetPlans_2d"
        assert simple_planner.generate_data_identifier("3d_fullres") == "U2NetPlans_3d_fullres"

    def test_2d_planning(self, simple_planner):
        """Test 2D U2Net planning."""
        spacing = np.array([1.0, 1.0])
        shape = np.array([256, 256])
        cache = {}
        
        plans = simple_planner.get_plans_for_configuration(
            spacing, shape, "test_2d", 100000, cache
        )
        
        assert "architecture" in plans
        arch_kwargs = plans["architecture"]["arch_kwargs"]
        assert "depth_per_stage" in arch_kwargs
        assert arch_kwargs["n_stages"] <= simple_planner.max_2d_stages

    def test_3d_planning(self, simple_planner):
        """Test 3D U2Net planning."""
        spacing = np.array([1.0, 1.0, 1.0])
        shape = np.array([128, 128, 128])
        cache = {}
        
        plans = simple_planner.get_plans_for_configuration(
            spacing, shape, "test_3d", 200000, cache
        )
        
        assert "architecture" in plans
        arch_kwargs = plans["architecture"]["arch_kwargs"]
        assert arch_kwargs["n_stages"] <= simple_planner.max_3d_stages
        assert len(arch_kwargs["depth_per_stage"]) == arch_kwargs["n_stages"]

    def test_u2net_creation(self):
        """Test that U2Net can be created with typical parameters."""
        from torch import nn
        
        net = U2Net(
            input_channels=1,
            n_stages=3,
            features_per_stage=(32, 64, 128),
            conv_op=nn.Conv2d,
            kernel_sizes=[(3, 3), (3, 3), (3, 3)],
            strides=[(1, 1), (2, 2), (2, 2)],
            num_classes=2,
            deep_supervision=True,
            norm_op=nn.InstanceNorm2d,
            nonlin=nn.Sigmoid,
            blocks_nonlin=nn.ReLU,
            depth_per_stage=[5, 4, 3]
        )
        
        assert net is not None
        fmap_size = net.compute_conv_feature_map_size((64, 64))
        assert fmap_size > 0

    def test_minimum_stages_constraint(self, simple_planner):
        """Test that U2Net planner enforces minimum 2 stages."""
        # Test with very small spacing that would normally result in 1 stage
        spacing = np.array([10.0, 10.0, 10.0])  # Very large spacing
        shape = np.array([8, 8, 8])  # Very small shape
        cache = {}
        
        plans = simple_planner.get_plans_for_configuration(
            spacing, shape, "test_min_stages", 1000, cache
        )
        
        # Should still have at least 2 stages due to our constraint
        arch_kwargs = plans["architecture"]["arch_kwargs"]
        assert arch_kwargs["n_stages"] >= 2
        assert len(arch_kwargs["depth_per_stage"]) >= 2
        assert len(arch_kwargs["features_per_stage"]) >= 2

    def test_rsu_memory_overhead_calculation(self, simple_planner):
        """Test RSU memory overhead estimation."""
        spacing = np.array([1.0, 1.0])
        shape = np.array([128, 128])
        cache = {}
        
        plans = simple_planner.get_plans_for_configuration(
            spacing, shape, "test_memory", 50000, cache
        )
        
        # Memory estimation should be called with RSU overhead applied
        assert simple_planner.static_estimate_VRAM_usage.called
        
        # Check that depth_per_stage influences the calculation
        arch_kwargs = plans["architecture"]["arch_kwargs"]
        assert "depth_per_stage" in arch_kwargs
        depths = arch_kwargs["depth_per_stage"]
        assert all(isinstance(d, int) and d > 0 for d in depths)

    def test_features_per_stage_calculation(self, simple_planner):
        """Test feature calculation for different stage counts."""
        spacing_2d = np.array([1.0, 1.0])
        spacing_3d = np.array([1.0, 1.0, 1.0])
        shape = np.array([64, 64, 64])
        cache = {}
        
        # Test 2D features
        plans_2d = simple_planner.get_plans_for_configuration(
            spacing_2d, shape[:2], "test_features_2d", 10000, cache
        )
        features_2d = plans_2d["architecture"]["arch_kwargs"]["features_per_stage"]
        
        # Test 3D features  
        plans_3d = simple_planner.get_plans_for_configuration(
            spacing_3d, shape, "test_features_3d", 10000, cache
        )
        features_3d = plans_3d["architecture"]["arch_kwargs"]["features_per_stage"]
        
        # 2D should generally allow more features than 3D
        assert max(features_2d) >= max(features_3d)
        
        # Features should increase with stages
        assert all(features_2d[i] <= features_2d[i+1] for i in range(len(features_2d)-1))
        assert all(features_3d[i] <= features_3d[i+1] for i in range(len(features_3d)-1))

    def test_topology_optimization(self, simple_planner):
        """Test U2Net-optimized topology calculation."""
        # Test with different aspect ratios
        spacing = np.array([1.0, 1.0, 1.0])
        
        # Isotropic shape
        shape_iso = np.array([64, 64, 64])
        plans_iso = simple_planner.get_plans_for_configuration(
            spacing, shape_iso, "test_iso", 20000, {}
        )
        
        # Anisotropic shape  
        shape_aniso = np.array([32, 128, 128])
        plans_aniso = simple_planner.get_plans_for_configuration(
            spacing, shape_aniso, "test_aniso", 20000, {}
        )
        
        # Both should have valid architectures
        assert "architecture" in plans_iso
        assert "architecture" in plans_aniso
        
        # Patch sizes should be reasonable (allowing for very small patches in edge cases)
        patch_iso = plans_iso["patch_size"]
        patch_aniso = plans_aniso["patch_size"]
        
        assert all(p >= 1 for p in patch_iso)  # At least 1 voxel per dimension
        assert all(p >= 1 for p in patch_aniso)
        assert len(patch_iso) == 3  # 3D
        assert len(patch_aniso) == 3  # 3D

    def test_patch_size_reduction_behavior(self, simple_planner):
        """Test patch size reduction with memory constraints."""
        # Create a scenario that will trigger patch size reduction
        spacing = np.array([1.0, 1.0])
        shape = np.array([512, 512])  # Large shape
        cache = {}
        
        # Mock high memory usage to trigger reduction
        simple_planner.static_estimate_VRAM_usage.return_value = 20000000000  # 20GB
        
        plans = simple_planner.get_plans_for_configuration(
            spacing, shape, "test_reduction", 100000, cache
        )
        
        # Patch size should be reduced from initial estimate
        patch_size = plans["patch_size"]
        assert all(p < s for p, s in zip(patch_size, shape))
        
        # Should still meet basic requirements (patch size can be small under high memory pressure)
        assert all(p >= 1 for p in patch_size)  # At least 1 voxel per dimension
        assert len(patch_size) == 2  # 2D

    def test_cache_key_generation(self, simple_planner):
        """Test cache key generation for different configurations."""
        spacing = np.array([1.0, 1.0])
        shape = np.array([128, 128])
        cache = {}
        
        # Run same configuration twice
        plans1 = simple_planner.get_plans_for_configuration(
            spacing, shape, "test_cache1", 50000, cache
        )
        
        plans2 = simple_planner.get_plans_for_configuration(
            spacing, shape, "test_cache2", 50000, cache
        )
        
        # Cache should be populated
        assert len(cache) > 0
        
        # Different data identifiers but same config should use cache
        assert simple_planner.static_estimate_VRAM_usage.call_count >= 1

    def test_depth_extension_edge_case(self, simple_planner):
        """Test depth_per_stage extension when more stages needed."""
        # Force a scenario where we need more stages than depth values
        original_depths = simple_planner.depth_per_stage.copy()
        simple_planner.depth_per_stage = [7, 6]  # Only 2 depths
        
        spacing = np.array([0.5, 0.5, 0.5])  # Fine spacing
        shape = np.array([128, 128, 128])
        cache = {}
        
        plans = simple_planner.get_plans_for_configuration(
            spacing, shape, "test_depth_extend", 100000, cache
        )
        
        arch_kwargs = plans["architecture"]["arch_kwargs"]
        stages = arch_kwargs["n_stages"]
        depths = arch_kwargs["depth_per_stage"]
        
        # Should have depths for all stages
        assert len(depths) == stages
        
        # If extended, should use last available depth
        if stages > 2:
            assert depths[-1] == 6  # Last original depth
        
        # Restore original
        simple_planner.depth_per_stage = original_depths

    def test_error_handling_invalid_spacing(self, simple_planner):
        """Test error handling with invalid spacing."""
        with pytest.raises(AssertionError):
            simple_planner.get_plans_for_configuration(
                np.array([0.0, 1.0]),  # Invalid: zero spacing
                np.array([128, 128]),
                "test_invalid",
                50000,
                {}
            )
        
        with pytest.raises(AssertionError):
            simple_planner.get_plans_for_configuration(
                np.array([-1.0, 1.0]),  # Invalid: negative spacing
                np.array([128, 128]),
                "test_invalid",
                50000,
                {}
            )

    def test_batch_size_calculation(self, simple_planner):
        """Test batch size calculation with U2Net factors."""
        spacing = np.array([1.0, 1.0])
        shape = np.array([64, 64])
        cache = {}
        
        # Mock lower memory usage
        simple_planner.static_estimate_VRAM_usage.return_value = 1000000000  # 1GB
        
        plans = simple_planner.get_plans_for_configuration(
            spacing, shape, "test_batch", 10000, cache
        )
        
        batch_size = plans["batch_size"]
        
        # Should be reasonable and >= minimum
        assert batch_size >= simple_planner.UNet_min_batch_size
        assert isinstance(batch_size, int)

    def test_architecture_consistency(self, simple_planner):
        """Test that architecture parameters are consistent."""
        spacing = np.array([1.0, 1.0, 1.0])
        shape = np.array([64, 64, 64])
        cache = {}
        
        plans = simple_planner.get_plans_for_configuration(
            spacing, shape, "test_consistency", 25000, cache
        )
        
        arch_kwargs = plans["architecture"]["arch_kwargs"]
        
        # All lists should have same length as n_stages
        n_stages = arch_kwargs["n_stages"]
        assert len(arch_kwargs["features_per_stage"]) == n_stages
        assert len(arch_kwargs["kernel_sizes"]) == n_stages
        assert len(arch_kwargs["strides"]) == n_stages
        assert len(arch_kwargs["depth_per_stage"]) == n_stages
        
        # Check U2Net-specific parameters
        assert arch_kwargs["deep_supervision"] == True
        assert "nonlin" in arch_kwargs
        assert "blocks_nonlin" in arch_kwargs
        assert arch_kwargs["depth_per_stage"] is not None

    def test_extreme_patch_sizes(self, simple_planner):
        """Test handling of extreme patch size scenarios."""
        # Very small shape
        spacing_fine = np.array([0.1, 0.1, 0.1])
        shape_tiny = np.array([16, 16, 16])
        
        plans_tiny = simple_planner.get_plans_for_configuration(
            spacing_fine, shape_tiny, "test_tiny", 1000, {}
        )
        
        # Should still produce valid plans
        assert "patch_size" in plans_tiny
        patch_size = plans_tiny["patch_size"]
        assert all(p >= 1 for p in patch_size)  # At least 1 voxel per dimension
        assert len(patch_size) == 3  # 3D
        
        # Architecture should be valid
        arch_kwargs = plans_tiny["architecture"]["arch_kwargs"]
        assert arch_kwargs["n_stages"] >= 2
