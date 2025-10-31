import numpy as np
from typing import Union, List, Tuple

from dynamic_network_architectures.architectures.u2net import U2Net
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm
from torch import nn

from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner

from nnunetv2.experiment_planning.experiment_planners.network_topology import get_pool_and_conv_props

__author__ = ["Stefano Petraccini"]
__email__ = ["stefano.petraccini@studio.unibo.it"]

class U2NetPlanner(ExperimentPlanner):
    """
    Experiment planner for U2Net architecture within the nnUNet framework.
    
    This planner extends the default ExperimentPlanner to work with the U2Net architecture,
    which uses Residual U-blocks (RSU) with nested U-structures. It provides optimized
    configuration of network parameters, memory estimation accounting for RSU overhead,
    and topology calculation tailored for U2Net's unique requirements.
    
    Key features:
    - RSU-aware memory estimation with overhead calculations
    - Conservative patch sizing to accommodate nested U-structures  
    - Configurable depths per stage for RSU blocks
    - Minimum 2-stage enforcement (required by RSUDecoder)
    - U2Net-specific batch size and topology optimization
    
    Parameters
    ----------
    dataset_name_or_id : Union[str, int]
        Dataset name or ID to plan experiments for.
    gpu_memory_target_in_gb : float, optional
        Target GPU memory usage in GB, by default 8.
    preprocessor_name : str, optional
        Name of the preprocessor to use, by default 'DefaultPreprocessor'.
    plans_name : str, optional
        Name for the plans file, by default 'U2NetPlans'.
    overwrite_target_spacing : Union[List[float], Tuple[float, ...]], optional
        Custom target spacing to use instead of computed spacing, by default None.
    suppress_transpose : bool, optional
        Whether to suppress data transposition during preprocessing, by default False.
    
    Attributes
    ----------
    depth_per_stage : List[int]
        RSU block depths for each network stage
    max_2d_stages : int
        Maximum number of stages for 2D configurations
    max_3d_stages : int
        Maximum number of stages for 3D configurations
    UNet_max_features_2d : int
        Maximum feature channels for 2D networks
    UNet_max_features_3d : int
        Maximum feature channels for 3D networks
    """

    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', 
                 plans_name: str = 'U2NetPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_class = U2Net
        
        # the following numbers are reference values for VRAM estimation
        self.UNet_reference_val_3d = 680000000
        self.UNet_reference_val_2d = 135000000
        self.UNet_reference_val_corresp_GB = 8  # Reference GPU memory in GB
        self.UNet_reference_val_corresp_bs_2d = 12  # Reference batch size for 2D
        self.UNet_reference_val_corresp_bs_3d = 2   # Reference batch size for 3D
        
        # can be useful to set a maximum number of stages without having to reduce UNet_reference_val_ in order to keep a reasonable patch size.
        self.max_2d_stages = 6  
        self.max_3d_stages = 5  
        
        # RSU depths for each stage
        self.depth_per_stage = [7, 6, 5, 4, 4, 4]  # if changing self.max_3d_stages or self.max_2d_stages, make sure this is consistent.

        # next two lines override the default values in ExperimentPlanner
        self.UNet_max_features_2d = 1024  # default is 512
        self.UNet_max_features_3d = 512  # default is 320
        
        

    def generate_data_identifier(self, configuration_name: str) -> str:
        """
        Generate a unique identifier for data associated with a configuration.
        
        Creates an identifier that reflects both the configuration name and the plans
        it originates from, ensuring uniqueness across different plans files that
        may contain configurations with the same name.
        
        Parameters
        ----------
        configuration_name : str
            Name of the configuration (e.g., '2d', '3d_fullres', '3d_lowres').
            
        Returns
        -------
        str
            Unique data identifier in format '{plans_identifier}_{configuration_name}'.
            
        Notes
        -----
        This method allows to distinguish between configurations from different
        plans files for the same dataset. 
        """
        return self.plans_identifier + '_' + configuration_name

    def get_plans_for_configuration(self,
                                    spacing: Union[np.ndarray, Tuple[float, ...], List[float]],
                                    median_shape: Union[np.ndarray, Tuple[int, ...]],
                                    data_identifier: str,
                                    approximate_n_voxels_dataset: float,
                                    _cache: dict) -> dict:
        """
        Generate optimized plans for a specific U2Net network configuration.
        
        This method determines network architecture, batch size, patch size, and other
        configuration parameters based on the provided data characteristics and 
        U2Net-specific constraints. It accounts for RSU block memory overhead and
        implements conservative sizing strategies for stable training.
        
        Parameters
        ----------
        spacing : Union[np.ndarray, Tuple[float, ...], List[float]]
            Voxel spacing of the data (e.g., [1.0, 1.0, 1.0] for isotropic).
        median_shape : Union[np.ndarray, Tuple[int, ...]]
            Median shape of images in the dataset in voxels.
        data_identifier : str
            Unique identifier for this data configuration.
        approximate_n_voxels_dataset : float
            Approximate total number of voxels in the entire dataset.
        _cache : dict
            Cache dictionary for storing intermediate VRAM calculations.
            
        Returns
        -------
        dict
            Configuration dictionary containing:
            - 'data_identifier': Configuration identifier
            - 'batch_size': Optimized batch size for U2Net
            - 'patch_size': Network input patch size  
            - 'architecture': Network architecture parameters including RSU depths
            - 'spacing': Target voxel spacing
            - 'normalization_schemes': Data normalization configuration
            - 'resampling_*': Resampling function configurations
            - Other preprocessing and training parameters
            
        Notes
        -----
        The method implements several U2Net-specific optimizations:
        - RSU memory overhead estimation (1.5-3x regular convolutions)
        - Conservative patch sizing (20% smaller initial patches)
        - Minimum 2 stages enforced (required by RSUDecoder)
        - U2Net batch size reduction (25% smaller than regular U-Net)
        - Iterative patch size reduction with 15% steps under memory pressure
        """

        def _features_per_stage(num_stages, max_num_features) -> Tuple[int, ...]:
            """
            Calculate feature channels for each network stage.
            
            Uses exponential scaling (base_features * 2^stage_index) capped at
            the maximum allowed features. U2Net typically uses the same scaling
            as regular U-Net but may use different maximum values.
            
            Parameters
            ----------
            num_stages : int
                Number of stages in the network.
            max_num_features : int
                Maximum number of feature channels allowed.
                
            Returns
            -------
            Tuple[int, ...]
                Feature channels for each stage (e.g., (32, 64, 128, 256, 512)).
            """
            return tuple([min(self.UNet_base_num_features * 2**i, max_num_features) for i in range(num_stages)])

        def _estimate_rsu_memory_overhead(depth, features, patch_size):
            """
            Estimate memory overhead multiplier for RSU blocks vs regular convolutions.
            
            RSU blocks contain nested U-structures that create multiple feature maps
            at different scales. This function estimates the additional memory required
            based on the RSU depth and nested pooling operations.
            
            Parameters
            ----------
            depth : int
                Depth of the RSU block (number of nested layers).
            features : int
                Number of feature channels in the block.
            patch_size : tuple
                Current patch size for scale calculations.
                
            Returns
            -------
            float
                Memory overhead multiplier. Values > 1.0 indicate RSU blocks require
                more memory than regular convolutions. Capped at 3.0x overhead.
                
            Notes
            -----
            The estimation considers:
            - Each nested level creates feature maps at 1/2^d resolution
            - Memory contribution decreases with resolution but accumulates
            - Overhead is capped to prevent overestimation affecting training
            """
            if depth <= 1:
                return 1.0
            
            # Base overhead increases with depth
            base_overhead = 1.0 + (depth - 1) * 0.3
            
            # Additional overhead from nested pooling operations
            nested_overhead = sum([1.0 / (2 ** i) for i in range(1, min(depth, 4))])
            
            total_overhead = base_overhead + nested_overhead * 0.2
            
            # Cap the overhead to prevent overestimation
            return min(total_overhead, 3.0)

        def _u2net_optimized_topology(spacing, initial_patch_size, min_edge_length, max_stages):
            """
            Calculate U2Net-optimized network topology parameters.
            
            Determines the network architecture (stages, pooling, convolution parameters)
            while considering RSU block memory requirements and U2Net constraints.
            Applies conservative patch sizing and ensures minimum stage requirements.
            
            Parameters
            ----------
            spacing : array-like
                Voxel spacing of the input data.
            initial_patch_size : array-like
                Initial patch size estimate before U2Net adjustments.
            min_edge_length : int
                Minimum edge length for feature maps at the bottleneck.
            max_stages : int
                Maximum number of network stages allowed for this dimensionality.
                
            Returns
            -------
            tuple
                (network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes,
                 adjusted_patch_size, shape_must_be_divisible_by)
                Network topology parameters optimized for U2Net architecture.
                
            Notes
            -----
            U2Net-specific adjustments:
            - Patch size reduced by 30-50% to account for RSU memory overhead
            - Minimum 2 stages enforced (required by RSUDecoder)
            - Stage count limited by available RSU depths
            - Topology extended/truncated to match required stages
            """
            # Apply U2Net memory factor to reduce patch size
            u2net_memory_factor = 1.5 if len(spacing) == 3 else 1.3
            conservative_patch_size = [max(16, int(p / u2net_memory_factor)) for p in initial_patch_size]
            
            # Get topology using the standard nnUNet method but with conservative patch size
            network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, \
            shape_must_be_divisible_by = get_pool_and_conv_props(spacing, conservative_patch_size, 
                                                                 min_edge_length, max_stages)
            
            # Get original stages count
            original_stages = len(pool_op_kernel_sizes)
            num_stages = original_stages
            
            # U2Net requires minimum 2 stages for RSUDecoder to work properly
            num_stages = max(num_stages, 2)
            
            # Apply maximum stages limit from planner configuration
            num_stages = min(num_stages, max_stages)
            
            # Limit stages by available RSU depths
            if len(self.depth_per_stage) < num_stages:
                num_stages = len(self.depth_per_stage)
                num_stages = max(num_stages, 2)  # Still enforce minimum
            
            # If we need to extend topology for more stages
            if num_stages > original_stages:
                # Extend with safe defaults
                while len(pool_op_kernel_sizes) < num_stages:
                    pool_op_kernel_sizes.append((1,) * len(spacing))
                while len(conv_kernel_sizes) < num_stages:
                    conv_kernel_sizes.append((3,) * len(spacing))
            
            # If we need to truncate topology for fewer stages
            elif num_stages < original_stages:
                pool_op_kernel_sizes = pool_op_kernel_sizes[:num_stages]
                conv_kernel_sizes = conv_kernel_sizes[:num_stages]
            
            return (network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, 
                    patch_size, shape_must_be_divisible_by)

        def _keygen(patch_size, strides, depths):
            """
            Generate cache key for VRAM estimation caching.
            
            Creates a unique string key based on patch size, network strides,
            and RSU depths to enable caching of expensive VRAM calculations.
            
            Parameters
            ----------
            patch_size : list or tuple
                Network input patch size.
            strides : list or tuple
                Pooling strides for each stage.
            depths : list or tuple
                RSU block depths for each stage.
                
            Returns
            -------
            str
                Cache key string combining all parameters.
            """
            return f"{list(patch_size)}_{list(strides)}_{list(depths)}"
        assert all([i > 0 for i in spacing]), f"Spacing must be > 0! Spacing: {spacing}"
        num_input_channels = len(self.dataset_json['channel_names'].keys()
                                 if 'channel_names' in self.dataset_json.keys()
                                 else self.dataset_json['modality'].keys())
        max_num_features = self.UNet_max_features_2d if len(spacing) == 2 else self.UNet_max_features_3d
        unet_conv_op = convert_dim_to_conv_op(len(spacing))

        # U2Net-specific initial patch size calculation
        # Start more conservatively due to RSU memory overhead
        tmp = 1 / np.array(spacing)
        
        if len(spacing) == 3:
            # Reduce initial size by 20% for 3D U2Net due to higher memory requirements
            initial_patch_size = [round(i) for i in tmp * (200 ** 3 / np.prod(tmp)) ** (1 / 3)]
        elif len(spacing) == 2:
            # Reduce initial size by 15% for 2D U2Net
            initial_patch_size = [round(i) for i in tmp * (1700 ** 2 / np.prod(tmp)) ** (1 / 2)]
        else:
            raise RuntimeError("Only 2D and 3D are supported")

        # Clip initial patch size to median_shape
        initial_patch_size = np.minimum(initial_patch_size, median_shape[:len(spacing)])

        # Apply stage limits
        max_stages_for_dim = self.max_2d_stages if len(spacing) == 2 else self.max_3d_stages

        # Get U2Net-optimized network topology
        network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, \
        shape_must_be_divisible_by = _u2net_optimized_topology(spacing, initial_patch_size,
                                                               self.UNet_featuremap_min_edge_length,
                                                               max_stages_for_dim)
        
        num_stages = len(pool_op_kernel_sizes)
        
        # U2Net requires minimum 2 stages for RSUDecoder to work properly
        num_stages = max(num_stages, 2)
        
        # Ensure we have enough depth values, extend with last value if needed
        depth_per_stage = self.depth_per_stage[:num_stages]
        if len(depth_per_stage) < num_stages:
            # Extend with the last available depth value
            last_depth = self.depth_per_stage[-1] if self.depth_per_stage else 4
            depth_per_stage.extend([last_depth] * (num_stages - len(depth_per_stage)))

        norm = get_matching_instancenorm(unet_conv_op)
        architecture_kwargs = {
            'network_class_name': self.UNet_class.__module__ + '.' + self.UNet_class.__name__,
            'arch_kwargs': {
                'n_stages': num_stages,
                'features_per_stage': _features_per_stage(num_stages, max_num_features),
                'conv_op': unet_conv_op.__module__ + '.' + unet_conv_op.__name__,
                'kernel_sizes': conv_kernel_sizes,
                'strides': pool_op_kernel_sizes,
                'deep_supervision': True,
                'conv_bias': True,
                'norm_op': norm.__module__ + '.' + norm.__name__,
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None,
                'dropout_op_kwargs': None,
                'nonlin': nn.Sigmoid.__module__ + '.' + nn.Sigmoid.__name__,
                'nonlin_kwargs': {'inplace': True},
                'blocks_nonlin': nn.ReLU.__module__ + '.' + nn.ReLU.__name__,
                'blocks_nonlin_kwargs': {'inplace': True},
                'depth_per_stage': depth_per_stage,
            },
            '_kw_requires_import': ('conv_op', 'norm_op', 'dropout_op', 'nonlin', 'blocks_nonlin'), 
        }

        # U2Net-aware VRAM estimation with RSU overhead
        cache_key = _keygen(patch_size, pool_op_kernel_sizes, depth_per_stage)
        if cache_key in _cache:
            base_estimate = _cache[cache_key]
        else:
            base_estimate = self.static_estimate_VRAM_usage(patch_size,
                                                           num_input_channels,
                                                           len(self.dataset_json['labels'].keys()),
                                                           architecture_kwargs['network_class_name'],
                                                           architecture_kwargs['arch_kwargs'],
                                                           architecture_kwargs['_kw_requires_import'])
        
        # Apply RSU memory overhead correction
        rsu_overhead = 1.0
        features_per_stage = _features_per_stage(num_stages, max_num_features)
        for stage_idx, (depth, features) in enumerate(zip(depth_per_stage, features_per_stage)):
            stage_overhead = _estimate_rsu_memory_overhead(depth, features, patch_size)
            rsu_overhead = max(rsu_overhead, stage_overhead)
        
        estimate = base_estimate * rsu_overhead
        _cache[cache_key] = estimate

        # Adjust reference values for U2Net
        reference = (self.UNet_reference_val_2d if len(spacing) == 2 else self.UNet_reference_val_3d) * \
                    (self.UNet_vram_target_GB / self.UNet_reference_val_corresp_GB)

        # Patch size reduction loop with U2Net-specific constraints
        iteration_count = 0
        max_iterations = 10  # Prevent infinite loops
        
        while estimate > reference and iteration_count < max_iterations:
            iteration_count += 1
            
            # More aggressive patch size reduction for U2Net
            axis_to_be_reduced = np.argsort([i / j for i, j in zip(patch_size, median_shape[:len(spacing)])])[-1]
            
            patch_size = list(patch_size)
            reduction_factor = 0.85  # Reduce by 15% each iteration (more aggressive than default)
            
            # Set minimum patch size for U2Net - needs to be large enough for at least 2 stages
            min_patch_size = 32 if len(spacing) == 3 else 64
            patch_size[axis_to_be_reduced] = max(min_patch_size, int(patch_size[axis_to_be_reduced] * reduction_factor))
            
            # Ensure divisibility requirements are met
            if len(shape_must_be_divisible_by) > axis_to_be_reduced:
                divisor = shape_must_be_divisible_by[axis_to_be_reduced]
                patch_size[axis_to_be_reduced] = max(divisor, 
                                                   (patch_size[axis_to_be_reduced] // divisor) * divisor)

            # Recompute topology with new patch size
            network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, \
            shape_must_be_divisible_by = _u2net_optimized_topology(spacing, patch_size,
                                                                   self.UNet_featuremap_min_edge_length,
                                                                   max_stages_for_dim)
            
            num_stages = len(pool_op_kernel_sizes)
            
            # U2Net requires minimum 2 stages for RSUDecoder to work properly
            num_stages = max(num_stages, 2)
            
            # Ensure we have enough depth values, extend with last value if needed
            depth_per_stage = self.depth_per_stage[:num_stages]
            if len(depth_per_stage) < num_stages:
                # Extend with the last available depth value
                last_depth = self.depth_per_stage[-1] if self.depth_per_stage else 4
                depth_per_stage.extend([last_depth] * (num_stages - len(depth_per_stage)))
            
            architecture_kwargs['arch_kwargs'].update({
                'n_stages': num_stages,
                'kernel_sizes': conv_kernel_sizes,
                'strides': pool_op_kernel_sizes,
                'features_per_stage': _features_per_stage(num_stages, max_num_features),
                'depth_per_stage': depth_per_stage,
            })
            
            # Recalculate estimate with RSU overhead
            cache_key = _keygen(patch_size, pool_op_kernel_sizes, depth_per_stage)
            if cache_key in _cache:
                base_estimate = _cache[cache_key]
            else:
                base_estimate = self.static_estimate_VRAM_usage(patch_size,
                                                               num_input_channels,
                                                               len(self.dataset_json['labels'].keys()),
                                                               architecture_kwargs['network_class_name'],
                                                               architecture_kwargs['arch_kwargs'],
                                                               architecture_kwargs['_kw_requires_import'])
            
            # Recalculate RSU overhead for new configuration
            rsu_overhead = 1.0
            features_per_stage = _features_per_stage(num_stages, max_num_features)
            for stage_idx, (depth, features) in enumerate(zip(depth_per_stage, features_per_stage)):
                stage_overhead = _estimate_rsu_memory_overhead(depth, features, patch_size)
                rsu_overhead = max(rsu_overhead, stage_overhead)
            
            estimate = base_estimate * rsu_overhead
            _cache[cache_key] = estimate

        # Determine batch size with U2Net considerations
        ref_bs = self.UNet_reference_val_corresp_bs_2d if len(spacing) == 2 else self.UNet_reference_val_corresp_bs_3d
        
        # U2Net typically requires smaller batch sizes due to complexity
        u2net_batch_factor = 0.75  # Reduce batch size by 25%
        batch_size = max(1, round((reference / estimate) * ref_bs * u2net_batch_factor))

        # Cap the batch size to cover at most 5% of the entire dataset
        bs_corresponding_to_5_percent = round(
            approximate_n_voxels_dataset * self.max_dataset_covered / np.prod(patch_size, dtype=np.float64))
        batch_size = max(min(batch_size, bs_corresponding_to_5_percent), self.UNet_min_batch_size)

        # Get resampling and normalization configurations
        resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs = self.determine_resampling()
        resampling_softmax, resampling_softmax_kwargs = self.determine_segmentation_softmax_export_fn()
        normalization_schemes, mask_is_used_for_norm = \
            self.determine_normalization_scheme_and_whether_mask_is_used_for_norm()

        plan = {
            'data_identifier': data_identifier,
            'preprocessor_name': self.preprocessor_name,
            'batch_size': batch_size,
            'patch_size': patch_size,
            'median_image_size_in_voxels': median_shape,
            'spacing': spacing,
            'normalization_schemes': normalization_schemes,
            'use_mask_for_norm': mask_is_used_for_norm,
            'resampling_fn_data': resampling_data.__name__,
            'resampling_fn_seg': resampling_seg.__name__,
            'resampling_fn_data_kwargs': resampling_data_kwargs,
            'resampling_fn_seg_kwargs': resampling_seg_kwargs,
            'resampling_fn_probabilities': resampling_softmax.__name__,
            'resampling_fn_probabilities_kwargs': resampling_softmax_kwargs,
            'architecture': architecture_kwargs
        }
        return plan





