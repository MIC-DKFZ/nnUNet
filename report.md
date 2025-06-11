# Report - Not final

@Monday Jun 9th
- Preprocess data + figure out planner function from nnUNetv2
- Trained baseline model

@Tuesday Jun 10th
- Segmentation evaluation on the baseline model
- Train and evaluate new ResNetEncoder

## Data
Ignoring the test data for now.
- Train = 252
- Val = 72

## Baseline
### Configuration name: 3d_fullres
```json
{
    'data_identifier': 'nnUNetPlans_3d_fullres',
    'preprocessor_name': 'DefaultPreprocessor',
    'batch_size': 3,
    'patch_size': [64, 128, 192],
    'median_image_size_in_voxels': [59.0, 117.0, 180.5],
    'spacing': [2.0, 0.732421875, 0.732421875],

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
        'network_class_name': 'dynamic_network_architectures.architectures.unet.PlainConvUNet',
        'arch_kwargs': {
            'n_stages': 6,
            'features_per_stage': [32, 64, 128, 256, 320, 320],
            'conv_op': 'torch.nn.modules.conv.Conv3d',
            'kernel_sizes': [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            'strides': [[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            'n_conv_per_stage': [2, 2, 2, 2, 2, 2],
            'n_conv_per_stage_decoder': [2, 2, 2, 2, 2],
            'conv_bias': True,
            'norm_op': 'torch.nn.modules.instancenorm.InstanceNorm3d',
            'norm_op_kwargs': {
                'eps': 1e-05,
                'affine': True
            },
            'dropout_op': None,
            'dropout_op_kwargs': None,
            'nonlin': 'torch.nn.LeakyReLU',
            'nonlin_kwargs': {'inplace': True}
    },
    '_kw_requires_import': ['conv_op', 'norm_op', 'dropout_op', 'nonlin']},
    'batch_dice': False
}
```
### plan.json settings:
```json
{
    'dataset_name': 'Dataset001_PancreasSegClassification',
    'plans_name': 'nnUNetPlans',
    'original_median_spacing_after_transp': [2.0, 0.732421875, 0.732421875],
    'original_median_shape_after_transp': [64, 119, 179],
    'image_reader_writer': 'SimpleITKIO',
    'transpose_forward': [0, 1, 2],
    'transpose_backward': [0, 1, 2],
    'experiment_planner_used': 'ExperimentPlanner',
    'label_manager': 'LabelManager',
    'foreground_intensity_properties_per_channel': {
        '0': {
            'max': 1929.0,
            'mean': 74.06402587890625,
            'median': 77.98674774169922,
            'min': -406.9988098144531,
            'percentile_00_5': -56.0,
            'percentile_99_5': 179.99807739257812,
            'std': 44.359100341796875
        }
    }
}
```