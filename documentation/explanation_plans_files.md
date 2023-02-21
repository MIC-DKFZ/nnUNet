# Modifying the nnU-Net Configurations

nnU-Net provides unprecedented out-of-the-box segmentation performance for essentially any dataset we have evaluated 
it on. That said, there is always room for improvements. A fool-proof strategy for sequeezing our the last bit of 
performance is to start with the default nnU-Net, and then further tune it manually to a concrete dataset at hand.
**This guide is about changes to the nnU-Net configuration you can make via the plans files. It does not cover code 
extensions of nnU-Net. For that, take a look [here](extending_nnunet.md)**

In nnU-Net V2, plans files are SO MUCH MORE powerful than they were in V1. There are a lot more knobs that you can 
turn without resorting to hacky solutions or even having to touch the nnU-Net code at all! And as an added bonus: 
plans files are now also .json files and no longer require users to fiddle with picke. Just open them in your text 
editor of choice!

If overwhelmed, look at our [Examples](#examples)!

# plans.json structure

Plans have global and local settings. Global settings are applied to all configurations in that plans file while 
local settings are attached to a specific configuration.

## Global settings

- `foreground_intensity_properties_by_modality`: Intensity statistics of the foreground regions (all labels except 
background and ignore label), computed over all training cases. Used by [CT normalization scheme](explanation_normalization.md).
- `image_reader_writer`: Name of the image reader/wrriter class that should be used with this dataset. You might want 
to change this if, for example, you would like to run inference with a files that have a different file format. The 
class that is names here must be located in nnunetv2.imageio!
- `label_manager`: The name of the class that does label handling. Take a look at 
nnunetv2.utilities.label_handling.LabelManager to see what it does. If you decide to change it, place your version 
in nnunetv2.utilities.label_handling!
- `transpose_forward`: nnU-Net transposes the input data so that the axes with the highest resolution (lowest spacing) 
come last. This is because the 2D U-Net operates on the trailing dimensions (more efficient slicing due to internal 
memory layout of arrays). Future work might move this to setting to affect only individual configurations. 
- transpose_backward is what numpy.transpose gets as new axis ordering.
- `transpose_backward`: the axis ordering that inverts "transpose_forward"
- \[`original_median_shape_after_transp`\]: just here for your information
- \[`original_median_spacing_after_transp`\]: just here for your information
- \[`plans_name`\]: do not change. Used internally
- \[`experiment_planner_used`\]: just here as metadata so that we know what planner originally generated this file
- \[`dataset_name`\]: do not change. This is the dataset these plans are intended for

## Local settings
Plans also have a `configurations` key in which the actual configurations are stored. `configurations` are again a 
dictionary, where the keys are the configuration names and the values are the local settings for each configuration.

To better understand the components describing the network topology in our plans files, please read section 6.2 
in the [supplementary information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41592-020-01008-z/MediaObjects/41592_2020_1008_MOESM1_ESM.pdf) 
(page 13) of our paper!

Local settings:
- `spacing`: the target spacing used in this configuration
- `patch_size`: the patch size used for training this configuration
- `data_identifier`: the preprocessed data for this configuration will be saved in
  nnUNet_preprocessed/DATASET_NAME/_data_identifier_. If you add a new configuration, remember to set a unique
  data_identifier in order to not create conflicts with other configurations (unless you plan to reuse the data from
  another configuration, for example as is done in the cascade)
- `batch_size`: batch size used for training
- `batch_dice`: whether to use batch dice (pretend all samples in the batch are one image, compute dice loss over that)
or not (each sample in the batch is a separate image, compute dice loss for each sample and average over samples)
- `preprocessor_name`: Name of the preprocessor class used for running preprocessing. Class must be located in 
nnunetv2.preprocessing.preprocessors
- `use_mask_for_norm`: whether to use the nonzero mask for normalization or not (relevant for BraTS and the like, 
probably False for all other datasets). Interacts with ImageNormalization class
- `normalization_schemes`: mapping of channel identifier to ImageNormalization class name. ImageNormalization 
classes must be located in nnunetv2.preprocessing.normalization. Also see [here](explanation_normalization.md)
- `resampling_fn_data`: name of resampling function to be used for resizing image data. resampling function must be 
callable(data, current_spacing, new_spacing, **kwargs). It must be located in nnunetv2.preprocessing.resampling
- `resampling_fn_data_kwargs`: kwargs for resampling_fn_data
- `resampling_fn_probabilities`: name of resampling function to be used for resizing predicted class probabilities/logits. 
resampling function must be callable(data, current_spacing, new_spacing, **kwargs). It must be located in 
nnunetv2.preprocessing.resampling
- `resampling_fn_probabilities_kwargs`: kwargs for resampling_fn_probabilities
- `resampling_fn_seg`: name of resampling function to be used for resizing segmentation maps (integer: 0, 1, 2, 3, etc). 
resampling function must be callable(data, current_spacing, new_spacing, **kwargs). It must be located in 
nnunetv2.preprocessing.resampling
- `resampling_fn_seg_kwargs`: kwargs for resampling_fn_seg
- `UNet_class_name`: UNet class name, can be used to integrate custom dynamic architectures
- `UNet_base_num_features`: The number of starting features for the UNet architecture. Default is 32. Default: Features
are doubled with each downsampling 
- `unet_max_num_features`: Maximum number of features (default: capped at 320 for 3D and 512 for 2d). Purpose is to 
prevent parameters from exploding too much. 
- `conv_kernel_sizes`: the convolutional kernel sizes used by nnU-Net in each stage of the encoder. The decoder 
  mirrors the encoder and is therefore not explicitly listed here! The list is as long as `n_conv_per_stage_encoder` has 
  entries
- `n_conv_per_stage_encoder`: number of convolutions used per stage (=at a feature map resolution in the encoder) in the encoder. 
  Default is 2. The list has as many entries as the encoder has stages
- `n_conv_per_stage_decoder`: number of convolutions used per stage in the decoder. Also see `n_conv_per_stage_encoder`
- `num_pool_per_axis`: number of times each of the spatial axes is pooled in the network. Needed to know how to pad 
  image sizes during inference (num_pool = 5 means input must be divisible by 2**5=32)
- `pool_op_kernel_sizes`: the pooling kernel sizes (and at the same time strides) for each stage of the encoder
- \[`median_image_size_in_voxels`\]: the median size of the images of the training set at the current target spacing. 
Do not modify this as this is not used. It is just here for your information.

Special local settings:
- `inherits_from`: configurations can inherit from each other. This makes it easy to add new configurations that only
differ in a few local settings from another. If using this, remember to set a new `data_identifier` (if needed)!
- `previous_stage`: if this configuration is part of a cascade, we need to know what the previous stage (for example 
the low resolution configuration) was. This needs to be specified here.
- `next_stage`: if this configuration is part of a cascade, we need to know what possible subsequent stages are! This 
is because we need to export predictions in the correct spacing when running the validation. `next_stage` can either 
be a string or a list of strings

# Examples

## Increasing the batch size for large datasets
If your dataset is large the training can benefit from larger batch_sizes. To do this, simply create a new 
configuration in the `configurations` dict

    "configurations": {
      "3d_fullres_bs40": {
        "inherits_from": "3d_fullres",
        "batch_size": 40
      }
    }

No need to change the data_identifier. `3d_fullres_bs40` will just use the preprocessed data from `3d_fullres`.
No need to rerun `nnUNetv2_preprocess` because we can use already existing data (if available) from `3d_fullres`.

## Using custom preprocessors
If you would like to use a different preprocessor class then this can be specified as follows:

    "configurations": {
      "3d_fullres_my_preprocesor": {
        "inherits_from": "3d_fullres",
        "preprocessor_name": MY_PREPROCESSOR,
        "data_identifier": "3d_fullres_my_preprocesor"
      }
    }

You need to run preprocessing for this new configuration: 
`nnUNetv2_preprocess -d DATASET_ID -c 3d_fullres_my_preprocesor` because it changes the preprocessing. Remember to 
set a unique `data_identifier` whenever you make modifications to the preprocessed data!

## Change target spacing

    "configurations": {
      "3d_fullres_my_spacing": {
        "inherits_from": "3d_fullres",
        "spacing": [X, Y, Z],
        "data_identifier": "3d_fullres_my_spacing"
      }
    }

You need to run preprocessing for this new configuration: 
`nnUNetv2_preprocess -d DATASET_ID -c 3d_fullres_my_spacing` because it changes the preprocessing. Remember to 
set a unique `data_identifier` whenever you make modifications to the preprocessed data!

## Adding a cascade to a dataset where it does not exist
Hippocampus is small. It doesn't have a cascade. It also doesn't really make sense to add a cascade here but hey for 
the sake of demonstration we can do that.
We change the following things here:

- spacing: The lowres stage should operate at a lower resolution
- we modify the `median_image_size_in_voxels` entry as a guide for what original image sizes we deal with
- we set some patch size that is inspired by `median_image_size_in_voxels`
- we need to remember that the patch size must be divisible by 2**num_pool in each axis!
- network parameters such as kernel sizes, pooling operations are changed accordingly
- we need to specify the name of the next stage
- we need to add the highres stage

This is how this would look like (comparisons with 3d_fullres given as reference):

    "configurations": {
      "3d_lowres": {
        "inherits_from": "3d_fullres",
        "data_identifier": "3d_lowres"
        "spacing": [2.0, 2.0, 2.0], # from [1.0, 1.0, 1.0] in 3d_fullres
        "median_image_size_in_voxels": [18, 25, 18], # from [36, 50, 35]
        "patch_size": [20, 28, 20], # from [40, 56, 40]
        "n_conv_per_stage_encoder": [2, 2, 2], # one less entry than 3d_fullres ([2, 2, 2, 2])
        "n_conv_per_stage_decoder": [2, 2], # one less entry than 3d_fullres
        "num_pool_per_axis": [2, 2, 2], # one less pooling than 3d_fullres in each dimension (3d_fullres: [3, 3, 3])
        "pool_op_kernel_sizes": [[1, 1, 1], [2, 2, 2], [2, 2, 2]], # one less [2, 2, 2]
        "conv_kernel_sizes": [[3, 3, 3], [3, 3, 3], [3, 3, 3]], # one less [3, 3, 3]
        "next_stage": "3d_cascade_fullres" # name of the next stage in the cascade
      },
      "3d_cascade_fullres": { # does not need a data_identifier because we can use the data of 3d_fullres
        "inherits_from": "3d_fullres",
        "previous_stage": "3d_lowres" # name of the previous stage
      }
    }

To better understand the components describing the network topology in our plans files, please read section 6.2 
in the [supplementary information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41592-020-01008-z/MediaObjects/41592_2020_1008_MOESM1_ESM.pdf) 
(page 13) of our paper!