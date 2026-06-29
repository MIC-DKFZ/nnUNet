# How nnU-Net Works

This page explains the core idea behind nnU-Net and what problem it is designed to solve.

## What nnU-Net is

nnU-Net is a semantic segmentation framework that automatically adapts its pipeline to a dataset. Instead of asking the user to manually tune architecture, preprocessing, and training details for each dataset, nnU-Net analyzes the dataset and configures a strong U-Net-based baseline automatically.

## Why this matters

Segmentation datasets vary widely:

- 2D vs 3D data
- modalities and channel counts
- image sizes and voxel spacings
- anisotropy
- class imbalance
- target-structure properties

Because these properties interact, manual pipeline design is easy to get wrong and hard to scale. nnU-Net addresses that by turning dataset properties into a reproducible configuration process.

## Scope

nnU-Net is built for supervised semantic segmentation. It handles 2D and 3D data and supports many file formats, but it expects:

- labeled training data
- consistent channel definitions
- images that can be processed during preprocessing and postprocessing on the available system

It is especially strong for training-from-scratch settings, biomedical datasets, and non-standard imaging setups.

## Main configurations

Depending on the dataset, nnU-Net may generate:

- `2d`
- `3d_fullres`
- `3d_lowres`
- `3d_cascade_fullres`

The cascade is only created when the dataset characteristics justify it.

## How adaptation works

nnU-Net uses three kinds of decisions:

1. Fixed parameters: stable defaults that are not adapted per dataset
2. Rule-based parameters: heuristics derived from the dataset fingerprint
3. Empirical parameters: comparisons such as choosing the best configuration or postprocessing

## Dataset fingerprint

The dataset fingerprint summarizes properties such as:

- image sizes
- spacings
- intensity information

nnU-Net uses that information to determine preprocessing, target spacing, patch size, and network topology.

## What happens in practice

At a high level, the workflow is:

1. Prepare the dataset in nnU-Net format
2. Extract the fingerprint, create plans, and preprocess
3. Train one or more configurations
4. Compare configurations and determine postprocessing
5. Run inference with the selected setup

## Where to go next

- Workflow: [How-to Guides](../how-to/README.md)
- Formats and commands: [Reference](../reference/README.md)
- Deeper customization: [Extending nnU-Net](../extending_nnunet.md)
- Legacy conceptual overview: [Root README](../../readme.md)
