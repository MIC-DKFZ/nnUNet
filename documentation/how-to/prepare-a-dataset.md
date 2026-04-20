# Prepare a Dataset

This guide covers the dataset preparation step before experiment planning and training.

## Required format

nnU-Net expects datasets in the nnU-Net dataset format. Start with the concise reference here:

- [Dataset and input format reference](../reference/dataset-format.md)

The key points are:

- each dataset lives in `nnUNet_raw/DatasetXXX_Name`
- training images go into `imagesTr`
- training labels go into `labelsTr`
- optional test images go into `imagesTs`
- `dataset.json` describes modalities and labels

## Existing data sources

If you already have data in a different layout:

- Medical Segmentation Decathlon: [Convert MSD datasets](../convert_msd_dataset.md)
- nnU-Net v1 datasets: use `nnUNetv2_convert_old_nnUNet_dataset`

## Input formats

nnU-Net v2 supports multiple file formats. The exact supported formats and image I/O details are documented in:

- [nnU-Net dataset format](../dataset_format.md#supported-file-formats)

## Inference inputs

Inference input folders follow the training dataset's naming and file-ending conventions. See:

- [Dataset and input format reference](../reference/dataset-format.md)

## Next step

Once your dataset is in place, continue to [Plan and preprocess](plan-and-preprocess.md).
