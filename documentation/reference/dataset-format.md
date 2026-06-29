# Dataset and Input Format Reference

This page is the concise reference for dataset layout and inference input naming in nnU-Net v2.

## Core rules

- Each dataset lives in `nnUNet_raw/DatasetXXX_Name`.
- Training images go in `imagesTr`.
- Training labels go in `labelsTr`.
- Optional test images go in `imagesTs`.
- Each dataset must include a `dataset.json`.

Example:

```text
nnUNet_raw/
└── Dataset123_MyDataset
    ├── dataset.json
    ├── imagesTr
    ├── imagesTs
    └── labelsTr
```

## Case naming

Each training case has a unique case identifier.

Image files use:

```text
{CASE_IDENTIFIER}_{XXXX}.{FILE_ENDING}
```

Segmentation files use:

```text
{CASE_IDENTIFIER}.{FILE_ENDING}
```

`XXXX` is the 4-digit channel identifier, for example `0000`, `0001`, and so on.

## Multi-channel inputs

- Each non-RGB input channel is stored in a separate file.
- All channels for a case must share geometry and be aligned.
- Channel order must be consistent across all cases.
- The same naming convention also applies at inference time.

## `dataset.json`

The most important fields are:

```json
{
  "channel_names": {
    "0": "T2",
    "1": "ADC"
  },
  "labels": {
    "background": 0,
    "PZ": 1,
    "TZ": 2
  },
  "numTraining": 32,
  "file_ending": ".nii.gz"
}
```

Notes:

- `labels` map from label name to integer.
- `channel_names` influence normalization behavior.
- `file_ending` defines both training and inference file format.
- `overwrite_image_reader_writer` is optional for selecting a specific reader/writer.

## Supported file formats

nnU-Net v2 supports multiple input file formats. Common built-in options include:

- `.nii.gz`, `.nrrd`, `.mha`
- `.png`, `.bmp`, `.tif`
- 3D TIFF with sidecar spacing JSON

Images and labels must use the same dataset-level format, and lossy formats such as `.jpg` are not suitable.

## Inference input format

Inference inputs must match the training dataset's file ending and channel naming.

Example for a two-channel case:

```text
input_folder/
├── case_001_0000.nii.gz
├── case_001_0001.nii.gz
├── case_002_0000.nii.gz
└── case_002_0001.nii.gz
```

Predictions are written as:

```text
case_001.nii.gz
case_002.nii.gz
```

## Migrations and conversions

- nnU-Net v1 datasets: use `nnUNetv2_convert_old_nnUNet_dataset`
- Medical Segmentation Decathlon datasets: see [Convert MSD datasets](../convert_msd_dataset.md)

## Detailed legacy pages

- [nnU-Net dataset format](../dataset_format.md)
- [Data format for inference](../dataset_format_inference.md)
