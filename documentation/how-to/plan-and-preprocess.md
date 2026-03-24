# Plan and Preprocess

This guide covers dataset fingerprint extraction, experiment planning, and preprocessing.

## Recommended command

For a new dataset, use:

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

`DATASET_ID` is the numeric dataset identifier. `--verify_dataset_integrity` is recommended the first time you run the command.

## What this does

The command performs three steps:

1. Extract a dataset fingerprint
2. Create one or more nnU-Net configurations
3. Preprocess the data for those configurations

The output is written into `nnUNet_preprocessed/DatasetXXX_Name`.

## Useful options

- Use `--no_pbar` in non-interactive environments.
- Use `-d 1 2 3` to process multiple datasets.
- Use `-c 3d_fullres` if you already know which configuration you want.
- Use `-h` to inspect all options.

## Split commands

If you need more control, you can run the steps individually:

```bash
nnUNetv2_extract_fingerprint -d DATASET_ID
nnUNetv2_plan_experiment -d DATASET_ID
nnUNetv2_preprocess -d DATASET_ID
```

## What to inspect afterward

After preprocessing, the dataset folder in `nnUNet_preprocessed` contains:

- `dataset_fingerprint.json`
- `nnUNetPlans.json`
- preprocessed data folders for the created configurations

## Next step

Continue to [Train models](train-models.md).
