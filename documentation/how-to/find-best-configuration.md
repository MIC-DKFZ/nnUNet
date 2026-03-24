# Find the Best Configuration

This guide covers automatic comparison of trained nnU-Net configurations.

## Requirement

Validation probabilities must be available. Train with `--npz` or rerun validation with `--val --npz` before using this command.

## Command

```bash
nnUNetv2_find_best_configuration DATASET_NAME_OR_ID -c CONFIGURATIONS
```

Example:

```bash
nnUNetv2_find_best_configuration 123 -c 2d 3d_fullres 3d_lowres
```

## What it evaluates

- individual configurations
- pairwise ensembles, unless `--disable_ensembling` is used
- postprocessing based on connected-component removal

## Output

The command prints the recommended inference commands and also writes:

- `inference_instructions.txt`
- `inference_information.json`

Both are stored in `nnUNet_results/DatasetXXX_Name`.

## Next step

Continue to [Run inference](run-inference.md).
