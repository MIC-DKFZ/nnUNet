# Train Models

This guide covers training nnU-Net configurations on a prepared dataset.

## Training overview

nnU-Net can create several configurations depending on the dataset:

- `2d`
- `3d_fullres`
- `3d_lowres`
- `3d_cascade_fullres`

Not every dataset gets every configuration. Small datasets may not create the cascade.

Training is usually done as a 5-fold cross-validation so nnU-Net can compare configurations and optionally ensemble them later.

## Basic training command

```bash
nnUNetv2_train DATASET_NAME_OR_ID CONFIGURATION FOLD
```

Examples:

```bash
nnUNetv2_train DATASET_NAME_OR_ID 2d 0
nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres 0
nnUNetv2_train DATASET_NAME_OR_ID 3d_lowres 0
nnUNetv2_train DATASET_NAME_OR_ID 3d_cascade_fullres 0
```

For the cascade, `3d_lowres` must be trained before `3d_cascade_fullres`.

## Important flag for later model selection

If you plan to use `nnUNetv2_find_best_configuration`, train with `--npz`:

```bash
nnUNetv2_train DATASET_NAME_OR_ID CONFIGURATION FOLD --npz
```

This stores validation probabilities needed for automatic configuration comparison and ensembling.

If you already trained without `--npz`, you can rerun validation:

```bash
nnUNetv2_train DATASET_NAME_OR_ID CONFIGURATION FOLD --val --npz
```

## Device selection

Use `-device` to choose `cpu`, `cuda`, or `mps`.

For multi-GPU systems, select the GPU with `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres 0 --npz
```

## Recommended multi-GPU usage

If you have multiple GPUs, the preferred strategy is usually one training per GPU:

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train DATASET_NAME_OR_ID 2d 0 --npz
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train DATASET_NAME_OR_ID 2d 1 --npz
```

Distributed training is also available:

```bash
nnUNetv2_train DATASET_NAME_OR_ID 2d 0 --npz -num_gpus X
```

## Output location

Training outputs are written under:

```text
nnUNet_results/DatasetXXX_Name/TRAINER__PLANS__CONFIGURATION/fold_X
```

Important artifacts include:

- `checkpoint_final.pth`
- `checkpoint_best.pth`
- `progress.png`
- `validation/summary.json`
- `validation/*.npz` if `--npz` was enabled

## Next steps

- [Find the best configuration](find-best-configuration.md)
- [Run inference](run-inference.md)

## Detailed legacy page

For the original combined workflow document, see [How to run nnU-Net on a new dataset](../how_to_use_nnunet.md).
