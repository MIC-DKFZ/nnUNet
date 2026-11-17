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

Distributed training (DDP) is also available. For a single node, use `-num_gpus`:

```bash
nnUNetv2_train DATASET_NAME_OR_ID 2d 0 --npz -num_gpus X
```

For multi-node training, or to use PyTorch's standard launcher, run via `torchrun` (do not pass `-num_gpus` in that case). See [Multi-GPU training](../multi_gpu_training.md) for the full guide covering both options.

## Faster training: progressive growing of patch size (PGPS)

The PGPS trainers train with the progressive patch size schedule of [Progressive Growing of Patch Size: Curriculum
Learning for Accelerated and Improved Medical Image Segmentation](https://arxiv.org/abs/2510.23241) (PGPS-Efficiency).
Training starts with the smallest patch the network accepts (one unit of the encoder's total stride per axis), grows one
axis by one unit at a time and gives every patch size the same number of epochs, so the last part of the training uses the
configured patch size. Batch size, iterations per epoch, the learning rate schedule and inference are unchanged.

```bash
nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres 0 -tr nnUNetTrainer_PGPS
```

| trainer | epochs | compute |
|---|---|---|
| `nnUNetTrainer_PGPS` | 1000 | about 0.4x that of `nnUNetTrainer` (1000 epochs) |
| `nnUNetTrainer_PGPS_2700epochs` | 2700 | about that of `nnUNetTrainer` |
| `nnUNetTrainer_DASegOrd0_NoMirroring_PGPS_2700epochs_SV` | 2700 | about that of `nnUNetTrainer_DASegOrd0_NoMirroring_SV` (1000 epochs) |
| `nnUNetTrainer_DASegOrd0_NoMirroring_PGPS_5400epochs_SV` | 5400 | about that of `nnUNetTrainer_DASegOrd0_NoMirroring_2000epochs_SV` |

The last two are for datasets with many classes (see `nnUNetTrainer_DASegOrd0_NoMirroring_SV` in
[Memory-efficient (streaming) inference](run-inference-streaming.md#streaming-final-validation-in-training)).

- `checkpoint_best` is only written in the last stage, because earlier stages validate on smaller patches.
- With `torch.compile` on a CUDA device, the stages with small patches train with CUDA graphs. Their epochs are 2-3x
  faster that way, with the same numerics.
- To give your own trainer PGPS, mix `PGPSEfficiencyMixin`
  (`nnunetv2/training/nnUNetTrainer/variants/patch_size/nnUNetTrainer_PGPS.py`) in first:
  `class MyTrainer_PGPS(PGPSEfficiencyMixin, MyTrainer)`.
- If you use PGPS, please cite the paper linked above.

## Output location

Training outputs are written under:

```text
nnUNet_results/DatasetXXX_Name/TRAINER__PLANS__CONFIGURATION/fold_X
```

Important artifacts include:

- `checkpoint_final.safetensors`
- `checkpoint_best.safetensors`
- `progress.png`
- `validation/summary.json`
- `validation/*.npz` if `--npz` was enabled

Models trained with older nnU-Net versions have `.pth` checkpoints instead; they are still read everywhere.

## Next steps

- [Find the best configuration](find-best-configuration.md)
- [Run inference](run-inference.md)

## Detailed legacy page

For the original combined workflow document, see [How to run nnU-Net on a new dataset](../how_to_use_nnunet.md).
