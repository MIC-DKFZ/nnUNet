# Logging Reference

This page summarizes how training logging works in nnU-Net v2.

## Logging architecture

Training logs are routed through `MetaLogger`, which fans out to:

- `LocalLogger`: always enabled
- optional external loggers such as `WandbLogger`

The local logger is the source of truth for training curves and `progress.png`.

## What is logged by default

Per epoch, nnU-Net stores values such as:

- `mean_fg_dice`
- `ema_fg_dice`
- `dice_per_class_or_region`
- `train_losses`
- `val_losses`
- `lrs`
- epoch start and end timestamps

These values drive `progress.png` in the fold output folder.

## Resume behavior

When checkpoints are saved or resumed, local logging state is saved and restored too, so training curves remain continuous.

## Enable Weights & Biases

1. Install W&B:

```bash
pip install wandb
```

2. Set the environment variables:

```bash
export nnUNet_wandb_enabled=1
export nnUNet_wandb_project=nnunet
export nnUNet_wandb_mode=online
```

3. Run training normally:

```bash
nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres 0
```

`nnUNet_wandb_mode` can also be `offline`.

## Custom loggers

A custom logger must implement the minimal interface expected by `MetaLogger`:

- `update_config(self, config: dict)`
- `log(self, key, value, step: int)`
- `log_summary(self, key, value)`

If you add a brand-new logged key, make sure the local logger knows about it as well.

## Related pages

- [Train models](../how-to/train-models.md)
- [Installation and setup](../getting-started/installation-and-setup.md)

## Detailed legacy page

- [Logging in nnU-Net v2](../explanation_logging.md)
