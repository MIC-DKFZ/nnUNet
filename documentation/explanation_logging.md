# Logging in nnU-Net v2

## Introduction

Logging in nnU-Net is intentionally simple and centralized in
`nnunetv2/training/logging/nnunet_logger.py`.

The trainer talks to one object, `MetaLogger`, and `MetaLogger` fans out logs to:

- `LocalLogger` (always enabled): the source of truth for training curves, checkpoint logging state, and `progress.png`
- optional external loggers (currently `WandbLogger`)

This keeps training code clean while still allowing external tracking backends.

## Default behaviour

Without any setup, nnU-Net uses only `LocalLogger`.

Per epoch, it stores:

- `mean_fg_dice` and `ema_fg_dice` (EMA is computed automatically)
- `dice_per_class_or_region`
- `train_losses`, `val_losses`
- `lrs`
- `epoch_start_timestamps`, `epoch_end_timestamps`

From these values, `progress.png` is updated in the fold output folder.
On checkpoint save/load, the local logging state is also saved/restored, so curves continue correctly after resume.

## How to enable W&B

1. Install W&B:

```bash
pip install wandb
```

2. Enable the backend via environment variables:

```bash
export nnUNet_wandb_enabled=1
export nnUNet_wandb_project=nnunet
export nnUNet_wandb_mode=online   # or offline
```

3. Run training normally:

```bash
nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres 0
```

Notes:

- `nnUNet_wandb_enabled` accepts `0/1` and `false/true` (case-insensitive). Other values raise an error.
- When resuming (`--c`), W&B resume metadata in `fold_x/wandb/latest-run` is reused and duplicate older steps are skipped.

## How to integrate a custom logger

Add a new logger class with the same minimal interface used by `MetaLogger`:

- `update_config(self, config: dict)`
- `log(self, key, value, step: int)`
- `log_summary(self, key, value)`

Example skeleton:

```python
class MyLogger:
    def __init__(self, output_folder, resume):
        self.output_folder = output_folder
        self.resume = resume

    def update_config(self, config: dict):
        ...

    def log(self, key, value, step: int):
        ...

    def log_summary(self, key, value):
        ...
```

Then register it in `MetaLogger.__init__` (for example behind an env var switch), similar to how `WandbLogger` is added.

Important integration detail:

- `MetaLogger.log(...)` always writes to `LocalLogger` first.
- If you introduce a brand-new per-epoch key, also add that key to `LocalLogger.my_fantastic_logging`, otherwise the local assertion will fail.
