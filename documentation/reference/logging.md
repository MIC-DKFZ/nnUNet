# Logging Reference

This page summarizes how training logging works in nnU-Net v2.

## Logging architecture

Training logs are routed through `MetaLogger`, which fans out to:

- `LocalLogger`: always enabled
- optional external loggers such as `WandbLogger` and `MlflowLogger`

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

## Enable MLflow

1. Install MLflow:

```bash
pip install mlflow
```

2. Set the environment variables:

```bash
export nnUNet_mlflow_enabled=1
export nnUNet_mlflow_experiment=nnunet
export nnUNet_mlflow_tracking_uri=mlruns    # or http://myserver:5000
export nnUNet_mlflow_system_metrics=0       # or 1 to measure system metrics -> requires: pip install psutil (+ pynvml for NVIDIA GPU metrics)
```

3. Run training normally:

```bash
nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres 0
```

`nnUNet_mlflow_tracking_uri` accepts any URI supported by MLflow: a local directory, `http://...` for a remote tracking server, or a `databricks` URI.

**Behaviour notes:**

- If `mlflow` is not installed but the env var is set, a warning is printed and training continues — it does not crash.
- For a remote server, use `mlflow server --host 0.0.0.0 --port 5000` on the host — `mlflow ui` is read-only and will reject writes.
- Each run is named `DatasetXXX_Name__configuration__fold_X` automatically.
- Runs are tagged with `dataset`, `configuration`, `fold`, and `trainer` for filtering in the UI.
- The run ID is saved in `fold_x/mlflow/run_id.txt` and reused on resume (`--c`), so all checkpoint restarts for a single model append to the same run.
- Runs are ended cleanly via `atexit` even on crash, so they do not remain stuck in `RUNNING` state.

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
