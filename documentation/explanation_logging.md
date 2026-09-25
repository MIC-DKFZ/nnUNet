# Logging in nnU-Net v2

Prefer the migrated reference page for the recommended path:

- [Logging reference](reference/logging.md)

## Introduction

Logging in nnU-Net is intentionally simple and centralized in
`nnunetv2/training/logging/nnunet_logger.py`.

The trainer talks to one object, `MetaLogger`, and `MetaLogger` fans out logs to:

- `LocalLogger` (always enabled): the source of truth for training curves, checkpoint logging state, and `progress.png`
- optional external loggers (currently `WandbLogger` and `MlflowLogger`)

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

## How to enable MLflow

1. Install MLflow:

```bash
pip install mlflow
```

2. Enable the backend via environment variables:

```bash
export nnUNet_mlflow_enabled=1
export nnUNet_mlflow_experiment=nnunet
export nnUNet_mlflow_tracking_uri=mlruns    # or a remote URI such as http://myserver:5000
export nnUNet_mlflow_system_metrics=0       # or 1 to measure system metrics -> requires: pip install psutil (+ pynvml for NVIDIA GPU metrics)
```

3. Run training normally:

```bash
nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres 0
```

Notes:

- `nnUNet_mlflow_enabled` accepts `0/1` and `false/true` (case-insensitive). Other values raise an error.
- If `mlflow` is not installed but the env var is set, a warning is printed and training continues with local logging only — it does not crash.
- The run ID is persisted in `fold_x/mlflow/run_id.txt`. When resuming (`--c`), that ID is reused so metrics append to the same run regardless of how many checkpoint restarts a single model training requires.
- `nnUNet_mlflow_tracking_uri` accepts any URI supported by MLflow: a local directory, `http://...` for a remote tracking server, or a `databricks` URI.
- For a **remote server**, start it with `mlflow server` (not `mlflow ui` — the UI is read-only and cannot accept metric writes): `mlflow server --backend-store-uri /path/to/mlruns --host 0.0.0.0 --port 5000`
- Each run is automatically named `DatasetXXX_Name__configuration__fold_X` and tagged with `dataset`, `configuration`, `fold`, and `trainer` for easy filtering in the MLflow UI.
- The run is ended cleanly via `atexit` even if training crashes, so runs do not stay stuck in `RUNNING` state.

### Optional: system metrics

To also log CPU usage, memory, and GPU utilisation per epoch:

```bash
export nnUNet_mlflow_system_metrics=1
```

This calls `mlflow.enable_system_metrics_logging()` and requires `psutil` (`pip install psutil`). GPU metrics additionally require `pynvml` for NVIDIA GPUs.

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
