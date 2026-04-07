# Checkpoint format

nnU-Net checkpoints are stored in a self-describing safetensors layout. This
page explains what gets written, why, and how to interoperate with the legacy
`.pth` format.

## Files written per checkpoint

For a checkpoint named `checkpoint_final.pth`, the trainer writes:

| File | Contents |
|---|---|
| `checkpoint_final.safetensors` | Network weights only. Inference artifact. Has a `weight_layout=torch_ncdhw` entry in the safetensors metadata header. |
| `checkpoint_final.trainer_state.safetensors` | Optimizer and grad_scaler tensors, flattened with dotted keys (`optimizer.state.<param_id>.<buffer_name>`). |
| `checkpoint_final.json` | Everything Python: `init_args`, `trainer_name`, `inference_allowed_mirroring_axes`, `current_epoch`, `_best_ema`, `logging`, plus skeletons for the optimizer and grad_scaler dicts with tensor placeholders. |
| `checkpoint_final.pth` | *(optional)* Legacy PyTorch pickle. Written by default; opt out with `nnUNet_save_pth=0`. |

Inference loaders need only the first file plus the JSON sidecar. Resume-from-checkpoint needs all three of the new files (or just the `.pth`).

## Why this layout

- **Safety.** safetensors does not execute pickled Python on load, so distributing trained models no longer requires the consumer to trust your `.pth`. This matters more than it used to: **CVE-2025-32434** (CVSS 9.3, fixed in PyTorch 2.6.0) demonstrated a remote-code-execution path through `torch.load` that bypassed even `weights_only=True`. nnU-Net checkpoints are dicts containing optimizer state and arbitrary `init_args`, so every legacy `.pth` load goes through the unsafe `weights_only=False` path. The safetensors format sidesteps the pickle attack surface entirely. Anyone distributing trained models to third parties should set `nnUNet_save_pth=0` and ship only the safetensors layout.
- **Speed.** safetensors loads faster than `torch.load`, especially for the inference path that only needs the network weights.
- **Portability.** `.pth` files are PyTorch pickle archives — non-PyTorch frameworks (MLX, JAX, ONNX runtimes, native C++ inference) genuinely cannot consume them without taking a hard dependency on `torch` purely to deserialize. safetensors is framework-agnostic by design: the MLX inference port for Apple Silicon, for example, reads the weights file directly with no `torch` import at runtime. The `weight_layout` metadata entry tells non-PyTorch loaders how the conv tensor axes are ordered so they can transpose deterministically.
- **Round-trip fidelity.** The recursive flatten/merge in `nnunetv2/utilities/checkpoint_io.py` walks arbitrary nested optimizer state, so warmed Adam buffers (`exp_avg`, `exp_avg_sq`, `step`) and grad_scaler scale tensors come back byte-equal.

## Loading

`nnunetv2.utilities.checkpoint_io.load_checkpoint(path, map_location, load_optimizer=True)` is the single entry point. It:

1. Looks for `<base>.safetensors` and prefers the new layout if present.
2. Falls back to `torch.load` on `<base>.pth` for legacy checkpoints.
3. Returns a dict with the same keys the legacy `torch.load` output had, so existing code is unchanged.

Inference callers should pass `load_optimizer=False` to skip the trainer-state file (they don't need it, and distribution-only artifacts may not ship it).

## Disabling the legacy `.pth` write

`.pth` is written by default for backwards compatibility with downstream tooling that calls `torch.load` directly. To stop writing it:

```bash
export nnUNet_save_pth=0
```

This roughly halves the disk footprint per checkpoint. The new format is canonical — disabling `.pth` does not lose any information.

The plan is to flip this default to off in a future release, then remove the writer entirely the release after. The `.pth` *reader* will be kept indefinitely so existing model zoos keep working.

## Converting existing `.pth` checkpoints

```bash
nnUNetv2_convert_to_safetensors path/to/checkpoint_final.pth
nnUNetv2_convert_to_safetensors path/to/model_dir --recursive
nnUNetv2_convert_to_safetensors path/to/checkpoint_final.pth --delete-pth
```

The converter loads the `.pth` once with `torch.load`, writes the new layout, and (with `--delete-pth`) removes the legacy file.

## Weight layout metadata

The safetensors weights file carries a `weight_layout` entry in its header. Today it is always `torch_ncdhw`, meaning Conv3d weights are stored as `(C_out, C_in, kD, kH, kW)`. Non-PyTorch loaders that need a different memory layout (e.g. MLX uses `(C_out, kD, kH, kW, C_in)`) should read this tag and transpose accordingly. Loaders that don't recognize the tag can default to `torch_ncdhw` — that is the layout for every checkpoint produced by every version of nnU-Net to date.

## Implementation notes

- The trainer's `save_checkpoint` calls `save_checkpoint(...)` from `nnunetv2.utilities.checkpoint_io` unconditionally. The optional `.pth` write is gated on `nnUNet_save_pth` and is the only path that still uses `torch.save`.
- `safetensors>=0.4.0` is a hard dependency. It is a small Rust extension and installs as a wheel on every supported platform.
- The optimizer state walker stringifies dict keys for JSON storage. Param ids in `optimizer_state["state"]` are restored to integers at load time. Other integer-keyed dicts in custom optimizers would need similar handling.
