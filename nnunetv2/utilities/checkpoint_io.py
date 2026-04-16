"""
Checkpoint I/O with safetensors as the canonical format.

Layout for a checkpoint named ``checkpoint_final.pth``:

  checkpoint_final.safetensors                -- network weights only
                                                 (clean inference artifact;
                                                  metadata: weight_layout=torch_ncdhw)
  checkpoint_final.trainer_state.safetensors  -- optimizer + grad_scaler tensors
                                                 (flattened, dotted keys)
  checkpoint_final.json                       -- everything Python: init_args,
                                                 trainer_name, mirroring axes,
                                                 epoch, _best_ema, logging, plus
                                                 skeletons for optimizer and
                                                 grad_scaler with tensor placeholders
  checkpoint_final.pth                        -- legacy PyTorch format (optional)

Loading prefers the safetensors layout, falls back to .pth. Inference paths
need only the first file plus the JSON metadata; resume needs all three.

Tensor placeholders in the JSON skeleton are dicts of the form
``{"__tensor__": "<flat_key>"}`` and are resolved against the trainer-state
safetensors file at load time.

The on-disk weight layout is recorded in the safetensors metadata header so
that non-PyTorch loaders (e.g. the MLX inference port) can transpose
deterministically rather than relying on key-pattern heuristics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import torch
from safetensors.torch import load_file, save_file

WEIGHT_LAYOUT_TORCH = "torch_ncdhw"
FORMAT_VERSION = 1

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _weights_path(pth: PathLike) -> Path:
    return Path(pth).with_suffix(".safetensors")


def _inference_meta_json_path(pth: PathLike) -> Path:
    """Inference-side metadata: init_args, trainer_name, mirroring axes.

    Pairs with the network weights .safetensors file. Both together are the
    full distribution artifact for a pretrained model.
    """
    return Path(pth).with_suffix(".json")


def _trainer_state_tensors_path(pth: PathLike) -> Path:
    p = Path(pth)
    return p.with_name(p.stem + ".trainer_state.safetensors")


def _trainer_state_json_path(pth: PathLike) -> Path:
    """Trainer Python state: epoch, _best_ema, logging history, and the
    optimizer/grad_scaler skeletons with tensor placeholders.

    Pairs with the trainer_state .safetensors file. Both together are
    needed to resume training; neither is needed for inference.
    """
    p = Path(pth)
    return p.with_name(p.stem + ".trainer_state.json")


# ---------------------------------------------------------------------------
# Recursive tensor / Python split
# ---------------------------------------------------------------------------

def _split(obj: Any, prefix: str, tensors: dict) -> Any:
    """Walk obj, extracting tensors into ``tensors`` and replacing them with
    ``{"__tensor__": "<flat_key>"}`` placeholders. Returns a JSON-friendly
    structure.

    Note: dict keys are stringified (JSON requirement). The optimizer ``state``
    map uses integer param ids; the loader restores those at the appropriate
    level.
    """
    if isinstance(obj, torch.Tensor):
        tensors[prefix] = obj.detach().contiguous().cpu()
        return {"__tensor__": prefix}
    if isinstance(obj, dict):
        return {str(k): _split(v, f"{prefix}.{k}", tensors) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_split(v, f"{prefix}.{i}", tensors) for i, v in enumerate(obj)]
    return obj


def _merge(obj: Any, tensors: dict) -> Any:
    """Inverse of ``_split``: walk obj and resolve tensor placeholders."""
    if isinstance(obj, dict):
        if len(obj) == 1 and "__tensor__" in obj:
            return tensors[obj["__tensor__"]]
        return {k: _merge(v, tensors) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_merge(v, tensors) for v in obj]
    return obj


def _restore_optimizer_int_keys(optimizer_state: dict) -> dict:
    """torch optimizers key ``state`` by integer param id; JSON stringified
    them. Convert back."""
    if "state" in optimizer_state and isinstance(optimizer_state["state"], dict):
        optimizer_state["state"] = {
            int(k): v for k, v in optimizer_state["state"].items()
        }
    return optimizer_state


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_checkpoint(checkpoint: dict, filename: PathLike) -> None:
    """Save an nnU-Net checkpoint as safetensors + JSON sidecars.

    Writes up to four files alongside ``filename`` (which is conventionally a
    .pth path; the suffix is used only to derive sibling paths):

      <base>.safetensors                -- network weights (inference artifact)
      <base>.json                       -- inference metadata (small, distributable)
      <base>.trainer_state.safetensors  -- optimizer + grad_scaler tensors
      <base>.trainer_state.json         -- trainer Python state + skeletons

    Distribution = first two files. Resume = all four. The trainer-state
    pair is omitted entirely if the checkpoint has no optimizer or scaler
    state (e.g. an inference-only checkpoint converted from .pth weights).
    """
    pth = Path(filename)
    weights_path = _weights_path(pth)
    inference_meta_path = _inference_meta_json_path(pth)
    state_tensors_path = _trainer_state_tensors_path(pth)
    state_json_path = _trainer_state_json_path(pth)

    # 1. Network weights — flat dict[str, Tensor], no recursion needed.
    network_weights = checkpoint["network_weights"]
    network_weights_cpu = {
        k: v.detach().contiguous().cpu() for k, v in network_weights.items()
    }
    save_file(
        network_weights_cpu,
        str(weights_path),
        metadata={
            "weight_layout": WEIGHT_LAYOUT_TORCH,
            "format_version": str(FORMAT_VERSION),
            "framework": f"torch=={torch.__version__}",
        },
    )

    # 2. Inference metadata — what predict_from_raw_data needs and nothing
    # more. This is the file that ships with a distributed model.
    inference_meta = {
        "format_version": FORMAT_VERSION,
        "weight_layout": WEIGHT_LAYOUT_TORCH,
        "trainer_name": checkpoint.get("trainer_name"),
        "init_args": checkpoint.get("init_args"),
        "inference_allowed_mirroring_axes": checkpoint.get(
            "inference_allowed_mirroring_axes"
        ),
    }
    with open(inference_meta_path, "w") as f:
        json.dump(inference_meta, f, indent=2, default=str)

    # 3 + 4. Trainer state: tensors flattened into one safetensors file,
    # everything Python (including skeletons with tensor placeholders) into
    # the paired JSON file. Both omitted if there is no trainer state.
    has_optimizer = checkpoint.get("optimizer_state") is not None
    has_grad_scaler = checkpoint.get("grad_scaler_state") is not None
    if not (has_optimizer or has_grad_scaler):
        return

    trainer_tensors: dict[str, torch.Tensor] = {}
    optimizer_skeleton = None
    if has_optimizer:
        optimizer_skeleton = _split(
            checkpoint["optimizer_state"], "optimizer", trainer_tensors
        )

    grad_scaler_skeleton = None
    if has_grad_scaler:
        grad_scaler_skeleton = _split(
            checkpoint["grad_scaler_state"], "grad_scaler", trainer_tensors
        )

    if trainer_tensors:
        save_file(trainer_tensors, str(state_tensors_path))

    trainer_state = {
        "format_version": FORMAT_VERSION,
        "current_epoch": checkpoint.get("current_epoch"),
        "_best_ema": checkpoint.get("_best_ema"),
        "logging": checkpoint.get("logging"),
        "optimizer_state": optimizer_skeleton,
        "grad_scaler_state": grad_scaler_skeleton,
    }
    with open(state_json_path, "w") as f:
        json.dump(trainer_state, f, indent=2, default=str)


# Backwards-compat alias for the old function name used in earlier drafts of
# this branch. Will be removed before merge to master.
save_checkpoint_safetensors = save_checkpoint


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_checkpoint(
    filename: PathLike,
    map_location: Optional[Union[str, torch.device]] = None,
    load_optimizer: bool = True,
) -> dict:
    """Load an nnU-Net checkpoint, preferring the safetensors layout.

    Parameters
    ----------
    filename
        Path to the .pth checkpoint. The function looks for the safetensors
        sibling files first; if they don't exist it falls back to ``torch.load``.
    map_location
        Device for loaded tensors. Accepts a torch.device or a string.
    load_optimizer
        If False, skip optimizer/grad_scaler state even if available. The
        returned dict will have ``optimizer_state`` and ``grad_scaler_state``
        set to None. Inference callers should pass False.

    Returns
    -------
    dict
        A dict with the same keys as the legacy torch.load output, suitable
        for direct consumption by ``nnUNetTrainer.load_checkpoint``.
    """
    pth = Path(filename)
    weights_path = _weights_path(pth)
    inference_meta_path = _inference_meta_json_path(pth)
    state_tensors_path = _trainer_state_tensors_path(pth)
    state_json_path = _trainer_state_json_path(pth)

    if weights_path.exists():
        return _load_safetensors(
            weights_path, state_tensors_path, inference_meta_path,
            state_json_path, map_location, load_optimizer,
        )

    if pth.exists():
        return torch.load(str(pth), map_location=map_location, weights_only=False)

    raise FileNotFoundError(
        f"No checkpoint found: neither {weights_path} nor {pth} exists."
    )


def _device_str(map_location) -> str:
    if map_location is None:
        return "cpu"
    if isinstance(map_location, torch.device):
        return str(map_location)
    return str(map_location)


def _load_safetensors(
    weights_path: Path,
    state_tensors_path: Path,
    inference_meta_path: Path,
    state_json_path: Path,
    map_location,
    load_optimizer: bool,
) -> dict:
    device = _device_str(map_location)

    network_weights = load_file(str(weights_path), device=device)

    checkpoint: dict = {
        "network_weights": network_weights,
        "trainer_name": None,
        "init_args": None,
        "inference_allowed_mirroring_axes": None,
        "current_epoch": None,
        "_best_ema": None,
        "logging": None,
        "optimizer_state": None,
        "grad_scaler_state": None,
    }

    # Inference metadata sidecar — small, distribution-friendly.
    if inference_meta_path.exists():
        with open(inference_meta_path) as f:
            inference_meta = json.load(f)
        checkpoint["trainer_name"] = inference_meta.get("trainer_name")
        checkpoint["init_args"] = inference_meta.get("init_args")
        checkpoint["inference_allowed_mirroring_axes"] = inference_meta.get(
            "inference_allowed_mirroring_axes"
        )

    if not load_optimizer:
        return checkpoint

    # Trainer-state pair: optional. Absent on distribution-only artifacts.
    if not state_json_path.exists():
        return checkpoint

    with open(state_json_path) as f:
        trainer_state = json.load(f)
    checkpoint["current_epoch"] = trainer_state.get("current_epoch")
    checkpoint["_best_ema"] = trainer_state.get("_best_ema")
    checkpoint["logging"] = trainer_state.get("logging")

    if state_tensors_path.exists():
        trainer_tensors = load_file(str(state_tensors_path), device=device)
        if trainer_state.get("optimizer_state") is not None:
            opt = _merge(trainer_state["optimizer_state"], trainer_tensors)
            checkpoint["optimizer_state"] = _restore_optimizer_int_keys(opt)
        if trainer_state.get("grad_scaler_state") is not None:
            checkpoint["grad_scaler_state"] = _merge(
                trainer_state["grad_scaler_state"], trainer_tensors
            )

    return checkpoint


# ---------------------------------------------------------------------------
# .pth → safetensors conversion
# ---------------------------------------------------------------------------

def convert_pth_to_safetensors(
    pth_path: PathLike, keep_pth: bool = True
) -> Path:
    """Convert a legacy .pth checkpoint to the safetensors layout.

    Returns the path to the network weights safetensors file.
    """
    pth_path = Path(pth_path)
    checkpoint = torch.load(str(pth_path), map_location="cpu", weights_only=False)
    save_checkpoint(checkpoint, pth_path)
    if not keep_pth:
        pth_path.unlink()
    return _weights_path(pth_path)


def convert_cli():
    """CLI entry point: convert .pth checkpoints to safetensors."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert nnU-Net .pth checkpoints to the safetensors layout."
    )
    parser.add_argument(
        "path",
        help="Path to a .pth file, a fold directory, or a model output directory.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively find and convert all .pth checkpoints under the path.",
    )
    parser.add_argument(
        "--delete-pth",
        action="store_true",
        help="Delete the original .pth after successful conversion.",
    )
    args = parser.parse_args()

    path = Path(args.path)

    if path.is_file() and path.suffix == ".pth":
        sf = convert_pth_to_safetensors(path, keep_pth=not args.delete_pth)
        print(f"Converted: {sf}")
        return

    if path.is_dir():
        pattern = "**/*.pth" if args.recursive else "*.pth"
        pth_files = sorted(path.glob(pattern))
        if not pth_files:
            pth_files = sorted(path.glob("fold_*/checkpoint_*.pth"))
        if not pth_files:
            print(f"No .pth files found in {path}")
            return
        for pth in pth_files:
            sf_path = _weights_path(pth)
            if sf_path.exists():
                print(f"Skipping (exists): {sf_path}")
                continue
            sf = convert_pth_to_safetensors(pth, keep_pth=not args.delete_pth)
            print(f"Converted: {sf}")
        return

    print(f"Error: {path} is not a .pth file or directory")
