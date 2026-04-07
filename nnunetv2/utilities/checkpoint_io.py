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


def _trainer_state_tensors_path(pth: PathLike) -> Path:
    p = Path(pth)
    return p.with_name(p.stem + ".trainer_state.safetensors")


def _trainer_state_json_path(pth: PathLike) -> Path:
    return Path(pth).with_suffix(".json")


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
    """Save an nnU-Net checkpoint as safetensors + JSON.

    Writes three files alongside ``filename`` (which is conventionally a .pth
    path; the suffix is used only to derive sibling paths).
    """
    pth = Path(filename)
    weights_path = _weights_path(pth)
    state_path = _trainer_state_tensors_path(pth)
    meta_path = _trainer_state_json_path(pth)

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

    # 2. Optimizer + grad_scaler tensors and skeletons.
    trainer_tensors: dict[str, torch.Tensor] = {}
    optimizer_skeleton = None
    if checkpoint.get("optimizer_state") is not None:
        optimizer_skeleton = _split(
            checkpoint["optimizer_state"], "optimizer", trainer_tensors
        )

    grad_scaler_skeleton = None
    if checkpoint.get("grad_scaler_state") is not None:
        grad_scaler_skeleton = _split(
            checkpoint["grad_scaler_state"], "grad_scaler", trainer_tensors
        )

    if trainer_tensors:
        save_file(trainer_tensors, str(state_path))

    # 3. JSON sidecar with everything Python.
    metadata = {
        "format_version": FORMAT_VERSION,
        "weight_layout": WEIGHT_LAYOUT_TORCH,
        "trainer_name": checkpoint.get("trainer_name"),
        "init_args": checkpoint.get("init_args"),
        "inference_allowed_mirroring_axes": checkpoint.get(
            "inference_allowed_mirroring_axes"
        ),
        "current_epoch": checkpoint.get("current_epoch"),
        "_best_ema": checkpoint.get("_best_ema"),
        "logging": checkpoint.get("logging"),
        "optimizer_state": optimizer_skeleton,
        "grad_scaler_state": grad_scaler_skeleton,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


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
    state_path = _trainer_state_tensors_path(pth)
    meta_path = _trainer_state_json_path(pth)

    if weights_path.exists():
        return _load_safetensors(
            weights_path, state_path, meta_path, map_location, load_optimizer
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
    state_path: Path,
    meta_path: Path,
    map_location,
    load_optimizer: bool,
) -> dict:
    device = _device_str(map_location)

    network_weights = load_file(str(weights_path), device=device)

    if not meta_path.exists():
        # Inference-only artifact (e.g. converted from a pth that had no
        # metadata). Return what we can.
        return {
            "network_weights": network_weights,
            "optimizer_state": None,
            "grad_scaler_state": None,
        }

    with open(meta_path) as f:
        metadata = json.load(f)

    checkpoint: dict = {
        "network_weights": network_weights,
        "trainer_name": metadata.get("trainer_name"),
        "init_args": metadata.get("init_args"),
        "inference_allowed_mirroring_axes": metadata.get(
            "inference_allowed_mirroring_axes"
        ),
        "current_epoch": metadata.get("current_epoch"),
        "_best_ema": metadata.get("_best_ema"),
        "logging": metadata.get("logging"),
        "optimizer_state": None,
        "grad_scaler_state": None,
    }

    if load_optimizer and state_path.exists():
        trainer_tensors = load_file(str(state_path), device=device)
        if metadata.get("optimizer_state") is not None:
            opt = _merge(metadata["optimizer_state"], trainer_tensors)
            checkpoint["optimizer_state"] = _restore_optimizer_int_keys(opt)
        if metadata.get("grad_scaler_state") is not None:
            checkpoint["grad_scaler_state"] = _merge(
                metadata["grad_scaler_state"], trainer_tensors
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
