"""Round-trip tests for nnunetv2.utilities.checkpoint_io.

These tests build a tiny torch model + Adam optimizer + GradScaler, run a real
forward/backward/step so the optimizer accumulates non-trivial state (exp_avg,
exp_avg_sq, step counters), then save and reload the resulting checkpoint via
the safetensors layout and assert byte-equal recovery.

They are deliberately self-contained — no nnU-Net Trainer dependencies — so
they catch regressions in the I/O layer in isolation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from nnunetv2.utilities.checkpoint_io import (
    WEIGHT_LAYOUT_TORCH,
    _trainer_state_json_path,
    _trainer_state_tensors_path,
    _weights_path,
    convert_pth_to_safetensors,
    load_checkpoint,
    save_checkpoint,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(2, 4, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm3d(4, affine=True)
        self.head = nn.Conv3d(4, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(self.conv(x)))


def _build_warm_checkpoint() -> dict:
    """Build a checkpoint dict whose optimizer state has been warmed up by
    a couple of real optimizer steps."""
    torch.manual_seed(0)
    net = _TinyNet()
    optim = torch.optim.AdamW(net.parameters(), lr=1e-3, betas=(0.9, 0.95),
                              weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    for _ in range(3):
        x = torch.randn(1, 2, 4, 4, 4)
        target = torch.randn(1, 3, 4, 4, 4)
        loss = ((net(x) - target) ** 2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()

    return {
        "network_weights": net.state_dict(),
        "optimizer_state": optim.state_dict(),
        "grad_scaler_state": scaler.state_dict(),
        "logging": {"train_losses": [1.0, 0.8, 0.6], "lrs": [1e-3, 1e-3, 1e-3]},
        "_best_ema": 0.42,
        "current_epoch": 7,
        "init_args": {"plans": "fake_plans", "configuration": "3d_fullres",
                      "fold": 0, "dataset_json": {"channel_names": {"0": "CT"}}},
        "trainer_name": "nnUNetTrainer",
        "inference_allowed_mirroring_axes": (0, 1, 2),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_state_dicts_equal(a: dict, b: dict) -> None:
    assert set(a.keys()) == set(b.keys())
    for k in a:
        assert torch.equal(a[k], b[k]), f"weights mismatch at key {k}"


def _assert_optimizer_states_equal(a: dict, b: dict) -> None:
    # param_groups: hyperparams should match exactly (after list/tuple coercion).
    assert len(a["param_groups"]) == len(b["param_groups"])
    for pg_a, pg_b in zip(a["param_groups"], b["param_groups"]):
        for key in pg_a:
            va, vb = pg_a[key], pg_b[key]
            if isinstance(va, tuple):
                va = list(va)
            if isinstance(vb, tuple):
                vb = list(vb)
            assert va == vb, f"param_groups mismatch at {key}: {va!r} vs {vb!r}"

    # state: same param ids, same buffer keys, byte-equal tensors and scalars.
    assert set(a["state"].keys()) == set(b["state"].keys())
    for pid in a["state"]:
        sa, sb = a["state"][pid], b["state"][pid]
        assert set(sa.keys()) == set(sb.keys())
        for name in sa:
            va, vb = sa[name], sb[name]
            if isinstance(va, torch.Tensor):
                assert isinstance(vb, torch.Tensor)
                assert torch.equal(va, vb), f"state[{pid}][{name}] tensor mismatch"
            else:
                assert va == vb, f"state[{pid}][{name}] scalar mismatch"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_save_creates_three_files(tmp_path: Path) -> None:
    ckpt = _build_warm_checkpoint()
    pth = tmp_path / "checkpoint_final.pth"
    save_checkpoint(ckpt, pth)

    assert _weights_path(pth).exists()
    assert _trainer_state_tensors_path(pth).exists()
    assert _trainer_state_json_path(pth).exists()
    assert not pth.exists()  # we don't write .pth ourselves


def test_metadata_records_weight_layout(tmp_path: Path) -> None:
    from safetensors import safe_open

    ckpt = _build_warm_checkpoint()
    pth = tmp_path / "checkpoint_final.pth"
    save_checkpoint(ckpt, pth)

    with safe_open(str(_weights_path(pth)), framework="pt") as f:
        meta = f.metadata() or {}
    assert meta.get("weight_layout") == WEIGHT_LAYOUT_TORCH


def test_roundtrip_network_weights(tmp_path: Path) -> None:
    ckpt = _build_warm_checkpoint()
    pth = tmp_path / "checkpoint_final.pth"
    save_checkpoint(ckpt, pth)

    loaded = load_checkpoint(pth)
    _assert_state_dicts_equal(ckpt["network_weights"], loaded["network_weights"])


def test_roundtrip_optimizer_state(tmp_path: Path) -> None:
    ckpt = _build_warm_checkpoint()
    pth = tmp_path / "checkpoint_final.pth"
    save_checkpoint(ckpt, pth)

    loaded = load_checkpoint(pth)
    _assert_optimizer_states_equal(ckpt["optimizer_state"], loaded["optimizer_state"])


def test_loaded_optimizer_state_loadable_into_real_optimizer(tmp_path: Path) -> None:
    """The reconstructed dict must be accepted by torch.optim.AdamW.load_state_dict
    without errors and produce equivalent state."""
    ckpt = _build_warm_checkpoint()
    pth = tmp_path / "checkpoint_final.pth"
    save_checkpoint(ckpt, pth)
    loaded = load_checkpoint(pth)

    torch.manual_seed(0)
    net = _TinyNet()
    optim = torch.optim.AdamW(net.parameters(), lr=1e-3, betas=(0.9, 0.95),
                              weight_decay=1e-4)
    optim.load_state_dict(loaded["optimizer_state"])
    _assert_optimizer_states_equal(ckpt["optimizer_state"], optim.state_dict())


def test_roundtrip_python_metadata(tmp_path: Path) -> None:
    ckpt = _build_warm_checkpoint()
    pth = tmp_path / "checkpoint_final.pth"
    save_checkpoint(ckpt, pth)
    loaded = load_checkpoint(pth)

    assert loaded["trainer_name"] == "nnUNetTrainer"
    assert loaded["current_epoch"] == 7
    assert loaded["_best_ema"] == 0.42
    assert loaded["logging"]["train_losses"] == [1.0, 0.8, 0.6]
    assert loaded["init_args"]["configuration"] == "3d_fullres"
    # Tuple → list is acceptable; the trainer indexes by position.
    assert list(loaded["inference_allowed_mirroring_axes"]) == [0, 1, 2]


def test_load_optimizer_false_skips_state(tmp_path: Path) -> None:
    ckpt = _build_warm_checkpoint()
    pth = tmp_path / "checkpoint_final.pth"
    save_checkpoint(ckpt, pth)

    loaded = load_checkpoint(pth, load_optimizer=False)
    assert loaded["network_weights"] is not None
    assert loaded["optimizer_state"] is None
    assert loaded["grad_scaler_state"] is None
    # Inference path still gets what it needs.
    assert loaded["trainer_name"] == "nnUNetTrainer"
    assert loaded["init_args"]["configuration"] == "3d_fullres"


def test_legacy_pth_fallback(tmp_path: Path) -> None:
    """When only a .pth exists (no safetensors siblings), load via torch.load."""
    ckpt = _build_warm_checkpoint()
    pth = tmp_path / "checkpoint_final.pth"
    torch.save(ckpt, pth)

    loaded = load_checkpoint(pth)
    _assert_state_dicts_equal(ckpt["network_weights"], loaded["network_weights"])
    _assert_optimizer_states_equal(ckpt["optimizer_state"], loaded["optimizer_state"])


def test_convert_pth_to_safetensors(tmp_path: Path) -> None:
    ckpt = _build_warm_checkpoint()
    pth = tmp_path / "checkpoint_final.pth"
    torch.save(ckpt, pth)

    out = convert_pth_to_safetensors(pth, keep_pth=True)
    assert out == _weights_path(pth)
    assert _weights_path(pth).exists()
    assert _trainer_state_tensors_path(pth).exists()
    assert _trainer_state_json_path(pth).exists()
    assert pth.exists()  # keep_pth=True

    loaded = load_checkpoint(pth)
    _assert_state_dicts_equal(ckpt["network_weights"], loaded["network_weights"])
    _assert_optimizer_states_equal(ckpt["optimizer_state"], loaded["optimizer_state"])


def test_missing_checkpoint_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nonexistent.pth")
