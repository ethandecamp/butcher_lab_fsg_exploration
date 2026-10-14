"""Conditioned PINN f(x, y | outline, Re) -> (u, v, p) with hard boundary conditions.

Architecture (all sizes in ``config.ModelConfig``):

* Shape encoder: the 256-point outline (normalized by fixed train-set statistics) is
  flattened to 512 numbers and mapped by an MLP to a latent code z (32 numbers).
* Trunk: per node, [Fourier features of (x/10, y), (x/10, y), z, normalized log Re] ->
  MLP with SiLU (smooth, so the second derivatives the PDE needs exist) -> (N_u, N_v, N_p).
* Hard-constraint ansatz (so walls and inlet are exact for *any* weights):

      m   = tanh(d_c / delta)      0 on the cushion, ~1 away from it
      w   = y (1 - y)              0 on the top and bottom walls
      g   = (x + 10) / 20          0 at the inlet, 1 at the outlet
      u_p = 6 y (1 - y)            Poiseuille inlet profile (mean speed 1)

      u = m (u_p + w g S_u N_u)    -> u = 0 on all walls, u = m u_p = u_p at the inlet
      v = m w g S_v N_v            -> v = 0 on all walls and at the inlet
      p = 12 (10 - x) + S_p N_p    -> Poiseuille pressure drop plus a learned correction

  The inlet value is exactly u_p because m = 1 there: the closest outline is 6.79 H from
  the inlet, so 1 - m <= 2.5e-118 (``geometry_np.inlet_mask_deficit``). The outlet
  condition is not hard-coded (it involves derivatives); it is a soft loss term.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pinn_surrogate.config import ModelConfig
from pinn_surrogate.geometry import polyline_distance

log = logging.getLogger(__name__)

INLET_X = -10.0
X_SCALE = 10.0  # x* / 10 brings the channel to [-1, 1] like y* in [0, 1]
POISEUILLE_DPDX = 12.0  # -dp*/dx* of plane Poiseuille flow with mean speed 1


def mlp(sizes: list[int]) -> nn.Sequential:
    """Linear layers with SiLU between them (none after the last)."""
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(pairwise(sizes)):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class FourierFeatures(nn.Module):
    """Random Fourier features [sin(2 pi v B), cos(2 pi v B)] with a fixed Gaussian B.

    Plain MLPs learn low frequencies first (spectral bias); these features give the trunk
    ready-made oscillations at a range of scales. B is a buffer: saved with the model,
    never trained.
    """

    def __init__(self, n_freq: int, sigma: float, seed: int) -> None:
        super().__init__()
        b = np.random.default_rng(seed).normal(0.0, sigma, size=(2, n_freq))
        self.register_buffer("B", torch.tensor(b, dtype=torch.float32))

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """Map (..., 2) coordinates to (..., 2 * n_freq) features."""
        proj = 2 * math.pi * (v @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class ShapeEncoder(nn.Module):
    """Outline (B, 256, 2) -> latent code (B, latent).

    Coordinates are normalized by fixed per-axis statistics of the training outlines
    (buffers ``shift``, ``scale``), not per shape, because the cushion's position and
    height in the channel matter.
    """

    def __init__(self, n_points: int, width: int, latent: int) -> None:
        super().__init__()
        self.net = mlp([2 * n_points, width, width, latent])
        self.register_buffer("shift", torch.zeros(2))
        self.register_buffer("scale", torch.ones(2))

    def forward(self, outline: torch.Tensor) -> torch.Tensor:
        """Encode (B, V, 2) outlines."""
        return self.net(((outline - self.shift) / self.scale).flatten(1))


class ConditionedPINN(nn.Module):
    """The surrogate network; see the module docstring for the ansatz."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(cfg.seed)  # reproducible initial weights
        self.encoder = ShapeEncoder(cfg.n_outline, cfg.enc_width, cfg.latent)
        self.fourier = FourierFeatures(cfg.n_fourier, cfg.fourier_sigma, cfg.seed)
        n_in = 2 * cfg.n_fourier + 2 + cfg.latent + 1
        self.trunk = mlp([n_in] + [cfg.width] * cfg.depth + [3])
        self.register_buffer("log_re_mean", torch.zeros(()))
        self.register_buffer("log_re_std", torch.ones(()))

    def n_params(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_normalization(self, outlines: np.ndarray, log_re: np.ndarray) -> None:
        """Fix the input normalization from training data (stored as buffers).

        Args:
            outlines: (S, 256, 2) nondim training outlines.
            log_re: (S,) natural log of the training Reynolds numbers.
        """
        flat = np.asarray(outlines, dtype=np.float64).reshape(-1, 2)
        std = np.maximum(flat.std(0), 1e-6)
        lr = np.asarray(log_re, dtype=np.float64)
        with torch.no_grad():
            self.encoder.shift.copy_(torch.tensor(flat.mean(0)))
            self.encoder.scale.copy_(torch.tensor(std))
            self.log_re_mean.fill_(float(lr.mean()))
            self.log_re_std.fill_(float(max(lr.std(), 1e-6)))

    def trunk_outputs(
        self, xy: torch.Tensor, z: torch.Tensor, log_re: torch.Tensor
    ) -> torch.Tensor:
        """Raw network outputs (N_u, N_v, N_p) at points xy, shape (B, N, 3).

        Args:
            xy: (B, N, 2) nondim points.
            z: (B, latent) shape codes.
            log_re: (B,) natural log of Re.
        """
        bsz, n = xy.shape[:2]
        coords = torch.stack([xy[..., 0] / X_SCALE, xy[..., 1]], dim=-1)
        re_in = ((log_re - self.log_re_mean) / self.log_re_std).reshape(bsz, 1, 1)
        feats = [
            self.fourier(coords),
            coords,
            z.unsqueeze(1).expand(bsz, n, z.shape[-1]),
            re_in.expand(bsz, n, 1),
        ]
        return self.trunk(torch.cat(feats, dim=-1))

    def parts(
        self, xy: torch.Tensor, outline: torch.Tensor, log_re: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Every factor of the ansatz, for tests and diagnostics (each (B, N))."""
        c = self.cfg
        x, y = xy[..., 0], xy[..., 1]
        raw = self.trunk_outputs(xy, self.encoder(outline), log_re)
        w = y * (1 - y)
        return {
            "m": torch.tanh(polyline_distance(xy, outline) / c.delta),
            "w": w,
            "g": (x - INLET_X) / (2 * X_SCALE),
            "u_p": 6 * w,
            "N_u": raw[..., 0],
            "N_v": raw[..., 1],
            "N_p": raw[..., 2],
            "p_base": POISEUILLE_DPDX * (-INLET_X - x),
        }

    def forward(
        self, xy: torch.Tensor, outline: torch.Tensor, log_re: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict (u, v, p), each (B, N), in nondimensional units.

        Args:
            xy: (B, N, 2) nondim points.
            outline: (B, 256, 2) nondim cushion outlines.
            log_re: (B,) natural log of Re.
        """
        c, f = self.cfg, self.parts(xy, outline, log_re)
        m, wg = f["m"], f["w"] * f["g"]
        u = m * (f["u_p"] + wg * c.s_u * f["N_u"])
        v = m * wg * c.s_v * f["N_v"]
        p = f["p_base"] + c.s_p * f["N_p"]
        return u, v, p

    def field_fn(
        self, outline: torch.Tensor, log_re: torch.Tensor
    ) -> Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """The model with its conditioning bound: xy -> (u, v, p), for ``physics``."""
        return lambda xy: self(xy, outline, log_re)


def atomic_save(obj: dict, path: Path) -> None:
    """``torch.save`` to a temporary file, then rename over ``path``.

    A VM restart in the middle of a write then leaves the previous file intact instead of
    a truncated one (``os.replace`` is atomic on the same filesystem).
    """
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def torch_load(path: Path) -> dict:
    """Load a checkpoint written by this package onto the CPU.

    ``weights_only=True`` (the default from torch 2.6) refuses arbitrary pickled objects;
    our checkpoints hold only tensors and plain Python types, so it always suffices.
    """
    return torch.load(path, map_location="cpu", weights_only=True)


def save_checkpoint(model: ConditionedPINN, path: Path, extra: dict | None = None) -> None:
    """Save weights, buffers and the model config (plain types only)."""
    atomic_save(
        {"state_dict": model.state_dict(), "model_config": asdict(model.cfg), **(extra or {})},
        path,
    )


def load_checkpoint(path: Path) -> tuple[ConditionedPINN, dict]:
    """Rebuild a model from ``save_checkpoint`` output.

    Returns:
        ``(model, checkpoint dict)``; the model is in eval mode on CPU.
    """
    ckpt = torch_load(path)
    model = ConditionedPINN(ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def main() -> None:
    """Print the parameter count of the default model, split by component."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    model = ConditionedPINN(ModelConfig())
    for name, part in (("encoder", model.encoder), ("trunk", model.trunk)):
        log.info("%-8s %8d", name, sum(p.numel() for p in part.parameters()))
    log.info("%-8s %8d", "total", model.n_params())


if __name__ == "__main__":
    main()
