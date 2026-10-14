"""Report figures for the ablation grid (static PNGs for REPORT.md).

* ``fig_scaling.png``: three panels against training-set size, physics on vs off, one dot
  per seed and a line through the seed mean: wall-shear error, wall-pressure error, and
  PDE residual (each a median over the held-out family-B solves). Separate panels, never
  a second y-axis, because the three quantities have different units and scales.
* ``fig_wall_curves.png``: signed wall shear along one held-out family-B cushion at its four
  flow rates: Dan's FEM vs the physics-on and physics-off models (n = 160, seed 0).

Usage::

    python3 -m pinn_surrogate.figures --runs runs/grid --out figures
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)

SIZES = (20, 80, 160)
# Categorical slots 1 and 2 of the reference palette (validated CVD-safe pair); FEM truth
# is drawn in neutral ink so it never competes with the two model arms for identity.
COLOR = {"on": "#2a78d6", "off": "#eb6834"}
LABEL = {"on": "physics on", "off": "physics off (data only)"}
INK, MUTED = "#222222", "#6b6b6b"


def _median(run: Path, metric: str, split: str = "test_B") -> float:
    agg = json.loads((run / "eval.json").read_text())["splits"][split]["aggregate"]["all"]
    return float(agg[metric]["median"])


def _style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)


def scaling_figure(runs: Path, out: Path) -> None:
    """Error and residual vs training-set size, both arms, every seed."""
    panels = (
        ("tau_rel_l2", "Wall shear error\n(median rel. L2)", False),
        ("p_rel_l2", "Wall pressure error\n(median rel. L2)", False),
        ("pde_res", "PDE residual\n(median, Dan's r_rel units)", True),
    )
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
    for ax, (metric, title, logy) in zip(axes, panels, strict=True):
        for arm in ("off", "on"):
            means = []
            for n in SIZES:
                vals = [
                    _median(r, metric)
                    for r in sorted(runs.glob(f"n{n}_phys{arm}_s*"))
                    if (r / "eval.json").exists()
                ]
                ax.scatter([n] * len(vals), vals, s=36, color=COLOR[arm], alpha=0.55, lw=0)
                means.append(np.mean(vals) if vals else np.nan)
            ax.plot(SIZES, means, color=COLOR[arm], lw=2, marker="o", ms=8, label=LABEL[arm])
        ax.set_xscale("log")
        ax.set_xticks(SIZES, [str(n) for n in SIZES])
        ax.minorticks_off()
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("training geometries", color=INK)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        _style(ax)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle(
        "Held-out FSG-derived shapes (family B); dots = seeds, line = seed mean",
        color=MUTED,
        fontsize=9,
        x=0.01,
        ha="left",
    )
    fig.savefig(out, dpi=160)
    plt.close(fig)
    log.info("wrote %s", out)


def wall_curves_figure(runs: Path, out: Path, n: int = 160, seed: int = 0) -> None:
    """Signed wall shear along one held-out family-B cushion, all four flows."""
    from pinn_surrogate import metrics
    from pinn_surrogate.data import load_packed
    from pinn_surrogate.dataset import SolveSet
    from pinn_surrogate.evaluate import predict_wall
    from pinn_surrogate.model import load_checkpoint

    pk = load_packed()
    nd = pk.nondim()
    test = pk.split_indices("test_B")
    gid = int(pk.arrays["geom_id"][test[0]])
    idx = np.array([i for i in test if int(pk.arrays["geom_id"][i]) == gid])
    s = SolveSet.from_packed(pk, nd, idx)
    models = {
        arm: load_checkpoint(runs / f"n{n}_phys{arm}_s{seed}" / "model.pt")[0]
        for arm in ("off", "on")
    }
    fig, axes = plt.subplots(1, len(idx), figsize=(12, 3.2), constrained_layout=True)
    for j, ax in enumerate(axes):
        frac = metrics.arc_fraction(s.outline[j].numpy())
        ax.plot(frac, s.wall_tau_t[j].numpy(), color=INK, lw=2.2, label="FEM (Dan)")
        for arm, model in models.items():
            tau = predict_wall(model, s, j)["tau_t"]
            ax.plot(frac, tau, color=COLOR[arm], lw=2, label=LABEL[arm])
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(f"Re = {float(s.re[j]):.2g}", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("arc length (upstream -> downstream)", color=INK, fontsize=9)
        _style(ax)
    axes[0].set_ylabel("wall shear  tau* = tau H / (mu U)", color=INK)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle(
        f"Held-out family-B geometry g_{gid:05d}: signed wall shear, n = {n} models, seed {seed}",
        color=MUTED,
        fontsize=9,
        x=0.01,
        ha="left",
    )
    fig.savefig(out, dpi=160)
    plt.close(fig)
    log.info("wrote %s", out)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=Path("runs/grid"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    scaling_figure(args.runs, args.out / "fig_scaling.png")
    wall_curves_figure(args.runs, args.out / "fig_wall_curves.png")


if __name__ == "__main__":
    main()
