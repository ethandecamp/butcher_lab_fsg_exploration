"""Wall-shear amplitude bias of trained models on the held-out family-B solves.

For every solve: peak ratio = max|tau_pred| / max|tau_true| and best-fit scale
a* = argmin_a ||a tau_true - tau_pred|| = (tau_pred . tau_true) / (tau_true . tau_true).
A median well below 1 with nearly every solve under-predicted means the error is a
systematic amplitude bias, not noise (REPORT.md, "The dominant error").

Usage::

    python3 -m pinn_surrogate.wall_bias runs/grid/n160_physon_s0 runs/grid/n160_physoff_s0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from pinn_surrogate.data import load_packed
from pinn_surrogate.dataset import SolveSet
from pinn_surrogate.evaluate import predict_wall
from pinn_surrogate.model import load_checkpoint

log = logging.getLogger(__name__)


def bias(run: Path, s: SolveSet) -> dict[str, float]:
    """Median peak ratio (+ IQR), median best-fit scale, share of solves under-predicted."""
    model, _ = load_checkpoint(run / "model.pt")
    peak, scale = [], []
    for i in range(len(s)):
        tp = predict_wall(model, s, i)["tau_t"]
        tt = s.wall_tau_t[i].numpy()
        peak.append(np.abs(tp).max() / np.abs(tt).max())
        scale.append(float(tp @ tt / (tt @ tt)))
    p = np.array(peak)
    return {
        "peak_ratio_median": float(np.median(p)),
        "peak_ratio_q25": float(np.percentile(p, 25)),
        "peak_ratio_q75": float(np.percentile(p, 75)),
        "best_fit_scale_median": float(np.median(scale)),
        "frac_under": float(np.mean(p < 1)),
    }


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_num_threads(args.threads)
    pk = load_packed()
    s = SolveSet.from_packed(pk, pk.nondim(), pk.split_indices("test_B"))
    for run in args.runs:
        b = bias(run, s)
        log.info(
            "%s: median peak ratio %.2f (IQR %.2f-%.2f); median best-fit scale %.2f; "
            "frac under-predicted peak %.2f",
            run.name,
            b["peak_ratio_median"],
            b["peak_ratio_q25"],
            b["peak_ratio_q75"],
            b["best_fit_scale_median"],
            b["frac_under"],
        )


if __name__ == "__main__":
    main()
