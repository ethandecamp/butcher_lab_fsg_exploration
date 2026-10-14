"""Summarize nondimensional magnitudes over the packed data (for choosing output scalings).

Usage::

    PYTHONPATH=src python3 -m pinn_surrogate.stats --out data/nondim_stats.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from pinn_surrogate.data import PACKED_PATH, load_packed

log = logging.getLogger(__name__)

PERCENTILES = (0, 1, 5, 50, 95, 99, 100)


def summarize(values: np.ndarray) -> dict[str, float]:
    """Percentiles (keys ``p0`` .. ``p100``), mean and std of a flattened array."""
    v = np.asarray(values, dtype=np.float64).ravel()
    out = {f"p{q}": float(x) for q, x in zip(PERCENTILES, np.percentile(v, PERCENTILES))}
    out.update(mean=float(v.mean()), std=float(v.std()))
    return out


def nondim_stats(path: Path = PACKED_PATH) -> dict[str, dict]:
    """Compute the summary for every training-relevant nondimensional quantity.

    Args:
        path: Packed data file.

    Returns:
        ``{"all": {quantity: summary}, "by_flow_k": {k: {quantity: summary}}}``. The
        ``p_span`` entries are the per-solve max - min of nodal p* (gauge-free size).
    """
    pk = load_packed(path)
    nd = pk.nondim()
    p_span = nd["p"].max(axis=1) - nd["p"].min(axis=1)
    quantities = {
        "x": nd["xy"][..., 0],
        "y": nd["xy"][..., 1],
        "u": nd["uv"][..., 0],
        "v": nd["uv"][..., 1],
        "p": nd["p"],
        "p_span": p_span,
        "wall_p": nd["wall_p"],
        "wall_tau_t": nd["wall_tau_t"],
        "wall_tau_mag": nd["wall_tau_mag"],
        "Re": nd["Re"],
    }
    flow_k = pk.arrays["flow_k"]
    by_k = {
        int(k): {
            name: summarize(q[flow_k == k])
            for name, q in quantities.items()
            if name in ("p", "p_span", "wall_tau_t", "wall_tau_mag", "Re")
        }
        for k in np.unique(flow_k)
    }
    return {"all": {n: summarize(q) for n, q in quantities.items()}, "by_flow_k": by_k}


def main() -> None:
    """Print the summary table and write it as JSON."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--packed", type=Path, default=PACKED_PATH)
    ap.add_argument("--out", type=Path, default=PACKED_PATH.parent / "nondim_stats.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    stats = nondim_stats(args.packed)
    cols = [f"p{q}" for q in PERCENTILES]
    log.info("%-16s" + " %10s" * len(cols), "quantity", *cols)
    for name, s in stats["all"].items():
        log.info("%-16s" + " %10.3g" * len(cols), name, *(s[c] for c in cols))
    for k, rows in stats["by_flow_k"].items():
        for name, s in rows.items():
            log.info("%-16s" + " %10.3g" * len(cols), f"k={k} {name}", *(s[c] for c in cols))
    args.out.write_text(json.dumps(stats, indent=1))
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
