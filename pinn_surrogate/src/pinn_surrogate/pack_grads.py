"""Companion file to ``packed_v1.npz``: Dan's velocity gradients at the same packed nodes.

Replays ``pack.py``'s seeded node sampling (same geometry order, same RNG sequence) and saves
(u_x, u_y, v_x, v_y) in SI units for every packed node, so the v1 pack stays untouched and
the v2 gradient loss can be switched on or off. ``xy`` of every solve is checked against the
existing pack; any mismatch aborts.

Usage (numpy only)::

    python3 -m pinn_surrogate.pack_grads --dataset ../steady_state_dataset/dataset_v1_1k
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from pinn_surrogate.pack import N_NODES, SEED, sample_nodes, select_geometries

log = logging.getLogger(__name__)
GRAD_KEYS = ("u_x", "u_y", "v_x", "v_y")


def pack_grads(dataset: Path, packed: Path, out: Path) -> None:
    """Write ``out`` with ``grad`` (n_solves, N_NODES, 4) float32, SI units (1/s)."""
    t0 = time.time()
    ref = np.load(packed)["xy"]
    index = json.loads((dataset / "cfd_index.json").read_text())
    split = select_geometries(index)
    order = split["train"] + split["val"] + split["test_B"]
    rng = np.random.default_rng(SEED)
    grad = np.zeros((4 * len(order), N_NODES, 4), np.float32)
    i = 0
    for gid in order:
        gdir = dataset / "geoms" / gid
        mesh = np.load(gdir / "mesh.npz")
        d0 = np.load(gdir / "cfd" / "derived_0.npz")
        idx, _ = sample_nodes(rng, mesh, d0["dist_arc"])
        for k in range(4):
            d = np.load(gdir / "cfd" / f"derived_{k}.npz")
            if not np.array_equal(d["xy"][idx].astype(np.float32), ref[i]):
                raise SystemExit(f"node mismatch with {packed} at solve {i} ({gid}, k={k})")
            grad[i] = np.stack([d[key][idx] for key in GRAD_KEYS], axis=-1)
            i += 1
    np.savez_compressed(out, grad=grad, keys=np.array(GRAD_KEYS))
    log.info("wrote %s: %s, xy verified against %s (%.0f s)", out, grad.shape, packed, time.time() - t0)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--packed", type=Path, default=Path("data/packed_v1.npz"))
    ap.add_argument("--out", type=Path, default=Path("data/packed_v1_grads.npz"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    pack_grads(args.dataset, args.packed, args.out)


if __name__ == "__main__":
    main()
