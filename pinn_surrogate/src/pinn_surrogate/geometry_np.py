"""Brute-force numpy reference for the cushion distance function (torch-free).

``geometry.py`` computes the same distance in torch (vectorized, differentiable). This file
is the slow, obviously-correct version it is tested against, plus the inlet check that the
hard velocity constraint m = tanh(d / delta) is 1 at the inlet for every packed outline.

Run as a script it prints the numbers behind the model's design values (inlet margin of
the hard constraint, and the output scales S_u, S_v, S_p in ``config.ModelConfig``)::

    PYTHONPATH=src python3 -m pinn_surrogate.geometry_np
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

log = logging.getLogger(__name__)

INLET_X = -10.0  # nondim inlet position (x = -4 mm / H)


def point_segment_distance_np(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point to each segment [a_j, b_j].

    Args:
        points: (N, 2).
        a: Segment start points (S, 2).
        b: Segment end points (S, 2).

    Returns:
        (N, S) Euclidean distances (float64).
    """
    p = np.asarray(points, dtype=np.float64)[:, None, :]
    a = np.asarray(a, dtype=np.float64)[None]
    ab = np.asarray(b, dtype=np.float64)[None] - a
    t = np.clip(((p - a) * ab).sum(-1) / (ab * ab).sum(-1), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1)


def polyline_distance_np(points: np.ndarray, outlines: np.ndarray) -> np.ndarray:
    """Unsigned distance from points to an open polyline, batched over solves.

    Args:
        points: (B, N, 2) query points.
        outlines: (B, V, 2) polyline vertices (V - 1 segments).

    Returns:
        (B, N) distances, float64. A loop over the batch keeps memory at N x (V - 1).
    """
    out = np.empty(points.shape[:2], dtype=np.float64)
    for i, (p, o) in enumerate(zip(points, outlines)):
        out[i] = point_segment_distance_np(p, o[:-1], o[1:]).min(axis=1)
    return out


def inlet_distance_np(outlines: np.ndarray, x_inlet: float = INLET_X) -> np.ndarray:
    """Exact minimum distance between the inlet segment {x_inlet} x [0, 1] and each outline.

    The minimum distance between two sets of segments is attained at an endpoint of one of
    them, so it is the smaller of (outline vertices -> inlet segment) and (inlet endpoints
    -> outline polyline).

    Args:
        outlines: (B, V, 2) nondimensional outlines.
        x_inlet: Inlet x position.

    Returns:
        (B,) distances.
    """
    o = np.asarray(outlines, dtype=np.float64)
    dx = o[..., 0] - x_inlet
    dy = o[..., 1] - np.clip(o[..., 1], 0.0, 1.0)
    vert_to_inlet = np.hypot(dx, dy).min(axis=1)
    ends = np.broadcast_to(np.array([[x_inlet, 0.0], [x_inlet, 1.0]]), (len(o), 2, 2))
    inlet_to_poly = polyline_distance_np(ends, o).min(axis=1)
    return np.minimum(vert_to_inlet, inlet_to_poly)


def inlet_mask_deficit(outlines: np.ndarray, delta: float) -> np.ndarray:
    """1 - tanh(d_inlet / delta) per outline, computed without cancellation.

    1 - tanh(x) = 2 / (exp(2x) + 1), which stays accurate when tanh(x) rounds to exactly 1.

    Args:
        outlines: (B, V, 2) nondimensional outlines.
        delta: Hard-constraint length scale.

    Returns:
        (B,) deficits; the hard inlet profile is u_p * (1 - deficit).
    """
    x = inlet_distance_np(outlines) / delta
    return 2.0 / (np.exp(np.minimum(2.0 * x, 700.0)) + 1.0)


def ansatz_targets(
    xy: np.ndarray, uv: np.ndarray, p: np.ndarray, outline: np.ndarray, delta: float
) -> dict[str, np.ndarray]:
    """What the network outputs would have to be for the model to reproduce one solve.

    Inverts the hard-constraint ansatz of ``model.ConditionedPINN`` at interior nodes:
    ``S_u N_u = (u/m - u_p)/(w g)``, ``S_v N_v = v/(m w g)``, ``S_p N_p = p - 12(10 - x)``.
    The velocity entries are kept only where m > 0.2 and w g > 0.01, because dividing by
    a factor that is ~0 there amplifies FEM noise into meaningless values.

    Args:
        xy: (N, 2) interior node coordinates (nondim).
        uv: (N, 2) velocities; p: (N,) pressure.
        outline: (V, 2) the solve's outline.
        delta: Hard-constraint length scale.

    Returns:
        ``{"u": ..., "v": ..., "p": ...}`` 1-D arrays of required S*N values.
    """
    x, y = xy[:, 0], xy[:, 1]
    m = np.tanh(polyline_distance_np(xy[None], outline[None])[0] / delta)
    w, g = y * (1 - y), (x - INLET_X) / 20.0
    keep = (m > 0.2) & (w * g > 0.01)
    mk, wgk = m[keep], (w * g)[keep]
    return {
        "u": (uv[keep, 0] / mk - 6 * w[keep]) / wgk,
        "v": uv[keep, 1] / (mk * wgk),
        "p": p - 12.0 * (-INLET_X - x),
    }


def main() -> None:
    """Print the inlet margin over all packed outlines and the S-scale statistics."""
    from pinn_surrogate.data import TAG_INTERIOR, load_packed  # data imports nothing heavy

    ap = argparse.ArgumentParser(description="Design numbers for the hard-constraint ansatz.")
    ap.add_argument("--delta", type=float, default=0.05)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pk = load_packed()
    nd = pk.nondim()
    d_in = inlet_distance_np(nd["outline"])
    log.info(
        "inlet: min distance to any outline %.4f H; worst 1 - m (delta %g) = %.3g",
        d_in.min(),
        args.delta,
        inlet_mask_deficit(nd["outline"], args.delta).max(),
    )
    parts: dict[str, list[np.ndarray]] = {"u": [], "v": [], "p": []}
    idx = pk.split_indices("train")
    for i in idx:
        interior = pk.arrays["tag"][i] == TAG_INTERIOR
        t = ansatz_targets(
            nd["xy"][i][interior], nd["uv"][i][interior], nd["p"][i][interior],
            nd["outline"][i], args.delta,
        )  # fmt: skip
        for k, chunks in parts.items():
            chunks.append(t[k])
    log.info("required S*N over %d train solves (delta %g):", len(idx), args.delta)
    for k, chunks in parts.items():
        v = np.concatenate(chunks)
        a = np.abs(v)
        log.info(
            "  %s: n %d  std %.3g  mean %.3g  |.| p50 %.3g  p95 %.3g  p99 %.3g  max %.3g",
            k, len(v), v.std(), v.mean(), np.median(a),
            np.percentile(a, 95), np.percentile(a, 99), a.max(),
        )  # fmt: skip


if __name__ == "__main__":
    main()
