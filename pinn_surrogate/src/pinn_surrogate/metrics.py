"""Evaluation metrics on the 256 wall points and their aggregation (numpy only).

``evaluate.py`` produces predictions with torch and hands plain arrays to these functions,
so every number in ``eval.json`` is computed by code that is unit-tested without torch.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

SIGN_THRESHOLD = 0.01  # compare signs only where |tau| > 1 % of max |tau| (noise floor)


def rel_l2(pred: np.ndarray, true: np.ndarray) -> float:
    """Relative L2 error ||pred - true|| / ||true||."""
    pred, true = np.asarray(pred, dtype=np.float64), np.asarray(true, dtype=np.float64)
    return float(np.linalg.norm(pred - true) / np.linalg.norm(true))


def arc_fraction(outline: np.ndarray) -> np.ndarray:
    """Cumulative arc length of the outline vertices, as a fraction of the total (0..1).

    Args:
        outline: (V, 2) vertices.

    Returns:
        (V,) values, 0 at the first vertex and 1 at the last.
    """
    seg = np.linalg.norm(np.diff(np.asarray(outline, dtype=np.float64), axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return s / s[-1]


def peak_location_error(tau_pred: np.ndarray, tau_true: np.ndarray, s_frac: np.ndarray) -> float:
    """Distance along the arc between the predicted and true peak |tau|, as a fraction of L.

    The peak WSS location is what the biology responds to, so it gets its own metric.
    """
    return float(abs(s_frac[np.argmax(np.abs(tau_pred))] - s_frac[np.argmax(np.abs(tau_true))]))


def sign_agreement(
    tau_pred: np.ndarray, tau_true: np.ndarray, rel_threshold: float = SIGN_THRESHOLD
) -> float:
    """Fraction of wall points where sign(pred) == sign(true), ignoring near-zero true shear.

    Recirculation shows up as negative tau_t; this checks the network finds the same
    reversed-flow regions. Points with |tau| <= rel_threshold * max|tau| are excluded
    because their sign is noise (and is exactly where the FEM itself is least trustworthy).
    """
    tau_true = np.asarray(tau_true, dtype=np.float64)
    keep = np.abs(tau_true) > rel_threshold * np.abs(tau_true).max()
    if not keep.any():
        return float("nan")
    return float(np.mean(np.sign(tau_pred[keep]) == np.sign(tau_true[keep])))


def summarize(values: Iterable[float]) -> dict[str, float]:
    """Median, mean and count of the finite values."""
    v = np.asarray(list(values), dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"median": float("nan"), "mean": float("nan"), "n": 0}
    return {"median": float(np.median(v)), "mean": float(v.mean()), "n": len(v)}


def aggregate(rows: list[dict], metric_names: list[str], group_key: str | None = None) -> dict:
    """Median/mean of each metric over rows, optionally grouped by ``row[group_key]``.

    Args:
        rows: One dict per solve.
        metric_names: Keys of ``rows`` to summarize.
        group_key: ``None`` for a single summary, else e.g. ``"k"`` or ``"family"``.

    Returns:
        ``{metric: summary}`` or ``{str(group): {metric: summary}}`` (groups sorted).
    """
    if group_key is None:
        return {m: summarize(r[m] for r in rows) for m in metric_names}
    groups = sorted({r[group_key] for r in rows})
    return {str(g): aggregate([r for r in rows if r[group_key] == g], metric_names) for g in groups}
