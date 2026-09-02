"""The six executable tests a hypothesis may compile into.

Each primitive has the signature ``fn(ds, **params) -> (passed, observed, summary)``.
They are the only things in this project allowed to decide whether a claim survives, and
they decide it arithmetically: no primitive consults prose, and none of them looks at the
data before its threshold is fixed.

Three rules hold across all six:

* **NaN is missing, not zero.** Wall shear stress is NaN at every interior solid node,
  and von Mises is finite everywhere. Coercing those NaNs to 0 would quietly deflate
  every mean, fraction and correlation involving WSS. Non-finite samples are dropped and
  the count that survived is always reported as ``n_used``.
* **Everything the decision touched is reported.** ``observed`` carries the thresholds
  and the comparison operands as well as the measurement, so a reader can re-derive the
  verdict without rerunning anything.
* **Failure to run is not failure.** A primitive raises on bad parameters, an unknown
  field or an empty selection; :mod:`llm_insights.harness.runner` converts that into an
  ``Outcome`` with ``error`` set. It never returns ``passed=False`` to mean "broken".

The metric registry (:mod:`llm_insights.summary.metrics`) is imported lazily inside
:func:`metric_value_or_raise` so this module imports cleanly before that one exists.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any, Final

import numpy as np

from llm_insights.io.dataset import Dataset, GRNField, NodeField, arc_order_and_s

LOG = logging.getLogger(__name__)

#: Signature every registered primitive obeys.
PrimitiveFn = Callable[..., tuple[bool, dict[str, float], str]]

#: Name -> primitive. Populated by the :func:`primitive` decorator at import time.
PRIMITIVES: dict[str, PrimitiveFn] = {}

#: Denominator floor for relative differences, so a near-zero reference cannot explode.
EPS: Final[float] = 1e-12

_OPS: Final[dict[str, Callable[[float, float], bool]]] = {
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
}

_OP_SYMBOL: Final[dict[str, str]] = {"gt": ">", "ge": ">=", "lt": "<", "le": "<="}

#: Fields :func:`peak_location` understands.
_PEAK_FIELDS: Final[tuple[str, ...]] = ("wss", "von_mises")
#: Fields :func:`fraction_above` understands.
_FRACTION_FIELDS: Final[tuple[str, ...]] = ("von_mises", "wss", "mech_norm", "growth")
#: Fields :func:`profile_monotonic` understands, all of them arc-resolved.
_PROFILE_FIELDS: Final[tuple[str, ...]] = ("wss", "von_mises", "pressure", "height")


def primitive(name: str) -> Callable[[PrimitiveFn], PrimitiveFn]:
    """Register a function in :data:`PRIMITIVES` under ``name``.

    Args:
        name: The key a :class:`~llm_insights.harness.spec.TestSpec` uses to select it.

    Returns:
        A decorator that registers and returns the function unchanged.

    Raises:
        ValueError: If ``name`` is already registered.
    """

    def wrap(fn: PrimitiveFn) -> PrimitiveFn:
        if name in PRIMITIVES:
            raise ValueError(f"primitive {name!r} is already registered")
        PRIMITIVES[name] = fn
        LOG.debug("registered primitive %s", name)
        return fn

    return wrap


# --- shared helpers ---------------------------------------------------------------


def compare(op: str, left: float, right: float) -> bool:
    """Apply one of the four ordering operators.

    Args:
        op: One of ``"gt"``, ``"ge"``, ``"lt"``, ``"le"``.
        left: Left-hand operand.
        right: Right-hand operand.

    Returns:
        The truth value of ``left OP right``.

    Raises:
        ValueError: If ``op`` is not one of the four supported operators.
    """
    try:
        fn = _OPS[op]
    except (KeyError, TypeError):
        raise ValueError(f"unknown comparison op {op!r}; expected one of {sorted(_OPS)}") from None
    return bool(fn(float(left), float(right)))


def _as_float(name: str, value: Any) -> float:
    """Coerce a parameter to float with an error message a non-programmer can act on.

    Args:
        name: Parameter name, for the error message.
        value: The supplied value.

    Returns:
        The value as a float.

    Raises:
        ValueError: If the value is not a real number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
        raise ValueError(f"parameter {name!r} must be a number, got {value!r}")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"parameter {name!r} must be a finite number, got {value!r}")
    return out


def _as_int(name: str, value: Any) -> int:
    """Coerce a parameter to int with a clear error message.

    Args:
        name: Parameter name, for the error message.
        value: The supplied value.

    Returns:
        The value as an int.

    Raises:
        ValueError: If the value is not a whole number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"parameter {name!r} must be a whole number, got {value!r}")
    return int(value)


def _check_choice(name: str, value: Any, allowed: Sequence[str]) -> str:
    """Validate that a parameter is one of a fixed set of strings.

    Args:
        name: Parameter name, for the error message.
        value: The supplied value.
        allowed: The permitted values.

    Returns:
        The validated value.

    Raises:
        ValueError: If the value is not in ``allowed``.
    """
    if value not in allowed:
        raise ValueError(f"parameter {name!r} must be one of {list(allowed)}, got {value!r}")
    return str(value)


def _resolve(ds: Dataset, case: Any) -> str:
    """Resolve a case name or condition label, re-raising as a ``ValueError``.

    Args:
        ds: The dataset, used only for its ``resolve_case``.
        case: A case folder name (``"flow_U0p0360"``) or label (``"Healthy"``).

    Returns:
        The canonical case folder name.

    Raises:
        ValueError: If the name matches neither form.
    """
    if not isinstance(case, str):
        raise ValueError(f"case must be a string, got {type(case).__name__} ({case!r})")
    try:
        return ds.resolve_case(case)
    except KeyError as exc:
        raise ValueError(str(exc).strip("'")) from None


def metric_value_or_raise(ds: Dataset, name: str, case: str) -> float:
    """Look up a scalar metric through the metric registry.

    The registry lives in :mod:`llm_insights.summary.metrics` and is imported here rather
    than at module scope, so this module stays importable while that one is still being
    written.

    Args:
        ds: The dataset to measure.
        name: Metric name, which must be listed in ``METRIC_NAMES``.
        case: Canonical case folder name.

    Returns:
        The metric value.

    Raises:
        ValueError: If the registry cannot be imported or the metric is unknown.
    """
    try:
        from llm_insights.summary.metrics import METRIC_NAMES, metric_value
    except ImportError as exc:  # registry not delivered yet, or broken
        raise ValueError(
            "the metric registry (llm_insights.summary.metrics) could not be loaded, so "
            f"metric {name!r} cannot be measured: {exc}"
        ) from None
    if not isinstance(name, str):
        raise ValueError(f"metric must be a string, got {type(name).__name__} ({name!r})")
    if name not in METRIC_NAMES:
        raise ValueError(f"unknown metric {name!r}; known metrics are {sorted(METRIC_NAMES)}")
    value = float(metric_value(ds, name, case))
    LOG.debug("metric %s [%s] = %g", name, case, value)
    return value


def arc_projection(node_field: NodeField, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Place a nodal quantity on normalized arc position.

    Only the surface-band nodes carry an arc position at all, so the interior is dropped
    first. The survivors are ordered by the same descending-angle rule the dataset uses
    for :class:`~llm_insights.io.dataset.ArcProfile`, via
    :func:`llm_insights.io.dataset.arc_order_and_s` — that function is reused rather than
    reimplemented, because a plain sort on x disagrees with it on the underflow case.

    Args:
        node_field: The nodal field supplying coordinates and the surface mask.
        values: A per-node array, same length as the node field.

    Returns:
        A tuple ``(s_norm, values_in_arc_order)``, both of length ``node_field.n_surface``,
        with ``s_norm`` ascending from 0 (atrial) to 1 (ventricular).

    Raises:
        ValueError: If ``values`` has the wrong length or no node is on the surface.
    """
    vals = np.asarray(values, dtype=float)
    mask = np.asarray(node_field.is_surface, dtype=bool)
    if vals.shape != mask.shape:
        raise ValueError(
            f"values has shape {vals.shape} but the node field has {mask.shape} nodes; "
            "they must line up one-to-one"
        )
    if not mask.any():
        raise ValueError("no surface-band nodes in this node field, so nothing to place on the arc")
    x = np.asarray(node_field.x_mm, dtype=float)[mask]
    y = np.asarray(node_field.y_mm, dtype=float)[mask]
    order, s_norm, total_mm = arc_order_and_s(x, y)
    LOG.debug("projected %d surface nodes onto %.4f mm of arc", order.size, total_mm)
    return s_norm, vals[mask][order]


def _finite_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Drop index positions where either array is non-finite.

    Args:
        x: First array.
        y: Second array, same length.

    Returns:
        A tuple ``(x_kept, y_kept, n_dropped)``.

    Raises:
        ValueError: If the arrays have different lengths.
    """
    if x.shape != y.shape:
        raise ValueError(f"the two series have different lengths: {x.size} and {y.size}")
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep], int(x.size - int(np.count_nonzero(keep)))


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Rank ``values`` ascending, giving tied entries their shared average rank.

    This is ``scipy.stats.rankdata(values, method="average")`` in numpy, which is the
    ranking Spearman's coefficient is defined over. Ties must be averaged rather than
    broken arbitrarily: this dataset has many exactly-equal values -- every node clipped
    at the 100 Pa ceiling reports the identical activity -- and breaking those ties by
    array position would invent an ordering the data does not have.

    Args:
        values: The values to rank. Must be finite; callers drop non-finite pairs first.

    Returns:
        Float ranks in ``[1, n]``, one per input position.
    """
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    n = values.size
    starts_group = np.empty(n, dtype=bool)
    starts_group[0] = True
    np.not_equal(ordered[1:], ordered[:-1], out=starts_group[1:])
    group_of = np.cumsum(starts_group) - 1
    starts = np.flatnonzero(starts_group)
    counts = np.diff(np.append(starts, n))
    # A group filling 0-based sorted positions [s, s+c) holds 1-based ranks s+1..s+c,
    # whose mean is s + (c + 1) / 2.
    means = starts + (counts + 1.0) / 2.0
    ranks = np.empty(n, dtype=float)
    ranks[order] = means[group_of]
    return ranks


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson's correlation coefficient, in numpy.

    Each vector is centred and divided by its own norm before the dot product, rather
    than dividing a covariance by a product of standard deviations at the end. The two
    are algebraically identical; normalising first is better conditioned and is what
    ``scipy.stats.pearsonr`` does, which keeps this agreeing with the reference
    implementation to floating-point noise.

    Args:
        x: First series, finite.
        y: Second series, finite, same length.

    Returns:
        The coefficient, clamped into ``[-1, 1]`` so accumulated rounding cannot report
        a correlation of 1.0000000000000002.

    Raises:
        ValueError: If either series is constant, which leaves the coefficient undefined.
    """
    xm = x - x.mean()
    ym = y - y.mean()
    nx = float(np.linalg.norm(xm))
    ny = float(np.linalg.norm(ym))
    if nx == 0.0 or ny == 0.0:
        raise ValueError("a correlation is undefined when a series is constant")
    r = float(np.dot(xm / nx, ym / ny))
    return float(min(max(r, -1.0), 1.0))


def spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman's rank correlation, in numpy.

    Spearman is exactly Pearson computed over average-tied ranks, so this composes
    :func:`average_ranks` with :func:`pearson_r` rather than reimplementing anything.

    Args:
        x: First series, finite.
        y: Second series, finite, same length.

    Returns:
        The coefficient in ``[-1, 1]``.

    Raises:
        ValueError: If either series has no variation once ranked.
    """
    return pearson_r(average_ranks(x), average_ranks(y))


def _grn_series(grn: GRNField, name: Any) -> np.ndarray:
    """Pull one named column out of the GRN table.

    Args:
        grn: The loaded GRN field.
        name: ``"von_mises_pa"``, ``"wss_dyn_cm2"``, ``"mech_norm"``, ``"wss_norm"``, or
            any ``"<Node>_<scenario>"`` activity key.

    Returns:
        The column as a float array.

    Raises:
        ValueError: If the name is not a column of the GRN table.
    """
    base = {
        "von_mises_pa": grn.von_mises_pa,
        "wss_dyn_cm2": grn.wss_dyn_cm2,
        "mech_norm": grn.mech_norm,
        "wss_norm": grn.wss_norm,
    }
    if isinstance(name, str):
        if name in base:
            return np.asarray(base[name], dtype=float)
        if name in grn.activities:
            return np.asarray(grn.activities[name], dtype=float)
    available = sorted(base) + sorted(grn.activities)
    raise ValueError(f"unknown GRN field {name!r}; available fields are {available}")


def _arc_series(ds: Dataset, field: str, case: str, step: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(s_norm, values)`` for an arc-resolved field.

    Args:
        ds: The dataset.
        field: One of :data:`_PROFILE_FIELDS`.
        case: Canonical case folder name.
        step: Growth step. Only step 14 is available for ``von_mises``.

    Returns:
        A tuple ``(s_norm, values)`` in ascending arc order.

    Raises:
        ValueError: If the field is unknown, or ``von_mises`` is asked for at a step
            other than 14.
    """
    if field == "von_mises":
        if step != 14:
            raise ValueError(
                "nodal von Mises is only stored at step 14, so it cannot be placed on the "
                f"arc at step {step}; use step=14 or field='wss'"
            )
        nodes = ds.nodes(case)
        return arc_projection(nodes, nodes.von_mises_pa)
    arc = ds.arc(case, step)
    if field == "wss":
        return np.asarray(arc.s_norm, dtype=float), np.asarray(arc.wss_dyn_cm2, dtype=float)
    if field == "pressure":
        return np.asarray(arc.s_norm, dtype=float), np.asarray(arc.pressure_pa, dtype=float)
    if field == "height":
        return np.asarray(arc.s_norm, dtype=float), np.asarray(arc.y_mm, dtype=float)
    raise ValueError(f"unknown field {field!r}; expected one of {list(_PROFILE_FIELDS)}")


# --- primitives -------------------------------------------------------------------


@primitive("compare_metric")
def compare_metric(
    ds: Dataset,
    metric: str,
    case_a: str,
    case_b: str,
    op: str,
    min_rel_diff: float = 0.0,
) -> tuple[bool, dict[str, float], str]:
    """Compare one scalar metric between two cases, with a noise guard.

    The claim passes only if the ordering holds *and* the two values differ by at least
    ``min_rel_diff`` in relative terms. Without that guard a claim like "overflow has
    higher peak stress" would pass on a difference of one part in 10^15.

    Args:
        ds: The dataset to measure.
        metric: A name from ``llm_insights.summary.metrics.METRIC_NAMES``.
        case_a: Case on the left of the comparison.
        case_b: Case on the right.
        op: ``"gt"``, ``"ge"``, ``"lt"`` or ``"le"``.
        min_rel_diff: Required ``abs(a - b) / max(abs(b), EPS)``. Default 0.0 admits any
            difference, including numerical noise.

    Returns:
        A tuple ``(passed, observed, summary)``.

    Raises:
        ValueError: On an unknown metric, case, or operator, or a non-numeric guard.
    """
    op = _check_choice("op", op, tuple(_OPS))
    min_rel_diff = _as_float("min_rel_diff", min_rel_diff)
    if min_rel_diff < 0:
        raise ValueError(f"parameter 'min_rel_diff' must not be negative, got {min_rel_diff}")
    a_case = _resolve(ds, case_a)
    b_case = _resolve(ds, case_b)
    a = metric_value_or_raise(ds, metric, a_case)
    b = metric_value_or_raise(ds, metric, b_case)

    abs_diff = abs(a - b)
    rel_diff = abs_diff / max(abs(b), EPS)
    ordering_ok = compare(op, a, b)
    margin_ok = rel_diff >= min_rel_diff
    passed = bool(ordering_ok and margin_ok)

    observed = {
        "value_a": a,
        "value_b": b,
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
        "min_rel_diff": min_rel_diff,
        "ordering_ok": float(ordering_ok),
        "margin_ok": float(margin_ok),
    }
    reason = (
        ""
        if passed
        else (
            " (ordering does not hold)"
            if not ordering_ok
            else f" (difference is only {rel_diff:.3g}, below the required {min_rel_diff:.3g})"
        )
    )
    summary = (
        f"{metric}: {a_case}={a:.6g} {_OP_SYMBOL[op]} {b_case}={b:.6g}? "
        f"relative difference {rel_diff:.3g} vs required {min_rel_diff:.3g}"
        f" -> {'holds' if passed else 'does not hold'}{reason}"
    )
    return passed, observed, summary


@primitive("metric_ordering")
def metric_ordering(
    ds: Dataset,
    metric: str,
    cases: Sequence[str],
    direction: str,
    strict: bool = True,
) -> tuple[bool, dict[str, float], str]:
    """Check that a metric is monotonically ordered across a list of cases.

    Args:
        ds: The dataset to measure.
        metric: A name from ``llm_insights.summary.metrics.METRIC_NAMES``.
        cases: Two or more cases, in the order the claim asserts.
        direction: ``"increasing"`` or ``"decreasing"``.
        strict: If True, equal neighbours count as a violation.

    Returns:
        A tuple ``(passed, observed, summary)``. ``observed`` carries one
        ``value_<case>`` entry per case plus the violation count.

    Raises:
        ValueError: On a bad direction, fewer than two cases, duplicate cases, or an
            unknown metric or case.
    """
    direction = _check_choice("direction", direction, ("increasing", "decreasing"))
    if not isinstance(strict, bool):
        raise ValueError(f"parameter 'strict' must be true or false, got {strict!r}")
    if isinstance(cases, str) or not isinstance(cases, Sequence):
        raise ValueError(f"parameter 'cases' must be a list of case names, got {cases!r}")
    if len(cases) < 2:
        raise ValueError(
            f"parameter 'cases' needs at least 2 cases to be ordered, got {len(cases)}"
        )
    resolved = [_resolve(ds, c) for c in cases]
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"parameter 'cases' lists the same case twice: {resolved}")

    values = [metric_value_or_raise(ds, metric, c) for c in resolved]
    diffs = np.diff(np.asarray(values, dtype=float))
    if direction == "increasing":
        ok = diffs > 0 if strict else diffs >= 0
    else:
        ok = diffs < 0 if strict else diffs <= 0
    n_violations = int(np.count_nonzero(~ok))
    passed = n_violations == 0

    observed: dict[str, float] = {f"value_{c}": v for c, v in zip(resolved, values, strict=True)}
    observed["n_cases"] = float(len(resolved))
    observed["n_violations"] = float(n_violations)
    observed["min_step"] = float(np.min(diffs))
    observed["max_step"] = float(np.max(diffs))
    chain = " -> ".join(f"{c}={v:.6g}" for c, v in zip(resolved, values, strict=True))
    summary = (
        f"{metric} across {chain}: {'strictly ' if strict else ''}{direction} "
        f"with {n_violations} violation(s) out of {diffs.size} step(s)"
    )
    return passed, observed, summary


@primitive("peak_location")
def peak_location(
    ds: Dataset,
    field: str,
    case: str,
    s_min: float,
    s_max: float,
    step: int = 14,
) -> tuple[bool, dict[str, float], str]:
    """Check that a field's maximum sits inside a window of normalized arc position.

    ``s_norm = 0`` is the atrial (upstream) end and ``s_norm = 1`` the ventricular end.
    ``wss`` is read from the fluid-side arc profile; ``von_mises`` is nodal and is placed
    on the arc by :func:`arc_projection`, which is only possible at step 14.

    Args:
        ds: The dataset to measure.
        field: ``"wss"`` or ``"von_mises"``.
        case: Case folder name or condition label.
        s_min: Lower edge of the accepted window, inclusive.
        s_max: Upper edge, inclusive.
        step: Growth step for ``wss``. Must be 14 for ``von_mises``.

    Returns:
        A tuple ``(passed, observed, summary)``.

    Raises:
        ValueError: On an unknown field or case, an inverted window, or a profile with
            no finite samples.
    """
    field = _check_choice("field", field, _PEAK_FIELDS)
    s_min = _as_float("s_min", s_min)
    s_max = _as_float("s_max", s_max)
    step = _as_int("step", step)
    if s_min > s_max:
        raise ValueError(f"s_min ({s_min}) must not exceed s_max ({s_max})")
    case = _resolve(ds, case)

    s, values = _arc_series(ds, field, case, step)
    finite = np.isfinite(values)
    n_used = int(np.count_nonzero(finite))
    if n_used == 0:
        raise ValueError(
            f"every {field} sample for {case} at step {step} is missing (NaN), so there is "
            "no peak to locate"
        )
    idx = int(np.argmax(np.where(finite, values, -np.inf)))
    s_peak = float(s[idx])
    peak = float(values[idx])
    passed = bool(s_min <= s_peak <= s_max)

    observed = {
        "s_at_peak": s_peak,
        "peak_value": peak,
        "s_min": s_min,
        "s_max": s_max,
        "n_used": float(n_used),
        "n_dropped": float(values.size - n_used),
    }
    summary = (
        f"peak {field} for {case} (step {step}) is {peak:.6g} at s={s_peak:.4f}, "
        f"{'inside' if passed else 'outside'} the window [{s_min:.4f}, {s_max:.4f}] "
        f"(from {n_used} finite samples)"
    )
    return passed, observed, summary


@primitive("fraction_above")
def fraction_above(
    ds: Dataset,
    field: str,
    case: str,
    threshold: float,
    op: str,
    value: float,
    step: int = 14,
) -> tuple[bool, dict[str, float], str]:
    """Compare the fraction of samples above a threshold against a target.

    Sources, all at step 14 unless noted: ``von_mises`` and ``wss`` come from the full
    solid mesh (``Dataset.nodes``), ``mech_norm`` from the GRN table (``Dataset.grn``),
    and ``growth`` from the per-cell multiplier at ``step`` (``Dataset.growth``).

    WSS is NaN at every interior node. Those samples are dropped, not counted as zero,
    so the fraction is out of the nodes that actually carry a value.

    Args:
        ds: The dataset to measure.
        field: ``"von_mises"``, ``"wss"``, ``"mech_norm"`` or ``"growth"``.
        case: Case folder name or condition label.
        threshold: Samples strictly greater than this count toward the fraction.
        op: ``"gt"``, ``"ge"``, ``"lt"`` or ``"le"``, comparing the fraction to ``value``.
        value: Target fraction in [0, 1].
        step: Growth step, used only by ``growth``.

    Returns:
        A tuple ``(passed, observed, summary)``.

    Raises:
        ValueError: On an unknown field, case or operator, or if no finite samples remain.
    """
    field = _check_choice("field", field, _FRACTION_FIELDS)
    op = _check_choice("op", op, tuple(_OPS))
    threshold = _as_float("threshold", threshold)
    value = _as_float("value", value)
    step = _as_int("step", step)
    case = _resolve(ds, case)

    if field == "von_mises":
        samples = np.asarray(ds.nodes(case).von_mises_pa, dtype=float)
    elif field == "wss":
        samples = np.asarray(ds.nodes(case).wss_dyn_cm2, dtype=float)
    elif field == "mech_norm":
        samples = np.asarray(ds.grn(case).mech_norm, dtype=float)
    else:
        samples = np.asarray(ds.growth(case, step).g, dtype=float)

    finite = np.isfinite(samples)
    n_used = int(np.count_nonzero(finite))
    n_dropped = int(samples.size - n_used)
    if n_used == 0:
        raise ValueError(
            f"every {field} sample for {case} is missing (NaN), so no fraction can be computed"
        )
    kept = samples[finite]
    n_above = int(np.count_nonzero(kept > threshold))
    fraction = n_above / n_used
    passed = compare(op, fraction, value)

    observed = {
        "fraction": float(fraction),
        "threshold": threshold,
        "target": value,
        "n_above": float(n_above),
        "n_used": float(n_used),
        "n_dropped": float(n_dropped),
    }
    summary = (
        f"{n_above}/{n_used} {field} samples for {case} exceed {threshold:g} "
        f"(fraction {fraction:.4f}); {fraction:.4f} {_OP_SYMBOL[op]} {value:g} is "
        f"{'true' if passed else 'false'}"
        + (f"; {n_dropped} non-finite sample(s) dropped" if n_dropped else "")
    )
    return passed, observed, summary


@primitive("correlation")
def correlation(
    ds: Dataset,
    field_x: str,
    field_y: str,
    case: str,
    op: str,
    value: float,
    method: str = "pearson",
) -> tuple[bool, dict[str, float], str]:
    """Correlate two GRN-table columns and compare the coefficient to a target.

    Both columns come from the same rows of ``results_spatial.csv``, so they are already
    co-registered. Rows where either column is non-finite are dropped pairwise; WSS is
    NaN at interior nodes, and treating those as zero would drag any WSS correlation
    toward the mechanical field's interior structure.

    Args:
        ds: The dataset to measure.
        field_x: A GRN column: ``"von_mises_pa"``, ``"wss_dyn_cm2"``, ``"mech_norm"``,
            ``"wss_norm"``, or a ``"<Node>_<scenario>"`` activity.
        field_y: The other column, same vocabulary.
        case: Case folder name or condition label.
        op: ``"gt"``, ``"ge"``, ``"lt"`` or ``"le"``, comparing the coefficient to ``value``.
        value: Target coefficient in [-1, 1].
        method: ``"pearson"`` or ``"spearman"``.

    Returns:
        A tuple ``(passed, observed, summary)``. ``observed["n_used"]`` is the number of
        row pairs that survived NaN removal.

    Raises:
        ValueError: On an unknown field, case, operator or method; on fewer than three
            usable pairs; or if either column is constant over those pairs.
    """
    op = _check_choice("op", op, tuple(_OPS))
    method = _check_choice("method", method, ("pearson", "spearman"))
    value = _as_float("value", value)
    case = _resolve(ds, case)

    grn = ds.grn(case)
    x_all = _grn_series(grn, field_x)
    y_all = _grn_series(grn, field_y)
    x, y, n_dropped = _finite_pair(x_all, y_all)
    n_used = int(x.size)
    if n_used < 3:
        raise ValueError(
            f"only {n_used} row(s) have a finite value for both {field_x!r} and {field_y!r} "
            f"in {case}; at least 3 are needed to correlate them"
        )
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        constant = field_x if float(np.std(x)) == 0.0 else field_y
        raise ValueError(
            f"{constant!r} is constant across the {n_used} usable rows in {case}, so a "
            "correlation is undefined"
        )

    r = pearson_r(x, y) if method == "pearson" else spearman_r(x, y)
    passed = compare(op, r, value)

    observed = {
        "r": r,
        "target": value,
        "n_used": float(n_used),
        "n_dropped": float(n_dropped),
    }
    summary = (
        f"{method} correlation of {field_x} with {field_y} in {case} is r={r:.4f} over "
        f"{n_used} rows ({n_dropped} dropped for missing values); r {_OP_SYMBOL[op]} "
        f"{value:g} is {'true' if passed else 'false'}"
    )
    return passed, observed, summary


@primitive("profile_monotonic")
def profile_monotonic(
    ds: Dataset,
    field: str,
    case: str,
    s_min: float,
    s_max: float,
    direction: str,
    min_fraction: float = 0.8,
    step: int = 14,
) -> tuple[bool, dict[str, float], str]:
    """Check that a profile trends one way over a window of normalized arc position.

    Consecutive differences are taken over the samples inside ``[s_min, s_max]``, in arc
    order. The claim passes if at least ``min_fraction`` of them have the required sign.
    Exact ties count against the claim, so a flat profile cannot pass as either trend.

    Args:
        ds: The dataset to measure.
        field: ``"wss"``, ``"von_mises"``, ``"pressure"`` or ``"height"``.
        case: Case folder name or condition label.
        s_min: Lower edge of the window, inclusive.
        s_max: Upper edge, inclusive.
        direction: ``"increasing"`` or ``"decreasing"``.
        min_fraction: Required share of correctly-signed differences, in [0, 1].
        step: Growth step. Must be 14 for ``von_mises``.

    Returns:
        A tuple ``(passed, observed, summary)``.

    Raises:
        ValueError: On an unknown field, case or direction, an inverted window, or a
            window holding fewer than two finite samples.
    """
    field = _check_choice("field", field, _PROFILE_FIELDS)
    direction = _check_choice("direction", direction, ("increasing", "decreasing"))
    s_min = _as_float("s_min", s_min)
    s_max = _as_float("s_max", s_max)
    min_fraction = _as_float("min_fraction", min_fraction)
    step = _as_int("step", step)
    if s_min > s_max:
        raise ValueError(f"s_min ({s_min}) must not exceed s_max ({s_max})")
    if not 0.0 <= min_fraction <= 1.0:
        raise ValueError(f"parameter 'min_fraction' must be between 0 and 1, got {min_fraction}")
    case = _resolve(ds, case)

    s, values = _arc_series(ds, field, case, step)
    window = (s >= s_min) & (s <= s_max)
    n_in_window = int(np.count_nonzero(window))
    keep = window & np.isfinite(values)
    n_used = int(np.count_nonzero(keep))
    if n_used < 2:
        raise ValueError(
            f"the window [{s_min}, {s_max}] holds only {n_used} finite {field} sample(s) for "
            f"{case} at step {step}; at least 2 are needed to see a trend"
        )
    diffs = np.diff(values[keep])
    good = diffs > 0 if direction == "increasing" else diffs < 0
    fraction = float(np.count_nonzero(good)) / float(diffs.size)
    passed = bool(fraction >= min_fraction)

    observed = {
        "fraction_monotonic": fraction,
        "min_fraction": min_fraction,
        "n_diffs": float(diffs.size),
        "n_used": float(n_used),
        "n_dropped": float(n_in_window - n_used),
        "s_min": s_min,
        "s_max": s_max,
    }
    summary = (
        f"{field} for {case} over s in [{s_min:.4f}, {s_max:.4f}]: "
        f"{fraction:.4f} of {diffs.size} consecutive differences are {direction}, "
        f"{'meeting' if passed else 'short of'} the required {min_fraction:.4f}"
    )
    return passed, observed, summary
