"""Named scalar metrics over the FSG + GRN output, each carrying its own provenance.

This is the coarsest of the three zoom levels in the summary layer. A metric is one
number per case, and every one of them declares its units and exactly how it was
computed, because a number without units or provenance is not defensible in a meeting.

The registry is deliberately small and closed. A hypothesis may only be tested against
a metric that appears here, which is what stops a language model from inventing a
plausible-sounding quantity that nothing actually computes.

**Grid dependence, stated once.** Von Mises statistics come from the full 11,615-node
solid mesh; GRN statistics come from the 80%-downsampled grid the GRN was actually
evaluated on. These are different denominators, so ``mech_clipped_fraction`` and
``grn_saturated_fraction`` measure the same physical saturation and still disagree
(0.337 vs 0.296 on the healthy case). Both are correct for their grid. Any claim about
"the fraction of the domain that saturates" must name which one it means.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np

from llm_insights.io.dataset import MECH_MAX_PA, Dataset

LOG = logging.getLogger(__name__)

#: Growth multiplier floor observed in the shipped runs (the source default is 0.3).
GROWTH_FLOOR: Final[float] = 0.5
#: Tolerance for calling a growth value "pinned" at the floor.
GROWTH_PIN_TOL: Final[float] = 1e-9
#: Final growth step, the only one with nodal stress available for all three cases.
FINAL_STEP: Final[int] = 14


@dataclass(frozen=True)
class MetricDef:
    """One named scalar metric.

    Attributes:
        name: Registry key, used verbatim in hypothesis specs.
        units: Physical units, or ``"dimensionless"``.
        description: What the number means, in one line a biologist can read.
        source: Which artifact it is computed from and how. This is the provenance
            string that travels with the number into the briefing and onto the cards.
        fn: Callable taking ``(dataset, case)`` and returning the value.
    """

    name: str
    units: str
    description: str
    source: str
    fn: Callable[[Dataset, str], float]


_REGISTRY: dict[str, MetricDef] = {}


def _register(name: str, units: str, description: str, source: str) -> Callable:
    """Return a decorator registering a metric function under ``name``.

    Args:
        name: Registry key.
        units: Physical units string.
        description: One-line meaning.
        source: Provenance string.

    Returns:
        A decorator that registers the wrapped function and returns it unchanged.
    """

    def wrap(fn: Callable[[Dataset, str], float]) -> Callable[[Dataset, str], float]:
        if name in _REGISTRY:
            raise RuntimeError(f"metric {name!r} registered twice")
        _REGISTRY[name] = MetricDef(name, units, description, source, fn)
        return fn

    return wrap


def _surface_arc(ds: Dataset, case: str, values_attr: str) -> tuple[np.ndarray, np.ndarray]:
    """Project a nodal field onto arc position using the shared ordering rule."""
    from llm_insights.harness.primitives import arc_projection

    nodes = ds.nodes(case)
    return arc_projection(nodes, getattr(nodes, values_attr))


def _asymmetry(s: np.ndarray, v: np.ndarray) -> float:
    """Ventricular-minus-atrial mean, normalized by the overall mean.

    Positive means the quantity is weighted toward the ventricular (downstream) half.
    Normalizing by the overall mean makes the number comparable across cases whose
    absolute magnitudes differ by a factor of two or more.
    """
    finite = np.isfinite(v)
    s, v = s[finite], v[finite]
    if v.size == 0:
        return float("nan")
    overall = float(np.mean(v))
    if abs(overall) < 1e-12:
        return float("nan")
    down = v[s >= 0.5]
    up = v[s < 0.5]
    if down.size == 0 or up.size == 0:
        return float("nan")
    return float((np.mean(down) - np.mean(up)) / overall)


# --- Surface mechanics -----------------------------------------------------------


@_register(
    "wss_peak_dyn_cm2",
    "dyn/cm^2",
    "Maximum wall shear stress magnitude on the cushion surface at the final step.",
    "max of tau_mag from step_014/arc_data.npz, converted Pa -> dyn/cm^2 (x10)",
)
def _wss_peak(ds: Dataset, case: str) -> float:
    return float(np.max(ds.arc(case, FINAL_STEP).wss_dyn_cm2))


@_register(
    "wss_mean_dyn_cm2",
    "dyn/cm^2",
    "Mean wall shear stress magnitude along the cushion surface at the final step.",
    "mean of tau_mag from step_014/arc_data.npz, converted Pa -> dyn/cm^2 (x10)",
)
def _wss_mean(ds: Dataset, case: str) -> float:
    return float(np.mean(ds.arc(case, FINAL_STEP).wss_dyn_cm2))


@_register(
    "wss_peak_s",
    "dimensionless",
    "Normalized arc position (0 atrial, 1 ventricular) of the wall shear stress maximum.",
    "argmax of tau_mag on the angle-ordered arc from step_014/arc_data.npz",
)
def _wss_peak_s(ds: Dataset, case: str) -> float:
    arc = ds.arc(case, FINAL_STEP)
    return float(arc.s_norm[int(np.argmax(arc.wss_dyn_cm2))])


@_register(
    "wss_lr_asymmetry",
    "dimensionless",
    "Ventricular-minus-atrial mean wall shear stress, divided by the overall mean.",
    "halves split at s=0.5 on the angle-ordered arc from step_014/arc_data.npz",
)
def _wss_asym(ds: Dataset, case: str) -> float:
    arc = ds.arc(case, FINAL_STEP)
    return _asymmetry(arc.s_norm, arc.wss_dyn_cm2)


# --- Solid mechanics -------------------------------------------------------------


@_register(
    "von_mises_peak_pa",
    "Pa",
    "Maximum von Mises stress anywhere in the cushion at the final step.",
    "max of von_mises_Pa over all 11,615 solid nodes (grn_input.npz), clipped at 0",
)
def _vm_peak(ds: Dataset, case: str) -> float:
    return float(np.max(ds.nodes(case).von_mises_pa))


@_register(
    "von_mises_mean_pa",
    "Pa",
    "Mean von Mises stress over the whole cushion at the final step.",
    "mean of von_mises_Pa over all 11,615 solid nodes (grn_input.npz), clipped at 0",
)
def _vm_mean(ds: Dataset, case: str) -> float:
    return float(np.mean(ds.nodes(case).von_mises_pa))


@_register(
    "von_mises_peak_s",
    "dimensionless",
    "Normalized arc position of the peak von Mises stress among surface-band nodes.",
    "argmax over surface-band nodes projected onto the angle-ordered arc",
)
def _vm_peak_s(ds: Dataset, case: str) -> float:
    s, v = _surface_arc(ds, case, "von_mises_pa")
    return float(s[int(np.argmax(v))])


@_register(
    "von_mises_lr_asymmetry",
    "dimensionless",
    "Ventricular-minus-atrial mean von Mises stress on the surface, over the overall mean.",
    "surface-band nodes projected onto arc, halves split at s=0.5",
)
def _vm_asym(ds: Dataset, case: str) -> float:
    s, v = _surface_arc(ds, case, "von_mises_pa")
    return _asymmetry(s, v)


@_register(
    "mech_clipped_fraction",
    "dimensionless",
    "Fraction of solid nodes whose von Mises exceeds the 100 Pa GRN normalization ceiling.",
    "count(von_mises_Pa > MECH_MAX_PA) / 11,615 solid nodes (full mesh denominator)",
)
def _mech_clipped(ds: Dataset, case: str) -> float:
    return ds.nodes(case).mech_clipped_fraction()


# --- Morphology ------------------------------------------------------------------


@_register(
    "cushion_height_mm",
    "mm",
    "Maximum cushion height above the channel floor at the final step.",
    "max y of step_014/arc_data.npz, converted m -> mm",
)
def _height(ds: Dataset, case: str) -> float:
    return ds.arc(case, FINAL_STEP).height_mm


@_register(
    "arc_length_mm",
    "mm",
    "Traced length of the cushion surface outline at the final step.",
    "cumulative polyline length of the angle-ordered arc; NOT the x-extent chord, "
    "which is what metrics_summary.csv's length_mm reports",
)
def _arc_len(ds: Dataset, case: str) -> float:
    return ds.arc(case, FINAL_STEP).arc_length_mm


# --- Growth ----------------------------------------------------------------------


@_register(
    "growth_mean",
    "dimensionless",
    "Mean per-cell growth multiplier at the final step.",
    "mean of step_014/g_field.npy over 22,769 cells",
)
def _growth_mean(ds: Dataset, case: str) -> float:
    return float(np.mean(ds.growth(case, FINAL_STEP).g))


@_register(
    "growth_pinned_low_fraction",
    "dimensionless",
    "Fraction of cells pinned at the growth floor, i.e. not growing at all.",
    "count(g <= 0.5 + 1e-9) / 22,769 cells in step_014/g_field.npy; note the shipped "
    "floor is 0.5 while solid_solver.py's current default is 0.3, so these runs are "
    "not reproducible from the source as it stands",
)
def _growth_pinned(ds: Dataset, case: str) -> float:
    g = ds.growth(case, FINAL_STEP).g
    return float(np.mean(g <= GROWTH_FLOOR + GROWTH_PIN_TOL))


# --- Gene regulatory network -----------------------------------------------------


@_register(
    "grn_saturated_fraction",
    "dimensionless",
    "Fraction of GRN grid nodes whose mechanical input has hit its normalization ceiling.",
    "count(mech_norm >= 1.0) / rows of results_spatial.csv (80% grid denominator; "
    "differs from mech_clipped_fraction, which uses the full mesh)",
)
def _grn_sat(ds: Dataset, case: str) -> float:
    return ds.grn(case).saturated_fraction()


@_register(
    "endmt_mean_combined",
    "dimensionless",
    "Mean EndMT activity under the combined shear+stress scenario.",
    "mean of EndMT_combined in results_spatial.csv (80% grid)",
)
def _endmt_mean(ds: Dataset, case: str) -> float:
    return float(np.nanmean(ds.grn(case).activity("EndMT", "combined")))


@_register(
    "endmt_sd_combined",
    "dimensionless",
    "Spatial standard deviation of EndMT activity under the combined scenario.",
    "population SD of EndMT_combined in results_spatial.csv (80% grid)",
)
def _endmt_sd(ds: Dataset, case: str) -> float:
    return float(np.nanstd(ds.grn(case).activity("EndMT", "combined")))


# --- Public surface --------------------------------------------------------------

METRICS: Final[dict[str, MetricDef]] = _REGISTRY
METRIC_NAMES: Final[tuple[str, ...]] = tuple(sorted(_REGISTRY))


def metric_value(ds: Dataset, name: str, case: str) -> float:
    """Compute one named metric for one case.

    Args:
        ds: The dataset to read from.
        name: A key of :data:`METRICS`.
        case: Case folder name or condition label.

    Returns:
        The metric value as a float.

    Raises:
        ValueError: If ``name`` is not a registered metric.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown metric {name!r}; registered metrics are {', '.join(METRIC_NAMES)}"
        )
    resolved = ds.resolve_case(case)
    value = float(_REGISTRY[name].fn(ds, resolved))
    LOG.debug("metric %s for %s = %r", name, resolved, value)
    return value


def metric_table(ds: Dataset) -> dict[str, dict[str, float]]:
    """Compute every metric for every case.

    Args:
        ds: The dataset to read from.

    Returns:
        A mapping of metric name to a mapping of case name to value.
    """
    return {
        name: {case: metric_value(ds, name, case) for case in ds.cases()}
        for name in METRIC_NAMES
    }
