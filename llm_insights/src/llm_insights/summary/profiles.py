"""Middle zoom level: fields collapsed onto normalized arc length.

About 94% of the solid mesh carries no wall shear stress at all — only a thin surface
band touches the fluid — and the cushion is a shallow cap whose interesting variation
runs along its outline. So the honest low-dimensional view of this simulation is a
one-dimensional profile in normalized arc length, not a 2-D field.

A profile here is a fixed-length resampling (25 points by default) of a field along
``s_norm`` in [0, 1], atrial to ventricular. That is small enough to put several of
them in a prompt and still leave room to think, and — this is the part that has to be
earned rather than assumed — it is checked against the full arrays by the
answer-preservation suite in ``tests/test_summary.py``. A compression is only correct
if it preserves the answers to the questions people actually ask of the data.

Resampling is linear interpolation onto a uniform grid, deliberately: it introduces no
smoothing that could hide a plateau. The `MECH_MAX` saturation plateau in the overflow
case must survive into the summary, because it is one of the things the summary exists
to surface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import numpy as np

from llm_insights.io.dataset import Dataset

LOG = logging.getLogger(__name__)

DEFAULT_N_POINTS: Final[int] = 25
FINAL_STEP: Final[int] = 14

#: Fields resolvable on the fluid-side arc, mapped to (ArcProfile attribute, units).
ARC_FIELDS: Final[dict[str, tuple[str, str]]] = {
    "wss": ("wss_dyn_cm2", "dyn/cm^2"),
    "wss_pa": ("wss_pa", "Pa"),
    "pressure": ("pressure_pa", "Pa"),
    "height": ("y_mm", "mm"),
}

#: Fields resolvable on surface-band solid nodes, mapped to (NodeField attribute, units).
NODE_FIELDS: Final[dict[str, tuple[str, str]]] = {
    "von_mises": ("von_mises_pa", "Pa"),
}


@dataclass(frozen=True)
class Profile:
    """One field resampled onto a uniform normalized-arc-length grid.

    Attributes:
        field: Field key, e.g. ``"wss"``.
        case: Case folder name.
        condition: Human label, e.g. ``"Overflow"``.
        step: Growth step the profile describes.
        units: Physical units of ``values``.
        s: Uniform grid over [0, 1], atrial to ventricular. Shape (n,).
        values: Field values interpolated onto ``s``. Shape (n,).
        n_source_points: How many raw points the profile was built from.
        source: Provenance string naming the artifact and the resampling used.
    """

    field: str
    case: str
    condition: str
    step: int
    units: str
    s: np.ndarray
    values: np.ndarray
    n_source_points: int
    source: str

    def peak_s(self) -> float:
        """Normalized arc position of this profile's maximum."""
        return float(self.s[int(np.nanargmax(self.values))])

    def peak_value(self) -> float:
        """Maximum value in this profile."""
        return float(np.nanmax(self.values))

    def as_rows(self, decimals: int = 3) -> list[tuple[float, float]]:
        """Return ``(s, value)`` pairs rounded for compact display."""
        return [
            (round(float(a), 3), round(float(b), decimals))
            for a, b in zip(self.s, self.values, strict=True)
        ]


def _resample(s_src: np.ndarray, v_src: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate a field onto a uniform grid of ``n`` points over [0, 1]."""
    finite = np.isfinite(v_src)
    if not finite.any():
        raise ValueError("field has no finite values to resample")
    s_clean, v_clean = s_src[finite], v_src[finite]
    order = np.argsort(s_clean)
    s_clean, v_clean = s_clean[order], v_clean[order]
    grid = np.linspace(0.0, 1.0, n)
    return grid, np.interp(grid, s_clean, v_clean)


def arc_profile(
    ds: Dataset, case: str, field: str, n: int = DEFAULT_N_POINTS, step: int = FINAL_STEP
) -> Profile:
    """Resample a fluid-side surface field onto the arc grid.

    Args:
        ds: The dataset to read from.
        case: Case folder name or condition label.
        field: One of :data:`ARC_FIELDS`.
        n: Number of grid points.
        step: Growth step.

    Returns:
        The resampled :class:`Profile`.

    Raises:
        ValueError: If ``field`` is not an arc field.
    """
    if field not in ARC_FIELDS:
        raise ValueError(
            f"{field!r} is not a surface field; choose from {', '.join(sorted(ARC_FIELDS))}"
        )
    attr, units = ARC_FIELDS[field]
    arc = ds.arc(case, step)
    grid, values = _resample(arc.s_norm, np.asarray(getattr(arc, attr), dtype=float), n)
    LOG.debug("arc profile %s %s step %d -> %d points", case, field, step, n)
    return Profile(
        field=field,
        case=arc.case,
        condition=arc.condition,
        step=step,
        units=units,
        s=grid,
        values=values,
        n_source_points=arc.n_points,
        source=(
            f"step_{step:03d}/arc_data.npz, angle-ordered, linearly resampled from "
            f"{arc.n_points} facets onto {n} uniform points"
        ),
    )


def node_profile(ds: Dataset, case: str, field: str, n: int = DEFAULT_N_POINTS) -> Profile:
    """Resample a surface-band nodal field onto the arc grid.

    Only nodes in the surface band have an arc position, so the interior — about 94%
    of the mesh — is dropped before resampling.

    Args:
        ds: The dataset to read from.
        case: Case folder name or condition label.
        field: One of :data:`NODE_FIELDS`.
        n: Number of grid points.

    Returns:
        The resampled :class:`Profile`.

    Raises:
        ValueError: If ``field`` is not a nodal field.
    """
    if field not in NODE_FIELDS:
        raise ValueError(
            f"{field!r} is not a nodal field; choose from {', '.join(sorted(NODE_FIELDS))}"
        )
    from llm_insights.harness.primitives import arc_projection

    attr, units = NODE_FIELDS[field]
    nodes = ds.nodes(case)
    s_src, v_src = arc_projection(nodes, getattr(nodes, attr))
    grid, values = _resample(s_src, v_src, n)
    LOG.debug("node profile %s %s -> %d points", case, field, n)
    return Profile(
        field=field,
        case=nodes.case,
        condition=nodes.condition,
        step=nodes.step,
        units=units,
        s=grid,
        values=values,
        n_source_points=nodes.n_surface,
        source=(
            f"grn_input.npz surface-band nodes ({nodes.n_surface} of {nodes.n_nodes}), "
            f"angle-ordered, linearly resampled onto {n} uniform points"
        ),
    )


def profile(ds: Dataset, case: str, field: str, n: int = DEFAULT_N_POINTS, step: int = FINAL_STEP):
    """Resolve a field to whichever profile builder owns it.

    Args:
        ds: The dataset to read from.
        case: Case folder name or condition label.
        field: A key of :data:`ARC_FIELDS` or :data:`NODE_FIELDS`.
        n: Number of grid points.
        step: Growth step, ignored for nodal fields, which exist only at step 14.

    Returns:
        The resampled :class:`Profile`.

    Raises:
        ValueError: If the field is unknown, or a nodal field is asked for at a step
            other than 14, where it does not exist.
    """
    if field in ARC_FIELDS:
        return arc_profile(ds, case, field, n=n, step=step)
    if field in NODE_FIELDS:
        if step != FINAL_STEP:
            raise ValueError(
                f"nodal field {field!r} exists only at step {FINAL_STEP}; step {step} was asked "
                "for. Per-step solid stress lives in solid_fields.h5, which needs h5py."
            )
        return node_profile(ds, case, field, n=n)
    known = sorted(set(ARC_FIELDS) | set(NODE_FIELDS))
    raise ValueError(f"unknown field {field!r}; choose from {', '.join(known)}")


def height_trajectory(ds: Dataset, case: str) -> tuple[np.ndarray, np.ndarray]:
    """Cushion height at every growth step, for temporal claims.

    Args:
        ds: The dataset to read from.
        case: Case folder name or condition label.

    Returns:
        A tuple of (step indices, height in mm). Step 0 is the pristine undeformed cap
        and is included; callers making growth claims usually want to drop it.
    """
    from llm_insights.io.dataset import N_STEPS

    steps = np.arange(N_STEPS)
    heights = np.array([ds.arc(case, int(s)).height_mm for s in steps], dtype=float)
    return steps, heights
