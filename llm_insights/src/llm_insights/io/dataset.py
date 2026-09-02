"""Read-only access to the baked FSG + GRN model output.

This module is the single place that knows where files live on disk and what their
arrays are called. Everything downstream consumes the dataclasses defined here, so a
change in Dan's output layout is a change in exactly one file.

Nothing here reads HDF5. The `.h5` field dumps require h5py, which is unavailable in
the environments this project runs in, so every quantity is sourced from the `.npz`,
`.npy`, and `.csv` artifacts that the model also emits. See ``NOTES_data_formats.md``
for the full provenance audit, including what is *only* available inside the `.h5`
files and therefore unreachable.

Conventions this module enforces, all inherited from ``one_way_fsg_model``:

* **Arc ordering.** Surface points are ordered by descending polar angle about the
  base-line centroid, matching ``visualize_growth.load_profiles``. ``s_norm = 0`` is
  the atrial (upstream, min-x) end; ``s_norm = 1`` is the ventricular end. A plain
  ``argsort(x)`` disagrees with this on the underflow case at every step, so the
  angle sort is reproduced here verbatim rather than approximated.
* **Units.** Coordinates are converted to mm at load. Wall shear stress is carried in
  both Pa (as ``arc_data.npz`` stores it) and dyn/cm^2 (as everything downstream of
  ``process_fsg_results`` stores it) so callers never have to guess which one they hold.
* **Step/geometry lag.** ``step_k/arc_data.npz`` describes the solid geometry at the
  *end of step k-1*. ``ArcProfile.geometry_lag_note`` records this; ``step_000`` is the
  pristine undeformed cap and is flagged by ``ArcProfile.is_pristine``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

# --- Case identity ---------------------------------------------------------------

CONDITION_BY_CASE: Final[dict[str, str]] = {
    "flow_U0p0180": "Underflow",
    "flow_U0p0360": "Healthy",
    "flow_U0p0540": "Overflow",
}
CASE_BY_CONDITION: Final[dict[str, str]] = {v: k for k, v in CONDITION_BY_CASE.items()}

#: Inlet velocity in m/s, from README.txt lines 38-42.
INLET_VELOCITY_M_S: Final[dict[str, float]] = {
    "flow_U0p0180": 0.018,
    "flow_U0p0360": 0.036,
    "flow_U0p0540": 0.054,
}

#: GRN result folders are named by the downsample fraction and the case suffix.
GRN_RESULT_DIR_BY_CASE: Final[dict[str, str]] = {
    "flow_U0p0180": "080_180",
    "flow_U0p0360": "080_360",
    "flow_U0p0540": "080_540",
}

N_STEPS: Final[int] = 15
PA_TO_DYN_CM2: Final[float] = 10.0

#: The five GRN nodes actually written to results_spatial.csv (of 23 simulated).
GRN_NODES: Final[tuple[str, ...]] = ("Snai1", "Snai2", "EndMT", "NICD", "YAP_TAZ")
#: The three input scenarios each node is reported under.
GRN_SCENARIOS: Final[tuple[str, ...]] = ("shear_only", "mech_only", "combined")

#: Normalization ceilings hard-coded in networkpoint.py. Values above these clip.
SHEAR_MAX_DYN_CM2: Final[float] = 30.0
MECH_MAX_PA: Final[float] = 100.0


# --- Value objects ---------------------------------------------------------------


@dataclass(frozen=True)
class ArcProfile:
    """Fluid-side surface fields along the cushion arc for one case and step.

    Attributes:
        case: Flow-case folder name, e.g. ``"flow_U0p0360"``.
        condition: Human label, e.g. ``"Healthy"``.
        step: Growth step index, 0-14.
        s_norm: Normalized arc length in [0, 1], atrial to ventricular. Shape (N,).
        x_mm: Surface point x in mm, deformed frame, in arc order. Shape (N,).
        y_mm: Surface point y in mm, deformed frame, in arc order. Shape (N,).
        wss_pa: Wall shear stress magnitude in Pa, in arc order. Shape (N,).
        wss_dyn_cm2: Same quantity in dyn/cm^2 (``wss_pa * 10``). Shape (N,).
        pressure_pa: Fluid gauge pressure in Pa, in arc order. Shape (N,).
        arc_length_mm: Total traced arc length in mm.
        is_pristine: True for step 0, the undeformed cap, which has no matching solid state.
    """

    case: str
    condition: str
    step: int
    s_norm: np.ndarray
    x_mm: np.ndarray
    y_mm: np.ndarray
    wss_pa: np.ndarray
    wss_dyn_cm2: np.ndarray
    pressure_pa: np.ndarray
    arc_length_mm: float
    is_pristine: bool

    geometry_lag_note: str = field(
        default=(
            "arc_data.npz in step_k is the solid geometry at the END of step k-1; "
            "pair arc(step=k+1) with node fields at step k for exact co-registration."
        ),
        repr=False,
    )

    @property
    def n_points(self) -> int:
        """Number of arc facets in this profile (varies 219-259 by case and step)."""
        return int(self.s_norm.size)

    @property
    def height_mm(self) -> float:
        """Maximum cushion height above the channel floor, in mm."""
        return float(np.max(self.y_mm))


@dataclass(frozen=True)
class NodeField:
    """Solid-mesh nodal fields at step 14, available for all three cases.

    Sourced from ``grn_inputs/downsampled_100pct/grn_input.npz`` (11,615 nodes, the
    full solid mesh; the "100pct" label is literal here, unlike the same-named folder
    under ``extracted_fields/``, which loses ~4.5% of nodes to grid decimation).

    Attributes:
        case: Flow-case folder name.
        condition: Human label.
        step: Always 14; this artifact exists only for the final step.
        x_mm: Node x in mm, deformed frame. Shape (11615,).
        y_mm: Node y in mm, deformed frame. Shape (11615,).
        von_mises_pa: Von Mises stress in Pa, every node, clipped at 0. Shape (11615,).
        wss_dyn_cm2: WSS in dyn/cm^2, NaN at interior nodes. Shape (11615,).
        is_surface: True where the node lies in the KDTree surface band. Shape (11615,).
        von_mises_raw_min_pa: Minimum before the clip-at-zero, for provenance.
    """

    case: str
    condition: str
    step: int
    x_mm: np.ndarray
    y_mm: np.ndarray
    von_mises_pa: np.ndarray
    wss_dyn_cm2: np.ndarray
    is_surface: np.ndarray
    von_mises_raw_min_pa: float

    @property
    def n_nodes(self) -> int:
        """Total solid mesh nodes."""
        return int(self.x_mm.size)

    @property
    def n_surface(self) -> int:
        """Nodes in the surface band (the only ones carrying WSS)."""
        return int(np.count_nonzero(self.is_surface))

    def mech_clipped_fraction(self) -> float:
        """Fraction of nodes whose von Mises exceeds ``MECH_MAX_PA`` and so saturates."""
        return float(np.mean(self.von_mises_pa > MECH_MAX_PA))


@dataclass(frozen=True)
class GRNField:
    """GRN steady-state activities on the 80%-downsampled grid at step 14.

    Attributes:
        case: Flow-case folder name.
        condition: Human label.
        x_mm: Node x in mm. Shape (M,).
        y_mm: Node y in mm. Shape (M,).
        von_mises_pa: Mechanical input in Pa. Shape (M,).
        wss_dyn_cm2: Shear input in dyn/cm^2, NaN at interior. Shape (M,).
        mech_norm: ``von_mises_pa / 100`` clipped to [0, 1]. Shape (M,).
        wss_norm: ``wss_dyn_cm2 / 30`` clipped to [0, 1], NaN at interior. Shape (M,).
        activities: Maps ``"<Node>_<scenario>"`` to its activity array in [0, 1].
    """

    case: str
    condition: str
    x_mm: np.ndarray
    y_mm: np.ndarray
    von_mises_pa: np.ndarray
    wss_dyn_cm2: np.ndarray
    mech_norm: np.ndarray
    wss_norm: np.ndarray
    activities: dict[str, np.ndarray]

    def activity(self, node: str, scenario: str = "combined") -> np.ndarray:
        """Return one activity array.

        Args:
            node: One of ``GRN_NODES``.
            scenario: One of ``GRN_SCENARIOS``.

        Returns:
            The activity array in [0, 1].

        Raises:
            KeyError: If the node/scenario pair was not written to the CSV.
        """
        key = f"{node}_{scenario}"
        if key not in self.activities:
            raise KeyError(
                f"{key!r} not in GRN results; available: {sorted(self.activities)}"
            )
        return self.activities[key]

    def saturated_fraction(self) -> float:
        """Fraction of nodes where ``mech_norm`` has hit its ceiling of exactly 1.0."""
        return float(np.mean(self.mech_norm >= 1.0))


@dataclass(frozen=True)
class GrowthField:
    """Per-cell growth multiplier ``g`` and the reference-frame cell centroids.

    Attributes:
        case: Flow-case folder name.
        condition: Human label.
        step: Growth step index.
        g: Growth multiplier per cell, shipped runs span [0.5, 1.3]. Shape (22769,).
        centroid_x_mm: Cell centroid x in mm, **reference** (undeformed) frame.
        centroid_y_mm: Cell centroid y in mm, reference frame.
    """

    case: str
    condition: str
    step: int
    g: np.ndarray
    centroid_x_mm: np.ndarray
    centroid_y_mm: np.ndarray

    @property
    def n_cells(self) -> int:
        """Number of solid mesh cells."""
        return int(self.g.size)


# --- Arc ordering ----------------------------------------------------------------


def arc_order_and_s(x_mm: np.ndarray, y_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Reproduce ``visualize_growth.load_profiles``' arc ordering and normalized length.

    Points are sorted by descending polar angle about the base-line centroid, which
    traces the outline atrial-to-ventricular even where the cushion leans and x stops
    being monotonic. This is a faithful reimplementation of lines 62-71 of
    ``one_way_fsg_model/visualize_growth.py``, not an approximation of it.

    Args:
        x_mm: Unordered surface x coordinates in mm.
        y_mm: Unordered surface y coordinates in mm.

    Returns:
        A tuple of (ordering indices, normalized arc length in [0, 1], total length in mm).
    """
    x_center = 0.5 * (float(x_mm.min()) + float(x_mm.max()))
    angle = np.arctan2(np.maximum(y_mm, 0.0) + 1e-9, x_mm - x_center)
    order = np.argsort(-angle)
    xs, ys = x_mm[order], y_mm[order]
    seg = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total_mm = float(s[-1])
    s_norm = s / total_mm if total_mm > 0 else s
    return order, s_norm, total_mm


# --- Dataset ---------------------------------------------------------------------


class Dataset:
    """Lazily-loaded, cached read-only view of one FSG results tree.

    Args:
        root: Path to the folder containing ``FSG Results`` and ``AHA GRN Plots``.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._fsg = self.root / "FSG Results"
        self._grn = self.root / "AHA GRN Plots" / "3scenarios_results"
        if not self._fsg.is_dir():
            raise FileNotFoundError(f"no 'FSG Results' under {self.root}")
        LOG.info("dataset root %s", self.root)

    # -- discovery --

    @staticmethod
    def cases() -> tuple[str, ...]:
        """All three flow cases, ordered by inlet velocity."""
        return tuple(CONDITION_BY_CASE)

    @staticmethod
    def condition(case: str) -> str:
        """Human-readable condition label for a case folder name."""
        return CONDITION_BY_CASE[case]

    @staticmethod
    def resolve_case(name: str) -> str:
        """Accept either a case folder name or a condition label and return the case.

        Args:
            name: ``"flow_U0p0360"`` or ``"Healthy"`` (case-insensitive on the label).

        Returns:
            The canonical case folder name.

        Raises:
            KeyError: If the name matches neither form.
        """
        if name in CONDITION_BY_CASE:
            return name
        key = name.strip().capitalize()
        if key in CASE_BY_CONDITION:
            return CASE_BY_CONDITION[key]
        raise KeyError(
            f"unknown case {name!r}; expected one of "
            f"{sorted(CONDITION_BY_CASE)} or {sorted(CASE_BY_CONDITION)}"
        )

    # -- loaders --

    @lru_cache(maxsize=64)
    def arc(self, case: str, step: int) -> ArcProfile:
        """Load the surface arc profile for one case and step.

        Args:
            case: Case folder name or condition label.
            step: Growth step, 0-14.

        Returns:
            The ordered :class:`ArcProfile`.
        """
        case = self.resolve_case(case)
        if not 0 <= step < N_STEPS:
            raise ValueError(f"step {step} out of range 0-{N_STEPS - 1}")
        path = self._fsg / case / f"step_{step:03d}" / "arc_data.npz"
        LOG.debug("loading arc %s step %d from %s", case, step, path)
        with np.load(path) as d:
            x_mm = np.asarray(d["x"]) * 1e3
            y_mm = np.asarray(d["y"]) * 1e3
            tau_pa = np.asarray(d["tau_mag"])
            p_pa = np.asarray(d["p"])
        order, s_norm, total_mm = arc_order_and_s(x_mm, y_mm)
        return ArcProfile(
            case=case,
            condition=CONDITION_BY_CASE[case],
            step=step,
            s_norm=s_norm,
            x_mm=x_mm[order],
            y_mm=y_mm[order],
            wss_pa=tau_pa[order],
            wss_dyn_cm2=tau_pa[order] * PA_TO_DYN_CM2,
            pressure_pa=p_pa[order],
            arc_length_mm=total_mm,
            is_pristine=(step == 0),
        )

    @lru_cache(maxsize=8)
    def nodes(self, case: str) -> NodeField:
        """Load full-mesh nodal von Mises and surface WSS at step 14.

        Args:
            case: Case folder name or condition label.

        Returns:
            The :class:`NodeField`. Von Mises is clipped at 0 to remove the CG1
            projection undershoot (min is about -0.82 Pa on one node); the raw
            minimum is preserved on the dataclass for provenance.
        """
        case = self.resolve_case(case)
        path = self._fsg / case / "grn_inputs" / "downsampled_100pct" / "grn_input.npz"
        LOG.debug("loading nodes %s from %s", case, path)
        with np.load(path) as d:
            vm_raw = np.asarray(d["von_mises_Pa"], dtype=float)
            return NodeField(
                case=case,
                condition=CONDITION_BY_CASE[case],
                step=14,
                x_mm=np.asarray(d["x_m"], dtype=float) * 1e3,
                y_mm=np.asarray(d["y_m"], dtype=float) * 1e3,
                von_mises_pa=np.clip(vm_raw, 0.0, None),
                wss_dyn_cm2=np.asarray(d["wss_mag_dyn_cm2"], dtype=float),
                is_surface=np.asarray(d["is_surface"]).astype(bool),
                von_mises_raw_min_pa=float(vm_raw.min()),
            )

    @lru_cache(maxsize=8)
    def grn(self, case: str) -> GRNField:
        """Load GRN steady-state activities on the 80% grid at step 14.

        Args:
            case: Case folder name or condition label.

        Returns:
            The :class:`GRNField`.
        """
        case = self.resolve_case(case)
        path = self._grn / GRN_RESULT_DIR_BY_CASE[case] / "results_spatial.csv"
        LOG.debug("loading grn %s from %s", case, path)
        df = pd.read_csv(path)
        activities = {
            f"{node}_{scen}": df[f"{node}_{scen}"].to_numpy(dtype=float)
            for node in GRN_NODES
            for scen in GRN_SCENARIOS
            if f"{node}_{scen}" in df.columns
        }
        return GRNField(
            case=case,
            condition=CONDITION_BY_CASE[case],
            x_mm=df["x_m"].to_numpy(dtype=float) * 1e3,
            y_mm=df["y_m"].to_numpy(dtype=float) * 1e3,
            von_mises_pa=df["von_mises_Pa"].to_numpy(dtype=float),
            wss_dyn_cm2=df["wss_mag_dyn_cm2"].to_numpy(dtype=float),
            mech_norm=df["mech_norm"].to_numpy(dtype=float),
            wss_norm=df["wss_norm"].to_numpy(dtype=float),
            activities=activities,
        )

    @lru_cache(maxsize=64)
    def growth(self, case: str, step: int) -> GrowthField:
        """Load the per-cell growth multiplier for one case and step.

        Args:
            case: Case folder name or condition label.
            step: Growth step. Only a subset of steps is staged; a missing step raises.

        Returns:
            The :class:`GrowthField`.
        """
        case = self.resolve_case(case)
        g_path = self._fsg / case / f"step_{step:03d}" / "g_field.npy"
        c_path = self._fsg / case / "cell_centroids.npy"
        LOG.debug("loading growth %s step %d", case, step)
        g = np.load(g_path)
        centroids = np.load(c_path)
        return GrowthField(
            case=case,
            condition=CONDITION_BY_CASE[case],
            step=step,
            g=np.asarray(g, dtype=float),
            centroid_x_mm=np.asarray(centroids[:, 0], dtype=float) * 1e3,
            centroid_y_mm=np.asarray(centroids[:, 1], dtype=float) * 1e3,
        )

    def available_growth_steps(self, case: str) -> tuple[int, ...]:
        """Steps for which ``g_field.npy`` is present on disk for this case."""
        case = self.resolve_case(case)
        found = [
            s
            for s in range(N_STEPS)
            if (self._fsg / case / f"step_{s:03d}" / "g_field.npy").is_file()
        ]
        return tuple(found)

    @lru_cache(maxsize=1)
    def metrics_summary(self) -> pd.DataFrame:
        """Dan's own per-case summary table, used as an independent cross-check.

        Returns:
            The ``comparison_figures/metrics_summary.csv`` table, indexed by case.

        Note:
            The script that generated this file is not in the handoff, so the exact
            definition of its ``peak_xnorm`` column is unknown. Treat it as a
            corroborating source, never as the definition of anything.
        """
        return pd.read_csv(self._fsg / "comparison_figures" / "metrics_summary.csv").set_index(
            "case"
        )
