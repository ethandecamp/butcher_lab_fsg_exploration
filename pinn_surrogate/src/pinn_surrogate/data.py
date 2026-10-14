"""Load the packed CFD subset and convert between SI and nondimensional units (numpy only).

Nondimensionalization (fixed in PLAN.md, implemented exactly here):

* lengths      x* = x / H                        (H = channel height = 0.4 mm)
* velocity     u* = u / U,   U = (2/3) u_inlet_max (mean speed of the parabolic inlet)
* pressure     p* = p H / (mu U)                 (viscous scale, well behaved as Re -> 0)
* stress       tau* = tau H / (mu U)             (same scale as pressure)

With these, steady Navier-Stokes reads ``Re (u*.grad*) u* = -grad* p* + lap* u*`` and
``div* u* = 0`` with ``Re = rho U H / mu``. Derivative scales follow by the chain rule
(see ``SCALES``). Torch code later wraps these arrays; nothing here imports torch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

PACKED_PATH = Path(__file__).resolve().parents[2] / "data" / "packed_v1.npz"
SPLIT_CODES = {"train": 0, "val": 1, "test_B": 2}
FLOWS_PER_GEOM = 4

# Trust-weight reference level for Dan's relative residual ``r_rel``. Nodes with r_rel <= R0
# get full weight; worse nodes are down-weighted in proportion. 1e-2 is ~70x the bulk
# median (1.3e-4 for packed interior nodes) and about the near-wall median (8e-3 within
# 5 um), so the bulk is untouched and only the clearly unconverged tail is discounted.
TRUST_R0 = 1e-2


# Each quantity's SI unit is U**a * H**b * mu**c; SCALES maps kind -> (a, b, c).
SCALES: dict[str, tuple[int, int, int]] = {
    "length": (0, 1, 0),  # H
    "velocity": (1, 0, 0),  # U
    "pressure": (1, -1, 1),  # mu U / H
    "stress": (1, -1, 1),  # mu U / H
    "velocity_gradient": (1, -1, 0),  # U / H
    "divergence": (1, -1, 0),  # U / H
    "velocity_laplacian": (1, -2, 0),  # U / H^2
    "pressure_gradient": (1, -2, 1),  # mu U / H^2
    "momentum": (1, -2, 1),  # mu U / H^2, units of a momentum residual (N/m^3)
}


def unit(kind: str, u_bar: np.ndarray | float, h: float, mu: float) -> np.ndarray | float:
    """Return the SI size of one nondimensional unit of ``kind``.

    Args:
        kind: One of the keys of ``SCALES``.
        u_bar: Mean inlet speed U in m/s (scalar or array broadcastable to the values).
        h: Channel height H in m.
        mu: Dynamic viscosity in Pa s.

    Returns:
        The dimensional scale, so that ``dimensional = nondimensional * unit``.
    """
    if kind not in SCALES:
        raise KeyError(f"unknown quantity kind {kind!r}; expected one of {sorted(SCALES)}")
    a, b, c = SCALES[kind]
    return np.asarray(u_bar, dtype=np.float64) ** a * h**b * mu**c


def mean_inlet_speed(u_inlet_max: np.ndarray | float) -> np.ndarray | float:
    """Mean speed of a parabolic (plane Poiseuille) inlet: U = (2/3) u_max."""
    return (2.0 / 3.0) * np.asarray(u_inlet_max, dtype=np.float64)


def reynolds(rho: float, u_bar: np.ndarray | float, h: float, mu: float) -> np.ndarray | float:
    """Reynolds number Re = rho U H / mu."""
    return rho * np.asarray(u_bar, dtype=np.float64) * h / mu


def to_nondim(
    value: np.ndarray, kind: str, u_bar: np.ndarray | float, h: float, mu: float
) -> np.ndarray:
    """Convert an SI array to nondimensional units (computed in float64).

    Args:
        value: SI values.
        kind: Quantity kind, a key of ``SCALES``.
        u_bar: Mean inlet speed, broadcastable against ``value``.
        h: Channel height in m.
        mu: Dynamic viscosity in Pa s.

    Returns:
        ``value / scale`` as float64.
    """
    return np.asarray(value, dtype=np.float64) / unit(kind, u_bar, h, mu)


def to_dim(
    value: np.ndarray, kind: str, u_bar: np.ndarray | float, h: float, mu: float
) -> np.ndarray:
    """Inverse of ``to_nondim``: nondimensional array back to SI (float64)."""
    return np.asarray(value, dtype=np.float64) * unit(kind, u_bar, h, mu)


def trust_weights(r_rel: np.ndarray, r0: float = TRUST_R0) -> np.ndarray:
    """Per-node data-loss weight w = min(1, r0 / r_rel).

    ``r_rel`` is Dan's pointwise relative momentum residual (|r_mom| divided by the sum of
    absolute momentum terms; see ``reference.relative_residual``). Inverse weighting means
    a node whose FEM derivatives violate the PDE by 10x more than r0 counts 10x less.

    Caveat: ``r_rel`` measures how well the *projected derivatives* satisfy the PDE, which is
    poor on boundary nodes (median ~3e-2 on the cushion) even where the nodal values are
    exact (no-slip velocity is imposed). Apply these weights to interior nodes only.

    Args:
        r_rel: Relative residual, any shape, non-negative.
        r0: Residual level at and below which a node gets full weight.

    Returns:
        Weights in (0, 1], same shape as ``r_rel``, float64.
    """
    r = np.asarray(r_rel, dtype=np.float64)
    with np.errstate(divide="ignore"):
        return np.minimum(1.0, r0 / r)


@dataclass(frozen=True)
class Packed:
    """The packed dataset, SI units exactly as stored by ``pack.py``.

    Attributes:
        arrays: Every array in the ``.npz`` file, keyed by its stored name.
        rho: Density in kg/m^3.
        mu: Dynamic viscosity in Pa s.
        h: Channel height in m.
    """

    arrays: dict[str, np.ndarray]
    rho: float
    mu: float
    h: float

    @property
    def n_solves(self) -> int:
        """Number of (geometry, flow) solves."""
        return len(self.arrays["Re"])

    @property
    def u_bar(self) -> np.ndarray:
        """Mean inlet speed U per solve, shape (n_solves,)."""
        return mean_inlet_speed(self.arrays["u_inlet_max"])

    def nondim(self) -> dict[str, np.ndarray]:
        """Nondimensional versions of the fields a PINN trains on.

        Returns:
            Dict with ``xy`` (n, nodes, 2), ``uv`` (n, nodes, 2), ``p`` (n, nodes),
            ``outline`` (n, 256, 2), ``wall_p``, ``wall_tau_t``, ``wall_tau_mag`` (n, 256),
            and ``Re`` (n,). Pressures keep Dan's gauge (traction-free outlet).
        """
        a, ub = self.arrays, self.u_bar
        node_u = ub[:, None, None]
        wall_u = ub[:, None]
        h, mu = self.h, self.mu
        return {
            "xy": to_nondim(a["xy"], "length", node_u, h, mu),
            "uv": to_nondim(a["uvp"][..., :2], "velocity", node_u, h, mu),
            "p": to_nondim(a["uvp"][..., 2], "pressure", wall_u, h, mu),
            "outline": to_nondim(a["outline"], "length", node_u, h, mu),
            "wall_p": to_nondim(a["wall_p"], "pressure", wall_u, h, mu),
            "wall_tau_t": to_nondim(a["wall_tau_t"], "stress", wall_u, h, mu),
            "wall_tau_mag": to_nondim(a["wall_tau_mag"], "stress", wall_u, h, mu),
            "Re": np.asarray(a["Re"], dtype=np.float64),
            # Momentum terms are pressure / length: scale mu U / H^2. Divergence: U / H.
            "mom_scale": np.asarray(a["mom_scale"], np.float64) * h**2 / (mu * wall_u),
            "div_scale": np.asarray(a["div_scale"], np.float64) * h / wall_u,
        }

    def split_indices(self, name: str) -> np.ndarray:
        """Solve indices (packed order) belonging to split ``name`` (train / val / test_B)."""
        return np.flatnonzero(self.arrays["split"] == SPLIT_CODES[name])

    def geometry_order(self, name: str) -> np.ndarray:
        """Geometry ids of split ``name`` in packed order (first appearance), unique."""
        gids = self.arrays["geom_id"][self.split_indices(name)]
        _, first = np.unique(gids, return_index=True)
        return gids[np.sort(first)]

    def train_subset(self, n_geoms: int) -> np.ndarray:
        """Solve indices of the first ``n_geoms`` train geometries (all their flows).

        ``pack.py`` interleaves the train families (A, C, D, E, A, C, ...), so prefixes whose
        length is a multiple of 4 are family-balanced; ``tests/test_data.py`` checks this.

        Args:
            n_geoms: Number of train geometries to keep, 1..n_train.

        Returns:
            Solve indices in packed order, length ``FLOWS_PER_GEOM * n_geoms``.
        """
        order = self.geometry_order("train")
        if not 1 <= n_geoms <= len(order):
            raise ValueError(f"n_geoms must be in [1, {len(order)}], got {n_geoms}")
        idx = self.split_indices("train")
        keep = np.isin(self.arrays["geom_id"][idx], order[:n_geoms])
        return idx[keep]


def load_packed(path: Path = PACKED_PATH) -> Packed:
    """Load ``packed_v1.npz`` fully into memory (~100 MB).

    Args:
        path: Location of the packed file.

    Returns:
        A ``Packed`` instance.
    """
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    log.info("loaded %s: %d solves", path, len(arrays["Re"]))
    return Packed(
        arrays=arrays,
        rho=float(arrays["rho"]),
        mu=float(arrays["mu"]),
        h=float(arrays["H"]),
    )


def load_full(path: Path) -> dict[str, np.ndarray]:
    """Load one full-field dump ``data/full/full_<gid>_k<k>.npz``.

    Floating fields are promoted to float64 (Dan stores nodal fields as float32) so that
    reference computations are not limited by float32 arithmetic.

    Args:
        path: The dump file.

    Returns:
        Dict of arrays keyed by stored name.
    """
    with np.load(path) as z:
        return {k: (z[k].astype(np.float64) if z[k].dtype.kind == "f" else z[k]) for k in z.files}


# Boundary tags of packed nodes (Dan's FSG convention; 0 = interior, set by pack.py).
TAG_INTERIOR, TAG_INLET, TAG_OUTLET, TAG_TOP, TAG_CUSHION, TAG_BOTTOM = 0, 10, 11, 12, 13, 14
CHANNEL_LENGTH = 20.0  # nondim, x* from -10 to 10


def node_weights(r_rel: np.ndarray, tag: np.ndarray, r0: float = TRUST_R0) -> np.ndarray:
    """Data-loss weight per packed node: trust weight on interior nodes, 1 on boundaries.

    Boundary nodes have large ``r_rel`` because projected derivatives are poor there, not
    because their nodal values are wrong (no-slip and the inlet profile are imposed
    exactly), so ``trust_weights`` is not applied to them.

    Args:
        r_rel: Dan's relative residual, any shape.
        tag: Boundary tag per node, same shape (0 = interior).
        r0: Trust reference level.

    Returns:
        Weights in (0, 1], float64, same shape.
    """
    return np.where(np.asarray(tag) == TAG_INTERIOR, trust_weights(r_rel, r0), 1.0)


def mean_pressure_gradient(p: np.ndarray, tag: np.ndarray) -> np.ndarray:
    """Per-solve mean pressure gradient (p_inlet - p_outlet) / L from the data, nondim.

    Used as the fixed per-solve scale of the momentum residual in the PINN loss: the
    residual's size grows with the pressure level of the solve (12 for an empty channel,
    12.1-114 over the packed solves, median 14.9), and dividing by this puts all solves on a
    comparable footing. It comes from the data, never from the network, so it cannot be
    gamed by the optimizer.

    Args:
        p: Nondimensional pressure, (n_solves, n_nodes).
        tag: Boundary tags, same shape.

    Returns:
        (n_solves,) values of (mean p* over inlet nodes - mean p* over outlet nodes) / 20.
    """
    tag = np.asarray(tag)
    p = np.asarray(p, dtype=np.float64)
    inlet = np.where(tag == TAG_INLET, p, 0.0).sum(1) / (tag == TAG_INLET).sum(1)
    outlet = np.where(tag == TAG_OUTLET, p, 0.0).sum(1) / (tag == TAG_OUTLET).sum(1)
    return (inlet - outlet) / CHANNEL_LENGTH
