"""Torch tensors for a set of packed solves, and random mini-batches drawn from them.

All the loading, nondimensionalization, trust weights and splits come from ``data.py``;
this module only selects solves, converts to float32 tensors and samples batches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from pinn_surrogate.data import (
    TAG_INTERIOR,
    TAG_OUTLET,
    Packed,
    mean_pressure_gradient,
    node_weights,
)

CORNER_EPS = 1e-6  # outlet nodes this close to y = 0 or 1 are wall corners (no-slip)


def _positions(tag: np.ndarray, value: int) -> np.ndarray:
    """(S, K) node positions with ``tag == value`` per solve; K must be equal for all."""
    rows = [np.flatnonzero(t == value) for t in tag]
    if len({len(r) for r in rows}) != 1:
        raise ValueError(f"solves have different numbers of tag-{value} nodes")
    return np.stack(rows)


@dataclass
class SolveSet:
    """Float32 tensors for ``S`` solves (nondimensional) plus numpy metadata.

    Attributes:
        index: (S,) packed solve indices.
        xy: (S, N, 2) node coordinates.
        uvp: (S, N, 3) true (u*, v*, p*).
        weight: (S, N) data-loss weight (``data.node_weights``).
        tag: (S, N) boundary tag.
        outline: (S, 256, 2) cushion outline.
        normals: (S, 256, 2) wall normals at the outline vertices, into the fluid.
        wall_p, wall_tau_t: (S, 256) Dan's wall pressure and signed wall shear.
        log_re, re: (S,) log Reynolds number and Reynolds number.
        c: (S,) mean pressure gradient from the data (PDE loss scale).
        interior: (S, P) positions of interior nodes.
        outlet: (S, K) positions of outlet nodes; ``outlet_mask`` (S, K) excludes corners.
        flow_k, family, geom_id: (S,) numpy metadata.
    """

    index: np.ndarray
    xy: torch.Tensor
    uvp: torch.Tensor
    weight: torch.Tensor
    tag: torch.Tensor
    outline: torch.Tensor
    normals: torch.Tensor
    wall_p: torch.Tensor
    wall_tau_t: torch.Tensor
    log_re: torch.Tensor
    re: torch.Tensor
    c: torch.Tensor
    mom_scale: torch.Tensor
    div_scale: torch.Tensor
    interior: torch.Tensor
    outlet: torch.Tensor
    outlet_mask: torch.Tensor
    flow_k: np.ndarray
    family: np.ndarray
    geom_id: np.ndarray

    def __len__(self) -> int:
        return len(self.index)

    @classmethod
    def from_packed(cls, pk: Packed, nd: dict[str, np.ndarray], idx: np.ndarray) -> SolveSet:
        """Select solves ``idx`` from the packed data.

        Args:
            pk: ``data.load_packed()`` output.
            nd: ``pk.nondim()`` output (passed in so it is computed once).
            idx: Packed solve indices.
        """
        a = pk.arrays
        tag = a["tag"][idx]
        p = nd["p"][idx]
        outlet = _positions(tag, TAG_OUTLET)
        y_out = np.take_along_axis(nd["xy"][idx][..., 1], outlet, axis=1)
        mask = (y_out > CORNER_EPS) & (y_out < 1 - CORNER_EPS)

        def f32(x: np.ndarray) -> torch.Tensor:
            return torch.tensor(np.asarray(x), dtype=torch.float32)

        return cls(
            index=np.asarray(idx),
            xy=f32(nd["xy"][idx]),
            uvp=f32(np.concatenate([nd["uv"][idx], p[..., None]], axis=-1)),
            weight=f32(node_weights(a["r_rel"][idx], tag)),
            tag=torch.tensor(tag.astype(np.int64)),
            outline=f32(nd["outline"][idx]),
            normals=f32(np.stack([a["wall_nx"][idx], a["wall_ny"][idx]], axis=-1)),
            wall_p=f32(nd["wall_p"][idx]),
            wall_tau_t=f32(nd["wall_tau_t"][idx]),
            log_re=f32(np.log(nd["Re"][idx])),
            re=f32(nd["Re"][idx]),
            c=f32(mean_pressure_gradient(p, tag)),
            mom_scale=f32(nd["mom_scale"][idx]),
            div_scale=f32(nd["div_scale"][idx]),
            interior=torch.tensor(_positions(tag, TAG_INTERIOR)),
            outlet=torch.tensor(outlet),
            outlet_mask=f32(mask),
            flow_k=a["flow_k"][idx].astype(int),
            family=a["families"][a["family"][idx]],
            geom_id=a["geom_id"][idx].astype(int),
        )


@dataclass
class Batch:
    """One training mini-batch (B solves)."""

    xy: torch.Tensor  # (B, n, 2) data nodes
    uvp: torch.Tensor  # (B, n, 3)
    weight: torch.Tensor  # (B, n)
    xy_pde: torch.Tensor  # (B, P, 2) interior collocation nodes
    xy_out: torch.Tensor  # (B, K, 2) outlet nodes
    out_mask: torch.Tensor  # (B, K)
    outline: torch.Tensor  # (B, 256, 2)
    log_re: torch.Tensor  # (B,)
    re: torch.Tensor  # (B, 1)
    c: torch.Tensor  # (B, 1)
    c_mom: torch.Tensor  # (B, P) local momentum-term size at the collocation nodes
    c_div: torch.Tensor  # (B, P) local divergence-term size


def _random_subsets(rng: np.random.Generator, n_rows: int, n_items: int, k: int) -> np.ndarray:
    """(n_rows, k) independent random subsets of range(n_items), without replacement."""
    k = min(k, n_items)
    return np.argsort(rng.random((n_rows, n_items)), axis=1)[:, :k]


def sample_batch(
    s: SolveSet, rng: np.random.Generator, n_solves: int, n_nodes: int, n_pde: int
) -> Batch:
    """Draw ``n_solves`` solves and random node subsets of each.

    Args:
        s: Training solves.
        rng: Numpy generator (seeded by the caller for reproducibility).
        n_solves: Solves per batch (with replacement only if the set is smaller).
        n_nodes: Data nodes per solve, from all packed nodes.
        n_pde: Collocation nodes per solve, from the interior nodes.
    """
    sol = rng.choice(len(s), size=n_solves, replace=len(s) < n_solves)
    nodes = torch.tensor(_random_subsets(rng, n_solves, s.xy.shape[1], n_nodes))
    pde_cols = _random_subsets(rng, n_solves, s.interior.shape[1], n_pde)
    sol_t = torch.tensor(sol)
    rows = sol_t[:, None]
    pde_pos = s.interior[rows, torch.tensor(pde_cols)]
    out_pos = s.outlet[sol_t]
    return Batch(
        xy=s.xy[rows, nodes],
        uvp=s.uvp[rows, nodes],
        weight=s.weight[rows, nodes],
        xy_pde=s.xy[rows, pde_pos],
        xy_out=s.xy[rows, out_pos],
        out_mask=s.outlet_mask[sol_t],
        outline=s.outline[sol_t],
        log_re=s.log_re[sol_t],
        re=s.re[sol_t][:, None],
        c=s.c[sol_t][:, None],
        c_mom=s.mom_scale[rows, pde_pos],
        c_div=s.div_scale[rows, pde_pos],
    )
