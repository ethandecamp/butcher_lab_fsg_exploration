"""Pack a small, fixed subset of Dan's steady CFD dataset into one training file.

The full dataset is 12 GB (1,000 geometries x 4 flows, ~21.6k mesh nodes each). A
proof-of-concept PINN needs far less: this script selects a seeded subset of geometries,
subsamples a fixed number of mesh nodes per solve (oversampling near the cushion wall,
where wall shear is decided), and writes everything into a single compressed ``.npz``.

Everything is stored in **SI units, exactly as Dan saved it**. Nondimensionalization
happens later in ``data.py`` so that a round-trip test can check it.

Usage (numpy only, no torch needed)::

    python3 -m pinn_surrogate.pack --dataset ../steady_state_dataset/dataset_v1_1k \
        --out data/packed_v1.npz
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Subset sizes (geometries). Each geometry contributes all 4 of its flows.
N_TRAIN_PER_FAMILY = 40  # families A, C, D, E -> 160 train geometries
N_VAL_PER_FAMILY = 10  # families A, C, D, E -> 40 in-distribution held-out geometries
N_TEST_B = 40  # family B (real FSG-derived shapes) -> out-of-distribution test
SYNTHETIC_FAMILIES = ("A", "C", "D", "E")
FAMILIES = ("A", "B", "C", "D", "E")

# Node budget per solve. Boundary counts are per mesh boundary tag (Dan's FSG convention:
# 10 inlet, 11 outlet, 12 top wall, 13 cushion arc, 14 bottom wall).
BOUNDARY_BUDGET = {10: 32, 11: 32, 12: 128, 14: 128, 13: 256}
N_NEAR_WALL = 1024  # interior nodes within NEAR_WALL_M of the cushion
N_FAR = 1472  # all other interior nodes
NEAR_WALL_M = 50e-6
N_NODES = sum(BOUNDARY_BUDGET.values()) + N_NEAR_WALL + N_FAR  # 3072

SEED = 0

# v2 pack (2026-10-21): every geometry except v1's val and test_B goes to train, and the
# near-wall band is denser. Near band = all interior nodes within NEAR20_M (capped at
# V2_NEAR20_CAP), topped up from 20-50 um to V2_NEAR_TOTAL. Boundary and far budgets as v1.
NEAR20_M = 20e-6
V2_NEAR20_CAP = 1024
V2_NEAR_TOTAL = 1536
V2_N_NODES = sum(BOUNDARY_BUDGET.values()) + V2_NEAR_TOTAL + N_FAR  # 3584


def _choose(rng: np.random.Generator, idx: np.ndarray, n: int) -> np.ndarray:
    """Pick ``n`` indices from ``idx``; with replacement only if there are too few."""
    if len(idx) == 0:
        raise ValueError("no candidate nodes to sample from")
    return rng.choice(idx, size=n, replace=len(idx) < n)


def select_geometries(index: list[dict], seed: int = SEED) -> dict[str, list[str]]:
    """Return the seeded train / val / test_B geometry ids.

    The train list is ordered so that its prefixes (first 20, 80, 160) are themselves
    family-balanced; the data-size ablation uses those prefixes.
    """
    rng = np.random.default_rng(seed)
    by_family: dict[str, list[str]] = {f: [] for f in FAMILIES}
    for row in index:
        if row["k"] == 0:
            by_family[row["family"]].append(row["geom"])
    split: dict[str, list[str]] = {"train": [], "val": [], "test_B": []}
    per_family_train: dict[str, list[str]] = {}
    for fam in SYNTHETIC_FAMILIES:
        ids = sorted(by_family[fam])
        perm = [ids[i] for i in rng.permutation(len(ids))]
        per_family_train[fam] = perm[:N_TRAIN_PER_FAMILY]
        split["val"] += perm[N_TRAIN_PER_FAMILY : N_TRAIN_PER_FAMILY + N_VAL_PER_FAMILY]
    # Interleave families so any prefix of length 4m holds m geometries per family.
    for i in range(N_TRAIN_PER_FAMILY):
        for fam in SYNTHETIC_FAMILIES:
            split["train"].append(per_family_train[fam][i])
    ids_b = sorted(by_family["B"])
    split["test_B"] = [ids_b[i] for i in rng.permutation(len(ids_b))[:N_TEST_B]]
    return split


def select_geometries_v2(index: list[dict], seed: int = SEED) -> dict[str, list[str]]:
    """v1's val and test_B unchanged; train = v1 train, then every other geometry.

    The extra geometries (all families, B included) follow in a seeded order that cycles
    through families, so v1's train list is a prefix and the 160-shape runs stay nested.
    """
    split = select_geometries(index, seed)
    held = set(split["val"]) | set(split["test_B"]) | set(split["train"])
    rng = np.random.default_rng(seed + 1)
    by_family: dict[str, list[str]] = {}
    for row in index:
        if row["k"] == 0 and row["geom"] not in held:
            by_family.setdefault(row["family"], []).append(row["geom"])
    queues = {f: [sorted(v)[i] for i in rng.permutation(len(v))] for f, v in sorted(by_family.items())}
    extra: list[str] = []
    while any(queues.values()):
        for f in sorted(queues):
            if queues[f]:
                extra.append(queues[f].pop(0))
    split["train"] = split["train"] + extra
    return split


def sample_nodes_v2(
    rng: np.random.Generator, mesh: np.lib.npyio.NpzFile, dist_arc: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """v2 node sampling (``V2_N_NODES`` per geometry); see the constants above."""
    lines, tags = mesh["lines"], mesh["line_tags"]
    on_boundary = np.zeros(len(mesh["points"]), dtype=bool)
    on_boundary[np.unique(lines[np.isin(tags, list(BOUNDARY_BUDGET))])] = True
    chosen, chosen_tag = [], []
    for tag, n in BOUNDARY_BUDGET.items():
        chosen.append(_choose(rng, np.unique(lines[tags == tag]), n))
        chosen_tag.append(np.full(n, tag, dtype=np.int8))
    interior = np.flatnonzero(~on_boundary)
    d = dist_arc[interior]
    p20 = interior[d < NEAR20_M]
    p50 = interior[(d >= NEAR20_M) & (d < NEAR_WALL_M)]
    far = interior[d >= NEAR_WALL_M]
    n20 = min(len(p20), V2_NEAR20_CAP)
    pick20 = rng.choice(p20, size=n20, replace=False)
    for pick, n in ((pick20, n20), (_choose(rng, p50, V2_NEAR_TOTAL - n20), V2_NEAR_TOTAL - n20),
                    (_choose(rng, far, N_FAR), N_FAR)):
        chosen.append(pick)
        chosen_tag.append(np.zeros(n, dtype=np.int8))
    return np.concatenate(chosen), np.concatenate(chosen_tag)


def sample_nodes(
    rng: np.random.Generator, mesh: np.lib.npyio.NpzFile, dist_arc: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (node indices, boundary tag per index; 0 = interior) for one geometry."""
    lines, tags = mesh["lines"], mesh["line_tags"]
    n_pts = len(mesh["points"])
    on_boundary = np.zeros(n_pts, dtype=bool)
    chosen, chosen_tag = [], []
    for tag, n in BOUNDARY_BUDGET.items():
        nodes = np.unique(lines[tags == tag])
        on_boundary[nodes] = True
    for tag, n in BOUNDARY_BUDGET.items():
        nodes = np.unique(lines[tags == tag])
        pick = _choose(rng, nodes, n)
        chosen.append(pick)
        chosen_tag.append(np.full(n, tag, dtype=np.int8))
    interior = np.flatnonzero(~on_boundary)
    near = interior[dist_arc[interior] < NEAR_WALL_M]
    far = interior[dist_arc[interior] >= NEAR_WALL_M]
    for pool, n in ((near, N_NEAR_WALL), (far, N_FAR)):
        chosen.append(_choose(rng, pool, n))
        chosen_tag.append(np.zeros(n, dtype=np.int8))
    return np.concatenate(chosen), np.concatenate(chosen_tag)


def pack(dataset: Path, out: Path, full_dir: Path | None = None, version: str = "v1") -> None:
    """Build the packed training file (and a few full-field dumps) from ``dataset``.

    ``version="v2"`` uses ``select_geometries_v2`` and ``sample_nodes_v2``.
    """
    t0 = time.time()
    index = json.loads((dataset / "cfd_index.json").read_text())
    v2 = version == "v2"
    split = select_geometries_v2(index) if v2 else select_geometries(index)
    sampler = sample_nodes_v2 if v2 else sample_nodes
    n_nodes = V2_N_NODES if v2 else N_NODES
    order = split["train"] + split["val"] + split["test_B"]
    split_of = {g: s for s, gs in split.items() for g in gs}
    rng = np.random.default_rng(SEED)

    n_solves = 4 * len(order)
    xy = np.zeros((n_solves, n_nodes, 2), np.float32)
    uvp = np.zeros((n_solves, n_nodes, 3), np.float32)
    r_rel = np.zeros((n_solves, n_nodes), np.float32)
    dist = np.zeros((n_solves, n_nodes), np.float32)
    tag = np.zeros((n_solves, n_nodes), np.int8)
    # Per-node size of the momentum terms, |rho(u.grad)u| + |grad p| + |mu lap u| summed over
    # x and y (the denominator of Dan's r_rel), and of the divergence terms |u_x| + |v_y|.
    # They let the PINN's residual be measured relative to the local physics, like r_rel.
    mom_scale = np.zeros((n_solves, n_nodes), np.float32)
    div_scale = np.zeros((n_solves, n_nodes), np.float32)
    outline = np.zeros((n_solves, 256, 2), np.float32)
    wall = {k: np.zeros((n_solves, 256), np.float32) for k in ("p", "tau_t", "tau_mag", "nx", "ny")}
    scal = {
        k: np.zeros(n_solves, np.float64)
        for k in ("Re", "u_inlet_max", "blockage", "height", "arc_length")
    }
    geom_id = np.zeros(n_solves, np.int32)
    flow_k = np.zeros(n_solves, np.int8)
    family = np.zeros(n_solves, np.int8)
    split_id = np.zeros(n_solves, np.int8)  # 0 train, 1 val, 2 test_B
    split_codes = {"train": 0, "val": 1, "test_B": 2}

    i = 0
    for gi, gid in enumerate(order):
        gdir = dataset / "geoms" / gid
        meta = json.loads((gdir / "meta.json").read_text())
        mesh = np.load(gdir / "mesh.npz")
        out256 = np.load(gdir / "arc_n256.npy")
        d0 = np.load(gdir / "cfd" / "derived_0.npz")
        idx, idx_tag = sampler(rng, mesh, d0["dist_arc"])
        for k in range(4):
            d = np.load(gdir / "cfd" / f"derived_{k}.npz")
            a = np.load(gdir / "cfd" / f"arc_{k}.npz")
            xy[i] = d["xy"][idx]
            uvp[i] = np.stack([d["u"][idx], d["v"][idx], d["p"][idx]], axis=-1)
            r_rel[i] = d["r_rel"][idx]
            dist[i] = d["dist_arc"][idx]
            rho_, mu_ = 1060.0, 0.0035
            conv_x = rho_ * (d["u"] * d["u_x"] + d["v"] * d["u_y"])
            conv_y = rho_ * (d["u"] * d["v_x"] + d["v"] * d["v_y"])
            mom = (np.abs(conv_x) + np.abs(d["p_x"]) + np.abs(mu_ * d["lap_u"])
                   + np.abs(conv_y) + np.abs(d["p_y"]) + np.abs(mu_ * d["lap_v"]))
            mom_scale[i] = mom[idx]
            div_scale[i] = (np.abs(d["u_x"]) + np.abs(d["v_y"]))[idx]
            tag[i] = idx_tag
            outline[i] = out256
            for key in wall:
                wall[key][i] = a[f"{key}_256"]
            scal["Re"][i] = meta["flow"][k]["Re"]
            scal["u_inlet_max"][i] = meta["flow"][k]["u_inlet_max"]
            scal["blockage"][i] = meta["geom"]["blockage"]
            scal["height"][i] = meta["geom"]["height"]
            scal["arc_length"][i] = meta["geom"]["arc_length"]
            geom_id[i] = meta["id"]
            flow_k[i] = k
            family[i] = FAMILIES.index(meta["family"])
            split_id[i] = split_codes[split_of[gid]]
            i += 1
        if gi % 20 == 0:
            log.info("packed %d/%d geometries (%.0f s)", gi + 1, len(order), time.time() - t0)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        xy=xy, uvp=uvp, r_rel=r_rel, dist_arc=dist, tag=tag, outline=outline,
        mom_scale=mom_scale, div_scale=div_scale,
        wall_p=wall["p"], wall_tau_t=wall["tau_t"], wall_tau_mag=wall["tau_mag"],
        wall_nx=wall["nx"], wall_ny=wall["ny"],
        Re=scal["Re"], u_inlet_max=scal["u_inlet_max"], blockage=scal["blockage"],
        height=scal["height"], arc_length=scal["arc_length"],
        geom_id=geom_id, flow_k=flow_k, family=family, split=split_id,
        families=np.array(FAMILIES), split_names=np.array(["train", "val", "test_B"]),
        rho=np.float64(1060.0), mu=np.float64(0.0035), H=np.float64(0.4e-3),
    )
    splits_name = "splits.json" if not v2 else out.stem + "_splits.json"
    (out.parent / splits_name).write_text(json.dumps(split, indent=1))
    log.info("wrote %s: %d solves x %d nodes (%.0f s)", out, n_solves, n_nodes, time.time() - t0)

    if full_dir is not None:
        # Full-field dumps (every node, every derived field) for physics validation and
        # the single-case PINN: one val geometry at low and high Re, one family-B shape.
        full_dir.mkdir(parents=True, exist_ok=True)
        picks = [(split["val"][0], 0), (split["val"][0], 3), (split["test_B"][0], 2)]
        for gid, k in picks:
            gdir = dataset / "geoms" / gid
            d = dict(np.load(gdir / "cfd" / f"derived_{k}.npz"))
            m = np.load(gdir / "mesh.npz")
            a = np.load(gdir / "cfd" / f"arc_{k}.npz")
            meta = json.loads((gdir / "meta.json").read_text())
            np.savez_compressed(
                full_dir / f"full_{gid}_k{k}.npz",
                **d,
                **{f"arc_{key}": a[key] for key in a.files},
                triangles=m["triangles"], lines=m["lines"], line_tags=m["line_tags"],
                outline=np.load(gdir / "arc_n256.npy"),
                Re=meta["flow"][k]["Re"], u_inlet_max=meta["flow"][k]["u_inlet_max"],
                rho=1060.0, mu=0.0035, H=0.4e-3,
            )
            log.info("wrote full dump %s k=%d", gid, k)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/packed_v1.npz"))
    ap.add_argument("--full-dir", type=Path, default=Path("data/full"))
    ap.add_argument("--version", choices=("v1", "v2"), default="v1")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    full = args.full_dir if args.version == "v1" else None  # v2 reuses v1's full dumps
    pack(args.dataset, args.out, full, args.version)


if __name__ == "__main__":
    main()
