"""v2 additions: wall-distance trunk input, v2 geometry split and node sampling.

The torch part checks the hard constraints still hold with the extra input and that a v1
checkpoint still reproduces its committed eval.json numbers after the model refactor.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from pinn_surrogate.config import ModelConfig
from pinn_surrogate.pack import (
    BOUNDARY_BUDGET,
    N_FAR,
    NEAR20_M,
    NEAR_WALL_M,
    V2_N_NODES,
    V2_NEAR_TOTAL,
    sample_nodes_v2,
    select_geometries,
    select_geometries_v2,
)

try:
    import torch

    from pinn_surrogate.data import load_packed
    from pinn_surrogate.dataset import SolveSet
    from pinn_surrogate.evaluate import frozen, predict_wall
    from pinn_surrogate.metrics import rel_l2
    from pinn_surrogate.model import ConditionedPINN, load_checkpoint

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

ROOT = Path(__file__).resolve().parents[1]
V1_RUN = ROOT / "runs" / "grid" / "n160_physoff_s0"


def fake_index() -> list[dict]:
    sizes = {"A": 300, "B": 175, "C": 175, "D": 175, "E": 175}
    rows, gid = [], 0
    for fam, n in sizes.items():
        for _ in range(n):
            for k in range(4):
                rows.append({"geom": f"g_{gid:05d}", "family": fam, "k": k})
            gid += 1
    return rows


class TestSplitV2(unittest.TestCase):
    def test_v1_held_out_sets_unchanged_and_train_is_superset(self) -> None:
        idx = fake_index()
        v1, v2 = select_geometries(idx), select_geometries_v2(idx)
        self.assertEqual(v1["val"], v2["val"])
        self.assertEqual(v1["test_B"], v2["test_B"])
        self.assertEqual(v2["train"][: len(v1["train"])], v1["train"])
        self.assertEqual(len(v2["train"]), 1000 - 80)
        self.assertEqual(len(set(v2["train"])), len(v2["train"]))
        self.assertFalse(set(v2["train"]) & (set(v2["val"]) | set(v2["test_B"])))


class TestNodesV2(unittest.TestCase):
    def _mesh(self, n_near20: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        n = 6000
        lines, tags, start = [], [], 0
        for tag, budget in BOUNDARY_BUDGET.items():
            seg = np.arange(start, start + budget + 10)
            lines.append(np.stack([seg[:-1], seg[1:]], 1)); tags.append(np.full(len(seg) - 1, tag))
            start += budget + 10
        dist = np.full(n, 1e-3)
        interior = np.arange(start, n)
        dist[interior[:n_near20]] = rng.uniform(0, NEAR20_M * 0.99, n_near20)
        dist[interior[n_near20 : n_near20 + 900]] = rng.uniform(NEAR20_M, NEAR_WALL_M * 0.99, 900)
        mesh = {"points": np.zeros((n, 2)), "lines": np.concatenate(lines), "line_tags": np.concatenate(tags)}
        return mesh, dist, start

    def test_counts_and_bands(self) -> None:
        for n20 in (320, 830, 2000):
            mesh, dist, first_interior = self._mesh(n20)
            idx, tag = sample_nodes_v2(np.random.default_rng(0), mesh, dist)
            self.assertEqual(len(idx), V2_N_NODES)
            inter = idx[tag == 0]
            self.assertEqual(len(inter), V2_NEAR_TOTAL + N_FAR)
            self.assertTrue((inter >= first_interior).all())  # no boundary node tagged interior
            near = inter[:V2_NEAR_TOTAL]
            self.assertTrue((dist[near] < NEAR_WALL_M).all())
            k20 = np.unique(near[dist[near] < NEAR20_M])
            self.assertEqual(len(k20), min(n20, 1024))  # all <20 um nodes taken, no repeats
            self.assertTrue((dist[inter[V2_NEAR_TOTAL:]] >= NEAR_WALL_M).all())


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestWallDistInput(unittest.TestCase):
    def test_constraints_hold_and_param_count(self) -> None:
        pk = load_packed(); nd = pk.nondim()
        o = torch.tensor(nd["outline"][:32], dtype=torch.float32)
        lr = torch.tensor(np.log(nd["Re"][:32]), dtype=torch.float32)
        cfg = ModelConfig(wall_dist_input=True, width=32, depth=3)
        model = ConditionedPINN(cfg)
        base = ConditionedPINN(ModelConfig(width=32, depth=3)).n_params()
        self.assertEqual(model.n_params(), base + 3 * 32)
        g = torch.Generator().manual_seed(3)
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)
            u, v, _ = model(o, o, lr)
            self.assertEqual(float(u.abs().max()), 0.0)
            self.assertEqual(float(v.abs().max()), 0.0)
            y = torch.linspace(0, 1, 21)
            pts = torch.stack([torch.full_like(y, -10.0), y], -1).expand(32, 21, 2)
            u, v, _ = model(pts, o, lr)
            torch.testing.assert_close(u, (6 * y * (1 - y)).expand_as(u), rtol=0, atol=1e-6)
            self.assertEqual(float(v.abs().max()), 0.0)

    @unittest.skipUnless((V1_RUN / "model.pt").exists(), "v1 model.pt not present")
    def test_v1_checkpoint_reproduces_committed_eval(self) -> None:
        """The model refactor must not change v1 predictions (eval.json was written by v1 code)."""
        model, _ = load_checkpoint(V1_RUN / "model.pt")
        rows = json.loads((V1_RUN / "eval.json").read_text())["splits"]["test_B"]["per_solve"][:6]
        pk = load_packed()
        s = SolveSet.from_packed(pk, pk.nondim(), np.array([r["solve"] for r in rows]))
        with frozen(model):
            for i, r in enumerate(rows):
                tau = predict_wall(model, s, i)["tau_t"]
                self.assertAlmostEqual(rel_l2(tau, s.wall_tau_t[i].numpy()), r["tau_rel_l2"], places=5)


if __name__ == "__main__":
    unittest.main()
