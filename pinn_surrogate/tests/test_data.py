"""Stage 1 checks: unit conversion round trip, Re, and split integrity on the packed data."""

from __future__ import annotations

import json
import unittest

import numpy as np

from pinn_surrogate.data import (
    PACKED_PATH,
    SCALES,
    load_full,
    load_packed,
    mean_inlet_speed,
    reynolds,
    to_dim,
    to_nondim,
    trust_weights,
)
from pinn_surrogate.reference import FULL_DIR

FAMILY_B = 1  # index of "B" in pack.FAMILIES


class TestNondim(unittest.TestCase):
    """Round trip and physical sanity of the fixed nondimensionalization."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.pk = load_packed()

    def test_round_trip_packed(self) -> None:
        """to_dim(to_nondim(q)) == q for every packed SI quantity.

        Computed in float64, the round trip is within 1 ulp (rtol 1e-14); cast back to the
        stored float32 it must be bit-identical to Dan's values.
        """
        a, h, mu = self.pk.arrays, self.pk.h, self.pk.mu
        ub = self.pk.u_bar
        cases = {
            "xy": ("length", ub[:, None, None]),
            "uv": ("velocity", ub[:, None, None]),
            "p": ("pressure", ub[:, None]),
            "outline": ("length", ub[:, None, None]),
            "wall_p": ("pressure", ub[:, None]),
            "wall_tau_t": ("stress", ub[:, None]),
            "wall_tau_mag": ("stress", ub[:, None]),
        }
        stored = {"uv": a["uvp"][..., :2], "p": a["uvp"][..., 2]}
        for name, (kind, u) in cases.items():
            x = stored.get(name, a.get(name))
            back = to_dim(to_nondim(x, kind, u, h, mu), kind, u, h, mu)
            with self.subTest(quantity=name):
                np.testing.assert_allclose(back, x.astype(np.float64), rtol=1e-14, atol=0)
                self.assertTrue(np.array_equal(back.astype(x.dtype), x))

    def test_round_trip_every_kind(self) -> None:
        """Every unit kind round-trips on real full-dump fields."""
        f = load_full(next(iter(sorted(FULL_DIR.glob("full_*.npz")))))
        ub, h, mu = float(mean_inlet_speed(f["u_inlet_max"])), float(f["H"]), float(f["mu"])
        field_of = {
            "length": "xy",
            "velocity": "u",
            "pressure": "p",
            "stress": "arc_tau_t",
            "velocity_gradient": "u_y",
            "divergence": "r_div",
            "velocity_laplacian": "lap_u",
            "pressure_gradient": "p_x",
            "momentum": "r_mom_x",
        }
        self.assertEqual(set(field_of), set(SCALES))
        for kind, name in field_of.items():
            with self.subTest(kind=kind):
                back = to_dim(to_nondim(f[name], kind, ub, h, mu), kind, ub, h, mu)
                np.testing.assert_allclose(back, f[name], rtol=1e-14, atol=0)

    def test_reynolds_matches_stored(self) -> None:
        """Re = rho U H / mu with U = (2/3) u_inlet_max reproduces the stored Re."""
        re = reynolds(self.pk.rho, self.pk.u_bar, self.pk.h, self.pk.mu)
        np.testing.assert_allclose(re, self.pk.arrays["Re"], rtol=1e-12)

    def test_poiseuille_pressure_gradient_is_minus_12(self) -> None:
        """Upstream of the cushion the flow is plane Poiseuille, where dp*/dx* = -12.

        This checks the pressure and length scales and U = (2/3) u_max independently of the
        stored Re (a wrong U by a factor 3/2 would give -8 or -18).
        """
        for path in sorted(FULL_DIR.glob("full_*.npz")):
            f = load_full(path)
            ub = float(mean_inlet_speed(f["u_inlet_max"]))
            upstream = f["xy"][:, 0] < -3e-3
            px = to_nondim(f["p_x"][upstream], "pressure_gradient", ub, f["H"], f["mu"])
            with self.subTest(dump=path.stem):
                self.assertAlmostEqual(float(np.median(px)), -12.0, delta=0.01)

    def test_trust_weights(self) -> None:
        """w = min(1, r0/r_rel) is in (0, 1] and is 1 at or below r0."""
        w = trust_weights(np.array([0.0, 1e-4, 1e-2, 1e-1]))
        np.testing.assert_allclose(w, [1.0, 1.0, 1.0, 0.1])


class TestSplits(unittest.TestCase):
    """Split integrity and family balance of the data-size ablation prefixes."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.pk = load_packed()

    def test_no_geometry_in_two_splits(self) -> None:
        """Each geometry id belongs to exactly one split."""
        sets = {s: set(self.pk.geometry_order(s).tolist()) for s in ("train", "val", "test_B")}
        self.assertEqual(sum(len(v) for v in sets.values()), len(set().union(*sets.values())))

    def test_test_b_is_all_family_b(self) -> None:
        """test_B holds only family B, and family B appears nowhere else."""
        fam, split = self.pk.arrays["family"], self.pk.arrays["split"]
        self.assertTrue(np.all(fam[split == 2] == FAMILY_B))
        self.assertFalse(np.any(fam[split != 2] == FAMILY_B))

    def test_packed_order_matches_splits_json(self) -> None:
        """The packed geometry order equals splits.json (train, val, test_B)."""
        split = json.loads((PACKED_PATH.parent / "splits.json").read_text())
        names = [f"g_{g:05d}" for g in self.pk.arrays["geom_id"][::4]]
        self.assertEqual(names, split["train"] + split["val"] + split["test_B"])

    def test_train_prefixes_are_family_balanced(self) -> None:
        """train_subset(n) for n = 20, 80, 160 holds n/4 geometries of each of A, C, D, E."""
        fam = self.pk.arrays["family"]
        for n in (20, 80, 160):
            idx = self.pk.train_subset(n)
            with self.subTest(n=n):
                self.assertEqual(len(idx), 4 * n)  # 4 flows per geometry
                fams, counts = np.unique(fam[idx], return_counts=True)
                self.assertEqual(fams.tolist(), [0, 2, 3, 4])  # A, C, D, E
                self.assertTrue(np.all(counts == 4 * n // 4))
                self.assertTrue(np.all(self.pk.arrays["split"][idx] == 0))


if __name__ == "__main__":
    unittest.main()
