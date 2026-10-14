"""Stage 2 checks: the reference residual and wall traction reproduce Dan's saved arrays.

Tolerances come from the numbers, not taste:

* Dan's fields are stored as float32 (machine epsilon EPS32 = 1.19e-7). A residual is a
  small difference of large terms, so it is compared pointwise against the size of those
  terms: |ours - Dan's| <= 2 EPS32 * (sum of |momentum terms|). Observed max: 6.5e-8.
* Wall pressure is P1, so the facet-midpoint value is exact up to float32 storage.
* Wall shear needs velocity gradients at the wall. We only have Dan's nodal gradient
  values (presumably a projection of the discontinuous P2 gradient; his solver code is not
  in this repo) and average them to the facet midpoint, so agreement is limited by
  discretization: observed relative L2 of 7.0e-3, 1.2e-2 and 7.7e-4 (growing with Re,
  worst at the upstream shoulder). The tolerance 1.5e-2 sits above that. A wrong
  convention is not hidden by it: the flipped sign gives relative L2 ~2.0.
"""

from __future__ import annotations

import unittest

import numpy as np

from pinn_surrogate.data import load_full, mean_inlet_speed, to_dim
from pinn_surrogate.reference import (
    FULL_DIR,
    momentum_terms_dim,
    nondim_fields,
    ns_residual_dim,
    ns_residual_nondim,
    outlet_traction,
    rel_l2,
    relative_residual,
    wall_traction,
)

EPS32 = float(np.finfo(np.float32).eps)
EPS64 = float(np.finfo(np.float64).eps)
TAU_T_REL_L2_TOL = 1.5e-2
TRACTION_REL_L2_TOL = 5e-3
P_REL_L2_TOL = 1e-6


def _dumps() -> list[tuple[str, dict[str, np.ndarray]]]:
    paths = sorted(FULL_DIR.glob("full_*.npz"))
    if len(paths) != 3:
        raise RuntimeError(f"expected 3 full dumps in {FULL_DIR}, found {len(paths)}")
    return [(p.stem, load_full(p)) for p in paths]


class TestResidual(unittest.TestCase):
    """NS residual: our formula vs Dan's r_mom_x, r_mom_y, r_div, r_rel."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dumps = _dumps()

    def test_momentum_and_divergence_match_dan(self) -> None:
        """Pointwise agreement within 2 EPS32 of the local term scale."""
        for name, f in self.dumps:
            rho, mu = float(f["rho"]), float(f["mu"])
            rx, ry, rdiv = ns_residual_dim(f, rho, mu)
            scale = sum(np.abs(t).sum(axis=0) for t in momentum_terms_dim(f, rho, mu).values())
            div_scale = np.abs(f["u_x"]) + np.abs(f["v_y"])
            with self.subTest(dump=name):
                self.assertTrue(np.all(np.abs(rx - f["r_mom_x"]) <= 2 * EPS32 * scale))
                self.assertTrue(np.all(np.abs(ry - f["r_mom_y"]) <= 2 * EPS32 * scale))
                self.assertTrue(np.all(np.abs(rdiv - f["r_div"]) <= 2 * EPS32 * div_scale))
                # The opposite sign convention must fail badly.
                self.assertGreater(rel_l2(-rx, f["r_mom_x"]), 1.9)

    def test_relative_residual_matches_dan(self) -> None:
        """r_rel = |r_mom|_2 / sum|terms| reproduces Dan's r_rel to float32 precision."""
        for name, f in self.dumps:
            r = relative_residual(f, float(f["rho"]), float(f["mu"]))
            with self.subTest(dump=name):
                self.assertLess(float(np.abs(r - f["r_rel"]).max()), 4 * EPS32)

    def test_nondim_residual_rescales_to_dim(self) -> None:
        """Nondim residual * (mu U / H^2) equals the dimensional residual (float64)."""
        for name, f in self.dumps:
            rho, mu, h = float(f["rho"]), float(f["mu"]), float(f["H"])
            ub = float(mean_inlet_speed(f["u_inlet_max"]))
            dim = ns_residual_dim(f, rho, mu)
            nd = ns_residual_nondim(nondim_fields(f, ub, h, mu), float(f["Re"]))
            # Same cancellation argument in float64: observed max error 1.4 EPS64 * scale.
            mom_scale = sum(np.abs(t).sum(axis=0) for t in momentum_terms_dim(f, rho, mu).values())
            div_scale = np.abs(f["u_x"]) + np.abs(f["v_y"])
            scales = (mom_scale, mom_scale, div_scale)
            kinds = ("momentum", "momentum", "divergence")
            with self.subTest(dump=name):
                for a, b, kind, s in zip(nd, dim, kinds, scales):
                    err = np.abs(to_dim(a, kind, ub, h, mu) - b)
                    self.assertTrue(np.all(err <= 4 * EPS64 * s))


class TestWallTraction(unittest.TestCase):
    """Wall p and signed shear from Dan's nodal gradients vs his arc arrays."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dumps = [(name, f, wall_traction(f, float(f["mu"]))) for name, f in _dumps()]

    def test_facets_are_dan_arc_points(self) -> None:
        """Arc points are the tag-13 facet midpoints in lines order; normals point the same way."""
        for name, f, w in self.dumps:
            n_dan = np.stack([f["arc_nx"], f["arc_ny"]], axis=1)
            with self.subTest(dump=name):
                np.testing.assert_allclose(w["mid"], np.c_[f["arc_x"], f["arc_y"]], atol=1e-10)
                self.assertGreater(float(np.einsum("ij,ij->i", w["n"], n_dan).min()), 1 - 1e-9)

    def test_pressure_matches(self) -> None:
        """arc_p is the fluid pressure at the facet midpoint (not -t.n)."""
        for name, f, w in self.dumps:
            with self.subTest(dump=name):
                self.assertLess(rel_l2(w["p"], f["arc_p"]), P_REL_L2_TOL)

    def test_traction_vector_matches(self) -> None:
        """(arc_tx, arc_ty) = sigma n with n into the fluid."""
        for name, f, w in self.dumps:
            ours, dan = np.c_[w["tx"], w["ty"]], np.c_[f["arc_tx"], f["arc_ty"]]
            with self.subTest(dump=name):
                self.assertLess(rel_l2(ours, dan), TRACTION_REL_L2_TOL)

    def test_signed_shear_matches(self) -> None:
        """tau_t = t . (n_y, -n_x) within discretization; the flipped sign fails."""
        for name, f, w in self.dumps:
            with self.subTest(dump=name):
                self.assertLess(rel_l2(w["tau_t"], f["arc_tau_t"]), TAU_T_REL_L2_TOL)
                self.assertLess(rel_l2(w["tau_mag"], f["arc_tau_mag"]), TAU_T_REL_L2_TOL)
                self.assertGreater(rel_l2(-w["tau_t"], f["arc_tau_t"]), 1.9)


if __name__ == "__main__":
    unittest.main()


class TestOutlet(unittest.TestCase):
    """Which traction-free condition Dan imposed at the outlet (sets the pressure gauge)."""

    def test_outlet_is_symmetric_traction_free(self) -> None:
        """sigma n = 0 with the symmetric stress holds; the do-nothing form does not.

        Nodal gradients are projections, so neither form is satisfied exactly; observed rms
        is <= 0.06 of mu max|u_y| for the symmetric form vs >= 0.4 for do-nothing.
        """
        for name, f in _dumps():
            out = outlet_traction(f, float(f["mu"]))
            with self.subTest(dump=name):
                self.assertLess(float(np.sqrt(np.mean(out["sym"] ** 2))), 0.1 * out["scale"])
                self.assertGreater(
                    float(np.sqrt(np.mean(out["do_nothing"] ** 2))), 0.3 * out["scale"]
                )
