"""Torch physics vs closed forms and vs the numpy reference (skipped without torch).

(a) Plane Poiseuille flow: zero residual, wall shear +6 (bottom) / -6 (top), outlet traction.
(b) Kovasznay flow (exact NS solution, zero residual) and a manufactured divergence-free
    field whose nonzero residual was derived by hand.
(c) ``wall_quantities`` on a synthetic mesh equals ``reference.wall_traction``.

Everything runs in float64 so tolerances can be tight.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from pinn_surrogate.reference import wall_traction

try:
    import torch

    from pinn_surrogate import physics

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

PI = math.pi


def _points(n: int, lo: tuple[float, float], hi: tuple[float, float], seed: int = 0):
    """(1, n, 2) float64 random points in a box."""
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.uniform(lo, hi, size=(1, n, 2)), dtype=torch.float64)


def poiseuille(xy):
    """u = 6y(1-y), v = 0, p = 12(10-x): the empty-channel solution in our units."""
    x, y = xy[..., 0], xy[..., 1]
    return 6 * y * (1 - y), 0 * x, 12 * (10 - x)


def kovasznay(re: float):
    """Kovasznay flow scaled to Re (u.grad)u = -grad p + lap u (so p = Re * p_textbook)."""
    lam = re / 2 - math.sqrt(re**2 / 4 + 4 * PI**2)

    def fn(xy):
        x, y = xy[..., 0], xy[..., 1]
        e = torch.exp(lam * x)
        u = 1 - e * torch.cos(2 * PI * y)
        v = lam / (2 * PI) * e * torch.sin(2 * PI * y)
        p = re * 0.5 * (1 - torch.exp(2 * lam * x))
        return u, v, p

    return fn


def manufactured(xy):
    """Stream function psi = sin(pi x) sin(pi y) (so div u = 0) with p = x^2 y."""
    x, y = xy[..., 0], xy[..., 1]
    u = PI * torch.sin(PI * x) * torch.cos(PI * y)
    v = -PI * torch.cos(PI * x) * torch.sin(PI * y)
    return u, v, x**2 * y


def manufactured_residual(xy, re: float):
    """Hand-derived residual of ``manufactured``.

    (u.grad)u = (pi^3/2)(sin 2 pi x, sin 2 pi y), lap u = -2 pi^2 u, grad p = (2xy, x^2), so
    r_x = Re pi^3/2 sin(2 pi x) + 2xy + 2 pi^2 u and r_y = Re pi^3/2 sin(2 pi y) + x^2 + 2 pi^2 v.
    """
    x, y = xy[..., 0], xy[..., 1]
    u, v, _ = manufactured(xy)
    r_x = re * PI**3 / 2 * torch.sin(2 * PI * x) + 2 * x * y + 2 * PI**2 * u
    r_y = re * PI**3 / 2 * torch.sin(2 * PI * y) + x**2 + 2 * PI**2 * v
    return r_x, r_y


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestResidual(unittest.TestCase):
    """NS residual via autograd."""

    def test_poiseuille_residual_is_zero(self) -> None:
        """(a) Momentum and continuity vanish everywhere for Poiseuille flow."""
        for re in (0.2, 60.0):
            r = physics.ns_residual(poiseuille, _points(200, (-10, 0), (10, 1)), re)
            for comp in r:
                self.assertLess(float(comp.abs().max()), 1e-11)

    def test_kovasznay_residual_is_zero(self) -> None:
        """(b) An exact nonlinear NS solution has zero residual (convection term included)."""
        for re in (1.0, 20.0, 60.0):
            xy = _points(300, (-0.5, -0.5), (1.0, 1.5), seed=1)
            r_x, r_y, r_div = physics.ns_residual(kovasznay(re), xy, re)
            u, v, _ = kovasznay(re)(xy)
            scale = re * float(torch.max(u.abs() + v.abs())) ** 2 + 4 * PI**2 * 10
            for comp in (r_x, r_y):
                self.assertLess(float(comp.abs().max()) / scale, 1e-12)
            self.assertLess(float(r_div.abs().max()), 1e-11)

    def test_manufactured_residual_matches_closed_form(self) -> None:
        """(b) A nonzero residual equals the hand-derived formula; continuity is zero."""
        re = 7.5
        xy = _points(300, (-1.0, -1.0), (1.0, 1.0), seed=2)
        r_x, r_y, r_div = physics.ns_residual(manufactured, xy, re)
        want_x, want_y = manufactured_residual(xy, re)
        torch.testing.assert_close(r_x, want_x, rtol=1e-12, atol=1e-10)
        torch.testing.assert_close(r_y, want_y, rtol=1e-12, atol=1e-10)
        self.assertLess(float(r_div.abs().max()), 1e-12)
        self.assertGreater(float(want_x.abs().max()), 10.0)  # the test is not vacuous

    def test_batched_re(self) -> None:
        """Re given per solve as (B, 1) broadcasts over that solve's points."""
        xy = torch.cat([_points(50, (-1, -1), (1, 1), s) for s in (3, 4)])
        re = torch.tensor([[2.0], [9.0]], dtype=torch.float64)
        r_x, _, _ = physics.ns_residual(manufactured, xy, re)
        for b in range(2):
            want, _ = manufactured_residual(xy[b : b + 1], float(re[b]))
            torch.testing.assert_close(r_x[b : b + 1], want, rtol=1e-12, atol=1e-10)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestWallAndOutlet(unittest.TestCase):
    """Wall traction and outlet traction."""

    def test_poiseuille_wall_shear(self) -> None:
        """(a) Bottom wall n = (0, 1): tau_t = +6; top wall n = (0, -1): tau_t = -6; p exact."""
        x = torch.linspace(-9, 9, 7, dtype=torch.float64)
        for y0, ny, want in ((0.0, 1.0, 6.0), (1.0, -1.0, -6.0)):
            xy = torch.stack([x, torch.full_like(x, y0)], dim=-1)[None]
            n = torch.tensor([0.0, ny], dtype=torch.float64).expand_as(xy)
            wq = physics.wall_quantities(poiseuille, xy, n, offset=0.0)
            torch.testing.assert_close(wq["tau_t"], torch.full_like(wq["tau_t"], want))
            torch.testing.assert_close(wq["p"], 12 * (10 - x)[None])
            torch.testing.assert_close(wq["tau_mag"], torch.full_like(wq["tau_t"], 6.0))

    def test_non_unit_normals_are_normalized(self) -> None:
        """Dan's interpolated normals can have |n| ~ 0.92; the result must not depend on |n|."""
        x = torch.linspace(-9, 9, 7, dtype=torch.float64)
        xy = torch.stack([x, torch.zeros_like(x)], dim=-1)[None]
        n = torch.tensor([0.0, 0.915], dtype=torch.float64).expand_as(xy)
        wq = physics.wall_quantities(poiseuille, xy, n, offset=0.0)
        torch.testing.assert_close(wq["tau_t"], torch.full_like(wq["tau_t"], 6.0))

    def test_outlet_residual_poiseuille(self) -> None:
        """At x = 10: -p + 2 u_x = 0, but u_y + v_x = 6 - 12 y (Poiseuille is not traction-free)."""
        y = torch.linspace(0.05, 0.95, 11, dtype=torch.float64)
        xy = torch.stack([torch.full_like(y, 10.0), y], dim=-1)[None]
        normal, shear = physics.outlet_residual(poiseuille, xy)
        self.assertLess(float(normal.abs().max()), 1e-12)
        torch.testing.assert_close(shear, (6 - 12 * y)[None])

    def test_wall_quantities_match_reference(self) -> None:
        """(c) Same p, tau_t, tau_mag, traction as reference.wall_traction on a synthetic mesh.

        Velocity is quadratic and pressure linear, so the reference's facet-midpoint average
        of nodal gradients (and of nodal p) is exact and both sides compute the same thing.
        """
        nx_ = 9
        xs = np.linspace(0.0, 1.0, nx_)
        y_wall = 0.1 + 0.05 * np.sin(3 * xs)  # a curved wall -> nontrivial normals
        xy = np.concatenate([np.c_[xs, y_wall], np.c_[xs, y_wall + 0.3]])
        tri = [(i, i + 1, nx_ + i) for i in range(nx_ - 1)]
        tri += [(i + 1, nx_ + i + 1, nx_ + i) for i in range(nx_ - 1)]
        lines = [(i, i + 1) for i in range(nx_ - 1)] + [
            (nx_ + i, nx_ + i + 1) for i in range(nx_ - 1)
        ]
        tags = [13] * (nx_ - 1) + [12] * (nx_ - 1)
        x, y = xy[:, 0], xy[:, 1]
        f = {
            "xy": xy,
            "triangles": np.array(tri),
            "lines": np.array(lines),
            "line_tags": np.array(tags),
            "p": 2.0 - 3.0 * x + 1.5 * y,
            "u_x": 1.2 + x - 0.4 * y,
            "u_y": -0.7 - 0.4 * x + 1.8 * y,
            "v_x": 0.4 - 0.6 * x + 0.6 * y,
            "v_y": 0.8 + 0.6 * x + 0.5 * y,
        }
        ref = wall_traction(f, mu=1.0)  # mu = 1: the dimensional formula = nondimensional

        def field(pts):
            px, py = pts[..., 0], pts[..., 1]
            u = 0.3 + 1.2 * px - 0.7 * py + 0.5 * px**2 - 0.4 * px * py + 0.9 * py**2
            v = -0.2 + 0.4 * px + 0.8 * py - 0.3 * px**2 + 0.6 * px * py + 0.25 * py**2
            return u, v, 2.0 - 3.0 * px + 1.5 * py

        mid = torch.tensor(ref["mid"][None], dtype=torch.float64)
        n = torch.tensor(ref["n"][None], dtype=torch.float64)
        wq = physics.wall_quantities(field, mid, n, offset=0.0)
        self.assertTrue(np.all(ref["n"][:, 1] > 0))  # normals point into the fluid (up)
        for key in ("p", "tx", "ty", "tau_t", "tau_mag"):
            with self.subTest(quantity=key):
                np.testing.assert_allclose(wq[key][0].detach().numpy(), ref[key], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
