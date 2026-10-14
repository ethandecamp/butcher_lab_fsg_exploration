"""Torch distance-to-outline vs the brute-force numpy reference (skipped without torch)."""

from __future__ import annotations

import unittest

import numpy as np

from pinn_surrogate.data import load_packed
from pinn_surrogate.geometry_np import polyline_distance_np

try:
    import torch

    from pinn_surrogate import geometry
    from pinn_surrogate.geometry import polyline_distance

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestPolylineDistanceTorch(unittest.TestCase):
    """Values, exact zeros and derivatives."""

    @classmethod
    def setUpClass(cls) -> None:
        pk = load_packed()
        nd = pk.nondim()
        pick = [0, 1, 300, 700, 959]  # mixed families and splits
        cls.outlines = nd["outline"][pick]  # (5, 256, 2)
        cls.nodes = nd["xy"][pick]  # real packed nodes of the same solves
        cls.interior = np.flatnonzero(pk.arrays["tag"][0] == 0)  # same layout in every solve

    def test_matches_numpy_on_packed_solves(self) -> None:
        """Random points and the real packed nodes, float64: agreement to 1e-12."""
        rng = np.random.default_rng(0)
        pts = np.concatenate([rng.uniform([-4, 0], [2, 1], size=(5, 500, 2)), self.nodes], 1)
        want = polyline_distance_np(pts, self.outlines)
        got = polyline_distance(torch.tensor(pts), torch.tensor(self.outlines))
        self.assertEqual(tuple(got.shape), want.shape)
        np.testing.assert_allclose(got.numpy(), want, rtol=0, atol=1e-12)

    def test_chunking_does_not_change_result(self) -> None:
        """Forcing many small chunks (uneven last chunk) gives bit-identical distances."""
        pts = torch.tensor(self.nodes)
        o = torch.tensor(self.outlines)
        whole = polyline_distance(pts, o)
        saved = geometry.MAX_PAIRS
        try:
            geometry.MAX_PAIRS = 5 * 255 * 7  # 7 points per chunk; 3072 % 7 != 0
            chunked = polyline_distance(pts, o)
        finally:
            geometry.MAX_PAIRS = saved
        torch.testing.assert_close(chunked, whole, rtol=0, atol=0)

    def test_float32_close_to_numpy(self) -> None:
        """In float32 (the training dtype) the error stays at float32 rounding level."""
        pts = self.nodes.astype(np.float32)
        got = polyline_distance(torch.tensor(pts), torch.tensor(self.outlines, dtype=torch.float32))
        np.testing.assert_allclose(got.numpy(), polyline_distance_np(pts, self.outlines), atol=5e-6)

    def test_exactly_zero_on_polyline(self) -> None:
        """Vertices and segment midpoints give exactly 0 (the no-slip mask is exact)."""
        o = torch.tensor(self.outlines)
        mids = 0.5 * (o[:, :-1] + o[:, 1:])
        for pts in (o, mids):
            self.assertTrue(bool((polyline_distance(pts, o) == 0).all()))

    def test_gradient_is_unit_normal_off_the_wall(self) -> None:
        """Just off a segment along its normal, grad d = that normal; no NaNs on the wall."""
        o = torch.tensor(self.outlines[:1])
        a, b = o[:, 100], o[:, 101]
        t = (b - a) / torch.linalg.norm(b - a)
        n = torch.stack([-t[:, 1], t[:, 0]], dim=-1)  # left normal = into the fluid
        pts = (0.5 * (a + b) + 1e-3 * n)[:, None].clone().requires_grad_(True)
        (g,) = torch.autograd.grad(polyline_distance(pts, o).sum(), pts)
        torch.testing.assert_close(g[:, 0], n, rtol=0, atol=1e-9)

        on_wall = o.clone().requires_grad_(True)
        (g0,) = torch.autograd.grad(polyline_distance(on_wall, o).sum(), on_wall)
        self.assertTrue(bool(torch.isfinite(g0).all()))

    def test_second_derivatives_are_finite(self) -> None:
        """Hessian entries exist (needed by the Laplacian of m = tanh(d / delta))."""
        o = torch.tensor(self.outlines[:2])
        pts = torch.tensor(self.nodes[:2, self.interior[:200]]).requires_grad_(True)
        d = polyline_distance(pts, o)
        (g,) = torch.autograd.grad(d.sum(), pts, create_graph=True)
        (h,) = torch.autograd.grad(g[..., 0].sum(), pts)
        self.assertTrue(bool(torch.isfinite(h).all()))


if __name__ == "__main__":
    unittest.main()
