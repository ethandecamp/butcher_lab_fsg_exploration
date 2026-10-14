"""Torch-free checks of the numpy distance reference and of the inlet hard-constraint margin."""

from __future__ import annotations

import logging
import unittest

import numpy as np

from pinn_surrogate.data import load_packed
from pinn_surrogate.geometry_np import (
    inlet_distance_np,
    inlet_mask_deficit,
    point_segment_distance_np,
    polyline_distance_np,
)

log = logging.getLogger(__name__)

# An L-shaped polyline: (0,0) -> (1,0) -> (1,1).
L_SHAPE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])


class TestPolylineDistance(unittest.TestCase):
    """Hand-checkable distances."""

    def test_known_points(self) -> None:
        """Distances to an L-shaped polyline worked out by hand."""
        pts = np.array([[0.5, 0.3], [2.0, 2.0], [1.5, 0.5], [-1.0, 0.0], [0.5, -0.2]])
        want = [0.3, np.sqrt(2.0), 0.5, 1.0, 0.2]
        got = polyline_distance_np(pts[None], L_SHAPE[None])[0]
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-15)

    def test_zero_on_polyline(self) -> None:
        """Exactly 0 at vertices and at points along segments."""
        on = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.25, 0.0], [1.0, 0.75]])
        self.assertTrue(np.all(polyline_distance_np(on[None], L_SHAPE[None]) == 0.0))

    def test_batched_equals_loop(self) -> None:
        """Batched call == one call per solve, on real packed outlines."""
        nd = load_packed().nondim()
        rng = np.random.default_rng(0)
        pts = rng.uniform([-10, 0], [10, 1], size=(3, 50, 2))
        out = nd["outline"][[0, 100, 900]]
        batched = polyline_distance_np(pts, out)
        for i in range(3):
            d = point_segment_distance_np(pts[i], out[i, :-1], out[i, 1:]).min(1)
            np.testing.assert_array_equal(batched[i], d)


class TestInletMargin(unittest.TestCase):
    """m = tanh(d / delta) must be 1 at the inlet for every packed outline."""

    def test_inlet_distance_exact_vs_dense_sampling(self) -> None:
        """Endpoint formula agrees with sampling the inlet segment densely."""
        nd = load_packed().nondim()
        out = nd["outline"][::97]
        y = np.linspace(0, 1, 2001)
        pts = np.stack([np.full_like(y, -10.0), y], axis=-1)
        dense = polyline_distance_np(np.broadcast_to(pts, (len(out),) + pts.shape), out).min(1)
        np.testing.assert_allclose(inlet_distance_np(out), dense, atol=1e-6)

    def test_inlet_mask_is_one(self) -> None:
        """Worst case over all 960 packed outlines: 1 - m < 1e-6 (it is ~1e-118)."""
        nd = load_packed().nondim()
        d = inlet_distance_np(nd["outline"])
        deficit = inlet_mask_deficit(nd["outline"], delta=0.05)
        log.info("inlet: min distance %.4f, worst 1 - m = %.3g", d.min(), deficit.max())
        self.assertGreater(d.min(), 6.0)
        self.assertLess(deficit.max(), 1e-6)


if __name__ == "__main__":
    unittest.main()
