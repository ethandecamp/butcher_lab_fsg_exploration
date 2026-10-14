"""Torch-free checks of the evaluation metrics and their aggregation."""

from __future__ import annotations

import math
import unittest

import numpy as np

from pinn_surrogate import metrics


class TestMetrics(unittest.TestCase):
    """Each metric on inputs with a known answer."""

    def test_rel_l2(self) -> None:
        """||(3,4) - (0,0)|| / ||(3,4)|| = 1; identical arrays give 0."""
        self.assertEqual(metrics.rel_l2(np.zeros(2), np.array([3.0, 4.0])), 1.0)
        self.assertEqual(metrics.rel_l2(np.array([3.0, 4.0]), np.array([3.0, 4.0])), 0.0)

    def test_arc_fraction(self) -> None:
        """Uneven spacing on a straight line: s/L follows the cumulative length."""
        o = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
        np.testing.assert_allclose(metrics.arc_fraction(o), [0, 0.25, 0.75, 1.0])

    def test_peak_location_error(self) -> None:
        """Peaks of |tau| (sign ignored) at s = 0.25 and 0.75 -> error 0.5."""
        s = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        pred = np.array([0, -5.0, 1, 1, 0])
        true = np.array([0, 1, 1, 3.0, 0])
        self.assertAlmostEqual(metrics.peak_location_error(pred, true, s), 0.5)
        self.assertEqual(metrics.peak_location_error(true, true, s), 0.0)

    def test_sign_agreement_ignores_small_values(self) -> None:
        """Points with |tau| <= 1 % of max are excluded; 2 of the 3 kept points agree."""
        true = np.array([100.0, -50.0, 20.0, 0.5, -0.9])
        pred = np.array([90.0, 10.0, 25.0, -1.0, 3.0])
        self.assertAlmostEqual(metrics.sign_agreement(pred, true), 2 / 3)

    def test_sign_agreement_all_zero(self) -> None:
        """No point above threshold -> NaN, not a crash."""
        self.assertTrue(math.isnan(metrics.sign_agreement(np.ones(3), np.zeros(3))))

    def test_aggregate(self) -> None:
        """Overall and grouped medians/means; NaNs are dropped."""
        rows = [
            {"k": 0, "e": 1.0},
            {"k": 0, "e": 3.0},
            {"k": 1, "e": 10.0},
            {"k": 1, "e": float("nan")},
        ]
        overall = metrics.aggregate(rows, ["e"])
        self.assertEqual(overall["e"], {"median": 3.0, "mean": 14 / 3, "n": 3})
        by_k = metrics.aggregate(rows, ["e"], "k")
        self.assertEqual(list(by_k), ["0", "1"])
        self.assertEqual(by_k["0"]["e"]["median"], 2.0)
        self.assertEqual(by_k["1"]["e"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
