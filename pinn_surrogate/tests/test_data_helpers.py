"""Torch-free checks of the training helpers added to data.py."""

from __future__ import annotations

import unittest

import numpy as np

from pinn_surrogate.data import (
    TAG_CUSHION,
    TAG_INLET,
    TAG_INTERIOR,
    TAG_OUTLET,
    load_packed,
    mean_pressure_gradient,
    node_weights,
)


class TestHelpers(unittest.TestCase):
    """node_weights and mean_pressure_gradient."""

    def test_node_weights(self) -> None:
        """Trust weights on interior nodes; boundary nodes always weight 1."""
        r = np.array([1e-4, 1e-1, 1e-1, 1e-1])
        tag = np.array([TAG_INTERIOR, TAG_INTERIOR, TAG_CUSHION, TAG_OUTLET])
        np.testing.assert_allclose(node_weights(r, tag), [1.0, 0.1, 1.0, 1.0])

    def test_poiseuille_gradient_is_12(self) -> None:
        """p = 12 (10 - x) sampled at inlet (x=-10) and outlet (x=10) nodes -> 12."""
        tag = np.array([[TAG_INLET, TAG_INLET, TAG_OUTLET, TAG_INTERIOR, TAG_OUTLET]])
        x = np.array([[-10.0, -10.0, 10.0, 0.0, 10.0]])
        self.assertAlmostEqual(float(mean_pressure_gradient(12 * (10 - x), tag)[0]), 12.0)

    def test_packed_values(self) -> None:
        """On real data the mean gradient is >= the empty-channel 12 and below ~120."""
        pk = load_packed()
        c = mean_pressure_gradient(pk.nondim()["p"], pk.arrays["tag"])
        self.assertEqual(c.shape, (pk.n_solves,))
        self.assertGreater(c.min(), 11.9)
        self.assertLess(c.max(), 120.0)


if __name__ == "__main__":
    unittest.main()
