"""wall_tau_loss: matches the evaluation metric, and its gradient reaches the weights (needs torch)."""

from __future__ import annotations

import unittest

import numpy as np

try:
    import torch

    from pinn_surrogate import physics
    from pinn_surrogate.config import ModelConfig
    from pinn_surrogate.losses import wall_tau_loss
    from pinn_surrogate.metrics import rel_l2
    from pinn_surrogate.model import ConditionedPINN

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestWallTauLoss(unittest.TestCase):
    def test_zero_at_truth_and_equals_metric_squared(self) -> None:
        rng = np.random.default_rng(0)
        t = torch.tensor(rng.normal(size=(3, 256)), dtype=torch.float64)
        self.assertEqual(float(wall_tau_loss(t, t)), 0.0)
        p = t * torch.tensor([[0.6], [1.0], [1.3]], dtype=torch.float64) + 0.1
        want = np.mean([rel_l2(p[i].numpy(), t[i].numpy()) ** 2 for i in range(3)])
        self.assertAlmostEqual(float(wall_tau_loss(p, t)), want, places=10)

    def test_gradient_reaches_trunk_weights(self) -> None:
        """Training through tau needs second-order autograd through wall_quantities."""
        model = ConditionedPINN(ModelConfig(width=16, depth=2))
        th = torch.linspace(0.05, np.pi - 0.05, 256)
        outline = torch.stack([-2 + 2 * torch.cos(th).flip(0) * -1, 0.4 * torch.sin(th)], -1)[None]
        outline = outline.float()
        normals = torch.stack([torch.cos(th).flip(0) * -1, torch.sin(th)], -1)[None].float()
        wq = physics.wall_quantities(
            model.field_fn(outline, torch.zeros(1)), outline, normals, offset=1e-4
        )
        loss = wall_tau_loss(wq["tau_t"], torch.ones(1, 256))
        loss.backward()
        g = model.trunk[-1].weight.grad
        self.assertIsNotNone(g)
        self.assertGreater(float(g.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
