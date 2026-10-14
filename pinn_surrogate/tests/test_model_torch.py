"""Hard constraints, shapes, parameter count and checkpointing of the model (needs torch)."""

from __future__ import annotations

import logging
import tempfile
import unittest
from itertools import pairwise
from pathlib import Path

import numpy as np

from pinn_surrogate.config import ModelConfig
from pinn_surrogate.data import load_packed

try:
    import torch

    from pinn_surrogate import physics
    from pinn_surrogate.model import ConditionedPINN, load_checkpoint, save_checkpoint

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

log = logging.getLogger(__name__)


def expected_params(c: ModelConfig) -> int:
    """Parameter count from the architecture, by hand (weights + biases per Linear)."""
    enc = [2 * c.n_outline, c.enc_width, c.enc_width, c.latent]
    trunk = [2 * c.n_fourier + 2 + c.latent + 1] + [c.width] * c.depth + [3]
    return sum(a * b + b for sizes in (enc, trunk) for a, b in pairwise(sizes))


def _randomize(model: ConditionedPINN, seed: int = 1) -> None:
    """Large random weights so N_u, N_v, N_p are far from zero (constraints must still hold)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.5)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestModel(unittest.TestCase):
    """The ansatz makes walls and inlet exact for any weights."""

    @classmethod
    def setUpClass(cls) -> None:
        pk = load_packed()
        nd = pk.nondim()
        cls.outline = torch.tensor(nd["outline"], dtype=torch.float32)  # all 960 solves
        cls.log_re = torch.tensor(np.log(nd["Re"]), dtype=torch.float32)
        cls.normals = torch.tensor(
            np.stack([pk.arrays["wall_nx"], pk.arrays["wall_ny"]], -1), dtype=torch.float32
        )
        cls.model = ConditionedPINN(ModelConfig())
        cls.model.set_normalization(nd["outline"], np.log(nd["Re"]))
        _randomize(cls.model)

    def test_param_count(self) -> None:
        """Count matches the hand formula (~133k with the defaults); printed for the log."""
        n = self.model.n_params()
        log.info("ConditionedPINN default parameter count: %d", n)
        print(f"\nConditionedPINN parameters: {n}")
        self.assertEqual(n, expected_params(ModelConfig()))

    def test_shapes(self) -> None:
        """(B, N, 2) points -> three (B, N) fields."""
        xy = torch.rand(3, 17, 2)
        u, v, p = self.model(xy, self.outline[:3], self.log_re[:3])
        for f in (u, v, p):
            self.assertEqual(tuple(f.shape), (3, 17))
            self.assertTrue(bool(torch.isfinite(f).all()))

    def test_no_slip_on_cushion_and_walls(self) -> None:
        """u = v = 0 exactly on outline vertices and y = 0, 1; mask ~0 at segment midpoints."""
        with torch.no_grad():
            for start in range(0, len(self.outline), 64):
                sl = slice(start, start + 64)
                o = self.outline[sl]
                mids = 0.5 * (o[:, :-1] + o[:, 1:])
                x = torch.linspace(-10, 10, 101).expand(len(o), 101)
                walls = torch.cat(
                    [
                        torch.stack([x, torch.zeros_like(x)], -1),
                        torch.stack([x, torch.ones_like(x)], -1),
                    ],
                    dim=1,
                )
                for pts in (o, walls):
                    u, v, _ = self.model(pts, o, self.log_re[sl])
                    self.assertEqual(float(u.abs().max()), 0.0)
                    self.assertEqual(float(v.abs().max()), 0.0)
                # Segment midpoints, rounded to float32, sit up to ~6e-8 off the segment, so
                # the mask there is ~1e-6 rather than exactly 0. Check the mask itself: the
                # velocity is the mask times O(1-10) for a trained model (randomized test
                # weights inflate that factor, so asserting on u would test the weights).
                m = self.model.parts(mids, o, self.log_re[sl])["m"]
                self.assertLess(float(m.abs().max()), 1e-5)

    def test_inlet_profile(self) -> None:
        """At x = -10: v = 0 and u = 6y(1-y) to within 1e-6 for every packed outline."""
        y = torch.linspace(0, 1, 41)
        pts = torch.stack([torch.full_like(y, -10.0), y], -1).expand(len(self.outline), 41, 2)
        with torch.no_grad():
            parts = self.model.parts(pts, self.outline, self.log_re)
            u, v, _ = self.model(pts, self.outline, self.log_re)
        deficit = float((1 - parts["m"].double()).max())
        log.info("inlet worst-case 1 - m in float32: %.3g", deficit)
        print(f"\ninlet worst-case 1 - m (float32): {deficit:.3g}")
        self.assertLess(deficit, 1e-6)
        torch.testing.assert_close(u, (6 * y * (1 - y)).expand_as(u), rtol=0, atol=1e-6)
        self.assertEqual(float(v.abs().max()), 0.0)

    def test_wall_shear_matches_analytic_limit(self) -> None:
        """Offset autograd shear ~= its exact wall limit (F . t_hat) / delta, float64.

        At the wall u = m F with m = 0 and grad m = n / delta, so grad u = F n^T / delta and
        tau_t = (F . t_hat) / delta exactly. The offset estimate differs by O(offset), plus a
        small angle error at the few vertices where Dan's (interpolated) normal is not
        perpendicular to the 256-point polyline, so grad d points along a segment normal.
        """
        model = ConditionedPINN(ModelConfig()).double()
        model.load_state_dict(self.model.state_dict())
        model = model.double()
        o = self.outline[:4].double()
        n = self.normals[:4].double()
        n = n / torch.linalg.norm(n, dim=-1, keepdim=True)  # as wall_quantities does
        lr = self.log_re[:4].double()
        wq = physics.wall_quantities(model.field_fn(o, lr), o, n, offset=model.cfg.wall_offset)
        with torch.no_grad():
            f = model.parts(o, o, lr)
            c = model.cfg
            wg = f["w"] * f["g"]
            fu = f["u_p"] + wg * c.s_u * f["N_u"]
            fv = wg * c.s_v * f["N_v"]
            exact = (fu * n[..., 1] - fv * n[..., 0]) / c.delta  # F . (n_y, -n_x) / delta
        err = torch.linalg.norm(wq["tau_t"].detach() - exact) / torch.linalg.norm(exact)
        log.info("offset vs exact wall shear relL2: %.3g", float(err))
        self.assertLess(float(err), 1e-2)

    def test_checkpoint_round_trip(self) -> None:
        """save -> load reproduces the outputs exactly (buffers included)."""
        xy = torch.rand(2, 9, 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.pt"
            save_checkpoint(self.model, path, {"note": "test"})
            loaded, ckpt = load_checkpoint(path)
        self.assertEqual(ckpt["note"], "test")
        with torch.no_grad():
            a = self.model(xy, self.outline[:2], self.log_re[:2])
            b = loaded(xy, self.outline[:2], self.log_re[:2])
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
