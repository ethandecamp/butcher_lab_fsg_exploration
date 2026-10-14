"""Loss terms and gradient-norm loss balancing for the PINN.

Three terms, each a mean of squares of dimensionless quantities:

* ``data_loss``: trust-weighted MSE on (u*, v*, p*), each channel divided by a fixed
  scale (its std over the packed data) so pressure does not dominate.
* ``pde_loss``: momentum residual / c and continuity residual / (c / 2), where c is the
  solve's mean pressure gradient taken from its data (``data.mean_pressure_gradient``).
  c / 2 is the wall shear rate of a Poiseuille flow driven by that gradient, i.e. the
  natural size of a velocity gradient in the same solve.
* ``outlet_loss``: outlet traction / 6, where 6 is the wall shear stress of the
  (near-Poiseuille, mean speed 1) outlet flow in these units.

``AdaptiveWeights`` rebalances them like Wang, Teng & Perdikaris (2021): a term whose
parameter gradients are small relative to the reference term gets a larger weight.
"""

from __future__ import annotations

import torch

OUTLET_STRESS_SCALE = 6.0


def data_loss(
    pred: tuple[torch.Tensor, ...],
    target: torch.Tensor,
    weight: torch.Tensor,
    scales: tuple[float, float, float],
) -> torch.Tensor:
    """Weighted MSE of (u, v, p) in units of each channel's scale.

    Args:
        pred: (u, v, p), each (B, N).
        target: (B, N, 3) true (u*, v*, p*).
        weight: (B, N) per-node weights (trust on interior nodes, 1 on boundaries).
        scales: Fixed normalizing scale per channel.

    Returns:
        Scalar: sum over channels of the weighted mean squared normalized error.
    """
    total = weight.sum()
    loss = pred[0].new_zeros(())
    for i, s in enumerate(scales):
        loss = loss + (weight * ((pred[i] - target[..., i]) / s) ** 2).sum() / total
    return loss


FLOOR = 0.01  # scale floor, as a fraction of the solve's c (stagnant zones have tiny terms)


def _scales(
    c: torch.Tensor, c_mom: torch.Tensor | None, c_div: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Momentum and divergence normalizers: local term sizes if given, else c and c / 2.

    The local version makes the momentum residual exactly Dan's ``r_rel`` (|r| over the sum
    of the absolute momentum terms of the true solution). A single per-solve constant let a
    few hundred near-wall nodes, where the terms are ~100x c, dominate the loss (LOG.md).
    """
    if c_mom is None or c_div is None:
        return c, 0.5 * c
    return c_mom + FLOOR * c, c_div + FLOOR * 0.5 * c


def pde_loss(
    residual: tuple[torch.Tensor, ...],
    c: torch.Tensor,
    c_mom: torch.Tensor | None = None,
    c_div: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean squared normalized NS residual.

    Args:
        residual: (r_x, r_y, r_div), each (B, N).
        c: (B, 1) per-solve mean pressure gradient from the data.
        c_mom, c_div: optional (B, N) local term sizes from the data (see ``_scales``).

    Returns:
        Scalar mean over points of (|r_mom| / s_mom)^2 + (r_div / s_div)^2.
    """
    return (pde_residual_magnitude(residual, c, c_mom, c_div) ** 2).mean()


def pde_residual_magnitude(
    residual: tuple[torch.Tensor, ...],
    c: torch.Tensor,
    c_mom: torch.Tensor | None = None,
    c_div: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pointwise sqrt of the ``pde_loss`` integrand (B, N); used by the evaluation."""
    r_x, r_y, r_div = residual
    s_mom, s_div = _scales(c, c_mom, c_div)
    return torch.sqrt((r_x**2 + r_y**2) / s_mom**2 + (r_div / s_div) ** 2)


def outlet_loss(residual: tuple[torch.Tensor, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
    """Mean squared outlet traction over the masked outlet points.

    Args:
        residual: (-p + 2 u_x, u_y + v_x), each (B, K).
        mask: (B, K) 1 for outlet points to use (the corners belong to the no-slip walls).

    Returns:
        Scalar.
    """
    sq = (residual[0] / OUTLET_STRESS_SCALE) ** 2 + (residual[1] / OUTLET_STRESS_SCALE) ** 2
    return (sq * mask).sum() / mask.sum()


def _mean_abs_grad(loss: torch.Tensor, params: list[torch.Tensor]) -> float:
    """Mean |d loss / d theta| over all parameters (unused parameters count as 0)."""
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    total = sum(float(g.abs().sum()) for g in grads if g is not None)
    return total / sum(p.numel() for p in params)


class AdaptiveWeights:
    """Gradient-norm loss balancing (Wang, Teng & Perdikaris 2021, mean/mean variant).

    Every ``update``: lambda_i <- (1 - alpha) lambda_i + alpha * share * G_ref / G_i, where G is the
    mean absolute parameter gradient of a term, clamped to ``[lo, hi]``. The reference term
    keeps lambda = 1. The moving average (alpha = 0.1) stops one noisy batch from swinging
    the weights.
    """

    def __init__(
        self,
        names: list[str],
        ref: str,
        alpha: float = 0.1,
        share: float = 1.0,
        lo: float = 1e-3,
        hi: float = 1e3,
    ) -> None:
        if ref not in names:
            raise ValueError(f"reference term {ref!r} not in {names}")
        self.ref, self.alpha, self.lo, self.hi = ref, alpha, lo, hi
        self.share = share  # target gradient size of each term relative to the reference
        self.lambdas = {n: 1.0 for n in names}
        self.last_grad: dict[str, float] = {}
        self.n_updates = 0

    def update(self, losses: dict[str, torch.Tensor], params: list[torch.Tensor]) -> None:
        """Recompute the weights from the current losses (graphs must still be alive)."""
        g = {n: _mean_abs_grad(losses[n], params) for n in self.lambdas}
        self.last_grad = g
        for n in self.lambdas:
            if n == self.ref or g[n] == 0.0:
                continue
            target = self.share * g[self.ref] / g[n]
            # First update: jump straight to the target. Starting every lambda at 1 and
            # easing in with alpha = 0.1 per update took ~2000 steps to reach the target, and
            # the physics gradient had already wrecked the data fit by then (LOG.md, Stage 4).
            new = target if self.n_updates == 0 else (
                (1 - self.alpha) * self.lambdas[n] + self.alpha * target
            )
            self.lambdas[n] = min(self.hi, max(self.lo, new))
        self.n_updates += 1

    def total(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        """Weighted sum of the losses."""
        return sum(self.lambdas[n] * losses[n] for n in self.lambdas)
