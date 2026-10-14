"""Nondimensional Navier-Stokes residual, wall traction and outlet traction via autograd.

All quantities are in the units of ``data.py`` (lengths / H, velocity / U, pressure and
stress / (mu U / H)). In these units

    momentum:    Re (u . grad) u + grad p - lap u = 0
    continuity:  div u = 0
    stress:      sigma = -p I + (grad u + grad u^T)

The sign convention of the momentum residual is Dan's (``reference.ns_residual_nondim``),
so a residual computed here can be compared with his saved arrays directly.

A "field function" maps points (B, N, 2) to (u, v, p), each (B, N): either the network
with its conditioning bound (``ConditionedPINN.field_fn``) or an analytic field in tests.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

FieldFn = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


def grad(f: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """d f / d xy for a field f (B, N) of points xy (B, N, 2); returns (B, N, 2).

    Each output depends only on its own point, so the gradient of ``f.sum()`` gives every
    pointwise gradient in one backward pass. ``create_graph=True`` keeps the result
    differentiable (needed for second derivatives and for training through it). A field
    that does not depend on xy (e.g. a constant in a test) has gradient zero.
    """
    if not f.requires_grad:
        return torch.zeros_like(xy)
    (g,) = torch.autograd.grad(
        f, xy, grad_outputs=torch.ones_like(f), create_graph=True, allow_unused=True
    )
    return torch.zeros_like(xy) if g is None else g


def _with_grad(xy: torch.Tensor) -> torch.Tensor:
    """A copy of xy that autograd tracks (so derivatives w.r.t. position exist)."""
    return xy.detach().clone().requires_grad_(True)


def ns_residual_from_fields(
    u: torch.Tensor, v: torch.Tensor, p: torch.Tensor, xy: torch.Tensor, re: torch.Tensor | float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Momentum-x, momentum-y and continuity residuals of fields built from ``xy``.

    Args:
        u, v, p: (B, N) fields computed from ``xy`` (which must require grad).
        xy: (B, N, 2) points.
        re: Reynolds number, a float or a tensor broadcastable to (B, N), e.g. (B, 1).

    Returns:
        ``(r_x, r_y, r_div)``, each (B, N).
    """
    gu, gv, gp = grad(u, xy), grad(v, xy), grad(p, xy)
    u_x, u_y = gu[..., 0], gu[..., 1]
    v_x, v_y = gv[..., 0], gv[..., 1]
    lap_u = grad(u_x, xy)[..., 0] + grad(u_y, xy)[..., 1]
    lap_v = grad(v_x, xy)[..., 0] + grad(v_y, xy)[..., 1]
    r_x = re * (u * u_x + v * u_y) + gp[..., 0] - lap_u
    r_y = re * (u * v_x + v * v_y) + gp[..., 1] - lap_v
    return r_x, r_y, u_x + v_y


def ns_residual(
    field_fn: FieldFn, xy: torch.Tensor, re: torch.Tensor | float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate ``field_fn`` at ``xy`` and return its NS residual (see above).

    Args:
        field_fn: Maps (B, N, 2) points to (u, v, p).
        xy: (B, N, 2) collocation points (need not require grad; a tracked copy is used).
        re: Reynolds number, float or broadcastable to (B, N).

    Returns:
        ``(r_x, r_y, r_div)``, each (B, N), differentiable w.r.t. the model parameters.
    """
    xy = _with_grad(xy)
    u, v, p = field_fn(xy)
    return ns_residual_from_fields(u, v, p, xy, re)


def traction(
    p: torch.Tensor,
    u_x: torch.Tensor,
    u_y: torch.Tensor,
    v_x: torch.Tensor,
    v_y: torch.Tensor,
    n: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Traction sigma n and its signed tangential part, same formula as ``reference.py``.

    Args:
        p, u_x, u_y, v_x, v_y: (..., K) pressure and velocity gradients at wall points.
        n: (..., K, 2) unit normal pointing out of the cushion into the fluid.

    Returns:
        ``tx, ty`` (force per area of the fluid on the wall), ``tau_t`` = tau . t_hat with
        t_hat = (n_y, -n_x) (positive = downstream on top of the cushion), ``tau_mag``.
    """
    nx, ny = n[..., 0], n[..., 1]
    sxx, syy, sxy = -p + 2 * u_x, -p + 2 * v_y, u_y + v_x
    tx, ty = sxx * nx + sxy * ny, sxy * nx + syy * ny
    t_n = tx * nx + ty * ny
    tau_x, tau_y = tx - t_n * nx, ty - t_n * ny
    return {
        "tx": tx,
        "ty": ty,
        "tau_t": tau_x * ny - tau_y * nx,
        "tau_mag": torch.sqrt(tau_x**2 + tau_y**2),
    }


def wall_quantities(
    field_fn: FieldFn, xy_wall: torch.Tensor, normals: torch.Tensor, *, offset: float
) -> dict[str, torch.Tensor]:
    """Wall pressure and wall shear at points on the cushion.

    Pressure is evaluated on the wall itself (it is not masked, and Dan's ``arc_p`` is the
    fluid pressure at the wall). Velocity gradients are evaluated at ``xy_wall + offset * n``:
    the distance d_c in the hard constraint has no usable gradient exactly on the polyline
    (see ``geometry.polyline_distance``), so ``offset`` must be > 0 for the network. It is
    keyword-only and has no default so a caller cannot silently get tau = 0. Analytic test
    fields may use ``offset=0``.

    Normals are re-normalized to unit length here. Dan's ``*_256`` normals (packed as
    ``wall_nx, wall_ny``) are interpolated from facet normals and ~5 % of them have
    |n| < 0.999 (minimum 0.915); tau_t = (sigma n) . (n_y, -n_x) is quadratic in n, so an
    unnormalized 0.915 normal would understate tau by ~16 % at those points.

    Args:
        field_fn: Maps (B, K, 2) points to (u, v, p).
        xy_wall: (B, K, 2) wall points.
        normals: (B, K, 2) normals into the fluid (any positive length).
        offset: Distance into the fluid (nondim) at which velocity gradients are taken.

    Returns:
        ``p`` and the entries of ``traction`` (``tx, ty, tau_t, tau_mag``), each (B, K).
    """
    normals = normals / torch.linalg.norm(normals, dim=-1, keepdim=True)
    _, _, p = field_fn(xy_wall)
    xy = _with_grad(xy_wall + offset * normals)
    u, v, _ = field_fn(xy)
    gu, gv = grad(u, xy), grad(v, xy)
    out = traction(p, gu[..., 0], gu[..., 1], gv[..., 0], gv[..., 1], normals)
    out["p"] = p
    return out


def outlet_residual(field_fn: FieldFn, xy_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Traction-free outlet with the symmetric stress: sigma n = 0 with n = (1, 0).

    Dan's outlet condition (verified in Stage 2): -p + 2 u_x = 0 and u_y + v_x = 0. It
    sets the pressure gauge; v is not zero there and the profile is not exactly Poiseuille.

    Args:
        field_fn: Maps (B, K, 2) points to (u, v, p).
        xy_out: (B, K, 2) outlet points.

    Returns:
        ``(-p + 2 u_x, u_y + v_x)``, each (B, K).
    """
    xy = _with_grad(xy_out)
    u, v, p = field_fn(xy)
    gu, gv = grad(u, xy), grad(v, xy)
    return -p + 2 * gu[..., 0], gu[..., 1] + gv[..., 0]
