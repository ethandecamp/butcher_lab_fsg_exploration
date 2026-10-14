"""Distance from points to the cushion outline, in torch (batched and differentiable).

The hard no-slip constraint multiplies the velocity by m = tanh(d_c / delta), where d_c is
the unsigned distance to the 256-vertex outline polyline. The PDE loss needs first and
second derivatives of d_c with respect to the query point, so this must be autograd-friendly.

Two-stage computation:

1. Under ``no_grad``, find the nearest of the 255 segments for every point (B x N x 255).
2. With gradients, recompute the distance to that one segment only (B x N).

The value equals the brute-force minimum (``geometry_np``), and the gradient is the same as
differentiating ``min`` (whose gradient also flows only through the arg-min), but the
autograd graph is 255x smaller, which matters when it is differentiated twice.
"""

from __future__ import annotations

import torch


def _segment_residual(p: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Vector from the closest point of segment [a, b] to p (all broadcastable, last dim 2)."""
    ab = b - a
    denom = (ab * ab).sum(-1).clamp_min(torch.finfo(ab.dtype).tiny)
    t = (((p - a) * ab).sum(-1) / denom).clamp(0.0, 1.0)
    return p - (a + t.unsqueeze(-1) * ab)


def _safe_norm(r: torch.Tensor) -> torch.Tensor:
    """|r| over the last dim, exactly 0 at r = 0, with a finite (zero) gradient there.

    A plain ``sqrt`` has an infinite derivative at 0, which turns into NaN gradients for
    points exactly on the polyline (e.g. packed cushion nodes in a batch that also has PDE
    terms). The ``where`` trick feeds sqrt a dummy 1 at those points instead.
    """
    r2 = (r * r).sum(-1)
    pos = r2 > 0
    return torch.where(pos, torch.sqrt(torch.where(pos, r2, torch.ones_like(r2))), r2 * 0)


# Upper bound on B x chunk x S point-segment pairs held at once in the nearest-segment search.
# 2**22 pairs x 2 coords x 4 bytes = 32 MB per temporary (a few temporaries live at once),
# small next to the VM's ~3.5 GB even with two runs in parallel. A training batch
# (8 x 512 x 255 = 1.0M pairs) fits in one chunk; only large evaluation calls are split.
MAX_PAIRS = 2**22


def nearest_segment(points: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Index of the closest segment [a_j, b_j] for every point, chunked over points.

    Args:
        points: (B, N, 2).
        a, b: (B, S, 2) segment endpoints.

    Returns:
        (B, N) int64 indices into the S segments. Computed without autograd.
    """
    bsz, n = points.shape[:2]
    chunk = max(1, MAX_PAIRS // max(1, bsz * a.shape[1]))
    out = []
    with torch.no_grad():
        for start in range(0, n, chunk):
            p = points[:, start : start + chunk, None, :]  # (B, c, 1, 2)
            r = _segment_residual(p, a[:, None], b[:, None])  # (B, c, S, 2)
            out.append((r * r).sum(-1).argmin(-1))
    return torch.cat(out, dim=1)


def polyline_distance(points: torch.Tensor, outline: torch.Tensor) -> torch.Tensor:
    """Unsigned distance from each point to its solve's outline polyline.

    Args:
        points: (B, N, 2) query points (nondim).
        outline: (B, V, 2) polyline vertices (nondim); segments join consecutive vertices.

    Returns:
        (B, N) distances; exactly 0 on the polyline. Differentiable w.r.t. ``points``
        (and ``outline``) to any order; note the gradient at d = 0 is 0, not the normal,
        so wall derivatives must be taken slightly off the wall (see physics.wall_quantities).
    """
    a, b = outline[:, :-1], outline[:, 1:]  # (B, S, 2)
    nearest = nearest_segment(points, a, b)  # (B, N)
    idx = nearest.unsqueeze(-1).expand(-1, -1, 2)
    a_sel = torch.gather(a, 1, idx)
    b_sel = torch.gather(b, 1, idx)
    return _safe_norm(_segment_residual(points, a_sel, b_sel))
