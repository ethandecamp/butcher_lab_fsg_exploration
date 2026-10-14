"""Numpy reference formulas that the torch physics code will be tested against.

Everything here takes Dan's saved nodal fields (``data/full/*.npz``) instead of a network:

* ``ns_residual_dim`` / ``ns_residual_nondim``: steady incompressible Navier-Stokes residual.
* ``relative_residual``: reproduces Dan's ``r_rel``.
* ``wall_traction``: wall pressure and signed wall shear on the cushion facets.

Conventions established by matching Dan's arrays (numbers in LOG.md, Stage 1-2):

* Dan's ``r_mom = rho (u.grad) u + grad p - mu lap u`` (this sign), ``r_div = u_x + v_y``.
* ``r_rel = |r_mom|_2 / (|c_x| + |c_y| + |p_x| + |p_y| + mu |lap u| + mu |lap v|)`` with
  ``c = rho (u.grad) u``; the divergence is not part of it.
* Wall normal ``n`` points out of the cushion, into the fluid. Traction ``t = sigma n``
  with ``sigma = -p I + mu (grad u + grad u^T)``, i.e. the force per area the fluid exerts
  on the cushion (``arc_tx, arc_ty``).
* Wall tangent ``t_hat = (n_y, -n_x)`` (n rotated 90 deg clockwise): on top of the cushion
  ``n = (0, 1)`` and ``t_hat = (1, 0)``, so ``tau_t > 0`` means downstream shear.
  ``tau_t = t . t_hat``; ``tau_mag = |t - (t.n) n|``.
* ``arc_p`` is the fluid pressure at the facet midpoint (P1 interpolation), not ``-t.n``.
* Arc points are the midpoints of the tag-13 boundary facets, in ``lines`` order.
* Outlet (tag 11) is traction-free with the *symmetric* stress, ``sigma n = 0``:
  ``-p + 2 mu u_x = 0`` and ``mu (u_y + v_x) = 0`` (``outlet_traction``). This, not
  ``p = 0``, sets the pressure gauge; the outlet profile is not exactly Poiseuille.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from pinn_surrogate.data import load_full, mean_inlet_speed, to_dim, to_nondim

log = logging.getLogger(__name__)

CUSHION_TAG = 13
FULL_DIR = Path(__file__).resolve().parents[2] / "data" / "full"

# Nodal field names -> nondimensional kind (see data.SCALES).
FIELD_KINDS = {
    "u": "velocity",
    "v": "velocity",
    "p": "pressure",
    "u_x": "velocity_gradient",
    "u_y": "velocity_gradient",
    "v_x": "velocity_gradient",
    "v_y": "velocity_gradient",
    "p_x": "pressure_gradient",
    "p_y": "pressure_gradient",
    "lap_u": "velocity_laplacian",
    "lap_v": "velocity_laplacian",
}


def momentum_terms_dim(f: dict[str, np.ndarray], rho: float, mu: float) -> dict[str, np.ndarray]:
    """Dimensional momentum terms, each of shape (2, n): convection, pressure, viscous.

    Args:
        f: Nodal fields ``u, v, u_x, u_y, v_x, v_y, p_x, p_y, lap_u, lap_v`` (SI).
        rho: Density in kg/m^3.
        mu: Dynamic viscosity in Pa s.

    Returns:
        ``{"conv": rho (u.grad) u, "pres": grad p, "visc": mu lap u}``.
    """
    conv = rho * np.stack(
        [f["u"] * f["u_x"] + f["v"] * f["u_y"], f["u"] * f["v_x"] + f["v"] * f["v_y"]]
    )
    pres = np.stack([f["p_x"], f["p_y"]])
    visc = mu * np.stack([f["lap_u"], f["lap_v"]])
    return {"conv": conv, "pres": pres, "visc": visc}


def ns_residual_dim(
    f: dict[str, np.ndarray], rho: float, mu: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dimensional steady NS residual, Dan's sign: rho (u.grad) u + grad p - mu lap u.

    Args:
        f: Nodal fields (SI), see ``momentum_terms_dim``.
        rho: Density in kg/m^3.
        mu: Dynamic viscosity in Pa s.

    Returns:
        ``(r_mom_x, r_mom_y, r_div)`` in N/m^3, N/m^3 and 1/s.
    """
    t = momentum_terms_dim(f, rho, mu)
    r = t["conv"] + t["pres"] - t["visc"]
    return r[0], r[1], f["u_x"] + f["v_y"]


def relative_residual(f: dict[str, np.ndarray], rho: float, mu: float) -> np.ndarray:
    """Dan's ``r_rel``: |r_mom|_2 over the sum of absolute momentum-term components."""
    t = momentum_terms_dim(f, rho, mu)
    rx, ry, _ = ns_residual_dim(f, rho, mu)
    scale = sum(np.abs(v).sum(axis=0) for v in t.values())
    return np.hypot(rx, ry) / scale


def nondim_fields(
    f: dict[str, np.ndarray], u_bar: float, h: float, mu: float
) -> dict[str, np.ndarray]:
    """Nondimensionalize the nodal fields that enter the NS residual (``FIELD_KINDS``)."""
    return {k: to_nondim(f[k], kind, u_bar, h, mu) for k, kind in FIELD_KINDS.items()}


def ns_residual_nondim(
    g: dict[str, np.ndarray], re: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nondimensional residual: Re (u*.grad*) u* + grad* p* - lap* u*, and div* u*.

    Multiplying the momentum part by mu U / H^2 (and the divergence by U / H) gives
    ``ns_residual_dim`` exactly, because Re mu U / H^2 = rho U^2 / H.

    Args:
        g: Nondimensional fields, as returned by ``nondim_fields``.
        re: Reynolds number rho U H / mu.

    Returns:
        ``(r_x*, r_y*, div*)``.
    """
    rx = re * (g["u"] * g["u_x"] + g["v"] * g["u_y"]) + g["p_x"] - g["lap_u"]
    ry = re * (g["u"] * g["v_x"] + g["v"] * g["v_y"]) + g["p_y"] - g["lap_v"]
    return rx, ry, g["u_x"] + g["v_y"]


def wall_facets(
    xy: np.ndarray,
    triangles: np.ndarray,
    lines: np.ndarray,
    line_tags: np.ndarray,
    tag: int = CUSHION_TAG,
) -> dict[str, np.ndarray]:
    """Geometry of the boundary facets with ``tag``, in ``lines`` order.

    The normal is oriented into the fluid: towards the third vertex of the (only) mesh
    triangle that owns the facet. The mesh covers the fluid only, so this needs no
    assumption about the order in which facets or their endpoints are stored.

    Args:
        xy: Node coordinates (n, 2).
        triangles: Fluid triangles (m, 3), node indices.
        lines: Boundary facets (k, 2), node indices.
        line_tags: Boundary tag per facet (k,).
        tag: Which boundary to return (13 = cushion).

    Returns:
        ``ends`` (k, 2) endpoint node indices, ``mid`` (k, 2), ``length`` (k,),
        ``n`` (k, 2) unit normal into the fluid, ``t_hat`` (k, 2) = (n_y, -n_x).
    """
    ends = lines[line_tags == tag]
    owner: dict[tuple[int, int], int] = {}
    for tri in triangles:
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i + 1) % 3])
            owner[(min(a, b), max(a, b))] = int(tri[(i + 2) % 3])
    third = np.array([owner[(min(a, b), max(a, b))] for a, b in ends])
    a, b = xy[ends[:, 0]], xy[ends[:, 1]]
    seg = b - a
    length = np.linalg.norm(seg, axis=1)
    n = np.stack([-seg[:, 1], seg[:, 0]], axis=1) / length[:, None]
    mid = 0.5 * (a + b)
    n *= np.sign(np.einsum("ij,ij->i", xy[third] - mid, n))[:, None]
    t_hat = np.stack([n[:, 1], -n[:, 0]], axis=1)
    return {"ends": ends, "mid": mid, "length": length, "n": n, "t_hat": t_hat}


def wall_traction(f: dict[str, np.ndarray], mu: float) -> dict[str, np.ndarray]:
    """Wall pressure and shear at the cushion facet midpoints from Dan's nodal fields.

    Pressure and velocity gradients at a facet midpoint are the mean of the facet's two
    endpoint nodes. That is exact for Dan's P1 pressure. For the gradients it interpolates
    his nodal values (presumably a projection of the discontinuous P2 gradient), which is
    the main error source: ~1e-3 to 1e-2 relative L2 in tau_t, see LOG.md.

    Args:
        f: A full dump (``load_full``): needs ``xy, triangles, lines, line_tags, p,
            u_x, u_y, v_x, v_y``.
        mu: Dynamic viscosity in Pa s.

    Returns:
        ``mid, n, t_hat`` (k, 2); ``p, tx, ty, tau_t, tau_mag`` (k,) in Pa, where
        ``(tx, ty) = sigma n`` is the traction the fluid exerts on the cushion.
    """
    geo = wall_facets(f["xy"], f["triangles"], f["lines"], f["line_tags"])
    e0, e1 = geo["ends"][:, 0], geo["ends"][:, 1]

    def at_mid(name: str) -> np.ndarray:
        return 0.5 * (f[name][e0] + f[name][e1])

    p = at_mid("p")
    ux, uy, vx, vy = at_mid("u_x"), at_mid("u_y"), at_mid("v_x"), at_mid("v_y")
    nx, ny = geo["n"][:, 0], geo["n"][:, 1]
    sxx, syy, sxy = -p + 2 * mu * ux, -p + 2 * mu * vy, mu * (uy + vx)
    tx, ty = sxx * nx + sxy * ny, sxy * nx + syy * ny
    t_n = tx * nx + ty * ny
    tau_x, tau_y = tx - t_n * nx, ty - t_n * ny
    return {
        "mid": geo["mid"],
        "n": geo["n"],
        "t_hat": geo["t_hat"],
        "p": p,
        "tx": tx,
        "ty": ty,
        "tau_t": tau_x * geo["t_hat"][:, 0] + tau_y * geo["t_hat"][:, 1],
        "tau_mag": np.hypot(tau_x, tau_y),
    }


def outlet_traction(f: dict[str, np.ndarray], mu: float) -> dict[str, np.ndarray]:
    """Outlet traction sigma n (n = +x) at the outlet nodes, wall corners excluded.

    Also returns the "do-nothing" variant ``(-p + mu u_x, mu v_x)`` that uses ``mu grad u``
    instead of the symmetric stress, so the two candidate outlet conditions can be compared.

    Args:
        f: A full dump with ``xy, lines, line_tags, p, u_x, u_y, v_x``.
        mu: Dynamic viscosity in Pa s.

    Returns:
        ``y`` (k,), ``sym`` (2, k), ``do_nothing`` (2, k) in Pa, and ``scale`` =
        mu max|u_y| over the outlet (Pa), the size of the viscous stress there.
    """
    nodes = np.unique(f["lines"][f["line_tags"] == 11])
    y = f["xy"][nodes, 1]
    nodes = nodes[(y > y.min()) & (y < y.max())]  # corners belong to the no-slip walls too
    p, ux, uy, vx = (f[k][nodes] for k in ("p", "u_x", "u_y", "v_x"))
    return {
        "y": f["xy"][nodes, 1],
        "sym": np.stack([-p + 2 * mu * ux, mu * (uy + vx)]),
        "do_nothing": np.stack([-p + mu * ux, mu * vx]),
        "scale": mu * np.abs(uy).max(),
    }


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    """Relative L2 error ||a - b|| / ||b||."""
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def check_dump(path: Path) -> dict[str, float]:
    """Compare every reference formula against one full dump; return the error metrics."""
    f = load_full(path)
    rho, mu, h = float(f["rho"]), float(f["mu"]), float(f["H"])
    u_bar = float(mean_inlet_speed(f["u_inlet_max"]))
    rx, ry, rdiv = ns_residual_dim(f, rho, mu)
    gx, _, _ = ns_residual_nondim(nondim_fields(f, u_bar, h, mu), float(f["Re"]))
    scale = sum(np.abs(t).sum(axis=0) for t in momentum_terms_dim(f, rho, mu).values())
    w = wall_traction(f, mu)
    n_dan = np.stack([f["arc_nx"], f["arc_ny"]], axis=1)
    tau_err = np.abs(w["tau_t"] - f["arc_tau_t"])
    out = outlet_traction(f, mu)
    return {
        "Re": float(f["Re"]),
        "r_mom_x_relL2": rel_l2(rx, f["r_mom_x"]),
        "r_mom_y_relL2": rel_l2(ry, f["r_mom_y"]),
        "r_mom_maxerr/scale": float(
            np.max(np.maximum(np.abs(rx - f["r_mom_x"]), np.abs(ry - f["r_mom_y"])) / scale)
        ),
        "r_div_relL2": rel_l2(rdiv, f["r_div"]),
        "r_rel_relL2": rel_l2(relative_residual(f, rho, mu), f["r_rel"]),
        "r_rel_maxabs_err": float(np.abs(relative_residual(f, rho, mu) - f["r_rel"]).max()),
        "nondim_vs_dim_relL2": rel_l2(to_dim(gx, "momentum", u_bar, h, mu), rx),
        "mid_vs_arc_xy_max_m": float(np.abs(w["mid"] - np.c_[f["arc_x"], f["arc_y"]]).max()),
        "normal_min_cos": float(np.einsum("ij,ij->i", w["n"], n_dan).min()),
        "p_relL2": rel_l2(w["p"], f["arc_p"]),
        "traction_relL2": rel_l2(np.c_[w["tx"], w["ty"]], np.c_[f["arc_tx"], f["arc_ty"]]),
        "tau_t_relL2": rel_l2(w["tau_t"], f["arc_tau_t"]),
        "tau_t_maxabs_Pa": float(tau_err.max()),
        "tau_t_max_Pa": float(np.abs(f["arc_tau_t"]).max()),
        "tau_t_worst_at_s/L": float(f["arc_s"][tau_err.argmax()] / f["arc_s"][-1]),
        "tau_t_flipped_relL2": rel_l2(-w["tau_t"], f["arc_tau_t"]),
        "tau_mag_relL2": rel_l2(w["tau_mag"], f["arc_tau_mag"]),
        "outlet_sym_rms/scale": float(np.sqrt(np.mean(out["sym"] ** 2)) / out["scale"]),
        "outlet_dn_rms/scale": float(np.sqrt(np.mean(out["do_nothing"] ** 2)) / out["scale"]),
        "outlet_max|v|/u_max": float(
            np.abs(f["v"][np.unique(f["lines"][f["line_tags"] == 11])]).max() / f["u_inlet_max"]
        ),
    }


def main() -> None:
    """Print the reference-vs-Dan agreement table for every full dump."""
    ap = argparse.ArgumentParser(description="Check reference formulas against Dan's dumps.")
    ap.add_argument("--full-dir", type=Path, default=FULL_DIR)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    paths = sorted(args.full_dir.glob("full_*.npz"))
    rows = {p.stem: check_dump(p) for p in paths}
    keys = list(next(iter(rows.values())))
    log.info("%-22s" + " %18s" * len(rows), "metric", *rows)
    for k in keys:
        log.info("%-22s" + " %18.3e" * len(rows), k, *(r[k] for r in rows.values()))


if __name__ == "__main__":
    main()
