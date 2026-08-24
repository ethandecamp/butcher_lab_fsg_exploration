#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
traction_mapper.py
------------------
Maps CFD-extracted arc traction data (pressure + WSS) onto FEniCS
UserExpression objects that can be plugged directly into the solid solver's
weak form as Neumann boundary tractions.

Design
------
The CFD returns arrays of  (x, p, tau_x, tau_y)  sampled at facet midpoints
along the arc, ordered by x-coordinate.  We interpolate these onto any spatial
point (x, y) that the FEniCS assembler queries during integration.

Two mappers are provided:
  PressureOnArc  – scalar normal pressure   p(x)
  WSSOnArc       – vector traction [tau_x, tau_y] (tangential + normal viscous)

Both use scipy 1-D interpolation (linear by default, cubic optional) along x.
This is exact enough given that the arc is a smooth monotone curve in x.

Usage (inside solid_solver.py)
------
    from traction_mapper import build_traction_expressions
    p_expr, wss_expr = build_traction_expressions(arc_data, mesh, degree=1)

    # Neumann term in weak form:
    traction = -p_expr * n + wss_expr   # vector
    Pi += - inner(traction, u) * ds(2)
"""

import numpy as np
from dolfin import UserExpression, MPI


class PressureOnArc(UserExpression):
    """Scalar pressure interpolated along arc x-coordinate."""
    def __init__(self, x_arr, p_arr, **kwargs):
        super().__init__(**kwargs)
        from scipy.interpolate import interp1d
        self._interp = interp1d(x_arr, p_arr, kind="linear",
                                bounds_error=False,
                                fill_value=(p_arr[0], p_arr[-1]))
    def eval(self, values, x):
        values[0] = float(self._interp(x[0]))
    def value_shape(self):
        return ()


class WSSOnArc(UserExpression):
    """
    Tangential WSS traction interpolated along arc x-coordinate.
    Returns the tangential component only (no pressure).
    tau_x, tau_y are the tangential components extracted from CFD.
    """
    def __init__(self, x_arr, tau_x_arr, tau_y_arr, **kwargs):
        super().__init__(**kwargs)
        from scipy.interpolate import interp1d
        self._ix = interp1d(x_arr, tau_x_arr, kind="linear",
                            bounds_error=False,
                            fill_value=(tau_x_arr[0], tau_x_arr[-1]))
        self._iy = interp1d(x_arr, tau_y_arr, kind="linear",
                            bounds_error=False,
                            fill_value=(tau_y_arr[0], tau_y_arr[-1]))
    def eval(self, values, x):
        values[0] = float(self._ix(x[0]))
        values[1] = float(self._iy(x[0]))
    def value_shape(self):
        return (2,)


def build_traction_expressions(arc_data, degree=1):
    """
    Returns (p_expr, wss_expr) separately so the solid solver
    can apply them the same way as the original valve_growth.py:
        traction = -p_expr * n  +  wss_expr
    where n is the FEniCS FacetNormal (outward from solid).
    """
    x    = np.asarray(arc_data['x'],     dtype=float)
    p    = np.asarray(arc_data['p'],     dtype=float)
    tx   = np.asarray(arc_data['tau_x'], dtype=float)
    ty   = np.asarray(arc_data['tau_y'], dtype=float)

    order = np.argsort(x)
    x, p, tx, ty = x[order], p[order], tx[order], ty[order]

    if MPI.comm_world.rank == 0:
        print(f"[TractionMapper] Arc points : {len(x)}")
        print(f"[TractionMapper] p     range: [{p.min():.4e}, {p.max():.4e}] Pa")
        tau_mag = np.sqrt(tx**2 + ty**2)
        print(f"[TractionMapper] |tau| range: [{tau_mag.min():.4e}, {tau_mag.max():.4e}] Pa")

    p_expr   = PressureOnArc(x, p,      degree=degree)
    wss_expr = WSSOnArc(x, tx, ty,      degree=degree)

    return p_expr, wss_expr


# ─────────────────────────────────────────────────────────────────────────────
# Utility: extract deformed arc coordinates from solid mesh + displacement
# ─────────────────────────────────────────────────────────────────────────────

def get_deformed_arc_coords(mesh, facet_tags, u_solid, arc_tag=2):
    """
    Given the solid mesh, its facet tags, and the current displacement field,
    return the deformed coordinates of the arc boundary nodes.

    Returns
    -------
    arc_xy_ref      : (N,2) reference (undeformed) arc coordinates (m)
    arc_xy_deformed : (N,2) deformed arc coordinates (m)
    arc_order       : indices sorted by x_ref (left to right)
    """
    from dolfin import vertices as dolfin_vertices, facets as dolfin_facets
    import numpy as np

    coords = mesh.coordinates()  # (n_verts, 2)

    arc_vertex_set = set()
    for f in dolfin_facets(mesh):
        if not f.exterior():
            continue
        if int(facet_tags[f]) != arc_tag:
            continue
        for v in dolfin_vertices(f):
            arc_vertex_set.add(v.index())

    arc_idx = sorted(arc_vertex_set)

    arc_xy_ref = coords[arc_idx, :].copy()

    # Evaluate displacement at each arc vertex
    disp = np.zeros_like(arc_xy_ref)
    for i, vi in enumerate(arc_idx):
        pt = arc_xy_ref[i]
        try:
            d = u_solid(pt)
            disp[i, 0] = float(d[0])
            disp[i, 1] = float(d[1])
        except Exception:
            pass  # boundary of function space – leave as 0

    arc_xy_deformed = arc_xy_ref + disp

    # Sort left → right by reference x
    order = np.argsort(arc_xy_deformed[:, 0])
    arc_xy_ref      = arc_xy_ref[order]
    arc_xy_deformed = arc_xy_deformed[order]

    return arc_xy_ref, arc_xy_deformed, order