#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_fsg.py
----------
Iterative fluid-solid-growth (FSG) loop.

Algorithm per FSG iteration:
  1. Build fluid mesh from current deformed solid arc (starts with your default dimensions)
  2. Solve steady state laminar Navier-Stokes at your prescribed flow rate  →  p(x), tau(x) on arc
  3. Map tractions onto solid mesh
  4. Short ramp solid mechanics (5 steps) for numerical stability - 5 is about the minimum I have found
  5. Hold + grow solid for N_HOLD steps (48) - approximates 2 days of growth right now
  6. Extract new deformed arc coordinates
  7. Save all outputs to  results/<flow_label>/step_<k>/

TO RUN:
  python3 run_fsg.py                          # uses default U_INLET
  python3 run_fsg.py --u_inlet 0.036          # normal flow
  python3 run_fsg.py --u_inlet 0.54         # high flow (150%)
  python3 run_fsg.py --u_inlet 0.018         # low flow (50%)

Output structure:
  results/
  └── flow_U0.084/
      ├── fsg_log.txt
      ├── step_000/
      │   ├── fluid_velocity.xdmf
      │   ├── fluid_pressure.xdmf
      │   ├── solid_displacement.xdmf
      │   ├── solid_fields.xdmf
      │   └── arc_data.npz
      ├── step_001/
      │   └── ...
      └── ...
"""

import os
import sys
import time
import math
import argparse
import numpy as np
from scipy.interpolate import splprep, splev

# ─────────────────────────────────────────────────────────────────────────────
# USER SETTINGS  (geometry dimensions, fluid channel dimensions, mesh settings, )
# ─────────────────────────────────────────────────────────────────────────────

# Geometry (metres)
LS           = 1e-3
A_CAP        = 0.40 * LS
B_CAP        = 0.14 * LS
XC           = 0.0
YC           = 0.0

# Fluid mesh
CHANNEL_X0   = -4.0  * LS
CHANNEL_X1   =  4.0  * LS
CHANNEL_Y0   =  0.0
CHANNEL_Y1   =  0.40 * LS
H_FLUID      =  0.020 * LS
H_ARC        =  0.004 * LS
N_ARC_PTS    =  300   # ~3.5 µm spacing along 1.05 mm arc — matches H_ARC

# Solid mesh
H_SOLID      = 0.004 * LS
N_ARC_SOLID  = 60

# Fluid properties
RHO          = 1060.0           # kg/m³
MU           = 3.5e-3           # Pa·s
U_INLET      = 0.040        # m/s  HH21-25 AV canal mid-range (override with --u_inlet)

# FSG loop
N_FSG        = 48           # number of geometry-update cycles
N_RAMP_FIRST = 10               # ramp steps on iteration 0 (cold start)
N_RAMP       = 1                 # subsequent iterations: solid already near equilibrium,
                                 # so go straight to full scale in one Newton solve
N_HOLD       = 10             # growth hold steps per FSG iteration
N_REMESH     = 5             # rebuild solid mesh every this many FSG iterations (0 = disabled)
N_RAMP_REMESH = 5            # ramp steps on first solid solve after a remesh

# Output root
RESULTS_ROOT = "FSG Results"

# ─────────────────────────────────────────────────────────────────────────────
# Command-line overrides
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--u_inlet", type=float, default=U_INLET)
    p.add_argument("--n_fsg",   type=int,   default=N_FSG)
    p.add_argument("--n_hold",  type=int,   default=N_HOLD)
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Logging: mirror stdout to file, watch with tail -f whatever_the_output_file_is.out
# ─────────────────────────────────────────────────────────────────────────────

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_step_dir(run_dir, k):
    d = os.path.join(run_dir, f"step_{k:03d}")
    os.makedirs(d, exist_ok=True)
    return d

def initial_arc_coords(a, b, xc, yc, n=180):
    """Undeformed ellipse arc, left → right, (N,2) metres."""
    pts = np.array([
        [xc + a * math.cos(math.pi * i / (n - 1)),
         yc + b * math.sin(math.pi * i / (n - 1))]
        for i in range(n)
    ])
    pts = pts[np.argsort(pts[:, 0])]
    return pts

def save_fluid_xdmf(step_dir, u_f, p_f):
    """Save velocity and pressure to separate XDMF files, check 'em out in paraview."""
    from dolfin import XDMFFile, MPI
    for func, name in [(u_f, "fluid_velocity"), (p_f, "fluid_pressure")]:
        path = os.path.join(step_dir, f"{name}.xdmf")
        xf = XDMFFile(MPI.comm_world, path)
        xf.parameters["flush_output"] = True
        xf.write(func, 0.0)
        xf.close()

def _write_solid_fields(xf, solver, t):
    """Write all solid fields to an open XDMFFile at time t."""
    from dolfin import project, Function

    def pw(expr, space, name):
        fn = project(expr, space)
        fn.rename(name, name)
        xf.write(fn, t)

    # Displacement (vector)
    u_out = Function(solver.V)
    u_out.assign(solver.u)
    u_out.rename("displacement", "displacement")
    xf.write(u_out, t)

    # Scalar stress invariants
    pw(solver._sigma_vm_expr,     solver.S,   "von_mises")
    pw(solver._sigma_max_expr,    solver.S,   "sigma_max_principal")
    pw(solver._sigma_min_expr,    solver.S,   "sigma_min_principal")
    pw(solver._sigma_ten_expr,    solver.S,   "sigma_tension")
    pw(solver._sigma_comp_expr,   solver.S,   "sigma_comp_mag")

    # Jacobians
    pw(solver._Je_expr,           solver.Vd0, "J_e")
    pw(solver._Jg_expr,           solver.Vd0, "J_g")

    # Growth factor
    g_out = Function(solver.Vg)
    g_out.assign(solver.g)
    g_out.rename("growth_factor_g", "growth_factor_g")
    xf.write(g_out, t)

    # Tensor fields
    pw(solver._F,                 solver.T,   "deformation_gradient_F")
    pw(solver._E_GL_expr,         solver.T,   "green_lagrange_strain_E")
    pw(solver._sigma_cauchy_expr, solver.T,   "cauchy_stress")


def save_solid_xdmf(step_dir, solver, step_label):
    """Save all solid fields to a per-step XDMF file."""
    from dolfin import XDMFFile, MPI

    f_path = os.path.join(step_dir, "solid_fields.xdmf")
    xf = XDMFFile(MPI.comm_world, f_path)
    xf.parameters["flush_output"] = True
    xf.parameters["functions_share_mesh"] = True
    _write_solid_fields(xf, solver, float(step_label))
    xf.close()

def save_arc_npz(step_dir, arc_data):
    path = os.path.join(step_dir, "arc_data.npz")
    np.savez(path, **arc_data)

def get_deformed_arc(mesh_s, ftags_s, u_solid, n_pts=None):
    """
    Build the deformed arc from the arc-boundary vertices of the solid mesh,
    then smooth and resample to give gmsh a clean, uniformly-spaced curve.

    Steps:
      1. Collect arc boundary vertices (facet tag 2) via DOF array —
         avoids FEniCS point-evaluation failures at boundary nodes.
      2. Sort by REFERENCE angle (π → 0, left to right) — robust under
         large asymmetric deformation where deformed-x ordering can fail.
      3. Clamp y ≥ CHANNEL_Y0 — prevents the arc dipping below the floor
         and intersecting the bottom-wall curve in gmsh.
      4. Fit a cubic spline and resample at n_pts uniform arc-length
         positions — removes node bunching near the base and gives gmsh
         a smooth BSpline input.
    """
    from dolfin import facets, vertices

    n = n_pts if n_pts is not None else N_ARC_PTS

    # ── 1. Collect arc boundary vertex indices (facet tag 2) ─────────────
    arc_vids = set()
    for facet in facets(mesh_s):
        if ftags_s[facet] == 2:
            for v in vertices(facet):
                arc_vids.add(v.index())
    arc_vids = np.array(sorted(arc_vids))

    coords  = mesh_s.coordinates()
    arc_ref = coords[arc_vids]

    # ── 2. Displacements via compute_vertex_values (safe at boundary, still gives me border issues occassionally) ─────
    n_verts = mesh_s.num_vertices()
    u_vals  = u_solid.compute_vertex_values(mesh_s)
    ux      = u_vals[:n_verts]
    uy      = u_vals[n_verts:]

    disp    = np.column_stack([ux[arc_vids], uy[arc_vids]])
    arc_def = arc_ref + disp

    # ── 3. Sort by reference angle (π → 0 = left → right) ────────────────
    ref_angles = np.arctan2(arc_ref[:, 1] - YC, arc_ref[:, 0] - XC)
    order      = np.argsort(-ref_angles)   # descending: π first
    arc_def    = arc_def[order]

    # Clamp: no node below the channel floor
    arc_def[:, 1] = np.maximum(arc_def[:, 1], CHANNEL_Y0)

    # ── 4. Spline fit + uniform resample at n_pts ─────────────────────────
    # Use s=0 (exact interpolating spline) to preserve all spatial detail
    # from the solid mesh. The solid Newton solve converges to rtol=1e-9 so
    # arc vertex positions carry negligible FEM noise.
    tck, _ = splprep([arc_def[:, 0], arc_def[:, 1]],
                     s=0, k=3, per=False)
    u_new  = np.linspace(0.0, 1.0, n)
    x_new, y_new = splev(u_new, tck)
    arc_smooth = np.column_stack([x_new, y_new])

    # Final clamp + endpoint snap
    arc_smooth[:, 1]  = np.maximum(arc_smooth[:, 1], CHANNEL_Y0)
    arc_smooth[0,  1] = CHANNEL_Y0
    arc_smooth[-1, 1] = CHANNEL_Y0

    return arc_smooth

# ─────────────────────────────────────────────────────────────────────────────
# Solid remeshing
# ─────────────────────────────────────────────────────────────────────────────

def remesh_solid(arc_coords, run_dir, k, rank):
    """
    Rebuild the solid mesh on the current deformed geometry (updated Lagrangian).

    The unloaded, grown shape becomes the new stress-free reference configuration;
    accumulated growth history is encoded in the mesh coordinates.  The growth
    factor g is reset to 1.0 and displacement u to 0 on the new mesh.

    Parameters
    ----------
    arc_coords : (N, 2) array  — deformed arc from get_deformed_arc (left→right)
    run_dir    : str           — run output directory (for .msh and .npy files)
    k          : int           — current FSG iteration index (for file naming)
    rank       : int           — MPI rank (for print guards)

    Returns
    -------
    (new_solver, new_mesh, new_ftags)
    """
    from mesh_builder import build_solid_mesh_from_arc, msh_to_fenics
    from solid_solver import SolidSolver

    if rank == 0:
        print(f"\n  [REMESH] Rebuilding solid mesh at end of FSG step {k}")

    # Downsample arc to N_ARC_SOLID control points for the gmsh spline
    # (get_deformed_arc returns N_ARC_PTS=300 pts; solid mesh only needs ~60)
    n_ctrl = N_ARC_SOLID
    if len(arc_coords) > n_ctrl:
        idx = np.round(np.linspace(0, len(arc_coords) - 1, n_ctrl)).astype(int)
        arc_for_mesh = arc_coords[idx]
    else:
        arc_for_mesh = arc_coords

    remesh_path = os.path.join(run_dir, f"solid_remesh_{k:03d}.msh")
    build_solid_mesh_from_arc(remesh_path, arc_for_mesh, h_size=H_SOLID)

    new_mesh, _, new_ftags = msh_to_fenics(remesh_path)

    if rank == 0:
        print(f"  [REMESH] New solid mesh: {new_mesh.num_cells()} cells, "
              f"{new_mesh.num_vertices()} vertices  (u=0, g=1 on new reference)")

    new_solver = SolidSolver(new_mesh, new_ftags)
    new_solver.initialize()

    # Save new cell centroids for post-processing growth maps
    coords_new = new_mesh.coordinates()
    cell_mp_new = np.array([coords_new[c].mean(axis=0) for c in new_mesh.cells()])
    np.save(os.path.join(run_dir, f"cell_centroids_remesh_{k:03d}.npy"), cell_mp_new)

    return new_solver, new_mesh, new_ftags


# ─────────────────────────────────────────────────────────────────────────────
# Main FSG loop
# ─────────────────────────────────────────────────────────────────────────────

def _save_cumulative_g(run_dir, n_fsg, n_remesh, final_mesh, Vg, final_dir):
    """
    Compute cumulative growth factor across all remesh generations and write
    two DG0 fields to final_dir/cumulative_growth.xdmf:
      cumulative_g        — area growth factor product (g₀ × g₁ × …) - track it over time
      cumulative_g_linear — sqrt(cumulative_g), linear stretch equivalent - should really be nonlinear? Don't use currently.
    Also saves cumulative_g.npy alongside for post-processing scripts.
    """
    from dolfin import Function, XDMFFile, MPI
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    # Detect last step of each generation (same logic as remesh trigger)
    last_step = n_fsg - 1
    remesh_steps = [k for k in range(n_fsg)
                    if (k + 1) % n_remesh == 0 and k < last_step]
    gen_last_steps = remesh_steps + [last_step]

    centroid_files = [os.path.join(run_dir, "cell_centroids.npy")]
    for rs in remesh_steps:
        centroid_files.append(
            os.path.join(run_dir, f"cell_centroids_remesh_{rs:03d}.npy"))

    # Final mesh cell centroids as query frame
    coords = final_mesh.coordinates()
    query_pts = np.array([coords[cell].mean(axis=0)
                          for cell in final_mesh.cells()])

    g_cumulative = np.ones(len(query_pts))

    for i, (last_k, cfile) in enumerate(zip(gen_last_steps, centroid_files)):
        gfile = os.path.join(run_dir, f"step_{last_k:03d}", "g_field.npy")
        if not os.path.exists(gfile) or not os.path.exists(cfile):
            print(f"    [WARN] Missing gen {i} data, skipping.")
            continue

        g_src = np.load(gfile)
        c_src = np.load(cfile)

        if i == len(gen_last_steps) - 1:
            g_here = g_src.copy()
        else:
            lin = LinearNDInterpolator(c_src, g_src, fill_value=np.nan)
            g_here = lin(query_pts)
            nan_mask = np.isnan(g_here)
            if nan_mask.any():
                nn = NearestNDInterpolator(c_src, g_src)
                g_here[nan_mask] = nn(query_pts[nan_mask])

        g_cumulative *= g_here

    g_linear = np.sqrt(np.maximum(g_cumulative, 0.0))
    np.save(os.path.join(run_dir, "cumulative_g.npy"), g_cumulative)

    print(f"    cumulative g      : [{g_cumulative.min():.4f}, {g_cumulative.max():.4f}]")
    print(f"    linear stretch √g : [{g_linear.min():.4f}, {g_linear.max():.4f}]")

    # Write to XDMF as DG0 fields on the final mesh
    xf_path = os.path.join(final_dir, "cumulative_growth.xdmf")
    xf = XDMFFile(MPI.comm_world, xf_path)
    xf.parameters["flush_output"] = True
    xf.parameters["functions_share_mesh"] = True

    for values, name in [(g_cumulative, "cumulative_g"),
                         (g_linear,     "cumulative_g_linear")]:
        fn = Function(Vg)
        fn.vector().set_local(values)
        fn.vector().apply("insert")
        fn.rename(name, name)
        xf.write(fn, 0.0)

    xf.close()
    print(f"    Saved → {xf_path}")


def main():
    args = parse_args()
    u_inlet = args.u_inlet
    n_fsg   = args.n_fsg
    n_hold  = args.n_hold

    # ── Output directory ──────────────────────────────────────────────────
    flow_label = f"flow_U{u_inlet:.4f}".replace(".", "p")
    run_dir    = os.path.join(RESULTS_ROOT, flow_label)
    os.makedirs(run_dir, exist_ok=True)

    # ── Logging ───────────────────────────────────────────────────────────
    from dolfin import MPI
    rank = MPI.comm_world.Get_rank()
    if rank == 0:
        log_path = os.path.join(run_dir, "fsg_log.txt")
        _log     = open(log_path, "w")
        sys.stdout = Tee(sys.stdout, _log)
        sys.stderr = sys.stdout

    # ── Imports ───────────────────────────────────────────────────────────
    from mesh_builder    import build_solid_mesh, build_fluid_mesh, msh_to_fenics
    from fluid_solver    import solve_navier_stokes
    from traction_mapper import build_traction_expressions
    from solid_solver    import SolidSolver

    t0 = time.time()

    if rank == 0:
        print("=" * 65)
        print(f"  FSG Loop  —  U_inlet = {u_inlet} m/s  ({u_inlet*100:.2f} cm/s)")
        print(f"  N_FSG={n_fsg}  N_RAMP_FIRST={N_RAMP_FIRST}  "
              f"N_RAMP={N_RAMP}  N_HOLD={n_hold}")
        print(f"  Re ≈ {RHO * u_inlet * (CHANNEL_Y1 - CHANNEL_Y0) / MU:.1f}")
        print(f"  Output: {run_dir}")
        print("=" * 65)

    # ── Build solid mesh (fixed throughout — remesh not needed for solid usually since this is steady state and a longer timeframe) ─
    solid_msh = os.path.join(run_dir, "solid.msh")
    build_solid_mesh(solid_msh,
                     a_cap=A_CAP, b_cap=B_CAP,
                     xc=XC, yc=YC,
                     h_size=H_SOLID, n_arc_pts=N_ARC_SOLID)

    mesh_s, _, ftags_s = msh_to_fenics(solid_msh)

    if rank == 0:
        print(f"Solid mesh: {mesh_s.num_cells()} cells, "
              f"{mesh_s.num_vertices()} vertices\n")

    # Save cell centroids once (for post-processing growth along arc and output to GRN)
    coords_s = mesh_s.coordinates()
    cell_mp  = np.array([coords_s[c].mean(axis=0) for c in mesh_s.cells()])
    np.save(os.path.join(run_dir, "cell_centroids.npy"), cell_mp)

    # ── Initialise solid solver (persists across all FSG iterations) ──────
    solver = SolidSolver(mesh_s, ftags_s)
    solver.initialize()

    # ── Initial arc (undeformed) ──────────────────────────────────────────
    arc_coords = initial_arc_coords(A_CAP, B_CAP, XC, YC, n=N_ARC_PTS)

    # ── Open final combined XDMF (one file per mesh generation) ──────────
    from dolfin import XDMFFile
    gen_idx  = 0   # increments each time the solid is remeshed
    final_dir = os.path.join(run_dir, "final_gen0")
    os.makedirs(final_dir, exist_ok=True)
    final_xf = XDMFFile(MPI.comm_world, os.path.join(final_dir, "solid_fields.xdmf"))
    final_xf.parameters["flush_output"] = True
    final_xf.parameters["functions_share_mesh"] = True

    # ── FSG loop ──────────────────────────────────────────────────────────
    global_solid_step = 0   # monotonically increasing step counter
    _remeshed_last_iter = False

    for k in range(n_fsg):

        if rank == 0:
            print()
            print(f"{'─' * 65}")
            print(f"  FSG iteration {k:03d} / {n_fsg - 1:03d}")
            print(f"{'─' * 65}")

        step_dir = make_step_dir(run_dir, k)

        # ── 1. Build fluid mesh from current arc ──────────────────────────
        fluid_msh = os.path.join(step_dir, "fluid.msh")
        try:
            build_fluid_mesh(fluid_msh, arc_coords,
                             channel_x0=CHANNEL_X0, channel_x1=CHANNEL_X1,
                             channel_y0=CHANNEL_Y0, channel_y1=CHANNEL_Y1,
                             h_fluid=H_FLUID, h_arc=H_ARC)
        except Exception as e:
            if rank == 0:
                print(f"  [WARN] Fluid mesh build failed at step {k}: {e}")
                print("  Falling back to undeformed arc.")
            arc_coords = initial_arc_coords(A_CAP, B_CAP, XC, YC, n=N_ARC_PTS)
            build_fluid_mesh(fluid_msh, arc_coords,
                             channel_x0=CHANNEL_X0, channel_x1=CHANNEL_X1,
                             channel_y0=CHANNEL_Y0, channel_y1=CHANNEL_Y1,
                             h_fluid=H_FLUID, h_arc=H_ARC)

        mesh_f, _, ftags_f = msh_to_fenics(fluid_msh)

        if rank == 0:
            print(f"  Fluid mesh: {mesh_f.num_cells()} cells")

        # ── 2. Solve fluid domain ────────────────────────────────────────
        try:
            u_f, p_f, arc_data = solve_navier_stokes(
                mesh_f, ftags_f,
                rho=RHO, mu=MU,
                u_inlet_max=u_inlet,
                channel_y0=CHANNEL_Y0,
                channel_y1=CHANNEL_Y1,
            )
        except Exception as e:
            if rank == 0:
                print(f"  [ERROR] NS solve failed at step {k}: {e}")
                print("  Skipping to next FSG iteration.")
            continue

        # ── 3. Save fluid outputs ─────────────────────────────────────────
        save_fluid_xdmf(step_dir, u_f, p_f)
        save_arc_npz(step_dir, arc_data)

        if rank == 0:
            print(f"  p    range: [{arc_data['p'].min():.3e}, "
                  f"{arc_data['p'].max():.3e}] Pa")
            print(f"  |tau| range: [{arc_data['tau_mag'].min():.3e}, "
                  f"{arc_data['tau_mag'].max():.3e}] Pa")

        # ── 4. Map traction onto solid ────────────────────────────────────
        p_expr, wss_expr = build_traction_expressions(arc_data, degree=1)
        solver.set_traction(p_expr, wss_expr)

        # ── 5. Solid ramp - can make longer, but it gets slower, albeit more stable... ─────────────────────────────────────────────────
        if k == 0:
            n_ramp = N_RAMP_FIRST
        elif _remeshed_last_iter:
            n_ramp = N_RAMP_REMESH
        else:
            n_ramp = N_RAMP
        _remeshed_last_iter = False

        solver.run_ramp(
            n_ramp      = n_ramp,
            p_magnitude = 1.0,
            xdmf_file   = None,          # fields saved per-step below
            step_offset = global_solid_step,
        )
        global_solid_step += n_ramp

        # ── 6. Solid hold + grow!!!! ──────────────────────────────────────────
        solver.run_hold(
            n_hold      = n_hold,
            xdmf_file   = None,
            step_offset = global_solid_step,
        )
        global_solid_step += n_hold

        # ── 7. Save solid outputs for this FSG step ───────────────────────
        save_solid_xdmf(step_dir, solver, step_label=k)
        _write_solid_fields(final_xf, solver, float(k))

        # Also save raw g array for post-processing
        np.save(os.path.join(step_dir, "g_field.npy"),
                solver.g.vector().get_local())

        # ── 8. Extract deformed arc for next iteration ────────────────────
        arc_coords = get_deformed_arc(mesh_s, ftags_s, solver.u)

        # ── 9. Step summary and let 'em know ───────────────────────────────────────────────
        g_arr = solver.g.vector().get_local()
        u_arr = solver.u.vector().get_local()

        if rank == 0:
            max_disp = float(np.max(np.abs(u_arr)))
            arc_shift = float(np.max(
                np.linalg.norm(
                    arc_coords - initial_arc_coords(A_CAP, B_CAP, XC, YC,
                                                    n=len(arc_coords)),
                    axis=1)
            ))
            print(f"\n  ── Step {k:03d} summary ──────────────────────────")
            print(f"  Max |u|     : {max_disp*1e6:.2f} µm")
            print(f"  Max arc shift: {arc_shift*1e6:.2f} µm")
            print(f"  g range     : [{g_arr.min():.4f}, {g_arr.max():.4f}]")
            print(f"  g mean      : {g_arr.mean():.4f}")
            print(f"  Elapsed     : {time.time() - t0:.1f} s")
            print(f"  Saved to    : {step_dir}")

        # ── 10. Remesh solid if due ───────────────────────────────────────
        # Skipped on the last FSG iteration (no subsequent solve to benefit).
        if N_REMESH > 0 and (k + 1) % N_REMESH == 0 and k < n_fsg - 1:
            solver, mesh_s, ftags_s = remesh_solid(arc_coords, run_dir, k, rank)

            # Each mesh generation gets its own time-series XDMF (mesh changes
            # between generations so a single file cannot span the remesh).
            final_xf.close()
            gen_idx += 1
            final_dir = os.path.join(run_dir, f"final_gen{gen_idx}")
            os.makedirs(final_dir, exist_ok=True)
            final_xf = XDMFFile(MPI.comm_world,
                                 os.path.join(final_dir, "solid_fields.xdmf"))
            final_xf.parameters["flush_output"] = True
            final_xf.parameters["functions_share_mesh"] = True

            _remeshed_last_iter = True

    # ── Done ──────────────────────────────────────────────────────────────
    final_xf.close()

    # ── Cumulative growth factor ───────────────────────────────────────────
    # Compute g_cumulative = product of g across all remesh generations,
    # interpolated onto the final mesh, and save as a DG0 XDMF field.
    if rank == 0:
        print("\n  [Post] Computing cumulative growth factor ...")
        _save_cumulative_g(run_dir, n_fsg, N_REMESH, mesh_s, solver.Vg, final_dir)

    if rank == 0:
        print()
        print("=" * 65)
        print("  FSG loop complete.")
        print(f"  Total runtime : {time.time() - t0:.1f} s")
        print(f"  Results in    : {run_dir}")
        print()
        print("  Per-step outputs (results/step_NNN/):")
        print("    fluid_velocity.xdmf   — velocity field (ParaView)")
        print("    fluid_pressure.xdmf   — pressure field (ParaView)")
        print("    solid_fields.xdmf     — all solid fields (ParaView)")
        print("    arc_data.npz          — p, tau arrays (plotting)")
        print("    g_field.npy           — raw growth factor array")
        print()
        print("  Combined time-series (final_genN/ per mesh generation):")
        print("    solid_fields.xdmf     — steps within that generation, scrub in ParaView")
        print("=" * 65)


if __name__ == "__main__":
    main()