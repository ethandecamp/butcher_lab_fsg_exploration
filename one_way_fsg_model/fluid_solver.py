#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fluid_solver.py
---------------
Steady incompressible Navier-Stokes on the fluid channel domain.

Formulation  : Taylor-Hood P2/P1 (velocity / pressure)
Stabilisation: none needed for Taylor-Hood at moderate Re
               (add SUPG/PSPG here if Re >> 100)

Boundary conditions (physical tag convention from mesh_builder.py):
  Tag 10 (inlet)      : parabolic u_x profile, u_y = 0
  Tag 11 (outlet)     : do-nothing (natural stress-free BC)
  Tag 12 (top wall)   : no-slip
  Tag 13 (solid arc)  : no-slip  ← traction extracted here
  Tag 14 (bottom wall): no-slip  (or symmetry – see SYMMETRY_BOTTOM flag)

Returns
-------
(u_f, p_f) : FEniCS Function objects (velocity, pressure) on the fluid mesh
arc_data   : dict with keys
               'x'    : (N,) arc x-coords (m)
               'p'    : (N,) pressure  (Pa)
               'tau_x': (N,) x-component of wall shear stress (Pa)
               'tau_y': (N,) y-component of wall shear stress (Pa)
               'tau_mag': (N,) |tau| (Pa)
               'nx'   : (N,) outward normal x (pointing INTO fluid)
               'ny'   : (N,) outward normal y
"""

import os
import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
# Main solver
# ═════════════════════════════════════════════════════════════════════════════

def solve_navier_stokes(mesh, facet_tags,
                        rho, mu,
                        u_inlet_max,
                        channel_y0, channel_y1,
                        symmetry_bottom=False,
                        newton_max_iter=50,
                        newton_rtol=1e-8,
                        newton_atol=1e-10,
                        newton_relax=0.8,
                        n_sample_arc=200):
    """
    Parameters
    ----------
    mesh, facet_tags : dolfin objects from mesh_builder.msh_to_fenics
    rho              : fluid density   (kg/m³)
    mu               : dynamic viscosity (Pa·s)
    u_inlet_max      : peak inlet velocity (m/s)  – parabolic profile
    channel_y0/y1    : channel bottom / top y-coordinate (m)
    symmetry_bottom  : if True, apply symmetry (du/dn=0, p free) on tag 14
    n_sample_arc     : number of quadrature-like sample points along arc

    Returns
    -------
    (u_f, p_f, arc_data)
    """
    from dolfin import (
        VectorFunctionSpace, FunctionSpace, MixedElement,
        Function, TestFunctions, TrialFunction, split,
        DirichletBC, Constant, FacetNormal, Identity,
        grad, div, inner, dot, dx, Measure, derivative,
        solve, parameters, Expression, MPI, assemble,
        project, sym
    )
    import ufl

    parameters["form_compiler"]["quadrature_degree"] = 4

    # ── Function spaces (Taylor-Hood P2/P1) ──────────────────────────────
    P2 = VectorFunctionSpace(mesh, "CG", 2)
    P1 = FunctionSpace(mesh, "CG", 1)

    # Mixed space
    element = MixedElement([P2.ufl_element(), P1.ufl_element()])
    W = FunctionSpace(mesh, element)  # NOTE: FunctionSpace on MixedElement

    w  = Function(W)
    (v, q) = TestFunctions(W)
    (u, p) = split(w)

    ds = Measure("ds", domain=mesh, subdomain_data=facet_tags)

    # ── Inlet: parabolic profile ──────────────────────────────────────────
    H = float(channel_y1 - channel_y0)
    # u_x = u_max * 4*(y-y0)*(y1-y)/(H^2),  u_y = 0
    inlet_expr = Expression(
        ("u_max * 4.0*(x[1]-y0)*(y1-x[1]) / (H*H)", "0.0"),
        u_max=float(u_inlet_max),
        y0=float(channel_y0),
        y1=float(channel_y1),
        H=float(H),
        degree=2
    )

    # ── Boundary conditions ───────────────────────────────────────────────
    bc_inlet  = DirichletBC(W.sub(0), inlet_expr,   facet_tags, 10)
    bc_top    = DirichletBC(W.sub(0), Constant((0.0, 0.0)), facet_tags, 12)
    bc_arc    = DirichletBC(W.sub(0), Constant((0.0, 0.0)), facet_tags, 13)

    bcs = [bc_inlet, bc_top, bc_arc]

    if not symmetry_bottom:
        bc_bot = DirichletBC(W.sub(0), Constant((0.0, 0.0)), facet_tags, 14)
        bcs.append(bc_bot)
    # outlet (tag 11): do-nothing → natural BC (zero traction) — no explicit BC needed

    # ── Weak form (steady NS, Newton linearisation) ───────────────────────
    rho_c = Constant(float(rho))
    mu_c  = Constant(float(mu))

    def epsilon(u_):
        return sym(grad(u_))

    def sigma_f(u_, p_):
        return 2.0 * mu_c * epsilon(u_) - p_ * Identity(2)

    # Residual
    F = (
        rho_c * inner(dot(u, grad(u)), v) * dx
        + inner(sigma_f(u, p), epsilon(v)) * dx
        - inner(Constant((0.0, 0.0)), v) * dx   # no body force
        + div(u) * q * dx
    )

    J = derivative(F, w)

    prm = {
        "newton_solver": {
            "maximum_iterations": int(newton_max_iter),
            "relative_tolerance": float(newton_rtol),
            "absolute_tolerance": float(newton_atol),
            "linear_solver":      "mumps",
            "report":             True,
            "relaxation_parameter": float(newton_relax),
        }
    }

    if MPI.comm_world.rank == 0:
        Re = rho * u_inlet_max * H / mu
        print(f"[NS] Re ≈ {Re:.1f}  (rho={rho}, mu={mu}, U={u_inlet_max}, H={H:.4e})")
        print(f"[NS] Solving Navier-Stokes ...")

    solve(F == 0, w, bcs=bcs, J=J, solver_parameters=prm)

    u_f, p_f = w.split(deepcopy=True)
    u_f.rename("velocity", "velocity")
    p_f.rename("pressure", "pressure")

    if MPI.comm_world.rank == 0:
        u_max_sol = u_f.vector().norm("linf")
        print(f"[NS] Solved. |u|_max = {u_max_sol:.4e} m/s")

    # ── Extract traction on arc (tag 13) ──────────────────────────────────
    arc_data = _extract_arc_traction(mesh, facet_tags, u_f, p_f, mu, n_sample_arc)

    return u_f, p_f, arc_data


# ═════════════════════════════════════════════════════════════════════════════
# Traction extraction on the solid arc
# ═════════════════════════════════════════════════════════════════════════════

def _extract_arc_traction(mesh, facet_tags, u_f, p_f, mu, n_sample=200):
    """
    Compute wall traction on arc (tag 13) using a projected stress tensor.
    Avoids finite-difference errors entirely.
    """
    from dolfin import (
        TensorFunctionSpace, FunctionSpace, VectorFunctionSpace,
        project, FacetNormal, Identity, sym, grad,
        facets as dolfin_facets, vertices as dolfin_vertices,
        Constant, MPI
    )
    import numpy as np

    # ── Project full Cauchy stress tensor onto CG1 ────────────────────────
    # sigma_f = -p*I + mu*(grad u + grad u^T)
    T  = TensorFunctionSpace(mesh, "CG", 1)
    mu_c = Constant(float(mu))
    I2   = Identity(2)

    sigma_proj = project(
        -p_f * I2 + mu_c * (grad(u_f) + grad(u_f).T),
        T
    )

    if MPI.comm_world.rank == 0:
        print("[NS] Stress tensor projected onto CG1.")

    # ── Walk arc facets, evaluate traction t = sigma · n ──────────────────
    records = []

    for f in dolfin_facets(mesh):
        if not f.exterior():
            continue
        if int(facet_tags[f]) != 13:
            continue

        # Facet midpoint
        mp  = f.midpoint()
        x_pt = np.array([mp.x(), mp.y()])

        # Outward normal from facet geometry
        verts = list(dolfin_vertices(f))
        if len(verts) < 2:
            continue
        c0 = verts[0].point().array()[:2]
        c1 = verts[1].point().array()[:2]

        tang = c1 - c0
        tang_len = np.linalg.norm(tang)
        if tang_len < 1e-30:
            continue
        tang /= tang_len

        # Two candidate normals
        n_cand = np.array([-tang[1], tang[0]])

        # Pick the one pointing AWAY from the mesh interior
        # (i.e. into the fluid — away from the solid cap)
        # Use the adjacent cell centroid to determine inward direction
        adj_cells = f.entities(2)
        if len(adj_cells) > 0:
            cell = mesh.cells()[adj_cells[0]]
            cell_verts = mesh.coordinates()[cell]
            cell_centroid = cell_verts.mean(axis=0)
            to_cell = cell_centroid - x_pt
            # n_cand should point INTO the fluid cell (away from solid)
            # i.e. in the same direction as to_cell
            if np.dot(n_cand, to_cell) < 0:
                n_cand = -n_cand  # flip so it points toward fluid cell

        nx, ny = float(n_cand[0]), float(n_cand[1])

        # Evaluate projected stress at midpoint
        try:
            S = sigma_proj(x_pt)   # returns flattened [s00, s01, s10, s11]
        except Exception:
            continue

        s00, s01, s10, s11 = float(S[0]), float(S[1]), float(S[2]), float(S[3])

        # Traction t = sigma · n
        tx = s00 * nx + s01 * ny
        ty = s10 * nx + s11 * ny

        # Pressure at midpoint (for diagnostic separation)
        try:
            p_val = float(p_f(x_pt))
        except Exception:
            p_val = 0.0

        # Decompose into normal (pressure-like) and tangential (WSS) parts
        t_normal_mag = tx * nx + ty * ny          # scalar: t·n
        t_tang_x = tx - t_normal_mag * nx         # tangential component
        t_tang_y = ty - t_normal_mag * ny
        tau_mag  = np.sqrt(t_tang_x**2 + t_tang_y**2)

        records.append({
            'x':       x_pt[0],
            'y':       x_pt[1],
            'p':       p_val,
            'tx':      tx,
            'ty':      ty,
            'tau_x':   t_tang_x,
            'tau_y':   t_tang_y,
            'tau_mag': tau_mag,
            'nx':      nx,
            'ny':      ny,
        })

    if not records:
        raise RuntimeError("No facets with tag 13 found.")

    records.sort(key=lambda r: r['x'])

    def arr(key): return np.array([r[key] for r in records])

    arc_data = {
        'x':       arr('x'),
        'y':       arr('y'),
        'p':       arr('p'),
        'tx':      arr('tx'),
        'ty':      arr('ty'),
        'tau_x':   arr('tau_x'),
        'tau_y':   arr('tau_y'),
        'tau_mag': arr('tau_mag'),
        'nx':      arr('nx'),
        'ny':      arr('ny'),
    }

    return arc_data

# ═════════════════════════════════════════════════════════════════════════════
# Transient Navier-Stokes solver (BDF1 + Picard linearization)
# ═════════════════════════════════════════════════════════════════════════════

def solve_transient_navier_stokes(mesh, facet_tags,
                                   rho, mu,
                                   waveform_fn,
                                   period,
                                   channel_y0, channel_y1,
                                   n_steps_per_cycle=100,
                                   n_warmup_cycles=2,
                                   symmetry_bottom=False,
                                   n_sample_arc=200,
                                   step_dir=None):
    """
    Transient incompressible Navier-Stokes, BDF1 + Picard linearization.

    Time integration: backward Euler (BDF1).
    Linearization: convective term uses velocity from the previous time step,
    making each step a single linear solve (no Newton iterations).

    Marches through (n_warmup_cycles + 1) cardiac cycles; arc traction
    statistics are accumulated on the final cycle only, sampled every
    max(1, n_steps_per_cycle//20) steps to limit stress-projection cost.

    Parameters
    ----------
    waveform_fn       : callable  cycle_fraction in [0,1] -> velocity (m/s)
    period            : float     cardiac cycle period (s)
    n_steps_per_cycle : int       time steps per cycle; dt = period / n_steps
    n_warmup_cycles   : int       cycles run before accumulating statistics

    Returns
    -------
    (u_f, p_f, arc_data)
    u_f, p_f   — velocity/pressure at the final time step (for XDMF viz)
    arc_data   — keys: 'x','y','nx','ny',
                       'p'       (TAWP  — time-averaged wall pressure, Pa),
                       'tau_mag' (TAWSS — time-averaged |WSS|, Pa),
                       'tau_x','tau_y' (time-averaged signed WSS components),
                       'tx','ty'       (time-averaged full traction components),
                       'osi'     (oscillatory shear index; 0=unidirectional,
                                  0.5=fully oscillatory)
    """
    from dolfin import (
        VectorFunctionSpace, FunctionSpace, MixedElement,
        Function, TrialFunction, TestFunctions, split,
        DirichletBC, Constant, Identity,
        grad, div, inner, dot, dx,
        assemble, solve as dolfin_solve,
        Expression, MPI, sym, parameters
    )

    parameters["form_compiler"]["quadrature_degree"] = 4

    H  = float(channel_y1 - channel_y0)
    dt = period / n_steps_per_cycle

    # ── Function spaces (Taylor-Hood P2/P1) ──────────────────────────────
    P2      = VectorFunctionSpace(mesh, "CG", 2)
    P1      = FunctionSpace(mesh, "CG", 1)
    element = MixedElement([P2.ufl_element(), P1.ufl_element()])
    W       = FunctionSpace(mesh, element)

    # w_prev: coefficient holding the previous time-step solution (zero = quiescent start)
    # du:     trial function used to build the bilinear form
    # w:      Function that receives the linear solve result each step
    w_prev   = Function(W)
    w        = Function(W)
    du       = TrialFunction(W)
    (du_u, du_p) = split(du)
    (u_prev, _p) = split(w_prev)
    (v, q)       = TestFunctions(W)

    rho_c = Constant(float(rho))
    mu_c  = Constant(float(mu))
    dt_c  = Constant(dt)

    inlet_expr = Expression(
        ("u_t * 4.0*(x[1]-y0)*(y1-x[1]) / (H*H)", "0.0"),
        u_t=0.0,
        y0=float(channel_y0), y1=float(channel_y1), H=H,
        degree=2
    )

    # ── Boundary conditions ───────────────────────────────────────────────
    bc_inlet = DirichletBC(W.sub(0), inlet_expr,             facet_tags, 10)
    bc_top   = DirichletBC(W.sub(0), Constant((0.0, 0.0)),  facet_tags, 12)
    bc_arc   = DirichletBC(W.sub(0), Constant((0.0, 0.0)),  facet_tags, 13)
    bcs = [bc_inlet, bc_top, bc_arc]
    if not symmetry_bottom:
        bcs.append(DirichletBC(W.sub(0), Constant((0.0, 0.0)), facet_tags, 14))
    # outlet (tag 11): do-nothing natural BC — no explicit entry needed

    # ── Picard weak form (BDF1 + linearized convection) ──────────────────
    # Treating u_prev as a known coefficient makes the system linear in (u,p),
    # so each time step requires exactly one linear solve.
    def epsilon_(u_): return sym(grad(u_))
    def sigma_f_(u_, p_): return 2.0 * mu_c * epsilon_(u_) - p_ * Identity(2)

    # Bilinear form a(du, v): trial functions du_u, du_p appear here.
    # u_prev is a Coefficient (known each step) used in the Picard convection term.
    a_form = (
        rho_c / dt_c * inner(du_u, v)
        + rho_c * inner(dot(u_prev, grad(du_u)), v)
        + inner(sigma_f_(du_u, du_p), epsilon_(v))
        + div(du_u) * q
    ) * dx

    # Linear form L(v): RHS, depends only on u_prev (no trial functions).
    L_form = rho_c / dt_c * inner(u_prev, v) * dx

    # ── Time loop ─────────────────────────────────────────────────────────
    total_steps   = (n_warmup_cycles + 1) * n_steps_per_cycle
    stats_start   = n_warmup_cycles * n_steps_per_cycle
    # Sample arc traction ~20 times per cycle to limit projection cost.
    sample_stride = max(1, n_steps_per_cycle // 20)
    log_stride    = max(1, n_steps_per_cycle // 5)

    acc_tau_mag = None
    acc_tau_x   = None
    acc_tau_y   = None
    acc_tx      = None
    acc_ty      = None
    acc_p       = None
    spatial     = None
    T_acc       = 0.0

    rank = MPI.comm_world.rank

    # ── Optional XDMF time-series output ─────────────────────────────────
    # Written every sample_stride steps on ALL cycles (warmup + stats),
    # so the exported video shows convergence to periodic steady state.
    xf_vel = xf_pres = None
    if step_dir is not None:
        from dolfin import XDMFFile
        xf_vel  = XDMFFile(MPI.comm_world,
                            os.path.join(step_dir, "fluid_velocity.xdmf"))
        xf_pres = XDMFFile(MPI.comm_world,
                            os.path.join(step_dir, "fluid_pressure.xdmf"))
        for xf in (xf_vel, xf_pres):
            xf.parameters["flush_output"]       = True
            xf.parameters["functions_share_mesh"] = True

    if rank == 0:
        print(f"[NS-T] {total_steps} steps  "
              f"({n_warmup_cycles} warmup + 1 stats)  "
              f"dt={dt*1e3:.2f} ms  "
              f"arc sampled every {sample_stride} steps"
              + (f"  XDMF saved every {sample_stride} steps" if step_dir else ""))

    # Capture the PEAK-flow-phase solution for the channel-Δp setpoint probe. The
    # march ends at t/T≈1 (end-diastole, near-zero inlet flow) where the inlet-
    # outlet Δp ≈ 0, so returning the LAST-step pressure made channel_pressure_drop
    # read noise (~0 Pa) and silently broke --setpoint-mode dp on transient runs.
    # Peak phase matches what the quasi-steady solver returns (p_f_peak).
    u_peak_val = -np.inf
    w_peak     = None

    for step in range(total_steps):
        t_norm = (step % n_steps_per_cycle) * dt / period
        t_sim  = step * dt                        # absolute simulation time (s)
        u_t    = float(waveform_fn(t_norm))
        inlet_expr.u_t = u_t

        A = assemble(a_form)
        b = assemble(L_form)
        for bc in bcs:
            bc.apply(A, b)
        dolfin_solve(A, w.vector(), b, "mumps")
        w_prev.assign(w)

        # Track the peak-velocity phase (once past warmup) for the Δp probe.
        if step >= stats_start and u_t > u_peak_val:
            u_peak_val = u_t
            w_peak     = w.copy(deepcopy=True)

        # Write fluid fields to XDMF every sample_stride steps
        if xf_vel is not None and step % sample_stride == 0:
            u_f_s, p_f_s = w.split(deepcopy=True)
            u_f_s.rename("velocity", "velocity")
            p_f_s.rename("pressure", "pressure")
            xf_vel.write(u_f_s,  t_sim)
            xf_pres.write(p_f_s, t_sim)

        # Accumulate arc traction on the stats cycle
        in_stats = step >= stats_start
        on_sample = (step - stats_start) % sample_stride == 0
        if in_stats and on_sample:
            u_f_s, p_f_s = w.split(deepcopy=True)
            arc_t  = _extract_arc_traction(mesh, facet_tags,
                                           u_f_s, p_f_s, mu, n_sample_arc)
            weight = dt * sample_stride

            if spatial is None:
                spatial     = {k: arc_t[k] for k in ('x', 'y', 'nx', 'ny')}
                acc_tau_mag = np.zeros_like(arc_t['tau_mag'])
                acc_tau_x   = np.zeros_like(arc_t['tau_x'])
                acc_tau_y   = np.zeros_like(arc_t['tau_y'])
                acc_tx      = np.zeros_like(arc_t['tx'])
                acc_ty      = np.zeros_like(arc_t['ty'])
                acc_p       = np.zeros_like(arc_t['p'])

            acc_tau_mag += arc_t['tau_mag'] * weight
            acc_tau_x   += arc_t['tau_x']   * weight
            acc_tau_y   += arc_t['tau_y']   * weight
            acc_tx      += arc_t['tx']       * weight
            acc_ty      += arc_t['ty']       * weight
            acc_p       += arc_t['p']        * weight
            T_acc       += weight

        if rank == 0 and step % log_stride == 0:
            print(f"  [NS-T] step {step+1:4d}/{total_steps}  "
                  f"t/T={t_norm:.3f}  u={u_t*100:.2f} cm/s")

    if xf_vel is not None:
        xf_vel.close()
        xf_pres.close()

    # ── Cycle-averaged quantities ─────────────────────────────────────────
    tawss   = acc_tau_mag / T_acc
    tau_x_m = acc_tau_x   / T_acc
    tau_y_m = acc_tau_y   / T_acc
    tx_mean = acc_tx       / T_acc
    ty_mean = acc_ty       / T_acc
    p_mean  = acc_p        / T_acc

    # OSI: ratio of |mean WSS vector| to TAWSS.
    # 0 = purely unidirectional; 0.5 = equal forward+retrograde.
    tau_mean_mag = np.sqrt(tau_x_m**2 + tau_y_m**2)
    osi = 0.5 * (1.0 - tau_mean_mag / np.maximum(tawss, 1e-30))
    np.clip(osi, 0.0, 0.5, out=osi)

    arc_data = {
        **spatial,
        'p':       p_mean,
        'tau_mag': tawss,
        'tau_x':   tau_x_m,
        'tau_y':   tau_y_m,
        'tx':      tx_mean,
        'ty':      ty_mean,
        'osi':     osi,
    }

    if rank == 0:
        print(f"[NS-T] TAWSS : [{tawss.min():.3e}, {tawss.max():.3e}] Pa")
        print(f"[NS-T] OSI   : [{osi.min():.3f},  {osi.max():.3f}]")

    # Return the PEAK-flow-phase fields (not the last, end-diastole step) so the
    # caller's channel_pressure_drop measures a representative hemodynamic load and
    # --setpoint-mode dp scales correctly across stages. Falls back to the final
    # step only if no peak was captured (e.g. a single-step stats window).
    w_out = w_peak if w_peak is not None else w
    if rank == 0 and w_peak is not None:
        print(f"[NS-T] Δp probe uses peak phase u={u_peak_val*100:.2f} cm/s")
    u_f, p_f = w_out.split(deepcopy=True)
    u_f.rename("velocity", "velocity")
    p_f.rename("pressure", "pressure")

    return u_f, p_f, arc_data


# ═════════════════════════════════════════════════════════════════════════════
# Save fluid outputs to XDMF
# ═════════════════════════════════════════════════════════════════════════════

def write_fluid_xdmf(xdmf_path, u_f, p_f, t_out):
    from dolfin import XDMFFile, MPI
    xdmf = XDMFFile(MPI.comm_world, xdmf_path)
    xdmf.parameters["flush_output"] = True
    xdmf.parameters["functions_share_mesh"] = True
    xdmf.write(u_f, float(t_out))
    xdmf.write(p_f, float(t_out))
    xdmf.close()