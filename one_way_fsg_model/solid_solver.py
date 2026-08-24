#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
solid_solver.py
---------------
Exponential isotropic hyperelastic solid with isotropic G&R (growth & resorption).
Refactored from valve_growth.py to accept CFD-derived traction expressions
instead of hard-coded pressure/WSS profiles.

Key change from original:
  - `traction_expr`  : a FEniCS vector Expression or Function (2-component)
                       representing the full fluid traction  t = σ_f · n
                       on the arc.  Built by traction_mapper.build_traction_expressions().
  - When traction_expr is None, falls back to the original hard-coded profiles
    (for standalone testing without CFD).

Usage
-----
    from solid_solver import SolidSolver

    solver = SolidSolver(mesh, facet_tags, config)
    solver.initialize()

    for fsi_step in range(N_FSI):
        # ... run CFD, build traction_expr ...
        solver.set_traction(traction_expr)
        solver.run_ramp(n_ramp=N_RAMP, p_final=P_FINAL)   # or just run_hold()
        u = solver.u
        g = solver.g
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Default config (mirrors original valve_growth.py constants)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = dict(
    # Material
    C_exp      = 200.0,
    alpha_exp  = 0.30,
    D_exp      = 6.0e-3,

    # Compression-homeostasis growth law (Buskohl-style)
    # Growth if sigma_comp < sigma_comp_home, resorption if sigma_comp > sigma_comp_home
    sigma_comp_home_Pa = 38.0,   # Pa — between downstream (~24 Pa) and upstream (~39 Pa) pressure
    sigma_comp_dead_Pa =  3.0,   # Pa — dead band half-width (no response within ±this)
    k_g                = 0.007,  # rate constant (~143 min half-response at 3 min/hold-step)
    dt_g               = 0.5,    # growth timestep; 10 steps/iter × 0.5 = 30 min/iter
    g_min              = 0.3,    # sqrt(0.3)≈0.55× max linear compaction per generation
    g_max              = 1.3,    # sqrt(1.3)≈1.14× max linear growth per generation
    dg_max_step        = 0.05,   # coarse cap on |Δg| per hold step (see _growth_dg);
                                 # the fine control is run_hold's adaptive sub-stepping.
    growth_substep_min_frac = 1.0/32,  # adaptive hold: smallest growth-increment
                                 # fraction tried before giving up (halves from 1.0).
                                 # High-load HH27/30 steps typically settle at ~1/4.

    # Atrial-face compaction (per-cell setpoint reduction).
    # Spatially LOWER the compression-homeostasis setpoint on the atrial (upstream,
    # x < x_center) flank so that face reads as over-compressed (sigma_comp > local
    # setpoint → g<1 → compaction) while the ventricular flank keeps the full
    # setpoint and grows — manufacturing the compact-atrial / grow-ventricular split
    # of a shark fin. The reduction is scaled live by a 0..1 gate (set_atrial_gate)
    # so the runner can engage it only once the fin is tall enough. Off by default.
    atrial_setpoint_frac = 1.0,  # atrial setpoint = base × this  (1.0 = off)
    atrial_xfrac         = 0.6,  # atrial band width as a fraction of the half-width,
                                 # measured inward from the atrial (x<center) edge
    # ── Anisotropic (tensorial) growth ───────────────────────────────────────
    # When True, replace the isotropic Fg = sqrt(g)*I with a growth TENSOR
    #   Fg = lam_g*(a⊗a) + lam_c*(n⊗n)     (a,n orthonormal; n ⊥ a)
    # grow along a (lam_g≥1), thin/compact across it (lam_c≤1). The direction a is
    # aligned to the stress eigenbasis (a = least-compressed / max-principal axis;
    # n = most-compressed axis), regularized for stability. det Fg = lam_g*lam_c.
    # anisotropic=False → byte-identical isotropic behaviour (scalar g path).
    anisotropic       = False,
    aniso_lam_g_max   = 2.0,     # per-generation grow-along-a ceiling
    aniso_lam_c_min   = 0.3,     # per-generation thin-across ceiling (floor on lam_c)
    aniso_dir_relax   = 0.30,    # per-step rotation limiter: a ← slerp(a_prev, a_new, this)
    aniso_smooth_passes = 2,     # SPATIAL smoothing passes on the direction (DG0→CG1→DG0
                                 # neighbour-averaging) so adjacent boundary cells grow
                                 # along similar axes → no kinked/bumpy arc. 0 = off.
    # ABSOLUTE atrial target (Pa). When > 0 this OVERRIDES the fractional form: the
    # atrial band target is pulled toward this fixed Pa value instead of base×frac.
    # Decouples compaction from the stage setpoint so it does NOT run away when the
    # setpoint jumps at a high stage (e.g. base 53→468 at HH23 would push a 0.55×
    # band target 29→257, reversing compaction; an absolute ~26 Pa target stays
    # reachable and the atrial front keeps compacting through HH23). 0 = off (use frac).
    atrial_setpoint_abs  = 0.0,

    # Tension-based growth law (disabled — kept for future use)
    # To re-enable: set use_tension_stimulus = True
    use_tension_stimulus = False,
    sigma_home_Pa    = 9999.0,
    sigma_dead_Pa    = 9999.0,
    k_r              = 0.005,

    # Newton
    newton_max_iters = 200,
    newton_rtol      = 1e-9,
    newton_atol      = 1e-10,
    newton_relax     = 0.20,

    # Projection
    proj_deg  = 1,
    quad_deg  = 4,

    # Ramp
    ramp_mode = "linear",

    # Ventricular spring (optional Robin / Winkler elastic foundation)
    # A distributed surface spring on the ventricular (downstream / right) side
    # that resists +x displacement, limiting expansion DOWN the canal so growth
    # is redirected UP into the canal (leaflet-like). Added to the strain energy
    # as 0.5*k*u_x^2 over the chosen boundary, so it enters the Newton solve
    # smoothly with no point-load singularity (unlike the pinned roller anchor).
    # vent_spring_k = 0.0 disables it → solver behaves exactly as before.
    vent_spring_k       = 0.0,     # Pa/m — surface spring modulus; 0 = off
    vent_spring_frac    = 0.5,     # right fraction of the width that gets the spring
    vent_spring_surface = "arc",   # "arc" (tag 2, downstream flank) or "base" (tag 1)
    vent_spring_dir     = "x",     # "x" = resist downstream only; "all" = isotropic

    # Base x-spring (only used by base_bc="yroller_spring"): distributed Winkler
    # spring along the base that carries the horizontal load and softly restrains
    # x, so the base slides / the atrial corner relaxes (no vertical wall) while a
    # single centre u_x=0 pin sets the absolute reference. Stiffer → less drift,
    # better conditioning; softer → corner relaxes more.
    base_spring_k       = 5.0e5,   # Pa/m
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _pos_part(x):
    import ufl
    return ufl.max_value(x, 0.0)


def _ramp_value(step, n_steps, p_final, mode="linear"):
    s = float(step) / float(n_steps)
    if mode == "quadratic":
        s = s * s
    return p_final * s


# ─────────────────────────────────────────────────────────────────────────────
# SolidSolver class
# ─────────────────────────────────────────────────────────────────────────────

class SolidSolver:

    def __init__(self, mesh, facet_tags, config=None):
        self.mesh       = mesh
        self.facet_tags = facet_tags
        self.cfg        = dict(DEFAULT_CONFIG)
        if config:
            self.cfg.update(config)

        self.u          = None   # displacement Function
        self.g          = None   # growth factor DG0 Function
        self._traction  = None   # current traction Expression (set externally)
        self._setup_done = False
        self._w_atrial   = None  # per-cell atrial-face weight (set in initialize)
        self._atrial_gate = 0.0  # live 0..1 gate on the atrial setpoint reduction

    # ── public API ────────────────────────────────────────────────────────

    def initialize(self):
        self._build_spaces()
        self._build_forms()
        self._precompute_height_weights()
        self._precompute_atrial_weight()
        self._setup_done = True

    def set_traction(self, p_expr, wss_expr):
        if not self._setup_done:
            raise RuntimeError("Call initialize() first.")
        self._traction = (p_expr, wss_expr)
        self._rebuild_residual()

    def set_growth_setpoint(self, sigma_comp_home_Pa):
        """
        Update the compression-homeostasis setpoint (Pa) used by the growth law.

        For the HH developmental sweep the characteristic loading climbs ~10x from
        the early tubular stages to the mature ones, so a FIXED setpoint makes the
        high-pressure late stages read as massively over-compressed and resorb to
        collapse. Raising the setpoint per stage keeps the tissue near homeostasis
        (still remodeling/growing) as pressures rise. The growth update reads this
        value live each step, so no form rebuild is needed.
        """
        from dolfin import Constant
        self.cfg["sigma_comp_home_Pa"] = float(sigma_comp_home_Pa)
        # Keep the stored Constant consistent (not used in the residual, but avoids
        # a stale value if any future form references it).
        self._sigma_comp_home = Constant(float(sigma_comp_home_Pa))

    def set_atrial_gate(self, gate):
        """
        Set the live 0..1 gate scaling the atrial-face setpoint reduction.

        0 → no reduction (uniform setpoint everywhere); 1 → full reduction so the
        atrial band's setpoint reaches base × atrial_setpoint_frac. The runner drives
        this from the fin-height fraction so compaction only engages once the cushion
        is tall enough (a fin-size gate). Read live in _growth_dg — no form rebuild.
        """
        self._atrial_gate = float(np.clip(gate, 0.0, 1.0))

    def run_ramp(self, n_ramp, p_magnitude, xdmf_file=None, step_offset=0):
        """
        Ramp the traction magnitude linearly from 0 → 1 over n_ramp steps.

        For FSI use: traction is already the full CFD traction at current geometry.
        The ramp here is a numerical convenience (helps Newton convergence on the
        first FSI iteration).  On subsequent FSI iterations you may want n_ramp=1.

        p_magnitude : unused in CFD mode (traction already carries magnitude).
                      Kept for backward-compat with standalone mode.
        xdmf_file   : open dolfin.XDMFFile or None
        step_offset : global step counter offset for XDMF time axis
        """
        if not self._setup_done:
            raise RuntimeError("Call initialize() first.")

        for step in range(1, int(n_ramp) + 1):
            scale = _ramp_value(step, n_ramp, 1.0, mode=self.cfg["ramp_mode"])
            self._scale.assign(float(scale))
            self._do_step(step + step_offset, xdmf_file, grow=False)

    def run_hold(self, n_hold, xdmf_file=None, step_offset=0):
        """
        Run n_hold growth steps at full traction (scale=1.0) with ADAPTIVE growth
        sub-stepping.

        Each step grows from the current converged state, then re-solves
        equilibrium. If that solve fails — a far-from-homeostasis, high-load stage
        (HH27/30) can perturb the stiff near-incompressible solid past element
        inversion — the growth increment is halved and the solve retried, down to
        growth_substep_min_frac. Easy low-load stages never trigger this (frac
        stays 1.0, so behaviour is identical to before); only the mature
        high-pressure stages auto-refine, keeping HH27/30 stable without slowing
        the fast early growth. Growth the solid can't yet absorb isn't lost — it
        re-accrues next step from the still-elevated stress.
        """
        if not self._setup_done:
            raise RuntimeError("Call initialize() first.")
        from dolfin import MPI

        self._scale.assign(1.0)
        frac_min = float(self.cfg.get("growth_substep_min_frac", 1.0 / 32))
        rank = MPI.comm_world.rank

        aniso = self.cfg.get("anisotropic", False)
        for step in range(1, int(n_hold) + 1):
            sg = step + step_offset

            # Growth increment from the CURRENT converged state (computed once).
            if aniso:
                # Direction (stress-aligned, rotation-limited) set once per step;
                # only the lam_g/lam_c MAGNITUDES are adaptively sub-stepped.
                ax_new, ay_new, dlg, dlc, sigma_comp_arr, stimulus = self._growth_aniso()
                self._set_direction(ax_new, ay_new)
                lg_base = self.lam_g.vector().get_local().copy()
                lc_base = self.lam_c.vector().get_local().copy()
            else:
                dg_masked, sigma_comp_arr, stimulus = self._growth_dg()
                g_base = self.g.vector().get_local().copy()
            u_base = self.u.vector().get_local().copy()

            frac  = 1.0
            tries = 0
            while True:
                tries += 1
                if aniso:
                    self._apply_growth_aniso(lg_base, lc_base, dlg, dlc, frac)
                else:
                    self._apply_growth(g_base, frac * dg_masked)
                try:
                    self._solve_u()
                    break
                except RuntimeError:
                    # Reset the displacement guess and shrink the growth step.
                    self.u.vector().set_local(u_base)
                    self.u.vector().apply("insert")
                    frac *= 0.5
                    if frac < frac_min:
                        if aniso:
                            self._apply_growth_aniso(lg_base, lc_base, dlg, dlc, 0.0)
                        else:
                            self._apply_growth(g_base, 0.0 * dg_masked)  # restore
                        raise RuntimeError(
                            f"Hold step {sg}: growth solve failed even at "
                            f"frac={frac * 2:.4g} (< growth_substep_min_frac="
                            f"{frac_min:.4g})")

            if rank == 0:
                sub = (f"  [adaptive: frac={frac:.4g}, {tries} tries]"
                       if frac < 1.0 else "")
                if aniso:
                    gstr = (f"lam_g=[{self.lam_g.vector().min():.3f}, "
                            f"{self.lam_g.vector().max():.3f}]  "
                            f"lam_c=[{self.lam_c.vector().min():.3f}, "
                            f"{self.lam_c.vector().max():.3f}]")
                else:
                    gstr = (f"g=[{self.g.vector().min():.4f}, "
                            f"{self.g.vector().max():.4f}]")
                print(f"\n  [Solid] hold step {sg}  scale=1.0  {gstr}  "
                      f"setpoint={float(self.cfg['sigma_comp_home_Pa']):.1f} Pa  "
                      f"sigma_comp=[{sigma_comp_arr.min():.1f}, {sigma_comp_arr.max():.1f}] Pa  "
                      f"stimulus=[{stimulus.min():.2f}, {stimulus.max():.2f}] Pa{sub}")

            if xdmf_file is not None:
                t_out = float(sg)
                xdmf_file.write(self.u, t_out)
                self._write_fields(xdmf_file, t_out)

    def _precompute_height_weights(self):
        """
        Compute a per-cell growth weight based on normalised height y_norm ∈ [0,1].

        Suppresses growth only in a narrow zone near the clamped base (y_norm ≈ 0)
        where displacement is constrained to zero.  The bulk of the cushion gets
        w = 1.0 so the stress field drives growth/resorption without spatial bias.

        Profile: Hermite smoothstep from W_FLOOR at y_norm=0 up to 1.0,
        with the taper completing at y_norm = BASE_FRAC + TAPER_FRAC.

        Config keys (all optional, defaults below):
          mask_base_frac  : fraction of height held at W_FLOOR  (default 0.05)
          mask_taper_frac : fraction of height over which w ramps 0→1 (default 0.15)
          mask_w_floor    : minimum weight inside the base zone   (default 0.05)
        """
        coords = self.mesh.coordinates()
        cell_mp = np.array([coords[cell].mean(axis=0)
                            for cell in self.mesh.cells()])

        y_vals = cell_mp[:, 1]
        y_min  = float(y_vals.min())
        y_max  = float(y_vals.max())
        y_norm = (y_vals - y_min) / (y_max - y_min + 1e-30)

        base_frac  = float(self.cfg.get("mask_base_frac",  0.05))
        taper_frac = float(self.cfg.get("mask_taper_frac", 0.15))
        w_floor    = float(self.cfg.get("mask_w_floor",    0.05))

        y_start = base_frac
        y_end   = base_frac + taper_frac

        # Hermite smoothstep: 0 → 1 over [y_start, y_end]
        t = np.clip((y_norm - y_start) / max(y_end - y_start, 1e-12), 0.0, 1.0)
        w_smooth = t * t * (3.0 - 2.0 * t)
        w = w_floor + (1.0 - w_floor) * w_smooth

        # ── Lateral corner knock-out (graduated, geometry-aware) ─────────────
        # The recurring gmsh degeneracy is the base corner leaning past vertical
        # into an overhang: tissue at high |x| (lateral edge) AND low y (near
        # base) bulges out and steepens the wall. Suppress growth in exactly that
        # low-y / high-|x| corner zone so the wall physically cannot grow past
        # vertical. Multiplicative on top of the height mask; graded (smoothstep)
        # so there is no sharp shoulder. Off by default (corner_w_floor=1.0).
        corner_wfloor = float(self.cfg.get("mask_corner_w_floor", 1.0))
        if corner_wfloor < 1.0:
            corner_xfrac = float(self.cfg.get("mask_corner_xfrac", 0.30))  # zone width from each edge
            corner_yfrac = float(self.cfg.get("mask_corner_yfrac", 0.30))  # "low" height band
            x_vals   = cell_mp[:, 0]
            x_center = 0.5 * (float(x_vals.min()) + float(x_vals.max()))
            half_w   = 0.5 * (float(x_vals.max()) - float(x_vals.min())) + 1e-30
            x_norm   = np.abs(x_vals - x_center) / half_w          # 0 centre → 1 edge
            # "how lateral": 0 until x_norm enters the corner band, →1 at the edge
            tl = np.clip((x_norm - (1.0 - corner_xfrac)) / max(corner_xfrac, 1e-12), 0.0, 1.0)
            lateral = tl * tl * (3.0 - 2.0 * tl)
            # "how low": 1 at base, →0 above the low band
            tlo = np.clip(y_norm / max(corner_yfrac, 1e-12), 0.0, 1.0)
            lowy = 1.0 - tlo * tlo * (3.0 - 2.0 * tlo)
            w_corner = 1.0 - (1.0 - corner_wfloor) * lateral * lowy
            w = w * w_corner

        self._w_height = w
        self._cell_mp_y = y_norm

        from dolfin import MPI
        if MPI.comm_world.rank == 0:
            extra = "" if corner_wfloor >= 1.0 else \
                f"  corner_wfloor={corner_wfloor} (xfrac={self.cfg.get('mask_corner_xfrac',0.30)}, yfrac={self.cfg.get('mask_corner_yfrac',0.30)})"
            print(f"  [Growth mask] base_frac={base_frac}  taper_frac={taper_frac}  "
                  f"w_floor={w_floor}  "
                  f"w range: [{w.min():.3f}, {w.max():.3f}]  "
                  f"w at midline: {w[np.argmin(np.abs(y_norm - 0.5))]:.3f}{extra}")

    def _precompute_atrial_weight(self):
        """
        Per-cell weight (0..1) selecting the ATRIAL (upstream, x < x_center) flank
        for setpoint reduction. 1 at the atrial edge, smoothstep → 0 over
        `atrial_xfrac` of the half-width, and 0 everywhere on the ventricular side.

        Built from REFERENCE cell centroids in the same cell ordering as the DG0
        growth vector (mirrors _precompute_height_weights), so it multiplies the
        projected stress array elementwise in _growth_dg. Rebuilt on every remesh
        (initialize() is called on each fresh generation mesh), so it tracks the arc
        as the fin grows and leans.
        """
        coords  = self.mesh.coordinates()
        cell_mp = np.array([coords[cell].mean(axis=0)
                            for cell in self.mesh.cells()])
        x_vals   = cell_mp[:, 0]
        x_center = 0.5 * (float(x_vals.min()) + float(x_vals.max()))
        half_w   = 0.5 * (float(x_vals.max()) - float(x_vals.min())) + 1e-30

        # Signed normalised x: -1 at the atrial (upstream) edge, +1 ventricular.
        x_sgn = (x_vals - x_center) / half_w
        xfrac = float(self.cfg.get("atrial_xfrac", 0.6))

        # Distance inward from the atrial edge, normalised by the band width, so the
        # band spans the atrial-most `xfrac` of the half-width. Smoothstep so the
        # weight = 1 at the edge and grades to 0 at the band's inner boundary (no
        # sharp shoulder). Everything ventral of the band is 0.
        t = np.clip((x_sgn + 1.0) / max(xfrac, 1e-12), 0.0, 1.0)
        self._w_atrial = 1.0 - t * t * (3.0 - 2.0 * t)

        from dolfin import MPI
        if MPI.comm_world.rank == 0:
            n_at = int((self._w_atrial > 0.01).sum())
            print(f"  [Atrial weight] xfrac={xfrac}  "
                  f"w range: [{self._w_atrial.min():.3f}, {self._w_atrial.max():.3f}]  "
                  f"cells>0.01: {n_at}/{len(self._w_atrial)}")

    # ── internals ─────────────────────────────────────────────────────────

    def _build_spaces(self):
        from dolfin import (
            VectorFunctionSpace, FunctionSpace, TensorFunctionSpace,
            Function, Constant
        )

        mesh = self.mesh
        V  = VectorFunctionSpace(mesh, "CG", 2)
        Vg = FunctionSpace(mesh, "DG", 0)
        T  = TensorFunctionSpace(mesh, "CG", int(self.cfg["proj_deg"]))
        S  = FunctionSpace(mesh, "CG", 1)
        Vd0= FunctionSpace(mesh, "DG", 0)

        self.V   = V
        self.Vg  = Vg
        self.T   = T
        self.S   = S
        self.Vd0 = Vd0

        self.u  = Function(V,  name="u")
        self.g  = Function(Vg, name="growth_factor_g")
        self.g.assign(Constant(1.0))

        # Anisotropic-growth fields (only used when cfg["anisotropic"]): per-cell
        # unit direction a (DG0 vector) + the two stretch factors lam_g, lam_c (DG0
        # scalar). Initialise to a=ŷ, lam_g=lam_c=1  →  Fg = ŷ⊗ŷ + x̂⊗x̂ = I, i.e.
        # identical to the isotropic g=1 state until the growth update moves them.
        if self.cfg.get("anisotropic", False):
            # Direction stored as two DG0 SCALARS (a_x, a_y) sharing Vg's cell
            # ordering, so they assign directly from the principal-stress arrays.
            self.a_x   = Function(Vg, name="growth_dir_ax")
            self.a_y   = Function(Vg, name="growth_dir_ay")
            self.lam_g = Function(Vg, name="lam_g")
            self.lam_c = Function(Vg, name="lam_c")
            self.a_x.assign(Constant(0.0))        # a = ŷ (vertical) to start
            self.a_y.assign(Constant(1.0))
            self.lam_g.assign(Constant(1.0))
            self.lam_c.assign(Constant(1.0))

        self._scale = Constant(0.0)   # traction scale factor for ramping

    def _build_forms(self):
        from dolfin import (
            TestFunction, TrialFunction, Constant, Identity,
            grad, det, tr, inner, dx, Measure, FacetNormal,
            DirichletBC, parameters, CompiledSubDomain, MPI
        )
        import ufl

        parameters["form_compiler"]["quadrature_degree"] = int(self.cfg["quad_deg"])

        mesh       = self.mesh
        facet_tags = self.facet_tags
        u          = self.u
        g          = self.g

        v  = TestFunction(self.V)
        du = TrialFunction(self.V)

        self._v  = v
        self._du = du

        self._ds = Measure("ds", domain=mesh, subdomain_data=facet_tags)

        # Material constants
        Cc = Constant(float(self.cfg["C_exp"]))
        aa = Constant(float(self.cfg["alpha_exp"]))
        Dd = Constant(float(self.cfg["D_exp"]))

        # Growth constants
        self._sigma_home = Constant(float(self.cfg["sigma_home_Pa"]))
        self._sigma_dead = Constant(float(self.cfg["sigma_dead_Pa"]))
        self._kG         = Constant(float(self.cfg["k_g"]))
        self._dtG        = Constant(float(self.cfg["dt_g"]))
        self._kR         = Constant(float(self.cfg["k_r"]))
        self._sigma_comp_home = Constant(float(self.cfg["sigma_comp_home_Pa"]))
        self._sigma_comp_dead = Constant(float(self.cfg["sigma_comp_dead_Pa"]))

        # ── Kinematics with multiplicative growth ────────────────────────
        I2 = Identity(2)
        F  = ufl.variable(I2 + grad(u))
        J  = det(F)

        if self.cfg.get("anisotropic", False):
            # Tensorial growth: grow along a (unit), thin across n = a^perp.
            #   Fg = lam_g*(a⊗a) + lam_c*(n⊗n)
            # a,lam_g,lam_c are DG0 Functions updated each generation; the residual
            # and all stress measures derive from Fe = F*Fg_inv, so nothing below
            # changes. (a=ŷ, lam=1 ⇒ Fg = I, matching the isotropic g=1 baseline.)
            a  = ufl.as_vector([self.a_x, self.a_y])
            nn = ufl.as_vector([-self.a_y, self.a_x])   # a^perp
            Fg = ufl.variable(self.lam_g * ufl.outer(a, a)
                              + self.lam_c * ufl.outer(nn, nn))
        else:
            g12 = ufl.variable(g ** 0.5)
            Fg  = ufl.variable(g12 * I2)
        Fg_inv= ufl.inv(Fg)
        Fe    = ufl.variable(F * Fg_inv)
        Je    = det(Fe)
        be    = Fe * Fe.T
        I1e   = tr(be)

        # ── Energy ──────────────────────────────────────────────────────
        psi_iso = (Cc / 2.0) * (ufl.exp(aa * (I1e - 2.0)) - 1.0)
        psi_vol = (1.0 / Dd) * (Je - 1.0) ** 2
        psi     = psi_iso + psi_vol

        # ── Stress measures ─────────────────────────────────────────────
        P_expr     = ufl.diff(psi, F)
        sigma_expr = (1.0 / J) * P_expr * F.T

        s_dev        = sigma_expr - 0.5 * tr(sigma_expr) * I2
        sigma_vm_expr= ufl.sqrt(1.5 * inner(s_dev, s_dev))

        tr_sig  = tr(sigma_expr)
        det_sig = ufl.det(sigma_expr)
        disc    = ufl.max_value(tr_sig*tr_sig - 4.0*det_sig, 0.0)

        sigma_max_expr  = 0.5 * (tr_sig + ufl.sqrt(disc))
        sigma_min_expr  = 0.5 * (tr_sig - ufl.sqrt(disc))
        sigma_ten_expr  = _pos_part(sigma_max_expr)
        sigma_comp_expr = _pos_part(-sigma_min_expr)

        # Jacobians
        J_total_expr = J
        Je_expr      = Je
        Jg_expr      = det(Fg)

        # Green-Lagrange strain  E = 0.5*(F^T F - I)
        E_GL_expr = 0.5 * (F.T * F - I2)

        # Store UFL expressions for use in residual + growth update
        self._F    = F
        self._J    = J
        self._psi  = psi
        self._sigma_cauchy_expr = sigma_expr
        self._E_GL_expr         = E_GL_expr
        self._sigma_vm_expr   = sigma_vm_expr
        self._sigma_max_expr  = sigma_max_expr
        self._sigma_min_expr  = sigma_min_expr
        self._sigma_ten_expr  = sigma_ten_expr
        self._sigma_comp_expr = sigma_comp_expr
        self._J_total_expr    = J_total_expr
        self._Je_expr         = Je_expr
        self._Jg_expr         = Jg_expr

        # Base boundary condition (facet tag 1 = base diameter).
        #   "clamp"  : fully fixed base  u=(0,0)  — original behaviour (pins width).
        #   "roller" : frictionless base  u_y=0, free in x, with u_x=0 pinned only
        #              at the single centre node to remove rigid x-translation. The
        #              footprint can WIDEN as the cushion grows (FIX #2), keeping
        #              sidewalls gentle and moving toward shark-fin morphology.
        self._Pi_base_spring = None   # set below for the yroller_spring mode
        base_bc = str(self.cfg.get("base_bc", "clamp")).lower()
        if base_bc == "roller":
            coords   = self.mesh.coordinates()
            hmin     = self.mesh.hmin()
            y_base   = coords[:, 1].min()
            x_centre = 0.5 * (coords[:, 0].min() + coords[:, 0].max())
            # Pin u_x at the REAL base node nearest the centre (use its exact
            # coords so the pointwise BC matches exactly one vertex — a guessed
            # x=0 might sit between nodes and pin nothing → singular x-system).
            base_pts = coords[np.abs(coords[:, 1] - y_base) < 0.25 * hmin]
            x_anchor, y_anchor = base_pts[np.argmin(np.abs(base_pts[:, 0] - x_centre))]
            tol = 0.25 * hmin
            bc_y = DirichletBC(self.V.sub(1), Constant(0.0), facet_tags, 1)
            centre = CompiledSubDomain(
                "near(x[0], xa, tol) && near(x[1], yb, tol)",
                xa=float(x_anchor), yb=float(y_anchor), tol=float(tol))
            bc_x = DirichletBC(self.V.sub(0), Constant(0.0), centre,
                               method="pointwise")
            self._bc = [bc_y, bc_x]
            if MPI.comm_world.rank == 0:
                print(f"  [Solid BC] ROLLER base: u_y=0 on tag 1, u_x=0 pinned at "
                      f"centre ({x_anchor:.4g}, {y_base:.4g}) — base free to widen")
        elif base_bc in ("asym_roller", "atrial_pin"):
            # Atrial (upstream) base anchored, ventricular (downstream) base on
            # rollers. u_y=0 on the WHOLE base; u_x=0 ONLY over the atrial fraction
            # (a DISTRIBUTED pin → no point-load singularity, unlike "roller"). The
            # ventricular base is free to slide downstream so the tip extends into
            # the canal. Flow is +x, so atrial = small x (left). This removes the
            # rigid null space without the single anchor node that crashed the
            # symmetric roller run.
            coords = self.mesh.coordinates()
            hmin   = self.mesh.hmin()
            y_base = coords[:, 1].min()
            x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
            pin_frac = float(self.cfg.get("base_pin_frac", 0.35))
            x_split  = x_min + pin_frac * (x_max - x_min)
            tol = 0.25 * hmin
            bc_y = DirichletBC(self.V.sub(1), Constant(0.0), facet_tags, 1)
            atrial = CompiledSubDomain(
                "on_boundary && x[1] < yb + tol && x[0] <= xs",
                yb=float(y_base), tol=float(tol), xs=float(x_split))
            bc_x = DirichletBC(self.V.sub(0), Constant(0.0), atrial)
            self._bc = [bc_y, bc_x]
            if MPI.comm_world.rank == 0:
                print(f"  [Solid BC] ASYMMETRIC: u_y=0 whole base; atrial u_x=0 for "
                      f"x<= {x_split:.4g} ({pin_frac:.0%} of base); ventricular base "
                      f"free to slide downstream")
        elif base_bc in ("yroller_spring", "yroller_xspring", "spring_base"):
            # u_y=0 on the WHOLE base (no vertical motion; this alone removes
            # rotation + y-translation). x is held by THREE complementary pieces,
            # none of which is the load-bearing point anchor that caused earlier
            # failures:
            #   (1) a DISTRIBUTED x-spring along the whole base (self._Pi_base_spring,
            #       0.5*k_base*u_x^2 over tag 1) — carries the net horizontal flow
            #       load and gives a soft restoring restraint, so the atrial corner
            #       can SLIDE/relax instead of steepening into the vertical wall the
            #       asym pin forced (which folded the arc and crashed the mesher);
            #   (2) a SINGLE u_x=0 pin at the centre base node — sets the absolute
            #       x-reference so the footprint can't drift across remesh
            #       generations. Because the spring carries the load, this pin's
            #       reaction is ~0 → none of the 721 Pa stress singularity that
            #       crashed the pointwise "roller", and being at the centre (not the
            #       atrial corner) it grows no wall.
            coords   = self.mesh.coordinates()
            hmin     = self.mesh.hmin()
            y_base   = coords[:, 1].min()
            x_centre = 0.5 * (coords[:, 0].min() + coords[:, 0].max())
            base_pts = coords[np.abs(coords[:, 1] - y_base) < 0.25 * hmin]
            x_anchor, y_anchor = base_pts[np.argmin(np.abs(base_pts[:, 0] - x_centre))]
            tol = 0.25 * hmin
            bc_y = DirichletBC(self.V.sub(1), Constant(0.0), facet_tags, 1)
            centre = CompiledSubDomain(
                "near(x[0], xa, tol) && near(x[1], yb, tol)",
                xa=float(x_anchor), yb=float(y_anchor), tol=float(tol))
            bc_x = DirichletBC(self.V.sub(0), Constant(0.0), centre,
                               method="pointwise")
            self._bc = [bc_y, bc_x]
            k_base = float(self.cfg.get("base_spring_k", 5.0e5))
            self._Pi_base_spring = 0.5 * Constant(k_base) * u[0] ** 2 * self._ds(1)
            if MPI.comm_world.rank == 0:
                print(f"  [Solid BC] Y-ROLLER + X-SPRING base: u_y=0 on tag 1; "
                      f"distributed x-spring k_base={k_base:.3e} Pa/m carries the load; "
                      f"u_x=0 reference pin at centre ({x_anchor:.4g}, {y_base:.4g}) — "
                      f"base slides, corner relaxes, no drift")
        else:
            self._bc = [DirichletBC(self.V, Constant((0.0, 0.0)), facet_tags, 1)]
            if MPI.comm_world.rank == 0:
                print("  [Solid BC] CLAMPED base: u=(0,0) on tag 1 (fixed width)")

        # Ventricular spring (optional Robin / Winkler elastic foundation).
        # Built once here; added into the residual by _rebuild_residual().
        self._setup_vent_spring()

        # Build initial residual (traction = 0 until set_traction is called)
        self._rebuild_residual()

    def _setup_vent_spring(self):
        """
        Build the optional ventricular-spring energy form  0.5*k*u_x^2  over the
        downstream (right) fraction of a boundary, stored as self._Pi_spring.

        Represents surrounding tissue resisting expansion DOWN the canal: it
        penalizes +x material-point displacement on the ventricular side, so
        growth (near-incompressible) is redirected UPWARD into the canal — the
        leaflet-elongation morphology we want.

        Linear symmetric spring; because the ventricular side moves predominantly
        downstream (+x), it acts mainly to oppose that motion. Implemented as a
        distributed energy term (NOT a pinned DOF), so there is no point-load
        stress singularity like the symmetric roller anchor that crashed nfsg60.

        Reference (material) coordinates + reference ds are used, so the spring
        is re-anchored to the current ventricular extent each remesh generation
        (a fresh SolidSolver is built on the relaxed grown mesh).

        Config keys (see DEFAULT_CONFIG): vent_spring_k, vent_spring_frac,
        vent_spring_surface, vent_spring_dir.
        """
        from dolfin import Constant, SpatialCoordinate, conditional, ge, MPI
        import ufl

        k_spring = float(self.cfg.get("vent_spring_k", 0.0))
        if k_spring <= 0.0:
            self._Pi_spring = None
            return

        surface = str(self.cfg.get("vent_spring_surface", "arc")).lower()
        tag     = 2 if surface == "arc" else 1
        frac    = float(self.cfg.get("vent_spring_frac", 0.5))
        frac    = min(max(frac, 0.0), 1.0)
        direction = str(self.cfg.get("vent_spring_dir", "x")).lower()

        coords = self.mesh.coordinates()
        x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
        x_split = x_max - frac * (x_max - x_min)   # ventricular side: x >= x_split

        x = SpatialCoordinate(self.mesh)
        region = conditional(ge(x[0], Constant(x_split)), 1.0, 0.0)

        k = Constant(k_spring)
        u = self.u
        if direction == "all":
            spring_density = 0.5 * k * ufl.inner(u, u)
        else:   # "x" — resist downstream expansion only
            spring_density = 0.5 * k * u[0] ** 2

        self._Pi_spring = region * spring_density * self._ds(tag)

        if MPI.comm_world.rank == 0:
            print(f"  [Solid spring] VENTRICULAR spring: k={k_spring:.3e} Pa/m on "
                  f"{'arc(tag2)' if tag == 2 else 'base(tag1)'} for x>={x_split:.4g} "
                  f"(right {frac:.0%} of width), dir={direction}")

    def _rebuild_residual(self):
        from dolfin import inner, derivative, FacetNormal
        import ufl

        ds = self._ds
        u = self.u
        v = self._v
        du = self._du
        n = FacetNormal(self.mesh)  # outward from solid, same as original

        Pi_int = self._psi * ufl.dx

        if self._traction is not None:
            p_expr, wss_expr = self._traction  # now a tuple

            # Exactly mirrors original valve_growth.py:
            #   traction = (-p) * n  +  wss_tangential
            traction = self._scale * (-p_expr * n + wss_expr)
            Pi = Pi_int - inner(traction, u) * ds(2)
        else:
            Pi = Pi_int

        # Optional ventricular spring (Robin / Winkler foundation). Independent of
        # the fluid traction, so it stays active during relaxation too.
        if getattr(self, "_Pi_spring", None) is not None:
            Pi = Pi + self._Pi_spring

        # Optional base x-spring (yroller_spring BC): carries the horizontal load
        # and gives the base a soft restoring x-restraint. Also load-independent.
        if getattr(self, "_Pi_base_spring", None) is not None:
            Pi = Pi + self._Pi_base_spring

        self._R = derivative(Pi, u, v)
        self._K = derivative(self._R, u, du)

    def _solve_u(self):
        """Run one Newton solve of the equilibrium residual for self.u (mumps)."""
        from dolfin import solve
        prm = {
            "newton_solver": {
                "maximum_iterations": int(self.cfg["newton_max_iters"]),
                "relative_tolerance": float(self.cfg["newton_rtol"]),
                "absolute_tolerance": float(self.cfg["newton_atol"]),
                "linear_solver": "mumps",
                "report": True,
                "relaxation_parameter": float(self.cfg["newton_relax"]),
            }
        }
        solve(self._R == 0, self.u, bcs=self._bc,
              J=self._K, solver_parameters=prm)

    def _do_step(self, step_global, xdmf_file=None, grow=True):
        from dolfin import MPI

        if MPI.comm_world.rank == 0:
            if self.cfg.get("anisotropic", False):
                gstr = (f"  lam_g=[{self.lam_g.vector().min():.3f}, "
                        f"{self.lam_g.vector().max():.3f}]  "
                        f"lam_c=[{self.lam_c.vector().min():.3f}, "
                        f"{self.lam_c.vector().max():.3f}]")
            else:
                gstr = f"  g=[{self.g.vector().min():.4f}, {self.g.vector().max():.4f}]"
            print(f"\n  [Solid] step {step_global}  scale={float(self._scale):.4f}{gstr}")

        self._solve_u()

        # Write outputs
        if xdmf_file is not None:
            t_out = float(step_global)
            xdmf_file.write(self.u, t_out)
            self._write_fields(xdmf_file, t_out)

        # Growth update (skipped during ramp — only runs during hold)
        if grow:
            self._growth_update()

    def principal_stress_field(self):
        """Per-cell principal Cauchy stresses + their directions, for visualizing
        stress alignment (e.g. to design stress-biased anisotropic growth).

        Returns a dict of 1-D arrays ordered to match cell-iteration order (same
        ordering as cell_centroids.npy and the g field):
          cx, cy      : cell-centroid coordinates (m)
          s1, s2      : max / min principal Cauchy stress (Pa), s1 >= s2
          e1x, e1y    : unit eigenvector of s1 (max-principal / most-tensile dir)
          e2x, e2y    : unit eigenvector of s2 (min-principal / most-compressive)
        Uses a DG0 projection of the Cauchy stress tensor + the closed-form 2x2
        symmetric eigendecomposition.
        """
        from dolfin import project
        sxx = project(self._sigma_cauchy_expr[0, 0], self.Vg).vector().get_local()
        sxy = project(self._sigma_cauchy_expr[0, 1], self.Vg).vector().get_local()
        syy = project(self._sigma_cauchy_expr[1, 1], self.Vg).vector().get_local()
        mean = 0.5 * (sxx + syy)
        R    = np.sqrt((0.5 * (sxx - syy)) ** 2 + sxy ** 2)
        s1, s2 = mean + R, mean - R
        theta  = 0.5 * np.arctan2(2.0 * sxy, sxx - syy)   # max-principal angle
        e1x, e1y = np.cos(theta), np.sin(theta)
        coords = self.mesh.coordinates()
        cc = np.array([coords[c].mean(axis=0) for c in self.mesh.cells()])
        return dict(cx=cc[:, 0], cy=cc[:, 1], s1=s1, s2=s2,
                    e1x=e1x, e1y=e1y, e2x=-e1y, e2y=e1x)

    def _write_fields(self, xdmf, t_out):
        from dolfin import project, Function, MPI

        def pw(expr, space, name):
            f = project(expr, space)
            f.rename(name, name)
            xdmf.write(f, t_out)
            return f

        pw(self._sigma_vm_expr,   self.S,   "von_mises")
        pw(self._sigma_max_expr,  self.S,   "sigma_max_principal")
        pw(self._sigma_min_expr,  self.S,   "sigma_min_principal")
        pw(self._sigma_ten_expr,  self.S,   "sigma_tension")
        pw(self._sigma_comp_expr, self.S,   "sigma_comp_mag")
        pw(self._J_total_expr,    self.Vd0, "J_total")
        pw(self._Je_expr,         self.Vd0, "J_e")
        pw(self._Jg_expr,         self.Vd0, "J_g")

        g_out = Function(self.Vg)
        g_out.assign(self.g)
        g_out.rename("growth_factor_g", "growth_factor_g")
        xdmf.write(g_out, t_out)

    def _growth_dg(self):
        """
        Compute the per-cell growth increment Δg from the CURRENT converged stress
        WITHOUT modifying self.g. Returns (dg_masked, sigma_comp_arr, stimulus).

        dg_masked already includes the per-step rate limiter and the height mask,
        so a caller applies  g_new = clip(g_base + frac*dg_masked, g_min, g_max)
        for a sub-step fraction frac in (0, 1] (see run_hold's adaptive loop).
        """
        from dolfin import project

        sigma_comp_home = float(self.cfg["sigma_comp_home_Pa"])
        sigma_comp_dead = float(self.cfg["sigma_comp_dead_Pa"])
        kG  = float(self.cfg["k_g"])
        dtG = float(self.cfg["dt_g"])

        # ── Compression-homeostasis stimulus (Buskohl-style) ──────────────
        # Project compressive stress magnitude to DG0 (one value per cell)
        sigma_comp_arr = project(self._sigma_comp_expr, self.Vg).vector().get_local()

        # Per-cell homeostatic setpoint: the base (scalar) setpoint everywhere,
        # LOWERED on the gated atrial band so that face reads as over-compressed and
        # resorbs (g<1) while the ventricular flank keeps the full setpoint and grows.
        # Two forms for the atrial band target (blended in by gate·w_atrial):
        #   ABSOLUTE (atrial_setpoint_abs > 0): band target → a fixed Pa value,
        #     decoupled from the stage setpoint so it does not run away at high stages.
        #       sp_field = base + gate·w_atrial·(abs_target − base)
        #   FRACTIONAL (else): band target → base × atrial_setpoint_frac.
        #       sp_field = base × (1 − gate · w_atrial · (1 − atrial_setpoint_frac))
        # gate=0 (or off) → sp_field ≡ base everywhere (original scalar behaviour).
        atrial_frac = float(self.cfg.get("atrial_setpoint_frac", 1.0))
        atrial_abs  = float(self.cfg.get("atrial_setpoint_abs", 0.0))
        if self._w_atrial is not None and self._atrial_gate > 0.0 and atrial_abs > 0.0:
            sp_field = sigma_comp_home + \
                self._atrial_gate * self._w_atrial * (atrial_abs - sigma_comp_home)
        elif self._w_atrial is not None and atrial_frac < 1.0 and self._atrial_gate > 0.0:
            sp_field = sigma_comp_home * (
                1.0 - self._atrial_gate * self._w_atrial * (1.0 - atrial_frac))
        else:
            sp_field = sigma_comp_home

        # Signed deviation from homeostatic setpoint:
        #   deviation > 0  →  under-compressed  →  grow
        #   deviation < 0  →  over-compressed   →  resorb
        deviation = sp_field - sigma_comp_arr

        # Dead band: suppress response within ±sigma_comp_dead of homeostasis
        stimulus = np.where(
            np.abs(deviation) > sigma_comp_dead,
            deviation - np.sign(deviation) * sigma_comp_dead,
            0.0,
        )

        # Normalised growth increment
        dg = dtG * kG * stimulus / (sigma_comp_home + 1e-12)

        # ── Per-step growth-rate limiter (numerical safeguard) ────────────
        # Cap |dg| so a single step can't collapse/inflate g enough to invert the
        # mesh. This is a coarse floor; the fine control is run_hold's adaptive
        # sub-stepping, which shrinks the increment further when a high-load solve
        # (HH27/30) can't absorb even a capped step.
        dg_max = float(self.cfg.get("dg_max_step", 0.05))
        dg = np.clip(dg, -dg_max, dg_max)

        # ── Tension-based stimulus (disabled — re-enable via use_tension_stimulus) ──
        # if self.cfg.get("use_tension_stimulus", False):
        #     ... (unchanged; see git history)

        # Height mask: suppress growth in the narrow clamped-base zone.
        return self._w_height * dg, sigma_comp_arr, stimulus

    def _apply_growth(self, g_base, increment):
        """Set self.g = clip(g_base + increment, g_min, g_max). g_base is a numpy
        array (the pre-growth values); increment the (possibly scaled) Δg array."""
        from dolfin import Function
        arr = np.clip(g_base + increment,
                      float(self.cfg["g_min"]), float(self.cfg["g_max"]))
        g_new = Function(self.Vg)
        g_new.vector().set_local(arr)
        g_new.vector().apply("insert")
        self.g.assign(g_new)

    # ── Anisotropic (tensorial) growth update ───────────────────────────────
    def _dg0_set(self, fn, arr):
        """Assign a per-cell numpy array to a DG0 Function (shared Vg ordering)."""
        from dolfin import Function
        f = Function(self.Vg)
        f.vector().set_local(np.ascontiguousarray(arr, dtype=float))
        f.vector().apply("insert")
        fn.assign(f)

    def _smooth_dg0(self, arr, n_passes=1):
        """Spatially smooth a per-cell DG0 array by projecting DG0→CG1 (continuous,
        node-averages neighbours) and back to DG0, n_passes times. Used to regularize
        the growth-direction field so it doesn't kink the boundary cell-to-cell."""
        from dolfin import Function, project
        f = Function(self.Vg)
        f.vector().set_local(np.ascontiguousarray(arr, dtype=float))
        f.vector().apply("insert")
        for _ in range(int(n_passes)):
            f = project(project(f, self.S), self.Vg)   # DG0 → CG1 → DG0
        return f.vector().get_local()

    def _growth_aniso(self):
        """
        Anisotropic growth increment from the CURRENT converged stress.

        DIRECTION: a ← e1 (max-principal = LEAST-compressed axis), with a sign
        convention (a_y ≥ 0) and a per-step rotation limiter (blend toward the new
        target from the previous a) so the field evolves smoothly and doesn't jitter
        the boundary. n = a^perp = e2 (MOST-compressed axis).

        MAGNITUDE: split the SAME compression-homeostasis stimulus used isotropically —
          under-compressed (stimulus>0) → grow along a  (lam_g up),
          over-compressed  (stimulus<0) → thin across n (lam_c down).
        Returns (a_x_new, a_y_new, dlam_g, dlam_c, sigma_comp_arr, stimulus).
        """
        from dolfin import project
        setpoint = float(self.cfg["sigma_comp_home_Pa"])
        dead     = float(self.cfg["sigma_comp_dead_Pa"])
        kG, dtG  = float(self.cfg["k_g"]), float(self.cfg["dt_g"])

        ps = self.principal_stress_field()
        e1x, e1y = ps["e1x"], ps["e1y"]
        sigma_comp_arr = project(self._sigma_comp_expr, self.Vg).vector().get_local()

        # Direction: e1 with consistent sign (upper half-plane), SPATIALLY smoothed
        # (so neighbouring boundary cells share an axis → no bumpy arc), then the
        # temporal rotation limiter.
        ax, ay = e1x.copy(), e1y.copy()
        flip = ay < 0.0
        ax[flip] *= -1.0
        ay[flip] *= -1.0
        npass = int(self.cfg.get("aniso_smooth_passes", 2))
        if npass > 0:
            ax = self._smooth_dg0(ax, npass)
            ay = self._smooth_dg0(ay, npass)
            nrm0 = np.sqrt(ax**2 + ay**2) + 1e-30
            ax /= nrm0
            ay /= nrm0
        relax = float(self.cfg.get("aniso_dir_relax", 0.30))
        ax_prev = self.a_x.vector().get_local()
        ay_prev = self.a_y.vector().get_local()
        ax_new = (1.0 - relax) * ax_prev + relax * ax
        ay_new = (1.0 - relax) * ay_prev + relax * ay
        nrm = np.sqrt(ax_new**2 + ay_new**2) + 1e-30
        ax_new /= nrm
        ay_new /= nrm

        # Magnitude: split the homeostasis stimulus (same dead-band as isotropic).
        deviation = setpoint - sigma_comp_arr
        stimulus = np.where(np.abs(deviation) > dead,
                            deviation - np.sign(deviation) * dead, 0.0)
        rate   = dtG * kG / (setpoint + 1e-12)
        dg_max = float(self.cfg.get("dg_max_step", 0.05))
        dlam_g =  np.clip(rate * np.maximum(stimulus, 0.0), 0.0, dg_max) * self._w_height
        dlam_c = -np.clip(rate * np.maximum(-stimulus, 0.0), 0.0, dg_max) * self._w_height
        return ax_new, ay_new, dlam_g, dlam_c, sigma_comp_arr, stimulus

    def _set_direction(self, ax_new, ay_new):
        """Update the per-cell growth direction a = (a_x, a_y) (already normalized)."""
        self._dg0_set(self.a_x, ax_new)
        self._dg0_set(self.a_y, ay_new)

    def _apply_growth_aniso(self, lg_base, lc_base, dlam_g, dlam_c, frac):
        """lam_g ← clip(base + frac·Δ, 1, lam_g_max);  lam_c ← clip(base + frac·Δ,
        lam_c_min, 1). Grow-along-a stays ≥1, thin-across-n stays ≤1."""
        lg_max = float(self.cfg.get("aniso_lam_g_max", 2.0))
        lc_min = float(self.cfg.get("aniso_lam_c_min", 0.3))
        self._dg0_set(self.lam_g, np.clip(lg_base + frac * dlam_g, 1.0, lg_max))
        self._dg0_set(self.lam_c, np.clip(lc_base + frac * dlam_c, lc_min, 1.0))

    def _growth_update(self):
        """Legacy single-shot growth update: compute Δg from the current stress,
        apply it fully, and log. Retained for any caller not using run_hold's
        adaptive sub-stepping path."""
        from dolfin import MPI
        dg_masked, sigma_comp_arr, stimulus = self._growth_dg()
        self._apply_growth(self.g.vector().get_local(), dg_masked)
        if MPI.comm_world.rank == 0:
            print(f"  [Solid] g updated → [{self.g.vector().min():.4f}, "
                  f"{self.g.vector().max():.4f}]  "
                  f"setpoint={float(self.cfg['sigma_comp_home_Pa']):.1f} Pa  "
                  f"sigma_comp range: [{sigma_comp_arr.min():.1f}, "
                  f"{sigma_comp_arr.max():.1f}] Pa  "
                  f"stimulus range: [{stimulus.min():.3f}, {stimulus.max():.3f}] Pa")