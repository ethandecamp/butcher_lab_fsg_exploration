#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_growth.py
-------------------
Splice the per-iteration cushion outlines from an FSG run into a single
comparison figure + an animation, so you can SEE size/shape change over time
without hand-loading each step in ParaView.

Reads step_NNN/arc_data.npz (the deformed arc x,y at each FSG iteration) from a
run directory and produces, in <run>/growth_viz/:
  - growth_overlay.png   : all cushion profiles overlaid, colored by iteration,
                           with the original blip dashed in black for reference,
                           plus height/area-vs-iteration trend plots.
  - growth_anim.gif      : animated cushion growing frame-by-frame.

USAGE
  python visualize_growth.py                     # default run below
  python visualize_growth.py "FSG Results/longterm_blip_HH21_np20_nfsg10"
  python visualize_growth.py <run_dir> --stride 1 --equal   # tweak

Run in the `fenics` conda env (matplotlib + numpy + Pillow available there).
"""

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.ticker import MaxNLocator


DEFAULT_RUN = "FSG Results/longterm_blip_HH21_np20_nfsg10"


def load_profiles(run_dir):
    """Return list of (step, x_mm, y_mm, s_norm, p_Pa, wss_Pa, osi) per
    step_NNN/arc_data.npz. Points are ordered atrial->ventricular (left->right
    by x); s_norm is normalized arc length [0,1] along the cushion; p and wss
    (=tau_mag) are the fluid pressure and wall shear stress along that same arc;
    osi is the oscillatory shear index (transient runs only; 0 for quasi-steady)."""
    step_dirs = sorted(glob.glob(os.path.join(run_dir, "step_*")))
    profs = []
    for sd in step_dirs:
        f = os.path.join(sd, "arc_data.npz")
        if not os.path.exists(f):
            continue
        k = int(os.path.basename(sd).split("_")[1])
        d = np.load(f)
        x = np.asarray(d["x"]) * 1e3   # m -> mm
        y = np.asarray(d["y"]) * 1e3
        p   = np.asarray(d["p"])       if "p" in d       else np.zeros_like(x)
        wss = np.asarray(d["tau_mag"]) if "tau_mag" in d else np.zeros_like(x)
        osi = np.asarray(d["osi"])     if "osi"     in d else np.zeros_like(x)
        # Order atrial -> ventricular ALONG THE OUTLINE by angle about the
        # base-centroid (pi -> 0), not by x. A plain argsort(x) scrambles the
        # steep flanks / incipient lean (points non-monotonic in x) into a
        # zigzag — the "jagged edges" seen in the GIF are that artifact, not
        # real boundary roughness. Angle order matches how get_deformed_arc
        # orders the simulation arc, so the plotted outline is the true shape
        # (and poly_area is computed on a correctly-traced polygon).
        xc = 0.5 * (x.min() + x.max())
        ang = np.arctan2(np.maximum(y, 0.0) + 1e-9, x - xc)
        order = np.argsort(-ang)
        x, y, p, wss, osi = x[order], y[order], p[order], wss[order], osi[order]
        seg = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        s = s / s[-1] if s[-1] > 0 else s
        profs.append((k, x, y, s, p, wss, osi))
    profs.sort(key=lambda t: t[0])
    # Drop mesher-fallback steps: when the fluid/solid mesh build fails or hangs,
    # the run reverts to the UNDEFORMED reference arc for that step (see the
    # "Falling back to undeformed arc" log line + the gmsh watchdog). Such a step's
    # outline is identical to the iteration-0 blip, which poisons the height/area
    # summary (blip-to-blip = +0%) and the GIF. Any k>0 whose (height,width) matches
    # step 0 within 1 um is a reset, not real geometry — hold it out.
    if profs:
        h0 = profs[0][2].max()
        w0 = profs[0][1].max() - profs[0][1].min()
        kept, dropped = [], []
        for pr in profs:
            k, x, y = pr[0], pr[1], pr[2]
            if k > 0 and abs(y.max() - h0) < 1e-3 and abs((x.max() - x.min()) - w0) < 2e-3:
                dropped.append(k)
            else:
                kept.append(pr)
        if dropped:
            print(f"  [viz] skipping {len(dropped)} mesher-fallback step(s) "
                  f"(outline == reference blip): {dropped}")
        profs = kept
    return profs


def anchor_profiles(profs, mode):
    """Shift each profile in x so one chosen edge stays fixed across iterations,
    removing the rigid downstream slide for clearer visualization (shape only —
    does not change height/area). Flow is +x, so atrial/upstream = LEFT edge
    (min x) and ventricular/downstream = RIGHT edge (max x). The reference is the
    iteration-0 profile's corresponding feature.

      mode 'atrial'/'left'/'upstream'      : pin the atrial (left) edge; growth
                                             then visibly extends toward the ventricle
      mode 'ventricular'/'right'/'downstream': pin the ventricular (right) edge
      mode 'center'                         : pin the mid-base point
      mode 'none'                           : no shift (true lab-frame positions)
    """
    mode = (mode or "none").lower()
    if mode in ("none", ""):
        return profs

    def feat(x):
        if mode in ("atrial", "left", "upstream"):
            return x.min()
        if mode in ("ventricular", "right", "downstream"):
            return x.max()
        if mode == "center":
            return 0.5 * (x.min() + x.max())
        sys.exit(f"unknown --anchor mode: {mode}")

    ref = feat(profs[0][1])
    return [(k, x + (ref - feat(x)), y, s, p, wss, osi)
            for (k, x, y, s, p, wss, osi) in profs]


def poly_area(x, y):
    """Cross-sectional area of cushion = arc closed by its base chord (mm^2)."""
    xs = np.r_[x, x[0]]
    ys = np.r_[y, y[0]]
    return 0.5 * abs(np.sum(xs[:-1] * ys[1:] - xs[1:] * ys[:-1]))


def load_stress_field(run_dir):
    """Latest step's principal Cauchy-stress field + that step's (unanchored)
    outline. Returns (k, sdict, (x_mm, y_mm)) or (None, None, None) when no
    stress_field.npz exists (runs predating the solver stress output)."""
    sfiles = sorted(glob.glob(os.path.join(run_dir, "step_*", "stress_field.npz")))
    if not sfiles:
        return None, None, None
    f = sfiles[-1]
    k = int(os.path.basename(os.path.dirname(f)).split("_")[1])
    sdat = {key: np.asarray(v) for key, v in np.load(f).items()}
    outline = None
    arc = os.path.join(os.path.dirname(f), "arc_data.npz")
    if os.path.exists(arc):
        d = np.load(arc); xo = np.asarray(d["x"]) * 1e3; yo = np.asarray(d["y"]) * 1e3
        xcn = 0.5 * (xo.min() + xo.max())               # angle order (see load_profiles)
        o = np.argsort(-np.arctan2(np.maximum(yo, 0.0) + 1e-9, xo - xcn))
        outline = (xo[o], yo[o])
    return k, sdat, outline


def load_aniso_field(run_dir):
    """Latest step's anisotropic growth fields (lam_g, lam_c, a_x, a_y) or None.
    run_dir may be a str or a list of dirs (latest step across all wins)."""
    run_dirs = [run_dir] if isinstance(run_dir, str) else list(run_dir)
    afiles = []
    for rd in run_dirs:
        afiles += glob.glob(os.path.join(rd, "step_*", "growth_aniso.npz"))
    if not afiles:
        return None
    afiles.sort(key=lambda p: int(os.path.basename(os.path.dirname(p)).split("_")[1]))
    return {k: np.asarray(v) for k, v in np.load(afiles[-1]).items()}


def _open_axes(ax):
    """Open 2-spine look: keep the x and y main axis lines, drop the top/right box.
    (Spine thickness comes from rcParams axes.linewidth.)"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)


def _placeholder_panel(ax, title, note="pending —\npopulates on next run"):
    """Empty but framed panel with title, for fields not yet in the saved data."""
    _open_axes(ax)
    ax.text(0.5, 0.5, note, ha="center", va="center", transform=ax.transAxes,
            fontsize=13, color="0.5")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=14)


def _glyph_panel(fig, ax, cx, cy, mag, ex, ey, cmap, title, outline,
                 clip=(5, 95)):
    """Cushion field as glyphs: color = scalar `mag` (clipped to clip-percentiles
    so a singularity can't wash it out), arrow = (ex,ey) direction, oriented to the
    +x hemisphere for consistent arrowheads. cx,cy in metres."""
    _open_axes(ax)
    sgn = np.where(ex < 0, -1.0, 1.0)
    ex, ey = ex * sgn, ey * sgn
    n = len(cx); stride = max(1, n // 120); sl = slice(None, None, stride)
    L = 0.022
    q = ax.quiver(cx[sl] * 1e3, cy[sl] * 1e3, ex[sl] * L, ey[sl] * L, mag[sl],
                  cmap=cmap, pivot="mid", angles="xy", scale_units="xy", scale=1,
                  width=0.007, headwidth=3.0, headlength=3.3, headaxislength=3.0,
                  edgecolor="0.3", linewidth=0.5)
    lo, hi = float(np.percentile(mag, clip[0])), float(np.percentile(mag, clip[1]))
    if hi <= lo:
        hi = lo + 1e-9
    q.set_clim(lo, hi)
    if outline is not None:
        ax.plot(outline[0], outline[1], color="0.3", lw=1.8)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("y (mm)")
    cb = fig.colorbar(q, ax=ax, shrink=0.85, aspect=11, pad=0.02, ticks=[lo, hi])
    cb.ax.tick_params(labelsize=12)


def make_overlay(profs, out_path, equal_aspect, anchor="none", run_dir=None):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 18, "font.weight": "bold",
        "axes.titleweight": "bold", "axes.titlesize": 18,
        "axes.labelweight": "bold", "axes.labelsize": 18,
        "xtick.labelsize": 16, "ytick.labelsize": 16,
        "axes.linewidth": 2.2,                       # thicker x/y axis lines
        "xtick.major.width": 2.0, "ytick.major.width": 2.0,
        "xtick.major.size": 7, "ytick.major.size": 7,
    })
    ks = [p[0] for p in profs]
    kmin, kmax = min(ks), max(ks)
    norm = colors.Normalize(vmin=kmin, vmax=kmax)
    cmap = cm.viridis

    heights = [p[2].max() for p in profs]
    areas   = [poly_area(p[1], p[2]) for p in profs]
    a0 = areas[0]

    def shade(k, base):
        """Darker line for later iterations (kept light enough to stay visible)."""
        t = 0.25 + 0.75 * ((k - kmin) / (kmax - kmin) if kmax > kmin else 1.0)
        return base(t)

    def three_ticks(ax):
        ax.xaxis.set_major_locator(MaxNLocator(nbins=2, min_n_ticks=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=2, min_n_ticks=3))

    _, sdat, soutline = load_stress_field(run_dir) if run_dir else (None, None, None)

    fig = plt.figure(figsize=(30, 11))
    # 2 rows x 5 columns. Col1 profile/σ1 | Col2 cum-growth/strain |
    # Col3 height/area | Col4 pressure/WSS | Col5 OSI/WSS-gradient.
    gs = fig.add_gridspec(2, 5, hspace=0.55, wspace=0.42)

    # ── COL 1: Cushion Profile (top) + σ1 Max Principal Stress (bottom) ──────
    axp = fig.add_subplot(gs[0, 0])
    for (k, x, y, *_rest) in profs:
        axp.plot(x, y, color=cmap(norm(k)), lw=2.0, alpha=0.9)
    axp.plot(profs[0][1], profs[0][2], "k--", lw=2.4)   # original blip reference
    if anchor not in ("none", ""):
        xref = (profs[0][1].min() if anchor in ("atrial", "left", "upstream")
                else profs[0][1].max() if anchor in ("ventricular", "right", "downstream")
                else 0.5 * (profs[0][1].min() + profs[0][1].max()))
        axp.axvline(xref, color="0.4", ls=":", lw=1.4)
    axp.set_title("Cushion Profile", fontsize=15)
    axp.set_ylabel("y (mm)")
    axp.set_xlabel("x (mm)")
    # x and y are the SAME physical units (mm) — always render at true 1:1 scale
    # so the profile shows real proportions, not a vertically-stretched dome.
    # (Forced on regardless of --equal, which now only affects other panels.)
    axp.set_aspect("equal")
    axp.set_ylim(bottom=0.0)          # base sits on the axis for a clean, honest shape
    _open_axes(axp)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=axp, shrink=0.85, aspect=11, pad=0.02, ticks=[kmin, kmax])
    cb.set_label("Iteration", fontsize=13); cb.ax.tick_params(labelsize=12)

    axs1 = fig.add_subplot(gs[1, 0])
    if sdat is not None:
        _glyph_panel(fig, axs1, sdat["cx"], sdat["cy"], sdat["s1"],
                     sdat["e1x"], sdat["e1y"], cm.Reds,
                     "σ1 Max Principal Stress\n(Tensile, Pa)", soutline)
        axs1.set_xlabel("x (mm)")
    else:
        _placeholder_panel(axs1, "σ1 Max Principal Stress (Tensile)")

    # ── COL 2: Cumulative Growth (top) + Max Principal Strain (bottom) ───────
    axg = fig.add_subplot(gs[0, 1])
    gcum = None
    if run_dir is not None and sdat is not None:
        gf = os.path.join(run_dir, "cumulative_g.npy")   # saved at end of a completed run
        if os.path.exists(gf):
            arr = np.load(gf)
            if len(arr) == len(sdat["cx"]):              # same final mesh as last stress_field
                gcum = arr
    adat = load_aniso_field(run_dir) if run_dir is not None else None
    if (adat is not None and sdat is not None
            and len(adat["lam_g"]) == len(sdat["cx"])):
        # Anisotropic growth: color = lam_g/lam_c (>1 elongate-dominant, =1 isotropic),
        # arrows = the growth axis a at each element (like the stress glyphs).
        ratio = adat["lam_g"] / np.maximum(adat["lam_c"], 1e-6)
        _glyph_panel(fig, axg, sdat["cx"], sdat["cy"], ratio,
                     adat["a_x"], adat["a_y"], cm.magma,
                     "Growth anisotropy  λg/λc\n[arrows = growth axis a]", soutline,
                     clip=(2, 98))
    elif gcum is not None:
        _glyph_panel(fig, axg, sdat["cx"], sdat["cy"], gcum,
                     sdat["e1x"], sdat["e1y"], cm.Greens,
                     "Cumulative Growth (g)\n[arrows = σ1 dir]", soutline,
                     clip=(2, 98))
    else:
        _placeholder_panel(axg, "Cumulative Growth (g)")

    axs2 = fig.add_subplot(gs[1, 1])
    if sdat is not None:
        _glyph_panel(fig, axs2, sdat["cx"], sdat["cy"], sdat["s2"],
                     sdat["e2x"], sdat["e2y"], cm.Blues,
                     "σ2 Min Principal Stress\n(Compressive, Pa)", soutline)
        axs2.set_xlabel("x (mm)")
    else:
        _placeholder_panel(axs2, "σ2 Min Principal Stress (Compressive, Pa)")

    # ── COL 3: Maximum Height (top) + Cross-Sectional Area (bottom) ──────────
    axh = fig.add_subplot(gs[0, 2])
    axh.plot(ks, heights, "o-", color="#c0392b", ms=6, lw=2.4)
    axh.set_title("Maximum Height")
    axh.set_ylabel("Height (mm)")
    _open_axes(axh); three_ticks(axh)

    axa = fig.add_subplot(gs[1, 2])
    axa.plot(ks, np.array(areas) / a0, "s-", color="#2c7fb8", ms=6, lw=2.4)
    axa.set_title("Cross-Sectional Area")
    axa.set_xlabel("FSG Iteration")
    axa.set_ylabel("Area (Normalized to Initial)")
    _open_axes(axa); three_ticks(axa)

    # ── COL 4: Pressure (top) + WSS (bottom), darker = later iteration ───────
    axpr = fig.add_subplot(gs[0, 3])
    for (k, x, y, s, p, wss, osi) in profs:
        axpr.plot(s, p, color=shade(k, cm.Reds), lw=1.8)
    axpr.set_title("Pressure (Pa)")
    _open_axes(axpr); three_ticks(axpr)

    axw = fig.add_subplot(gs[1, 3])
    for (k, x, y, s, p, wss, osi) in profs:
        axw.plot(s, wss * 10.0, color=shade(k, cm.Blues), lw=1.8)   # Pa -> dyn/cm²
    axw.set_title("WSS (dynes/cm²)")
    axw.set_xlabel("Arc Length (Atrial → Ventricular)")
    _open_axes(axw); three_ticks(axw)

    # ── COL 5: OSI (top) + WSS Gradient (bottom) ─────────────────────────────
    # OSI along the arc, per iteration (from the transient solver's arc_data['osi'];
    # quasi-steady runs carry osi=0). Darker = later iteration, atrial→ventricular.
    axo = fig.add_subplot(gs[0, 4])
    osi_max = max((float(np.max(osi)) for (k, x, y, s, p, wss, osi) in profs),
                  default=0.0)
    for (k, x, y, s, p, wss, osi) in profs:
        axo.plot(s, osi, color=shade(k, cm.Oranges), lw=1.8)
    axo.set_ylim(-0.02, 0.5)                        # OSI is bounded [0, 0.5]
    axo.set_title("OSI")
    axo.set_xlabel("Arc Length (Atrial → Ventricular)")
    if osi_max < 1e-3:
        axo.text(0.5, 0.86, "≈0 (unidirectional flow\nat this stage)", ha="center",
                 va="center", transform=axo.transAxes, fontsize=12, color="0.5")
    _open_axes(axo); three_ticks(axo)

    axwg = fig.add_subplot(gs[1, 4])
    for (k, x, y, s, p, wss, osi) in profs:
        sphys = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
        grad = np.gradient(wss * 10.0, sphys)                       # (dyn/cm²)/mm
        axwg.plot(s, grad, color=shade(k, cm.Purples), lw=1.8)
    axwg.set_title("WSS Gradient (dyn/cm²/mm)", fontsize=15)
    axwg.set_xlabel("Arc Length (Atrial → Ventricular)")
    _open_axes(axwg); three_ticks(axwg)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return heights, areas


def make_anim(profs, out_path, anchor="none"):
    from matplotlib.animation import FuncAnimation, PillowWriter

    ks = [p[0] for p in profs]
    norm = colors.Normalize(vmin=min(ks), vmax=max(ks))
    cmap = cm.viridis

    # Data-driven x-limits (so an asymmetric downstream extension is fully shown).
    xlo = min(p[1].min() for p in profs)
    xhi = max(p[1].max() for p in profs)
    xpad = 0.05 * (xhi - xlo)
    ymax = max(p[2].max() for p in profs) * 1.20
    xref = None
    if anchor not in ("none", ""):
        xref = (profs[0][1].min() if anchor in ("atrial", "left", "upstream")
                else profs[0][1].max() if anchor in ("ventricular", "right", "downstream")
                else 0.5 * (profs[0][1].min() + profs[0][1].max()))

    fig, ax = plt.subplots(figsize=(7, 4.2))

    def draw(i):
        ax.clear()
        # ghost: all earlier profiles faint
        for j in range(i + 1):
            k, x, y = profs[j][:3]
            ax.plot(x, y, color=cmap(norm(k)),
                    lw=(2.2 if j == i else 0.8),
                    alpha=(1.0 if j == i else 0.25))
        k, x, y = profs[i][:3]
        ax.fill(np.r_[x, x[0]], np.r_[y, y[0]],
                color=cmap(norm(k)), alpha=0.25)
        x0, y0 = profs[0][1], profs[0][2]
        ax.plot(x0, y0, "k--", lw=1.2)
        ax.axhline(0, color="0.6", lw=0.8)
        if xref is not None:
            ax.axvline(xref, color="0.4", ls=":", lw=1.0)
        ax.set_xlim(xlo - xpad, xhi + xpad); ax.set_ylim(-0.002, ymax)
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
        ax.set_title(f"FSG iteration {k:02d}   |   peak height {y.max():.4f} mm   "
                     f"(orig 0.0500 mm, dashed)")

    anim = FuncAnimation(fig, draw, frames=len(profs), interval=500)
    anim.save(out_path, writer=PillowWriter(fps=2))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default=DEFAULT_RUN)
    ap.add_argument("--stride", type=int, default=1,
                    help="plot every Nth iteration (useful for long runs)")
    ap.add_argument("--equal", action="store_true",
                    help="equal aspect ratio (true mm shape; thin cushions look flat)")
    ap.add_argument("--no-anim", action="store_true", help="skip the GIF")
    ap.add_argument("--anchor", default="none",
                    choices=["none", "atrial", "left", "upstream",
                             "ventricular", "right", "downstream", "center"],
                    help="hold one edge fixed in x across iterations (visualization "
                         "only). 'atrial' pins the upstream/left edge so the rigid "
                         "downstream slide is removed and growth visibly extends "
                         "toward the ventricle.")
    args = ap.parse_args()

    profs = load_profiles(args.run_dir)
    if not profs:
        sys.exit(f"No step_*/arc_data.npz found in {args.run_dir}")
    if args.stride > 1:
        profs = profs[::args.stride]
    profs = anchor_profiles(profs, args.anchor)

    out_dir = os.path.join(args.run_dir, "growth_viz")
    os.makedirs(out_dir, exist_ok=True)
    overlay = os.path.join(out_dir, "growth_overlay.png")
    heights, areas = make_overlay(profs, overlay, args.equal, args.anchor,
                                  run_dir=args.run_dir)
    print(f"  {len(profs)} profiles  |  height {heights[0]:.4f} -> {heights[-1]:.4f} mm "
          f"(+{(heights[-1]/heights[0]-1)*100:.1f}%)  |  "
          f"area x{areas[-1]/areas[0]:.3f}")
    print(f"  saved: {overlay}")

    if not args.no_anim:
        gif = os.path.join(out_dir, "growth_anim.gif")
        try:
            make_anim(profs, gif, args.anchor)
            print(f"  saved: {gif}")
        except Exception as e:
            print(f"  [WARN] GIF skipped ({e})")


if __name__ == "__main__":
    main()
