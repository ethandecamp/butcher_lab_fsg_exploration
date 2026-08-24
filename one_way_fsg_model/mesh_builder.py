#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mesh_builder.py
---------------
Gmsh routines for:
  - Solid domain  : upper half-ellipse cap
  - Fluid domain  : rectangular channel MINUS the deformed solid cap

Physical tags (shared convention):
  Solid:
    1  -> diameter (base, clamped)
    2  -> arc (loaded surface)
    10 -> solid surface

  Fluid:
    10 -> inlet  (left wall)
    11 -> outlet (right wall)
    12 -> top wall
    13 -> solid arc (no-slip, traction extraction)
    14 -> bottom wall (if present; unused in simple channel – set as symmetry or wall)
    20 -> fluid surface
"""

import math
import numpy as np
import gmsh


# ─────────────────────────────────────────────────────────────────────────────
# Solid mesh (half-ellipse cap)
# ─────────────────────────────────────────────────────────────────────────────

def build_solid_mesh(msh_path,
                     a_cap, b_cap,
                     xc, yc,
                     h_size,
                     n_arc_pts=60):
    """
    Solid domain: upper half of ellipse (xc±a_cap, yc to yc+b_cap).
    All coordinates in METRES.

    Physical groups:
      Line 1  -> diameter (y≈yc)
      Line 2  -> arc
      Surface 10 -> solid domain
    """
    if n_arc_pts < 3:
        raise ValueError("n_arc_pts must be >= 3")

    gmsh.initialize()
    gmsh.model.add("solid_cap")

    t_vals = [math.pi * i / (n_arc_pts - 1) for i in range(n_arc_pts)]
    arc_pts = []
    for t in t_vals:
        x = xc + a_cap * math.cos(t)
        y = yc + b_cap * math.sin(t)
        arc_pts.append(gmsh.model.occ.addPoint(x, y, 0.0, h_size))

    pL = arc_pts[0]
    pR = arc_pts[-1]

    arc_tag = gmsh.model.occ.addSpline(arc_pts)
    diam_tag = gmsh.model.occ.addLine(pR, pL)

    wire = gmsh.model.occ.addWire([arc_tag, diam_tag])
    surf = gmsh.model.occ.addPlaneSurface([wire])

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h_size)

    gmsh.model.addPhysicalGroup(1, [diam_tag], tag=1)
    gmsh.model.addPhysicalGroup(1, [arc_tag],  tag=2)
    gmsh.model.addPhysicalGroup(2, [surf],     tag=10)

    gmsh.model.mesh.generate(2)
    gmsh.write(msh_path)
    gmsh.finalize()


# ─────────────────────────────────────────────────────────────────────────────
# Solid mesh from arbitrary arc (remeshing after each FSG step)
# ─────────────────────────────────────────────────────────────────────────────

def build_solid_mesh_from_arc(msh_path, arc_coords, h_size, h_corner=None):
    """
    Build a solid mesh whose arc boundary follows arbitrary deformed coordinates.

    Used during FSG remeshing: after each iteration the grown/deformed solid
    geometry becomes the new reference configuration, so the solid mesh must
    be rebuilt around the new shape rather than the original ellipse.

    Parameters
    ----------
    msh_path : str
        Output .msh file path.
    arc_coords : (N, 2) array
        Deformed arc points in metres, ordered LEFT → RIGHT,
        with endpoints on y ≈ 0 (the base line).
    h_size : float
        Target element size (metres).

    Physical groups  (same convention as build_solid_mesh):
      Line 1    → base / diameter  (clamped, Dirichlet BC)
      Line 2    → arc surface      (traction loading surface)
      Surface 10 → solid domain
    """
    if len(arc_coords) < 3:
        raise ValueError("arc_coords must have at least 3 points")

    if h_corner is None:
        h_corner = h_size * 0.5   # match the fluid mesh: resolve the base
                                  # corners finely so the two meshes agree
                                  # there and the junction stays clean

    # Clean gmsh session in case a prior build raised before finalizing.
    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.model.add("solid_cap_deformed")

    # Add arc points
    arc_pts = [
        gmsh.model.occ.addPoint(float(x), float(y), 0.0, h_size)
        for (x, y) in arc_coords
    ]

    pL = arc_pts[0]    # left  endpoint (should be on y≈0)
    pR = arc_pts[-1]   # right endpoint (should be on y≈0)

    arc_tag  = gmsh.model.occ.addSpline(arc_pts)
    diam_tag = gmsh.model.occ.addLine(pR, pL)   # base: right → left

    wire = gmsh.model.occ.addWire([arc_tag, diam_tag])
    surf = gmsh.model.occ.addPlaneSurface([wire])

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h_size)
    # Localized corner refinement (Distance+Threshold), CONFINED to a small
    # radius around the two base endpoints — not setSize on the endpoints, which
    # refined the whole base edge and ~4x'd the solid remesh. Grades h_corner →
    # h_size within corner_radius.
    corner_radius = 6.0 * h_size
    fdist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(fdist, "PointsList", [pL, pR])
    fthr = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(fthr, "InField", fdist)
    gmsh.model.mesh.field.setNumber(fthr, "SizeMin", h_corner)
    gmsh.model.mesh.field.setNumber(fthr, "SizeMax", h_size)
    gmsh.model.mesh.field.setNumber(fthr, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(fthr, "DistMax", corner_radius)
    gmsh.model.mesh.field.setAsBackgroundMesh(fthr)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)

    gmsh.model.addPhysicalGroup(1, [diam_tag], tag=1)
    gmsh.model.addPhysicalGroup(1, [arc_tag],  tag=2)
    gmsh.model.addPhysicalGroup(2, [surf],     tag=10)

    gmsh.model.mesh.generate(2)

    _, tri_tags, _ = gmsh.model.mesh.getElements(2)
    if int(sum(len(t) for t in tri_tags)) == 0:
        gmsh.finalize()
        raise RuntimeError(
            "Solid mesh generation produced 0 triangles (degenerate arc — "
            "likely self-intersecting near the base corners).")

    gmsh.write(msh_path)
    gmsh.finalize()


# ─────────────────────────────────────────────────────────────────────────────
# Fluid mesh (channel minus deformed cap)
# ─────────────────────────────────────────────────────────────────────────────

def build_fluid_mesh(msh_path,
                     arc_coords_m,
                     channel_x0, channel_x1,
                     channel_y0, channel_y1,
                     h_fluid,
                     h_arc=None,
                     h_corner=None):
    """
    Fluid domain: rectangle [x0,x1] x [y0,y1]  MINUS  the solid cap.

    arc_coords_m : (N,2) array of deformed arc points in metres,
                   ordered LEFT → RIGHT (i.e. from (xc-a,yc) to (xc+a,yc)).
                   The diameter endpoints must lie on y≈channel_y0 (the base line
                   of the cap).  The mesh builder closes the cutout with the
                   horizontal chord connecting the two endpoints.

    Physical groups:
      Line 10 -> inlet  (x = channel_x0)
      Line 11 -> outlet (x = channel_x1)
      Line 12 -> top wall  (y = channel_y1)
      Line 13 -> solid arc (no-slip; traction extraction surface)
      Line 14 -> bottom wall (y = channel_y0, excluding cap footprint)
      Surface 20 -> fluid domain
    """
    if h_arc is None:
        h_arc = h_fluid * 0.2   # finer mesh on the arc
    if h_corner is None:
        h_corner = h_arc * 0.5  # extra-fine at the two base corners (arc↔base
                                # junction) where the acute re-entrant angle
                                # otherwise breeds sliver / inverted elements

    # Guarantee a clean gmsh session even if a previous build raised before it
    # could finalize — otherwise the caller's fallback-rebuild inherits a
    # half-open model and fails for a spurious reason.
    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.model.add("fluid_channel")

    # ── channel rectangle corners ──────────────────────────────────────────
    p_bl = gmsh.model.occ.addPoint(channel_x0, channel_y0, 0.0, h_fluid)
    p_br = gmsh.model.occ.addPoint(channel_x1, channel_y0, 0.0, h_fluid)
    p_tr = gmsh.model.occ.addPoint(channel_x1, channel_y1, 0.0, h_fluid)
    p_tl = gmsh.model.occ.addPoint(channel_x0, channel_y1, 0.0, h_fluid)

    l_bot    = gmsh.model.occ.addLine(p_bl, p_br)
    l_outlet = gmsh.model.occ.addLine(p_br, p_tr)
    l_top    = gmsh.model.occ.addLine(p_tr, p_tl)
    l_inlet  = gmsh.model.occ.addLine(p_tl, p_bl)

    outer_wire = gmsh.model.occ.addWire([l_bot, l_outlet, l_top, l_inlet])
    outer_surf = gmsh.model.occ.addPlaneSurface([outer_wire])

    # ── solid cap cutout (spline of deformed arc + closing chord) ──────────
    # arc_coords_m is ordered left→right; gmsh spline needs explicit point tags
    arc_pts = []
    for (x, y) in arc_coords_m:
        arc_pts.append(gmsh.model.occ.addPoint(float(x), float(y), 0.0, h_arc))

    pAL = arc_pts[0]   # left  endpoint of arc (xc-a, yc)
    pAR = arc_pts[-1]  # right endpoint of arc (xc+a, yc)

    arc_spline  = gmsh.model.occ.addSpline(arc_pts)          # arc curve (going UP)
    chord_line  = gmsh.model.occ.addLine(pAR, pAL)           # closing chord (going LEFT along base)

    cap_wire   = gmsh.model.occ.addWire([arc_spline, chord_line])
    cap_surf   = gmsh.model.occ.addPlaneSurface([cap_wire])

    gmsh.model.occ.synchronize()

    # ── Boolean cut: fluid = channel MINUS cap ─────────────────────────────
    cut_result, cut_map = gmsh.model.occ.cut(
        [(2, outer_surf)],
        [(2, cap_surf)],
        removeObject=True,
        removeTool=True
    )

    gmsh.model.occ.synchronize()

    if not cut_result:
        gmsh.finalize()
        raise RuntimeError("Boolean cut produced no result – check cap is inside channel bounds.")

    fluid_surf_tag = cut_result[0][1]

    # ── Identify boundary curves by geometry ──────────────────────────────
    # After the boolean cut, we need to re-identify which curves are which.
    # We do this by sampling the midpoint of each 1D entity.

    all_curves = gmsh.model.getEntities(1)

    inlet_tags   = []
    outlet_tags  = []
    top_tags     = []
    bottom_tags  = []
    arc_tags_out = []

    tol = max(h_fluid * 0.1, 1e-10)

    # bounding box of the arc footprint on y≈channel_y0
    arc_x_min = float(np.min(arc_coords_m[:, 0]))
    arc_x_max = float(np.max(arc_coords_m[:, 0]))

    for (dim, tag) in all_curves:
        bbox = gmsh.model.getBoundingBox(dim, tag)
        xmid = 0.5 * (bbox[0] + bbox[3])
        ymid = 0.5 * (bbox[1] + bbox[4])

        if abs(xmid - channel_x0) < tol:
            inlet_tags.append(tag)
        elif abs(xmid - channel_x1) < tol:
            outlet_tags.append(tag)
        elif abs(ymid - channel_y1) < tol:
            top_tags.append(tag)
        elif abs(ymid - channel_y0) < tol:
            # bottom wall — distinguish between cap chord and outer bottom
            if arc_x_min - tol <= xmid <= arc_x_max + tol:
                # This is the cap chord (inside cap footprint) → skip (interior)
                pass
            else:
                bottom_tags.append(tag)
        else:
            # Anything else touching the cap region = the arc
            arc_tags_out.append(tag)

    # ── Physical groups ─────────────────────────────────────────────────────
    if inlet_tags:
        gmsh.model.addPhysicalGroup(1, inlet_tags,   tag=10)
    if outlet_tags:
        gmsh.model.addPhysicalGroup(1, outlet_tags,  tag=11)
    if top_tags:
        gmsh.model.addPhysicalGroup(1, top_tags,     tag=12)
    if arc_tags_out:
        gmsh.model.addPhysicalGroup(1, arc_tags_out, tag=13)
    if bottom_tags:
        gmsh.model.addPhysicalGroup(1, bottom_tags,  tag=14)

    gmsh.model.addPhysicalGroup(2, [fluid_surf_tag], tag=20)

    # ── Mesh refinement near arc + LOCALIZED corner field ─────────────────
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h_fluid)
    if arc_pts:
        gmsh.model.mesh.setSize([(0, p) for p in arc_pts], h_arc)
        # Refine the two base corners with a Distance+Threshold field CONFINED to
        # a small radius, instead of setSize on the endpoints. The endpoints are
        # the shared endpoints of the base chord AND the arc, so setSize there
        # refined the whole base edge and ~4x'd the mesh; a bounded field grades
        # h_corner → h_fluid within corner_radius, densifying only the corner
        # neighbourhood. gmsh takes min(field, point-sizes), so the arc keeps its
        # h_arc sizing.
        corner_radius = 6.0 * h_arc
        fdist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(fdist, "PointsList",
                                         [arc_pts[0], arc_pts[-1]])
        fthr = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(fthr, "InField", fdist)
        gmsh.model.mesh.field.setNumber(fthr, "SizeMin", h_corner)
        gmsh.model.mesh.field.setNumber(fthr, "SizeMax", h_fluid)
        gmsh.model.mesh.field.setNumber(fthr, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(fthr, "DistMax", corner_radius)
        gmsh.model.mesh.field.setAsBackgroundMesh(fthr)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)

    gmsh.model.mesh.generate(2)

    # A degenerate cap cutout (self-intersecting arc, or the arc touching a
    # channel wall) makes gmsh silently emit a mesh with ZERO 2D elements —
    # meshio/FEniCS then hard-crash in msh_to_fenics. Detect it here and raise
    # so the caller can fall back to the undeformed arc or skip the iteration
    # instead of losing a multi-hour run.
    _, tri_tags, _ = gmsh.model.mesh.getElements(2)
    n_tri = int(sum(len(t) for t in tri_tags))
    if n_tri == 0:
        gmsh.finalize()
        raise RuntimeError(
            "Fluid mesh generation produced 0 triangles (degenerate cap "
            "cutout — arc likely self-intersecting or touching a wall)."
        )

    gmsh.write(msh_path)
    gmsh.finalize()


# ─────────────────────────────────────────────────────────────────────────────
# meshio → FEniCS (triangles only, with physical tags)
# ─────────────────────────────────────────────────────────────────────────────

def msh_to_fenics(msh_path, length_scale=1.0):
    import meshio
    import numpy as np
    from dolfin import Mesh, MeshEditor, MeshFunction

    msh = meshio.read(msh_path)

    pts = np.asarray(msh.points, dtype=float)[:, :2] * float(length_scale)

    # ── collect ALL line and triangle blocks with their physical tags ──────
    line_cells_list = []
    line_tags_list  = []
    tri_cells_list  = []
    tri_tags_list   = []

    phys = msh.cell_data.get("gmsh:physical", [])

    for i, cb in enumerate(msh.cells):
        tag_arr = np.asarray(phys[i], dtype=np.int64) if i < len(phys) else None
        if cb.type == "line":
            line_cells_list.append(np.asarray(cb.data, dtype=np.int64))
            if tag_arr is not None:
                line_tags_list.append(tag_arr)
        elif cb.type == "triangle":
            tri_cells_list.append(np.asarray(cb.data, dtype=np.int64))
            if tag_arr is not None:
                tri_tags_list.append(tag_arr)

    if not tri_cells_list:
        raise RuntimeError("No triangle elements found in .msh")

    tri_cells = np.vstack(tri_cells_list)
    tri_data  = np.concatenate(tri_tags_list) if tri_tags_list else None

    line_cells = np.vstack(line_cells_list)  if line_cells_list else None
    line_data  = np.concatenate(line_tags_list) if line_tags_list else None

    # ── build dolfin Mesh ─────────────────────────────────────────────────
    mesh = Mesh()
    editor = MeshEditor()
    editor.open(mesh, "triangle", 2, 2)
    editor.init_vertices(int(pts.shape[0]))
    editor.init_cells(int(tri_cells.shape[0]))
    for i, (x, y) in enumerate(pts):
        editor.add_vertex(int(i), [float(x), float(y)])
    for c, (v0, v1, v2) in enumerate(tri_cells):
        editor.add_cell(int(c), [int(v0), int(v1), int(v2)])
    editor.close()
    mesh.init()
    mesh.init(1, 0); mesh.init(2, 0); mesh.init(1, 2)

    # ── cell tags ─────────────────────────────────────────────────────────
    cell_tags = MeshFunction("size_t", mesh, 2, 0)
    if tri_data is not None:
        for c, tag in enumerate(tri_data):
            cell_tags[int(c)] = int(tag)

    # ── facet tags from line elements ─────────────────────────────────────
    facet_tags = MeshFunction("size_t", mesh, 1, 0)
    if line_cells is not None and line_data is not None:
        from dolfin import facets as dolfin_facets, vertices as dolfin_vertices

        # build vertex-pair → facet index map
        vp_to_facet = {}
        for f in dolfin_facets(mesh):
            key = tuple(sorted(v.index() for v in dolfin_vertices(f)))
            vp_to_facet[key] = f.index()

        matched   = 0
        unmatched = 0
        for seg, tag in zip(line_cells, line_data):
            key = tuple(sorted(seg))
            if key in vp_to_facet:
                facet_tags[vp_to_facet[key]] = int(tag)
                matched += 1
            else:
                unmatched += 1

        print(f"  Facet tag matching: {matched} matched, {unmatched} unmatched")

    return mesh, cell_tags, facet_tags


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess worker (mesh-build watchdog)
# ─────────────────────────────────────────────────────────────────────────────
# gmsh can HANG (self-intersection infinite loop) inside mesh.generate(); a native
# C loop can't be interrupted by a Python signal, so the only robust timeout is to
# run the build in a separate, killable process. run_fsg_longterm invokes this via
#   python -c "import mesh_builder; mesh_builder._worker_main()" <config.json>
# with subprocess timeout; a hang → TimeoutExpired → the child is killed → caller
# falls back. Config JSON keys: kind ("fluid"|"solid"), msh_path, arc_npy, and the
# per-kind size/channel params.

def _worker_main():
    import sys, json
    import numpy as np
    with open(sys.argv[1]) as fh:
        p = json.load(fh)
    arc = np.load(p["arc_npy"])
    if p["kind"] == "fluid":
        build_fluid_mesh(p["msh_path"], arc,
                         p["channel_x0"], p["channel_x1"],
                         p["channel_y0"], p["channel_y1"],
                         p["h_fluid"], p.get("h_arc"), p.get("h_corner"))
    elif p["kind"] == "solid":
        build_solid_mesh_from_arc(p["msh_path"], arc,
                                  p["h_size"], p.get("h_corner"))
    else:
        raise ValueError(f"unknown mesh kind: {p['kind']}")


if __name__ == "__main__":
    _worker_main()