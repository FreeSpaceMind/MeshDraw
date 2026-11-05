"""Pure-NumPy mesh helpers the 3D views build and export with: revolved profiles, STL bytes, cut-aways, polygon
triangulation, swept outlines, boxes. Faces are VTK's flat ``[n, i0, ..., n, ...]`` layout throughout.
"""

from __future__ import annotations

import numpy as np


def revolved_mesh_arrays(
    z_profile,
    radius_profile,
    n_phi: int = 72,
    sweep_deg: float = 360.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Revolve a (z, r) polyline about the z axis into quad-mesh arrays.

    Returns ``(points, faces)`` — an ``(n, 3)`` float array and a flat
    int64 connectivity array in VTK polydata layout (``[4, a, b, c, d,
    4, ...]``) — or ``None`` for fewer than two valid samples. This is
    the exact construction UmbraAntennaDesigner revolves its committed
    feed with, kept as pure NumPy so it is testable without pyvista.

    ``sweep_deg`` below 360 makes a partial revolve (no wrap face), the
    cutaway view that shows the corrugations inside the horn.
    """
    z_profile = np.asarray(z_profile, dtype=float)
    radius_profile = np.asarray(radius_profile, dtype=float)
    n = min(len(z_profile), len(radius_profile))
    if n < 2:
        return None
    z_profile = z_profile[:n]
    radius_profile = np.maximum(radius_profile[:n], 0.0)
    valid = np.isfinite(z_profile) & np.isfinite(radius_profile)
    z_profile = z_profile[valid]
    radius_profile = radius_profile[valid]
    if len(z_profile) < 2:
        return None

    sweep = float(np.clip(sweep_deg, 1.0, 360.0))
    full = sweep >= 360.0 - 1e-9
    theta = np.linspace(0.0, np.deg2rad(sweep), int(n_phi),
                        endpoint=not full)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    points = []
    for z_val, r_val in zip(z_profile, radius_profile):
        points.extend(
            np.column_stack((
                r_val * cos_theta,
                r_val * sin_theta,
                np.full_like(theta, z_val),
            ))
        )
    points = np.asarray(points, dtype=float)

    faces = []
    n_ring = len(theta)
    n_profile = len(z_profile)
    last_j = n_ring if full else n_ring - 1
    for i in range(n_profile - 1):
        ring0 = i * n_ring
        ring1 = (i + 1) * n_ring
        for j in range(last_j):
            jp1 = (j + 1) % n_ring
            faces.extend([4, ring0 + j, ring0 + jp1, ring1 + jp1, ring1 + j])

    return points, np.asarray(faces, dtype=np.int64)


def quad_mesh_stl_bytes(meshes) -> bytes:
    """Binary STL for a list of ``(points, faces)`` quad meshes.

    Each VTK-layout quad becomes two triangles; facet normals come from
    the triangle winding. Pure NumPy — no extra dependency — so the 3D
    view's geometry can go straight into CAD or a slicer.
    """
    triangles = []
    for points, faces in meshes:
        if points is None or faces is None or len(faces) == 0:
            continue
        quads = np.asarray(faces, dtype=np.int64).reshape(-1, 5)[:, 1:]
        vertices = np.asarray(points, dtype=float)
        triangles.append(vertices[quads[:, [0, 1, 2]]])
        triangles.append(vertices[quads[:, [0, 2, 3]]])
    if not triangles:
        raise ValueError("no mesh data to export")
    tri = np.concatenate(triangles, axis=0)

    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.where(lengths > 1e-30, normals / np.maximum(lengths, 1e-30),
                       0.0)

    record = np.zeros(tri.shape[0], dtype=[
        ("normal", "<f4", 3),
        ("v0", "<f4", 3), ("v1", "<f4", 3), ("v2", "<f4", 3),
        ("attr", "<u2"),
    ])
    record["normal"] = normals
    record["v0"] = tri[:, 0]
    record["v1"] = tri[:, 1]
    record["v2"] = tri[:, 2]

    header = b"meshdraw revolved body (binary STL)".ljust(80, b"\0")
    count = np.uint32(tri.shape[0]).tobytes()
    return header + count + record.tobytes()


def vtk_faces_to_triangles(faces) -> np.ndarray:
    """``(n_tri, 3)`` vertex indices from a VTK face array of triangles, quads or polygons (fan split)."""
    faces = np.asarray(faces, dtype=np.int64).reshape(-1)
    tris = []
    i = 0
    while i < len(faces):
        n = int(faces[i])
        poly = faces[i + 1 : i + 1 + n]
        for k in range(1, n - 1):
            tris.append((poly[0], poly[k], poly[k + 1]))
        i += n + 1
    return np.asarray(tris, dtype=np.int64).reshape(-1, 3)


def cutaway_mesh(points, faces, center_xy=(0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """``(points, faces)`` without the polygons whose centroid lies in the quadrant ``x > cx, y > cy``.

    The cutaway of a posed body: a quarter section opened about the
    vertical line through ``center_xy`` (the chain axis), so the inside of
    every member shows at once. Polygons are kept or dropped whole; on a
    revolved mesh (``revolved_mesh_arrays``, 5 deg steps by default) the
    quadrant edges fall on mesh lines and the cut is clean.
    """
    points = np.asarray(points, dtype=float)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1)
    cx, cy = (float(v) for v in center_xy)
    kept = []
    i = 0
    while i < len(faces):
        n = int(faces[i])
        poly = faces[i + 1 : i + 1 + n]
        c = points[poly].mean(axis=0)
        if not (c[0] > cx and c[1] > cy):
            kept.append(faces[i : i + 1 + n])
        i += n + 1
    out = np.concatenate(kept) if kept else np.zeros(0, dtype=np.int64)
    return points, out


def mesh_stl_bytes(meshes) -> bytes:
    """Binary STL for ``(points, faces)`` meshes with any VTK polygon faces (triangles, quads, ...)."""
    triangles = []
    for points, faces in meshes:
        if points is None or faces is None or len(faces) == 0:
            continue
        tri = vtk_faces_to_triangles(faces)
        triangles.append(np.asarray(points, dtype=float)[tri])
    if not triangles:
        raise ValueError("no mesh data to export")
    tri = np.concatenate(triangles, axis=0)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.where(lengths > 1e-30, normals / np.maximum(lengths, 1e-30), 0.0)
    record = np.zeros(tri.shape[0], dtype=[("normal", "<f4", 3), ("v0", "<f4", 3), ("v1", "<f4", 3), ("v2", "<f4", 3), ("attr", "<u2")])
    record["normal"], record["v0"], record["v1"], record["v2"] = normals, tri[:, 0], tri[:, 1], tri[:, 2]
    header = b"meshdraw body (binary STL)".ljust(80, b"\0")
    return header + np.uint32(tri.shape[0]).tobytes() + record.tobytes()


def read_binary_stl(data: bytes) -> np.ndarray:
    """``(n_tri, 3, 3)`` vertices of a binary STL (the inverse of :func:`mesh_stl_bytes`, for tests and imports)."""
    n = int(np.frombuffer(data[80:84], dtype="<u4")[0])
    rec = np.frombuffer(data[84 : 84 + 50 * n], dtype=[("normal", "<f4", 3), ("v0", "<f4", 3), ("v1", "<f4", 3), ("v2", "<f4", 3), ("attr", "<u2")])
    return np.stack([rec["v0"], rec["v1"], rec["v2"]], axis=1).astype(float)


def triangulate_polygon(outline) -> np.ndarray:
    """``(n_tri, 3)`` vertex indices triangulating a simple polygon ``(k, 2)``, in the polygon's own orientation sense.

    Ear clipping (any simple polygon, convex or not; consecutive
    collinear or coincident vertices are dropped as zero-area ears), so a
    concave outline -- an arm pair's section, a ridged guide -- is covered
    by its own area only. A VTK polygon or a fan from one vertex of such
    an outline spans its concavities with sheets. Falls back to a fan if
    the polygon is not simple (self-crossing), which is the best a
    display can do with it.
    """
    pts = np.asarray(outline, dtype=float).reshape(-1, 2)
    n = len(pts)
    if n < 3:
        return np.zeros((0, 3), dtype=np.int64)
    x, y = pts[:, 0], pts[:, 1]
    area2 = float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    flip = area2 < 0
    idx = list(range(n))[::-1] if flip else list(range(n))
    scale = float(np.ptp(pts, axis=0).max()) or 1.0
    eps = 1e-12 * scale * scale
    tris = []
    remaining = np.ones(n, dtype=bool)
    i, misses = 0, 0
    while len(idx) > 3:
        m = len(idx)
        a, b, c = idx[(i - 1) % m], idx[i], idx[(i + 1) % m]
        pa, pb, pc = pts[a], pts[b], pts[c]
        cross = (pb[0] - pa[0]) * (pc[1] - pb[1]) - (pb[1] - pa[1]) * (pc[0] - pb[0])
        if abs(cross) <= eps:  # collinear (or a coincident / spike vertex): no area to cover, drop b
            remaining[b] = False
            idx.pop(i)
            i %= len(idx)
            misses = 0
            continue
        if cross > 0:  # a convex corner: an ear when no other remaining vertex lies in its triangle
            remaining[[a, b, c]] = False
            others = pts[remaining]
            remaining[[a, b, c]] = True
            if len(others) == 0 or not _any_in_triangle(others, pa, pb, pc):
                tris.append((a, b, c))
                remaining[b] = False
                idx.pop(i)
                i %= len(idx)
                misses = 0
                continue
        i = (i + 1) % m
        misses += 1
        if misses > m:  # not a simple polygon: fan the rest
            tris.extend((idx[0], idx[k], idx[k + 1]) for k in range(1, m - 1))
            idx = []
            break
    if len(idx) == 3:
        tris.append(tuple(idx))
    out = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    if flip:  # back to the polygon's own sense
        out = out[:, [0, 2, 1]]
    return out


def _any_in_triangle(points, pa, pb, pc, tol: float = 1e-9) -> bool:
    """Whether any of ``points`` lies in the closed triangle ``(pa, pb, pc)`` (barycentric, vectorised)."""
    v0, v1 = pc - pa, pb - pa
    v2 = points - pa
    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
    d20, d21 = v2 @ v0, v2 @ v1
    den = d00 * d11 - d01 * d01
    if abs(den) < 1e-300:
        return False
    u = (d11 * d20 - d01 * d21) / den
    v = (d00 * d21 - d01 * d20) / den
    return bool(np.any((u >= -tol) & (v >= -tol) & (u + v <= 1.0 + tol)))


def swept_mesh(origins, xhats, yhats, outline, cap_ends: bool = False):
    """A tube: ``outline`` (closed loop in each station's (x-hat, y-hat)) placed at every station, neighbours joined by quads.

    ``cap_ends`` closes both ends (a solid's look; a guide's wall stays
    open so its ports show) with the outline's triangulation
    (:func:`triangulate_polygon`): one polygon per cap drew a concave
    outline with sheets across its re-entrant parts. Returns ``(points,
    VTK faces)``.
    """
    o = np.asarray(origins, float)
    x = np.asarray(xhats, float)
    y = np.asarray(yhats, float)
    ol = np.asarray(outline, float)
    k, m = len(ol), len(o)
    pts = (o[:, None, :] + ol[None, :, 0, None] * x[:, None, :] + ol[None, :, 1, None] * y[:, None, :]).reshape(-1, 3)
    faces = []
    for i in range(m - 1):
        for j in range(k):
            jn = (j + 1) % k
            faces.extend((4, i * k + j, i * k + jn, (i + 1) * k + jn, (i + 1) * k + j))
    if cap_ends:
        tri = triangulate_polygon(ol)
        base = (m - 1) * k
        for a, b, c in tri:
            faces.extend((3, int(c), int(b), int(a)))  # the first cap faces backwards along the sweep
            faces.extend((3, base + int(a), base + int(b), base + int(c)))
    return pts, np.asarray(faces, dtype=np.int64)


def box_mesh(lo, hi):
    """A closed axis-aligned box from corner ``lo`` to ``hi`` (8 points, 6 quads)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    pts = np.array([[(lo[0], hi[0])[i & 1], (lo[1], hi[1])[(i >> 1) & 1], (lo[2], hi[2])[(i >> 2) & 1]] for i in range(8)])
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return pts, np.array([v for q in quads for v in (4, *q)], dtype=np.int64)


__all__ = ["box_mesh", "cutaway_mesh", "revolved_mesh_arrays", "quad_mesh_stl_bytes", "swept_mesh",
           "triangulate_polygon", "vtk_faces_to_triangles", "mesh_stl_bytes", "read_binary_stl"]
