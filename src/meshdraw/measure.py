"""Measuring the 3D view's geometry the SolidWorks way: pick faces, edges and vertices, read distances, angles and sizes.

The view draws surface meshes (posed device bodies, revolved horn walls),
not CAD topology, so :class:`MeshTopology` rebuilds the topology a CAD
measure works on from a mesh: coincident points are welded; triangles
are grown into FACES across every mesh edge whose dihedral angle is below
``feature_angle`` (25 deg, the angle the view draws its dark feature
lines at, so a face is what the eye sees bounded by lines); the mesh edges
between two faces, or on a boundary, are chained into EDGES -- a chain
broken wherever the pair of faces it separates changes -- and the chains'
ends are the VERTICES. Each face is classified as a plane (normal, offset),
a cylinder (axis, radius: from its normals and a circle fit) or a surface;
each edge as a line, a circle (centre, radius) or a curve.

:func:`measure` reports one or two entities: the minimum distance between
any two (exact on the tessellation: vertex-triangle and segment-segment
pairs, with the two closest points and their dx, dy, dz), the normal
distance between parallel planes or from a vertex to a plane, the angle
between planes / lines, the centre distance of cylinders and circles, a
face's area, an edge's length and diameter, a vertex's coordinates. A
curved face is its facets, so a distance to it is to the drawn surface
(within the view's chord error). Pure NumPy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

#: faces are split where neighbouring triangles turn by more than this [deg] (the view's feature-line angle)
FEATURE_ANGLE_DEG = 25.0
#: directions this close (1 - |cos|) count as parallel
PARALLEL_TOL = 1e-6


@dataclass(frozen=True)
class Entity:
    """A picked entity: ``kind`` face / edge / vertex, the mesh it is on, its index there."""

    kind: str
    mesh: int
    index: int


def _cells(faces: Sequence[int]) -> List[NDArray]:
    """The point indices of each VTK cell in a flat ``[n, i0, ..., n, ...]`` array."""
    f = np.asarray(faces, dtype=np.int64)
    out, k = [], 0
    while k < len(f):
        n = int(f[k])
        out.append(f[k + 1:k + 1 + n])
        k += n + 1
    return out


class MeshTopology:
    """Faces, edges and vertices of a surface mesh (see the module docstring). ``faces`` is a flat VTK cell array."""

    def __init__(self, points, faces, feature_angle_deg: float = FEATURE_ANGLE_DEG) -> None:
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        cells = _cells(faces)
        size = float(np.ptp(pts, axis=0).max()) if len(pts) else 1.0
        self.tol = max(1e-9 * max(size, 1e-9), 1e-15)
        # weld coincident points (meshes are often written cell by cell)
        key = np.round(pts / (self.tol * 10)).astype(np.int64)
        _uniq, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)
        self.points = pts[first]
        tris, cell_of = [], []
        for c, cell in enumerate(cells):
            idx = inverse[cell]
            for j in range(1, len(idx) - 1):  # a fan per cell (planar quads and polygons)
                t = (idx[0], idx[j], idx[j + 1])
                if len(set(t)) == 3:
                    tris.append(t)
                    cell_of.append(c)
        self.tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
        self.cell_of_tri = np.asarray(cell_of, dtype=np.int64)
        self.tri_of_cell: Dict[int, int] = {}
        for t, c in enumerate(cell_of):
            self.tri_of_cell.setdefault(int(c), t)
        p = self.points[self.tris]
        cr = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        area2 = np.linalg.norm(cr, axis=1)
        self.tri_area = 0.5 * area2
        self.tri_normal = np.where(area2[:, None] > 0, cr / np.maximum(area2, 1e-300)[:, None], 0.0)
        if len(p) and float(np.einsum("ij,ij->i", p[:, 0], cr).sum()) < 0:
            self.tri_normal = -self.tri_normal  # a closed mesh wound inward: its normals point out (a plane's normal reads outward)
        self._grow(math.cos(math.radians(feature_angle_deg)))
        self._chain()

    # -- faces ----------------------------------------------------------------------------------------------
    def _grow(self, cos_limit: float) -> None:
        nt = len(self.tris)
        edge_tris: Dict[Tuple[int, int], List[int]] = {}
        for t, (a, b, c) in enumerate(self.tris):
            for i, j in ((a, b), (b, c), (c, a)):
                edge_tris.setdefault((min(i, j), max(i, j)), []).append(t)
        self._edge_tris = edge_tris
        face = np.full(nt, -1, dtype=np.int64)
        nb: List[List[int]] = [[] for _ in range(nt)]
        for ts in edge_tris.values():
            if len(ts) == 2:
                a, b = ts
                # either winding: a mesh written cell by cell need not orient its cells alike
                if abs(float(self.tri_normal[a] @ self.tri_normal[b])) >= cos_limit:
                    nb[a].append(b)
                    nb[b].append(a)
        k = 0
        for s in range(nt):
            if face[s] >= 0:
                continue
            stack = [s]
            face[s] = k
            while stack:
                t = stack.pop()
                for u in nb[t]:
                    if face[u] < 0:
                        face[u] = k
                        if float(self.tri_normal[u] @ self.tri_normal[t]) < 0:  # one face, one side: its normals agree
                            self.tri_normal[u] = -self.tri_normal[u]
                        stack.append(u)
            k += 1
        self.face_of_tri = face
        self.faces: List[NDArray] = [np.flatnonzero(face == f) for f in range(k)]

    # -- edges and vertices ---------------------------------------------------------------------------------------
    def _chain(self) -> None:
        sides: Dict[Tuple[int, int], Tuple[int, ...]] = {}
        for e, ts in self._edge_tris.items():
            fs = tuple(sorted({int(self.face_of_tri[t]) for t in ts}))
            if len(ts) == 1 or len(fs) > 1:
                sides[e] = fs if len(ts) > 1 else (fs[0], -1)
        at: Dict[int, List[Tuple[int, int]]] = {}
        for e in sides:
            at.setdefault(e[0], []).append(e)
            at.setdefault(e[1], []).append(e)
        # a vertex: where the sharp edges do not simply continue (not 2 of them), or the faces they part change
        corner = {p for p, es in at.items() if len(es) != 2 or sides[es[0]] != sides[es[1]]}
        used = set()
        edges: List[NDArray] = []
        edge_faces: List[Tuple[int, ...]] = []
        for e0 in sides:
            if e0 in used:
                continue
            used.add(e0)
            chain = [e0[0], e0[1]]
            for forward in (True, False):  # walk both ways from the first segment until a vertex or the start
                while True:
                    tip = chain[-1] if forward else chain[0]
                    if tip in corner or (len(chain) > 2 and chain[0] == chain[-1]):
                        break
                    nxt = [e for e in at[tip] if e not in used and sides[e] == sides[e0]]
                    if not nxt:
                        break
                    e = nxt[0]
                    used.add(e)
                    other = e[1] if e[0] == tip else e[0]
                    if forward:
                        chain.append(other)
                    else:
                        chain.insert(0, other)
            edges.append(np.asarray(chain, dtype=np.int64))
            edge_faces.append(sides[e0])
        self.edges = edges
        self.edge_faces = edge_faces
        ends = set()
        for ch in edges:
            if ch[0] != ch[-1]:
                ends |= {int(ch[0]), int(ch[-1])}
        self.vertices = np.asarray(sorted(ends | corner), dtype=np.int64)

    # -- geometry of one entity ------------------------------------------------------------------------------------
    def face_points(self, f: int) -> NDArray:
        return self.points[np.unique(self.tris[self.faces[f]])]

    def edge_points(self, e: int) -> NDArray:
        return self.points[self.edges[e]]

    def face_info(self, f: int) -> Dict[str, object]:
        """``{kind: plane | cylinder | surface, area, ...}``: a plane's ``normal`` and a ``point``; a cylinder's ``axis``,
        ``center`` (a point on the axis) and ``radius``."""
        ts = self.faces[f]
        area = float(self.tri_area[ts].sum())
        nrm = self.tri_normal[ts]
        w = self.tri_area[ts]
        mean = (nrm * w[:, None]).sum(axis=0)
        pts = self.face_points(f)
        if np.linalg.norm(mean) > 0 and np.all(nrm @ (mean / np.linalg.norm(mean)) > 1 - 1e-9):
            n = mean / np.linalg.norm(mean)
            return {"kind": "plane", "area": area, "normal": n, "point": pts.mean(axis=0)}
        if len(ts) >= 3:
            _u, s, vt = np.linalg.svd(nrm, full_matrices=False)
            axis = vt[-1]
            if float(np.max(np.abs(nrm @ axis))) < 0.02:  # every normal square to one direction: a cylinder
                fit = _circle_fit(pts, axis)
                if fit is not None:
                    c, r, resid = fit
                    if resid < 0.02 * r:
                        return {"kind": "cylinder", "area": area, "axis": axis, "center": c, "radius": r}
        return {"kind": "surface", "area": area}

    def edge_info(self, e: int) -> Dict[str, object]:
        """``{kind: line | circle | curve, length, ...}``: a line's ``direction``; a circle's ``center``, ``radius`` and
        ``normal`` (an arc is a circle too)."""
        p = self.edge_points(e)
        length = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
        d = p[-1] - p[0]
        c0 = p.mean(axis=0)
        _u, s, vt = np.linalg.svd(p - c0, full_matrices=False)
        span = max(float(s[0]), 1e-300)
        closed = len(p) > 3 and np.linalg.norm(p[-1] - p[0]) <= self.tol
        if len(p) == 2 or (len(s) > 1 and s[1] < 1e-7 * span):
            dd = vt[0] if np.linalg.norm(d) < self.tol else d / np.linalg.norm(d)
            return {"kind": "line", "length": length, "direction": dd, "start": p[0], "end": p[-1]}
        if len(p) >= 4 and (len(s) < 3 or s[2] < 1e-7 * span):
            fit = _circle_fit(p, vt[2])
            if fit is not None and fit[2] < 1e-3 * fit[1]:
                # a whole circle's true circumference (the chords' sum is short of it by the view's facetting)
                return {"kind": "circle", "length": 2 * math.pi * fit[1] if closed else length, "center": fit[0], "radius": fit[1],
                        "normal": vt[2], "closed": closed}
        return {"kind": "curve", "length": length}

    def primitives(self, ent: Entity) -> Tuple[NDArray, NDArray, NDArray]:
        """``(points (k, 3), segments (m, 2, 3), triangles (t, 3, 3))`` an entity is made of, for distances."""
        if ent.kind == "vertex":
            return self.points[[int(self.vertices[ent.index])]], np.zeros((0, 2, 3)), np.zeros((0, 3, 3))
        if ent.kind == "edge":
            p = self.edge_points(ent.index)
            return p, np.stack([p[:-1], p[1:]], axis=1), np.zeros((0, 3, 3))
        tri = self.points[self.tris[self.faces[ent.index]]]
        segs = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
        return self.face_points(ent.index), segs, tri

    # -- picking ---------------------------------------------------------------------------------------------
    def pick(self, cell: int, at: Sequence[float], tol: float) -> Entity:
        """What a click on cell ``cell`` at ``at`` [m] means: the nearest vertex within ``tol`` [m], else the nearest edge
        within it, else the cell's face (mesh index 0: the caller sets its own)."""
        q = np.asarray(at, float)
        if len(self.vertices):
            dv = np.linalg.norm(self.points[self.vertices] - q, axis=1)
            k = int(np.argmin(dv))
            if dv[k] <= tol:
                return Entity("vertex", 0, k)
        best, best_d = None, tol
        for e, ch in enumerate(self.edges):
            p = self.points[ch]
            d = _point_segments(q, p[:-1], p[1:])[0]
            if d <= best_d:
                best, best_d = e, d
        if best is not None:
            return Entity("edge", 0, best)
        t = self.tri_of_cell.get(int(cell))
        if t is None:
            t = int(np.argmin(np.linalg.norm(self.points[self.tris].mean(axis=1) - q, axis=1)))
        return Entity("face", 0, int(self.face_of_tri[t]))


def _circle_fit(pts: NDArray, normal: NDArray) -> Optional[Tuple[NDArray, float, float]]:
    """``(centre (3,), radius, rms residual)`` of the circle best through ``pts`` projected on the plane square to
    ``normal`` (an algebraic fit), or None."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    o = pts.mean(axis=0)
    x, y = (pts - o) @ u, (pts - o) @ v
    if len(x) < 3:
        return None
    M = np.column_stack([x, y, np.ones_like(x)])
    sol, *_ = np.linalg.lstsq(M, x * x + y * y, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r2 = sol[2] + cx * cx + cy * cy
    if not r2 > 0:
        return None
    r = math.sqrt(r2)
    resid = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)))
    return o + cx * u + cy * v + float(((pts - o) @ n).mean()) * n, r, resid


# -- distances (vectorized) -----------------------------------------------------------------------------------------


def _point_segments(q: NDArray, a: NDArray, b: NDArray) -> Tuple[float, NDArray]:
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", q - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-300), 0.0, 1.0)
    c = a + t[:, None] * ab
    d = np.linalg.norm(c - q, axis=1)
    k = int(np.argmin(d))
    return float(d[k]), c[k]


def _points_triangles(P: NDArray, T: NDArray) -> Tuple[float, NDArray, NDArray]:
    """Min distance from points ``P`` (k, 3) to triangles ``T`` (t, 3, 3): ``(d, the point, the closest point on T)``."""
    best = (math.inf, None, None)
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    n = np.cross(b - a, c - a)
    nn = np.einsum("ij,ij->i", n, n)
    for start in range(0, len(P), 256):
        Q = P[start:start + 256][:, None, :]  # (q, 1, 3)
        # projection into the plane and its barycentric coordinates
        w = Q - a[None]
        dist_plane = np.einsum("qtj,tj->qt", w, n) / np.maximum(nn, 1e-300)[None]
        proj = Q - dist_plane[..., None] * n[None]
        v0, v1 = b - a, c - a
        v2 = proj - a[None]
        d00, d01, d11 = np.einsum("ij,ij->i", v0, v0), np.einsum("ij,ij->i", v0, v1), np.einsum("ij,ij->i", v1, v1)
        d20, d21 = np.einsum("qtj,tj->qt", v2, v0), np.einsum("qtj,tj->qt", v2, v1)
        den = np.maximum(d00 * d11 - d01 * d01, 1e-300)[None]
        vv = (d11[None] * d20 - d01[None] * d21) / den
        ww = (d00[None] * d21 - d01[None] * d20) / den
        inside = (vv >= 0) & (ww >= 0) & (vv + ww <= 1)
        dd = np.where(inside, np.linalg.norm(proj - Q, axis=2), np.inf)
        k = np.unravel_index(int(np.argmin(dd)), dd.shape)
        if dd[k] < best[0]:
            best = (float(dd[k]), Q[k[0], 0], proj[k])
    # the triangles' edges cover a foot outside every triangle
    segs = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    for p in P:
        d, cpt = _point_segments(p, segs[:, 0], segs[:, 1])
        if d < best[0]:
            best = (d, p, cpt)
    return best


def _segments_segments(A: NDArray, B: NDArray) -> Tuple[float, NDArray, NDArray]:
    """Min distance between segment sets ``A`` (m, 2, 3) and ``B`` (n, 2, 3), with the two closest points."""
    best = (math.inf, None, None)
    for start in range(0, len(A), 256):
        a0, a1 = A[start:start + 256, 0][:, None], A[start:start + 256, 1][:, None]
        b0, b1 = B[None, :, 0], B[None, :, 1]
        d1, d2, r = a1 - a0, b1 - b0, a0 - b0
        a = np.einsum("ijk,ijk->ij", d1, d1)
        e = np.einsum("ijk,ijk->ij", d2, d2)
        f = np.einsum("ijk,ijk->ij", d2, r)
        c = np.einsum("ijk,ijk->ij", d1, r)
        bb = np.einsum("ijk,ijk->ij", d1, d2)
        den = a * e - bb * bb
        s = np.where(den > 1e-300, np.clip((bb * f - c * e) / np.where(den > 1e-300, den, 1.0), 0, 1), 0.0)
        t = (bb * s + f) / np.maximum(e, 1e-300)
        t_c = np.clip(t, 0, 1)
        s = np.where((t < 0) | (t > 1), np.clip((bb * t_c - c) / np.maximum(a, 1e-300), 0, 1), s)
        t = t_c
        pa = a0 + s[..., None] * d1
        pb = b0 + t[..., None] * d2
        dd = np.linalg.norm(pa - pb, axis=2)
        k = np.unravel_index(int(np.argmin(dd)), dd.shape)
        if dd[k] < best[0]:
            best = (float(dd[k]), pa[k], pb[k])
    return best


def min_distance(a: Tuple[NDArray, NDArray, NDArray], b: Tuple[NDArray, NDArray, NDArray]) -> Tuple[float, NDArray, NDArray]:
    """The minimum distance between two entities given as :meth:`MeshTopology.primitives`: ``(d, point on a, point on b)``.
    Exact for flat-faced geometry: the closest pair is a vertex against a triangle or a segment against a segment."""
    (pa, sa, ta), (pb, sb, tb) = a, b
    cands = []
    if len(pa) and len(pb):
        diff = pa[:, None] - pb[None]
        dd = np.linalg.norm(diff, axis=2)
        i, j = np.unravel_index(int(np.argmin(dd)), dd.shape)
        cands.append((float(dd[i, j]), pa[i], pb[j]))
    if len(ta) and len(pb):
        d, p, q = _points_triangles(pb, ta)
        cands.append((d, q, p))
    if len(tb) and len(pa):
        cands.append(_points_triangles(pa, tb))
    if len(sa) and len(sb):
        cands.append(_segments_segments(sa, sb))
    if len(sa) and len(pb):
        for p in pb:
            d, c = _point_segments(p, sa[:, 0], sa[:, 1])
            cands.append((d, c, p))
    if len(sb) and len(pa):
        for p in pa:
            d, c = _point_segments(p, sb[:, 0], sb[:, 1])
            cands.append((d, p, c))
    return min(cands, key=lambda x: x[0])


# -- reports ----------------------------------------------------------------------------------------------------


def _formatter(unit: Tuple[str, float]) -> Callable[[float], str]:
    label, factor = unit

    def fmt(v: float) -> str:
        x = float(v) * factor
        return f"{0.0 if abs(x) < 1e-9 else x:.6g} {label}"
    return fmt


def _angle(u: NDArray, v: NDArray, lines: bool = False) -> float:
    c = float(np.clip(np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-300), -1.0, 1.0))
    a = math.degrees(math.acos(c))
    return min(a, 180.0 - a) if lines else a


def info(topo: MeshTopology, ent: Entity) -> Dict[str, object]:
    if ent.kind == "face":
        return topo.face_info(ent.index)
    if ent.kind == "edge":
        return topo.edge_info(ent.index)
    return {"kind": "vertex", "point": topo.points[int(topo.vertices[ent.index])]}


def measure(topos: Sequence[MeshTopology], ents: Sequence[Entity], unit: Tuple[str, float] = ("mm", 1e3)) -> Dict[str, object]:
    """What the measure tool reports for one or two picked entities: ``{"lines": [text], "closest": (p, q) or None}``.

    One entity: a face's area (a plane's normal, a cylinder's diameter), an edge's length (a circle's diameter), a vertex's
    coordinates. Two: the minimum distance and its dx / dy / dz, plus -- where they apply -- the normal distance (parallel
    planes, a vertex and a plane), the angle (two planes, two lines, a line and a plane), the centre distance (two
    cylinders or circles: their axes / centres). ``unit``: the display unit's label and its factor from metres."""
    fmt = _formatter(unit)
    lines: List[str] = []
    closest = None
    infos = [info(topos[e.mesh], e) for e in ents]
    names = {"face": "Face", "edge": "Edge", "vertex": "Vertex"}

    def label(e: Entity, inf) -> str:
        kind = inf["kind"]
        return f"{names[e.kind]} ({kind})" if e.kind != "vertex" else "Vertex"

    if len(ents) == 1:
        e, inf = ents[0], infos[0]
        lines.append(label(e, inf))
        if e.kind == "face":
            lines.append(f"Area: {inf['area'] * unit[1] ** 2:.6g} {unit[0]}²")
            if inf["kind"] == "plane":
                n = inf["normal"]
                lines.append(f"Normal: ({n[0]:.4g}, {n[1]:.4g}, {n[2]:.4g})")
            elif inf["kind"] == "cylinder":
                a = inf["axis"]
                lines.append(f"Diameter: {fmt(2 * inf['radius'])}  (radius {fmt(inf['radius'])})")
                lines.append(f"Axis: ({a[0]:.4g}, {a[1]:.4g}, {a[2]:.4g})")
        elif e.kind == "edge":
            lines.append(f"Length: {fmt(inf['length'])}")
            if inf["kind"] == "circle":
                lines.append(f"Diameter: {fmt(2 * inf['radius'])}  (radius {fmt(inf['radius'])})")
                c = inf["center"]
                lines.append(f"Centre: ({fmt(c[0])}, {fmt(c[1])}, {fmt(c[2])})")
            elif inf["kind"] == "line":
                d = inf["direction"]
                lines.append(f"Direction: ({d[0]:.4g}, {d[1]:.4g}, {d[2]:.4g})")
        else:
            p = inf["point"]
            lines.append(f"X {fmt(p[0])}, Y {fmt(p[1])}, Z {fmt(p[2])}")
        return {"lines": lines, "closest": None}
    if len(ents) != 2:
        return {"lines": ["Pick one or two faces, edges or vertices"], "closest": None}
    (ea, eb), (ia, ib) = ents, infos
    prims = [topos[e.mesh].primitives(e) for e in ents]
    d, p, q = min_distance(prims[0], prims[1])
    closest = (p, q)
    lines.append(f"{label(ea, ia)} to {label(eb, ib)}")
    lines.append(f"Distance: {fmt(d)}")
    dv = q - p
    lines.append(f"dX {fmt(abs(dv[0]))}, dY {fmt(abs(dv[1]))}, dZ {fmt(abs(dv[2]))}")
    kinds = (ia["kind"], ib["kind"])
    if kinds == ("plane", "plane"):
        na, nb = ia["normal"], ib["normal"]
        if 1 - abs(float(na @ nb)) < PARALLEL_TOL:
            lines.append(f"Normal distance: {fmt(abs(float(na @ (ib['point'] - ia['point']))))} (parallel)")
        else:
            lines.append(f"Angle: {_angle(na, nb, lines=True):.6g}°")
    elif "plane" in kinds and "vertex" in kinds:
        pl, vx = (ia, ib) if kinds[0] == "plane" else (ib, ia)
        lines.append(f"Normal distance: {fmt(abs(float(pl['normal'] @ (vx['point'] - pl['point']))))}")
    elif kinds == ("line", "line"):
        lines.append(f"Angle: {_angle(ia['direction'], ib['direction'], lines=True):.6g}°")
    elif "plane" in kinds and "line" in kinds:
        pl, ln = (ia, ib) if kinds[0] == "plane" else (ib, ia)
        lines.append(f"Angle to the plane: {90.0 - _angle(pl['normal'], ln['direction'], lines=True):.6g}°")
        if abs(float(pl["normal"] @ ln["direction"])) < 1e-6:
            lines.append(f"Normal distance: {fmt(abs(float(pl['normal'] @ (ln['start'] - pl['point']))))} (parallel)")
    elif kinds == ("cylinder", "cylinder"):
        aa, ab = ia["axis"], ib["axis"]
        if 1 - abs(float(aa @ ab)) < 1e-4:
            r = ib["center"] - ia["center"]
            lines.append(f"Centre distance: {fmt(float(np.linalg.norm(r - (r @ aa) * aa)))} (parallel axes)")
        else:
            lines.append(f"Axis angle: {_angle(aa, ab, lines=True):.6g}°")
    elif kinds == ("circle", "circle"):
        lines.append(f"Centre distance: {fmt(float(np.linalg.norm(ib['center'] - ia['center'])))}")
    elif "cylinder" in kinds and "plane" in kinds:
        cy, pl = (ia, ib) if kinds[0] == "cylinder" else (ib, ia)
        if abs(float(cy["axis"] @ pl["normal"])) < 1e-4:
            lines.append(f"Axis to plane: {fmt(abs(float(pl['normal'] @ (cy['center'] - pl['point']))))}")
    return {"lines": lines, "closest": closest}


__all__ = ["Entity", "FEATURE_ANGLE_DEG", "MeshTopology", "info", "measure", "min_distance"]
