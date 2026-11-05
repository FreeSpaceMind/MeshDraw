"""Section planes for the 3D views: where they sit, what they cut away, the caps and outlines they leave (no Qt).

A view holds up to :data:`MAX_PLANES` section planes. Each keeps the side
its normal points to (``n . (x - origin) >= 0``); the scene shows the UNION
of what the planes keep -- CAD's section view: one plane removes a half,
two remove a quarter (the old Cutaway), three an octant. A mesh is drawn as
one piece per plane, piece ``i`` clipped to keep plane ``i``'s side and the
CUT sides of planes ``0 .. i-1`` (``pieces``): the pieces are disjoint and
together are the union, using only the renderer's clipping planes (cheap
enough to follow a slider live).

Every plane's trace on every visible mesh is drawn as a dark outline
(``outline``). Where the traces on the surfaces that bound METAL (walls,
exteriors, a radiating part's metal: not a STEP part's air) close into
loops, the loops are filled (``cap``), so a solid reads as solid and a
hollow stays hollow.
Both are drawn on plane ``i`` and limited to where it bounds the removed
region: the cut sides of the other planes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

MAX_PLANES = 3
#: a trace whose ends are this close (a fraction of its extent) is closed for filling: the seams drawn meshes leave open
CLOSE_GAP = 0.05
AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


@dataclass
class SectionPlane:
    """One plane: on an axis (``x`` / ``y`` / ``z``) or ``free`` (``normal``), at ``fraction`` of the scene's extent along it
    (0..1), removing the side its axis points to unless ``flip``."""

    enabled: bool = False
    axis: str = "x"
    fraction: float = 0.5
    flip: bool = False
    normal: Tuple[float, float, float] = (1.0, 0.0, 0.0)

    def base(self) -> np.ndarray:
        """The unit direction the plane's position is measured along (its axis, or its free normal)."""
        n = np.asarray(AXES.get(self.axis, self.normal), float)
        return n / (np.linalg.norm(n) or 1.0)

    def place(self, bounds: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """``(origin, normal)`` in the scene ``bounds`` (xmin, xmax, ymin, ymax, zmin, zmax): ``fraction`` of the way along
        ``base`` across the scene. The plane REMOVES the side ``base`` points to (toward +X for an X plane: the opening faces
        the default iso view, which looks from +X +Y +Z), so the normal returned (the KEPT side) is ``-base``; ``flip``
        removes the other side."""
        b = self.base()
        lo, hi = extent_along(bounds, b)
        c = box_centre(bounds)
        t = lo + float(np.clip(self.fraction, 0.0, 1.0)) * (hi - lo)
        return c + (t - float(c @ b)) * b, (b if self.flip else -b)

    def set_from(self, origin, normal, bounds: Sequence[float]) -> None:
        """Become the ``free`` plane through ``origin`` with ``normal`` (what the drag handle reports)."""
        n = np.asarray(normal, float)
        n = n / (np.linalg.norm(n) or 1.0)  # the handle's normal points at the side it removes
        self.axis, self.flip, self.normal = "free", False, tuple(float(v) for v in n)
        lo, hi = extent_along(bounds, n)
        self.fraction = float(np.clip((float(np.asarray(origin, float) @ n) - lo) / ((hi - lo) or 1.0), 0.0, 1.0))
        self.enabled = True


def box_centre(bounds: Sequence[float]) -> np.ndarray:
    b = np.asarray(bounds, float).reshape(3, 2)
    return b.mean(axis=1)


def extent_along(bounds: Sequence[float], direction) -> Tuple[float, float]:
    """The scene box's extent along ``direction``: (min, max) of ``x . d`` over its corners."""
    b = np.asarray(bounds, float).reshape(3, 2)
    d = np.asarray(direction, float)
    d = d / (np.linalg.norm(d) or 1.0)
    corners = np.array([[x, y, z] for x in b[0] for y in b[1] for z in b[2]])
    proj = corners @ d
    return float(proj.min()), float(proj.max())


@dataclass
class Section:
    """The view's planes, placed in its scene's bounds."""

    planes: List[SectionPlane] = field(default_factory=lambda: [SectionPlane(axis=a) for a in ("x", "y", "z")][:MAX_PLANES])

    def active(self, bounds) -> List[Tuple[np.ndarray, np.ndarray]]:
        """``(origin, normal)`` of every enabled plane, in order."""
        return [p.place(bounds) for p in self.planes if p.enabled]

    def any(self) -> bool:
        return any(p.enabled for p in self.planes)


def pieces(active: Sequence[Tuple[np.ndarray, np.ndarray]]) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
    """The clipping planes of each disjoint piece of a mesh (``[(origin, normal), ...]`` per piece; each keeps its normal's
    side): piece ``i`` keeps plane ``i``'s side and the cut side of every plane before it. Their union is everything any
    plane keeps. No plane: one unclipped piece."""
    if not active:
        return [[]]
    out = []
    for i, (o, n) in enumerate(active):
        out.append([(np.asarray(o, float), np.asarray(n, float))] + [(np.asarray(oj, float), -np.asarray(nj, float))
                                                                      for oj, nj in active[:i]])
    return out


def kept(points, active) -> np.ndarray:
    """Which ``points`` the section shows (on the kept side of at least one plane; all of them with no plane)."""
    p = np.asarray(points, float).reshape(-1, 3)
    if not active:
        return np.ones(len(p), bool)
    keep = np.zeros(len(p), bool)
    for o, n in active:
        keep |= (p - np.asarray(o, float)) @ np.asarray(n, float) >= 0.0
    return keep


def cap_limits(active, i: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """The clipping planes of plane ``i``'s cap / outline: the CUT sides of the other planes (where plane ``i`` bounds what
    was removed)."""
    return [(np.asarray(o, float), -np.asarray(n, float)) for j, (o, n) in enumerate(active) if j != i]


def outline(points, faces, origin, normal):
    """The plane's trace on a mesh: a PolyData of line segments (None when it misses)."""
    import pyvista as pv

    mesh = pv.PolyData(np.asarray(points, float), np.asarray(faces, np.int64))
    try:
        s = mesh.slice(normal=tuple(np.asarray(normal, float)), origin=tuple(np.asarray(origin, float)))
    except Exception:
        return None
    return s if s.n_points else None


def cap(traces):
    """The cut face where a plane crosses solid metal: the plane's traces on every surface that bounds metal (a horn's wall
    and its exterior meet at the aperture rim; a radiating part's metal is closed alone) merged, and the loops that close
    filled (holes kept: VTK's contour triangulator). Open traces (a zero-thickness wall) fill nothing. None when nothing
    closes."""
    import vtk
    import pyvista as pv

    traces = [t for t in traces if t is not None and t.n_points]
    if not traces:
        return None
    merged = traces[0].merge(traces[1:]) if len(traces) > 1 else traces[0]
    size = float(np.ptp(np.asarray(merged.points), axis=0).max()) or 1.0
    try:
        lines = merged.clean(tolerance=1e-7 * size, absolute=False).strip()
    except Exception:  # pragma: no cover - VTK dependent
        return None
    loops = []
    pts = np.asarray(lines.points)
    for k in range(lines.n_cells):
        ids = np.asarray(lines.get_cell(k).point_ids)
        if len(ids) < 3:
            continue
        gap = float(np.linalg.norm(pts[ids[0]] - pts[ids[-1]]))
        extent = float(np.ptp(pts[ids], axis=0).max()) or 1.0
        # closed, or nearly: a seam no drawn mesh carries (a horn's back face round its throat port, 1 mm on a 50 mm
        # Scrimp) is bridged; a lone wall's trace (throat to rim) is far from closing and fills nothing
        if ids[0] == ids[-1] or gap <= CLOSE_GAP * extent:
            loops.append(ids if ids[0] == ids[-1] else np.r_[ids, ids[0]])
    if not loops:
        return None
    cells = np.concatenate([np.r_[len(ids), ids] for ids in loops])
    closed = pv.PolyData(pts, lines=cells.astype(np.int64))
    try:
        tri = vtk.vtkContourTriangulator()
        tri.SetInputData(closed)
        tri.Update()
        out = pv.wrap(tri.GetOutput())
    except Exception:  # pragma: no cover - VTK dependent
        return None
    return out if out.n_cells else None


def is_closed(points, faces) -> bool:
    """Whether a mesh (VTK flat faces: triangles, quads, polygons) is closed: every edge of its triangles shared by exactly two."""
    from meshdraw.mesh import vtk_faces_to_triangles

    tri = vtk_faces_to_triangles(faces)
    if not len(tri):
        return False
    # coincident points merged first: a revolved body repeats its seam (0 and 360 deg) as separate vertices
    pts = np.asarray(points, float).reshape(-1, 3)
    tol = 1e-9 * max(float(np.ptp(pts, axis=0).max()), 1e-12)
    _u, inverse = np.unique(np.round(pts / tol).astype(np.int64), axis=0, return_inverse=True)
    tri = np.asarray(inverse).reshape(-1)[tri]
    tri = tri[(tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 2] != tri[:, 0])]  # collapsed at the axis
    e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    _u, counts = np.unique(e, axis=0, return_counts=True)
    return bool(len(counts)) and bool(np.all(counts == 2))


def plane_mesh(origin, normal, size: float):
    """A square of side ``size`` on the plane (the drawn plane while it is dragged)."""
    import pyvista as pv

    return pv.Plane(center=tuple(np.asarray(origin, float)), direction=tuple(np.asarray(normal, float)), i_size=size, j_size=size)


__all__ = ["AXES", "MAX_PLANES", "Section", "SectionPlane", "box_centre", "cap", "cap_limits", "extent_along", "is_closed", "kept",
           "outline", "pieces", "plane_mesh"]
