"""What a 3D view draws, as plain data (no Qt, no VTK): items built from meshes, grids and lines, and port glyphs.

An ITEM is a dict the view renders: ``{"kind": "mesh", points (n, 3), faces
(VTK flat), color, opacity, label, member, solid}``, optionally with
``scalars`` (one per point or per cell), ``cmap``, ``clim`` and
``scalar_bar`` (a title: the bar is shown once per title), or ``{"kind":
"lines", points, lines (VTK flat), color, width, label}`` with optional
``scalars`` for coloured polylines. Labels are ``member/leaf``: the leaf
chooses a default tint (``LEAF_STYLES``), the member a tint of its own
(``MEMBER_TINTS``, in order of first appearance, so neighbours differ).
``solid`` says whether a section cap may fill the item's trace (a metal
body) or only outline it (an air surface, an open wall).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from meshdraw.style import EXTERIOR_OPACITY, EXTERIOR_TINT, FEM_TINT, MEMBER_TINTS, METAL_TINT

#: leaves whose surface bounds AIR, not metal: outlined by a section, never filled
AIR_LEAVES = ("fem_body",)
#: a display-only stand-in is see-through, so it never reads as real geometry
NOTIONAL_OPACITY = 0.45
#: leaf name -> (color, opacity); ``None`` for the colour means the member's own tint
LEAF_STYLES: Dict[str, Tuple[Optional[str], float]] = {
    "exterior": (EXTERIOR_TINT, EXTERIOR_OPACITY),
    "fem_body": (FEM_TINT, 1.0),
    "fem_metal": (METAL_TINT, 1.0),
    "notional": (None, NOTIONAL_OPACITY),
}


def body_items(meshes: Sequence[Tuple[Any, Any]], labels: Sequence[str], *, colors: Optional[Mapping[str, str]] = None,
               cutaway: bool = False, leaf_styles: Optional[Mapping[str, Tuple[Optional[str], float]]] = None,
               tints: Sequence[str] = MEMBER_TINTS, air_leaves: Iterable[str] = AIR_LEAVES) -> List[Dict]:
    """Mesh items for a posed body: ``[(points, faces)]`` with one label each (``member/leaf``).

    Each member gets its own tint, a leaf in ``leaf_styles`` its style,
    ``colors`` maps a full label to an explicit colour, and ``cutaway``
    opens the quadrant ``x > cx, y > cy`` about the body's axis on every
    mesh. Pure, so a view's content is testable without a renderer."""
    from meshdraw.mesh import cutaway_mesh

    meshes = list(meshes)
    if not meshes:
        return []
    styles = dict(LEAF_STYLES if leaf_styles is None else leaf_styles)
    air = set(air_leaves)
    allpts = np.concatenate([np.asarray(p, dtype=float).reshape(-1, 3) for p, _ in meshes])
    center = 0.5 * (allpts[:, :2].min(axis=0) + allpts[:, :2].max(axis=0))
    tint_of: Dict[str, str] = {}
    out = []
    for (points, faces), label in zip(meshes, labels):
        member, _, leaf = label.rpartition("/")
        member = member or label
        color, opacity = styles.get(leaf, (None, 1.0))
        if color is None:
            if member not in tint_of:  # only walls (and stand-ins) take a tint, so neighbours always differ
                tint_of[member] = tints[len(tint_of) % len(tints)]
            color = tint_of[member]
        color = (colors or {}).get(label) or color
        points = np.asarray(points, dtype=float)
        faces = np.asarray(faces, dtype=np.int64)
        if cutaway:
            points, faces = cutaway_mesh(points, faces, center)
        if len(faces) == 0:
            continue
        out.append({"kind": "mesh", "points": points, "faces": faces, "color": color, "opacity": float(opacity), "member": member,
                    "label": label, "solid": leaf not in air})
    return out


def mesh_item(points, faces, label: str, *, color: str = "#c9d3dd", opacity: float = 1.0, solid: bool = True, scalars=None,
              cmap: str = "viridis", clim: Optional[Tuple[float, float]] = None, scalar_bar: Optional[str] = None) -> Dict:
    """One surface item; ``scalars`` (per point or per cell) colour it through ``cmap`` and show ``scalar_bar``."""
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim == 2:  # (t, 3) triangles -> VTK flat
        faces = np.hstack([np.full((len(faces), 1), faces.shape[1], dtype=np.int64), faces]).ravel()
    item = {"kind": "mesh", "points": np.asarray(points, dtype=float).reshape(-1, 3), "faces": faces, "color": color,
            "opacity": float(opacity), "member": label.rpartition("/")[0] or label, "label": label, "solid": bool(solid)}
    if scalars is not None:
        item.update(scalars=np.asarray(scalars, dtype=float), cmap=cmap, clim=clim, scalar_bar=scalar_bar)
    return item


def grid_item(X, Y, Z, label: str, *, scalars=None, **kw) -> Dict:
    """A structured ``(m, n)`` surface (a reflector) as a mesh item of quads; ``scalars`` (m, n) per point."""
    X, Y, Z = (np.asarray(a, dtype=float) for a in (X, Y, Z))
    m, n = X.shape
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    i, j = np.meshgrid(np.arange(m - 1), np.arange(n - 1), indexing="ij")
    a = (i * n + j).ravel()
    quads = np.column_stack([np.full(len(a), 4), a, a + 1, a + n + 1, a + n]).astype(np.int64).ravel()
    keep = np.isfinite(pts).all(axis=1)
    if not keep.all():  # drop quads with a missing corner
        q = quads.reshape(-1, 5)
        ok = keep[q[:, 1:]].all(axis=1)
        quads = q[ok].ravel()
        pts = np.where(keep[:, None], pts, 0.0)
    return mesh_item(pts, quads, label, scalars=None if scalars is None else np.asarray(scalars, dtype=float).ravel(), **kw)


def lines_item(polylines: Sequence[Any], label: str, *, color: str = "#d32f2f", width: float = 2.0, opacity: float = 1.0,
               scalars=None, cmap: str = "viridis", clim: Optional[Tuple[float, float]] = None, scalar_bar: Optional[str] = None) -> Dict:
    """Polylines (each ``(k, 3)``) as one item; ``scalars`` one per polyline, or one per point over all of them."""
    pts, lines, per_line = [], [], []
    off = 0
    for k, pl in enumerate(polylines):
        p = np.asarray(pl, dtype=float).reshape(-1, 3)
        if len(p) < 2:
            continue
        pts.append(p)
        lines.append(np.concatenate([[len(p)], np.arange(off, off + len(p))]))
        per_line.append(np.full(len(p), k))
        off += len(p)
    item = {"kind": "lines", "points": np.vstack(pts) if pts else np.zeros((0, 3)), "lines": np.concatenate(lines).astype(np.int64) if lines else np.zeros(0, np.int64),
            "color": color, "width": float(width), "opacity": float(opacity), "label": label, "member": label.rpartition("/")[0] or label, "solid": False}
    if scalars is not None:
        s = np.asarray(scalars, dtype=float).ravel()
        if pts and len(s) == len(pts):  # one value per polyline: spread over its points
            s = s[np.concatenate(per_line)]
        item.update(scalars=s, cmap=cmap, clim=clim, scalar_bar=scalar_bar)
    return item


def port_glyph(name: str, origin, axis, radius: float, *, outline=None, xhat=None, kind: str = "guide") -> Dict:
    """A port drawn as a translucent labelled face: its ``outline`` (k, 3) when known, else a disc of ``radius``."""
    return {"name": str(name), "origin": np.asarray(origin, dtype=float), "axis": np.asarray(axis, dtype=float), "radius": float(radius),
            "kind": kind, "outline": None if outline is None else np.asarray(outline, dtype=float), "xhat": None if xhat is None else np.asarray(xhat, dtype=float)}


def port_face_cells(loop) -> np.ndarray:
    """VTK cells covering a port's outline ``(k, 3)``: its ear-clipped triangles. One polygon cell of a concave outline (a
    double ridge) is fanned from one corner by VTK, drawing sheets across the ridges' notches."""
    from meshdraw.mesh import triangulate_polygon

    loop = np.asarray(loop, float)
    c = loop.mean(axis=0)
    _u, _s, vt = np.linalg.svd(loop - c, full_matrices=False)
    uv = (loop - c) @ vt[:2].T  # the outline in its own plane
    tri = triangulate_polygon(uv)
    if not len(tri):
        return np.array([len(loop), *range(len(loop))], dtype=np.int64)
    return np.hstack([np.full((len(tri), 1), 3, dtype=np.int64), tri.astype(np.int64)]).ravel()


__all__ = ["AIR_LEAVES", "LEAF_STYLES", "NOTIONAL_OPACITY", "body_items", "grid_item", "lines_item", "mesh_item", "port_face_cells", "port_glyph"]
