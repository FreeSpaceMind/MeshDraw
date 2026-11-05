"""The 3D views' look: the CAD style every embedded view shares (``make_interactor`` applies it; offscreen renders too).

* Light neutral colours: walls in a family of light metal tints (one per
  member, so neighbours differ), exterior bodies a light translucent grey,
  a radiating part's metal a light steel blue, ports blue.
* Three-point lighting plus a weak headlight, a vertical gradient
  background and an orientation triad.
* Anti-aliasing per MACHINE (``settings``): 4x MSAA by
  default (crisp edges and silhouettes; FXAA, the old default, softens the
  whole image), FXAA or none. Optional screen-space ambient occlusion
  (``settings``) for depth in cavities.
* Depth peeling, so overlapping translucent surfaces sort correctly.
* Display meshes carry normals split at ``FEATURE_ANGLE``: curved walls
  shade smoothly, corners stay sharp. Feature edges are drawn as crisp
  dark lines pulled toward the viewer (polygon offset), never z-fighting
  with their faces.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

#: the corner a surface may turn through and still shade as one smooth face; beyond it, an edge line and split normals
FEATURE_ANGLE = 30.0
#: walls: a member's tint, in order of first appearance (light metals: neighbours always differ)
MEMBER_TINTS = ("#c9d3dd", "#e3d3b5", "#c8dccb", "#dccbe0", "#d9c6bb", "#c4dbe0", "#e0dcc0", "#cfd0e6")
EXTERIOR_TINT = "#a9b2bc"
EXTERIOR_OPACITY = 0.4
#: a radiating part's metal (the air's surface less its open box) and a STEP part's air wall
METAL_TINT = "#b9c6d3"
FEM_TINT = "#bcd2e6"
PORT_COLOR = "#1e88e5"
PORT_EDGE_COLOR = "#0d47a1"
EDGE_COLOR = "#1f2a36"
#: the cut face of a capped solid, and the outline a section leaves on a wall
SECTION_CAP_COLOR = "#8a97a6"
SECTION_LINE_COLOR = "#0b0f14"
BACKGROUND_BOTTOM = "#e6eaef"
BACKGROUND_TOP = "#ffffff"
#: anti-aliasing choices
ANTIALIASING = ("msaa", "fxaa", "none")
DEFAULT_ANTIALIASING = "msaa"
MSAA_SAMPLES = 4


#: a radiating part's open boundary (its air box per frequency) and a PML's outer box
BOUNDARY_COLOR = "#00897b"
PML_COLOR = "#8e24aa"
#: a solver mesh drawn on the part
MESH_LINE_COLOR = "#283593"
#: a mesh refinement box
REFINE_COLOR = "#c2185b"
#: reference geometry: planes and axes
REFERENCE_COLOR = "#5e35b1"
#: a scene this heavy (triangles, a translucent one counting three times, plus mesh lines) draws a cheaper frame while the
#: camera moves: no depth peeling, no ambient occlusion, no feature edges or mesh lines (restored when it stops)
FAST_INTERACTION_COST = 300_000


def _mix(color: str, other: str, t: float) -> str:
    """``color`` moved ``t`` of the way to ``other`` (both #rrggbb)."""
    a, b = str(color).lstrip("#"), str(other).lstrip("#")
    if len(a) != 6 or len(b) != 6:
        return color
    return "#" + "".join(f"{int(round((1 - t) * int(a[k:k + 2], 16) + t * int(b[k:k + 2], 16))):02x}" for k in (0, 2, 4))


def _darker(color: str, factor: float) -> str:
    """``color`` (#rrggbb) scaled toward black by ``factor``."""
    c = str(color).lstrip("#")
    if len(c) != 6:
        return color
    rgb = [max(0, min(255, int(int(c[k:k + 2], 16) * factor))) for k in (0, 2, 4)]
    return "#" + "".join(f"{v:02x}" for v in rgb)


class Settings:
    """Where this machine's display choices live: a ``get(key) -> value`` / ``set(key, value)`` pair a host supplies
    (a host's own machine store), in memory by default."""

    ANTIALIASING = "view_antialiasing"
    AMBIENT_OCCLUSION = "view_ambient_occlusion"

    def __init__(self) -> None:
        self._values: Dict[str, Any] = {}
        self._get = self._values.get
        self._set = self._values.__setitem__

    def configure(self, get, set) -> None:
        """Back the settings by a host's store."""
        self._get, self._set = get, set

    def get(self, key: str, default=None):
        v = self._get(key)
        return default if v is None else v

    def set(self, key: str, value) -> None:
        self._set(key, value)


#: the one settings store every view reads (``Settings.configure`` to back it by a host's)
settings = Settings()


def antialiasing() -> str:
    """This machine's anti-aliasing (``msaa`` / ``fxaa`` / ``none``)."""
    v = str(settings.get(Settings.ANTIALIASING) or DEFAULT_ANTIALIASING).lower()
    return v if v in ANTIALIASING else DEFAULT_ANTIALIASING


def ambient_occlusion() -> bool:
    return bool(settings.get(Settings.AMBIENT_OCCLUSION) or False)


def _set_polygon_offset() -> None:
    """Surfaces pushed back a little, lines not: edges drawn on a face never flicker through it (global VTK state)."""
    import vtk

    vtk.vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()
    vtk.vtkMapper.SetResolveCoincidentTopologyPolygonOffsetParameters(1.0, 1.0)
    vtk.vtkMapper.SetResolveCoincidentTopologyLineOffsetParameters(0.0, -4.0)


def style_plotter(plotter, *, aa: Optional[str] = None, ssao: Optional[bool] = None) -> None:
    """Give a pyvista plotter the CAD look: lights, background, anti-aliasing, depth peeling, triad. Idempotent."""
    import warnings

    import pyvista as pv

    _set_polygon_offset()
    plotter.set_background(BACKGROUND_BOTTOM, top=BACKGROUND_TOP)
    try:
        plotter.remove_all_lights()
        plotter.enable_3_lights()
        plotter.add_light(pv.Light(light_type="headlight", intensity=0.3))
    except Exception:  # pragma: no cover - renderer dependent
        pass
    aa = antialiasing() if aa is None else aa
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plotter.disable_anti_aliasing()
            if aa == "msaa":
                plotter.enable_anti_aliasing("msaa", multi_samples=MSAA_SAMPLES)
            elif aa == "fxaa":
                plotter.enable_anti_aliasing("fxaa")
        if any("SSAA" in str(w.message) for w in caught):
            plotter.disable_anti_aliasing()  # OSMesa / EGL builds swap FXAA for SSAA (a render at several times the size)
    except Exception:  # pragma: no cover - renderer dependent
        pass
    try:
        plotter.enable_depth_peeling(number_of_peels=8)
    except Exception:  # pragma: no cover - renderer dependent
        pass
    if ambient_occlusion() if ssao is None else ssao:
        try:
            plotter.enable_ssao(radius=_ssao_radius(plotter), bias=0.01, kernel_size=64)
        except Exception:  # pragma: no cover - renderer dependent
            pass
    try:
        if getattr(plotter, "renderer", None) is not None and getattr(plotter.renderer, "axes_widget", None) is None:
            plotter.add_axes(line_width=2, labels_off=False)
    except Exception:  # pragma: no cover - renderer dependent
        pass


def _ssao_radius(plotter) -> float:
    try:
        b = np.asarray(plotter.bounds, float).reshape(3, 2)
        return max(float(np.ptp(b, axis=1).max()) * 0.02, 1e-4)
    except Exception:
        return 1e-3


def display_mesh(points, faces):
    """A PolyData that shades well: normals split at ``FEATURE_ANGLE`` (curved faces smooth, corners sharp)."""
    import pyvista as pv

    mesh = pv.PolyData(np.asarray(points, float), np.asarray(faces, np.int64))
    try:
        return mesh.compute_normals(split_vertices=True, feature_angle=FEATURE_ANGLE, consistent_normals=True,
                                    auto_orient_normals=False, inplace=False)
    except Exception:  # pragma: no cover - degenerate input
        return mesh


def feature_edges(points, faces):
    """The mesh's corners and open borders as lines (None when there are none)."""
    import pyvista as pv

    try:
        e = pv.PolyData(np.asarray(points, float), np.asarray(faces, np.int64)).extract_feature_edges(
            feature_angle=FEATURE_ANGLE, boundary_edges=True, non_manifold_edges=False, manifold_edges=False)
    except Exception:  # pragma: no cover - degenerate input
        return None
    return e if e.n_points else None


#: feature-line density (total edge length / sqrt(surface area)) up to which edges draw at full strength; a cube is 4.9, a
#: quad-ridge horn 24, an axial horn 25, a 130-corrugation radial horn 200 (its lines swamped the surface at full strength)
EDGE_DENSITY_FULL = 40.0


def edge_strength(points, faces, edges) -> float:
    """How strongly a mesh's feature edges draw (an opacity, 0.15 .. 1): full below ``EDGE_DENSITY_FULL``, fading in
    proportion beyond, so a long corrugated horn's hundreds of rings read as texture, not a black surface."""
    import pyvista as pv

    if edges is None:
        return 1.0
    try:
        length = float(edges.compute_cell_sizes(length=True, area=False, volume=False)["Length"].sum())
        area = float(pv.PolyData(np.asarray(points, float), np.asarray(faces, np.int64)).area)
    except Exception:  # pragma: no cover - degenerate input
        return 1.0
    density = length / np.sqrt(area) if area > 0 else 0.0
    return float(np.clip(EDGE_DENSITY_FULL / density, 0.15, 1.0)) if density > EDGE_DENSITY_FULL else 1.0


def surface_props(opacity: float) -> Dict[str, Any]:
    """``add_mesh`` keywords of a shaded surface: matte metal, smooth shading on the split normals."""
    return {"smooth_shading": True, "split_sharp_edges": False, "ambient": 0.18, "diffuse": 0.78, "specular": 0.22,
            "specular_power": 24.0, "opacity": float(opacity)}


__all__ = ["Settings", "settings", "BOUNDARY_COLOR", "FAST_INTERACTION_COST", "MESH_LINE_COLOR", "PML_COLOR", "REFERENCE_COLOR", "REFINE_COLOR", "ANTIALIASING", "BACKGROUND_BOTTOM", "BACKGROUND_TOP", "DEFAULT_ANTIALIASING", "EDGE_COLOR", "EXTERIOR_OPACITY",
           "EXTERIOR_TINT", "FEATURE_ANGLE", "FEM_TINT", "MEMBER_TINTS", "METAL_TINT", "PORT_COLOR", "PORT_EDGE_COLOR",
           "SECTION_CAP_COLOR", "SECTION_LINE_COLOR", "ambient_occlusion", "antialiasing", "display_mesh", "edge_strength", "feature_edges",
           "style_plotter", "surface_props"]
