"""Rendering without Qt: the same items and look on an off-screen pyvista plotter, to a PNG (help figures, scripts)."""

from __future__ import annotations

from typing import Optional, Sequence


def screenshot(items: Sequence[dict], path: str, *, title: str = "", window_size=(1200, 800), camera: Optional[str] = "iso",
               show_edges: bool = True) -> None:
    """Draw scene items (``meshdraw.scene``) in the CAD style to ``path``."""
    import numpy as np
    import pyvista as pv

    from meshdraw.style import EDGE_COLOR, display_mesh, edge_strength, feature_edges, style_plotter, surface_props

    plotter = pv.Plotter(off_screen=True, window_size=tuple(window_size))
    style_plotter(plotter, aa="none", ssao=False)
    bars: set = set()
    for item in items:
        if item.get("kind") == "lines":
            poly = pv.PolyData(np.asarray(item["points"], float), lines=np.asarray(item["lines"], np.int64))
            plotter.add_mesh(poly, color=item["color"], line_width=float(item.get("width", 2.0)), opacity=float(item.get("opacity", 1.0)))
            continue
        mesh = display_mesh(item["points"], item["faces"])
        kw = {"color": item["color"]}
        s = item.get("scalars")
        if s is not None:
            s = np.asarray(s, float).ravel()
            if len(s) == mesh.n_points:
                mesh.point_data["value"] = s
            elif len(s) == mesh.n_cells:
                mesh.cell_data["value"] = s
            if "value" in mesh.point_data or "value" in mesh.cell_data:
                title_bar = item.get("scalar_bar")
                kw = {"scalars": "value", "cmap": item.get("cmap") or "viridis", "show_scalar_bar": bool(title_bar) and title_bar not in bars}
                if item.get("clim") is not None:
                    kw["clim"] = tuple(item["clim"])
                if kw["show_scalar_bar"]:
                    bars.add(title_bar)
                    kw["scalar_bar_args"] = {"title": str(title_bar)}
        plotter.add_mesh(mesh, **kw, **surface_props(item.get("opacity", 1.0)))
        if show_edges:
            edges = feature_edges(item["points"], item["faces"])
            if edges is not None:
                plotter.add_mesh(edges, color=EDGE_COLOR, line_width=1.2, opacity=edge_strength(item["points"], item["faces"], edges))
    if title:
        plotter.add_text(title, position="upper_left", font_size=10)
    if camera == "iso":
        plotter.view_isometric()
    elif camera:
        plotter.camera_position = camera
    plotter.reset_camera()
    plotter.screenshot(path)
    plotter.close()


__all__ = ["screenshot"]
