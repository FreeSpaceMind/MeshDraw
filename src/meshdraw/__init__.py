"""MeshDraw: one 3D drawer for antenna_feeds, UmbraAntennaDesigner and mastmodeler.

Qt-free core: ``style`` (the CAD look and this machine's display settings),
``scene`` (items: meshes, scalar-coloured surfaces, polylines, ports),
``sections`` (section planes, pieces, caps), ``appearance`` (per-object
display choices), ``measure`` (faces, edges, vertices and distances from a
drawn mesh), ``mesh`` (revolve, STL, cut-away, triangulation). The widget
(``meshdraw.widget.View3DWidget``, the ``qt`` extra: PyQt6, pyvista,
pyvistaqt) renders them; ``offscreen`` renders without Qt.
"""

from meshdraw.appearance import Appearance, Choice
from meshdraw.scene import body_items, grid_item, lines_item, mesh_item, port_glyph
from meshdraw.sections import Section, SectionPlane
from meshdraw.style import Settings, settings

__all__ = ["Appearance", "Choice", "Section", "SectionPlane", "Settings", "body_items", "grid_item", "lines_item", "mesh_item",
           "port_glyph", "settings"]
__version__ = "0.1.0"
