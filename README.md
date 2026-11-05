# MeshDraw

One 3D drawer for antenna_feeds, UmbraAntennaDesigner and mastmodeler: the
CAD look, section planes with filled cuts, per-object hide / transparency /
colour, a SolidWorks-style measure, and a scene model of meshes,
scalar-coloured surfaces, polylines and port glyphs.

- `meshdraw.widget.View3DWidget` (needs the `qt` extra: PyQt6, pyvista,
  pyvistaqt): `set_body(body)` for a device body, `set_items(items)` for
  scene items, `set_overlays`, `view_state` / `set_view_state`,
  `tools=False` for a plain view.
- `meshdraw.offscreen.screenshot(items, path)`: the same look without Qt.
- Qt-free: `scene`, `style`, `sections`, `appearance`, `measure`, `mesh`.

Rules in `CLAUDE.md`; the record in `docs/notes/view.md`.
