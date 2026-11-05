# The 3D view: record

MeshDraw was extracted (2026-11) from antenna_feeds' device workspace,
where the drawer was overhauled on the owner's "it sucks: not crisp, hard
to see details; we need cuts, hide objects, transparency": the CAD look,
section planes with a drag handle and filled cuts, right-click display
choices, the measure tool. The decisions and measurements behind each piece
(anti-aliasing per machine, depth peeling, the feature-line fade above an
edge density of 40, the per-piece actor names, the plane widget) are
recorded in antenna_feeds' `docs/notes/workspace.md` ("The 3D view
overhaul", "second batch"); they are not repeated here.

What the extraction changed:

- `HornView3DWidget` -> `View3DWidget` (`widget.py`): the horn entry points
  (axial / radial profiles, the notional-body export filter, the
  workspace's tooltips, length unit, file dialogs and machine store) stay
  in antenna_feeds as a subclass and hooks (`tips`, `units`, `save_file`,
  `exportable`, `style.settings.configure`).
- `subject.body_render_items` -> `scene.body_items` with the leaf styles as
  data (`LEAF_STYLES`: exterior, fem_body, fem_metal, notional) and
  `AIR_LEAVES`; new item kinds: a scalar-coloured surface (`mesh_item` with
  `scalars` / `cmap` / `clim` / `scalar_bar`, per point or per cell), a
  structured grid (`grid_item`) and polylines (`lines_item`), what Umbra's
  reflector scene needs (deformation and current maps, rays).
- `tools=False`: the toolbar, section bar and measure readout hidden (a
  plain view for Umbra's 3D tab); right-click stays.
- `offscreen.screenshot`: the items in the same style without Qt (help
  figures, mastmodeler's example scripts).
