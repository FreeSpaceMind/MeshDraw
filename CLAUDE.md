# MeshDraw - rules

One 3D drawer for antenna_feeds, UmbraAntennaDesigner and mastmodeler.
This file holds the RULES; the record (what was decided, how each piece
works, the measurements behind thresholds) is in `docs/notes/view.md`.
Read the note before changing an area and update it with the change.

- **The core is Qt-free** (`scene`, `style`, `sections`, `appearance`,
  `measure`, `mesh`); only `widget` and `interactor` import PyQt6 /
  pyvistaqt, and `offscreen` pyvista. Nothing in the package imports a
  consumer (antenna_feeds, Umbra, mastmodeler): a consumer's naming rules
  reach the drawer as data (`scene.LEAF_STYLES`, tints, labels), never as
  imports.
- **Every view comes from `interactor.make_interactor`** (no auto-update,
  the CAD style of `style.style_plotter`); scenes render once per change.
- **Items are plain dicts** (`scene.mesh_item` / `grid_item` /
  `lines_item`); labels are `member/leaf`, and a label is what display
  choices, highlights and the STL export are keyed by.
- **Section planes** cut away the region beyond every plane, drawn as
  disjoint GPU-clipped pieces (one actor per piece, each its own name);
  only `solid` items' traces are filled.
- **Display settings are the machine's** (`style.settings`): a host backs
  the store with its own (`Settings.configure`); never a project's.
- **The measure reads the meshes it draws** (`measure.MeshTopology`): faces
  grown at the 25 deg feature-line angle, edges and vertices from them.
- No widget tree is left to Python's garbage collector: a dropped view is
  `deleteLater`ed by its host, and `cleanup_plotter` finalizes the window.
- Tests: `pytest` from the repository root (a `qapp` fixture in
  `tests/conftest.py`; pyvista tests skip without it).
