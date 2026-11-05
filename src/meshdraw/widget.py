"""The 3D view widget: a posed body or scene items in the CAD style, with section planes, per-object display choices and measure.

Ported from antenna_feeds' device workspace (``HornView3DWidget``, 2026-11)
so antenna_feeds, UmbraAntennaDesigner and mastmodeler draw with ONE
widget. pyvista and pyvistaqt are optional (the ``qt`` extra); without
them the widget degrades to a label saying how to enable the view.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from meshdraw.appearance import Appearance
from meshdraw.interactor import PYVISTAQT_AVAILABLE, PYVISTA_AVAILABLE, make_interactor, pv
from meshdraw.sections import Section
from meshdraw.scene import port_face_cells
from meshdraw.style import (
    BOUNDARY_COLOR,
    EDGE_COLOR,
    FAST_INTERACTION_COST,
    MESH_LINE_COLOR,
    PML_COLOR,
    REFERENCE_COLOR,
    REFINE_COLOR,
    _darker,
    _mix,
)

#: a port's frame: outward normal and x-hat (the polarization reference)
PORT_NORMAL_COLOR = "#1a237e"
PORT_XHAT_COLOR = "#d50000"
#: feature lines (corners, open borders) on the meshes
EDGE_LINE_COLOR = EDGE_COLOR
#: the measure tool's picked entities and its closest-points line
MEASURE_COLOR = "#00b8d4"
MEASURE_LINE_COLOR = "#ff6d00"
#: how close (screen pixels) a click must come to a vertex or an edge to pick it
MEASURE_PICK_PX = 8
MEASURE_HINT = ("Measure: click faces, edges or vertices (one or two). Click one again to drop it, a third starts again, "
                "a click on empty space clears.")
#: the section sliders' resolution
SECTION_STEPS = 1000
#: the draggable section plane's handle
PORT_COLOR_HANDLE = "#ff6d00"
#: an object the host's tree selected
HIGHLIGHT_COLOR = "#ff8f00"

DEFAULT_TIPS = {
    "view.fit": "Fit the whole scene in the view.",
    "view.perspective": "Perspective projection; off: parallel (orthographic).",
    "view.show": "What the scene draws besides the bodies.",
    "view.edges": "Feature lines: corners and open borders of the meshes.",
    "view.port_frames": "Each port's outward normal and x-hat.",
    "view.open_boundary": "A radiating part's open boundary.",
    "view.mesh": "The solver mesh, where a preview holds one.",
    "view.refine_boxes": "Mesh refinement boxes.",
    "view.section": "Section planes: cut the scene away beyond up to three planes.",
    "view.measure": "Measure: distances, angles and sizes between picked faces, edges and vertices.",
    "view.section_plane": "Turn this plane on.",
    "view.section_axis": "The plane's normal: an axis, or free (moved by its handle).",
    "view.section_position": "Where along its axis the plane sits.",
    "view.section_flip": "Keep the other side.",
    "view.section_handle": "Drag the plane in the view.",
}


class MillimetreUnits:
    """The default length readout: millimetres to three decimals."""

    @staticmethod
    def format_length(metres: float) -> str:
        return f"{float(metres) * 1e3:.3f} mm"

    @staticmethod
    def length():
        return ("mm", 1e3)


class View3DWidget(QWidget):
    """The 3D view: any posed body (``set_body``) or scene items (``set_items``: surfaces, scalar-coloured surfaces, polylines), in the CAD style.

    ``tools`` False hides the toolbar, the section bar and the measure
    readout: a plain view (right-click still works). ``tips`` is a mapping
    of tooltip keys (``view.fit``, ``view.section`` ...), ``units`` an object
    with ``format_length(m) -> str`` and ``length() -> (unit, per_metre)``
    (millimetres by default), ``save_file(parent, title, name, filter) ->
    path`` a host's save dialog.

    * Views: Iso / Top / Front / Side, Fit, Perspective (off: parallel
      projection), Edges (the feature lines), Port frames, Measure.
    * Section: up to three planes (X / Y / Z or free), each with a position
      slider and a flip, one of them draggable in the view (its handle);
      together they cut away the region beyond every plane (one: a half,
      two: a quarter). Closed solids are capped where cut; walls and air
      bodies show the cut as an outline (``sections``).
    * Right-click an object: hide it, show only it, its transparency, its
      colour; right-click empty space: show all, bring back a hidden
      object, reset, and this machine's display settings (anti-aliasing,
      ambient occlusion). The choices are kept by object label across
      redraws (``appearance``).
    """

    #: an object was clicked (its label: ``member/leaf``, ``port:<name>``): a host selects it in its own tree
    object_clicked = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None, *, tools: bool = True, tips: Mapping[str, str] | None = None, units=None,
                 save_file=None):
        super().__init__(parent)
        self._available = PYVISTA_AVAILABLE and PYVISTAQT_AVAILABLE
        self._tips = dict(tips or {})
        self._units = units if units is not None else MillimetreUnits()
        self._save_file = save_file
        self._tools = bool(tools)
        self._last_body = None
        self.appearance = Appearance()
        self.section = Section()
        self._items: list = []  # the scene's body meshes (subject.body_render_items) and port glyphs
        self._objects: dict = {}  # label -> {"actors": [...], "color", "opacity", "kind"}
        self._actor_label: dict = {}  # an actor's VTK address -> its object's label
        self._section_actors: list = []
        self._vtk_planes: dict = {}  # (plane index, +1 / -1) -> vtkPlane shared by every clipped mapper
        self._display_cache: dict = {}  # id of an item's points -> (display mesh, feature edges)
        self._highlight: set = set()  # labels drawn highlighted (the navigator's selection)
        self._overlays: list = []  # open boundaries and meshes (``set_overlays``): drawn with the scene, never part of it
        self._fast = None  # what a moving camera switched off (``_on_interaction_start``), restored when it stops
        self._title = ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.plotter = None
        if not self._available:
            note = QLabel(
                "3D view requires pyvista and pyvistaqt.\n"
                "Install them with:  pip install meshdraw[qt]\n"
                "(or: pip install pyvista pyvistaqt)")
            note.setWordWrap(True)
            layout.addWidget(note)
            layout.addStretch()
            return

        self.toolbar_widget = QWidget()
        toolbar = QHBoxLayout(self.toolbar_widget)
        toolbar.setContentsMargins(4, 4, 4, 0)
        for label, position in (("Iso", "iso"), ("Top", "xy"), ("Front", "yz"), ("Side", "xz")):
            button = QPushButton(label)
            button.setToolTip(f"View from the {label.lower()} direction")
            button.clicked.connect(lambda _=False, pos=position: self._set_view(pos))
            toolbar.addWidget(button)
        self.fit_button = QPushButton("Fit")
        self.fit_button.setToolTip(self._tip("view.fit"))
        self.fit_button.clicked.connect(self.fit)
        toolbar.addWidget(self.fit_button)
        self.perspective_check = QCheckBox("Perspective")
        self.perspective_check.setChecked(True)
        self.perspective_check.setToolTip(self._tip("view.perspective"))
        self.perspective_check.toggled.connect(self._on_perspective)
        toolbar.addWidget(self.perspective_check)
        # what the scene draws besides the bodies: one menu of toggles
        self.show_button = QToolButton()
        self.show_button.setText("Show")
        self.show_button.setToolTip(self._tip("view.show"))
        self.show_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        show_menu = QMenu(self.show_button)
        self.show_button.setMenu(show_menu)
        self.edges_check = show_menu.addAction("Edges")
        self.edges_check.setToolTip(self._tip("view.edges"))
        self.edges_check.toggled.connect(self._on_edges)
        self.frames_check = show_menu.addAction("Port frames")
        self.frames_check.setToolTip(self._tip("view.port_frames"))
        self.frames_check.toggled.connect(self._rerender_last)
        self.boundary_check = show_menu.addAction("Open boundary")
        self.boundary_check.setToolTip(self._tip("view.open_boundary"))
        self.boundary_check.toggled.connect(lambda _on: self._refresh_appearance())
        self.mesh_check = show_menu.addAction("Mesh")
        self.mesh_check.setToolTip(self._tip("view.mesh"))
        self.mesh_check.toggled.connect(lambda _on: self._refresh_appearance())
        self.refine_check = show_menu.addAction("Refinement boxes")
        self.refine_check.setToolTip(self._tip("view.refine_boxes"))
        self.refine_check.toggled.connect(lambda _on: self._refresh_appearance())
        for act, on in ((self.edges_check, True), (self.frames_check, True), (self.boundary_check, True), (self.mesh_check, True),
                        (self.refine_check, True)):
            act.setCheckable(True)
            act.blockSignals(True)
            act.setChecked(on)
            act.blockSignals(False)
        show_menu.setToolTipsVisible(True)
        self._show_menu = show_menu
        #: an overlay's category -> the Show toggle that hides it (``add_show_toggle`` adds one)
        self._category_checks = {"boundary": self.boundary_check, "mesh": self.mesh_check, "refine": self.refine_check}
        toolbar.addWidget(self.show_button)
        self.section_button = QPushButton("Section")
        self.section_button.setCheckable(True)
        self.section_button.setToolTip(self._tip("view.section"))
        self.section_button.toggled.connect(self._on_section_toggled)
        toolbar.addWidget(self.section_button)
        self.measure_button = QPushButton("Measure")
        self.measure_button.setCheckable(True)
        self.measure_button.setToolTip(self._tip("view.measure"))
        self.measure_button.toggled.connect(self._on_measure_toggled)
        toolbar.addWidget(self.measure_button)
        self.export_stl_button = QPushButton("Export STL…")
        self.export_stl_button.setToolTip("Write the drawn body as a binary STL (metres).")
        self.export_stl_button.clicked.connect(self._on_export_stl)
        toolbar.addWidget(self.export_stl_button)
        toolbar.addStretch()
        layout.addWidget(self.toolbar_widget)
        self.toolbar_widget.setVisible(self._tools)
        self.section_bar = self._build_section_bar()
        self.section_bar.setVisible(False)
        layout.addWidget(self.section_bar)

        self.plotter = make_interactor(self)
        layout.addWidget(self.plotter, 1)
        self.measure_label = QLabel("")
        self.measure_label.setWordWrap(True)
        self.measure_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.measure_label.setVisible(False)
        layout.addWidget(self.measure_label)
        self._measure_meshes: list = []  # (points, faces) of every measurable mesh drawn, in drawing order
        self._measure_actors: dict = {}  # the actor's VTK address -> index into _measure_meshes
        self._topologies: dict = {}  # index -> MeshTopology, built on the first pick
        self.measure_selection: list = []  # the picked Entities (at most two)
        self.measure_result: dict = {}
        self._measure_overlays: list = []
        self._press = None
        self._rpress = None
        import vtk

        self._picker = vtk.vtkCellPicker()
        self._picker.SetTolerance(0.0005)
        iren = self.plotter.iren
        iren.add_observer("LeftButtonPressEvent", self._on_press)
        iren.add_observer("LeftButtonReleaseEvent", self._on_release)
        iren.add_observer("RightButtonPressEvent", self._on_rpress)
        iren.add_observer("RightButtonReleaseEvent", self._on_rrelease)
        try:  # the trackball style starts and ends every rotate / pan / spin
            style = iren.style
            style.AddObserver("StartInteractionEvent", lambda *_: self._on_interaction_start())
            style.AddObserver("EndInteractionEvent", lambda *_: self._on_interaction_end())
        except Exception:  # pragma: no cover - interactor dependent
            pass
        self.clear("Nothing to draw yet.")

    # -- the section bar ------------------------------------------------------------------------------------------------

    def _build_section_bar(self) -> QWidget:
        from meshdraw.sections import MAX_PLANES

        bar = QWidget()
        grid = QGridLayout(bar)
        grid.setContentsMargins(4, 0, 4, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(1)
        self._plane_rows = []
        for i in range(MAX_PLANES):
            on = QCheckBox(f"Plane {i + 1}")
            axis = QComboBox()
            for text, key in (("X", "x"), ("Y", "y"), ("Z", "z"), ("Free", "free")):
                axis.addItem(text, key)
            axis.setCurrentIndex(i if i < 3 else 0)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, SECTION_STEPS)
            slider.setValue(SECTION_STEPS // 2)
            where = QLabel("")
            where.setMinimumWidth(90)
            flip = QToolButton()
            flip.setText("Flip")
            flip.setCheckable(True)
            handle = QToolButton()
            handle.setText("Handle")
            handle.setCheckable(True)
            for w, key in ((on, "view.section_plane"), (axis, "view.section_axis"), (slider, "view.section_position"),
                           (flip, "view.section_flip"), (handle, "view.section_handle")):
                w.setToolTip(self._tip(key))
            for col, w in enumerate((on, axis, slider, where, flip, handle)):
                grid.addWidget(w, i, col)
            grid.setColumnStretch(2, 1)
            on.toggled.connect(lambda _on, i=i: self._on_plane_changed(i, structural=True))
            axis.currentIndexChanged.connect(lambda _k, i=i: self._on_plane_changed(i, structural=True))
            slider.valueChanged.connect(lambda _v, i=i: self._on_plane_changed(i))
            flip.toggled.connect(lambda _f, i=i: self._on_plane_changed(i, structural=True))
            handle.toggled.connect(lambda checked, i=i: self._on_handle(i, checked))
            self._plane_rows.append({"on": on, "axis": axis, "slider": slider, "where": where, "flip": flip, "handle": handle})
        return bar

    def _read_planes(self) -> None:
        """The section bar into ``self.section`` (a free plane keeps its normal)."""
        for row, plane in zip(self._plane_rows, self.section.planes):
            plane.enabled = row["on"].isChecked()
            plane.axis = str(row["axis"].currentData())
            plane.fraction = row["slider"].value() / SECTION_STEPS
            plane.flip = row["flip"].isChecked()

    def _on_section_toggled(self, on: bool) -> None:
        self.section_bar.setVisible(on)
        if on and not self.section.any():
            self._plane_rows[0]["on"].setChecked(True)  # a first plane through the middle, as the old cutaway opened
        self._read_planes()
        if not on:
            self._clear_plane_widget()
        self._apply_section(structural=True)

    def _on_plane_changed(self, i: int, structural: bool = False) -> None:
        if self.plotter is None or getattr(self, "_syncing", False):
            return
        self._read_planes()
        self._label_planes()
        self._apply_section(structural=structural)
        row = self._plane_rows[i]
        if row["handle"].isChecked() and structural:
            self._place_plane_widget(i)

    def _label_planes(self) -> None:
        bounds = self._scene_bounds()
        for row, plane in zip(self._plane_rows, self.section.planes):
            if bounds is None:
                row["where"].setText("")
                continue
            origin, _n = plane.place(bounds)
            if plane.axis in ("x", "y", "z"):
                row["where"].setText(f"{plane.axis} = {self._units.format_length(float(origin['xyz'.index(plane.axis)]))}")
            else:
                row["where"].setText(f"{plane.fraction * 100:.0f} %")

    def _on_handle(self, i: int, checked: bool) -> None:
        if getattr(self, "_syncing", False):
            return
        self._syncing = True
        try:
            for j, row in enumerate(self._plane_rows):  # one handle at a time
                if j != i:
                    row["handle"].setChecked(False)
            if checked:
                self._plane_rows[i]["on"].setChecked(True)
        finally:
            self._syncing = False
        self._read_planes()
        if checked:
            self._place_plane_widget(i)
        else:
            self._clear_plane_widget()
        self._apply_section(structural=True)

    def _place_plane_widget(self, i: int) -> None:
        bounds = self._scene_bounds()
        if self.plotter is None or bounds is None:
            return
        self._clear_plane_widget()
        origin, normal = self.section.planes[i].place(bounds)
        pad = 0.05 * float(np.ptp(np.asarray(bounds, float).reshape(3, 2), axis=1).max())
        b = np.asarray(bounds, float).reshape(3, 2) + np.array([-pad, pad])
        self.plotter.add_plane_widget(lambda n, o, i=i: self._on_plane_widget(i, o, n), normal=tuple(normal), origin=tuple(origin),
                                      bounds=tuple(b.ravel()), factor=1.0, implicit=False, outline_translation=False,
                                      interaction_event="always", color=PORT_COLOR_HANDLE)
        self._handle_for = i

    def _clear_plane_widget(self) -> None:
        if self.plotter is not None:
            try:
                self.plotter.clear_plane_widgets()
            except Exception:
                pass
        self._handle_for = None

    def _on_plane_widget(self, i: int, origin, normal) -> None:
        """The handle moved: plane ``i`` becomes free through its origin, its controls follow."""
        bounds = self._scene_bounds()
        if bounds is None:
            return
        plane = self.section.planes[i]
        plane.set_from(origin, normal, bounds)
        row = self._plane_rows[i]
        self._syncing = True
        try:
            row["on"].setChecked(True)
            row["axis"].setCurrentIndex(row["axis"].findData("free"))
            row["flip"].setChecked(False)
            row["slider"].setValue(int(round(plane.fraction * SECTION_STEPS)))
        finally:
            self._syncing = False
        self._label_planes()
        self._apply_section(structural=not self._vtk_planes)

    # -- plumbing -------------------------------------------------------------------------------------------------------

    def _set_view(self, position: str) -> None:
        if self.plotter is not None:
            self.plotter.camera_position = position
            self.plotter.reset_camera()
            self.plotter.render()

    def fit(self) -> None:
        if self.plotter is not None:
            self.plotter.reset_camera()
            self.plotter.render()

    def _on_perspective(self, on: bool) -> None:
        if self.plotter is None:
            return
        if on:
            self.plotter.disable_parallel_projection()
        else:
            self.plotter.enable_parallel_projection()
        self.plotter.render()

    def _on_edges(self, _on: bool) -> None:
        for obj in self._objects.values():
            self._apply_appearance(obj["label"], render=False)
        if self.plotter is not None:
            self.plotter.render()

    def clear(self, message: str = "") -> None:
        if self.plotter is None:
            return
        self._forget_measurables()
        self._clear_plane_widget()
        self.plotter.clear()
        self._items, self._objects, self._actor_label, self._section_actors = [], {}, {}, []
        self._overlays = []
        if message:
            self.plotter.add_text(message, position="upper_left", font_size=10, render=False)
        self.plotter.render()

    def _rerender_last(self) -> None:
        if self._last_body is not None:
            self.set_body(*self._last_body, keep_camera=True)

    def _on_export_stl(self) -> None:  # pragma: no cover - file dialog
        from PyQt6.QtWidgets import QFileDialog, QMessageBox

        if self._last_body is None:
            QMessageBox.information(self, "Export STL", "Nothing drawn yet.")
            return
        if self._save_file is not None:
            path = self._save_file(self, "Export STL", "body.stl", "STL mesh (*.stl)")
        else:
            path, _ = QFileDialog.getSaveFileName(self, "Export STL", "body.stl", "STL mesh (*.stl)")
        if not path:
            return
        try:
            self.export_stl(path)
        except Exception as exc:
            QMessageBox.critical(self, "STL export failed", str(exc))

    def export_stl(self, path) -> None:
        """Write the drawn body (every mesh item ``exportable``) as a binary STL."""
        from pathlib import Path as _Path

        _Path(path).write_bytes(self.body_stl_bytes())

    def exportable(self, label: str) -> bool:
        """Whether a drawn mesh belongs in the STL export (a host's display-only stand-ins do not)."""
        return True

    def set_body(self, body, title: str = "", colors=None, ports=None, *, keep_camera: bool = False) -> None:
        """Render a posed device ``Body`` (a ``CombinedDevice.body()``): any mix of meshes in world coordinates.

        Each member's wall gets its own light tint, exterior shells are light
        grey and translucent, a radiating part's metal steel blue
        (``subject.body_render_items``); every mesh carries its feature edges
        -- corrugation corners, steps, member ends -- as dark lines. The STL
        export writes the full meshes. ``ports`` (glyph dicts: ``name``,
        ``origin``, ``axis``, ``radius``) draw as translucent labelled faces.
        A new body resets the camera; ``keep_camera`` (a redraw of the same
        scene) keeps it.
        """
        self._last_body = (body, title, colors, ports)
        self._show_body(body, title, colors, ports, keep_camera=keep_camera)

    def set_items(self, items, title: str = "", ports=None, *, keep_camera: bool = False) -> None:
        """Render scene ITEMS (``scene.mesh_item`` / ``grid_item`` / ``lines_item`` dicts) in world coordinates."""
        self.set_body(list(items), title, None, ports, keep_camera=keep_camera)

    def _show_body(self, body, title, colors, ports, keep_camera: bool = False) -> None:
        if self.plotter is None:
            return
        camera = self.plotter.camera_position if keep_camera else None
        self._forget_measurables()
        self._clear_plane_widget()
        self.plotter.clear()
        self._objects, self._actor_label, self._section_actors, self._vtk_planes = {}, {}, [], {}
        self._title = title
        items = self.render_items(body, colors)
        if not items and not ports:
            self._items = []
            self.clear("No geometry to draw.")
            return
        self._items = items
        self._ports = list(ports or ())
        self._draw_scene()
        if title:
            self.plotter.add_text(title, position="upper_left", font_size=10, render=False)
        pending, self._pending_camera = getattr(self, "_pending_camera", None), None
        if pending is not None:
            self._set_camera(pending)  # a restored view state's
        elif camera is not None:
            self.plotter.camera_position = camera
        else:
            self.plotter.reset_camera()
            self.plotter.view_isometric()
            self.plotter.reset_camera()
        self._label_planes()
        if self.section_button.isChecked() and getattr(self, "_handle_for", None) is not None:
            self._place_plane_widget(self._handle_for)
        self.plotter.render()

    @staticmethod
    def render_items(body, colors=None) -> list:
        """The scene's items for what ``set_body`` was handed: a list of item dicts as they are, a ``(meshes, labels)`` pair or
        an object with ``meshes`` and ``labels`` (a device body) through ``scene.body_items``; None draws nothing."""
        from meshdraw.scene import body_items

        if body is None:
            return []
        if isinstance(body, (list, tuple)) and body and isinstance(body[0], dict):
            return [dict(it) for it in body]
        if isinstance(body, (list, tuple)) and len(body) == 2:
            meshes, labels = body
            return body_items(meshes, labels, colors=colors)
        meshes = getattr(body, "meshes", None)
        if not meshes:
            return []
        return body_items(meshes, getattr(body, "labels", [f"mesh{i}" for i in range(len(meshes))]), colors=colors)

    # -- the scene ------------------------------------------------------------------------------------------------------

    def _scene_bounds(self):
        pts = [np.asarray(it["points"], float).reshape(-1, 3) for it in self._items if len(it["points"])]
        pts += [np.asarray(g["origin"], float).reshape(1, 3) for g in getattr(self, "_ports", [])]
        if not pts:
            return None
        allp = np.vstack(pts)
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        pad = 1e-6 * max(float(np.ptp(allp, axis=0).max()), 1e-9)
        return (lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad, lo[2] - pad, hi[2] + pad)

    def _display(self, item):
        """An item's shaded display mesh and feature edges (made once per drawing of these points)."""
        from meshdraw.style import display_mesh, edge_strength, feature_edges

        key = (id(item["points"]), len(item["points"]), len(item["faces"]))
        if key not in self._display_cache:
            if len(self._display_cache) > 64:
                self._display_cache.clear()
            edges = feature_edges(item["points"], item["faces"])
            self._display_cache[key] = (display_mesh(item["points"], item["faces"]), edges,
                                        edge_strength(item["points"], item["faces"], edges), item["points"])
        mesh, edges, strength, _keep = self._display_cache[key]
        return mesh, edges, strength

    def _plane(self, i: int, sign: int, origin, normal):
        import vtk

        key = (i, sign)
        p = self._vtk_planes.get(key)
        if p is None:
            p = self._vtk_planes[key] = vtk.vtkPlane()
        p.SetOrigin(*[float(v) for v in origin])
        p.SetNormal(*[float(sign * v) for v in normal])
        return p

    def _clip_sets(self):
        """The vtkPlane list of each disjoint piece (``sections.pieces``): one empty list without a section."""
        bounds = self._scene_bounds()
        active = self.section.active(bounds) if (self.section_button.isChecked() and bounds is not None) else []
        if not active:
            return [[]], []
        sets = []
        for i, (o, n) in enumerate(active):
            sets.append([self._plane(i, +1, o, n)] + [self._plane(j, -1, oj, nj) for j, (oj, nj) in enumerate(active[:i])])
        return sets, active

    def _register(self, label: str, actor, *, color: str, opacity: float, kind: str) -> None:
        obj = self._objects.setdefault(label, {"label": label, "actors": [], "edges": [], "color": color, "opacity": opacity,
                                               "kind": kind})
        if kind == "edges":
            obj["edges"].append(actor)
        else:
            obj["actors"].append(actor)
        self._actor_label[_vtk_key(actor)] = label

    def _draw_scene(self) -> None:
        """Every body mesh as its section pieces with its edges; the ports; then the section's caps and outlines."""
        from meshdraw.style import EDGE_COLOR, surface_props

        sets, _active = self._clip_sets()
        bars_shown: set = set()
        for item in self._items:
            label = item["label"]
            if item.get("kind") == "lines":
                self._draw_lines(item, sets, bars_shown)
                continue
            mesh, edges, strength = self._display(item)
            scalar_kw = self._scalar_kwargs(item, mesh, bars_shown)
            for k, planes in enumerate(sets):
                # each piece its own NAME: pyvista names an actor after its dataset, and every piece shares the mesh, so
                # without one each piece replaced the last
                actor = self.plotter.add_mesh(mesh, render=False, name=f"{label}|piece{k}", **surface_props(item["opacity"]),
                                              **(scalar_kw if scalar_kw else {"color": item["color"]}))
                for p in planes:
                    actor.mapper.AddClippingPlane(p)
                self._register(label, actor, color=item["color"], opacity=item["opacity"], kind="mesh")
                self._add_measurable(actor, item["points"], item["faces"])
                if edges is not None:
                    e_actor = self.plotter.add_mesh(edges, color=EDGE_COLOR, line_width=1.4 if strength >= 1.0 else 1.0, render=False,
                                                    name=f"{label}|edges{k}", opacity=strength)
                    for p in planes:
                        e_actor.mapper.AddClippingPlane(p)
                    self._register(label, e_actor, color=EDGE_COLOR, opacity=1.0, kind="edges")
                    self._objects[label]["edge_strength"] = strength
        self._draw_ports()
        self._draw_overlays(sets)
        self._draw_section_marks()
        for label in list(self._objects):
            self._apply_appearance(label, render=False)

    def _scalar_kwargs(self, item, mesh, bars_shown: set) -> dict:
        """``add_mesh`` keywords colouring ``mesh`` by the item's scalars (per point or per cell), its bar shown once per title."""
        s = item.get("scalars")
        if s is None:
            return {}
        s = np.asarray(s, dtype=float).ravel()
        if len(s) == mesh.n_points:
            mesh.point_data["value"] = s
        elif len(s) == mesh.n_cells:
            mesh.cell_data["value"] = s
        else:
            return {}
        kw = {"scalars": "value", "cmap": item.get("cmap") or "viridis"}
        if item.get("clim") is not None:
            kw["clim"] = tuple(item["clim"])
        title = item.get("scalar_bar")
        if title and title not in bars_shown:
            bars_shown.add(title)
            kw["show_scalar_bar"] = True
            kw["scalar_bar_args"] = {"title": str(title)}
        else:
            kw["show_scalar_bar"] = False
        return kw

    def _draw_lines(self, item, sets, bars_shown: set) -> None:
        """A polyline item: one actor per section piece, coloured flat or by its scalars."""
        label = item["label"]
        if not len(item["points"]):
            return
        poly = pv.PolyData(np.asarray(item["points"], float), lines=np.asarray(item["lines"], np.int64))
        kw = self._scalar_kwargs(item, poly, bars_shown)
        for k, planes in enumerate(sets):
            actor = self.plotter.add_mesh(poly, line_width=float(item.get("width", 2.0)), opacity=float(item.get("opacity", 1.0)),
                                          render=False, name=f"{label}|lines{k}", **(kw if kw else {"color": item["color"]}))
            for p in planes:
                actor.mapper.AddClippingPlane(p)
            self._register(label, actor, color=item["color"], opacity=float(item.get("opacity", 1.0)), kind="lines")

    def _draw_ports(self) -> None:
        from meshdraw.style import PORT_COLOR, PORT_EDGE_COLOR

        for g in getattr(self, "_ports", []):
            label = f"port:{g['name']}"
            center = np.asarray(g["origin"], float)
            loop = g.get("outline")
            if loop is not None and len(loop) >= 3:
                # the port's own cross-section: a rectangle stays a rectangle
                loop = np.asarray(loop, float)
                face = pv.PolyData(loop, port_face_cells(loop))
                self._register(label, self.plotter.add_mesh(face, color=PORT_COLOR, opacity=0.45, render=False), color=PORT_COLOR,
                               opacity=0.45, kind="port")
                edge = pv.PolyData(np.vstack([loop, loop[:1]]), lines=np.array([len(loop) + 1, *range(len(loop) + 1)], dtype=np.int64))
                self._register(label, self.plotter.add_mesh(edge, color=PORT_EDGE_COLOR, line_width=2.0, render=False),
                               color=PORT_EDGE_COLOR, opacity=1.0, kind="port_line")
            else:
                disc = pv.Disc(center=center, inner=0.0, outer=max(float(g["radius"]), 1e-4), normal=np.asarray(g["axis"], float), c_res=48)
                self._register(label, self.plotter.add_mesh(disc, color=PORT_COLOR, opacity=0.45, render=False), color=PORT_COLOR,
                               opacity=0.45, kind="port")
            self._register(label, self.plotter.add_point_labels(center[None, :], [g["name"]], font_size=10, point_size=1,
                                                                shape_opacity=0.3, render=False),
                           color="", opacity=1.0, kind="port_label")
            if self.frames_check.isChecked() and g.get("xhat") is not None:
                for direction, color, frac in ((g["axis"], PORT_NORMAL_COLOR, 1.2), (g["xhat"], PORT_XHAT_COLOR, 0.9)):
                    arrow = pv.Arrow(start=center, direction=np.asarray(direction, float), scale=frac * max(float(g["radius"]), 1e-4),
                                     tip_length=0.25, tip_radius=0.08, shaft_radius=0.025)
                    self._register(label, self.plotter.add_mesh(arrow, color=color, render=False), color=color, opacity=1.0,
                                   kind="port_arrow")

    def _overlay_grid(self, ov):
        """A mesh overlay's tetrahedra as an UnstructuredGrid and its surface's element edges (made once per overlay)."""
        if "_grid" not in ov:
            nodes = np.asarray(ov["nodes"], float).reshape(-1, 3)
            tets = np.asarray(ov["tets"], np.int64).reshape(-1, 4)
            grid = pv.UnstructuredGrid({pv.CellType.TETRA: tets}, nodes)
            ov["_grid"] = grid
            ov["_edges"] = grid.extract_surface(algorithm="dataset_surface").extract_all_edges()
        return ov["_grid"], ov["_edges"]

    def add_show_toggle(self, category: str, text: str, tip: str = "", on: bool = True):
        """A Show-menu toggle for the overlays (and layers) of ``category`` (a subclass's own kinds of overlay)."""
        if self.plotter is None:
            return None
        act = self._show_menu.addAction(text)
        act.setCheckable(True)
        act.setChecked(on)
        if tip:
            act.setToolTip(tip)
        act.toggled.connect(lambda _on: self._refresh_appearance())
        self._category_checks[category] = act
        return act

    def category_shown(self, category: str) -> bool:
        act = self._category_checks.get(category)
        return act is None or act.isChecked()

    def _draw_overlays(self, sets) -> None:
        """The open boundaries and refinement boxes (each box's twelve edges and a tag, never clipped: they are regions, not
        surfaces), the meshes (every surface element's edges, clipped with the bodies; their cut through the volume is drawn
        on each section plane by ``_draw_section_marks``), and what a subclass draws besides (reference ``plane`` squares
        and ``line`` axes, ``points`` with labels, ``patch`` triangles: each in its overlay's ``category``)."""
        for k, ov in enumerate(self._overlays):
            label = ov["label"]
            category = ov.get("category")
            if ov["kind"] == "box":
                category = category or "boundary"
                c = np.asarray(ov["corners"], float).reshape(8, 3)
                pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
                lines = np.array([[2, a, b] for a, b in pairs], dtype=np.int64).ravel()
                color = ov.get("color") or (PML_COLOR if ov.get("pml") else REFINE_COLOR if category == "refine" else BOUNDARY_COLOR)
                actor = self.plotter.add_mesh(pv.PolyData(c, lines=lines), color=color, line_width=1.6, render=False, name=f"{label}|box",
                                              pickable=True)
                self._register(label, actor, color=color, opacity=1.0, kind="overlay")
                if ov.get("tag"):
                    tag = self.plotter.add_point_labels(c[7:8], [ov["tag"]], font_size=9, point_size=1, shape_opacity=0.25,
                                                        text_color=color, render=False, name=f"{label}|tag")
                    self._register(label, tag, color="", opacity=1.0, kind="overlay")
                self._objects[label]["category"] = category
            elif ov["kind"] == "plane":  # a reference plane: a translucent square and its outline
                c = np.asarray(ov["corners"], float).reshape(4, 3)
                color = ov.get("color") or REFERENCE_COLOR
                quad = pv.PolyData(c, np.array([4, 0, 1, 2, 3], dtype=np.int64))
                self._register(label, self.plotter.add_mesh(quad, color=color, opacity=0.18, render=False, name=f"{label}|plane",
                                                            pickable=False), color=color, opacity=0.18, kind="overlay")
                rim = pv.PolyData(np.vstack([c, c[:1]]), lines=np.array([5, 0, 1, 2, 3, 4], dtype=np.int64))
                self._register(label, self.plotter.add_mesh(rim, color=color, line_width=1.4, render=False, name=f"{label}|rim",
                                                            pickable=False), color=color, opacity=1.0, kind="overlay")
                self._overlay_tag(label, c[2:3], ov)
                self._objects[label]["category"] = category or "reference"
            elif ov["kind"] == "line":  # a reference axis
                p = np.asarray(ov["points"], float).reshape(-1, 3)
                color = ov.get("color") or REFERENCE_COLOR
                line = pv.PolyData(p, lines=np.array([len(p), *range(len(p))], dtype=np.int64))
                self._register(label, self.plotter.add_mesh(line, color=color, line_width=float(ov.get("width", 2.0)), render=False,
                                                            name=f"{label}|line", pickable=False), color=color, opacity=1.0, kind="overlay")
                self._overlay_tag(label, p[-1:], ov)
                self._objects[label]["category"] = category or "reference"
            elif ov["kind"] == "points":  # markers (a thin spot), each with its label
                p = np.asarray(ov["points"], float).reshape(-1, 3)
                color = ov.get("color") or HIGHLIGHT_COLOR
                self._register(label, self.plotter.add_mesh(pv.PolyData(p), color=color, point_size=float(ov.get("size", 12)),
                                                            render_points_as_spheres=True, render=False, name=f"{label}|pts",
                                                            pickable=False), color=color, opacity=1.0, kind="overlay")
                if ov.get("texts"):
                    lab = self.plotter.add_point_labels(p, list(ov["texts"]), font_size=9, point_size=1, shape_opacity=0.35,
                                                        text_color=color, always_visible=True, render=False, name=f"{label}|txt")
                    lab.SetPickable(False)
                    self._register(label, lab, color="", opacity=1.0, kind="overlay")
                self._objects[label]["category"] = category or "check"
            elif ov["kind"] == "patch":  # triangles drawn over a body (clipped with it)
                p = np.asarray(ov["points"], float).reshape(-1, 3)
                t = np.asarray(ov["triangles"], np.int64).reshape(-1, 3)
                if not len(t):
                    continue
                color = ov.get("color") or HIGHLIGHT_COLOR
                cells = np.hstack([np.full((len(t), 1), 3, np.int64), t]).ravel()
                patch = pv.PolyData(p, cells)
                for j, planes in enumerate(sets):
                    actor = self.plotter.add_mesh(patch, color=color, opacity=float(ov.get("opacity", 0.6)), render=False,
                                                  name=f"{label}|patch{j}", pickable=False)
                    for pl in planes:
                        actor.mapper.AddClippingPlane(pl)
                    self._register(label, actor, color=color, opacity=float(ov.get("opacity", 0.6)), kind="overlay")
                self._objects[label]["category"] = category or "highlight"
            elif ov["kind"] == "tets":
                _grid, edges = self._overlay_grid(ov)
                for j, planes in enumerate(sets):
                    actor = self.plotter.add_mesh(edges, color=MESH_LINE_COLOR, line_width=0.8, opacity=0.55, render=False,
                                                  name=f"{label}|edges{j}", pickable=False)
                    for p in planes:
                        actor.mapper.AddClippingPlane(p)
                    self._register(label, actor, color=MESH_LINE_COLOR, opacity=0.55, kind="overlay")
                self._objects[label]["category"] = "mesh"

    def _overlay_tag(self, label: str, at, ov) -> None:
        if ov.get("tag"):
            tag = self.plotter.add_point_labels(np.asarray(at, float).reshape(-1, 3), [ov["tag"]], font_size=9, point_size=1,
                                                shape_opacity=0.25, text_color=ov.get("color") or REFERENCE_COLOR, render=False,
                                                name=f"{label}|tag")
            tag.SetPickable(False)
            self._register(label, tag, color="", opacity=1.0, kind="overlay")

    def _overlay_on(self, label: str) -> bool:
        obj = self._objects.get(label) or {}
        on = self._category_checks.get(obj.get("category"))
        return on is None or on.isChecked()

    def set_overlays(self, overlays, render: bool = True) -> None:
        """What is drawn with the scene but is not geometry: ``[{label, kind: "box", corners (8, 3), tag, pml}]`` (a
        radiating part's open boundary, ``subject.air_boxes``) and ``[{label, kind: "tets", nodes (n, 3), tets (m, 4)}]``
        (the EMerge mesh of a Mesh preview), in the scene's coordinates. Kept across redraws until replaced or cleared;
        hidden from the Show menu (Open boundary, Mesh) or one by one (right-click)."""
        self._overlays = [dict(o) for o in (overlays or ())]
        if render and self.plotter is not None and (self._items or self._last_body is not None):
            self._rerender_last()

    def overlay_labels(self, category=None) -> list:
        """The overlays drawn (``category``: "boundary" or "mesh" alone)."""
        return [lbl for lbl, o in self._objects.items() if o["kind"] == "overlay" and (category is None or o.get("category") == category)]

    def _draw_section_marks(self) -> None:
        """On each plane: every visible mesh's trace as a dark outline, and the metal it cuts filled (``sections.cap``)."""
        from meshdraw import sections as sec
        from meshdraw.style import SECTION_CAP_COLOR, SECTION_LINE_COLOR

        for a in self._section_actors:
            try:
                self.plotter.remove_actor(a, render=False)
            except Exception:
                pass
        self._section_actors = []
        _sets, active = self._clip_sets()
        for i, (o, n) in enumerate(active):
            limits = [self._plane(j, -1, oj, nj) for j, (oj, nj) in enumerate(active) if j != i]
            metal = []
            for k, item in enumerate(self._items):
                if item.get("kind") == "lines":
                    continue
                visible, _color, _opacity = self.appearance.resolve(item["label"], item["color"], item["opacity"])
                if not visible:
                    continue
                trace = sec.outline(item["points"], item["faces"], o, n)
                if trace is None:
                    continue
                if item.get("solid", True):
                    metal.append(trace)
                actor = self.plotter.add_mesh(trace, color=SECTION_LINE_COLOR, line_width=2.2, render=False, name=f"section{i}|line{k}")
                self._limit(actor, limits, item["label"])
            filled = sec.cap(metal)
            if filled is not None:
                actor = self.plotter.add_mesh(filled, color=SECTION_CAP_COLOR, ambient=0.35, diffuse=0.65, specular=0.0, render=False,
                                              name=f"section{i}|cap")
                self._limit(actor, limits, None)
            for k, ov in enumerate(self._overlays):  # a mesh's elements where the plane cuts the volume
                if ov["kind"] != "tets" or not self.mesh_check.isChecked() or not self.appearance.get(ov["label"]).visible:
                    continue
                grid, _edges = self._overlay_grid(ov)
                try:
                    cut = grid.slice(normal=tuple(n), origin=tuple(o))
                except Exception:
                    continue
                if not cut.n_cells:
                    continue
                actor = self.plotter.add_mesh(cut.extract_all_edges(), color=MESH_LINE_COLOR, line_width=0.8, render=False,
                                              name=f"section{i}|mesh{k}", pickable=False)
                self._limit(actor, limits, ov["label"])

    def _limit(self, actor, planes, label) -> None:
        for p in planes:
            actor.mapper.AddClippingPlane(p)
        if label is not None:
            self._actor_label[_vtk_key(actor)] = label
        self._section_actors.append(actor)

    def _apply_section(self, structural: bool = False) -> None:
        """The section changed: a new set of planes redraws the scene; a moved plane moves its clipping planes and remakes the
        caps and outlines (fast enough to follow a slider)."""
        if self.plotter is None or not self._items:
            return
        _sets, active = self._clip_sets()
        n_built = len({k[0] for k in self._vtk_planes})
        if structural or len(active) != n_built:
            camera = self.plotter.camera_position
            if self._last_body is not None:
                self._show_body(*self._last_body, keep_camera=True)
            self.plotter.camera_position = camera
            self.plotter.render()
            return
        self._draw_section_marks()
        self.plotter.render()

    # -- what the user chose about each object (right-click) ------------------------------------------------------------

    def _apply_appearance(self, label: str, render: bool = True) -> None:
        obj = self._objects.get(label)
        if obj is None:
            return
        visible, color, opacity = self.appearance.resolve(label, obj["color"], obj["opacity"])
        show_edges = self.edges_check.isChecked()
        lit = label in self._highlight
        if obj["kind"] == "overlay":
            visible = visible and self._overlay_on(label)
        for a in obj["actors"]:
            a.SetVisibility(bool(visible))
            prop = getattr(a, "prop", None)
            if prop is not None and obj["kind"] in ("mesh", "lines"):
                prop.opacity = opacity
                if not getattr(a, "mapper", None) or a.mapper.GetScalarVisibility() == 0:  # a scalar-coloured object keeps its map
                    prop.color = _mix(color, HIGHLIGHT_COLOR, 0.35) if lit and color else color
            elif prop is not None and obj["kind"] == "port" and self.appearance.get(label).opacity is not None:
                prop.opacity = opacity
        strength = float(obj.get("edge_strength", 1.0))
        for e in obj["edges"]:
            e.SetVisibility(bool(visible and (show_edges or lit)))
            if getattr(e, "prop", None) is not None:  # dense lines fade (``edge_strength``); a see-through object's fade more
                e.prop.opacity = strength if opacity >= 1.0 else max(0.1, 0.6 * opacity * strength)
                e.prop.color = HIGHLIGHT_COLOR if lit else EDGE_LINE_COLOR
                e.prop.line_width = 2.4 if lit else (1.4 if strength >= 1.0 else 1.0)
        if render and self.plotter is not None:
            self._draw_section_marks()
            self.plotter.render()

    def highlight(self, labels) -> None:
        """Draw ``labels`` highlighted (an accent tint and edges): the navigator's selection. Empty clears."""
        old, self._highlight = self._highlight, set(labels or ())
        for label in old | self._highlight:
            self._apply_appearance(label, render=False)
        if self.plotter is not None:
            self.plotter.render()

    def labels_of(self, member: str) -> list:
        """The objects drawn for device ``member`` (its meshes ``member/...``), or ``port:<member>`` for a port."""
        return [lbl for lbl in self.object_labels() if lbl.rpartition("/")[0] == member or lbl == f"port:{member}"]

    def object_labels(self) -> list:
        """Every object the scene draws, in drawing order (meshes, then ports)."""
        return list(self._objects)

    def hide(self, label: str) -> None:
        self.appearance.hide(label)
        self._apply_appearance(label)

    def show_object(self, label: str) -> None:
        self.appearance.show(label)
        self._apply_appearance(label)

    def isolate(self, label: str) -> None:
        self.appearance.isolate([label], self.object_labels())
        self._refresh_appearance()

    def set_object_opacity(self, label: str, opacity: float) -> None:
        self.appearance.set_opacity(label, opacity)
        self._apply_appearance(label)

    def set_object_color(self, label: str, color: str) -> None:
        self.appearance.set_color(label, color)
        self._apply_appearance(label)

    def show_all(self) -> None:
        self.appearance.show_all()
        self._refresh_appearance()

    def reset_appearance(self, label=None) -> None:
        self.appearance.reset(label)
        self._refresh_appearance()

    def _refresh_appearance(self) -> None:
        for label in self.object_labels():
            self._apply_appearance(label, render=False)
        if self.plotter is not None:
            self._draw_section_marks()
            self.plotter.render()

    # -- the view's state (saved with the project per subject) --------------------------------------------------------

    def view_state(self) -> dict:
        """Where the user left this view: the section and its planes, every object's display choice, the Show toggles and
        the camera. Plain data (JSON): the workspace keeps one per subject and saves them with the project."""
        from dataclasses import asdict

        state = {
            "section": bool(self.section_button.isChecked()) if self.plotter is not None else False,
            "planes": [{"enabled": p.enabled, "axis": p.axis, "fraction": float(p.fraction), "flip": bool(p.flip),
                        "normal": [float(v) for v in p.normal]} for p in self.section.planes],
            "appearance": {lbl: asdict(c) for lbl, c in self.appearance.choices.items()},
        }
        if self.plotter is None:
            return state
        state["show"] = {"edges": self.edges_check.isChecked(), "frames": self.frames_check.isChecked(),
                         "boundary": self.boundary_check.isChecked(), "mesh": self.mesh_check.isChecked(),
                         "refine": self.refine_check.isChecked()}
        cam = self.plotter.camera
        state["camera"] = {"position": [float(v) for v in cam.GetPosition()], "focal_point": [float(v) for v in cam.GetFocalPoint()],
                           "view_up": [float(v) for v in cam.GetViewUp()], "parallel": bool(cam.GetParallelProjection()),
                           "parallel_scale": float(cam.GetParallelScale()), "view_angle": float(cam.GetViewAngle())}
        return state

    def set_view_state(self, state, redraw: bool = True) -> None:
        """Put the view back as :meth:`view_state` described it (what is missing stays as it is; a stale label is harmless:
        a choice applies to an object only if it is drawn). ``redraw`` False: only remember it, the next ``set_body`` draws
        it (and takes its camera)."""
        from meshdraw.appearance import Appearance, Choice
        from meshdraw.sections import SectionPlane

        state = dict(state or {})
        if "appearance" in state:
            self.appearance = Appearance({str(k): Choice(**{f: v for f, v in dict(c).items() if f in ("visible", "opacity", "color")})
                                          for k, c in dict(state["appearance"] or {}).items()})
        for plane, d in zip(self.section.planes, state.get("planes") or ()):
            d = dict(d)
            fresh = SectionPlane(enabled=bool(d.get("enabled", False)), axis=str(d.get("axis", plane.axis)),
                                 fraction=float(d.get("fraction", 0.5)), flip=bool(d.get("flip", False)),
                                 normal=tuple(float(v) for v in d.get("normal", plane.normal)))
            plane.__dict__.update(fresh.__dict__)
        if self.plotter is None:
            return
        self._syncing = True
        try:
            for row, plane in zip(self._plane_rows, self.section.planes):
                row["on"].setChecked(plane.enabled)
                row["axis"].setCurrentIndex(max(0, row["axis"].findData(plane.axis)))
                row["slider"].setValue(int(round(plane.fraction * SECTION_STEPS)))
                row["flip"].setChecked(plane.flip)
                row["handle"].setChecked(False)
            show = dict(state.get("show") or {})
            for key, act in (("edges", self.edges_check), ("frames", self.frames_check), ("boundary", self.boundary_check),
                             ("mesh", self.mesh_check), ("refine", self.refine_check)):
                if key in show:
                    act.blockSignals(True)
                    act.setChecked(bool(show[key]))
                    act.blockSignals(False)
            on = bool(state.get("section", False))
            self.section_button.blockSignals(True)
            self.section_button.setChecked(on)
            self.section_button.blockSignals(False)
            self.section_bar.setVisible(on)
        finally:
            self._syncing = False
        self._clear_plane_widget()
        cam = state.get("camera")
        if cam is not None:
            self.perspective_check.blockSignals(True)
            self.perspective_check.setChecked(not cam.get("parallel", False))
            self.perspective_check.blockSignals(False)
        if not redraw:
            self._pending_camera = cam
            return
        self._rerender_last()  # the planes, the toggles and every choice, drawn
        self._set_camera(cam)
        self._label_planes()
        if self.plotter is not None:
            self.plotter.render()

    def _set_camera(self, cam) -> None:
        if cam is not None and self.plotter is not None:
            c = self.plotter.camera
            c.SetPosition(*cam["position"])
            c.SetFocalPoint(*cam["focal_point"])
            c.SetViewUp(*cam["view_up"])
            c.SetParallelProjection(bool(cam.get("parallel", False)))
            if cam.get("parallel_scale"):
                c.SetParallelScale(float(cam["parallel_scale"]))
            if cam.get("view_angle"):
                c.SetViewAngle(float(cam["view_angle"]))
            self.plotter.renderer.ResetCameraClippingRange()

    # -- a cheaper frame while the camera moves ------------------------------------------------------------------------

    def scene_cost(self) -> int:
        """What a frame costs, roughly: the triangles drawn (a translucent object's three times: depth peeling draws it
        in layers) plus the mesh overlay's lines."""
        cost = 0
        for item in self._items:
            visible, _c, opacity = self.appearance.resolve(item["label"], item["color"], item["opacity"])
            if visible:
                cost += (len(item.get("faces", item.get("lines", ()))) // 4) * (3 if opacity < 1.0 else 1)
        for ov in self._overlays:
            if ov["kind"] == "tets" and "_edges" in ov:
                cost += int(ov["_edges"].n_cells)
        return cost

    def fast_interaction_cost(self) -> int:
        """The scene cost above which a moving camera draws the cheaper frame (``style.FAST_INTERACTION_COST``)."""
        return FAST_INTERACTION_COST

    def _on_interaction_start(self) -> None:
        if self.plotter is None or self._fast is not None or self.scene_cost() < self.fast_interaction_cost():
            return
        ren = self.plotter.renderer
        hidden = [a for obj in self._objects.values() for a in (obj["edges"] + (obj["actors"] if obj.get("category") == "mesh" else []))
                  if a.GetVisibility()]
        hidden += [a for a in self._section_actors if self._actor_label.get(_vtk_key(a), "").startswith("mesh:") and a.GetVisibility()]
        for a in hidden:
            a.SetVisibility(False)
        from meshdraw.style import ambient_occlusion

        self._fast = {"hidden": hidden, "peeling": bool(ren.GetUseDepthPeeling()), "ssao": ambient_occlusion()}
        ren.SetUseDepthPeeling(False)
        if self._fast["ssao"]:
            try:
                self.plotter.disable_ssao()
            except Exception:  # pragma: no cover - renderer dependent
                pass

    def _on_interaction_end(self) -> None:
        fast, self._fast = self._fast, None
        if fast is None or self.plotter is None:
            return
        ren = self.plotter.renderer
        ren.SetUseDepthPeeling(fast["peeling"])
        if fast["ssao"]:
            from meshdraw.style import _ssao_radius

            try:
                self.plotter.enable_ssao(radius=_ssao_radius(self.plotter), bias=0.01, kernel_size=64)
            except Exception:  # pragma: no cover - renderer dependent
                pass
        for a in fast["hidden"]:
            a.SetVisibility(True)
        self.plotter.render()

    def label_at(self, x: int, y: int):
        """The label of the object under the view pixel ``(x, y)`` (VTK's coordinates), or None."""
        if self.plotter is None:
            return None
        self._picker.Pick(x, y, 0, self.plotter.renderer)
        actor = self._picker.GetActor()
        return None if actor is None else self._actor_label.get(_vtk_key(actor))

    def context_menu(self, label=None) -> QMenu:
        """The right-click menu: the object's commands when ``label`` names one, then the scene's and the display's."""
        from meshdraw.appearance import OPACITIES, display_name

        menu = QMenu(self)
        if label is not None:
            head = menu.addAction(display_name(label))
            head.setEnabled(False)
            menu.addAction("Hide", lambda: self.hide(label))
            menu.addAction("Show only this", lambda: self.isolate(label))
            sub = menu.addMenu("Transparency")
            current = self.appearance.get(label).opacity
            for value, text in OPACITIES:
                act = sub.addAction(text, lambda v=value: self.set_object_opacity(label, v))
                act.setCheckable(True)
                act.setChecked(current is not None and abs(current - value) < 1e-6)
            menu.addAction("Colour…", lambda: self._pick_color(label))
            menu.addAction("Reset this object", lambda: self.reset_appearance(label))
            menu.addSeparator()
        hidden = self.appearance.hidden(self.object_labels())
        show = menu.addAction("Show all", self.show_all)
        show.setEnabled(bool(hidden))
        if hidden:
            sub = menu.addMenu(f"Hidden ({len(hidden)})")
            for lbl in hidden:
                sub.addAction(display_name(lbl), lambda lbl=lbl: self.show_object(lbl))
        menu.addAction("Reset all objects", lambda: self.reset_appearance())
        menu.addSeparator()
        display = menu.addMenu("Display (this machine)")
        from meshdraw.style import ambient_occlusion, antialiasing

        aa = antialiasing()
        for key, text in (("msaa", "Anti-aliasing: MSAA 4x (crisp)"), ("fxaa", "Anti-aliasing: FXAA (soft, cheap)"),
                          ("none", "Anti-aliasing: off")):
            act = display.addAction(text, lambda k=key: self.set_display(antialiasing=k))
            act.setCheckable(True)
            act.setChecked(aa == key)
        ao = display.addAction("Ambient occlusion (depth in cavities)", lambda: self.set_display(ambient_occlusion=not ambient_occlusion()))
        ao.setCheckable(True)
        ao.setChecked(ambient_occlusion())
        return menu

    def _pick_color(self, label: str) -> None:  # pragma: no cover - dialog
        from PyQt6.QtGui import QColor
        from PyQt6.QtWidgets import QColorDialog

        obj = self._objects.get(label, {})
        start = QColor(self.appearance.get(label).color or obj.get("color") or "#cccccc")
        c = QColorDialog.getColor(start, self, "Object colour")
        if c.isValid():
            self.set_object_color(label, c.name())

    def set_display(self, *, antialiasing=None, ambient_occlusion=None) -> None:
        """This machine's display settings (``view_style``); applied to this view now, to every view built after."""
        from meshdraw.style import Settings, settings, style_plotter

        if antialiasing is not None:
            settings.set(Settings.ANTIALIASING, str(antialiasing))
        if ambient_occlusion is not None:
            settings.set(Settings.AMBIENT_OCCLUSION, bool(ambient_occlusion))
        if self.plotter is not None:
            try:
                if ambient_occlusion is False:
                    self.plotter.disable_ssao()
                style_plotter(self.plotter)
            except Exception:  # pragma: no cover - renderer dependent
                pass
            self.plotter.render()

    def _on_rpress(self, *_):
        self._rpress = self._event_xy()

    def _on_rrelease(self, *_):  # pragma: no cover - needs a live window
        start, self._rpress = self._rpress, None
        if start is None:
            return
        x, y = self._event_xy()
        if abs(x - start[0]) > 4 or abs(y - start[1]) > 4:
            return  # a drag: the view zoomed
        from PyQt6.QtGui import QCursor

        self.context_menu(self.label_at(x, y)).exec(QCursor.pos())

    def _add_measurable(self, actor, points, faces) -> None:
        if not hasattr(self, "_measure_meshes"):
            return
        self._measure_actors[_vtk_key(actor)] = len(self._measure_meshes)
        self._measure_meshes.append((np.asarray(points, float), np.asarray(faces, np.int64)))

    def _forget_measurables(self) -> None:
        if not hasattr(self, "_measure_meshes"):
            return
        self._measure_meshes, self._measure_actors, self._topologies = [], {}, {}
        self._measure_overlays = []
        self.measure_selection, self.measure_result = [], {}
        if self.measure_button.isChecked():
            self.measure_label.setText(MEASURE_HINT)

    def topology(self, index: int):
        """The faces / edges / vertices of drawn mesh ``index`` (built once per drawing)."""
        from meshdraw.measure import MeshTopology

        if index not in self._topologies:
            pts, faces = self._measure_meshes[index]
            self._topologies[index] = MeshTopology(pts, faces)
        return self._topologies[index]

    def _on_measure_toggled(self, on: bool) -> None:
        self.measure_label.setVisible(on)
        self.measure_clear()
        if on:
            self.measure_label.setText(MEASURE_HINT)

    def measure_clear(self) -> None:
        self.measure_selection, self.measure_result = [], {}
        self._draw_measure()
        if self.measure_button.isChecked():
            self.measure_label.setText(MEASURE_HINT)

    def measure_pick(self, index: int, cell: int, point, tol: float):
        """Add (or, picked again, remove) what is at ``point`` [m] on cell ``cell`` of drawn mesh ``index``: a vertex or an
        edge within ``tol`` [m] of it, else the face; a third pick starts a new measurement. Returns the report."""
        from meshdraw.measure import Entity

        ent = self.topology(index).pick(cell, point, tol)
        ent = Entity(ent.kind, index, ent.index)
        if ent in self.measure_selection:
            self.measure_selection.remove(ent)
        elif len(self.measure_selection) >= 2:
            self.measure_selection = [ent]
        else:
            self.measure_selection.append(ent)
        return self._report()

    def _report(self) -> dict:
        from meshdraw import measure as measure3d
        if not self.measure_selection:
            self.measure_result = {}
            self.measure_label.setText(MEASURE_HINT)
        else:
            topos = {i: self.topology(i) for i in {e.mesh for e in self.measure_selection}}
            self.measure_result = measure3d.measure(topos, self.measure_selection, unit=self._units.length())
            self.measure_label.setText("\n".join(self.measure_result["lines"]))
        self._draw_measure()
        return self.measure_result

    def _draw_measure(self) -> None:
        """Highlight the picked entities (a face tinted, an edge thick, a vertex a ball) and the closest-points line."""
        if self.plotter is None:
            return
        for a in self._measure_overlays:
            try:
                self.plotter.remove_actor(a, render=False)
            except Exception:
                pass
        self._measure_overlays = []
        for e in self.measure_selection:
            topo = self.topology(e.mesh)
            size = float(np.ptp(topo.points, axis=0).max()) or 1.0
            if e.kind == "face":
                ts = topo.faces[e.index]
                tri = topo.tris[ts]
                lift = topo.tri_normal[ts].mean(axis=0) * 2e-4 * size  # just off the surface: no z-fight
                cells = np.hstack([np.full((len(tri), 1), 3), np.arange(3 * len(tri)).reshape(-1, 3)]).ravel()
                patch = pv.PolyData(topo.points[tri].reshape(-1, 3) + lift, cells)
                self._measure_overlays.append(self.plotter.add_mesh(patch, color=MEASURE_COLOR, opacity=0.65, pickable=False,
                                                                    render=False))
            elif e.kind == "edge":
                p = topo.edge_points(e.index)
                line = pv.PolyData(p, lines=np.array([len(p), *range(len(p))], dtype=np.int64))
                self._measure_overlays.append(self.plotter.add_mesh(line, color=MEASURE_COLOR, line_width=6, pickable=False,
                                                                    render=False))
            else:
                p = topo.points[[int(topo.vertices[e.index])]]
                self._measure_overlays.append(self.plotter.add_mesh(pv.PolyData(p), color=MEASURE_COLOR, point_size=16,
                                                                    render_points_as_spheres=True, pickable=False, render=False))
        closest = (self.measure_result or {}).get("closest")
        if closest is not None:
            p, q = (np.asarray(x, float) for x in closest)
            seg = pv.PolyData(np.vstack([p, q]), lines=np.array([2, 0, 1], dtype=np.int64))
            self._measure_overlays.append(self.plotter.add_mesh(seg, color=MEASURE_LINE_COLOR, line_width=3, pickable=False,
                                                                render=False))
            self._measure_overlays.append(self.plotter.add_mesh(pv.PolyData(np.vstack([p, q])), color=MEASURE_LINE_COLOR, point_size=9,
                                                                render_points_as_spheres=True, pickable=False, render=False))
        self.plotter.render()

    def _event_xy(self):
        return tuple(self.plotter.iren.interactor.GetEventPosition())

    def _on_press(self, *_):
        self._press = self._event_xy()

    def _on_release(self, *_):
        start, self._press = self._press, None
        if start is None:
            return
        x, y = self._event_xy()
        if abs(x - start[0]) > 4 or abs(y - start[1]) > 4:
            return  # a drag: the view turned
        if not self.measure_button.isChecked():
            label = self.label_at(x, y)  # a click on an object selects it in the navigator (the host listens)
            if label is not None:
                self.object_clicked.emit(label)
            return
        self._picker.Pick(x, y, 0, self.plotter.renderer)
        actor = self._picker.GetActor()
        index = self._measure_actors.get(_vtk_key(actor)) if actor is not None else None
        if index is None:
            self.measure_clear()  # a click on empty space starts again
            return
        point = np.asarray(self._picker.GetPickPosition(), float)
        self.measure_pick(index, int(self._picker.GetCellId()), point, self._pixel_size(point) * MEASURE_PICK_PX)

    def _pixel_size(self, point) -> float:  # pragma: no cover - needs a live render window
        """Metres per screen pixel at ``point`` (what the vertex / edge snap distance is measured in)."""
        cam = self.plotter.camera
        h = max(int(self.plotter.window_size[1]), 1)
        if cam.GetParallelProjection():
            return 2.0 * cam.GetParallelScale() / h
        d = float(np.linalg.norm(np.asarray(cam.GetPosition()) - np.asarray(point)))
        return 2.0 * d * math.tan(math.radians(cam.GetViewAngle()) / 2.0) / h

    def body_stl_bytes(self) -> bytes:
        """STL of the drawn body: every mesh item ``exportable`` (raises when nothing is drawn)."""
        from meshdraw.mesh import mesh_stl_bytes

        last = getattr(self, "_last_body", None)
        if last is None or last[0] is None:
            raise ValueError("nothing drawn")
        items = self.render_items(last[0], last[2])
        real = [(it["points"], it["faces"]) for it in items if it.get("kind", "mesh") == "mesh" and self.exportable(it["label"])]
        if not real:
            raise ValueError("nothing drawn")
        return mesh_stl_bytes(real)

    def _tip(self, key: str) -> str:
        return self._tips.get(key, DEFAULT_TIPS.get(key, ""))

    def cleanup_plotter(self) -> None:
        """Finalize the render window before close (Umbra's teardown)."""
        if self.plotter is None:
            return
        try:
            ren_win = self.plotter.ren_win
            if ren_win is not None:
                ren_win.Finalize()
        except Exception:
            pass
        try:
            self.plotter.close()
        except Exception:
            pass
        self.plotter = None

    def closeEvent(self, event) -> None:  # pragma: no cover - GUI teardown
        self.cleanup_plotter()
        super().closeEvent(event)



__all__ = ["DEFAULT_TIPS", "HIGHLIGHT_COLOR", "MEASURE_HINT", "MEASURE_PICK_PX", "MillimetreUnits", "SECTION_STEPS", "View3DWidget",
           "port_face_cells"]


def _vtk_key(actor) -> str:
    """An actor's identity as VTK knows it: its C++ address (a Python wrapper of it may be another object, with another id)."""
    return actor.GetAddressAsString("vtkObject")
