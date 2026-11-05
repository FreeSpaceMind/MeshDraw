"""The widget: scenes, sections, display choices, scalar items, polylines, a plain view, the view state, measure."""

import json

import numpy as np
import pytest

from meshdraw import scene
from meshdraw.mesh import box_mesh

pytest.importorskip("PyQt6.QtWidgets")


def _view(qapp, **kw):
    pytest.importorskip("pyvistaqt")
    from meshdraw.widget import View3DWidget

    w = View3DWidget(**kw)
    if w.plotter is None:
        pytest.skip("no 3D view here")
    return w


def _box(lo, hi):
    return box_mesh(np.array(lo, float), np.array(hi, float))


def test_the_view_cuts_hides_and_fills_and_its_menu_offers_the_hidden(qapp):
    from meshdraw.widget import _vtk_key

    w = _view(qapp)
    w.set_body(([_box((-1, -1, 0), (1, 1, 2)), _box((-3, -3, 0), (3, 3, 0.2))], ["blk/notional", "base/wall"]), "t")
    assert w.object_labels() == ["blk/notional", "base/wall"]
    w.section_button.setChecked(True)
    w._plane_rows[1]["on"].setChecked(True)
    for lbl in w.object_labels():
        actors = w._objects[lbl]["actors"]
        assert len(actors) == 2 and [a.mapper.GetClippingPlanes().GetNumberOfItems() for a in actors] == [1, 2]
    assert [a for a in w._section_actors if _vtk_key(a) not in w._actor_label]  # the closed block is filled where cut
    w.hide("base/wall")
    assert not any(a.GetVisibility() for a in w._objects["base/wall"]["actors"])
    menu = w.context_menu(None)
    hidden = [a for a in menu.actions() if a.menu() is not None and a.text().startswith("Hidden")]
    assert hidden and [x.text() for x in hidden[0].menu().actions()] == ["base wall"]
    w.set_object_opacity("base/wall", 0.25)
    w._rerender_last()
    assert w._objects["base/wall"]["actors"][0].prop.opacity == pytest.approx(0.25)
    assert w.body_stl_bytes()[:5] == b"meshd"
    w.cleanup_plotter()
    w.deleteLater()


def test_scalar_surfaces_and_polylines_draw_and_a_plain_view_has_no_tools(qapp):
    w = _view(qapp, tools=False)
    assert not w.toolbar_widget.isVisibleTo(w)
    x, y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 8), indexing="ij")
    dish = scene.grid_item(x, y, 0.2 * (x**2 + y**2), "dish/main", scalars=x, scalar_bar="x [m]")
    rays = scene.lines_item([np.array([[0, 0, 1], [0.5, 0.5, 0]]), np.array([[0, 0, 1], [-0.5, 0.2, 0]])], "rays", scalars=[0.0, 1.0])
    block = scene.mesh_item(*_box((-0.1, -0.1, 0.9), (0.1, 0.1, 1.1)), "feed/body")
    w.set_items([dish, rays, block], "scene", ports=[scene.port_glyph("p", (0, 0, 1.1), (0, 0, 1), 0.05)])
    assert w.object_labels() == ["dish/main", "rays", "feed/body", "port:p"]
    assert w._objects["rays"]["kind"] == "lines"
    assert w._objects["dish/main"]["actors"][0].mapper.GetScalarVisibility() == 1
    assert len([a for a in w.plotter.renderer.actors if "ScalarBar" in type(w.plotter.renderer.actors[a]).__name__]) == 1
    w.section_button.setChecked(True)  # the section still works on a plain view (right-click, a host's own controls)
    assert len(w._objects["rays"]["actors"]) == 1 and len(w._objects["dish/main"]["actors"]) == 1
    w.cleanup_plotter()
    w.deleteLater()


def test_the_view_state_round_trips(qapp):
    w = _view(qapp)
    body = ([_box((-1, -1, 0), (1, 1, 2)), _box((-3, -3, 0), (3, 3, 0.2))], ["blk/wall", "base/wall"])
    w.set_body(body, "t")
    w.section_button.setChecked(True)
    w._plane_rows[0]["slider"].setValue(300)
    w.hide("base/wall")
    w.set_object_opacity("blk/wall", 0.5)
    w.plotter.camera.SetPosition(10.0, -7.0, 4.0)
    state = json.loads(json.dumps(w.view_state()))
    w.set_view_state({"section": False, "planes": [{"enabled": False}] * 3, "appearance": {}})
    assert not w.section_button.isChecked() and not w.appearance.hidden(w.object_labels())
    w.set_view_state(state)
    assert w.section_button.isChecked() and w.section.planes[0].fraction == pytest.approx(0.3)
    assert not any(a.GetVisibility() for a in w._objects["base/wall"]["actors"])
    np.testing.assert_allclose(w.plotter.camera.GetPosition(), [10.0, -7.0, 4.0])
    w.cleanup_plotter()
    w.deleteLater()


def test_the_view_measures_what_it_draws(qapp):
    w = _view(qapp)
    b1 = box_mesh(np.zeros(3), np.array([0.02, 0.01, 0.04]))
    b2 = box_mesh(np.array([0.03, 0.0, 0.0]), np.array([0.05, 0.01, 0.04]))
    w.set_body(([b1, b2], ["a/wall", "b/wall"]), "two blocks")
    assert len(w._measure_meshes) == 2
    w.measure_button.setChecked(True)
    t0, t1 = w.topology(0), w.topology(1)

    def face(t, normal, where):
        best = None
        for i in range(len(t.faces)):
            inf = t.face_info(i)
            if inf["kind"] == "plane" and abs(abs(float(inf["normal"] @ np.asarray(normal, float))) - 1) < 1e-9:
                d = float(np.linalg.norm(inf["point"] - np.asarray(where)))
                best = (d, i) if best is None or d < best[0] else best
        return best[1]

    f0, f1 = face(t0, (1, 0, 0), (0.02, 0.005, 0.02)), face(t1, (1, 0, 0), (0.03, 0.005, 0.02))
    w.measure_pick(0, int(t0.cell_of_tri[t0.faces[f0][0]]), np.array([0.02, 0.005, 0.02]), 1e-4)
    r = w.measure_pick(1, int(t1.cell_of_tri[t1.faces[f1][0]]), np.array([0.03, 0.005, 0.02]), 1e-4)
    assert "Normal distance: 10 mm (parallel)" in r["lines"] and len(w._measure_overlays) == 4
    w.measure_clear()
    assert w.measure_selection == []
    w.cleanup_plotter()
    w.deleteLater()


def test_offscreen_screenshot_writes_a_png(tmp_path):
    pytest.importorskip("pyvista")
    from meshdraw.offscreen import screenshot

    items = [scene.mesh_item(*_box((0, 0, 0), (1, 1, 1)), "a/wall", scalars=np.linspace(0, 1, 8), scalar_bar="s")]
    try:
        screenshot(items, str(tmp_path / "s.png"), title="box")
    except Exception as exc:  # no OpenGL offscreen on this machine
        pytest.skip(f"offscreen render unavailable: {exc}")
    assert (tmp_path / "s.png").stat().st_size > 1000
