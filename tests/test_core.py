"""The Qt-free core: section planes, display choices, scene items, mesh helpers."""

import numpy as np
import pytest

from meshdraw import appearance as ap
from meshdraw import mesh as mm
from meshdraw import scene
from meshdraw import sections as sec
from meshdraw import style

BOUNDS = (0.0, 10.0, 0.0, 20.0, 0.0, 30.0)


def test_a_plane_removes_the_side_its_axis_points_to_and_flip_the_other():
    p = sec.SectionPlane(enabled=True, axis="x", fraction=0.25)
    o, n = p.place(BOUNDS)
    np.testing.assert_allclose(o, [2.5, 10.0, 15.0])
    np.testing.assert_allclose(n, [-1.0, 0.0, 0.0])
    p.flip = True
    np.testing.assert_allclose(p.place(BOUNDS)[1], [1.0, 0.0, 0.0])
    q = sec.SectionPlane()
    q.set_from((5.0, 5.0, 12.0), (0.0, 0.0, 3.0), BOUNDS)
    assert q.enabled and q.axis == "free" and q.fraction == pytest.approx(0.4)


def test_two_planes_cut_away_a_quarter_three_an_octant_and_the_pieces_are_disjoint():
    planes = [sec.SectionPlane(enabled=True, axis=a) for a in ("x", "y", "z")]
    corners = np.array([[x, y, z] for x in (1.0, 9.0) for y in (1.0, 19.0) for z in (1.0, 29.0)])
    three = [p.place(BOUNDS) for p in planes]
    assert sec.kept(corners, three).sum() == 7
    pts = np.random.default_rng(0).uniform([0, 0, 0], [10, 20, 30], size=(2000, 3))
    owners = np.zeros(len(pts), int)
    for clips in sec.pieces(three):
        inside = np.ones(len(pts), bool)
        for o, n in clips:
            inside &= (pts - o) @ n >= 0
        owners += inside
    np.testing.assert_array_equal(owners, sec.kept(pts, three).astype(int))
    assert sec.pieces([]) == [[]]


def test_a_closed_solid_is_filled_where_cut_and_an_open_wall_is_only_outlined():
    pytest.importorskip("pyvista")
    pts, faces = mm.box_mesh(np.array([-1.0, -2.0, 0.0]), np.array([1.0, 2.0, 3.0]))
    assert sec.is_closed(pts, faces)
    filled = sec.cap([sec.outline(pts, faces, (0, 0, 1.5), (0, 0, 1))])
    assert filled is not None and filled.area == pytest.approx(8.0)


def test_display_choices_are_kept_by_label():
    a = ap.Appearance()
    a.hide("h.section/wall")
    a.set_opacity("h.exterior/exterior", 0.25)
    a.set_color("h.exterior/exterior", "#ff0000")
    assert a.resolve("h.exterior/exterior", "#aaaaaa", 0.4) == (True, "#ff0000", 0.25)
    assert a.hidden(["h.section/wall", "h.exterior/exterior"]) == ["h.section/wall"]
    a.isolate(["port:h.p1"], ["h.section/wall", "h.exterior/exterior", "port:h.p1"])
    assert a.hidden(["h.section/wall", "h.exterior/exterior", "port:h.p1"]) == ["h.section/wall", "h.exterior/exterior"]
    a.show_all()
    assert not a.hidden(["h.section/wall", "h.exterior/exterior"])
    assert ap.display_name("port:h.p1") == "Port h.p1" and ap.display_name("qr/fem_metal") == "qr metal"


def test_body_items_tint_members_and_style_leaves_as_data():
    box = mm.box_mesh(np.zeros(3), np.ones(3))
    items = scene.body_items([box, box, box, box], ["a/wall", "b/wall", "a/exterior", "c/fem_body"])
    assert [it["label"] for it in items] == ["a/wall", "b/wall", "a/exterior", "c/fem_body"]
    assert items[0]["color"] != items[1]["color"] and items[0]["color"] == style.MEMBER_TINTS[0]
    assert items[2]["color"] == style.EXTERIOR_TINT and items[2]["opacity"] == style.EXTERIOR_OPACITY
    assert items[3]["solid"] is False and items[0]["solid"] is True
    custom = scene.body_items([box], ["dish/surface"], leaf_styles={"surface": ("#123456", 0.9)}, colors={"dish/surface": "#abcdef"})
    assert custom[0]["color"] == "#abcdef" and custom[0]["opacity"] == 0.9


def test_grid_and_lines_items_carry_scalars():
    x, y = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 3), indexing="ij")
    z = x**2 + y**2
    g = scene.grid_item(x, y, z, "dish/main", scalars=z, scalar_bar="z")
    assert g["points"].shape == (12, 3) and g["faces"].reshape(-1, 5).shape == (6, 5) and len(g["scalars"]) == 12
    lines = scene.lines_item([np.array([[0, 0, 0], [1, 0, 0]]), np.array([[0, 1, 0], [1, 1, 0], [2, 1, 0]])], "rays", scalars=[1.0, 2.0])
    assert lines["kind"] == "lines" and lines["points"].shape == (5, 3) and lines["lines"].tolist() == [2, 0, 1, 3, 2, 3, 4]
    np.testing.assert_allclose(lines["scalars"], [1, 1, 2, 2, 2])
    m = scene.mesh_item(np.eye(3), [[0, 1, 2]], "t/sheet", scalars=[0.5], solid=False)
    assert m["faces"].tolist() == [3, 0, 1, 2] and m["solid"] is False


def test_the_settings_store_can_be_backed_by_a_host():
    store = {}
    style.settings.configure(store.get, store.__setitem__)
    try:
        assert style.antialiasing() == "msaa" and not style.ambient_occlusion()
        style.settings.set(style.Settings.ANTIALIASING, "fxaa")
        assert style.antialiasing() == "fxaa" and store[style.Settings.ANTIALIASING] == "fxaa"
    finally:
        style.settings.__init__()


def test_display_meshes_split_at_corners_and_dense_edges_fade():
    pytest.importorskip("pyvista")
    pts, faces = mm.box_mesh(np.zeros(3), np.ones(3))
    m = style.display_mesh(pts, faces)
    assert "Normals" in m.point_data and m.n_points == 24
    edges = style.feature_edges(pts, faces)
    assert edges.n_cells == 12 and style.edge_strength(pts, faces, edges) == 1.0
    z = np.linspace(0.0, 1.0, 400)
    r = 0.05 + 0.02 * (np.arange(400) % 2)  # 200 corrugations: a forest of rings
    rev = mm.revolved_mesh_arrays(z, r, n_phi=48)
    e = style.feature_edges(*rev)
    assert style.edge_strength(rev[0], rev[1], e) < 0.5
