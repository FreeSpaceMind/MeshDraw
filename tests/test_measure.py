"""Measuring drawn meshes: topology from a box, distances and angles."""

import numpy as np
import pytest

from meshdraw.measure import Entity, MeshTopology, measure, min_distance
from meshdraw.mesh import box_mesh

UNIT = ("mm", 1e3)


def _box(lo, hi):
    return MeshTopology(*box_mesh(np.array(lo, float), np.array(hi, float)))


def _face(t, normal, where=None):
    cands = []
    for i in range(len(t.faces)):
        inf = t.face_info(i)
        if inf["kind"] == "plane" and abs(abs(float(inf["normal"] @ np.asarray(normal, float))) - 1) < 1e-9:
            cands.append((0.0 if where is None else float(np.linalg.norm(inf["point"] - np.asarray(where))), i))
    return min(cands)[1]


def test_a_box_has_six_planes_twelve_lines_and_eight_corners():
    t = _box((0, 0, 0), (0.02286, 0.01016, 0.04))
    assert len(t.faces) == 6 and len(t.edges) == 12 and len(t.vertices) == 8
    assert all(t.face_info(i)["kind"] == "plane" for i in range(6))


def test_two_faces_report_distance_normal_distance_and_angle():
    t = _box((0, 0, 0), (0.02, 0.01, 0.04))
    a = _face(t, (1, 0, 0), (0.02, 0.005, 0.02))
    b = _face(t, (1, 0, 0), (0.0, 0.005, 0.02))
    r = measure([t], [Entity("face", 0, a), Entity("face", 0, b)], unit=UNIT)
    assert "Normal distance: 20 mm (parallel)" in r["lines"]
    top = _face(t, (0, 0, 1), (0.01, 0.005, 0.04))
    r = measure([t], [Entity("face", 0, a), Entity("face", 0, top)], unit=UNIT)
    assert any("Angle: 90" in line for line in r["lines"])


def test_min_distance_is_exact_between_skew_segments_and_a_point_over_a_triangle():
    seg_a = (np.zeros((0, 3)), np.array([[[0, 0, 0], [1, 0, 0]]], float), np.zeros((0, 3, 3)))
    seg_b = (np.zeros((0, 3)), np.array([[[0.5, -1, 1], [0.5, 1, 1]]], float), np.zeros((0, 3, 3)))
    d, p, q = min_distance(seg_a, seg_b)
    assert d == pytest.approx(1.0)
    tri = (np.zeros((0, 3)), np.zeros((0, 2, 3)), np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], float))
    pt = (np.array([[0.2, 0.2, 0.7]]), np.zeros((0, 2, 3)), np.zeros((0, 3, 3)))
    d, _p, _q = min_distance(pt, tri)
    assert d == pytest.approx(0.7)
