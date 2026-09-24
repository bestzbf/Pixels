#!/usr/bin/env python3
"""Tests for the Agisoft Metashape ingestor (`l4d/data/metashape.py`).

The quaternion round-trip always runs. The real-project checks skip when the capture is not on this
machine, because the .psz lives in the user's data directory rather than in git.
"""
from __future__ import annotations

import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from l4d.data.colmap import qvec2rotmat
from l4d.data.metashape import _rotmat2qvec, mesh_vertices, read_project, tie_points

PROJECT = "/home/zbf/Desktop/mvs示例数据/准确版_多视角输入/建筑_Agisoft_Building/building/building.psz"
MESH = "/home/zbf/Desktop/mvs示例数据/准确版_多视角输入/建筑_Agisoft_Building/building/OBJ_export/building.obj"


def test_quaternion_roundtrip_matches_colmap_convention():
    """A wrong (w,x,y,z) order or a transposed rotation survives parsing and silently breaks every pose."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        vector, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(vector) < 0:
            vector[:, 0] *= -1
        translation = rng.normal(size=3)
        matrix = np.eye(4)
        matrix[:3, :3], matrix[:3, 3] = vector, translation
        qvec, tvec = _rotmat2qvec(matrix[:3, :3]), matrix[:3, 3]
        assert abs(np.linalg.norm(qvec) - 1.0) < 1e-9
        assert np.allclose(qvec2rotmat(qvec), vector, atol=1e-8)
        assert np.allclose(tvec, translation)


def test_written_model_is_readable_by_the_colmap_stager(tmp_path: str = "/tmp/pixels_metashape_model"):
    """prepare_colmap.py must be able to read what the converter emits, with the same intrinsics."""
    from l4d.data.colmap import read_cameras, read_images
    from l4d.data.metashape import write_colmap_model

    sensor = {"width": 1000, "height": 800, "fx": 900.0, "fy": 900.0, "cx": 500.0, "cy": 400.0,
              "k1": -0.1, "k2": 0.02, "k3": 0.0}
    camera = {"name": "IMG_1.JPG", "sensor": sensor, "camera_from_world": np.eye(4),
              "qvec": np.array([1.0, 0.0, 0.0, 0.0]), "tvec": np.zeros(3)}
    report = write_colmap_model(tmp_path, [camera], np.array([[1.0, 2.0, 3.0]]))
    assert report["images"] == 1 and report["points3D"] == 1
    cameras = read_cameras(os.path.join(tmp_path, "cameras.txt"))
    assert cameras[1].model == "OPENCV" and cameras[1].width == 1000
    assert np.allclose(cameras[1].intrinsics()[:4], [900.0, 900.0, 500.0, 400.0])
    assert np.allclose(cameras[1].distortion(), [-0.1, 0.02, 0, 0, 0.0])
    entries = read_images(os.path.join(tmp_path, "images.txt"))
    assert entries[0]["name"] == "IMG_1.JPG"


def _skip_if_missing() -> bool:
    if not os.path.exists(PROJECT):
        print("skip  real Metashape project tests (capture not on this machine)")
        return True
    return False


def test_real_project_pose_and_geometry_agree():
    """The one check that catches a chunk/world frame mix-up: reconstructed points must project into frame."""
    if _skip_if_missing():
        return
    project = read_project(PROJECT)
    cameras = project["cameras"]
    assert len(cameras) >= 40, f"only {len(cameras)} calibrated cameras parsed"
    points = tie_points(PROJECT, project["chunk_from_world"], limit=20_000)
    assert len(points) > 100, "the project carries no tie points"

    inside = 0
    for camera in cameras[:10]:
        sensor = camera["sensor"]
        matrix = camera["camera_from_world"]
        transformed = np.column_stack([points, np.ones(len(points))]) @ matrix.T
        depth = transformed[:, 2]
        visible = depth > 1e-6
        x = sensor["fx"] * transformed[visible, 0] / depth[visible] + sensor["cx"]
        y = sensor["fy"] * transformed[visible, 1] / depth[visible] + sensor["cy"]
        inside += int(((x > 0) & (x < sensor["width"]) & (y > 0) & (y < sensor["height"])).sum())
    fraction = inside / (10 * len(points))
    assert fraction > 0.02, f"only {fraction:.4f} of the cloud projects into the sampled frames"

    for camera in cameras:
        rotation = camera["camera_from_world"][:3, :3]
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
        assert abs(np.linalg.det(rotation) - 1.0) < 1e-6


def test_mesh_vertices_are_metric_and_finite():
    if _skip_if_missing() or not os.path.exists(MESH):
        return
    points = mesh_vertices(MESH, limit=50_000)
    assert len(points) > 1000
    assert np.isfinite(points).all()
    extent = float(np.linalg.norm(points.max(0) - points.min(0)))
    assert 1.0 < extent < 1000.0, f"a building capture reported an extent of {extent:.1f}"


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"pass  {name}")
