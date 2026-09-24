"""COLMAP ingestion checks: camera models, pose convention, projection consistency, staging."""
from __future__ import annotations

import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from l4d.data.colmap import (
    ColmapCamera,
    find_model_dirs,
    qvec2rotmat,
    read_cameras,
    read_images,
    read_points3D,
    scene_to_scannet_tree,
)
from l4d.data.scannet import load_scene

ROOT = "/tmp/pixels_colmap_test"


def write_model(root: str, model: str = "PINHOLE", params: tuple = (300.0, 300.0, 160.0, 120.0), views: int = 5) -> str:
    """Two views of a cube-ish cloud, COLMAP text layout, images named views/im-%02d.jpg."""
    import cv2

    model_dir = os.path.join(root, "sparse", "0")
    image_dir = os.path.join(root, "images", "views")
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    width, height = 320, 240

    with open(os.path.join(model_dir, "cameras.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"0 {model} {width} {height} " + " ".join(f"{value:.6f}" for value in params) + "\n")

    points = np.stack(
        np.meshgrid(np.linspace(-1, 1, 6), np.linspace(-1, 1, 6), np.linspace(2.0, 4.0, 6), indexing="ij"), -1
    ).reshape(-1, 3)
    with open(os.path.join(model_dir, "points3D.txt"), "w", encoding="utf-8") as fh:
        for index, point in enumerate(points):
            fh.write(f"{index} {point[0]:.6f} {point[1]:.6f} {point[2]:.6f} 128 128 128 0 " + "-1 -1 " * 4 + "\n")

    with open(os.path.join(model_dir, "images.txt"), "w", encoding="utf-8") as fh:
        fh.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n")
        for view in range(views):
            canvas = np.full((height, width, 3), 90, dtype=np.uint8)
            cv2.imwrite(os.path.join(image_dir, f"im-{view:02d}.jpg"), canvas)
            tvec = -np.array([0.15 * view, 0.0, 0.0])  # identity rotation, so t = -C
            fh.write(f"{view + 1} 1 0 0 0 {tvec[0]} {tvec[1]} {tvec[2]} 0 views/im-{view:02d}.jpg\n\n")
    return model_dir


def test_camera_models_and_poses():
    model_dir = write_model(ROOT)
    cameras = read_cameras(os.path.join(model_dir, "cameras.txt"))
    camera = cameras[0]
    assert camera.model == "PINHOLE"
    assert np.allclose(camera.matrix()[0, 0], 300.0)
    assert camera.is_fisheye is False
    entries = read_images(os.path.join(model_dir, "images.txt"))
    assert len(entries) == 5 and entries[0]["name"] == "views/im-00.jpg"
    rotation = qvec2rotmat(entries[0]["qvec"])
    assert np.allclose(rotation, np.eye(3))
    # COLMAP stores camera-from-world; the camera centre must be -R^T t
    center = -rotation.T @ entries[0]["tvec"]
    assert np.allclose(center, [0.0, 0.0, 0.0]), center
    center2 = -rotation.T @ entries[1]["tvec"]
    assert np.allclose(center2, [0.15, 0.0, 0.0]), center2
    assert np.allclose(-rotation.T @ entries[4]["tvec"], [0.6, 0.0, 0.0])
    assert len(read_points3D(os.path.join(model_dir, "points3D.txt"))) == 216


def test_fisheye_models_report_their_four_intrinsics():
    camera = ColmapCamera(0, "THIN_PRISM_FISHEYE", 6048, 4032,
                          np.array([3437.84, 3435.95, 3040.23, 2010.15, 0.2, 0.19, 0.0004, -7.8e-5, 0.0, 0.0, 0.0, 0.0]))
    fx, fy, cx, cy, extras = camera.intrinsics()
    assert (fx, fy) != (0.0, 0.0) and camera.is_fisheye
    assert camera.distortion().shape == (4,)  # fisheye keeps k1..k4
    pinhole = ColmapCamera(0, "SIMPLE_RADIAL", 640, 480, np.array([500.0, 320.0, 240.0, 0.1]))
    fx, fy, cx, cy, extras = pinhole.intrinsics()
    assert (fx, fy, cx, cy) == (500.0, 500.0, 320.0, 240.0) and float(extras[0]) == 0.1
    assert pinhole.distortion().shape == (5,)


def test_staging_then_scannet_reader_round_trip():
    model_dir = write_model(ROOT)
    staged = scene_to_scannet_tree(
        model_dir, os.path.join(ROOT, "images"), os.path.join(ROOT, "staged"), "cube",
        size=(120, 160), max_frames=9, splat_radius=1,
    )
    assert staged is not None
    loaded = load_scene(os.path.join(ROOT, "staged"), "cube_00")
    assert loaded is not None and len(loaded.colors) == 5  # 5 staged frames = 4k+1 with k=1
    intrinsic = np.loadtxt(os.path.join(staged, "intrinsic", "intrinsic_depth.txt"))
    assert abs(intrinsic[0, 0] - 300.0 * 160 / 320) < 1e-3  # focal scaled by the width ratio
    import cv2

    depth = np.asarray(cv2.imread(loaded.depths[0], cv2.IMREAD_UNCHANGED), np.float64) / 1000.0
    assert depth.max() > 0, "sparse depth should be non-empty for a scene in front of the camera"
    assert 1.9 <= depth[depth > 0].min() <= 4.1  # the cloud sits 2-4 m in front of the camera


def test_dense_depth_maps_beat_points3D_splatting():
    """Converted NeRF-style scenes name frames differently; the on-disk depth must still be used."""
    import cv2

    model_dir = write_model(ROOT)
    scene_root = os.path.dirname(os.path.dirname(model_dir))          # .../converted-style root
    depth_dir = os.path.join(scene_root, "depths")
    os.makedirs(depth_dir, exist_ok=True)
    for view in range(5):
        dense = np.full((240, 320), 2500 + 100 * view, dtype=np.uint16)   # millimetres, renamed frames
        cv2.imwrite(os.path.join(depth_dir, f"{view:03d}_depth.png"), dense)
    staged = scene_to_scannet_tree(
        model_dir, os.path.join(ROOT, "images"), os.path.join(ROOT, "staged2"), "dense",
        size=(120, 160), max_frames=9, splat_radius=1,
    )
    assert staged is not None
    import glob as glob_module

    files = sorted(glob_module.glob(os.path.join(staged, "*-depth.png")))
    assert len(files) == 5
    values = cv2.imread(files[1], cv2.IMREAD_UNCHANGED)
    assert float((values > 0).mean()) == 1.0                            # every pixel supervised
    assert abs(int(values[60, 80]) - 2600) < 50                         # the renamed dense map, not a splat


def test_model_discovery_finds_the_scene():
    write_model(ROOT)
    assert find_model_dirs(ROOT) == [os.path.join(ROOT, "sparse", "0")]


def main() -> None:
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"pass  {name}")


if __name__ == "__main__":
    main()
