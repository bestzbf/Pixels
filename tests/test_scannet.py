"""ScanNet loader / clip-export checks against a synthetic ScanNet-format fixture.

Also usable as the fixture builder for offline GPU runs:
    python tests/test_scannet.py build <root> --scenes 2 --frames 30
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from l4d.data.scannet import build_clip, discover_scenes, export_manifest, load_scene

SCENE = "scene0000_00"
WIDTH, HEIGHT = 160, 120
INTRINSIC = np.array([[120.0, 0.0, WIDTH / 2], [0.0, 120.0, HEIGHT / 2], [0.0, 0.0, 1.0]])


def write_fixture(root: str, scenes: int = 2, frames: int = 30) -> str:
    """Official ScanNet-v2 layout: frame-%06d.jpg / -depth.png / -pose.txt + intrinsic/intrinsic_depth.txt."""
    import shutil

    from PIL import Image

    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(0)
    for scene_index in range(scenes):
        scene = f"scene{scene_index:04d}_00"
        directory = os.path.join(root, scene)
        os.makedirs(os.path.join(directory, "intrinsic"), exist_ok=True)
        np.savetxt(os.path.join(directory, "intrinsic", "intrinsic_depth.txt"), INTRINSIC, fmt="%.6f")
        for index in range(frames):
            depth = np.clip(1.5 + 0.4 * np.sin(index / 3.0) + 0.02 * rng.standard_normal((HEIGHT, WIDTH)), 0.3, 8.0)
            Image.fromarray((depth * 1000).astype(np.uint16)).save(os.path.join(directory, f"frame-{index:06d}-depth.png"))
            color = (255 * (depth / depth.max())[..., None] * np.array([[[0.4, 0.6, 0.8]]])).astype(np.uint8)
            Image.fromarray(color).save(os.path.join(directory, f"frame-{index:06d}.jpg"))
            angle = index * 0.02
            rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
            pose = np.eye(4)
            pose[:3, :3] = rotation
            pose[:3, 3] = [0.05 * index, 0.0, 0.0]
            np.savetxt(os.path.join(directory, f"frame-{index:06d}-pose.txt"), pose, fmt="%.6f")
    return root


def test_scannet_layout_is_detected():
    root = "/tmp/pixels_scannet_test"
    write_fixture(root, scenes=1, frames=12)
    scenes = discover_scenes(root)
    assert scenes == [SCENE], scenes
    loaded = load_scene(root, SCENE)
    assert loaded is not None and len(loaded.colors) == 12
    assert np.allclose(loaded.intrinsic[0, 0], 120.0)


def test_clip_gt_is_geometrically_consistent():
    root = "/tmp/pixels_scannet_test"
    write_fixture(root, scenes=1, frames=12)
    loaded = load_scene(root, SCENE)
    clip = build_clip(loaded, list(range(6)), (HEIGHT, WIDTH), "/tmp/pixels_scannet_clip")
    assert clip["points"].shape == (6, HEIGHT, WIDTH, 3)
    assert clip["depth"].shape == (6, HEIGHT, WIDTH)
    assert clip["camera_rotation"].shape == (6, 3, 3)
    assert clip["fov"].shape == (6, 2)
    assert clip["mask"].dtype == bool and clip["mask"].mean() > 0.9
    # world points must reproduce the camera-space ray at the GT depth of frame 0
    rotation, center = clip["camera_rotation"][0], clip["camera_centers"][0]
    camera_space = np.einsum("ij,hwj->hwi", rotation.T, clip["points"][0] - center)
    assert np.abs(camera_space[..., 2] - clip["depth"][0]).max() < 1e-3
    # orthonormal rotations
    assert np.abs(clip["camera_rotation"] @ np.transpose(clip["camera_rotation"], (0, 2, 1)) - np.eye(3)).max() < 1e-4


def test_export_manifest_produces_loadable_clips():
    root = "/tmp/pixels_scannet_test"
    out = "/tmp/pixels_scannet_out"
    import shutil

    shutil.rmtree(out, ignore_errors=True)
    write_fixture(root, scenes=2, frames=25)
    summary = export_manifest(root, out, clip_length=10, clip_stride=10, size=(HEIGHT, WIDTH))
    assert summary["clips"] >= 2, summary
    assert summary["skipped"] == []
    from l4d.data.dataset import ReconstructionClips, collate_clips, load_manifest, manifest_report

    records = load_manifest(summary["manifest"])
    report = manifest_report(records)
    assert report["clips"] == summary["clips"]
    dataset = ReconstructionClips(records, image_size=(HEIGHT, WIDTH), frames=10)
    sample = collate_clips([dataset[0]])
    assert sample["video"].shape == (1, 3, 10, HEIGHT, WIDTH)
    assert float(sample["video"].min()) >= -1.001 and float(sample["video"].max()) <= 1.001  # VAE range
    assert sample["gt_points"].shape == (1, 10, HEIGHT, WIDTH, 3)
    assert torch.isfinite(sample["gt_points"]).all()


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        root = sys.argv[2] if len(sys.argv) > 2 else "/tmp/pixels_scannet"
        scenes = int(os.environ.get("FIXTURE_SCENES", 4))
        frames = int(os.environ.get("FIXTURE_FRAMES", 63))
        write_fixture(root, scenes=scenes, frames=frames)
        print(f"wrote {scenes} ScanNet-format scenes x {frames} frames to {root}")
        return
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"pass  {name}")


if __name__ == "__main__":
    main()
