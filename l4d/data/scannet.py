"""ScanNet (v2) -> L4AR clip records: colour frames, depth, poses -> metric 4D supervision.

Two on-disk layouts are auto-detected per scene:

  A. official sens-processed tree   <scene>/<scene>_00/{frame-%06d.jpg, frame-%06d-depth.png,
                                                    frame-%06d-pose.txt, intrinsic/intrinsic_depth.txt}
  B. mvsanywhere png2-style tree    <scene>/{%05d.png, depth/%05d.png, pose/%05d.txt, intrinsics.txt}

ScanNet depth is uint16 in millimetres and poses are 4x4 camera-to-world, which is exactly the
annotation the paper's geometry supervision needs (world point maps, cameras, normals). Static scenes
give zero inter-frame motion, so clips are drawn from camera motion only - see docs note in
REPRODUCTION_PLAN.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

DEPTH_SCALE_MM = 1000.0
SCANNET_INTRINSIC = np.array([[1.17254926, 0.0, 325.0], [0.0, 1.17254926, 243.6], [0.0, 0.0, 1.0]])


@dataclass
class SceneFrames:
    scene: str
    colors: list[str]
    depths: list[str]
    poses: list[str]
    intrinsic: np.ndarray


def _read_intrinsic(path: str) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(3, 3) if values.size == 9 else values.reshape(4, 4)[:3, :3]
    return values[:3, :3].astype(np.float64)


def _sorted_by_index(paths: Iterable[str], pattern: re.Pattern) -> list[str]:
    matched = [(int(pattern.search(p).group(1)), p) for p in paths if pattern.search(p)]
    return [p for _, p in sorted(matched)]


def load_scene(root: str, scene: str) -> Optional[SceneFrames]:
    """Return the frame triplets of one scene directory, or None when the layout is unknown."""
    scene_dir = os.path.join(root, scene)
    if not os.path.isdir(scene_dir):
        return None
    entries = sorted(os.listdir(scene_dir))

    # layout A: official ScanNet frame-*.jpg / -depth.png / -pose.txt
    if any(name.startswith("frame-") for name in entries):
        paths = [os.path.join(scene_dir, name) for name in entries]
        colors = _sorted_by_index([p for p in paths if re.search(r"frame-\d+\.jpg$", p)], re.compile(r"frame-(\d+)"))
        depths = _sorted_by_index([p for p in paths if re.search(r"frame-\d+-depth\.png$", p)], re.compile(r"frame-(\d+)"))
        poses = _sorted_by_index([p for p in paths if re.search(r"frame-\d+-pose\.txt$", p)], re.compile(r"frame-(\d+)"))
        intrinsic_file = os.path.join(scene_dir, "intrinsic", "intrinsic_depth.txt")
        intrinsic = _read_intrinsic(intrinsic_file) if os.path.exists(intrinsic_file) else SCANNET_INTRINSIC
        return _align(scene, colors, depths, poses, intrinsic)

    # layout B: png2-style with per-directory frames and a whole-scene 4x4 poses file
    color_dir = next((os.path.join(scene_dir, name) for name in entries if name in {"images", "color"}), None)
    depth_dir = next((os.path.join(scene_dir, name) for name in entries if name in {"depth", "depths"}), None)
    pose_dir = next((os.path.join(scene_dir, name) for name in entries if name in {"pose", "poses"}), None)
    if color_dir and depth_dir and pose_dir:
        colors = _sorted_by_index([os.path.join(color_dir, n) for n in os.listdir(color_dir)], re.compile(r"(\d+)"))
        depths = _sorted_by_index([os.path.join(depth_dir, n) for n in os.listdir(depth_dir)], re.compile(r"(\d+)"))
        poses = _sorted_by_index([os.path.join(pose_dir, n) for n in os.listdir(pose_dir)], re.compile(r"(\d+)"))
        intrinsic_file = next(
            (os.path.join(scene_dir, name) for name in entries if "intrinsic" in name.lower()), None
        )
        intrinsic = _read_intrinsic(intrinsic_file) if intrinsic_file else SCANNET_INTRINSIC
        return _align(scene, colors, depths, poses, intrinsic)
    return None


def _align(scene: str, colors: list[str], depths: list[str], poses: list[str], intrinsic: np.ndarray):
    count = min(len(colors), len(depths), len(poses))
    if count == 0:
        return None
    return SceneFrames(scene=scene, colors=colors[:count], depths=depths[:count], poses=poses[:count], intrinsic=intrinsic)


def discover_scenes(root: str, split_file: Optional[str] = None) -> list[str]:
    scenes = sorted(name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name)))
    if split_file and os.path.exists(split_file):
        with open(split_file, "r", encoding="utf-8") as fh:
            wanted = {line.strip().split("_00")[0] + "_00" for line in fh if line.strip()}
        scenes = [s for s in scenes if s in wanted or any(s.startswith(w + "_") for w in wanted)]
    return scenes


def _load_depth(path: str) -> np.ndarray:
    """ScanNet stores depth as uint16 millimetres; float maps are already metres."""
    from PIL import Image

    with Image.open(path) as handle:
        raw = np.asarray(handle)
    if np.issubdtype(raw.dtype, np.integer):
        return raw.astype(np.float64) / DEPTH_SCALE_MM
    return raw.astype(np.float64)


def _load_pose(path: str) -> np.ndarray:
    pose = np.loadtxt(path, dtype=np.float64)
    return pose.reshape(4, 4) if pose.size == 16 else np.eye(4)


def build_clip(frames: SceneFrames, positions: list[int], size: tuple[int, int], clip_dir: str) -> dict[str, np.ndarray]:
    """Read one clip's frames, resize them to `size`, and write the GT tensors used by Eq. 7 supervision."""
    from PIL import Image

    os.makedirs(clip_dir, exist_ok=True)
    colors, depths, poses = [], [], []
    native = None
    for order, index in enumerate(positions):
        with Image.open(frames.colors[index]) as handle:
            native = handle.size if native is None else native
            image = handle.convert("RGB").resize((size[1], size[0]))
        array = np.asarray(image, dtype=np.float32) / 255.0
        colors.append(array)
        Image.fromarray((array * 255.0).round().astype(np.uint8)).save(os.path.join(clip_dir, f"{order:04d}.png"))
        depths.append(_load_depth(frames.depths[index]))
        poses.append(_load_pose(frames.poses[index]))
    intrinsic = scale_intrinsic(frames.intrinsic, native, size)
    return _annotate(np.stack(colors, 0), np.stack(depths, 0), np.stack(poses, 0), intrinsic, size, native)


def export_manifest(
    root: str,
    out_dir: str,
    clip_length: int = 21,
    clip_stride: int = 21,
    frame_stride: int = 1,
    size: tuple[int, int] = (192, 256),
    scenes: Optional[list[str]] = None,
    max_clips: int = 0,
) -> dict:
    """Write `manifest.jsonl`, per-clip frame directories and GT .npz files under `out_dir`."""
    os.makedirs(os.path.join(out_dir, "clips"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "gt"), exist_ok=True)
    scene_names = scenes or discover_scenes(root)
    records, skipped = [], []
    for scene in scene_names:
        loaded = load_scene(root, scene)
        if loaded is None:
            skipped.append(scene)
            continue
        for start in range(0, len(loaded.colors) - clip_length + 1, clip_stride):
            step = max(frame_stride, 1)
            positions = list(range(start, start + clip_length * step, step))[:clip_length]
            if len(positions) < clip_length:
                break
            clip_id = f"{scene}_{start:06d}"
            clip_dir = os.path.join(out_dir, "clips", clip_id)
            os.makedirs(clip_dir, exist_ok=True)
            try:
                clip = build_clip(loaded, positions, size, clip_dir)
            except Exception as error:  # noqa: BLE001 - one unreadable scene must not abort the dataset
                skipped.append(f"{scene}:{type(error).__name__}")
                break
            gt_path = os.path.join(out_dir, "gt", f"{clip_id}.npz")
            np.savez_compressed(gt_path, **clip)
            records.append(
                {
                    "clip_id": clip_id,
                    "dataset": "scannet",
                    "video": clip_dir,
                    "gt": gt_path,
                    "num_frames": clip_length,
                    "resolution": list(size),
                    "split": "train",
                }
            )
            if max_clips and len(records) >= max_clips:
                break
        if max_clips and len(records) >= max_clips:
            break

    manifest = os.path.join(out_dir, "manifest.jsonl")
    with open(manifest, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return {"manifest": manifest, "clips": len(records), "scenes": len(scene_names), "skipped": skipped[:20]}

def scale_intrinsic(intrinsic: np.ndarray, native_size: tuple[int, int], size: tuple[int, int]) -> np.ndarray:
    """Re-express the native-resolution intrinsics for the resized clip (fx, cx scale by width ratio)."""
    native_w, native_h = float(native_size[0]), float(native_size[1])
    scaled = np.array(intrinsic, dtype=np.float64, copy=True)
    sx = size[1] / native_w
    sy = size[0] / native_h
    scaled[0, 0] *= sx
    scaled[0, 2] *= sx
    scaled[1, 1] *= sy
    scaled[1, 2] *= sy
    return scaled


def _annotate(
    colors: np.ndarray,
    depths: np.ndarray,
    poses: np.ndarray,
    intrinsic: np.ndarray,
    size: tuple[int, int],
    native_size: tuple[int, int] | None = None,
) -> dict[str, np.ndarray]:
    """Metric 4D supervision for one clip, expressed in a single shared world frame."""
    from PIL import Image

    height, width = size
    depth_stack = np.stack(
        [
            np.asarray(
                Image.fromarray((d * DEPTH_SCALE_MM).astype(np.uint16)).resize(
                    (width, height), resample=Image.Resampling.NEAREST
                ),
                np.float64,
            )
            / DEPTH_SCALE_MM
            for d in depths
        ],
        0,
    ).astype(np.float32)
    t = poses.shape[0]
    grid_v, grid_u = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    pixel = np.stack([grid_u + 0.5, grid_v + 0.5, np.ones_like(grid_v)], -1).astype(np.float32)
    rays_camera = np.einsum("ij,whj->whi", np.linalg.inv(intrinsic).astype(np.float32), pixel)
    camera_points = rays_camera[None] * depth_stack[..., None]
    rotation = poses[:, :3, :3].astype(np.float32)
    translation = poses[:, :3, 3].astype(np.float32)
    world_points = np.einsum("tij,trwj->trwi", rotation, camera_points) + translation[:, None, None, :]
    world_rays = world_points - translation[:, None, None, :]
    world_rays = world_rays / np.maximum(np.linalg.norm(world_rays, axis=-1, keepdims=True), 1e-8)
    valid = (depth_stack > 1e-3) & (depth_stack < 50.0)
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    fov = np.stack([2 * np.arctan(height / (2 * fy)), 2 * np.arctan(width / (2 * fx))], -1).astype(np.float32)
    return {
        "depth": depth_stack,
        "mask": valid,
        "points": world_points.astype(np.float32),
        "ray_dirs": world_rays.astype(np.float32),
        "camera_rotation": rotation,
        "camera_centers": translation,
        "fov": np.repeat(fov[None], t, axis=0),
        "intrinsic": intrinsic.astype(np.float32),
        "colors": colors.transpose(3, 0, 1, 2),
    }
