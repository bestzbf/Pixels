"""COLMAP sparse reconstruction -> ScanNet-style staging tree, so the tested ScanNet exporter can eat it.

A COLMAP model directory (`cameras.txt` + `images.txt` + `points3D.txt`, or the binary equivalents)
carries real intrinsics, real world<-camera poses and a real sparse point cloud. This module turns it
into the layout `l4d/data/scannet.py` already reads:

    <scene>_00/frame-%06d.jpg          undistorted, resized
    <scene>_00/frame-%06d-depth.png    uint16 millimetres, projected from points3D (0 = no observation)
    <scene>_00/frame-%06d-pose.txt     4x4 camera-to-world
    <scene>_00/intrinsic/intrinsic_depth.txt   pinhole K at the resized resolution

Fisheye/radial distortion is removed while remapping, and the ideal K used for the output resolution is
the same one the point projection uses, so images, intrinsics and depth stay mutually consistent.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

FISHEYE_MODES = {"FISHEYE", "RADIAL_FISHEYE", "THIN_PRISM_FISHEYE"}
#: COLMAP models whose first four parameters are [fx, fy, cx, cy]; the others are [f, cx, cy].
FOUR_TERM_MODELS = {
    "PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "FOV",
    "THIN_PRISM_FISHEYE", "RADIAL_FISHEYE", "SIMPLE_FISHEYE", "DIVIDE_PYRAMID_REMOVAL",
}


@dataclass
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    @property
    def is_fisheye(self) -> bool:
        return self.model.upper() in FISHEYE_MODES

    def intrinsics(self) -> tuple[float, float, float, float, np.ndarray]:
        """(fx, fy, cx, cy, distortion) normalised across COLMAP camera models."""
        model = self.model.upper()
        if model in FOUR_TERM_MODELS:
            fx, fy, cx, cy = (float(value) for value in self.params[:4])
            extras = self.params[4:]
        else:
            f, cx, cy = (float(value) for value in self.params[:3])
            fx = fy = f
            extras = self.params[3:]
        if fx <= 0 or fy <= 0:
            fx = fy = 1.2 * max(self.width, self.height)
        return fx, fy, cx, cy, np.asarray(extras, dtype=np.float64)

    def matrix(self, scale_x: float = 1.0, scale_y: float = 1.0) -> np.ndarray:
        fx, fy, cx, cy, _ = self.intrinsics()
        return np.array(
            [[fx * scale_x, 0, cx * scale_x], [0, fy * scale_y, cy * scale_y], [0, 0, 1]], dtype=np.float64
        )

    def distortion(self) -> np.ndarray:
        """OpenCV layout [k1,k2,p1,p2,k3]; fisheye wants only [k1..k4]."""
        extras = self.intrinsics()[4]
        if self.is_fisheye:
            return np.concatenate([extras[:4], np.zeros(4 - min(4, extras.size))])
        values = np.zeros(5)
        values[: min(5, extras.size)] = extras[:5]
        return values


def read_cameras(path: str) -> dict[int, ColmapCamera]:
    cameras = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        camera_id, model, width, height = int(fields[0]), fields[1].upper(), int(fields[2]), int(fields[3])
        params = np.array([float(value) for value in fields[4:]], dtype=np.float64)
        cameras[camera_id] = ColmapCamera(camera_id, model, width, height, params)
    return cameras


def read_images(path: str) -> list[dict]:
    """Pose records, in file order (capture order).

    COLMAP writes two lines per image, but the POINTS2D line is empty for undistorted/derived models, so
    pairing by position is unreliable: a pose record is identified by having exactly ten fields whose
    quaternion is unit norm.
    """
    entries = []
    for line in open(path, encoding="utf-8"):
        fields = line.strip().split()
        if len(fields) != 10 or fields[0].startswith("#"):
            continue
        try:
            quaternion = np.array([float(value) for value in fields[1:5]], dtype=np.float64)
            tx, ty, tz = (float(value) for value in fields[5:8])
        except ValueError:
            continue
        if abs(np.linalg.norm(quaternion) - 1.0) > 1e-4:
            continue
        qw, qx, qy, qz = quaternion
        name = fields[9]
        tx, ty, tz = (float(value) for value in fields[5:8])
        entries.append(
            {
                "image_id": int(fields[0]),
                "camera_id": int(fields[8]),
                "name": name,
                "qvec": quaternion,
                "tvec": np.array([tx, ty, tz], dtype=np.float64),
            }
        )
    return entries


def read_points3D(path: str, limit: int = 2_000_000) -> np.ndarray:
    coordinates = []
    for line in open(path, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        fields = line.split()
        if len(fields) < 7:
            continue
        coordinates.append([float(fields[1]), float(fields[2]), float(fields[3])])
        if len(coordinates) >= limit:
            break
    return np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec / np.linalg.norm(qvec)
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def find_model_dirs(root: str) -> list[str]:
    """Directories holding a COLMAP model in text form anywhere below `root`.

    Binary `.bin` models are intentionally not claimed here; convert with
    `colmap model_converter --output_type TXT` first.
    """
    found = []
    for base, _, files in os.walk(root):
        if {"cameras.txt", "images.txt"} <= set(files):
            found.append(base)
    return sorted(found)


def _undistort_maps(camera: ColmapCamera, out_size: tuple[int, int]):
    height, width = out_size
    native = camera.matrix()
    ideal = camera.matrix(width / camera.width, height / camera.height)
    distortion = camera.distortion()
    if camera.is_fisheye:
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            native, distortion.reshape(1, 4), np.eye(3), ideal, (width, height), cv2.CV_16SC2
        )
    else:
        map1, map2 = cv2.initUndistortRectifyMap(
            native, distortion, np.eye(3), ideal, (width, height), cv2.CV_16SC2
        )
    return map1, map2, ideal


def _splat_depth(points: np.ndarray, entry: dict, camera: ColmapCamera, ideal: np.ndarray,
                 out_size: tuple[int, int], splat_radius: int) -> np.ndarray:
    height, width = out_size
    depth = np.zeros((height, width), dtype=np.float64)
    if points.size == 0:
        return depth
    rotation = qvec2rotmat(entry["qvec"])
    camera_points = points @ rotation.T + entry["tvec"]
    in_front = camera_points[:, 2] > 1e-6
    projected = camera_points[in_front] @ ideal.T
    z = projected[:, 2:3]
    pixel = (projected[:, :2] / np.maximum(z, 1e-9)).round().astype(np.int64)
    distance = camera_points[in_front][:, 2]
    inside = (pixel[:, 0] >= 0) & (pixel[:, 0] < width) & (pixel[:, 1] >= 0) & (pixel[:, 1] < height)
    pixel, distance = pixel[inside], distance[inside]
    near_first = np.argsort(-distance)  # far first so nearer points overwrite
    pixel, distance = pixel[near_first], distance[near_first]
    radius = max(splat_radius, 0)
    for offset_y in range(-radius, radius + 1):
        for offset_x in range(-radius, radius + 1):
            shifted_x = pixel[:, 0] + offset_x
            shifted_y = pixel[:, 1] + offset_y
            valid = (shifted_x >= 0) & (shifted_x < width) & (shifted_y >= 0) & (shifted_y < height)
            depth[shifted_y[valid], shifted_x[valid]] = distance[valid]
    return depth


def _find_depth(depth_dir: Optional[str], name: str, order: int = 0, total: int = 0) -> Optional[str]:
    """Prefer a real per-view depth map over reprojecting the sparse cloud.

    Renamed conversions (`000_depth.png` against a `frame_000.png` pose record) fall back to the
    k-th sorted file, which is how those datasets are written out.
    """
    if not depth_dir or not os.path.isdir(depth_dir):
        return None
    stem = os.path.splitext(os.path.basename(name))[0]
    for suffix in (".png", ".npy", ".npz", ".exr", ".tiff"):
        candidate = os.path.join(depth_dir, stem + suffix)
        if os.path.exists(candidate):
            return candidate
    entries = sorted(os.listdir(depth_dir))
    matches = [entry for entry in entries if os.path.splitext(entry)[0] == stem]
    if matches:
        return os.path.join(depth_dir, matches[0])
    files = [entry for entry in entries if entry.lower().endswith((".png", ".npy", ".npz", ".exr", ".tiff"))]
    if files and total and len(files) >= total and order < len(files):
        return os.path.join(depth_dir, files[order])
    return None


def _depth_to_millimetres(path: str, size: tuple[int, int]) -> np.ndarray:
    height, width = size
    if path.endswith(".npy"):
        values = np.load(path).astype(np.float64)
        return (values * 1000.0 if values.max() < 1000 else values).astype(np.uint16)
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(path)
    values = raw.astype(np.float64)
    if values.ndim == 3:
        values = values[..., 0]
    if np.issubdtype(raw.dtype, np.floating) or values.max() < 1000:
        values = values * 1000.0  # float depth maps are metres; uint16 maps are already millimetres
    if values.shape[:2] != (height, width):
        values = cv2.resize(values, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.clip(values, 0, 65535).astype(np.uint16)


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def _candidate_folders(image_root: str, model_dir: str) -> list[str]:
    roots = [image_root, os.path.dirname(image_root), model_dir, os.path.dirname(model_dir),
             os.path.dirname(os.path.dirname(model_dir))]
    folders: list[str] = []
    for root in roots:
        for folder in (root, os.path.join(root, "images")):
            if folder not in folders:
                folders.append(folder)
    return folders


def _ordered_files(folder: str) -> list[str]:
    return sorted(
        os.path.join(folder, name) for name in os.listdir(folder) if name.lower().endswith(IMAGE_SUFFIXES)
    )


def _resolve_source(image_root: str, model_dir: str, name: str, order: int = 0) -> Optional[str]:
    """COLMAP names may be relative to the model dir, the images/ folder, or the scene root.

    Converted NeRF/LLFF-style scenes keep `tiny_poses.txt` order but rename frames (`000_color.png`), so
    when no name matches the k-th pose falls back to the k-th sorted image of the folder.
    """
    base = os.path.basename(name)
    stem = os.path.splitext(base)[0]
    for folder in _candidate_folders(image_root, model_dir):
        if not os.path.isdir(folder):
            continue
        for candidate in (os.path.join(folder, name), os.path.join(folder, base)):
            if os.path.isfile(candidate):
                return candidate
        matches = [path for path in _ordered_files(folder) if os.path.splitext(os.path.basename(path))[0] == stem]
        if matches:
            return matches[0]
    for folder in _candidate_folders(image_root, model_dir):
        if os.path.isdir(folder):
            files = _ordered_files(folder)
            if files and order < len(files):
                return files[order]
    return None


def _nearest_dir_ancestor_with(model_dir: str, names: tuple[str, ...]) -> Optional[str]:
    """COLMAP layouts differ: `scene/sparse/0` plus `scene/images` and `scene/depths`."""
    current = os.path.abspath(model_dir)
    for _ in range(5):
        for name in names:
            candidate = os.path.join(current, name)
            if os.path.isdir(candidate):
                return candidate
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return None


def scene_to_scannet_tree(
    model_dir: str,
    image_root: str,
    out_root: str,
    scene_name: str,
    size: tuple[int, int] = (240, 320),
    max_frames: int = 21,
    splat_radius: int = 2,
    frame_stride: int = 1,
    min_frames: int = 5,
    depth_dir: Optional[str] = None,
) -> Optional[str]:
    """Materialise one COLMAP scene as a ScanNet-style directory; returns its path or None.

    Per-view depth maps on disk win over reprojecting `points3D`; a scene whose depth ends up covering
    under 1% of pixels is dropped rather than exported as an unsupervisable clip.
    """
    cameras = read_cameras(os.path.join(model_dir, "cameras.txt"))
    entries = read_images(os.path.join(model_dir, "images.txt"))
    points_file = os.path.join(model_dir, "points3D.txt")
    points = read_points3D(points_file) if os.path.exists(points_file) else np.zeros((0, 3))
    if not entries or not cameras:
        return None

    positions = list(range(0, len(entries), max(frame_stride, 1)))[:max_frames]
    positions = positions[: 1 + 4 * ((len(positions) - 1) // 4)]  # the Wan VAE needs 4k+1 frames
    if len(positions) < min_frames:
        return None

    if depth_dir is None:
        depth_dir = _nearest_dir_ancestor_with(model_dir, ("depths", "depth", "render_depth"))
    expected_frames = len(positions)
    used_dense = False
    scene_dir = os.path.join(out_root, f"{scene_name}_00")
    shutil.rmtree(scene_dir, ignore_errors=True)
    os.makedirs(os.path.join(scene_dir, "intrinsic"), exist_ok=True)

    ideal: Optional[np.ndarray] = None
    coverages: list[float] = []
    for order, index in enumerate(positions):
        entry = entries[index]
        camera = cameras.get(entry["camera_id"])
        source = _resolve_source(image_root, model_dir, entry["name"], order)
        if camera is None or source is None:
            continue
        image = cv2.imread(source)
        if image is None:
            continue
        map1, map2, ideal = _undistort_maps(camera, size)
        cv2.imwrite(
            os.path.join(scene_dir, f"frame-{order:06d}.jpg"),
            cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE),
        )
        depth_file = _find_depth(depth_dir, entry["name"], len(coverages), expected_frames)
        if depth_file is not None:
            depth_mm = _depth_to_millimetres(depth_file, size)
            used_dense = used_dense or bool((depth_mm > 0).any())
        else:
            sparse = _splat_depth(points, entry, camera, ideal, size, splat_radius)
            depth_mm = (sparse * 1000.0).astype(np.uint16)
        cv2.imwrite(os.path.join(scene_dir, f"frame-{order:06d}-depth.png"), depth_mm)
        rotation = qvec2rotmat(entry["qvec"])
        pose = np.eye(4)
        pose[:3, :3] = rotation.T
        pose[:3, 3] = -rotation.T @ entry["tvec"]
        np.savetxt(os.path.join(scene_dir, f"frame-{order:06d}-pose.txt"), pose, fmt="%.8f")
        coverages.append(float((depth_mm > 0).mean()))

    if len(coverages) < min_frames or ideal is None:
        shutil.rmtree(scene_dir, ignore_errors=True)
        return None
    coverage = float(np.mean(coverages))
    if coverage < 0.01:
        print(f"[skip ] {scene_name}: depth coverage {coverage*100:.2f}% cannot supervise Eq. 7", flush=True)
        shutil.rmtree(scene_dir, ignore_errors=True)
        return None
    np.savetxt(os.path.join(scene_dir, "intrinsic", "intrinsic_depth.txt"), ideal, fmt="%.8f")
    print(f"[depth ] {scene_name}: {'dense per-view maps' if used_dense else 'points3D splat'} "
          f"coverage={coverage*100:.1f}% over {len(coverages)} frames", flush=True)
    return scene_dir


def scene_stats(model_dir: str) -> dict:
    cameras = read_cameras(os.path.join(model_dir, "cameras.txt"))
    entries = read_images(os.path.join(model_dir, "images.txt"))
    points_file = os.path.join(model_dir, "points3D.txt")
    return {
        "cameras": {camera.model: (camera.width, camera.height) for camera in cameras.values()},
        "images": len(entries),
        "points3D": len(read_points3D(points_file)) if os.path.exists(points_file) else 0,
    }
