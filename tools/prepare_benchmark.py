#!/usr/bin/env python3
"""Convert a 7-Scenes or NRGBD capture into the tree this repo already stages.

    python tools/prepare_benchmark.py --root /mnt/data/pixels-benchmarks/7scenes --out data/bm_models
    python tools/prepare_colmap.py --root data/bm_models --staging data/bm_staging --out data/bm \
           --height 480 --width 640 --frames 64 --dataset seven

Output is a COLMAP text model per sequence (`sparse/cameras.txt|images.txt`) plus `depths/frame-*.png` in
uint16 millimetres, which `l4d/data/colmap.py` prefers over splatting a point cloud. Undistortion, clip
export, the QC gate and the manifest are then the same code paths used for the ETH3D and photogrammetry
scenes, so a benchmark sequence and a local capture are indistinguishable downstream - which is the point:
it lets `tools/eval_gt.py --benchmark 7scenes` be scored on real data.

Dialects recognised (both datasets circulate in a Shotton layout and a DenseFusion layout):

    flat      frame-%06d.color.png|.depth.png|.pose.txt      (DenseFusion repackaging, what mirrors ship)
    shotton   scene-%04d-%06d-rgb.png / -camera-framedata.txt / <seq>-depth.tgz

Depth scale: the canonical Shotton PNGs are read as `raw / 5000` metres, but the mirror this came from
stores **millimetres**. That was measured, not assumed: triangulating SIFT matches between frame pairs with
the shipped poses gives `triangulated / (raw/5000) = 4.6-5.6`, i.e. a factor of 5, and the median valid
depth is then 1.6 m for an indoor walkthrough instead of an implausible 0.32 m. `--depth-divisor` overrides
it if a different repackaging disagrees. Pixels at uint16 saturation (>= 60000) are sensor sentinels, not
geometry, and are zeroed.

Intrinsics are the standard 7-Scenes factory calibration for the 640x480 undistossed frames
(fx=fy=585, cx=320.5, cy=240.5); the same triangulation residuals suggest up to ~10 % bias in the derived
depth, so treat absolute Acc/Comp here as carrying that uncertainty.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from l4d.data.metashape import _rotmat2qvec

FLAT = re.compile(r"^(frame-\d{6})\.color\.(png|jpg)$")
SHOTTON = re.compile(r"^(scene-\d{4}-\d{6})-rgb\.png$")
SENTINEL = 60000  # uint16 saturation: these pixels carry no measurement
SHOTTON_INTRINSICS = (640, 480, 585.0, 585.0, 320.5, 240.5)


def _world_to_camera(pose_camera_to_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    world_to_camera = np.linalg.inv(pose_camera_to_world)
    return _rotmat2qvec(world_to_camera[:3, :3]), world_to_camera[:3, 3]


def _write_depth(source: str, destination: str, divisor: float) -> int:
    """Store the depth map as uint16 millimetres, dropping sentinel pixels."""
    raw = cv2.imread(source, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return 0
    if raw.dtype != np.uint16:
        raw = raw.astype(np.uint16)
    millimetres = np.where(raw >= SENTINEL, 0, np.rint(raw.astype(np.float64) * (1000.0 / divisor))).astype(np.uint16)
    cv2.imwrite(destination, millimetres)
    return int((millimetres > 0).sum())


def scan_flat(root: str) -> list[dict]:
    entries = []
    for dirpath, _, filenames in os.walk(root):
        colors = {FLAT.match(name).group(1): os.path.join(dirpath, name)
                  for name in filenames if FLAT.match(name)}
        for stem in sorted(colors):
            pose = os.path.join(dirpath, f"{stem}.pose.txt")
            depth = os.path.join(dirpath, f"{stem}.depth.png")
            if os.path.exists(pose) and os.path.exists(depth):
                sequence = os.path.basename(dirpath.rstrip("/"))
                entries.append({"scene": f"{os.path.basename(os.path.dirname(dirpath.rstrip('/')))}_{sequence}",
                                "name": f"{stem}", "rgb": colors[stem], "pose": pose, "depth": depth})
    return entries


def scan_shotton(root: str) -> list[dict]:
    entries = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            match = SHOTTON.match(name)
            if not match:
                continue
            stem = match.group(1)
            pose = os.path.join(dirpath, f"{stem}-camera-framedata.txt")
            depth = next((os.path.join(dirpath, candidate) for candidate in filenames
                          if candidate.startswith(f"{stem}") and "depth" in candidate
                          and candidate.endswith((".png", ".tif"))), None)
            if os.path.exists(pose) and depth:
                entries.append({"scene": os.path.basename(dirpath.rstrip("/")), "name": stem,
                                "rgb": os.path.join(dirpath, name), "pose": pose, "depth": depth})
    return entries


def _split_key(tag: str) -> str:
    """"sequence1", "seq-01" and "stairs_seq-01" all name the same first capture."""
    digits = re.sub(r"\D", "", str(tag))
    return digits.lstrip("0") or digits


def read_split(root: str) -> dict[str, str]:
    """Honour the benchmark's own TrainSplit.txt / TestSplit.txt instead of inventing a split."""
    split: dict[str, str] = {}
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if not re.match(r"(Train|Test)Split\.txt$", name):
                continue
            kind = "train" if name.startswith("Train") else "test"
            for line in open(os.path.join(dirpath, name), encoding="utf-8", errors="ignore"):
                if line.strip():
                    split[_split_key(line)] = kind
    return split


def convert(root: str, out: str, divisor: float = 1000.0, limit: int = 0) -> dict:
    entries = scan_flat(root)
    dialect = "flat"
    if not entries:
        entries, dialect = scan_shotton(root), "shotton"
    if not entries:
        raise SystemExit(f"no 7-Scenes/NRGBD frames recognised under {root}")
    calibration = SHOTTON_INTRINSICS
    intrinsic_file = next((os.path.join(dirpath, "camera-intrinsics.txt")
                           for dirpath, _, files in os.walk(root) if "camera-intrinsics.txt" in files), None)
    if intrinsic_file:
        k = np.loadtxt(intrinsic_file)
        probe = cv2.imread(entries[0]["rgb"], cv2.IMREAD_COLOR)
        height, width = probe.shape[:2]
        calibration = (width, height, float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]))
    split = read_split(root)

    scenes: dict[str, list[dict]] = {}
    for entry in entries:
        scenes.setdefault(entry["scene"], []).append(entry)
    if limit:
        scenes = {name: frames[:limit] for name, frames in scenes.items()}

    written = []
    width, height, fx, fy, cx, cy = calibration
    for scene, frames in sorted(scenes.items()):
        model_dir = os.path.join(out, scene, "sparse")
        image_dir = os.path.join(out, scene, "images")
        depth_dir = os.path.join(out, scene, "depths")
        for folder in (model_dir, image_dir, depth_dir):
            os.makedirs(folder, exist_ok=True)
        lines = ["# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME"]
        usable = 0
        for index, frame in enumerate(sorted(frames, key=lambda item: item["name"]), start=1):
            pose = np.loadtxt(frame["pose"])
            if pose.size != 16:
                continue
            stamp = f"{index - 1:06d}"
            colour = cv2.imread(frame["rgb"], cv2.IMREAD_COLOR)
            if colour is None or not _write_depth(frame["depth"], os.path.join(depth_dir, f"frame-{stamp}.png"), divisor):
                continue
            cv2.imwrite(os.path.join(image_dir, f"frame-{stamp}.jpg"), colour)
            qvec, tvec = _world_to_camera(pose.reshape(4, 4))
            lines.append(f"{index} {qvec[0]:.10g} {qvec[1]:.10g} {qvec[2]:.10g} {qvec[3]:.10g} "
                         f"{tvec[0]:.10g} {tvec[1]:.10g} {tvec[2]:.10g} 1 frame-{stamp}.jpg")
            lines.append("")
            usable += 1
        if usable < 5:
            continue
        with open(os.path.join(model_dir, "images.txt"), "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        with open(os.path.join(model_dir, "cameras.txt"), "w", encoding="utf-8") as handle:
            handle.write(f"1 PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")
        with open(os.path.join(model_dir, "points3D.txt"), "w", encoding="utf-8") as handle:
            handle.write("# per-view depth maps are authoritative for this dataset\n")
        written.append({"scene": scene, "frames": usable, "split": split.get(_split_key(scene), "train")})
    return {"dialect": dialect, "intrinsics": list(calibration), "depth_divisor": divisor, "scenes": written}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="extracted dataset directory")
    parser.add_argument("--out", required=True, help="where to write COLMAP-readable scene models")
    parser.add_argument("--depth-divisor", type=float, default=1000.0,
                        help="raw PNG value per metre: 1000 for millimetres, 5000 for the Shotton original")
    parser.add_argument("--limit", type=int, default=0, help="cap frames per sequence (0 = all)")
    args = parser.parse_args()
    print(json.dumps(convert(args.root, args.out, args.depth_divisor, args.limit), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
