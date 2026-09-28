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

FLAT = re.compile(r"^(frame-\d{4,6})\.color\.(png|jpe?g)$")
SHOTTON = re.compile(r"^(scene-\d{4}-\d{6})-rgb\.(png|jpg)$")
SENTINEL = 60000  # uint16 saturation: these pixels carry no measurement
SHOTTON_INTRINSICS = (640, 480, 585.0, 585.0, 320.5, 240.5)


def _world_to_camera(pose_camera_to_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    world_to_camera = np.linalg.inv(pose_camera_to_world)
    return _rotmat2qvec(world_to_camera[:3, :3]), world_to_camera[:3, 3]


def _write_depth(source: str, destination: str, divisor: float) -> int:
    """Store the depth map as uint16 millimetres, dropping sentinel pixels."""
    if source.endswith(".pfm"):
        millimetres = np.rint(read_pfm(source) * (1000.0 / divisor)).astype(np.uint16)
        cv2.imwrite(destination, millimetres)
        return int((millimetres > 0).sum())
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


def _capture_key(scene: str) -> str:
    """7-Scenes splits are per *capture*, and capture 3 of chess is not capture 3 of fire, so a bare
    digit key lets whichever split file os.walk reaches last reassign every other scene's test set.
    Scene names are built as "<scene>_<sequence>", so "chess_seq-02" and chess's "sequence2" line have
    to meet on the same key."""
    digits = _split_key(scene)
    return f"{scene.split('_')[0]}_{digits}" if digits else ""


def read_split(root: str) -> dict[str, str]:
    """Honour the benchmark's own TrainSplit.txt / TestSplit.txt instead of inventing a split."""
    split: dict[str, str] = {}
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if not re.match(r"(Train|Test)Split\.txt$", name):
                continue
            kind = "train" if name.startswith("Train") else "test"
            scene = os.path.basename(dirpath.rstrip("/"))
            for line in open(os.path.join(dirpath, name), encoding="utf-8", errors="ignore"):
                if line.strip():
                    split[f"{scene}_{_split_key(line)}"] = kind
    return split


def calibrate_dtu_scan(scan_dir: str, pairs: tuple = ((0, 12), (6, 18), (20, 32))) -> dict:
    """Pick this scan's pose convention and depth unit from the data, not from the format's reputation.

    The mirror mixes preprocessing generations: some scans store a world->camera extrinsic with millimetre
    depth, others the inverse, and a few carry metre depth (where rounding millimetres from a 0.6 m value
    silently produced all-zero maps and 0% coverage). Both choices are decided here by reprojecting two
    distant views of the same static object and taking the reading that puts the two clouds on top of each
    other, so a wrong convention shows up as the losing score instead of as a plausible-looking dataset.
    """
    from scipy.spatial import cKDTree

    cams = sorted(name for name in os.listdir(os.path.join(scan_dir, "cams")) if name.endswith("_cam.txt"))
    depths = sorted(name for name in os.listdir(os.path.join(scan_dir, "gt_depths")) if name.endswith(".pfm"))
    if len(cams) < max(pair[1] for pair in pairs) + 1 or len(depths) < len(cams):
        return {"ok": False, "reason": f"{len(cams)} cams, {len(depths)} depths"}

    # What has to agree is the *ratio* between the depth unit and the camera-translation unit: DTU's
    # preprocessing stores both in millimetres, and dividing only the depth by 1000 put the object 1000x
    # further from the camera than it is, which scores ~0.99 instead of ~0.04. The ratio is measured, and
    # the absolute unit is read off the depth values themselves (>5 means the map is already millimetres).
    raw_median = float(np.median(read_pfm(os.path.join(scan_dir, "gt_depths", depths[0]))))
    depth_is_mm = raw_median > 5.0
    best = None
    for invert in (False, True):
        for ratio in (1.0, 1000.0, 0.001):     # translation unit / depth unit
            scores, usable = [], True
            for a, b in pairs:
                cloud = {}
                for index in (a, b):
                    E, K = _read_dtu_cam(os.path.join(scan_dir, "cams", cams[index]))
                    pose = E if invert else np.linalg.inv(E)        # pose := camera->world
                    metres = read_pfm(os.path.join(scan_dir, "gt_depths", depths[index])) * (1.0 if depth_is_mm else 1000.0)
                    valid = np.isfinite(metres) & (metres > 20.0) & (metres < 20000.0)
                    v, u = np.nonzero(valid)
                    if len(v) < 200:
                        usable = False
                        break
                    step = max(1, len(v) // 1500)
                    v, u = v[::step], u[::step]
                    rays = np.linalg.inv(np.array(K, dtype=np.float64)) @ np.column_stack([u, v, np.ones(len(u))]).T
                    points = np.vstack([rays * metres[v, u][None, :], np.ones((1, len(u)))])
                    world = pose[:3, :3] @ points[:3] + pose[:3, 3:4] * ratio * (1.0 if depth_is_mm else 1000.0)
                    cloud[index] = world.T
                if not usable:
                    break
                joined = np.concatenate([cloud[a], cloud[b]])
                extent = float(np.linalg.norm(joined.max(0) - joined.min(0)))
                scores.append(float(np.median(cKDTree(cloud[b]).query(cloud[a])[0])) / max(extent, 1e-9))
            if not scores:
                continue
            value = float(np.mean(scores))
            if best is None or value < best["wide_baseline_rel"]:
                best = {"ok": value < 0.05, "invert_extrinsic": invert, "translation_over_depth": ratio,
                        "depth_is_mm": depth_is_mm, "wide_baseline_rel": round(value, 5)}
    if best and best["ok"]:
        return best
    return {"ok": False, "reason": f"no convention reached rel < 0.05 (best {best})"}


def _read_dtu_cam(path: str) -> tuple[np.ndarray, list[list[float]]]:
    key, extrinsic, intrinsic = None, [], []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        if line in ("extrinsic", "intrinsic"):
            key = line
            continue
        row = [float(value) for value in line.split()]
        if key == "extrinsic" and len(extrinsic) < 4:
            extrinsic.append(row)
        elif key == "intrinsic" and len(intrinsic) < 3:
            intrinsic.append(row)
    while len(extrinsic) < 4:
        extrinsic.append([0.0, 0.0, 0.0, 1.0])
    return np.array(extrinsic, dtype=np.float64), intrinsic


def scan_dtu(root: str) -> list[dict]:
    """mvsnet-preprocessed DTU: <scan>/{cams,images,gt_depths}/%08d.(txt|png|pfm) + scan.ply."""
    entries = []
    for dirpath, dirnames, _ in os.walk(root):
        if "cams" not in dirnames:
            continue
        scan = os.path.basename(dirpath.rstrip("/"))
        calibration = calibrate_dtu_scan(dirpath)
        if not calibration.get("ok"):
            print(f"[skip ] {scan}: {calibration.get('reason')}", flush=True)
            continue
        print(f"[cal  ] {scan}: {calibration}", flush=True)
        cams = os.path.join(dirpath, "cams")
        for name in sorted(os.listdir(cams)):
            if not name.endswith("_cam.txt"):
                continue
            stem = name[: -len("_cam.txt")]
            image = os.path.join(dirpath, "images", f"{stem}.png")
            depth = os.path.join(dirpath, "gt_depths", f"{stem}.pfm")
            if not (os.path.exists(image) and os.path.exists(depth)):
                continue
            E, _ = _read_dtu_cam(os.path.join(cams, name))
            camera_to_world = (E if calibration["invert_extrinsic"] else np.linalg.inv(E)).copy()
            # staging stores depth as uint16 millimetres but the world frame in metres, so the translation
            # has to leave the depth's own unit (DTU: millimetres) and arrive in metres
            depth_to_mm = 1.0 if calibration["depth_is_mm"] else 1000.0
            depth_unit_in_m = 0.001 if calibration["depth_is_mm"] else 1.0
            camera_to_world[:3, 3] *= calibration["translation_over_depth"] * depth_unit_in_m
            entries.append({"scene": scan, "name": stem, "rgb": image, "pose": os.path.join(cams, name),
                            "depth": depth, "pose_matrix": camera_to_world, "depth_divisor": depth_to_mm})
    return entries


def read_dtu_camera(path: str) -> tuple[np.ndarray, tuple]:
    """Parse one mvsnet `cam.txt`: 4x4 world->camera extrinsic, 3x3 intrinsic, then near/far."""
    lines = [line.strip() for line in open(path, encoding="utf-8") if line.strip()]
    blocks: dict[str, list[str]] = {}
    key = None
    for line in lines:
        if line in ("extrinsic", "intrinsic"):
            key = line
            blocks[key] = []
        elif key == "extrinsic" and len(blocks[key]) < 4:
            blocks[key].append(line)
        elif key == "intrinsic" and len(blocks[key]) < 3:
            blocks[key].append(line)
        elif key == "intrinsic" and len(blocks[key]) == 3:
            blocks["range"] = line.split()
    extrinsic = np.array([[float(v) for v in row.split()] + ([0.0] if len(row.split()) == 3 else [])
                          for row in blocks["extrinsic"]], dtype=np.float64)
    if extrinsic.shape == (3, 4):
        extrinsic = np.vstack([extrinsic, [0, 0, 0, 1]])
    intrinsic = np.array([[float(v) for v in row.split()] for row in blocks["intrinsic"]], dtype=np.float64)
    return np.linalg.inv(extrinsic), (float(intrinsic[0, 0]), float(intrinsic[1, 1]),
                                      float(intrinsic[0, 2]), float(intrinsic[1, 2]))


def read_pfm(path: str) -> np.ndarray:
    """Single- or three-band float pixmap; a negative scale means the rows are stored mirrored."""
    raw = open(path, "rb").read()
    header, pos = [], 0
    for _ in range(3):
        index = raw.index(b"\n", pos)
        header.append(raw[pos:index].decode("ascii"))
        pos = index + 1
    width, height = (int(value) for value in header[1].split())
    scale = float(header[2])
    values = np.frombuffer(raw[pos:], dtype="<f4")
    bands = values.size // (width * height)
    grid = values.reshape(height, width * bands)[:, :width] if bands > 1 else values.reshape(height, width)
    return grid[:, ::-1] if scale < 0 else grid


def scan_nrgbd(root: str) -> list[dict]:
    """Neural RGB-D (NRGBD): <scene>/{images/imgN.png, depth/imgN.png, poses.txt, focal.txt}.

    `poses.txt` is one 4x4 camera->world block per image, four lines each, with no index column and no
    companion file list - so which block belongs to which image is not stated. Reading the blocks in numeric
    image order and taking the poses as delivered is what the released scenes are consistent with, but the
    claim is checked downstream by tools/check_dataset.py: a wrong pairing shows up as cross-view error far
    above threshold and the clip is dropped, not as a plausible-looking number.
    """
    entries = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "poses.txt" not in filenames or "images" not in dirnames:
            continue
        colors = os.path.join(dirpath, "images")
        depths = os.path.join(dirpath, "depth")
        names = [name for name in os.listdir(colors) if name.rsplit(".", 1)[0].startswith("img")]
        names.sort(key=lambda name: int(re.match(r"img(\d+)", name).group(1)))
        poses = _pose_blocks(os.path.join(dirpath, "poses.txt"))
        if len(poses) != len(names):
            print(f"[warn ] {dirpath}: {len(names)} images vs {len(poses)} pose blocks, taking the min", flush=True)
        focal_file = os.path.join(dirpath, "focal.txt")
        focal = float(open(focal_file).read().split()[0]) if os.path.exists(focal_file) else 554.2
        # images are img<i>.png and depths depth<i>.png with i the numeric frame index, so the join has to
        # go through the index: looking for img364.png inside depth/ finds nothing and silently drops a scene
        for index, name in enumerate(names[:len(poses)]):
            stem = name.rsplit(".", 1)[0]
            depth = os.path.join(depths, f"depth{stem[3:]}.png")
            if not os.path.exists(depth):
                continue
            probe = cv2.imread(os.path.join(colors, name), cv2.IMREAD_COLOR)
            entries.append({
                "scene": os.path.basename(dirpath.rstrip("/")),
                "name": f"frame-{index:06d}",
                "rgb": os.path.join(colors, name),
                "depth": depth,
                "pose_matrix": poses[index],
                "focal": focal,
                "size": probe.shape[:2],
            })
    return entries


def _pose_blocks(path: str) -> list[np.ndarray]:
    values = np.loadtxt(path, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[1] != 4 or values.shape[0] % 4:
        raise SystemExit(f"{path}: expected 4-line 4x4 blocks, got shape {values.shape}")
    return [values[index:index + 4] for index in range(0, len(values), 4)]


def convert(root: str, out: str, divisor: float = 1000.0, limit: int = 0) -> dict:
    entries, dialect = scan_nrgbd(root), "nrgbd"
    if not entries:
        entries, dialect = scan_dtu(root), "dtu"
    if not entries:
        entries, dialect = scan_flat(root), "flat"
    if not entries:
        entries, dialect = scan_shotton(root), "shotton"
    if not entries:
        raise SystemExit(f"no DTU / 7-Scenes / NRGBD frames recognised under {root}")
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
        if dialect == "nrgbd":
            height, width = frames[0]["size"]
            fx = fy = frames[0]["focal"]
            cx, cy = width / 2.0, height / 2.0
        elif dialect == "dtu":
            # DTU ships one intrinsic per view file; they are identical within a scan, so read the first
            probe = cv2.imread(frames[0]["rgb"], cv2.IMREAD_COLOR)
            height, width = probe.shape[:2]
            _, (fx, fy, cx, cy) = read_dtu_camera(frames[0]["pose"])
        elif not intrinsic_file:
            # only the fallback constant is left, and it belongs to the 640x480 Shotton frames
            height, width, fx, fy, cx, cy = SHOTTON_INTRINSICS
        lines = ["# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME"]
        usable = 0
        for index, frame in enumerate(sorted(frames, key=lambda item: item["name"]), start=1):
            if dialect in ("nrgbd", "dtu"):
                pose = frame["pose_matrix"]        # already camera->world: calibrated per scan for DTU
            else:
                values = np.loadtxt(frame["pose"])
                if values.size != 16:
                    continue
                pose = values.reshape(4, 4)
            stamp = f"{index - 1:06d}"
            colour = cv2.imread(frame["rgb"], cv2.IMREAD_COLOR)
            if colour is None or not _write_depth(frame["depth"], os.path.join(depth_dir, f"frame-{stamp}.png"),
                                                  frame.get("depth_divisor", divisor)):
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
        written.append({"scene": scene, "frames": usable, "split": split.get(_capture_key(scene), "train")})
    counts: dict[str, int] = {}
    for row in written:
        counts[row["split"]] = counts.get(row["split"], 0) + row["frames"]
    # a benchmark that is supposed to have a test split and reports none has a keying bug, not a dataset;
    # saying it here is what stops every downstream tool from quietly training on the held-out set
    return {"dialect": dialect, "intrinsics": list(calibration), "depth_divisor": divisor,
            "frame_splits": counts, "scenes": written}


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
