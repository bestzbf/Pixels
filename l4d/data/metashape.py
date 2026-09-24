"""Agisoft Metashape project (.psz, a zip holding doc.xml) -> the COLMAP text model this repo reads.

Metashape and COLMAP agree on the camera frame (x right, y down, z forward), so a project converts
without any axis gymnastics:

    camera -> chunk    <camera><transform>          4x4, homogeneous, row-major
    chunk  -> world    <transform><rotation>/<translation>/<scale>

which gives camera-from-world, and COLMAP wants its inverse. The photos themselves are referenced by
`<frame><cameras><camera><photo path=...>`; the sensor block carries a PINHOLE-style calibration plus
Brown k1/k2/k3, which is emitted as an OPENCV camera so `l4d.data.colmap` undistorts it while staging.

Geometry comes from the exported dense mesh (millions of triangles photogrammetrically reconstructed
from these very cameras), sampled to vertices; the point cloud inside the project (`points*.ply`) is the
far sparser SfM tie-point set, so it is only a fallback.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile

import numpy as np


def _matrix(flat: str | None, size: int = 4) -> np.ndarray:
    values = np.array([float(token) for token in (flat or "").split()], dtype=np.float64)
    if values.size == size * size:
        return values.reshape(size, size)
    if size == 3 and values.size == 9:
        return values.reshape(3, 3)
    raise ValueError(f"expected {size * size} numbers, got {values.size}")


def read_project(psz_path: str) -> dict:
    """Parse doc.xml into sensors, world-from-camera poses and the photo file names."""
    with zipfile.ZipFile(psz_path) as archive:
        document = ET.fromstring(archive.read("doc.xml"))

    chunk = next(document.iter("chunk"))
    sensors: dict[str, dict] = {}
    for sensor in chunk.iter("sensor"):
        calibration = sensor.find("calibration")
        if calibration is None:
            continue
        resolution = calibration.find("resolution")
        block = {
            "width": int(resolution.get("width")),
            "height": int(resolution.get("height")),
            "fx": float(calibration.findtext("fx", "0")),
            "fy": float(calibration.findtext("fy", "0")),
            "cx": float(calibration.findtext("cx", "0")),
            "cy": float(calibration.findtext("cy", "0")),
            "k1": float(calibration.findtext("k1", "0")),
            "k2": float(calibration.findtext("k2", "0")),
            "k3": float(calibration.findtext("k3", "0")),
        }
        sensors[sensor.get("id", "0")] = block

    # the chunk->world transform sits next to <region>, after the camera list; Metashape's convention is
    # world = scale * (rotation @ chunk) + translation
    model = chunk.find("transform")
    chunk_from_world = np.eye(4)
    if model is not None:
        rotation = model.find("rotation")
        scale = model.find("scale")
        translation = model.find("translation")
        factor = float(scale.text) if scale is not None and (scale.text or "").strip() else 1.0
        if rotation is not None:
            chunk_from_world[:3, :3] = _matrix(rotation.text, 3) * factor
        if translation is not None:
            chunk_from_world[:3, 3] = np.array([float(v) for v in translation.text.split()])

    cameras = []
    for camera in chunk.iter("camera"):
        transform = camera.find("transform")
        if transform is None or not (transform.text or "").strip():
            continue  # <camera> elements also appear inside <frame> without their own transform
        sensor = sensors.get(camera.get("sensor_id", "0"))
        if sensor is None or not camera.get("label"):
            continue
        camera_from_chunk = _matrix(transform.text)
        camera_from_world = np.linalg.inv(chunk_from_world @ camera_from_chunk)
        rotation = camera_from_world[:3, :3]
        cameras.append({
            "name": camera.get("label"),
            "sensor": sensor,
            "camera_from_world": camera_from_world,
            "qvec": _rotmat2qvec(rotation),
            "tvec": camera_from_world[:3, 3],
        })
    return {"cameras": cameras, "sensors": sensors, "chunk_from_world": chunk_from_world}


def _rotmat2qvec(rotation: np.ndarray) -> np.ndarray:
    """COLMAP's (w, x, y, z) unit quaternion from a world->camera rotation matrix."""
    trace = np.trace(rotation)
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
        w = (rotation[2, 1] - rotation[1, 2]) / scale
        x = 0.25 * scale
        y = (rotation[0, 1] + rotation[1, 0]) / scale
        z = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
        w = (rotation[0, 2] - rotation[2, 0]) / scale
        x = (rotation[0, 1] + rotation[1, 0]) / scale
        y = 0.25 * scale
        z = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
        w = (rotation[1, 0] - rotation[0, 1]) / scale
        x = (rotation[0, 2] + rotation[2, 0]) / scale
        y = (rotation[1, 2] + rotation[2, 1]) / scale
        z = 0.25 * scale
    quaternion = np.array([w, x, y, z])
    return quaternion / np.linalg.norm(quaternion)


def mesh_vertices(obj_path: str, limit: int = 400_000) -> np.ndarray:
    """Vertex positions of an exported mesh/cloud, thinned to roughly `limit` points."""
    total = 0
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                total += 1
    if not total:
        return np.zeros((0, 3))
    stride = max(1, total // limit)
    collected = np.zeros((min(total // stride + 1, limit), 3))
    seen = kept = 0
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            if seen % stride == 0 and kept < len(collected):
                collected[kept] = np.fromstring(line[2:], dtype=np.float64, sep=" ")[:3]
                kept += 1
            seen += 1
    return collected[:kept]


def tie_points(psz_path: str, chunk_from_world: np.ndarray, limit: int = 400_000) -> np.ndarray:
    """Fallback geometry: the sparse tie-point cloud stored in the project, lifted to world units."""
    with zipfile.ZipFile(psz_path) as archive:
        name = next((entry for entry in archive.namelist() if entry.startswith("points") and entry.endswith(".ply")), None)
        if name is None:
            return np.zeros((0, 3))
        raw = archive.read(name)
    head, body, count = _ply_header(raw)
    if head["format"] != "binary_little_endian" or not body:
        return np.zeros((0, 3))
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")] + [(key, "<f4") for key in head["others"]])
    records = np.frombuffer(body, dtype=dtype, count=count)
    points = np.stack([records["x"], records["y"], records["z"]], axis=1).astype(np.float64)
    homogeneous = np.column_stack([points, np.ones(len(points))]) @ chunk_from_world.T
    return homogeneous[:, :3][:limit]


def _ply_header(raw: bytes) -> tuple[dict, bytes, int]:
    text = raw.split(b"end_header", 1)
    lines = text[0].decode("ascii", "ignore").splitlines()
    head = {"format": "ascii", "others": []}
    count = 0
    for line in lines:
        parts = line.split()
        if parts[:2] == ["format", "ascii"] or parts[:2] == ["format", "binary_little_endian"]:
            head["format"] = parts[1]
        elif parts[:2] == ["element", "vertex"]:
            count = int(parts[2])
        elif parts[0] == "property" and parts[1] != "comment" and len(parts) >= 4:
            head["others"].append(parts[3])
    body = text[1].lstrip(b"\r\n") if len(text) > 1 else b""
    return head, body, count


def write_colmap_model(out_dir: str, cameras: list[dict], points: np.ndarray) -> dict:
    """Emit cameras.txt / images.txt / points3D.txt so prepare_colmap.py can stage this scene."""
    import os

    os.makedirs(out_dir, exist_ok=True)
    models: dict[tuple, int] = {}
    lines = ["# Camera list with one line of data per camera:", "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"]
    for index, camera in enumerate(_unique(cameras), start=1):
        sensor = camera["sensor"]
        models[_sensor_key(sensor)] = index
        lines.append(
            f"{index} OPENCV {sensor['width']} {sensor['height']} "
            f"{sensor['fx']} {sensor['fy']} {sensor['cx']} {sensor['cy']} "
            f"{sensor['k1']} {sensor['k2']} 0 0 {sensor['k3']}"
        )
    with open(os.path.join(out_dir, "cameras.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    image_lines = ["# Image list with two lines of data per image:", "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME"]
    for index, camera in enumerate(cameras, start=1):
        qvec, tvec = camera["qvec"], camera["tvec"]
        image_lines.append(
            f"{index} {qvec[0]:.10g} {qvec[1]:.10g} {qvec[2]:.10g} {qvec[3]:.10g} "
            f"{tvec[0]:.10g} {tvec[1]:.10g} {tvec[2]:.10g} {models[_sensor_key(camera['sensor'])]} {camera['name']}"
        )
        image_lines.append("")  # COLMAP's per-image error/points2D line
    with open(os.path.join(out_dir, "images.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(image_lines) + "\n")

    point_lines = ["# 3D point list:", "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]"]
    for index, point in enumerate(points, start=1):
        point_lines.append(f"{index} {point[0]:.8g} {point[1]:.8g} {point[2]:.8g} 128 128 128 1 0:0:0")
    with open(os.path.join(out_dir, "points3D.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(point_lines) + "\n")
    return {"cameras": len(models), "images": len(cameras), "points3D": int(len(points)), "model_dir": out_dir}


def _sensor_key(sensor: dict) -> tuple:
    return (sensor["width"], sensor["height"], sensor["fx"], sensor["fy"], sensor["cx"], sensor["cy"])


def _unique(cameras: list[dict]) -> list[dict]:
    seen: dict[tuple, dict] = {}
    for camera in cameras:
        seen.setdefault(_sensor_key(camera["sensor"]), camera)
    return list(seen.values())
