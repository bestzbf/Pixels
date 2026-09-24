#!/usr/bin/env python3
"""Turn an Agisoft Metashape project into a scene this repository can stage, export and train on.

    python tools/prepare_metashape.py \
        --psz /path/building.psz --mesh /path/OBJ_export/building.obj --photos /path/building \
        --out data/big_staging_model/building

Writes `<out>/sparse/cameras.txt|images.txt|points3D.txt` plus `<out>/images/` as links to the original
photos, so `tools/prepare_colmap.py --root <out>` picks the scene up like any other COLMAP model: same
undistortion, same depth splatting, same manifest and same `tools/check_dataset.py` gate.

Metashape exports meshes either in world coordinates or in the chunk's own frame; `--mesh-in-chunk-coords`
lifts the vertices by the chunk transform instead of assuming. Which frame is right is decided by
`check_dataset.py`'s cross-view consistency on the staged result, not by this script's default.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from l4d.data.metashape import mesh_vertices, read_project, tie_points, write_colmap_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--psz", required=True, help="Metashape project archive (doc.xml inside)")
    parser.add_argument("--mesh", default=None, help="exported .obj used as the scene geometry")
    parser.add_argument("--photos", required=True, help="directory holding the original images")
    parser.add_argument("--out", required=True, help="scene directory to write (sparse/ + images/)")
    parser.add_argument("--mesh-in-chunk-coords", action="store_true", help="lift mesh vertices by the chunk transform")
    parser.add_argument("--limit", type=int, default=400_000, help="max geometry points kept")
    args = parser.parse_args()

    project = read_project(args.psz)
    cameras = project["cameras"]
    if not cameras:
        raise SystemExit(f"{args.psz}: no calibrated camera carries a transform; the project is unaligned")

    if args.mesh:
        points = mesh_vertices(args.mesh, limit=args.limit)
        source = args.mesh
    else:
        points = tie_points(args.psz, project["chunk_from_world"], limit=args.limit)
        source = f"{args.psz} (tie points, already world)"
        args.mesh_in_chunk_coords = False
    if args.mesh_in_chunk_coords and len(points):
        transform = project["chunk_from_world"]
        points = points @ transform[:3, :3].T + transform[:3, 3]
    if not len(points):
        raise SystemExit(f"no scene geometry found in {source}")

    available = set(os.listdir(args.photos))
    missing = [camera["name"] for camera in cameras if camera["name"] not in available]
    if missing:
        print(f"[warn ] {len(missing)} camera photos absent from {args.photos}: {missing[:3]}", flush=True)
        cameras = [camera for camera in cameras if camera["name"] in available]

    images = os.path.join(args.out, "images")
    os.makedirs(images, exist_ok=True)
    for camera in cameras:
        link = os.path.join(images, camera["name"])
        if not os.path.lexists(link):
            os.symlink(os.path.abspath(os.path.join(args.photos, camera["name"])), link)

    report = write_colmap_model(os.path.join(args.out, "sparse"), cameras, points)
    report["geometry"] = source
    report["extent_m"] = round(float(np.linalg.norm(points.max(0) - points.min(0))), 2)
    print("metashape model:", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
