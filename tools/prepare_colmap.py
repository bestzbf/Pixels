#!/usr/bin/env python3
"""Local COLMAP scenes -> ScanNet-style staging tree -> L4AR clip manifest (real poses + intrinsics).

    python tools/prepare_colmap.py --root "<dir with COLMAP scenes>" --out data/colmap
    python tools/prepare_colmap.py --root ... --report-only      # inventory first

Depth is the scene's own `points3D.txt` cloud reprojected per view, so supervision is sparse but real;
every term in Eq. 7 is masked to observed pixels.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from l4d.data.colmap import find_model_dirs, scene_stats, scene_to_scannet_tree
from l4d.data.dataset import load_manifest, manifest_report
from l4d.data.scannet import export_manifest


def resolve_image_root(model_dir: str) -> str:
    """COLMAP names are relative to the reconstruction root, usually a sibling `images/` folder."""
    parent = os.path.dirname(model_dir)
    for candidate in (os.path.join(parent, "images"), parent, os.path.join(model_dir, "images")):
        if os.path.isdir(candidate):
            return candidate
    return parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="directory searched recursively for COLMAP models")
    parser.add_argument("--staging", default="data/colmap_staging", help="ScanNet-style tree written here")
    parser.add_argument("--out", default="data/colmap", help="clips/, gt/, manifest.jsonl")
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--frames", type=int, default=21, help="frames per clip; rounded down to 4k+1")
    parser.add_argument("--splat-radius", type=int, default=2, help="px radius each sparse point is painted over")
    parser.add_argument("--dataset", default="colmap", help="dataset tag written into the manifest")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    models = find_model_dirs(args.root)
    if not models:
        raise SystemExit(f"no COLMAP text models (cameras.txt + images.txt) found under {args.root}")

    if args.report_only:
        print(json.dumps({model: scene_stats(model) for model in models}, indent=2, ensure_ascii=False))
        return

    staging_scenes = []
    for model_dir in models:
        name = os.path.basename(os.path.dirname(model_dir)) or os.path.basename(model_dir)
        name = name.replace(" ", "_")[:48]
        created = scene_to_scannet_tree(
            model_dir, resolve_image_root(model_dir), args.staging, name,
            size=(args.height, args.width), max_frames=args.frames, splat_radius=args.splat_radius,
        )
        if created:
            staging_scenes.append(os.path.basename(created))
            print(f"[stage] {name}: {len(os.listdir(os.path.join(created)))//3} frames", flush=True)
        else:
            print(f"[skip ] {model_dir}: unreadable model or too few images", flush=True)
    if not staging_scenes:
        raise SystemExit("nothing staged; check --root and whether images resolve against cameras.txt names")

    summary = export_manifest(
        args.staging, args.out, clip_length=1 + 4 * ((args.frames - 1) // 4), clip_stride=args.frames,
        size=(args.height, args.width), scenes=staging_scenes, dataset=args.dataset,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("manifest:", json.dumps(manifest_report(load_manifest(summary["manifest"]))))


if __name__ == "__main__":
    main()
