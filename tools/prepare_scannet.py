#!/usr/bin/env python3
"""ScanNet -> L4AR training pool (manifest + GT npz + resized clip frames).

    python tools/prepare_scannet.py --root /path/to/scannet --out data/scannet \
        --split data_splits/scannetv2_train.txt --clips 1143

Then encode the clips with the frozen Wan VAE:
    python tools/precompute_latents.py --data configs/data/scannet.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from l4d.data.dataset import manifest_report
from l4d.data.scannet import discover_scenes, export_manifest, load_scene


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="ScanNet dense-data root (contains <scene>_00/ dirs)")
    parser.add_argument("--out", default="data/scannet")
    parser.add_argument("--split", default=None, help="text file of scene names (e.g. scannetv2_train.txt)")
    parser.add_argument("--scenes", nargs="*", default=None, help="explicit scene subset, overrides --split")
    parser.add_argument("--clip-length", type=int, default=21)
    parser.add_argument("--clip-stride", type=int, default=21)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--clips", type=int, default=0, help="stop after N clips (0 = all)")
    parser.add_argument("--report-only", action="store_true", help="inventory scenes without writing clips")
    args = parser.parse_args()

    scenes = args.scenes or discover_scenes(args.root, args.split)
    if args.report_only:
        inventory = {}
        for scene in scenes:
            loaded = load_scene(args.root, scene)
            inventory[scene] = len(loaded.colors) if loaded else "unrecognised layout"
        usable = sum(1 for value in inventory.values() if isinstance(value, int))
        print(json.dumps({"scenes": len(scenes), "usable": usable, "detail": inventory}, indent=2))
        if usable == 0 and scenes:
            print("\nno scene matched either supported layout; top-level listing of the first scene:", flush=True)
            first = os.path.join(args.root, scenes[0])
            print(sorted(os.listdir(first))[:20])
        return

    summary = export_manifest(
        args.root, args.out, clip_length=args.clip_length, clip_stride=args.clip_stride,
        frame_stride=args.frame_stride, size=(args.height, args.width), scenes=scenes, max_clips=args.clips,
    )
    print(json.dumps(summary, indent=2))
    from l4d.data.dataset import load_manifest

    print("manifest:", json.dumps(manifest_report(load_manifest(summary["manifest"]))))


if __name__ == "__main__":
    main()
