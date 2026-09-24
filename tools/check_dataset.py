#!/usr/bin/env python3
"""Dataset quality gate: cross-view geometric consistency and duplicate detection per clip.

    python tools/check_dataset.py --data configs/data/real.yaml --threshold 0.05

A clip whose frames are metric 4D annotations must agree with itself: the surface reconstructed from one
view should sit on the surface reconstructed from another. Sparse-depth scenes score well; depth maps that
were rendered independently of the poses do not, and training on the latter would teach wrong geometry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from l4d.data.dataset import ReconstructionClips, load_manifest


def cross_view_error(points: np.ndarray, mask: np.ndarray, max_pairs: int = 8, samples: int = 4000,
                     seed: int = 0) -> tuple[float, float]:
    """Median nearest-neighbour distance between *consecutive* views, absolute and relative to extent.

    Only neighbouring frames are compared: opposite sides of an object legitimately see disjoint surfaces,
    which would make an arbitrary pair score meaningless.
    """
    from scipy.spatial import cKDTree

    frames = [index for index in range(points.shape[0]) if mask[index].sum() > 100]
    if len(frames) < 2:
        return float("nan"), float("nan")
    consecutive = [(first, second) for first, second in zip(frames, frames[1:])]
    pairs = consecutive[:max_pairs]
    clouds = {index: points[index][mask[index]][:samples] for index in {i for pair in pairs for i in pair}}
    extent = float(np.linalg.norm(np.concatenate(list(clouds.values())).max(0)
                                  - np.concatenate(list(clouds.values())).min(0)))
    errors = []
    for first, second in pairs:
        distance, _ = cKDTree(clouds[second]).query(clouds[first])
        errors.append(float(np.median(distance)))
    median = float(np.mean(errors))
    return median, (median / extent if extent > 0 else float("nan"))


def fingerprint(record, dataset: ReconstructionClips, index: int) -> str:
    """Scale-normalised geometry hash, so re-converted copies of one asset are detectable."""
    sample = dataset[index]
    points = sample["gt_points"].numpy().astype(np.float64)
    mask = sample["point_mask"].numpy()
    selected = points[mask]
    if len(selected) < 8:
        return "empty"
    rng = np.random.default_rng(0)
    selected = selected[rng.choice(len(selected), min(2048, len(selected)), replace=False)]
    centre = selected.mean(0)
    extent = np.linalg.norm(selected.max(0) - selected.min(0))
    quantised = np.round((selected - centre) / max(extent, 1e-9) * 48).astype(np.int32)
    order = np.lexsort(quantised.T[::-1])
    digest = hashlib.sha256(np.ascontiguousarray(quantised[order]).tobytes()).hexdigest()[:16]
    return f"{digest}@{extent:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="configs/data/real.yaml")
    parser.add_argument("--threshold", type=float, default=0.05, help="max tolerated relative cross-view error")
    parser.add_argument("--ascent", action="store_true", help="also report the best depth scale per clip")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    sys.path.insert(0, os.getcwd())
    from l4d.utils.config import load_config

    cfg = load_config(args.data).to_dict()
    records = load_manifest(cfg["manifest"], split=cfg.get("split"))
    dataset = ReconstructionClips(records, image_size=tuple(cfg["resolution"]), frames=int(cfg["frames"]))

    seen: dict[str, str] = {}
    report = []
    for index, record in enumerate(records):
        sample = dataset[index]
        points = sample["gt_points"].numpy().astype(np.float64)
        mask = sample["point_mask"].numpy()
        depth = sample["gt_depth"].numpy().astype(np.float64)
        rays = sample["gt_ray_dirs"].numpy().astype(np.float64)
        centers = sample["gt_camera_centers"].numpy().astype(np.float64)
        absolute, relative = cross_view_error(points, mask)
        best_scale, best_relative = 1.0, relative
        if args.ascent and np.isfinite(relative):
            for scale in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0):
                clouds = []
                for view in (0, min(4, points.shape[0] - 1)):
                    sel = mask[view]
                    clouds.append(centers[view][None, :] + (depth[view] * scale)[sel][:, None] * rays[view][sel])
                from scipy.spatial import cKDTree

                distance, _ = cKDTree(clouds[1][:4000]).query(clouds[0][:4000])
                joined = np.concatenate(clouds)
                extent = np.linalg.norm(joined.max(0) - joined.min(0))
                value = float(np.median(distance)) / max(float(extent), 1e-9)
                if value < best_relative:
                    best_scale, best_relative = scale, value
        code = fingerprint(record, dataset, index)
        duplicate_of = seen.get(code)
        seen.setdefault(code, record.clip_id)
        verdict = "drop" if (not np.isfinite(relative) or relative > args.threshold or duplicate_of) else "keep"
        row = {
            "clip": record.clip_id, "frames": int(points.shape[0]), "depth_coverage": round(float(mask.mean()), 4),
            "cross_view_m": None if not np.isfinite(absolute) else round(absolute, 4),
            "cross_view_rel": None if not np.isfinite(relative) else round(relative, 5),
            "best_depth_scale": best_scale, "best_rel": round(best_relative, 5),
            "fingerprint": code, "duplicate_of": duplicate_of, "verdict": verdict,
        }
        report.append(row)
        print(f"{row['clip']:26s} cover={row['depth_coverage']*100:5.1f}% rel={row['cross_view_rel']} "
              f"best_s={best_scale}({best_relative:.4f}) dup={duplicate_of or '-'} -> {verdict}", flush=True)

    kept = [row for row in report if row["verdict"] == "keep"]
    print(f"\n{len(kept)}/{len(report)} clips pass rel<={args.threshold} and are unique")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"threshold": args.threshold, "rows": report, "kept": [row["clip"] for row in kept]}, fh, indent=2)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
