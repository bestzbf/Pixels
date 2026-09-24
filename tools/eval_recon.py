#!/usr/bin/env python3
"""Per-clip reconstruction check on real local data: relative point error, Acc/Comp, PLY dumps.

    python tools/eval_recon.py --model configs/model/l4ar_probe.yaml \
        --checkpoint runs/gpu_colmap_300/stage3_lora.pt --data configs/data/colmap.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from l4d.data.dataset import ReconstructionClips, load_manifest
from l4d.eval.gt_metrics import accuracy_completeness
from l4d.losses.objectives import align_points_scale
from l4d.models.l4ar import L4ARConfig, build_l4ar, vae_spec_from_dict
from l4d.models.video_interface import SyntheticVideoVAE, WanVideoVAE
from l4d.utils.config import load_config


def write_ply(path: str, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    step = max(1, points.shape[0] // 200_000)
    points, colors = points[::step], None if colors is None else colors[::step]
    with open(path, "w", encoding="ascii") as fh:
        fh.write("ply\nformat ascii 1.0\nelement vertex %d\n" % points.shape[0])
        for axis in ("x", "y", "z"):
            fh.write(f"property float {axis}\n")
        if colors is not None:
            for channel in ("red", "green", "blue"):
                fh.write(f"property uchar {channel}\n")
        fh.write("end_header\n")
        for index, point in enumerate(points):
            tail = "" if colors is None else " " + " ".join(str(int(value)) for value in colors[index])
            fh.write(f"{point[0]:.5f} {point[1]:.5f} {point[2]:.5f}{tail}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="configs/model/l4ar_probe.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--from-scratch", action="store_true", help="random initialisation, for comparison")
    parser.add_argument("--data", default="configs/data/colmap.yaml")
    parser.add_argument("--index", type=int, default=0, help="clip index to dump PLYs for")
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model_cfg = load_config(args.model).to_dict()
    data_cfg = load_config(args.data).to_dict()
    model = build_l4ar(L4ARConfig.from_dict(model_cfg["model"])).to(args.device).eval()
    if args.checkpoint and not args.from_scratch:
        state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        model.load_state_dict(state.get("model", state), strict=False)
    else:
        print("baseline: randomly initialised model (no checkpoint loaded)")

    spec = vae_spec_from_dict(model_cfg["model"]["vae"])
    vae = (SyntheticVideoVAE(spec) if model_cfg["model"]["vae"].get("backend") == "synthetic"
           else WanVideoVAE(spec, pretrained=spec.checkpoint_id)).to(args.device).eval()

    records = load_manifest(data_cfg["manifest"])
    resolution = tuple(data_cfg["resolution"])
    dataset = ReconstructionClips(records, image_size=resolution, frames=int(data_cfg["frames"]))
    rows = []
    for index, record in enumerate(records):
        sample = dataset[index]
        video = sample["video"].unsqueeze(0).to(args.device)
        with torch.no_grad():
            latent = vae.posterior_mean(video)
            prediction = model(latent)
        frames = min(prediction["points"].shape[1], sample["gt_points"].shape[0])
        predicted = prediction["points"][:, :frames][0].float().cpu()
        ground_truth = sample["gt_points"][:frames]
        mask = sample["point_mask"][:frames].bool()
        aligned = align_points_scale(predicted.unsqueeze(0), ground_truth.unsqueeze(0), mask.unsqueeze(0))[0]
        extent = float((ground_truth[mask].float().abs().amax(0) - ground_truth[mask].float().amin(0)).norm())
        error = (aligned - ground_truth).norm(dim=-1)
        selected = mask
        scores = accuracy_completeness(
            aligned[selected].float(), ground_truth[selected].float(), threshold=max(0.02 * extent, 0.05)
        )
        rows.append({
            "clip": record.clip_id, "frames": frames, "scene_extent_m": round(extent, 1),
            "mean_rel_err": round(float(error[selected].mean() / max(extent, 1e-6)), 4),
            "acc_m": round(scores["accuracy"], 3), "comp_m": round(scores["completeness"], 3),
        })
        if index == args.index and args.out:
            os.makedirs(args.out, exist_ok=True)
            write_ply(os.path.join(args.out, "gt.ply"), ground_truth[selected].numpy().astype("float32"))
            write_ply(os.path.join(args.out, "pred.ply"), aligned[selected].numpy().astype("float32"))
            print(f"wrote {args.out}/gt.ply and pred.ply ({int(selected.sum())} points)")
    for row in rows:
        print(f"{row['clip']:26s} T={row['frames']:2d} extent={row['scene_extent_m']:6.1f}m "
              f"rel_err={row['mean_rel_err']:.4f} acc={row['acc_m']:5.3f}m comp={row['comp_m']:5.3f}m")
    usable = [row for row in rows if np.isfinite(row["acc_m"])]
    if usable:
        print(f"mean relative point error over {len(usable)} clips: "
              f"{sum(row['mean_rel_err'] for row in usable) / len(usable):.4f} of scene extent")


if __name__ == "__main__":
    main()
