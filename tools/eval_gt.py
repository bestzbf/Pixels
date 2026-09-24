#!/usr/bin/env python3
"""Table 3: component ablation on the ground-truth benchmarks (7-Scenes / NRGBD).

    python tools/eval_gt.py --benchmark 7scenes --variants Full "w/o 3D Conv" "w/o Frame" "w/o Global" "w/o Grid"
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from l4d.data.dataset import SyntheticClips
from l4d.eval.gt_metrics import NRGBD, SEVEN_SCENES, evaluate_prediction
from l4d.eval.protocol import TABLE_3_REFERENCE, TABLE_3_VARIANTS
from l4d.models.l4ar import L4ARConfig, build_l4ar
from l4d.models.video_interface import SyntheticVideoVAE
from l4d.utils.config import load_config

BENCHMARKS = {B.name: B for B in (SEVEN_SCENES, NRGBD)}


def build_variant(model_cfg: dict, variant: str, device: str, checkpoint: str | None):
    cfg = copy.deepcopy(model_cfg["model"])
    for key, value in TABLE_3_VARIANTS[variant].items():
        cfg[key] = value
    model = build_l4ar(L4ARConfig.from_dict(cfg)).to(device).eval()
    if checkpoint:
        from l4d.utils.checkpoint import load_checkpoint
        report = load_checkpoint(model, checkpoint, device)
        if report.get("frozen_mismatch"):
            print(f"WARNING: frozen backbone differs from the training init ({report['frozen_mismatch'][:3]})", flush=True)
    return model


@torch.no_grad()
def evaluate_variant(model, vae, dataset: SyntheticClips, benchmark, device: str, threshold_cm: float) -> dict[str, float]:
    accumulates = {"Acc": [], "Comp": [], "NC": []}
    for index in range(len(dataset)):
        sample = dataset[index]
        video = sample["video"].unsqueeze(0).to(device)
        latent = vae.posterior_mean(video)
        out = model(latent)
        frames = min(out["points"].shape[1], sample["gt_points"].shape[0])
        prediction = out["points"][0, :frames]
        ground_truth = sample["gt_points"][:frames]
        scores = evaluate_prediction(prediction, ground_truth, benchmark, threshold_cm=threshold_cm)
        for key in accumulates:
            if torch.isfinite(torch.tensor(scores[key])):
                accumulates[key].append(scores[key])
    return {key: (sum(values) / len(values) if values else float("nan")) for key, values in accumulates.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="configs/model/l4ar_tiny.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--benchmark", default="7scenes", choices=sorted(BENCHMARKS))
    parser.add_argument("--variants", nargs="+", default=list(TABLE_3_VARIANTS), choices=list(TABLE_3_VARIANTS))
    parser.add_argument("--clips", type=int, default=2)
    parser.add_argument("--threshold-cm", type=float, default=5.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    model_cfg = load_config(args.model).to_dict()
    vae = SyntheticVideoVAE(L4ARConfig.from_dict(model_cfg["model"]).vae)
    dataset = SyntheticClips(size=args.clips, frames=21, height=32, width=40)
    benchmark = BENCHMARKS[args.benchmark]
    rows = {}
    for variant in args.variants:
        model = build_variant(model_cfg, variant, args.device, args.checkpoint)
        metrics = evaluate_variant(model, vae, dataset, benchmark, args.device, args.threshold_cm)
        rows[variant] = metrics
        reference = TABLE_3_REFERENCE[variant][args.benchmark]
        print(f"{variant:14s} Acc={metrics['Acc']:.3f} Comp={metrics['Comp']:.3f} NC={metrics['NC']:.3f} "
              f"| paper Acc={reference['Acc']:.3f} Comp={reference['Comp']:.3f} NC={reference['NC']:.3f}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"benchmark": args.benchmark, "rows": rows, "reference": TABLE_3_REFERENCE}, fh, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
