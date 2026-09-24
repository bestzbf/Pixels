#!/usr/bin/env python3
"""Table 3: component ablation, on the paper's 7-Scenes / NRGBD protocol or on any local clip pool.

    # the paper's benchmarks
    python tools/eval_gt.py --benchmark 7scenes --variants Full "w/o 3D Conv" "w/o Frame" "w/o Global" "w/o Grid"
    # whatever real data is actually on this machine (ScanNet until it arrives)
    python tools/eval_gt.py --data configs/data/real.yaml --model configs/model/l4ar_probe.yaml \
        --checkpoint runs/gpu_real_4rc_camdec/stage3_lora.pt --benchmark local
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from l4d.data.dataset import ReconstructionClips, SyntheticClips, load_manifest
from l4d.eval.gt_metrics import GTBenchmark, NRGBD, SEVEN_SCENES, evaluate_prediction
from l4d.eval.protocol import TABLE_3_REFERENCE, TABLE_3_VARIANTS
from l4d.models.l4ar import L4ARConfig, apply_pretrained_init, build_l4ar
from l4d.models.video_interface import SyntheticVideoVAE, WanVideoVAE
from l4d.utils.config import load_config

BENCHMARKS = {B.name: B for B in (SEVEN_SCENES, NRGBD)}


def build_variant(model_cfg: dict, variant: str, device: str, checkpoint: str | None, seed: int = 0):
    cfg = copy.deepcopy(model_cfg["model"])
    for key, value in TABLE_3_VARIANTS[variant].items():
        cfg[key] = value
    # a randomly initialised backbone is only reproducible through the seed that drew it
    torch.manual_seed(seed)
    model = build_l4ar(L4ARConfig.from_dict(cfg)).to(device)
    # same contract as tools/eval_recon.py: rebuild the frozen backbone training saw before restoring heads
    apply_pretrained_init(model, model_cfg.get("pretrained_init", {}))
    model.eval()
    if checkpoint:
        from l4d.utils.checkpoint import load_checkpoint
        report = load_checkpoint(model, checkpoint, device)
        if report.get("frozen_mismatch"):
            print(
                f"WARNING: frozen backbone differs from the training init ({report['frozen_mismatch'][:3]}) - "
                "this model is not the one that was trained, so its numbers are not comparable",
                flush=True,
            )
    return model


@torch.no_grad()
def evaluate_variant(model, vae, dataset, benchmark, device: str, threshold_cm: float) -> dict[str, float]:
    from l4d.models.video_interface import SharedLatentInterface

    interface = SharedLatentInterface(model.cfg.vae)
    accumulates = {"Acc": [], "Comp": [], "NC": []}
    for index in range(len(dataset)):
        sample = dataset[index]
        video = sample["video"].unsqueeze(0).to(device)
        latent = sample["latent"].unsqueeze(0).to(device) if torch.is_tensor(sample.get("latent")) \
            else vae.posterior_mean(video)
        latent = interface.normalize(latent)
        frames = video.shape[2]
        grid = (frames, max(latent.shape[3] // model.cfg.patch_size, 1), max(latent.shape[4] // model.cfg.patch_size, 1))
        out = model(latent, grid=grid, output_size=(video.shape[3], video.shape[4]))
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
    parser.add_argument("--data", default=None, help="clip pool yaml (e.g. configs/data/real.yaml); omit for synthetic")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--benchmark", default="7scenes", help="7scenes | nrgbd | any local pool name")
    parser.add_argument("--variants", nargs="+", default=list(TABLE_3_VARIANTS), choices=list(TABLE_3_VARIANTS))
    parser.add_argument("--clips", type=int, default=2)
    parser.add_argument("--threshold-cm", type=float, default=5.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=0,
                        help="must match the training seed: a randomly initialised backbone is only reproducible through it")
    args = parser.parse_args()

    model_cfg = load_config(args.model).to_dict()
    spec = L4ARConfig.from_dict(model_cfg["model"]).vae
    if args.data:
        data_cfg = load_config(args.data).to_dict()
        backend = model_cfg["model"]["vae"].get("backend", "wan")
        vae = (SyntheticVideoVAE(spec) if backend == "synthetic"
               else WanVideoVAE(spec, pretrained=spec.checkpoint_id))
        records = load_manifest(data_cfg["manifest"], split=data_cfg.get("split"))
        dataset = ReconstructionClips(
            records, image_size=tuple(data_cfg["resolution"]), frames=int(data_cfg["frames"])
        )
    else:
        vae = SyntheticVideoVAE(spec)
        dataset = SyntheticClips(size=args.clips, frames=21, height=32, width=40)
    benchmark = BENCHMARKS.get(args.benchmark) or GTBenchmark(args.benchmark, sequences=len(dataset))
    rows = {}
    for variant in args.variants:
        model = build_variant(model_cfg, variant, args.device, args.checkpoint, args.seed)
        metrics = evaluate_variant(model, vae, dataset, benchmark, args.device, args.threshold_cm)
        rows[variant] = metrics
        reference = TABLE_3_REFERENCE.get(variant, {}).get(args.benchmark)
        paper = (f"| paper Acc={reference['Acc']:.3f} Comp={reference['Comp']:.3f} NC={reference['NC']:.3f}"
                 if reference else "| (local pool: no paper reference row)")
        print(f"{variant:14s} Acc={metrics['Acc']:.3f} Comp={metrics['Comp']:.3f} NC={metrics['NC']:.3f} {paper}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"benchmark": args.benchmark, "rows": rows, "reference": TABLE_3_REFERENCE}, fh, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
