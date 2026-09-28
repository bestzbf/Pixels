#!/usr/bin/env python3
"""Input-dependence probe: is the predicted point map actually a function of the video?

A reconstruction metric can't tell two failures apart. A model that learned nothing and a model that
learned the wrong thing both report a large rel_err, and the difference matters: the first is an
implementation or loss problem, the second is a budget or data problem. So feed the same weights this
clip and a *different* clip, and measure how far the point maps move. If swapping the video moves the
output far less than the output is wrong, the network is emitting a near-constant shape.

    python tools/input_dependence.py --model configs/model/l4ar_paper.yaml \
        --data configs/data/benchmarks_test.local.yaml --checkpoint runs/benchmarks/stage3_lora.pt \
        --device cuda --clips 8
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from l4d.data.dataset import ReconstructionClips, load_manifest
from l4d.losses.objectives import align_points_scale
from l4d.models.l4ar import build_initialised_model, vae_spec_from_dict
from l4d.models.video_interface import SyntheticVideoVAE, WanVideoVAE, alignment_inputs
from l4d.utils.checkpoint import load_checkpoint
from l4d.utils.config import load_config


@torch.no_grad()
def point_map(model, vae, dataset, index, device):
    """Predicted points for one clip, plus its GT and valid mask."""
    sample = dataset[index]
    video = sample["video"].unsqueeze(0).to(device)
    batch = {"video": video}
    if torch.is_tensor(sample.get("latent")):
        batch["latent"] = sample["latent"].unsqueeze(0)
    latent, grid, output_size = alignment_inputs(model, vae, batch, device)
    out = model(latent, grid=grid, output_size=output_size)
    frames = min(out["points"].shape[1], sample["gt_points"].shape[0])
    return out["points"][0, :frames], sample["gt_points"][:frames], sample["point_mask"][:frames].bool()


def extent_of(ground_truth, mask) -> float:
    return float((ground_truth[mask].abs().amax(0) - ground_truth[mask].abs().amin(0)).norm())


def score(model, vae, dataset, indices, device) -> tuple[float, float]:
    """Mean over clips of (cross-video displacement, rel_err), both in units of the clip's own extent."""
    moves, errors = [], []
    for position, index in enumerate(indices):
        points, ground_truth, mask = point_map(model, vae, dataset, index, device)
        if not bool(mask.any()):
            continue
        # a different clip's latent, fed to the same weights: how much of the output is the video's doing
        foreign, _, _ = point_map(model, vae, dataset, indices[(position + 1) % len(indices)], device)
        extent = max(extent_of(ground_truth, mask), 1e-6)
        aligned = align_points_scale(points.unsqueeze(0), ground_truth.unsqueeze(0), mask.unsqueeze(0))[0]
        errors.append(float((aligned - ground_truth)[mask].norm(dim=-1).mean() / extent))
        # the two clips can differ in frame count, so compare on the overlap
        shared = min(points.shape[0], foreign.shape[0])
        moves.append(float((points[:shared] - foreign[:shared])[mask[:shared]].norm(dim=-1).mean() / extent))
    if not moves:
        return float("nan"), float("nan")
    return sum(moves) / len(moves), sum(errors) / len(errors)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="configs/model/l4ar_paper.yaml")
    parser.add_argument("--data", default="configs/data/benchmarks_test.local.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--clips", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model_cfg = load_config(args.model).to_dict()
    torch.manual_seed(args.seed)                     # a random init is only reproducible through the seed
    model = build_initialised_model(model_cfg, args.device)
    if args.checkpoint:
        report = load_checkpoint(model, args.checkpoint, args.device)
        if report.get("frozen_mismatch"):
            print(f"WARNING: frozen backbone differs from the training init ({report['frozen_mismatch'][:3]})",
                  flush=True)
    else:
        print("baseline: pretrained init only, no trained tensors loaded", flush=True)
    spec = vae_spec_from_dict(model_cfg["model"]["vae"])
    vae = (SyntheticVideoVAE(spec) if model_cfg["model"]["vae"].get("backend") == "synthetic"
           else WanVideoVAE(spec, pretrained=spec.checkpoint_id)).to(args.device).eval()
    data_cfg = load_config(args.data).to_dict()
    records = load_manifest(data_cfg["manifest"], split=data_cfg.get("split"))
    dataset = ReconstructionClips(records, image_size=tuple(data_cfg["resolution"]), frames=int(data_cfg["frames"]))
    indices = list(range(min(args.clips, len(dataset))))

    move, error = score(model, vae, dataset, indices, args.device)
    print(f"{len(indices)} clips: cross-video point displacement {move:.4f} of extent, "
          f"rel_err against GT {error:.4f} of extent")
    print("input-INDEPENDENT: the point map is a near-constant shape, so no scoring protocol will find "
          "an effect in it" if move < 0.1 * error else
          "input-dependent: the output tracks the video, so a large rel_err is a fitting or budget "
          "problem rather than a collapsed head")


if __name__ == "__main__":
    main()
