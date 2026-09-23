#!/usr/bin/env python3
"""Precompute frozen-VAE posterior-mean latents for the training clips (z^obs = mu(E_v(V))).

The VAE and all video models stay frozen; caching keeps training free of video decoding.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from l4d.data.dataset import LatentCache, ReconstructionClips, SyntheticClips, collate_clips, load_manifest, manifest_report
from l4d.models.l4ar import vae_spec_from_dict
from l4d.models.video_interface import SyntheticVideoVAE, WanVideoVAE
from l4d.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="configs/data/clips_final.yaml")
    parser.add_argument("--model", default="configs/model/l4ar_paper.yaml")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data_cfg = load_config(args.data).to_dict()
    model_cfg = load_config(args.model).to_dict()
    spec = vae_spec_from_dict(model_cfg["model"]["vae"])
    if model_cfg["model"]["vae"].get("backend") == "synthetic":
        vae = SyntheticVideoVAE(spec)
        dataset = SyntheticClips(size=int(data_cfg.get("size", 4)), frames=int(data_cfg.get("frames", 21)))
    else:
        vae = WanVideoVAE(spec, pretrained=data_cfg.get("checkpoint_id", spec.checkpoint_id))
        records = load_manifest(data_cfg["manifest"], split=data_cfg.get("split"))
        print("manifest:", json.dumps(manifest_report(records)), flush=True)
        dataset = ReconstructionClips(records, image_size=tuple(data_cfg["resolution"]), frames=int(data_cfg["frames"]))
    vae = vae.to(args.device).eval()
    cache = LatentCache(data_cfg["latent_cache"])
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_clips)
    written = 0
    for index, batch in enumerate(loader):
        if args.limit and written >= args.limit:
            break
        with torch.no_grad():
            latent = vae.posterior_mean(batch["video"].to(args.device))
        for case_id in batch["clip_id"]:
            cache.save(case_id, latent.detach().cpu(), {"vae": spec.name, "shape": list(latent.shape)})
            written += 1
        if (index + 1) % 25 == 0:
            print(f"encoded {written} clips", flush=True)
    print(f"done: {written} latents cached in {data_cfg['latent_cache']}")


if __name__ == "__main__":
    main()
