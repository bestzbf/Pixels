#!/usr/bin/env python3
"""Build / verify the locked 200-case benchmarks (Text4D-200, I4D-200).

The paper's benchmarks are locked and every method runs on every case; a lock file with the case ids and
sha256 digests makes the comparison reproducible. Generation mode uses a frozen compatible DiT to produce
the *terminal* latents (the same latents feed the matched cascade, so the comparison is controlled).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from l4d.models.l4ar import vae_spec_from_dict
from l4d.models.video_interface import CompatibleGenerators, DiTFinalLatentExtractor, SyntheticVideoVAE
from l4d.utils.config import load_config


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_latent(vae, generator: str, prompt: str, image: str | None, frames: int, size: tuple[int, int],
                  steps: int, seed: int, device: str) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Placeholder sampler: draws a latent in the shared VAE space.

    Replace `diffusers_call` with the real frozen DiT pipeline call; the returned tensor must be the
    latent *before* VAE decoding, and the trajectory must keep z_45 / z_50 for the Fig. 6 diagnostic.
    """
    def diffusers_call(condition, callback):
        shape = (1, spec.channels, 1 + (frames - 1) // spec.temporal_compression, size[0] // spec.spatial_compression,
                 size[1] // spec.spatial_compression)
        noise = torch.randn(shape, generator=torch.Generator(device="cpu").manual_seed(seed), device=device)
        latent = noise
        for step in range(steps):
            latent = 0.98 * latent + 0.02 * noise
            callback(step, latent)
        return latent

    spec = vae.spec
    extractor = DiTFinalLatentExtractor(spec, generator=generator, record_indices=(45, 50))
    condition = {"prompt": prompt, "image": image}
    with torch.no_grad():
        z, provenance = extractor(diffusers_call, condition, num_steps=steps)
    return z, provenance.trajectory or []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", dest="eval_cfg", default="configs/eval/text4d200.yaml")
    parser.add_argument("--model", default="configs/model/l4ar_paper.yaml")
    parser.add_argument("--conditions", default=None, help="jsonl with {case_id, prompt, image?}")
    parser.add_argument("--generator", default="Wan2.1-T2V-1.3B")
    parser.add_argument("--frames", type=int, default=21)
    parser.add_argument("--resolution", type=int, nargs=2, default=(192, 256))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.generator not in CompatibleGenerators().all_compatible:
        raise SystemExit(f"{args.generator} is outside the shared-VAE family; only cascade baselines apply")

    eval_cfg = load_config(args.eval_cfg).to_dict()
    model_cfg = load_config(args.model).to_dict()
    spec = vae_spec_from_dict(model_cfg["model"]["vae"])
    vae = SyntheticVideoVAE(spec) if model_cfg["model"]["vae"].get("backend") == "synthetic" else None
    if vae is None:
        from l4d.models.video_interface import WanVideoVAE

        vae = WanVideoVAE(spec, pretrained=spec.checkpoint_id)

    cases_dir = eval_cfg["cases_dir"]
    os.makedirs(cases_dir, exist_ok=True)
    if args.conditions:
        conditions = [json.loads(line) for line in open(args.conditions, encoding="utf-8") if line.strip()]
    else:
        conditions = [{"case_id": f"case_{i:03d}", "prompt": f"synthetic prompt {i}"} for i in range(args.size)]
    conditions = conditions[: args.size]
    if len(conditions) != 200:
        print(f"warning: benchmark holds {len(conditions)} cases, the paper's suite is locked at 200")

    manifest = []
    for offset, condition in enumerate(conditions):
        case_id = str(condition["case_id"])
        latent, trajectory = sample_latent(
            vae, args.generator, condition.get("prompt"), condition.get("image"), args.frames,
            tuple(args.resolution), args.steps, args.seed + offset, args.device,
        )
        payload = {
            "case_id": case_id,
            "latent": latent.detach().cpu(),
            "text": condition.get("prompt"),
            "generator": args.generator,
            "vae": spec.name,
            "trajectory": [step.detach().cpu() for step in trajectory],
        }
        target = os.path.join(cases_dir, f"{case_id}.pt")
        torch.save(payload, target)
        manifest.append({"case_id": case_id, "path": target, "sha256": sha256(target)})

    lock_path = eval_cfg["lock_file"]
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as fh:
        json.dump({"benchmark": eval_cfg["benchmark"], "cases": len(manifest), "generator": args.generator,
                   "items": manifest}, fh, indent=2)
    print(f"locked {len(manifest)} cases -> {lock_path}")


if __name__ == "__main__":
    main()
