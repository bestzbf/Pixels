#!/usr/bin/env python3
"""Inference: latent -> L4AR -> cameras + dynamic world-space point maps (Eq. 4 and 6).

    # from an observed video (training-time interface, z_obs = mu(E_v(V)))
    python tools/infer.py --video data/clips/scene0/frame_0 --out runs/infer/scene0
    # from a generated DiT latent (generation-time interface, z_gen)
    python tools/infer.py --latent runs/latents/case_007.pt --dit Wan2.1-T2V-1.3B --out runs/infer/case_007
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from l4d.models.l4ar import L4ARConfig, build_l4ar, vae_spec_from_dict
from l4d.models.video_interface import CompatibleGenerators, LatentProvenance, SharedLatentInterface, SyntheticVideoVAE, WanVideoVAE
from l4d.utils.config import load_config


def load_model(config_path: str, checkpoint: str | None, device: str):
    model_cfg = load_config(config_path).to_dict()
    model = build_l4ar(L4ARConfig.from_dict(model_cfg["model"])).to(device).eval()
    if checkpoint:
        from l4d.utils.checkpoint import load_checkpoint

        report = load_checkpoint(model, checkpoint, device)
        if report.get("frozen_mismatch"):
            print(f"WARNING: {len(report['frozen_mismatch'])} frozen tensors differ from the ones that trained "
                  f"(e.g. {report['frozen_mismatch'][:3]}) - the pretrained init of this build is not the "
                  "training init, so outputs are not meaningful", flush=True)
    return model, model_cfg


def load_video(path: str, frames: int, size: tuple[int, int], device: str) -> torch.Tensor:
    from PIL import Image

    names = sorted(os.listdir(path)) if os.path.isdir(path) else [path]
    names = [name for name in names if name.lower().endswith((".png", ".jpg", ".jpeg"))]
    index = np.linspace(0, len(names) - 1, frames).round().astype(int)
    tensors = []
    for position in index:
        image = Image.open(os.path.join(path, names[position])).convert("RGB").resize((size[1], size[0]))
        tensors.append(torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1))
    return torch.stack(tensors, dim=1).unsqueeze(0).to(device)


def write_ply(path: str, points: np.ndarray, colors: np.ndarray | None) -> None:
    with open(path, "w", encoding="ascii") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {points.shape[0]}\n")
        for axis in ("x", "y", "z"):
            fh.write(f"property float {axis}\n")
        if colors is not None:
            for channel in ("red", "green", "blue"):
                fh.write(f"property uchar {channel}\n")
        fh.write("end_header\n")
        for i, point in enumerate(points):
            tail = "" if colors is None else " " + " ".join(str(int(c)) for c in (colors[i] * 255).clip(0, 255))
            fh.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}{tail}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="configs/model/l4ar_tiny.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--video", default=None)
    parser.add_argument("--latent", default=None)
    parser.add_argument("--frames", type=int, default=21)
    parser.add_argument("--resolution", type=int, nargs=2, default=(192, 256))
    parser.add_argument("--dit", default=None, help="compatible generator name for provenance checking")
    parser.add_argument("--out", default="runs/infer")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    model, model_cfg = load_model(args.model, args.checkpoint, device)
    spec = vae_spec_from_dict(model_cfg["model"]["vae"])
    backend_cfg = model_cfg["model"]["vae"]
    vae = SyntheticVideoVAE(spec) if backend_cfg.get("backend") == "synthetic" else WanVideoVAE(spec, backend_cfg.get("checkpoint_id"))
    vae = vae.to(device).eval()
    interface = SharedLatentInterface(spec)

    provenance = None
    if args.video:
        video = load_video(args.video, args.frames, tuple(args.resolution), device)
        latent = interface.normalize(vae.posterior_mean(video))
        provenance = LatentProvenance(source="observed_vae", generator="observed-video", vae=spec)
    elif args.latent:
        payload = torch.load(args.latent, map_location=device, weights_only=False)
        latent = payload["latent"] if isinstance(payload, dict) and "latent" in payload else payload
        latent = latent.unsqueeze(0) if latent.dim() == 4 else latent
        generator = args.dit or (payload.get("generator") if isinstance(payload, dict) else "unknown")
        provenance = LatentProvenance(source="generated_dit", generator=str(generator), vae=spec)
        interface.check_compatible(provenance, latent)
    else:
        frames_latent = 1 + (args.frames - 1) // spec.temporal_compression
        latent = torch.randn(1, spec.channels, frames_latent, args.resolution[0] // spec.spatial_compression,
                             args.resolution[1] // spec.spatial_compression, device=device)
        provenance = LatentProvenance(source="generated_dit", generator=args.dit or "random-latent-probe", vae=spec)
        print(f"no --video/--latent given: sampling a random {tuple(latent.shape)} latent in {spec.name} space")

    with torch.no_grad():
        out = model(latent)

    os.makedirs(args.out, exist_ok=True)
    summary = {key: tuple(value.shape) for key, value in out.items() if torch.is_tensor(value)}
    print("outputs:", summary)
    print("interface provenance:", provenance.source, provenance.generator)
    if args.dit and args.dit in CompatibleGenerators().incompatible:
        print(f"note: {args.dit} is outside the shared-VAE family; only cascade baselines apply to it")
    points = out["points"][0].cpu().numpy()
    colors = None
    if "colors" in out:
        colors = out["colors"][0].cpu().numpy()
    save = {key: value.float().cpu().numpy() for key, value in out.items() if torch.is_tensor(value)}
    np.savez_compressed(os.path.join(args.out, "prediction.npz"), **save)
    write_ply(
        os.path.join(args.out, "frame0.ply"),
        points[0].reshape(-1, 3),
        None if colors is None else colors[0].reshape(-1, 3),
    )
    print(f"wrote {os.path.join(args.out, 'prediction.npz')} and frame0.ply")


if __name__ == "__main__":
    main()
