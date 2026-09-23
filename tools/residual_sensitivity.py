#!/usr/bin/env python3
"""Fig. 6: DiT-residual sensitivity of the direct latent pathway vs the RGB cascade.

The perturbation is the near-terminal residual z_45 - z_50 projected onto the Grid-Align width null
space, added to an observed-video latent, and applied identically to both pathways.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from l4d.eval.residual_probe import ResidualSensitivityStudy, near_terminal_residual, null_space_component, width_null_space
from l4d.models.l4ar import L4ARConfig, build_l4ar, vae_spec_from_dict
from l4d.models.video_interface import SyntheticVideoVAE
from l4d.utils.config import load_config


def make_trajectory(shape: tuple[int, ...], steps: int, seed: int = 0) -> list[torch.Tensor]:
    """A converging sampler: starts from noise, moves towards a denoised endpoint (z_50 != z_45)."""
    generator = torch.Generator().manual_seed(seed)
    start = torch.randn(*shape, generator=generator)
    endpoint = 0.55 * start + 0.45 * torch.randn(*shape, generator=generator)
    trajectory = [start]
    for _ in range(steps):
        trajectory.append(0.88 * trajectory[-1] + 0.12 * endpoint)
    return trajectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="configs/model/l4ar_tiny.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--rho", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model_cfg = load_config(args.model).to_dict()
    spec = vae_spec_from_dict(model_cfg["model"]["vae"])
    model = build_l4ar(L4ARConfig.from_dict(model_cfg["model"])).to(args.device).eval()
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        model.load_state_dict(state.get("model", state), strict=False)
    vae = SyntheticVideoVAE(spec)

    frames, height, width = 21, 96, 128
    video = torch.rand(1, 3, frames, height, width, generator=torch.Generator().manual_seed(1))
    observed = vae.posterior_mean(video.to(args.device))
    trajectory = make_trajectory(tuple(observed.shape), args.steps)
    basis = width_null_space(model.alignment.local.conv)
    residual = near_terminal_residual(trajectory, 45, 50)
    component = null_space_component(residual, model.alignment.local.conv, basis)
    print(f"Grid-Align width-null space: {basis.shape[0]} input dims, nullity {basis.shape[1]}")
    print(f"residual ||z45-z50||={float(residual.norm()):.4f} -> null-space component ||.||={float(component.norm()):.4f} "
          f"({100 * float(component.norm() / residual.norm()):.1f}% of the residual)")

    def rgb_round_trip(x: torch.Tensor) -> torch.Tensor:
        frames = x.shape[2]
        return vae.posterior_mean(vae.decode(x))[:, :, :frames]

    def cascade(perturbed_latent: torch.Tensor) -> dict[str, torch.Tensor]:
        """Generate-then-reconstruct path: decode to RGB, re-encode, then reconstruct."""
        return cascade_model(rgb_round_trip(perturbed_latent))

    def cascade_model(x: torch.Tensor) -> dict[str, torch.Tensor]:
        return model(x)

    def token_change(clean_latent: torch.Tensor, perturbed_latent: torch.Tensor) -> float:
        """Fraction of the perturbation that survives Grid-Align (direct path) or the RGB round trip."""
        base = model.alignment(clean_latent)[0]
        moved = model.alignment(perturbed_latent)[0]
        return float((moved - base).norm() / base.norm().clamp(min=1e-12))

    study = ResidualSensitivityStudy(rhos=tuple(args.rho), late_index=45, final_index=50)
    results = study.run(model, cascade=cascade, observed_latent=observed, trajectory=trajectory)
    for result in results:
        perturbed = observed + result.rho * component
        direct = token_change(observed, perturbed)
        via_rgb = token_change(rgb_round_trip(observed), rgb_round_trip(perturbed))
        print(f"rho={result.rho:.1f} token-change ours={direct:.3e} cascade={via_rgb:.3e} ratio={via_rgb / max(direct, 1e-12):.1f}x | "
              f"point drift ours={result.ours_point_drift:.3e} cascade={result.baseline_point_drift:.3e} "
              f"camera drift ours={result.ours_camera_drift:.3e} cascade={result.baseline_camera_drift:.3e} "
              f"favors_ours={result.favors_ours()}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump([result.__dict__ for result in results], fh, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
