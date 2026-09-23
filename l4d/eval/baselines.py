"""Generate-then-reconstruct cascades and the Table 1 reference numbers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

#: Paper Table 1 (x100, higher is better): method -> (Text CLIP, CLIP-I, DINO global, DINO match, DINO F1)
TABLE_1_REFERENCE: dict[str, dict[str, float]] = {
    "CogVideoX-5B + 4RC": {"text_clip": 26.887, "clip_i": 75.33, "dino_global": 42.71, "dino_match": 53.47, "dino_f1": 53.27},
    "CogVideoX-5B + pi3": {"text_clip": 26.293, "clip_i": 74.84, "dino_global": 38.74, "dino_match": 50.29, "dino_f1": 49.72},
    "CogVideoX-5B + Any4D": {"text_clip": 25.414, "clip_i": 72.66, "dino_global": 29.46, "dino_match": 45.97, "dino_f1": 45.12},
    "Wan2.1-14B + 4RC": {"text_clip": 28.116, "clip_i": 71.20, "dino_global": 42.45, "dino_match": 54.31, "dino_f1": 53.56},
    "Wan2.1-14B + pi3": {"text_clip": 26.594, "clip_i": 68.70, "dino_global": 34.79, "dino_match": 48.69, "dino_f1": 47.50},
    "Wan2.1-14B + Any4D": {"text_clip": 26.261, "clip_i": 67.56, "dino_global": 29.97, "dino_match": 46.80, "dino_f1": 45.55},
    "Wan2.1-1.3B + 4RC": {"text_clip": 27.829, "clip_i": 71.34, "dino_global": 43.30, "dino_match": 54.97, "dino_f1": 54.21},
    "Wan2.1-1.3B + pi3": {"text_clip": 27.034, "clip_i": 70.23, "dino_global": 38.51, "dino_match": 51.23, "dino_f1": 50.10},
    "Wan2.1-1.3B + Any4D": {"text_clip": 26.326, "clip_i": 68.19, "dino_global": 31.00, "dino_match": 47.26, "dino_f1": 46.06},
    "Ours (Wan2.1-14B)": {"text_clip": 28.544, "clip_i": 72.24, "dino_global": 45.43, "dino_match": 57.52, "dino_f1": 57.01},
    "Ours (Wan2.1-1.3B)": {"text_clip": 28.434, "clip_i": 72.32, "dino_global": 46.02, "dino_match": 57.64, "dino_f1": 57.09},
    "4DNeX": {"text_clip": 22.844, "clip_i": 61.11, "dino_global": 11.31, "dino_match": 30.53, "dino_f1": 28.33},
    "Wan2.2-I2V-A14B + 4RC": {"text_clip": 26.340, "clip_i": 70.55, "dino_global": 47.83, "dino_match": 56.25, "dino_f1": 55.79},
    "Wan2.2-I2V-A14B + pi3": {"text_clip": 24.678, "clip_i": 66.65, "dino_global": 33.50, "dino_match": 45.08, "dino_f1": 43.82},
    "Wan2.2-I2V-A14B + Any4D": {"text_clip": 24.362, "clip_i": 65.30, "dino_global": 27.21, "dino_match": 43.54, "dino_f1": 42.17},
    "Ours (Wan2.2-I2V-A14B)": {"text_clip": 27.340, "clip_i": 72.87, "dino_global": 54.85, "dino_match": 61.85, "dino_f1": 61.60},
}

CLAIMED_HEADLINE_GAINS = {"text4d200_dino_f1": (2.88, 3.45), "i4d200_dino_f1": 5.81}


@dataclass
class ReconstructorAdapter:
    """Wraps an RGB-conditioned feed-forward reconstructor (4RC / pi3 / Any4D)."""

    name: str
    run: Callable[[torch.Tensor], dict[str, torch.Tensor]]
    module: Optional[torch.nn.Module] = None

    def __call__(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.run(rgb)


class GenerateThenReconstruct(torch.nn.Module):
    """Eq. (3): Y_hat_rgb = R(D_v(z_v)) — the cascade this paper argues against."""

    def __init__(self, vae_decode: Callable[[torch.Tensor], torch.Tensor], reconstructor: ReconstructorAdapter):
        super().__init__()
        self.vae_decode = vae_decode
        self.reconstructor = reconstructor

    @torch.no_grad()
    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        rgb = self.vae_decode(latent)
        out = self.reconstructor(rgb)
        out["decoded_rgb"] = rgb
        return out


def matched_cascade_pairs() -> list[tuple[str, str]]:
    """Same-latent controlled comparisons: (Ours, matched cascade) sharing one generated latent."""
    return [
        ("Ours (Wan2.1-14B)", "Wan2.1-14B + 4RC"),
        ("Ours (Wan2.1-1.3B)", "Wan2.1-1.3B + 4RC"),
        ("Ours (Wan2.2-I2V-A14B)", "Wan2.2-I2V-A14B + 4RC"),
    ]


def dino_f1_gains(report: dict[str, dict[str, float]]) -> dict[str, float]:
    gains = {}
    for ours, baseline in matched_cascade_pairs():
        if ours in report and baseline in report:
            gains[f"{ours} vs {baseline}"] = report[ours]["dino_f1"] - report[baseline]["dino_f1"]
    return gains


def registry_entry(name: str) -> dict[str, Any]:
    """Metadata for the benchmark/leaderboard tooling."""
    task = "image-to-4d" if "I2V" in name or name == "4DNeX" else "text-to-4d"
    kind = "ours" if name.startswith("Ours") else "cascade" if "+" in name else "native-feed-forward"
    return {"name": name, "task": task, "kind": kind, "reference": TABLE_1_REFERENCE[name]}
