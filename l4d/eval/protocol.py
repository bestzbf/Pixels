"""Table 3 ablation variants and Table 2 human-evaluation statistics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

#: Each variant maps to L4ARConfig overrides (see docs/PAPER_ANALYSIS.md section 5 for the mapping).
TABLE_3_VARIANTS: dict[str, dict[str, Any]] = {
    "w/o Grid": {"grid_mode": "none"},
    "w/o 3D Conv": {"use_local_conv": False},
    "w/o Frame": {"scope_policy": "global_only"},
    "w/o Global": {"scope_policy": "frame_only"},
    "Full": {},
}

#: Paper Table 3 (Acc/Comp in cm, NC dimensionless).
TABLE_3_REFERENCE: dict[str, dict[str, dict[str, float]]] = {
    "w/o Grid": {"7scenes": {"Acc": 3.783, "Comp": 6.844, "NC": 0.608}, "nrgbd": {"Acc": 5.823, "Comp": 9.686, "NC": 0.726}},
    "w/o 3D Conv": {"7scenes": {"Acc": 6.944, "Comp": 15.806, "NC": 0.554}, "nrgbd": {"Acc": 12.439, "Comp": 26.511, "NC": 0.594}},
    "w/o Frame": {"7scenes": {"Acc": 6.688, "Comp": 20.806, "NC": 0.513}, "nrgbd": {"Acc": 12.367, "Comp": 36.982, "NC": 0.502}},
    "w/o Global": {"7scenes": {"Acc": 6.754, "Comp": 19.742, "NC": 0.559}, "nrgbd": {"Acc": 13.818, "Comp": 36.207, "NC": 0.515}},
    "Full": {"7scenes": {"Acc": 3.121, "Comp": 5.418, "NC": 0.628}, "nrgbd": {"Acc": 5.202, "Comp": 8.187, "NC": 0.766}},
}

#: Paper Table 2: case-averaged preference for Ours (%) with 95% bootstrap intervals.
TABLE_2_REFERENCE: dict[str, dict[str, tuple[float, float, float]]] = {
    "text-to-4d": {
        "condition_fidelity": (59.2, 54.1, 64.3),
        "geometry_completeness": (66.8, 62.0, 71.5),
        "temporal_stability": (63.5, 58.4, 68.5),
        "overall_quality": (65.7, 60.8, 70.5),
    },
    "image-to-4d": {
        "condition_fidelity": (66.4, 61.8, 70.9),
        "geometry_completeness": (72.1, 67.8, 76.3),
        "temporal_stability": (68.3, 63.7, 72.8),
        "overall_quality": (70.6, 66.2, 74.9),
    },
}

HUMAN_EVAL_PROTOCOL = {
    "participants": 50,
    "cases_per_benchmark": 50,
    "ratings_per_case": 10,
    "design": "randomized anonymized pairwise comparison, multi-view replay allowed",
    "axes": ("condition fidelity", "geometry and completeness", "temporal stability", "overall quality"),
}


def preference_rate(wins: torch.Tensor, cases: torch.Tensor) -> float:
    """Case-averaged preference: mean over cases of the fraction of ratings that preferred Ours."""
    per_case = []
    for case in cases.unique():
        pick = cases == case
        if pick.any():
            per_case.append(wins[pick].float().mean())
    return float(torch.stack(per_case).mean()) * 100.0 if per_case else float("nan")


def bootstrap_interval(
    values: torch.Tensor, statistic: Callable[[torch.Tensor], float], samples: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    generator = torch.Generator().manual_seed(seed)
    draws = torch.randint(0, values.numel(), (samples, values.numel()), generator=generator)
    stats = torch.tensor([statistic(values[draw]) for draw in draws])
    return float(statistic(values)), float(stats.quantile(0.025)), float(stats.quantile(0.975))


@dataclass
class AblationRun:
    variant: str
    overrides: dict[str, Any]
    dataset: str
    metrics: dict[str, float]

    def delta_vs_full(self, full: dict[str, float]) -> dict[str, float]:
        return {key: self.metrics[key] - full[key] for key in self.metrics if key in full}
