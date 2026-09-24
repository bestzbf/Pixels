"""Ground-truth benchmarks (7-Scenes / NRGBD): accuracy, completeness, normal consistency, drift."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..utils.geometry import surface_normals


@dataclass
class GTBenchmark:
    name: str
    sequences: int
    scale: float = 100.0  # metres -> centimetres


SEVEN_SCENES = GTBenchmark("7scenes", sequences=18)
NRGBD = GTBenchmark("nrgbd", sequences=9)


def _sample_indices(total: int, max_points: int, seed: int) -> torch.Tensor:
    """Chamfer on 1e5-point clouds is quadratic; a fixed seeded subsample is the standard estimate."""
    if max_points <= 0 or total <= max_points:
        return torch.arange(total)
    return torch.randperm(total, generator=torch.Generator().manual_seed(seed))[:max_points]


def _nn_distances(query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Chunked nearest-neighbour distance from every query point to the reference cloud."""
    chunk = max(1, int(4e6 / max(reference.shape[0], 1)))
    out = []
    for start in range(0, query.shape[0], chunk):
        block = query[start : start + chunk]
        dist = torch.cdist(block, reference)
        out.append(dist.min(dim=-1).values)
    return torch.cat(out)


def accuracy_completeness(
    pred: torch.Tensor, gt: torch.Tensor, threshold: float = 5.0, normal_pred: torch.Tensor | None = None,
    normal_gt: torch.Tensor | None = None, angle_threshold: float = 30.0, max_points: int = 20000,
) -> dict[str, float]:
    """Acc/Comp in the same length unit as the inputs; NC = normal consistency under the angle gate.

    Both clouds are subsampled once, up front, so the normal lookup uses the very points that were scored.
    """
    pred_all, gt_all = pred.reshape(-1, 3), gt.reshape(-1, 3)
    pred_idx = _sample_indices(pred_all.shape[0], max_points, 0)
    gt_idx = _sample_indices(gt_all.shape[0], max_points, 1)
    pred, gt = pred_all[pred_idx], gt_all[gt_idx]
    forward = _nn_distances(pred, gt)
    backward = _nn_distances(gt, pred)
    accuracy = forward[forward < threshold]
    completeness = backward[backward < threshold]
    result = {
        "accuracy": float(accuracy.mean()) if accuracy.numel() else float("nan"),
        "completeness": float(completeness.mean()) if completeness.numel() else float("nan"),
    }
    if normal_pred is not None and normal_gt is not None:
        nearest = torch.cdist(pred, gt).argmin(dim=-1)
        normals_pred = normal_pred.reshape(-1, 3)[pred_idx]
        normals_gt = normal_gt.reshape(-1, 3)[gt_idx]
        cosine = (normals_pred * normals_gt[nearest]).sum(-1).abs()
        gated = cosine[forward < threshold]
        result["normal_consistency"] = float((gated > torch.cos(torch.tensor(angle_threshold * 3.14159265 / 180.0))).float().mean()) if gated.numel() else 0.0
    return result


def evaluate_prediction(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    benchmark: GTBenchmark,
    pred_normals: torch.Tensor | None = None,
    gt_normals: torch.Tensor | None = None,
    threshold_cm: float = 5.0,
) -> dict[str, float]:
    """Aligns scale, then reports Acc / Comp (cm) and NC for one sequence."""
    pred_points = pred_points.detach().cpu().float()
    gt_points = gt_points.detach().cpu().float()
    if pred_normals is not None:
        pred_normals = pred_normals.detach().cpu().float()
    if gt_normals is not None:
        gt_normals = gt_normals.detach().cpu().float()
    scale = (gt_points.norm(dim=-1).mean() / pred_points.norm(dim=-1).mean().clamp(min=1e-8)).clamp(1e-3, 1e3)
    aligned = pred_points * scale
    if pred_normals is None:
        pred_normals = surface_normals(aligned.reshape(gt_points.shape))
    if gt_normals is None:
        gt_normals = surface_normals(gt_points.reshape(gt_points.shape))
    scores = accuracy_completeness(
        aligned * benchmark.scale, gt_points * benchmark.scale, threshold_cm,
        pred_normals.reshape(-1, 3), gt_normals.reshape(-1, 3),
    )
    return {
        "Acc": scores["accuracy"],
        "Comp": scores["completeness"],
        "NC": scores.get("normal_consistency", float("nan")),
        "scale_alignment": float(scale),
    }


def point_map_drift(pred_before: torch.Tensor, pred_after: torch.Tensor, valid: torch.Tensor | None = None) -> float:
    """Mean displacement caused by a latent perturbation (Fig. 6 diagnostic)."""
    error = (pred_after - pred_before).norm(dim=-1)
    if valid is not None:
        error = error[valid]
    return float(error.mean())


def camera_drift(center_before: torch.Tensor, center_after: torch.Tensor) -> float:
    return float((center_after - center_before).norm(dim=-1).mean())
