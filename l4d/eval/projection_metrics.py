"""Projection-based generation metrics: two off-axis views, DINO global / valid-patch match / set F1, CLIP."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from ..utils.geometry import look_at_camera


@dataclass
class OffAxisProtocol:
    """Paper renders each predicted point sequence from two off-axis cameras."""

    elevations: tuple[float, ...] = (0.35, -0.15)
    azimuths: tuple[float, ...] = (0.6, -0.6)
    distance_scale: float = 2.2
    fovy: float = 0.9
    image_size: int = 336
    patch: int = 14


class FeatureExtractor:
    """Interface: returns (global_features, patch_features) for a batch of RGB images (B,3,H,W)."""

    name = "surrogate"
    stride = 14

    def __call__(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class SurrogateExtractor(FeatureExtractor):
    """Fixed random projections, used when DINOv2/CLIP weights are unavailable (shape/protocol tests only)."""

    def __init__(self, dim: int = 384, stride: int = 14, seed: int = 0, device: str = "cpu"):
        torch.manual_seed(seed)
        self.stride = stride
        self.proj = torch.nn.Conv2d(3, dim, stride, stride).to(device).eval()
        for param in self.proj.parameters():
            param.requires_grad_(False)
        self.name = f"surrogate-{dim}"

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patches = self.proj(images)
        globals_ = patches.mean(dim=(-2, -1))
        return F.normalize(globals_, dim=-1), F.normalize(patches.flatten(2).transpose(1, 2), dim=-1)


class DinoV2Extractor(FeatureExtractor):
    """Real DINOv2 backbone (torch.hub), loaded lazily."""

    def __init__(self, variant: str = "dinov2_vitb14", device: str = "cuda"):
        self.variant = variant
        self.stride = 14
        self.device = device
        self._model: Optional[torch.nn.Module] = None
        self.name = variant

    def load(self) -> torch.nn.Module:
        if self._model is None:
            self._model = torch.hub.load("facebookresearch/dinov2", self.variant).to(self.device).eval()
        return self._model

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.load()
        out = model.forward_features(images.to(next(model.parameters()).device))
        return F.normalize(out["x_norm_clstoken"], dim=-1), F.normalize(out["x_norm_patchtokens"], dim=-1)


def render_off_axis_views(
    points: torch.Tensor,
    colors: Optional[torch.Tensor],
    protocol: OffAxisProtocol,
    frame: int = 0,
) -> list[torch.Tensor]:
    """Projects frame `frame` of a (T,H,W,3) point map into each off-axis view; returns (3,S,S) images."""
    from ..utils.geometry import render_point_map  # local import: renderer is optional-weight

    target = points[frame].reshape(-1, 3).mean(0)
    extent = float((points[frame] - target).abs().max()) + 1e-6
    views = []
    for elevation, azimuth in zip(protocol.elevations, protocol.azimuths):
        _, encoding = look_at_camera(
            target,
            elevation,
            azimuth,
            distance=extent * protocol.distance_scale,
            fovy=protocol.fovy,
            aspect=1.0,
            device=points.device,
        )
        intrinsics = torch.eye(3, device=points.device)
        intrinsics[0, 2] = intrinsics[1, 2] = protocol.image_size / 2.0
        focal = protocol.image_size / (2.0 * torch.tan(torch.tensor(protocol.fovy / 2.0, device=points.device)))
        intrinsics[0, 0] = intrinsics[1, 1] = focal
        flat_points = points[frame].reshape(-1, 3)
        flat_colors = None if colors is None else colors[frame].reshape(-1, 3)
        image = render_point_map(flat_points, flat_colors, encoding, intrinsics, protocol.image_size)
        image = image.permute(2, 0, 1)
        views.append(image.expand(3, -1, -1).contiguous() if image.shape[0] == 1 else image.contiguous())
    return views


def dino_global(view: torch.Tensor, reference: torch.Tensor, extractor: FeatureExtractor) -> float:
    features_pred, _ = extractor(view.unsqueeze(0))
    features_ref, _ = extractor(reference.unsqueeze(0))
    return float(F.cosine_similarity(features_pred, features_ref, dim=-1).mean()) * 100.0


def render_validity_mask(view: torch.Tensor, protocol: "OffAxisProtocol") -> torch.Tensor:
    """Empty background pixels are rendered black, so non-black pixels mark valid projections.

    view: (3,S,S) float image in [0,1].
    """
    occupancy = (view.abs().sum(0) > 0).float()
    patches = F.avg_pool2d(occupancy[None, None], protocol.patch, protocol.patch).flatten()
    return patches > 0.25


def dino_match(
    view: torch.Tensor, reference: torch.Tensor, extractor: FeatureExtractor, valid: Optional[torch.Tensor] = None
) -> float:
    """Valid-patch DINO matching: mean best-match cosine over non-empty predicted patches."""
    _, patches_pred = extractor(view.unsqueeze(0))
    _, patches_ref = extractor(reference.unsqueeze(0))
    best = (patches_pred @ patches_ref.transpose(1, 2)).max(dim=-1).values
    if valid is not None:
        keep = valid.reshape(-1)
        best = best.reshape(-1)[keep[: best.numel()]]
    if best.numel() == 0:
        return 0.0
    return float(best.mean()) * 100.0


def dino_set_f1(
    view: torch.Tensor, reference: torch.Tensor, extractor: FeatureExtractor, threshold: float = 0.65
) -> dict[str, float]:
    """Set-based F1 over patch features: a predicted patch counts as a true positive when its nearest
    reference patch exceeds the similarity threshold (and vice versa for recall)."""
    _, patches_pred = extractor(view.unsqueeze(0))
    _, patches_ref = extractor(reference.unsqueeze(0))
    forward = (patches_pred @ patches_ref.transpose(1, 2)).max(dim=-1).values
    backward = (patches_ref @ patches_pred.transpose(1, 2)).max(dim=-1).values
    precision = float((forward > threshold).float().mean())
    recall = float((backward > threshold).float().mean())
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": 100 * precision, "recall": 100 * recall, "f1": 100 * f1}


@dataclass
class ProjectionReport:
    text_clip: float = 0.0
    clip_i: float = 0.0
    dino_global: float = 0.0
    dino_match: float = 0.0
    dino_f1: float = 0.0
    per_case: list[dict[str, float]] = field(default_factory=list)

    def as_row(self) -> dict[str, float]:
        return {
            "Text CLIP": self.text_clip,
            "CLIP-I": self.clip_i,
            "DINO global": self.dino_global,
            "DINO match": self.dino_match,
            "DINO F1": self.dino_f1,
        }


class ProjectionEvaluator:
    """Aggregates the Table 1 metrics over a benchmark; RGB is used only for colouring/references."""

    def __init__(
        self,
        extractor: FeatureExtractor,
        clip_text_scorer=None,
        clip_image_scorer=None,
        protocol: OffAxisProtocol = OffAxisProtocol(),
    ):
        self.extractor = extractor
        self.clip_text_scorer = clip_text_scorer
        self.clip_image_scorer = clip_image_scorer
        self.protocol = protocol

    @torch.no_grad()
    def evaluate_case(
        self,
        points: torch.Tensor,
        colors: Optional[torch.Tensor],
        reference_rgb: Optional[torch.Tensor],
        text: Optional[str] = None,
        frame: int = 0,
    ) -> dict[str, float]:
        views = render_off_axis_views(points, colors, self.protocol, frame=frame)
        score = {"text_clip": 0.0, "clip_i": 0.0, "dino_global": 0.0, "dino_match": 0.0, "dino_f1": 0.0}
        if reference_rgb is None:
            return score
        for view in views:
            validity = render_validity_mask(view, self.protocol)
            score["dino_global"] += dino_global(view, reference_rgb, self.extractor)
            score["dino_match"] += dino_match(view, reference_rgb, self.extractor, validity)
            score["dino_f1"] += dino_set_f1(view, reference_rgb, self.extractor)["f1"]
            if text is not None and self.clip_text_scorer is not None:
                score["text_clip"] += float(self.clip_text_scorer(text, view))
            if self.clip_image_scorer is not None:
                score["clip_i"] += float(self.clip_image_scorer(view, reference_rgb))
        count = float(len(views))
        return {key: value / count for key, value in score.items()}

    def aggregate(self, case_scores: list[dict[str, float]]) -> ProjectionReport:
        keys = ("text_clip", "clip_i", "dino_global", "dino_match", "dino_f1")
        means = {key: sum(case[key] for case in case_scores) / max(len(case_scores), 1) for key in keys}
        return ProjectionReport(**means, per_case=case_scores)
