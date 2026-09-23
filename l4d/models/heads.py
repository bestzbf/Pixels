"""Prediction heads of the 4D decoder D_omega (paper Eq. 6): multi-level geometry head + 9D camera head."""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLevelProjector(nn.Module):
    """Fuses the tapped frame++global features of several depths into one per-frame feature map."""

    def __init__(self, level_dim: int, out_dim: int, n_levels: int):
        super().__init__()
        self.n_levels = n_levels
        self.projections = nn.ModuleList([nn.Linear(level_dim, out_dim) for _ in range(n_levels)])
        self.fuse = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, padding=1), nn.GroupNorm(min(8, out_dim), out_dim), nn.GELU()
        )

    def forward(self, levels: Sequence[torch.Tensor], grid: tuple[int, int, int]) -> torch.Tensor:
        t, hp, wp = grid
        merged = None
        for index, project in enumerate(self.projections):
            level = levels[min(index, len(levels) - 1)]
            b, _, m, dim = level.shape
            feat = project(level).transpose(1, 2).reshape(b * t, -1, m)
            map_hw = feat.view(b * t, -1, hp, wp) if m == hp * wp else F.interpolate(
                feat, size=(hp, wp), mode="bilinear", align_corners=False
            )
            merged = map_hw if merged is None else merged + map_hw
        return self.fuse(merged)


class GeometryHead(nn.Module):
    """Predicts per-pixel depth and world-space rays (unit direction), each with a confidence value."""

    def __init__(
        self,
        token_dim: int,
        n_levels: int = 5,
        hidden: int = 256,
        ray_space: str = "world",
        log_conf_clamp: tuple[float, float] = (-8.0, 6.0),
    ):
        super().__init__()
        assert ray_space in {"world", "camera"}
        self.ray_space = ray_space
        self.conf_clamp = log_conf_clamp
        self.projector = MultiLevelProjector(token_dim * 2, hidden, n_levels)
        self.refine = nn.Sequential(nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(min(8, hidden), hidden), nn.GELU())
        self.depth_head = nn.Conv2d(hidden, 2, 1)
        self.ray_head = nn.Conv2d(hidden, 4, 1)
        with torch.no_grad():
            self.depth_head.bias[1] = 0.0
            self.ray_head.bias[3] = 0.0

    def forward(
        self,
        levels: Sequence[torch.Tensor],
        grid: tuple[int, int, int],
        output_size: Optional[tuple[int, int]] = None,
    ) -> dict[str, torch.Tensor]:
        t, hp, wp = grid
        feats = self.refine(self.projector(levels, grid))
        if output_size is not None and tuple(feats.shape[-2:]) != tuple(output_size):
            feats = F.interpolate(feats, size=tuple(output_size), mode="bilinear", align_corners=False)
        depth_raw = self.depth_head(feats)
        ray_raw = self.ray_head(feats)
        b_t = feats.shape[0]
        b = b_t // t
        spatial = tuple(depth_raw.shape[-2:])
        depth = F.softplus(depth_raw[:, 0]).view(b, t, *spatial)
        depth_log_conf = depth_raw[:, 1].clamp(*self.conf_clamp).view(b, t, *spatial)
        rays = F.normalize(ray_raw[:, :3], dim=1).permute(0, 2, 3, 1).reshape(b, t, *spatial, 3)
        ray_log_conf = ray_raw[:, 3].clamp(*self.conf_clamp).view(b, t, *spatial)
        return {
            "depth": depth,
            "depth_log_conf": depth_log_conf,
            "ray_dirs": rays,
            "ray_log_conf": ray_log_conf,
            "output_size": spatial,
        }


class CameraHead(nn.Module):
    """9D pose-FOV camera encoding per frame, from the per-frame camera-token states."""

    def __init__(self, token_dim: int, hidden: int = 512, encoding_dim: int = 9):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.out = nn.Linear(hidden, encoding_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, camera_tokens: torch.Tensor) -> torch.Tensor:
        return self.out(self.mlp(camera_tokens))
