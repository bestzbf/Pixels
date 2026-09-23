"""Alignment module A_phi (paper Eq. 5): fixed grid resampling R, then learned 3D conv S_phi."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GridResample(nn.Module):
    """R: parameter-free trilinear resampling of the VAE latent onto the 4D token grid."""

    def __init__(self, mode: str = "trilinear"):
        super().__init__()
        if mode not in {"trilinear", "nearest", "none"}:
            raise ValueError(f"unknown grid mode: {mode}")
        self.mode = mode

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    def forward(self, z: torch.Tensor, target: tuple[int, int, int]) -> torch.Tensor:
        if not self.enabled:
            return z
        size = tuple(int(s) for s in target)
        if tuple(z.shape[2:5]) == size:
            return z
        return F.interpolate(z.float(), size=size, mode=self.mode, align_corners=False).to(z.dtype)


class LocalSpatiotemporalConv(nn.Module):
    """S_phi: Conv3d aggregating local spatiotemporal neighborhoods and projecting to token dim d."""

    def __init__(
        self,
        in_channels: int,
        token_dim: int,
        patch_size: int = 2,
        kernel_size: tuple[int, int, int] = (1, 2, 2),
        enabled: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        if enabled:
            kernel = tuple(int(k) for k in kernel_size)
        else:
            kernel = (1, 1, 1)
        self.kernel_size = kernel
        self.conv = nn.Conv3d(in_channels, token_dim, kernel_size=kernel, stride=(1, patch_size, patch_size))

    def required_input_grid(self, grid: tuple[int, int, int]) -> tuple[int, int, int]:
        t, hp, wp = grid
        kt, kh, kw = self.kernel_size
        p = self.patch_size
        return (t + kt - 1, hp * p + kh - p, wp * p + kw - p)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.conv(z)


class AlignmentModule(nn.Module):
    """Maps a Wan-style VAE latent (B, Cz, Tz, Hz, Wz) to tokens Q0 in R^{T x M x d} (Eq. 5)."""

    def __init__(
        self,
        in_channels: int,
        token_dim: int,
        patch_size: int = 2,
        kernel_size: tuple[int, int, int] = (1, 2, 2),
        grid_mode: str = "trilinear",
        use_local_conv: bool = True,
    ):
        super().__init__()
        self.grid = GridResample(grid_mode)
        self.local = LocalSpatiotemporalConv(
            in_channels, token_dim, patch_size=patch_size, kernel_size=kernel_size, enabled=use_local_conv
        )
        self.patch_size = patch_size
        self.token_dim = token_dim

    def native_grid(self, z: torch.Tensor) -> tuple[int, int, int]:
        t, h, w = z.shape[2], z.shape[3], z.shape[4]
        p = 1 if not self.grid.enabled else self.patch_size
        return (t, max(h // p, 1), max(w // p, 1))

    def forward(self, z: torch.Tensor, grid: Optional[tuple[int, int, int]] = None):
        """Returns (tokens, (T, Hp, Wp)). `grid` is the requested 4D token grid."""
        if not self.grid.enabled:
            # w/o Grid: the latent keeps its native spatiotemporal grid, so the token count that the
            # 4D hierarchy receives is whatever the conv arithmetic yields (this is the ablation).
            tokens = self.local(z)
            b, d, t_out, hp_out, wp_out = tokens.shape
            return tokens.permute(0, 2, 3, 4, 1).reshape(b, t_out, hp_out * wp_out, d), (t_out, hp_out, wp_out)
        if grid is None:
            grid = self.native_grid(z)
        t, hp, wp = grid
        target = self.local.required_input_grid((t, hp, wp))
        z_aligned = self.grid(z, target)
        tokens = self.local(z_aligned)
        b, d, t_out, hp_out, wp_out = tokens.shape
        assert (t_out, hp_out, wp_out) == (t, hp, wp), (
            f"alignment produced grid {(t_out, hp_out, wp_out)} but expected {(t, hp, wp)}"
        )
        return tokens.permute(0, 2, 3, 4, 1).reshape(b, t_out, hp_out * wp_out, d), (t_out, hp_out, wp_out)
