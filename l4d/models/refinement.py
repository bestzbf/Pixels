"""Spatiotemporal refinement hierarchy H_{psi,Delta psi}: blocks alternating frame-wise / global attention."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import attach_lora

FRAME = "frame"
GLOBAL = "global"


def sinusoidal_token_positions(t: int, h: int, w: int, dim: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Parameter-free (T*H*W, dim) embedding: separable sinusoids over time and the two spatial axes."""
    per_axis = (dim // 3) + 1
    per_axis -= per_axis % 2
    halves = []
    for size in (t, h, w):
        pos = torch.arange(size, device=device, dtype=torch.float32)
        omega = 1.0 / (10000 ** (torch.arange(per_axis // 2, device=device, dtype=torch.float32) / max(per_axis // 2, 1)))
        angles = torch.outer(pos, omega)
        halves.append(torch.cat([angles.sin(), angles.cos()], -1))
    tt, th, tw = halves
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(t, device=device), torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
        ),
        -1,
    ).reshape(-1, 3)
    emb = torch.cat([tt[grid[:, 0]], th[grid[:, 1]], tw[grid[:, 2]]], -1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb[:, :dim].to(dtype)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 16, qkv_bias: bool = True):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} not divisible by heads {heads}")
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d)
        return self.proj(out)


class Mlp(nn.Module):
    def __init__(self, dim: int, ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, scope: str, mlp_ratio: float = 4.0, drop_path: float = 0.0):
        super().__init__()
        self.scope = scope
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio)
        self.drop_path = nn.Identity() if drop_path <= 0 else nn.Dropout(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class TokenBank(nn.Module):
    """Frozen camera tokens (per frame) and time tokens (per frame index), reused from the 4RC hierarchy."""

    def __init__(self, dim: int, max_frames: int = 64, n_camera_tokens: int = 1):
        super().__init__()
        self.camera = nn.Parameter(torch.zeros(1, 1, n_camera_tokens, dim))
        self.time = nn.Parameter(torch.zeros(1, max_frames, 1, dim))
        nn.init.trunc_normal_(self.camera, std=0.02)
        nn.init.trunc_normal_(self.time, std=0.02)

    def freeze(self) -> "TokenBank":
        self.camera.requires_grad_(False)
        self.time.requires_grad_(False)
        return self


@dataclass
class RefinementOutput:
    levels: list[torch.Tensor]  # each (B, T, M, 2*dim): recent frame-wise ++ recent global features
    fused: torch.Tensor  # (B, T, M, 2*dim)
    camera_tokens: torch.Tensor  # (B, T, dim)
    grid: tuple[int, int, int]


class RefinementHierarchy(nn.Module):
    """H_{psi,Delta psi}: frozen pretrained blocks psi plus trainable LoRA updates Delta psi (rank 16)."""

    def __init__(
        self,
        dim: int = 1024,
        depth: int = 31,
        heads: int = 16,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        tap_after: tuple[int, ...] = (6, 13, 20, 27, 30),
        initial_frame_blocks: int = 1,
        lora_rank: int = 16,
        lora_alpha: Optional[float] = None,
        lora_targets: tuple[str, ...] = ("qkv", "proj", "fc1", "fc2"),
        use_camera_tokens: bool = True,
        use_time_tokens: bool = True,
        n_camera_tokens: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.initial_frame_blocks = initial_frame_blocks
        self.tap_after = tuple(sorted({int(i) for i in tap_after}))
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim,
                    heads,
                    self.scope_for(i, depth, initial_frame_blocks, "full"),
                    mlp_ratio,
                    drop_path * i / max(depth - 1, 1),
                )
                for i in range(depth)
            ]
        )
        self.tokens = TokenBank(dim, n_camera_tokens=n_camera_tokens)
        self.use_camera_tokens = use_camera_tokens
        self.use_time_tokens = use_time_tokens
        self.n_camera_tokens = n_camera_tokens
        if lora_rank > 0:
            attach_lora(self, rank=lora_rank, alpha=lora_alpha, target_modules=lora_targets)

    @staticmethod
    def scope_for(index: int, depth: int, initial_frame_blocks: int, policy: str) -> str:
        if policy == "global_only":  # ablation "w/o Frame"
            return GLOBAL
        if policy == "frame_only":  # ablation "w/o Global"
            return FRAME
        if index < initial_frame_blocks:
            return FRAME
        return GLOBAL if (index - initial_frame_blocks) % 2 == 0 else FRAME

    def set_scope_policy(self, policy: str = "full", initial_frame_blocks: Optional[int] = None) -> None:
        if initial_frame_blocks is not None:
            self.initial_frame_blocks = initial_frame_blocks
        for i, block in enumerate(self.blocks):
            block.scope = self.scope_for(i, self.depth, self.initial_frame_blocks, policy)

    def _run_scope(
        self,
        block: Block,
        patches: torch.Tensor,
        camera: Optional[torch.Tensor],
        time: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """patches: (B,T,M,d); camera: (B,T,Nc,d); time: (B,T,1,d)."""
        b, t, m, d = patches.shape
        if block.scope == FRAME:
            parts = [camera] if camera is not None else []
            parts.append(patches)
            seq = torch.cat(parts, dim=2)
            n_cam = 0 if camera is None else camera.shape[2]
            out = block(seq.reshape(b * t, -1, d)).reshape(b, t, n_cam + m, d)
            if n_cam:
                return out[:, :, n_cam:], out[:, :, :n_cam]
            return out, None
        parts = []
        if camera is not None:
            parts.append(camera)
        if time is not None:
            parts.append(time)
        parts.append(patches)
        prefix = sum(p.shape[2] for p in parts[:-1])
        seq = torch.cat(parts, dim=2).reshape(b, t * (prefix + m), d)
        out = block(seq).reshape(b, t, prefix + m, d)
        return out[:, :, prefix:], out[:, :, : camera.shape[2]] if camera is not None else None

    def forward(self, x: torch.Tensor, grid: Optional[tuple[int, int, int]] = None) -> RefinementOutput:
        b, t, m, d = x.shape
        if grid is None:
            side = int(round(math.sqrt(m)))
            grid = (t, side, m // max(side, 1))
        x = x + sinusoidal_token_positions(*grid, d, device=x.device, dtype=x.dtype).view(1, t, m, d)
        camera = self.tokens.camera.expand(b, t, -1, d) if self.use_camera_tokens else None
        time = self.tokens.time[:, :t].expand(b, -1, -1, d) if self.use_time_tokens else None
        stream = last_frame = last_global = x
        levels: list[torch.Tensor] = []
        taps = set(self.tap_after)
        for i, block in enumerate(self.blocks):
            stream, camera = self._run_scope(block, stream, camera, time)
            if block.scope == FRAME:
                last_frame = stream
            else:
                last_global = stream
            if i in taps:
                levels.append(torch.cat([last_frame, last_global], dim=-1))
        if not levels:
            levels.append(torch.cat([last_frame, last_global], dim=-1))
        fused = levels[-1]
        camera_out = camera.reshape(b, t, -1)[..., :d] if camera is not None else stream.mean(dim=2)
        return RefinementOutput(levels=levels, fused=fused, camera_tokens=camera_out, grid=grid)
