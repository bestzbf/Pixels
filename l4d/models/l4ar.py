"""L4AR: Latent-to-4D Alignment and Refinement network (alignment -> refinement -> 4D decoding)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn

from .alignment import AlignmentModule
from .decoder import FourDDecoder
from .lora import lora_parameters
from .refinement import RefinementHierarchy


@dataclass
class VAESpec:
    """Latent convention of a video VAE; two DiTs are compatible only if all fields match."""

    name: str = "wan2.1-vae"
    checkpoint_id: str = "Wan-AI/Wan2.1-T2V-1.3B"
    channels: int = 16
    spatial_compression: int = 8
    temporal_compression: int = 4
    scaling: float = 1.0
    shift: float = 0.0
    latents_mean: tuple = ()
    latents_std: tuple = ()
    layout: str = "B,C,T,H,W"

    def channel_stats(self, device=None, dtype=None):
        """Per-channel mean/std as (1,C,1,1,1) tensors, or None when the spec is not channel-wise."""
        if len(self.latents_mean) != self.channels or len(self.latents_std) != self.channels:
            return None
        mean = torch.tensor(self.latents_mean, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
        std = torch.tensor(self.latents_std, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
        return mean, std

    def latent_shape_for(self, num_frames: int, height: int, width: int) -> tuple[int, int, int]:
        t = 1 + (num_frames - 1) // self.temporal_compression
        return (self.channels, t, height // self.spatial_compression, width // self.spatial_compression)


def vae_spec_from_dict(raw: dict[str, Any]) -> VAESpec:
    """`backend` selects a wrapper class, it is not part of the latent convention."""
    fields = set(VAESpec.__dataclass_fields__)
    unknown = set(raw) - fields - {"backend"}
    if unknown:
        raise KeyError(f"unknown vae config keys: {sorted(unknown)}")
    return VAESpec(**{k: v for k, v in raw.items() if k in fields})


@dataclass
class L4ARConfig:
    token_dim: int = 1024
    heads: int = 16
    depth: int = 31
    mlp_ratio: float = 4.0
    drop_path: float = 0.0
    patch_size: int = 2
    align_kernel: tuple[int, int, int] = (1, 2, 2)
    grid_mode: str = "trilinear"
    use_local_conv: bool = True
    tap_after: tuple[int, ...] = (6, 13, 20, 27, 30)
    initial_frame_blocks: int = 1
    scope_policy: str = "full"
    lora_rank: int = 16
    lora_alpha: Optional[float] = None
    geometry_hidden: int = 256
    camera_hidden: int = 512
    camera_encoding_dim: int = 9
    ray_space: str = "world"
    use_camera_tokens: bool = True
    use_time_tokens: bool = True
    vae: VAESpec = field(default_factory=VAESpec)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "L4ARConfig":
        kwargs = {k: (tuple(v) if isinstance(v, list) else v) for k, v in data.items() if k != "vae"}
        vae = vae_spec_from_dict(data.get("vae", {})) if isinstance(data.get("vae"), dict) else VAESpec()
        for key, value in list(kwargs.items()):
            if isinstance(value, list):
                kwargs[key] = tuple(value)
        known = {f for f in L4ARConfig.__dataclass_fields__} - {"vae"}
        unknown = set(kwargs) - known
        if unknown:
            raise KeyError(f"unknown L4AR config keys: {sorted(unknown)}")
        return L4ARConfig(vae=vae, **{k: v for k, v in kwargs.items() if k in known})


class L4AR(nn.Module):
    """Paper Eq. (4): Y_hat = D_omega(H_{psi,Delta psi}(A_phi(z); S))."""

    def __init__(self, cfg: L4ARConfig):
        super().__init__()
        self.cfg = cfg
        self.alignment = AlignmentModule(
            in_channels=cfg.vae.channels,
            token_dim=cfg.token_dim,
            patch_size=cfg.patch_size,
            kernel_size=cfg.align_kernel,
            grid_mode=cfg.grid_mode,
            use_local_conv=cfg.use_local_conv,
        )
        self.refinement = RefinementHierarchy(
            dim=cfg.token_dim,
            depth=cfg.depth,
            heads=cfg.heads,
            mlp_ratio=cfg.mlp_ratio,
            drop_path=cfg.drop_path,
            tap_after=cfg.tap_after,
            initial_frame_blocks=cfg.initial_frame_blocks,
            lora_rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            use_camera_tokens=cfg.use_camera_tokens,
            use_time_tokens=cfg.use_time_tokens,
        )
        self.refinement.set_scope_policy(cfg.scope_policy, cfg.initial_frame_blocks)
        self.decoder = FourDDecoder(
            token_dim=cfg.token_dim,
            n_levels=len(cfg.tap_after),
            geometry_hidden=cfg.geometry_hidden,
            camera_hidden=cfg.camera_hidden,
            camera_encoding_dim=cfg.camera_encoding_dim,
            ray_space=cfg.ray_space,
        )
        self.trainable_stage = 3
        self.freeze_pretrained()

    def grid_for_latent(self, latent: torch.Tensor) -> tuple[int, int, int]:
        """Token grid implied by the VAE latent and the shared compression conventions."""
        _, _, t_lat, h_lat, w_lat = latent.shape
        vae = self.cfg.vae
        frames = 1 + (t_lat - 1) * vae.temporal_compression
        p = self.cfg.patch_size
        return (frames, max(h_lat // p, 1), max(w_lat // p, 1))

    def output_size_for_latent(self, latent: torch.Tensor) -> tuple[int, int]:
        vae = self.cfg.vae
        return (latent.shape[3] * vae.spatial_compression, latent.shape[4] * vae.spatial_compression)

    def forward(
        self,
        latent: torch.Tensor,
        grid: Optional[tuple[int, int, int]] = None,
        output_size: Optional[tuple[int, int]] = None,
    ) -> dict[str, torch.Tensor]:
        if latent.dim() != 5:
            raise ValueError(f"expected latent (B,C,T,H,W), got {tuple(latent.shape)}")
        grid = self.grid_for_latent(latent) if grid is None else grid
        output_size = self.output_size_for_latent(latent) if output_size is None else output_size
        tokens, used_grid = self.alignment(latent, grid)
        refined = self.refinement(tokens, used_grid)
        out = self.decoder(refined, output_size)
        out["token_grid"] = torch.tensor(used_grid)
        return out

    # ---- freezing / staged training -------------------------------------------
    def freeze_pretrained(self) -> None:
        """Video models, VAE, original transformer weights, tokens and unused heads stay frozen."""
        for param in self.parameters():
            param.requires_grad_(False)
        self.refinement.tokens.freeze()
        for param in self.alignment.parameters():
            param.requires_grad_(True)
        for param in self.decoder.parameters():
            param.requires_grad_(True)
        for param in lora_parameters(self.refinement):
            param.requires_grad_(True)

    def set_stage(self, stage: int) -> list[str]:
        """Progressively activate trainable components so the pretrained geometric prior is not destabilised."""
        if stage not in {1, 2, 3}:
            raise ValueError("stage must be 1 (alignment), 2 (+heads) or 3 (+LoRA refinement)")
        self.freeze_pretrained()
        for param in self.parameters():
            param.requires_grad_(False)
        for param in self.alignment.parameters():
            param.requires_grad_(True)
        active = ["alignment"]
        if stage >= 2:
            for param in self.decoder.parameters():
                param.requires_grad_(True)
            active.append("decoder_heads")
        if stage >= 3:
            for param in lora_parameters(self.refinement):
                param.requires_grad_(True)
            active.append("refinement_lora")
        self.trainable_stage = stage
        return active

    def trainable_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        return {
            "alignment": [p for n, p in self.alignment.named_parameters() if p.requires_grad],
            "decoder": [p for n, p in self.decoder.named_parameters() if p.requires_grad],
            "refinement_lora": [p for p in lora_parameters(self.refinement) if p.requires_grad],
            "frozen": [p for p in self.parameters() if not p.requires_grad],
        }

    def count_parameters(self) -> dict[str, int]:
        groups = self.trainable_parameter_groups()
        total = sum(p.numel() for p in self.parameters())
        return {
            "total": total,
            "alignment": sum(p.numel() for p in groups["alignment"]),
            "decoder": sum(p.numel() for p in groups["decoder"]),
            "refinement_lora": sum(p.numel() for p in groups["refinement_lora"]),
            "frozen": sum(p.numel() for p in groups["frozen"]),
        }


def build_l4ar(config: dict[str, Any] | L4ARConfig) -> L4AR:
    cfg = config if isinstance(config, L4ARConfig) else L4ARConfig.from_dict(config)
    return L4AR(cfg)


def load_4rc_init(model: "L4AR", checkpoint: str, strict: bool = False) -> dict[str, Any]:
    """Initialise the refinement hierarchy from a 4RC checkpoint.

    4RC's backbone is a CroCo/DINOv2-style stack (`blocks.N.attn.qkv`, `mlp.fc1`, `norm1`...), so the
    same key normalisation used for a plain ViT applies, including looking through our LoRA wrappers to
    reach `base.weight`. Its DualDPT / CameraDec internals differ in shape from our heads, so head
    tensors are reported as unmapped rather than silently forced.
    """
    from .init_from import load_vit_into_refinement

    report = load_vit_into_refinement(model.refinement, checkpoint, max_blocks=model.cfg.depth)
    report["unmapped"] = "heads (DualDPT/CameraDec layouts differ); align manually if transferring them"
    if strict:
        raise ValueError("strict=True is unsupported for foreign head layouts: only backbone blocks map cleanly")
    return report
