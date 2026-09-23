"""Shared-latent interface: frozen video VAE / DiT wrappers and the compatibility check.

Training consumes z_obs = mu(E_v(V)); generation consumes z_gen = G_theta(c, eps), the final
denoised latent immediately before VAE decoding. Both live in the same VAE space (paper Sec. 3.2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .l4ar import VAESpec


@dataclass
class LatentProvenance:
    source: str  # "observed_vae" or "generated_dit"
    generator: str
    vae: VAESpec
    steps: Optional[int] = None
    trajectory: Optional[list[torch.Tensor]] = None

    def __post_init__(self) -> None:
        if self.source not in {"observed_vae", "generated_dit"}:
            raise ValueError(self.source)


class SharedLatentInterface(nn.Module):
    """Validates that a latent may be consumed by L4AR, regardless of which DiT produced it."""

    def __init__(self, vae: VAESpec, supported_frames: Optional[tuple[int, ...]] = None):
        super().__init__()
        self.vae = vae
        self.supported_frames = supported_frames
        self.register_buffer("_dummy", torch.zeros(1), persistent=False)

    def check_compatible(self, provenance: LatentProvenance, latent: torch.Tensor) -> None:
        if provenance.vae != self.vae:
            raise ValueError(
                f"DiT {provenance.generator} uses {provenance.vae.name} "
                f"(ckpt={provenance.vae.checkpoint_id}, scale={provenance.vae.scaling}, "
                f"layout={provenance.vae.layout}); interface expects {self.vae.name}"
            )
        if latent.shape[1] != self.vae.channels:
            raise ValueError(f"latent channels {latent.shape[1]} != expected {self.vae.channels}")
        if self.supported_frames is not None and latent.shape[2] not in self.supported_frames:
            raise ValueError(f"latent temporal length {latent.shape[2]} not in supported {self.supported_frames}")

    def normalize(self, latent: torch.Tensor) -> torch.Tensor:
        """Apply the shared VAE scaling/shift convention before the alignment module."""
        return (latent - self.vae.shift) * self.vae.scaling


class VideoVAEBackend(nn.Module):
    """Encodes observed video into the posterior-mean latent z_obs with a frozen VAE."""

    def __init__(self, spec: VAESpec):
        super().__init__()
        self.spec = spec

    def posterior_mean(self, video: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.posterior_mean(video)


class SyntheticVideoVAE(VideoVAEBackend):
    """Deterministic stand-in used for offline smoke tests and CI (no Wan weights required)."""

    def __init__(self, spec: VAESpec, hidden: int = 32):
        super().__init__(spec)
        self.encoder = nn.Sequential(
            nn.Conv3d(3, hidden, (spec.temporal_compression, spec.spatial_compression, spec.spatial_compression),
                      stride=(spec.temporal_compression, spec.spatial_compression, spec.spatial_compression)),
            nn.GELU(),
            nn.Conv3d(hidden, spec.channels, 1),
        )
        for p in self.parameters():
            p.requires_grad_(False)

    def _ensure_decoder(self, latent: torch.Tensor) -> nn.Module:
        if not hasattr(self, "_decoder"):
            kernel = (self.spec.temporal_compression, self.spec.spatial_compression, self.spec.spatial_compression)
            self._decoder = nn.ConvTranspose3d(self.spec.channels, 3, kernel, stride=kernel).to(
                latent.device, latent.dtype
            )
            for param in self._decoder.parameters():
                param.requires_grad_(False)
        return self._decoder

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Reconstruction of the RGB observation; used by the generate-then-reconstruct baselines."""
        return self._ensure_decoder(latent)(latent).clamp(-1.0, 1.0)

    def posterior_mean(self, video: torch.Tensor) -> torch.Tensor:
        """Latent length follows the Wan convention Tz = 1 + (T - 1) / kt (T - 1 divisible by kt)."""
        kt = self.spec.temporal_compression
        first, rest = video[:, :, :1], video[:, :, 1:]
        padded = torch.cat([first.repeat(1, 1, kt - 1, 1, 1), rest], dim=2)
        tail = (kt - (padded.shape[2] % kt)) % kt
        if tail:
            padded = F.pad(padded, (0, 0, 0, 0, 0, tail), mode="replicate")
        return self.encoder(padded)


class WanVideoVAE(VideoVAEBackend):
    """Frozen Wan VAE wrapper (diffusers `AutoencoderKLWan`), loaded lazily."""

    def __init__(self, spec: VAESpec, pretrained: str = "Wan-AI/Wan2.1-T2V-1.3B"):
        super().__init__(spec)
        self.pretrained = pretrained
        self._module: Optional[nn.Module] = None

    def load(self, device: str = "cuda") -> nn.Module:
        if self._module is None:
            from diffusers import AutoencoderKLWan  # lazy: heavy optional dependency

            self._module = AutoencoderKLWan.from_pretrained(self.pretrained, subfolder="vae").to(device).eval()
            for param in self._module.parameters():
                param.requires_grad_(False)
        return self._module

    @torch.no_grad()
    def posterior_mean(self, video: torch.Tensor) -> torch.Tensor:
        module = self.load(video.device.type if video.device.type != "cpu" else "cpu")
        posterior = module.encode(video).latent_dist
        return posterior.mean


class DiTFinalLatentExtractor(nn.Module):
    """Runs a compatible frozen video DiT and returns its final denoised latent (plus trajectory)."""

    def __init__(self, spec: VAESpec, generator: str = "Wan2.1-T2V-1.3B", record_indices: tuple[int, ...] = ()):
        super().__init__()
        self.spec = spec
        self.generator = generator
        self.record_indices = tuple(record_indices)

    @torch.no_grad()
    def forward(self, sample_fn, condition: Any, num_steps: int = 50) -> tuple[torch.Tensor, LatentProvenance]:
        """`sample_fn(condition, step_callback)` must yield the terminal latent of the sampler."""
        trajectory: list[torch.Tensor] = []

        def callback(step: int, latents: torch.Tensor) -> None:
            if step in self.record_indices or step >= num_steps - 1:
                trajectory.append(latents.detach().clone())

        z = sample_fn(condition, callback)
        provenance = LatentProvenance(
            source="generated_dit", generator=self.generator, vae=self.spec, steps=num_steps, trajectory=trajectory
        )
        return z, provenance

    @property
    def compatibility_key(self) -> dict[str, Any]:
        return {
            "vae_checkpoint": self.spec.checkpoint_id,
            "latent_normalization": (self.spec.scaling, self.spec.shift),
            "tensor_layout": self.spec.layout,
            "compression": (self.spec.spatial_compression, self.spec.temporal_compression),
            "latent_channels": self.spec.channels,
        }


@dataclass
class CompatibleGenerators:
    """The evaluated common-VAE family: one checkpoint serves all of these unchanged."""

    t2v: tuple[str, ...] = ("Wan2.1-T2V-14B", "Wan2.1-T2V-1.3B")
    i2v: tuple[str, ...] = ("Wan2.2-I2V-A14B",)
    incompatible: tuple[str, ...] = ("CogVideoX-5B",)

    @property
    def all_compatible(self) -> tuple[str, ...]:
        return tuple(self.t2v) + tuple(self.i2v)
