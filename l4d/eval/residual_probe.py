"""Fig. 6 diagnostic: sensitivity to DiT-derived residuals projected onto the Grid-Align null space."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .gt_metrics import camera_drift, point_map_drift


def alignment_width_matrix(alignment_conv: torch.nn.Conv3d) -> torch.Tensor:
    """Flattened weight of S_phi: rows = output token channels, columns = input latent neighbourhood."""
    weight = alignment_conv.weight.detach()
    return weight.reshape(weight.shape[0], -1)


def width_null_space(alignment_conv: torch.nn.Conv3d, tolerance: float = 1e-6) -> torch.Tensor:
    """Directions of a token's receptive field that Grid-Align cannot express.

    The exact null space of the per-token operator is empty whenever the operator is injective
    (kernel taps <= token width), so the most-attenuated right-singular directions are used instead;
    the appendix definition of "width-null space" is the one open ambiguity here.
    """
    matrix = alignment_width_matrix(alignment_conv)
    _, singular, vh = torch.linalg.svd(matrix, full_matrices=True)
    rank = int((singular > tolerance * singular.max()).sum())
    nullity = vh.shape[0] - rank
    if nullity > 0:
        return vh[rank:].T  # (input_dim, nullity)
    tail = min(max(1, matrix.shape[1] - matrix.shape[0]), 8)
    return vh[-tail:].T


def project_to_null(vector: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    flat = vector.reshape(-1)
    if flat.numel() != basis.shape[0]:
        raise ValueError(f"vector has {flat.numel()} entries but the null-space basis expects {basis.shape[0]}")
    return (basis @ (basis.T @ flat)).reshape(vector.shape)


def _grid(latent: torch.Tensor, conv: torch.nn.Conv3d) -> tuple[int, int, int, int, int, int]:
    channels, kt, kh, kw = latent.shape[1], *conv.kernel_size
    st, sh, sw = conv.stride
    if (kh, kw) != (sh, sw):
        raise ValueError("null-space projection needs non-overlapping windows (stride == kernel)")
    return channels, kt, kh, kw, latent.shape[2] // kt, latent.shape[3] // kh, latent.shape[4] // kw


def receptive_field_patches(latent: torch.Tensor, conv: torch.nn.Conv3d) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Per-token receptive-field vectors of a latent field, in S_phi's own flattening order."""
    channels, kt, kh, kw, gt, gh, gw = _grid(latent, conv)
    split = latent.reshape(latent.shape[0], channels, gt, kt, gh, kh, gw, kw)
    patches = split.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(-1, channels * kt * kh * kw)
    return patches, (latent.shape[0], gt, gh, gw, channels, kt, kh, kw)


def null_space_component(latent: torch.Tensor, conv: torch.nn.Conv3d, basis: torch.Tensor) -> torch.Tensor:
    """Field-wide projection of a latent perturbation onto the Grid-Align width null space."""
    patches, (b, gt, gh, gw, channels, kt, kh, kw) = receptive_field_patches(latent, conv)
    arranged = (patches @ basis @ basis.T).reshape(b, gt, gh, gw, channels, kt, kh, kw)
    folded = arranged.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(b, channels, gt * kt, gh * kh, gw * kw)
    return folded[:, :, : latent.shape[2], : latent.shape[3], : latent.shape[4]]


def near_terminal_residual(latents: list[torch.Tensor], late_index: int = 45, final_index: int = 50) -> torch.Tensor:
    """z_45 - z_50 from a recorded denoising trajectory (indices are the sampler's own step numbering)."""
    if len(latents) <= max(late_index, final_index):
        raise ValueError(f"trajectory has {len(latents)} entries, need index >= {max(late_index, final_index) + 1}")
    return latents[late_index] - latents[final_index]


@dataclass
class ResidualSensitivityResult:
    rho: float
    ours_point_drift: float
    baseline_point_drift: float
    ours_camera_drift: float
    baseline_camera_drift: float

    def favors_ours(self) -> bool:
        return self.ours_point_drift < self.baseline_point_drift and self.ours_camera_drift < self.baseline_camera_drift


@dataclass
class ResidualSensitivityStudy:
    """Applies one perturbation to both pathways and measures how far the outputs move."""

    rhos: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
    late_index: int = 45
    final_index: int = 50
    results: list[ResidualSensitivityResult] = field(default_factory=list)

    @torch.no_grad()
    def run(
        self,
        model,
        cascade,
        observed_latent: torch.Tensor,
        trajectory: list[torch.Tensor],
        vae=None,
    ) -> list[ResidualSensitivityResult]:
        residual = near_terminal_residual(trajectory, self.late_index, self.final_index)
        component = null_space_component(residual, model.alignment.local.conv, width_null_space(model.alignment.local.conv))
        self.results = []
        for rho in self.rhos:
            perturbed = observed_latent + rho * component
            ours = model(perturbed)
            ours_clean = model(observed_latent)
            if vae is None:
                baseline = cascade(perturbed)
                baseline_clean = cascade(observed_latent)
            else:
                baseline = cascade(vae.decode(perturbed))
                baseline_clean = cascade(vae.decode(observed_latent))
            self.results.append(
                ResidualSensitivityResult(
                    rho=rho,
                    ours_point_drift=point_map_drift(ours_clean["points"], ours["points"]),
                    baseline_point_drift=point_map_drift(baseline_clean["points"], baseline["points"]),
                    ours_camera_drift=camera_drift(ours_clean["camera_centers"], ours["camera_centers"]),
                    baseline_camera_drift=camera_drift(baseline_clean["camera_centers"], baseline["camera_centers"]),
                )
            )
        return self.results
