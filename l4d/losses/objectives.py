"""Training objective, paper Eq. (7): L = L_unc + L_cam + L_geom."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..utils.geometry import surface_normals


def confidence_weighted(error: torch.Tensor, log_conf: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """exp(-s) * error + s, averaged over valid pixels (loss weights predicted by the confidences)."""
    weight = torch.exp(-log_conf)
    terms = weight * error + log_conf
    mask = valid.to(terms.dtype)
    return (terms * mask).sum() / mask.sum().clamp(min=1.0)


def gradient_loss(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Finite-difference (depth-gradient) consistency in image space, pred/target: (B,T,H,W)."""
    losses = []
    for axis in (-2, -1):
        dp = torch.diff(pred, dim=axis)
        dt = torch.diff(target, dim=axis)
        pair = torch.index_select(valid, axis, torch.arange(1, valid.shape[axis], device=valid.device)) & \
            torch.index_select(valid, axis, torch.arange(0, valid.shape[axis] - 1, device=valid.device))
        losses.append(confidence_weighted((dp - dt).abs(), torch.zeros_like(dp), pair))
    return torch.stack(losses).mean()


def align_points_scale(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Least-squares scale+shift alignment of predicted world points to GT (generated scenes lack metric scale)."""
    mask = valid.unsqueeze(-1).to(pred.dtype)
    denom = mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
    mu_p = (pred * mask).sum(dim=(1, 2, 3), keepdim=True) / denom.view(-1, 1, 1, 1, 1)
    mu_t = (target * mask).sum(dim=(1, 2, 3), keepdim=True) / denom.view(-1, 1, 1, 1, 1)
    num = ((pred - mu_p) * (target - mu_t) * mask).sum(dim=(1, 2, 3), keepdim=True)
    den = (((pred - mu_p) ** 2) * mask).sum(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
    scale = (num / den).clamp(1e-3, 1e3)
    return (pred - mu_p) * scale + mu_t


def tail_error(error: torch.Tensor, valid: torch.Tensor, quantile: float = 0.9) -> torch.Tensor:
    flat = error[valid]
    if flat.numel() == 0:
        return error.sum() * 0.0
    threshold = torch.quantile(flat.float(), quantile)
    return flat[flat >= threshold].mean()


@dataclass
class LossConfig:
    depth_weight: float = 1.0
    depth_gradient_weight: float = 1.0
    ray_weight: float = 1.0
    camera_translation_weight: float = 1.0
    camera_rotation_weight: float = 1.0
    camera_fov_weight: float = 0.5
    point_weight: float = 1.0
    point_tail_weight: float = 0.5
    normal_weight: float = 1.0
    tail_quantile: float = 0.9
    quantiles: tuple[float, float] = (0.2, 0.8)


class LatentTo4DLoss(torch.nn.Module):
    """Combines the uncertainty-aware, camera and geometry terms over a batch of 4D annotations."""

    def __init__(self, cfg: LossConfig | None = None):
        super().__init__()
        self.cfg = cfg or LossConfig()

    def forward(self, pred: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        c = self.cfg
        depth, gt_depth = pred["depth"], batch["gt_depth"]
        depth_mask = batch["depth_mask"]

        l_depth = confidence_weighted((depth - gt_depth).abs(), pred["depth_log_conf"], depth_mask)
        l_grad = gradient_loss(torch.log(depth.clamp(min=1e-3)), torch.log(gt_depth.clamp(min=1e-3)), depth_mask)
        gt_rays = batch["gt_ray_dirs"]
        l_ray = confidence_weighted((pred["ray_dirs"] - gt_rays).norm(dim=-1), pred["ray_log_conf"], depth_mask)
        loss_unc = c.depth_weight * l_depth + c.depth_gradient_weight * l_grad + c.ray_weight * l_ray

        rot_pred, rot_gt = pred["camera_rotation"], batch["gt_camera_rotation"]
        cosine = ((rot_pred @ rot_gt.transpose(-1, -2)).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
        geodesic = torch.acos(cosine.clamp(-1.0 + 1e-6, 1.0 - 1e-6))
        l_cam = (
            c.camera_translation_weight * (pred["camera_centers"] - batch["gt_camera_centers"]).abs().mean()
            + c.camera_rotation_weight * geodesic.abs().mean()
            + c.camera_fov_weight * (pred["fov"] - batch["gt_fov"]).abs().mean()
        )

        points_gt = batch["gt_points"]
        valid_points = batch["point_mask"]
        aligned = align_points_scale(pred["points"], points_gt, valid_points)
        err = (aligned - points_gt).norm(dim=-1)
        mask = valid_points
        denom = mask.sum().clamp(min=1.0)
        mean_err = (err * mask).sum() / denom
        tail = tail_error(err, mask, c.tail_quantile)
        normal_pred = surface_normals(aligned, mask)
        normal_gt = surface_normals(points_gt, mask)
        normal_err = (1.0 - (normal_pred * normal_gt).sum(-1).clamp(-1, 1)) * mask
        loss_geom = (
            c.point_weight * mean_err
            + c.point_tail_weight * tail
            + c.normal_weight * normal_err.sum() / denom
        )
        total = loss_unc + l_cam + loss_geom
        return {
            "loss": total,
            "loss_unc": loss_unc.detach(),
            "loss_depth": l_depth.detach(),
            "loss_depth_grad": l_grad.detach(),
            "loss_ray": l_ray.detach(),
            "loss_cam": l_cam.detach(),
            "loss_geom": loss_geom.detach(),
            "point_mean_err": mean_err.detach(),
            "point_tail_err": tail.detach(),
        }
