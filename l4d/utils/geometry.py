"""Tensor geometry helpers: rotations, cameras, rays, point maps, projections."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        -1,
    ).reshape(*q.shape[:-1], 3, 3)


def matrix_to_quaternion(m: torch.Tensor) -> torch.Tensor:
    m = m.reshape(-1, 3, 3)
    t = m.diagonal(dim1=-2, dim2=-1).sum(-1) + 1
    w = 0.5 * t.clamp(min=1e-12).sqrt()
    s = 1.0 / (4 * w)
    x = (m[:, 2, 1] - m[:, 1, 2]) * s
    y = (m[:, 0, 2] - m[:, 2, 0]) * s
    z = (m[:, 1, 0] - m[:, 0, 1]) * s
    return F.normalize(torch.stack([w, x, y, z], -1), dim=-1).reshape(*m.shape[:-2], 4)


def decode_camera_9d(enc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """9D pose-FOV encoding -> (world_from_cam rotation, camera center, fov).

    Layout (documented assumption, see docs/PAPER_ANALYSIS.md section 8):
    [0:3] translation, [3:7] quaternion (w,x,y,z), [7:9] (fovy, fovx) in radians.
    """
    translation = enc[..., 0:3]
    quat = enc[..., 3:7]
    fov = enc[..., 7:9]
    rot_cam_from_world = quaternion_to_matrix(quat)
    rot_world_from_cam = rot_cam_from_world.transpose(-1, -2)
    camera_center = torch.einsum("...ij,...j->...i", rot_world_from_cam, -translation)
    return rot_world_from_cam, camera_center, fov


def encode_camera_9d(
    rot_world_from_cam: torch.Tensor, camera_center: torch.Tensor, fov: torch.Tensor
) -> torch.Tensor:
    rot_cam_from_world = rot_world_from_cam.transpose(-1, -2)
    translation = torch.einsum("...ij,...j->...i", rot_cam_from_world, -camera_center)
    return torch.cat([translation, matrix_to_quaternion(rot_cam_from_world), fov], dim=-1)


def fov_to_focal(fov: torch.Tensor, resolution: float) -> torch.Tensor:
    """Vertical/horizontal fov in radians -> focal length in pixels."""
    return resolution / (2.0 * torch.tan(fov.clamp(min=1e-4) / 2.0))


def intrinsics_from_fov(fovy: torch.Tensor, fovx: torch.Tensor, h: int, w: int) -> torch.Tensor:
    fy = fov_to_focal(fovy, h)
    fx = fov_to_focal(fovx, w)
    zero = torch.zeros_like(fx)
    one = torch.ones_like(fx)
    flat = torch.stack([fx, zero, (w / 2.0) * one, zero, fy, (h / 2.0) * one, zero, zero, one], dim=-1)
    return flat.reshape(*fx.shape, 3, 3)


def unproject_rays(
    directions: torch.Tensor, origins: torch.Tensor, depth: torch.Tensor
) -> torch.Tensor:
    """Paper Eq. (6): P_t(u) = o_t + d_t(u) * r_t(u)."""
    directions = F.normalize(directions, dim=-1)
    while origins.dim() < directions.dim():
        origins = origins.unsqueeze(-2)
    return origins + depth.unsqueeze(-1) * directions


def depth_to_points(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Depth map (B,T,H,W) + K (B,T,3,3) -> camera-space points (B,T,H,W,3)."""
    b, t, h, w = depth.shape
    device, dtype = depth.device, depth.dtype
    v, u = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype), torch.arange(w, device=device, dtype=dtype), indexing="ij"
    )
    pix = torch.stack([u + 0.5, v + 0.5, torch.ones_like(u)], -1)
    inv_k = torch.linalg.inv(intrinsics)
    cam = torch.einsum("btij,...j->...i", inv_k.unsqueeze(-3).expand(b, t, 3, 3), pix)
    return cam * depth.unsqueeze(-1)


def points_to_camera(points: torch.Tensor, rot_world_from_cam: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """World points (...,3) -> camera coordinates: x_cam = R_world<-cam^T (x_world - C)."""
    rel = points - center.unsqueeze(-2)
    return torch.einsum("...ji,...j->...i", rot_world_from_cam.unsqueeze(-3), rel)


def surface_normals(points: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Normals from finite differences on a (B,T,H,W,3) point grid."""
    du = points[..., :, 1:, :] - points[..., :, :-1, :]
    dv = points[..., 1:, :, :] - points[..., :-1, :, :]
    du = F.pad(du, (0, 0, 0, 1, 0, 0))
    dv = F.pad(dv, (0, 0, 0, 0, 0, 1))
    n = F.normalize(torch.linalg.cross(du, dv, dim=-1), dim=-1)
    if mask is not None:
        n = n * mask.unsqueeze(-1).to(n.dtype)
    return n


def look_at_camera(
    target: torch.Tensor,
    elevation: float,
    azimuth: float,
    distance: float,
    fovy: float,
    aspect: float,
    device=None,
    dtype=torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Off-axis eval camera: returns (rot_world_from_cam (3,3), translation 9D-style center)."""
    el, az = torch.tensor(elevation, device=device, dtype=dtype), torch.tensor(azimuth, device=device, dtype=dtype)
    dir_vec = torch.stack(
        [torch.cos(el) * torch.sin(az), -torch.sin(el), torch.cos(el) * torch.cos(az)]
    )
    center = target + distance * dir_vec
    forward = F.normalize(target - center, dim=0)
    right = F.normalize(torch.linalg.cross(forward, torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)), dim=0)
    down = torch.linalg.cross(forward, right)
    rot_cam_from_world = torch.stack([right, down, forward])
    rot_world_from_cam = rot_cam_from_world.transpose(0, 1)
    translation = rot_cam_from_world @ (-center)
    quat = matrix_to_quaternion(rot_cam_from_world)[0]
    fov = torch.stack([torch.tensor(fovy, dtype=dtype, device=device), torch.tensor(fovy * aspect, dtype=dtype, device=device)])
    return rot_world_from_cam, torch.cat([translation, quat, fov])


def render_point_map(
    points: torch.Tensor,
    colors: Optional[torch.Tensor],
    extrinsic_9d: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: int,
) -> torch.Tensor:
    """Splat a world-space point cloud into one off-axis view with a painter's-algorithm depth order.

    points: (N,3), colors: (N,3) or None, intrinsics: (3,3). Returns (image_size,image_size,3) in [0,1].
    """
    rot_world_from_cam, center, _ = decode_camera_9d(extrinsic_9d)
    camera = points_to_camera(points, rot_world_from_cam, center)
    depth = camera[:, 2]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    horizontal = (camera[:, 0] * fx / depth.clamp(min=1e-6) + cx).long()
    vertical = (camera[:, 1] * fy / depth.clamp(min=1e-6) + cy).long()
    visible = (
        (depth > 1e-4)
        & (horizontal >= 0) & (horizontal < image_size)
        & (vertical >= 0) & (vertical < image_size)
    )
    surface = torch.ones_like(depth).unsqueeze(-1) if colors is None else colors
    order = torch.argsort(depth[visible])  # far first, so nearer splats overwrite them
    slots = (vertical[visible] * image_size + horizontal[visible])[order]
    canvas = torch.zeros(image_size * image_size, surface.shape[-1], device=points.device, dtype=surface.dtype)
    canvas[slots] = surface[visible][order]
    return canvas.reshape(image_size, image_size, -1)
