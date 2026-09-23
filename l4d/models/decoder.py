"""4D decoder D_omega: turns the refined token hierarchy into cameras and dynamic world-space point maps."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..utils.geometry import decode_camera_9d, intrinsics_from_fov, unproject_rays
from .heads import CameraHead, GeometryHead
from .refinement import RefinementOutput


class FourDDecoder(nn.Module):
    """Initialized from a pretrained 4D reconstructors' heads; camera/time tokens stay frozen upstream."""

    def __init__(
        self,
        token_dim: int,
        n_levels: int = 5,
        geometry_hidden: int = 256,
        camera_hidden: int = 512,
        camera_encoding_dim: int = 9,
        ray_space: str = "world",
    ):
        super().__init__()
        self.geometry_head = GeometryHead(token_dim, n_levels=n_levels, hidden=geometry_hidden, ray_space=ray_space)
        self.camera_head = CameraHead(token_dim * 3, hidden=camera_hidden, encoding_dim=camera_encoding_dim)
        self.ray_space = ray_space

    def forward(
        self, refined: RefinementOutput, output_size: Optional[tuple[int, int]] = None
    ) -> dict[str, torch.Tensor]:
        geometry = self.geometry_head(refined.levels, refined.grid, output_size)
        camera_input = torch.cat([refined.camera_tokens, refined.fused.mean(dim=2)], dim=-1)
        camera_encoding = self.camera_head(camera_input)
        rot_world_from_cam, origins, fov = decode_camera_9d(camera_encoding)
        fovy, fovx = fov[..., 0], fov[..., 1]
        height, width = geometry["output_size"]
        intrinsics = intrinsics_from_fov(fovy, fovx, height, width)
        directions = geometry["ray_dirs"]
        if self.ray_space == "camera":
            directions = torch.einsum("btij,bthwj->bthwi", rot_world_from_cam, directions)
            directions = torch.nn.functional.normalize(directions, dim=-1)
        points = unproject_rays(directions, origins, geometry["depth"])
        return {
            "camera_encoding": camera_encoding,
            "camera_rotation": rot_world_from_cam,
            "camera_centers": origins,
            "fov": fov,
            "intrinsics": intrinsics,
            "depth": geometry["depth"],
            "depth_log_conf": geometry["depth_log_conf"],
            "ray_dirs": directions,
            "ray_log_conf": geometry["ray_log_conf"],
            "ray_origins": origins,
            "points": points,
        }
