"""4D decoder D_omega: turns the refined token hierarchy into cameras and dynamic world-space point maps."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..utils.geometry import decode_camera_9d, intrinsics_from_fov, unproject_rays, z_depth_rays
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
        camera_layout: str = "mlp",
    ):
        super().__init__()
        self.geometry_head = GeometryHead(token_dim, n_levels=n_levels, hidden=geometry_hidden, ray_space=ray_space)
        # cam_dec layout consumes the concatenated frame++global tokens (2 * token_dim), like 4RC does.
        camera_in = token_dim * 2 if camera_layout == "cam_dec" else token_dim * 3
        self.camera_head = CameraHead(camera_in, hidden=camera_hidden, encoding_dim=camera_encoding_dim,
                                      layout=camera_layout)
        self.ray_space = ray_space

    def forward(
        self, refined: RefinementOutput, output_size: Optional[tuple[int, int]] = None
    ) -> dict[str, torch.Tensor]:
        geometry = self.geometry_head(refined.levels, refined.grid, output_size)
        pooled = refined.fused.mean(dim=2)
        camera_input = pooled if self.camera_head.layout == "cam_dec" else torch.cat(
            [refined.camera_tokens, pooled], dim=-1)
        camera_encoding = self.camera_head(camera_input)
        rot_world_from_cam, origins, fov = decode_camera_9d(camera_encoding)
        fovy, fovx = fov[..., 0], fov[..., 1]
        height, width = geometry["output_size"]
        intrinsics = intrinsics_from_fov(fovy, fovx, height, width)
        directions = geometry["ray_dirs"]
        if self.ray_space == "camera":
            directions = torch.einsum("btij,bthwj->bthwi", rot_world_from_cam, directions)
        directions = torch.nn.functional.normalize(directions, dim=-1)
        # the depth head is supervised with depth along the camera z axis (that is what every RGB-D release
        # ships), so Eq. (6) needs the ray whose camera-z component is 1, not the unit ray: unprojecting
        # with a unit ray under-reaches every off-axis pixel by cos(theta) - ~8 % averaged over a 4:3 frame,
        # ~25 % at its corners - while the depth term pushes the other way, and the two targets then cannot
        # both be satisfied by any head.
        points = unproject_rays(z_depth_rays(directions, rot_world_from_cam), origins, geometry["depth"])
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
