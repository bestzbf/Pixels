"""Training data: reconstruction clips with metric 4D annotations, plus the VAE-latent cache.

The paper trains the final stage on 1,143 clips drawn from six reconstruction datasets and keeps every
benchmark held out. The exact dataset list / clip split is appendix-only content, so it is externalised
into a manifest (`data/manifest.jsonl`) and the six names below are placeholders to be confirmed.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

SIX_RECONSTRUCTION_DATASETS: tuple[str, ...] = (
    "7scenes",
    "nrgbd",
    "realestate10k",
    "co3d_v2",
    "dl3df",
    "omni3d_4d",
)  # TODO(verify against appendix): paper does not name the six datasets in the main text

TARGET_FINAL_STAGE_CLIPS = 1143


@dataclass
class ClipRecord:
    clip_id: str
    dataset: str
    video: str
    gt: str
    num_frames: int = 21
    resolution: tuple[int, int] = (192, 256)
    split: str = "train"

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "ClipRecord":
        return ClipRecord(
            clip_id=str(raw["clip_id"]),
            dataset=str(raw["dataset"]),
            video=str(raw["video"]),
            gt=str(raw["gt"]),
            num_frames=int(raw.get("num_frames", 21)),
            resolution=tuple(raw.get("resolution", (192, 256))),
            split=str(raw.get("split", "train")),
        )


def load_manifest(path: str, split: Optional[str] = None) -> list[ClipRecord]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = ClipRecord.from_dict(json.loads(line))
            if split is None or record.split == split:
                records.append(record)
    return records


def manifest_report(records: Iterable[ClipRecord]) -> dict[str, Any]:
    records = list(records)
    per_dataset: dict[str, int] = {}
    for record in records:
        per_dataset[record.dataset] = per_dataset.get(record.dataset, 0) + 1
    return {
        "clips": len(records),
        "per_dataset": per_dataset,
        "target_final_stage_clips": TARGET_FINAL_STAGE_CLIPS,
        "meets_target": len(records) >= TARGET_FINAL_STAGE_CLIPS,
    }


class ReconstructionClips(Dataset):
    """Yields (video, gt 4D annotations); the frozen VAE encoding happens in `precompute_latents`."""

    def __init__(self, records: list[ClipRecord], image_size: tuple[int, int] = (192, 256), frames: int = 21):
        self.records = records
        self.image_size = image_size
        self.frames = frames

    def __len__(self) -> int:
        return len(self.records)

    def _load_video(self, record: ClipRecord) -> torch.Tensor:
        from PIL import Image

        paths = sorted(
            os.path.join(record.video, name)
            for name in os.listdir(record.video)
            if name.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        index = np.linspace(0, len(paths) - 1, self.frames).round().astype(int)
        frames = []
        for position in index:
            image = Image.open(paths[position]).convert("RGB").resize((self.image_size[1], self.image_size[0]))
            array = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)
            frames.append(array.permute(2, 0, 1))
        video = torch.stack(frames, dim=1)  # (3,T,H,W)
        return video * 2.0 - 1.0  # the video VAEs are trained on [-1,1]

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        payload = np.load(record.gt)
        gt_points = torch.from_numpy(np.asarray(payload["points"], dtype=np.float32))
        gt_depth = torch.from_numpy(np.asarray(payload["depth"], dtype=np.float32))
        mask = torch.from_numpy(np.asarray(payload.get("mask", np.ones(gt_depth.shape, dtype=bool))))
        return {
            "clip_id": record.clip_id,
            "dataset": record.dataset,
            "video": self._load_video(record),
            "gt_points": gt_points.permute(1, 0, 4, 2, 3) if gt_points.dim() == 5 else gt_points,
            "gt_depth": gt_depth,
            "depth_mask": mask,
            "gt_camera_rotation": torch.from_numpy(np.asarray(payload["camera_rotation"], dtype=np.float32)),
            "gt_camera_centers": torch.from_numpy(np.asarray(payload["camera_centers"], dtype=np.float32)),
            "gt_fov": torch.from_numpy(np.asarray(payload["fov"], dtype=np.float32)),
            "gt_ray_dirs": torch.from_numpy(np.asarray(payload["ray_dirs"], dtype=np.float32)),
            "point_mask": mask,
        }


class SyntheticClips(Dataset):
    """Offline stand-in with analytically consistent depth/ray/point supervision (shapes only)."""

    def __init__(self, size: int = 4, frames: int = 5, height: int = 32, width: int = 40, seed: int = 0):
        self.size = size
        self.frames = frames
        self.height = height
        self.width = width
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, Any]:
        t, h, w = self.frames, self.height, self.width
        depth = 1.0 + 0.5 * torch.rand(1, t, h, w, generator=self.generator)
        dirs = torch.nn.functional.normalize(torch.randn(1, t, h, w, 3, generator=self.generator), dim=-1)
        origins = 0.1 * torch.randn(1, t, 3, generator=self.generator)
        points = origins.unsqueeze(2).unsqueeze(2) + depth.unsqueeze(-1) * dirs
        rotation = torch.eye(3).view(1, 1, 3, 3).expand(1, t, 3, 3).clone()
        fov = torch.full((1, t, 2), 0.9)
        mask = torch.ones(1, t, h, w, dtype=torch.bool)
        return {
            "clip_id": f"synthetic-{index}",
            "dataset": "synthetic",
            "video": torch.rand(3, t, h, w, generator=self.generator) * 2 - 1,
            "gt_depth": depth.squeeze(0),
            "gt_ray_dirs": dirs.squeeze(0),
            "gt_points": points.squeeze(0),
            "gt_camera_rotation": rotation.squeeze(0),
            "gt_camera_centers": origins.squeeze(0),
            "gt_fov": fov.squeeze(0),
            "depth_mask": mask.squeeze(0),
            "point_mask": mask.squeeze(0),
        }


def collate_clips(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            out[key] = torch.stack(values)
        else:
            out[key] = values
    return out


class LatentCache:
    """Stores frozen-VAE posterior-mean latents on disk so training never re-encodes video."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def path(self, clip_id: str) -> str:
        return os.path.join(self.root, f"{clip_id}.pt")

    def save(self, clip_id: str, latent: torch.Tensor, meta: Optional[dict] = None) -> str:
        target = self.path(clip_id)
        torch.save({"latent": latent.detach().cpu(), "meta": meta or {}}, target)
        return target

    def load(self, clip_id: str) -> Optional[torch.Tensor]:
        target = self.path(clip_id)
        if not os.path.exists(target):
            return None
        return torch.load(target, map_location="cpu", weights_only=False)["latent"]

    def missing(self, clip_ids: list[str]) -> list[str]:
        return [clip_id for clip_id in clip_ids if not os.path.exists(self.path(clip_id))]
