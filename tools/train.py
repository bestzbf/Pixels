#!/usr/bin/env python3
"""Stage-wise geometry-supervised training of L4AR (paper Eq. 4 and 7).

CPU smoke run:
    python tools/train.py --model configs/model/l4ar_tiny.yaml --data configs/data/smoke.yaml \
        --train configs/train/staged.yaml --stage stage1_adapter --steps 3 --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Dataset

from l4d.data.dataset import (
    SyntheticClips,
    collate_clips,
    load_manifest,
    manifest_report,
    ReconstructionClips,
)
from l4d.losses.objectives import LatentTo4DLoss, LossConfig
from l4d.models.l4ar import L4AR, L4ARConfig, build_l4ar
from l4d.models.video_interface import SyntheticVideoVAE, WanVideoVAE
from l4d.utils.config import load_config


def build_model_from_config(model_cfg: dict) -> L4AR:
    return build_l4ar(L4ARConfig.from_dict(model_cfg["model"]))


def build_vae(model_cfg: dict):
    from l4d.models.l4ar import vae_spec_from_dict

    spec = vae_spec_from_dict(model_cfg["model"]["vae"])
    if model_cfg["model"]["vae"].get("backend", "wan") == "synthetic":
        return SyntheticVideoVAE(spec)
    return WanVideoVAE(spec, pretrained=spec.checkpoint_id)


def temporal_grid(model: L4AR, latent: torch.Tensor, video: torch.Tensor):
    frames = video.shape[2]
    _, _, _, height, width = latent.shape
    patch = model.cfg.patch_size
    return (frames, max(height // patch, 1), max(width // patch, 1)), (video.shape[3], video.shape[4])


def parameter_groups(model: L4AR, base_lr: float, lora_lr_scale: float = 1.0) -> list[dict]:
    groups = []
    for name, params in model.trainable_parameter_groups().items():
        if not params or name == "frozen":
            continue
        lr = base_lr * (lora_lr_scale if name == "refinement_lora" else 1.0)
        groups.append({"params": params, "lr": lr, "name": name})
    return groups


def training_step(model: L4AR, vae, loss_fn: LatentTo4DLoss, batch: dict, device: str) -> tuple[dict, torch.Tensor]:
    video = batch["video"].to(device)
    latent = vae.posterior_mean(video)
    grid, output_size = temporal_grid(model, latent, video)
    prediction = model(latent, grid=grid, output_size=output_size)
    ground_truth = {
        key: (batch[key].to(device) if torch.is_tensor(batch[key]) else batch[key])
        for key in ("gt_depth", "gt_ray_dirs", "gt_points", "gt_camera_rotation", "gt_camera_centers", "gt_fov", "depth_mask", "point_mask")
    }
    frames = prediction["depth"].shape[1]
    ground_truth = {
        key: (value[:, :frames] if torch.is_tensor(value) and value.dim() >= 2 and value.shape[1] >= frames else value)
        for key, value in ground_truth.items()
    }
    losses = loss_fn(prediction, ground_truth)
    return losses, losses["loss"]


def summarize(losses: dict) -> str:
    return " ".join(f"{key}={float(value):.4f}" for key, value in losses.items() if key != "loss")


def build_dataloader(data_cfg: dict) -> DataLoader:
    if data_cfg.get("backend", "manifest") == "synthetic":
        dataset: Dataset = SyntheticClips(
            size=int(data_cfg.get("size", 4)),
            frames=int(data_cfg.get("frames", 21)),
            height=int(data_cfg.get("height", 32)),
            width=int(data_cfg.get("width", 40)),
        )
    else:
        records = load_manifest(data_cfg["manifest"], split=data_cfg.get("split"))
        print("manifest:", json.dumps(manifest_report(records)), flush=True)
        dataset = ReconstructionClips(
            records, image_size=tuple(data_cfg.get("resolution", (192, 256))), frames=int(data_cfg.get("frames", 21))
        )
    return DataLoader(dataset, batch_size=int(data_cfg.get("batch_size", 2)), shuffle=True, collate_fn=collate_clips)


def run_stage(model: L4AR, stage: dict, data_cfg: dict, train_cfg: dict, device: str, log_dir: str) -> str:
    activate = {"alignment": 1, "decoder_heads": 2, "refinement_lora": 3}
    stage_index = max(activate[name] for name in stage["activate"])
    model.set_stage(stage_index)
    print(f"[{stage['name']}] trainable groups activated: {model.set_stage(stage_index)}", flush=True)
    print(f"[{stage['name']}] parameters: {json.dumps(model.count_parameters())}", flush=True)

    vae = build_vae(train_cfg["model_cfg"]).to(device).eval()
    for param in vae.parameters():
        param.requires_grad_(False)
    batch_size = stage.get("batch_size") or data_cfg.get("batch_size", 1)
    loader = build_dataloader({**data_cfg, "batch_size": batch_size})
    loss_fn = LatentTo4DLoss(LossConfig(**train_cfg.get("loss", {})))
    optimizer = torch.optim.AdamW(
        parameter_groups(model, float(stage.get("lr", 1e-4)), float(stage.get("lora_lr_scale", 1.0))),
        weight_decay=float(stage.get("weight_decay", 0.0)),
    )
    steps = int(stage.get("steps", 3))
    clip = float(stage.get("gradient_clip", 0) or 0)
    model.train()
    iterator = iter(loader)
    started = time.time()
    for step in range(1, steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        losses, objective = training_step(model, vae, loss_fn, batch, device)
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if step % int(train_cfg.get("log_every", 1)) == 0 or step == steps:
            print(f"[{stage['name']}] step {step}/{steps} loss={losses['loss'].item():.4f} {summarize(losses)}", flush=True)
    from l4d.utils.checkpoint import save_checkpoint

    checkpoint = os.path.join(log_dir, f"{stage['name']}.pt")
    info = save_checkpoint(checkpoint, model, train_cfg["model_cfg"], stage,
                           full=train_cfg.get("save_full", False), init_source=train_cfg.get("init_source", "random"))
    print(f"[{stage['name']}] saved {checkpoint} in {time.time() - started:.1f}s "
          f"({info['kind']}: {info['tensors']} tensors, {info['mb']} MB)", flush=True)
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="configs/model/l4ar_tiny.yaml")
    parser.add_argument("--data", default="configs/data/smoke.yaml")
    parser.add_argument("--train", default="configs/train/staged.yaml")
    parser.add_argument("--stage", default=None, help="run one named stage instead of the whole schedule")
    parser.add_argument("--steps", type=int, default=None, help="override the stage step budget")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--save-full", action="store_true", help="store the frozen backbone too (~4.6 GB per stage at paper scale)")
    args = parser.parse_args()

    model_cfg = load_config(args.model).to_dict()
    data_cfg = load_config(args.data).to_dict()
    train_cfg = load_config(args.train).to_dict()
    train_cfg["model_cfg"] = model_cfg
    train_cfg["save_full"] = args.save_full
    device = args.device or train_cfg.get("device", "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("cuda unavailable, falling back to cpu", flush=True)
        device = "cpu"
    torch.manual_seed(int(train_cfg.get("seed", 0)))
    log_dir = args.output or os.path.join(train_cfg.get("output_dir", "runs/l4ar"), time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)

    model = build_model_from_config(model_cfg).to(device)
    init = model_cfg.get("pretrained_init", {})
    train_cfg["init_source"] = "random"
    checkpoint = init.get("checkpoint")
    if checkpoint and not os.path.exists(checkpoint):
        print(
            f"WARNING: 4RC init weights absent ({checkpoint}) - continuing with random initialisation. "
            "Structure and training dynamics are still valid; the paper's numbers are not comparable until "
            "the pretrained hierarchy is fetched (see scripts/fetch_weights.sh).",
            flush=True,
        )
    elif checkpoint:
        from l4d.models.l4ar import load_4rc_init

        try:
            report = load_4rc_init(model, checkpoint, bool(init.get("strict", False)))
            train_cfg["init_source"] = "4RC"
            print("4RC init:", json.dumps(report), flush=True)
        except Exception as error:  # noqa: BLE001 - a half-fetched checkpoint must not kill a run
            print(
                f"WARNING: 4RC checkpoint unusable ({type(error).__name__}: {str(error)[:120]}) - if a download "
                "is still running, wait for scripts/fetch_4rc_mirror.sh to print DONE before relying on this init; "
                "continuing with random initialisation.",
                flush=True,
            )
    elif init.get("vit"):
        from l4d.models.init_from import load_vit_into_refinement

        report = load_vit_into_refinement(model.refinement, init["vit"], max_blocks=model.cfg.depth)
        train_cfg["init_source"] = "vit"
        print(
            f"pretrained ViT init from {init['vit']}: copied {report['copied']}/{report['block_params']} "
            f"block params from {report['source_blocks']} source blocks",
            flush=True,
        )
        if report["shape_mismatches"]:
            print("  shape mismatches (first 6):", json.dumps(report["shape_mismatches"]), flush=True)
        if not report["copied"]:
            print("  nothing copied - check token_dim/heads against the checkpoint", flush=True)
    stages = train_cfg["stages"]
    if args.stage:
        stages = [stage for stage in stages if stage["name"] == args.stage]
        if not stages:
            raise SystemExit(f"unknown stage {args.stage}; available: {[s['name'] for s in train_cfg['stages']]}")
    for stage in stages:
        if args.steps:
            stage = {**stage, "steps": args.steps}
        run_stage(model, stage, data_cfg, train_cfg, device, log_dir)


if __name__ == "__main__":
    main()
