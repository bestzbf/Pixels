"""Initialise the frozen refinement hierarchy from an available pretrained ViT.

4RC is the paper's own initialiser, but its weights are not reachable from every network while
DINOv2 (which 4RC itself builds on, cf. Oquab et al. 2024) usually is. Loading a pretrained ViT into
the refinement blocks keeps the hierarchy *pretrained* rather than random, which is the assumption
the paper's staged, low-lr recipe rests on.

Accepted layouts (keys are normalised to `blocks.N.*`):
  * transformers ViT/DINOv2: `encoder.layer.N.attention.attention.{query,key,value}`,
    `encoder.layer.N.attention.output.dense`, `encoder.layer.N.mlp.fc1|fc2`,
    `encoder.layer.N.layernorm_before|layernorm_after`
  * torch.hub dinov2: `blocks.N.attn.qkv`, `blocks.N.attn.proj`, `blocks.N.mlp.fc1|fc2`, `blocks.N.norm1|norm2`
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any, Optional

import torch

LAYER_RE = re.compile(r"^(?:encoder\.)?layer\.(\d+)\.(.+)$")
# Any wrapper prefix is allowed (`backbone.pretrained.blocks.N` in the 4RC checkpoint) as long as the
# segment is exactly `blocks` - `self_blocks.0` must not match.
HUB_RE = re.compile(r"(?:^|\.)blocks\.(\d+)\.(.+)$")
_PREFIXES = ("model.", "vit.", "backbone.", "encoder.")


def _normalise_key(key: str) -> str:
    for prefix in _PREFIXES:
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key


def _mapped(suffix: str, hub_style: bool) -> Optional[str]:
    if hub_style:
        return suffix if suffix.startswith(("attn.", "mlp.", "norm")) else None
    if suffix.startswith("attention.attention."):
        return "attn." + suffix[len("attention.attention."):]
    if suffix.startswith("attention.output."):
        tail = suffix[len("attention.output."):]
        return "attn.proj" + tail[len("dense"):] if tail.startswith("dense") else None
    if suffix.startswith("mlp."):
        return "mlp." + suffix[len("mlp."):]
    if suffix.startswith("layernorm_before"):
        return "norm1" + suffix[len("layernorm_before"):]
    if suffix.startswith("layernorm_after"):
        return "norm2" + suffix[len("layernorm_after"):]
    # some DINOv2 ports already use norm1/norm2 naming; layer_scale* has no counterpart in our blocks
    if suffix.startswith(("norm1", "norm2")):
        return suffix
    return None


def _read_state_dict(path: str) -> dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        payload = load_file(path)
    else:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        payload = raw.get("model", raw) if isinstance(raw, dict) else raw
    return {key: value.float() for key, value in payload.items() if torch.is_tensor(value)}


def load_pretrained_vit(ckpt: str) -> dict[str, torch.Tensor]:
    """Reads a checkpoint file, or the first weight file found in a model directory."""
    if os.path.isdir(ckpt):
        for name in ("model.safetensors", "pytorch_model.bin"):
            candidate = os.path.join(ckpt, name)
            if os.path.exists(candidate):
                return _read_state_dict(candidate)
        raise FileNotFoundError(
            f"no model.safetensors / pytorch_model.bin under {ckpt}; sharded checkpoints must be merged first"
        )
    return _read_state_dict(ckpt)


def to_block_tensors(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Flatten either layout into `blocks.N.*`, fusing query/key/value into a single qkv."""
    grouped: dict[int, dict[str, torch.Tensor]] = defaultdict(dict)
    for key, value in state.items():
        normalised = _normalise_key(key)
        hub = HUB_RE.search(normalised)
        layer = LAYER_RE.match(normalised)
        match = hub if (hub and not layer) else layer
        if not match:
            continue
        suffix = _mapped(match.group(2), hub is not None and layer is None)
        if suffix:
            grouped[int(match.group(1))][suffix] = value

    out: dict[str, torch.Tensor] = {}
    for index, tensors in grouped.items():
        parts = [tensors.get(f"attn.{name}.weight") for name in ("query", "key", "value")]
        if all(part is not None for part in parts):
            out[f"blocks.{index}.attn.qkv.weight"] = torch.cat(parts, dim=0)
            biases = [tensors.get(f"attn.{name}.bias") for name in ("query", "key", "value")]
            if all(bias is not None for bias in biases):
                out[f"blocks.{index}.attn.qkv.bias"] = torch.cat(biases, dim=0)
        for name, value in tensors.items():
            if re.match(r"attn\.(query|key|value)\.", name):
                continue
            out[f"blocks.{index}.{name}"] = value

        # SwiGLU FFN -> plain two-layer MLP: w12 is [w1; w2] fused along the output dim, so w1 becomes
        # fc1 and w3 becomes fc2. The gate branch w2 has no counterpart in our block and is dropped.
        if "mlp.w12.weight" in tensors:
            w12 = tensors["mlp.w12.weight"]
            half = w12.shape[0] // 2
            out[f"blocks.{index}.mlp.fc1.weight"] = w12[:half]
            out[f"blocks.{index}.mlp.fc2.weight"] = tensors["mlp.w3.weight"]
            if "mlp.w12.bias" in tensors:
                out[f"blocks.{index}.mlp.fc1.bias"] = tensors["mlp.w12.bias"][:half]
                out[f"blocks.{index}.mlp.fc2.bias"] = tensors["mlp.w3.bias"]
    return out


def norm_gain_report(blocks: dict[str, torch.Tensor]) -> dict[str, float]:
    """Summary of the source LayerNorm gains, for the log.

    4RC's gains range continuously from ~0 to ~1.02 across depth, which is a learned distribution rather
    than a `1 + gamma` convention, so the values are transferred unchanged - a well-meaning +1 here would
    rescale every normalised activation.
    """
    gains = [value.float().mean() for name, value in blocks.items() if ".norm" in name and name.endswith(".weight")]
    if not gains:
        return {}
    stacked = torch.stack(gains)
    return {
        "norm_gains": int(stacked.numel()),
        "min_mean_gain": round(float(stacked.min()), 4),
        "max_mean_gain": round(float(stacked.max()), 4),
        "near_zero_gains": int((stacked.abs() < 0.05).sum()),
        "transfer": "raw (no offset correction)",
    }


def load_vit_into_refinement(
    refinement: torch.nn.Module, ckpt: str, max_blocks: Optional[int] = None
) -> dict[str, Any]:
    """Copy shape-compatible pretrained tensors into `refinement.blocks`.

    Returns counts plus the first few shape mismatches, so a wrong `token_dim`/`heads` assumption in the
    config shows up immediately instead of silently leaving the hierarchy random.
    """
    blocks = to_block_tensors(load_pretrained_vit(ckpt))
    own = dict(refinement.named_parameters())

    def resolve(key: str) -> Optional[str]:
        """LoRA wraps each Linear, moving its weights to `<name>.base.weight`."""
        for candidate in (key, key.replace(".weight", ".base.weight").replace(".bias", ".base.bias")):
            if candidate in own:
                return candidate
        return None

    copied: dict[str, torch.Tensor] = {}
    mismatches: list[str] = []
    for key, value in blocks.items():
        if max_blocks is not None and int(key.split(".")[1]) >= max_blocks:
            continue
        target = resolve(key)
        if target is None:
            continue
        if tuple(own[target].shape) != tuple(value.shape):
            mismatches.append(f"{target}: source {tuple(value.shape)} vs model {tuple(own[target].shape)}")
            continue
        copied[target] = value
    if copied:
        refinement.load_state_dict(copied, strict=False)
    return {
        **norm_gain_report(blocks),
        "source_blocks": len({key.split(".")[1] for key in blocks}),
        "copied": len(copied),
        "block_params": len([name for name in own if name.startswith("blocks.")]),
        "shape_mismatches": mismatches[:6],
    }
