"""Compact checkpoints for PEFT-style training.

Only 22.8 M of a 1.156 B-parameter L4AR is trainable (alignment, heads, rank-16 LoRA), so storing the
frozen backbone in every stage checkpoint wastes ~4.6 GB per file and implies nothing about its content.
Trainable-only checkpoints are restored onto a model built from the same config, and the frozen weights
are fingerprinted at save time so a later load can detect that the backbone is *not* the one that trained.
"""
from __future__ import annotations

import torch

FROZEN_SAMPLE_LIMIT = 64


def trainable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    state = model.state_dict()
    return {name: value for name, value in state.items() if name in trainable}


def frozen_fingerprint(model: torch.nn.Module) -> dict[str, list[float]]:
    frozen = {name for name, param in model.named_parameters() if not param.requires_grad}
    names = sorted(frozen)[:FROZEN_SAMPLE_LIMIT]
    state = model.state_dict()
    return {
        name: [float(state[name].shape[-1]) if state[name].dim() else 0.0,
               float(state[name].float().mean()), float(state[name].float().std())]
        for name in names
    }


def save_checkpoint(path: str, model: torch.nn.Module, config: dict, stage: dict, full: bool = False,
                    init_source: str = "random", seed: int | None = None) -> dict:
    state = model.state_dict() if full else trainable_state_dict(model)
    torch.save(
        {
            "model": state,
            "model_kind": "full" if full else "trainable-only",
            # Randomly drawn frozen weights make a fingerprint meaningless; only a pretrained init is worth pinning.
            "init_source": init_source,
            # a random frozen backbone has no pinable statistics, so reproducing an eval needs this seed
            "seed": seed,
            "frozen_fingerprint": frozen_fingerprint(model) if init_source != "random" else {},
            "config": config,
            "stage": stage,
        },
        path,
    )
    bytes_on_disk = sum(value.numel() * value.element_size() for value in state.values())
    return {"tensors": len(state), "mb": round(bytes_on_disk / 1e6, 1), "kind": "full" if full else "trainable-only"}


def load_checkpoint(model: torch.nn.Module, path: str, device: str = "cpu", tolerance: float = 2e-3) -> dict:
    """Restores a checkpoint and reports whether the frozen backbone still matches what trained."""
    state = torch.load(path, map_location=device, weights_only=False)
    saved_state = state.get("model", state)
    model.load_state_dict(saved_state, strict=False)
    # A trainable-only checkpoint restored into a changed architecture would otherwise lose its trained
    # heads silently: report any stored tensor the model no longer has.
    target = dict(model.state_dict())
    dropped = [name for name in saved_state if name not in target]
    if dropped:
        result = {"kind": state.get("model_kind", "legacy-full"), "frozen_mismatch": [], "dropped_tensors": dropped[:8]}
        return result
    saved = state.get("frozen_fingerprint") or {}
    if not saved or state.get("model_kind") == "full":
        return {"kind": state.get("model_kind", "legacy-full"), "frozen_mismatch": []}
    current = frozen_fingerprint(model)
    mismatched = []
    for name, (shape, mean, std) in saved.items():
        local = current.get(name)
        if local is None:
            continue
        if abs(local[1] - mean) > tolerance or abs(local[2] - std) > tolerance:
            mismatched.append(name)
    return {"kind": state.get("model_kind"), "frozen_mismatch": mismatched[:8], "checked": len(saved)}
