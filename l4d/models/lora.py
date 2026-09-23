"""Rank-L (paper: rank-16) LoRA adapters injected into the frozen refinement hierarchy."""
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float | None = None, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = float(rank if alpha is None else alpha)
        self.scale = self.alpha / self.rank
        in_features, out_features = base.in_features, base.out_features
        self.lora_a = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @torch.no_grad()
    def reset_null(self) -> None:
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (self.dropout(x) @ self.lora_a.T @ self.lora_b.T) * self.scale


def _target_modules(model: nn.Module, patterns: Iterable[str]) -> list[tuple[nn.Module, str, str]]:
    found = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(p in module_name for p in patterns):
            parent_path, _, child = module_name.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            found.append((parent, child, module))
    return found


def attach_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: float | None = None,
    dropout: float = 0.0,
    target_modules: Iterable[str] = ("qkv", "proj", "fc1", "fc2"),
) -> list[LoRALinear]:
    """Wrap matching Linear layers in-place; returns the inserted adapters."""
    inserted: list[LoRALinear] = []
    for parent, child, module in _target_modules(model, list(target_modules)):
        adapter = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, child, adapter)
        inserted.append(adapter)
    return inserted


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for m in model.modules() if isinstance(m, LoRALinear) for p in (m.lora_a, m.lora_b)]
