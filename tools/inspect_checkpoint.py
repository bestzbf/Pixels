#!/usr/bin/env python3
"""Inspect a pretrained checkpoint and derive the L4AR architecture fields from it.

    python tools/inspect_checkpoint.py /mnt/data/pixels-weights/4rc/model.safetensors --match l4d/models

Prints the tensor inventory, then the inferred `depth / token_dim / heads / mlp_ratio / tap` candidates,
and finally which of the framework's parameter names the checkpoint can actually fill.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

BLOCK_RE = re.compile(r"(?:^|\.)(\w*blocks?[\._]?\d+|layers?[\._]\d+|h\.\d+)")
INDEX_RE = re.compile(r"^(.*?)\.(\d+)\.")


def load_index(path: str) -> dict[str, tuple[tuple[int, ...], str]]:
    if path.endswith(".safetensors"):
        from safetensors import safe_open

        index: dict[str, tuple[tuple[int, ...], str]] = {}
        with safe_open(path, framework="pt") as handle:
            for key in handle.keys():
                tensor = handle.get_slice(key)
                index[key] = (tuple(tensor.get_shape()), str(tensor.get_dtype()))
        return index
    state = torch.load(path, map_location="cpu", weights_only=False)
    state = state.get("model", state) if isinstance(state, dict) else state
    flat: dict[str, tuple[tuple[int, ...], str]] = {}
    for prefix, value in state.items():
        if isinstance(value, dict):
            for inner, tensor in flatten(value, prefix).items():
                flat[inner] = tensor
        elif torch.is_tensor(value):
            flat[prefix] = (tuple(value.shape), str(value.dtype))
    return flat


def flatten(mapping: dict, prefix: str) -> dict:
    out: dict[str, tuple[tuple[int, ...], str]] = {}
    for key, value in mapping.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten(value, name))
        elif torch.is_tensor(value):
            out[name] = (tuple(value.shape), str(value.dtype))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--model", default="configs/model/l4ar_tiny.yaml")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    index = load_index(args.checkpoint)
    total = sum(math.prod(shape) if shape else 1 for shape, _ in index.values())
    print(f"{len(index)} tensors, {total / 1e6:.1f} M parameters")

    layers: dict[str, set[int]] = collections.defaultdict(set)
    for key in index:
        match = INDEX_RE.match(key)
        if match:
            layers[match.group(1)].add(int(match.group(2)))
    repeated = {name: max(ids) + 1 for name, ids in layers.items() if len(ids) > 2}
    print("repeated block families (name -> depth):")
    for name, depth in sorted(repeated.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {name:60s} {depth}")

    dim_votes: collections.Counter[int] = collections.Counter()
    for key, (shape, _) in index.items():
        if key.endswith(("qkv.weight", "q.weight", "attn.qkv.weight")) and len(shape) == 2:
            dim_votes[shape[1]] += 1
        elif key.endswith(("weight",)) and len(shape) == 2 and shape[0] % shape[1] == 0 and shape[0] // shape[1] in {3, 4}:
            dim_votes[shape[1]] += 1
    print("most common token widths:", dim_votes.most_common(5))

    print(f"\nlargest {args.top} tensors:")
    for key, (shape, dtype) in sorted(index.items(), key=lambda kv: -math.prod(kv[1][0]))[: args.top]:
        print(f"  {key:70s} {str(shape):22s} {dtype}")

    if os.path.exists(args.model):
        from l4d.models.l4ar import L4ARConfig, build_l4ar
        from l4d.utils.config import load_config

        cfg = L4ARConfig.from_dict(load_config(args.model).to_dict()["model"])
        own = dict(build_l4ar(cfg).named_parameters())
        shape_match = [k for k, v in index.items() if k in own and own[k].shape == v[0]]
        name_match = [k for k in index if k in own]
        print(f"\nagainst {args.model}: name overlap {len(name_match)}, name+shape overlap {len(shape_match)}")
        print("example keys of each side:")
        print("  ckpt:", sorted(index)[:5])
        print("  ours:", sorted(own)[:5])


if __name__ == "__main__":
    main()
