#!/usr/bin/env python3
"""Fetch the real weights the framework needs (Wan VAE, 4RC, DINOv2, CLIP) via the HF Python API.

huggingface.co is unreachable from some networks, so HF_ENDPOINT defaults to the hf-mirror.com mirror.
Downloads resume across retries, which matters on throttled links.

    HF_ENDPOINT=https://hf-mirror.com python3 scripts/fetch_weights.py --dest /mnt/data/pixels-weights
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download

#: name -> (repo id, allow_patterns, purpose). All listed repos are ungated.
WEIGHTS = {
    "wan-vae": (
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        ["vae/config.json", "vae/diffusion_pytorch_model.safetensors"],
        "the shared video VAE: z^obs encoder and the common latent space of the Wan2.1/2.2 family (507.6 MB)",
    ),
    "4rc": (
        "Luo-Yihang/4RC",
        ["model.safetensors", "README.md", "LICENSE"],
        "pretrained 4D hierarchy initialising the 31-block refinement network and both heads (arXiv:2602.10094, 6.08 GB)",
    ),
    "dinov2-base": (
        "facebook/dinov2-base",
        ["config.json", "preprocessor_config.json", "model.safetensors"],
        "projection metrics backbone (346.3 MB)",
    ),
    "clip-vit-large-patch14": (
        "openai/clip-vit-large-patch14",
        ["config.json", "preprocessor_config.json", "tokenizer.json", "vocab.json", "merges.txt",
         "special_tokens_map.json", "model.safetensors"],
        "CLIP text/image scoring for Table 1 (1.71 GB)",
    ),
}

#: ~17 GB each and only needed to sample generated latents for the benchmark suites.
OPTIONAL = {
    "wan2.1-t2v-1.3b": (
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        ["model_index.json", "tokenizer/*", "text_encoder/*", "transformer/*", "scheduler/*"],
        "text-to-video DiT that produces z^gen (~17 GB)",
    ),
    "wan2.2-i2v-a14b": (
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        ["model_index.json", "tokenizer/*", "text_encoder/*", "transformer/*", "scheduler/*"],
        "image-to-video DiT that produces z^gen (~30 GB+)",
    ),
}


def expand(repo: str, pattern: str, endpoint: str) -> list[str]:
    """Resolve a glob like 'vae/*' against the repo's file listing."""
    import fnmatch
    import json
    import urllib.request

    with urllib.request.urlopen(f"{endpoint}/api/models/{repo}?blobs=true", timeout=60) as response:
        listing = json.load(response)
    return [s["rfilename"] for s in listing.get("siblings", []) if fnmatch.fnmatch(s["rfilename"], pattern)]


def repo_sizes(repo: str, endpoint: str) -> dict[str, int]:
    """File name -> byte size, from the mirror's API listing (HEAD requests stall on this network)."""
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(f"{endpoint}/api/models/{repo}?blobs=true", timeout=45) as response:
            listing = json.load(response)
        return {entry["rfilename"]: int(entry.get("size") or 0) for entry in listing.get("siblings", [])}
    except Exception as error:  # noqa: BLE001 - size is only used to detect completion
        print(f"    size lookup failed for {repo}: {type(error).__name__}", flush=True)
        return {}


def fetch_file(repo: str, name: str, target: str, endpoint: str, total: int = 0, attempts: int = 400) -> None:
    """Resume-friendly single-file download; aborts a stalled connection instead of hanging.

    --speed-limit/--speed-time cut any transfer that drops below 2 KB/s for 30 s, which is what
    actually keeps this going on a link that repeatedly stalls mid-request.
    """
    url = f"{endpoint}/{repo}/resolve/main/{name}"
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if total and os.path.exists(target) and os.path.getsize(target) == total:
        print(f"    already complete: {name} ({total / 1e6:.1f} MB)", flush=True)
        return
    for attempt in range(1, attempts + 1):
        have = os.path.getsize(target) if os.path.exists(target) else 0
        if total:
            print(f"    [{attempt}] {name}: {have / 1e6:.0f}/{total / 1e6:.0f} MB", flush=True)
        subprocess.run(
            ["curl", "-sL", "-C", "-", "--speed-limit", "2048", "--speed-time", "30",
             "--max-time", "3600", "-o", target, url],
            check=False,
        )
        now = os.path.getsize(target) if os.path.exists(target) else 0
        if total and now == total:
            print(f"    done: {name} ({total / 1e6:.1f} MB)", flush=True)
            return
        if not total and now > 0:
            print(f"    done(?): {name} ({now / 1e6:.1f} MB, no content-length)", flush=True)
            return
        time.sleep(min(2 * attempt, 20))
    raise RuntimeError(f"giving up on {repo}/{name} after {attempts} attempts")


def fetch(repo: str, files: list[str], target: str, endpoint: str) -> None:
    sizes = repo_sizes(repo, endpoint)
    for name in files:
        fetch_file(repo, name, os.path.join(target, name), endpoint, sizes.get(name, 0))


def size_of(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return total / 1e9


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="/mnt/data/pixels-weights")
    parser.add_argument("--only", nargs="*", default=None, help="subset of: " + " ".join(WEIGHTS))
    parser.add_argument("--include-optional", action="store_true", help="also pull the DiT checkpoints (~17 GB each)")
    parser.add_argument("--verify-only", action="store_true", help="report what is already on disk")
    args = parser.parse_args()

    wanted = dict(WEIGHTS)
    if args.only:
        wanted = {key: wanted[key] for key in args.only}
    if args.include_optional:
        wanted.update(OPTIONAL)

    os.makedirs(args.dest, exist_ok=True)
    endpoint = os.environ["HF_ENDPOINT"].rstrip("/")
    failures = []
    for name, (repo, allow, purpose) in wanted.items():
        target = os.path.join(args.dest, name)
        have = size_of(target) if os.path.isdir(target) else 0.0
        print(f"[{name}] {repo} :: {purpose} (on disk: {have:.2f} GB)", flush=True)
        if args.verify_only:
            continue
        files = [name for pattern in allow for name in expand(repo, pattern, endpoint)] if any("*" in a for a in allow) else allow
        try:
            fetch(repo, files, target, endpoint)
        except RuntimeError as error:
            failures.append(str(error))
        print(f"[{name}] now {size_of(target):.2f} GB", flush=True)

    if args.verify_only:
        return
    print("\npaths:")
    for name in wanted:
        print(f"  {name:28s} {os.path.join(args.dest, name)}")
    print(f"\nconfigs/model/l4ar_paper.yaml -> pretrained_init.checkpoint: {os.path.join(args.dest, '4rc', 'model.safetensors')}")
    print(f"                                      vae.checkpoint_id:          {os.path.join(args.dest, 'wan-vae')} (subfolder: vae)")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
