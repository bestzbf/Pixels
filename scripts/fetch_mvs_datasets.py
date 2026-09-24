#!/usr/bin/env python3
"""Fetch the *complete* MVS training corpora that are actually reachable from this machine.

    python scripts/fetch_mvs_datasets.py dtu                 # full DTU: 23 scans, ~14 GB
    python scripts/fetch_mvs_datasets.py hypersim --scenes 8 # first 8 Hypersim scenes
    python scripts/fetch_mvs_datasets.py hypersim            # the whole 451-scene release (~265 GB)

Why these two: the official university hosts measured 10-19 KB/s from this box (Princeton 19 KB/s, Stanford
11 KB/s), so "full dataset" here means "full on the one channel that answers" - hf-mirror at ~2.3 MB/s.
BlendedMVS was dropped because its mirror is missing `image.part3` (HTTP 404): a multi-part archive with a
hole is not a complete dataset, and only the images depend on that part.

Downloads are resumable (HTTP Range, verified against the size the API reports), each scene is extracted as
soon as its archive lands so training can start on what exists, and nothing already on disk is refetched.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.request

BASE = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
DEST = os.environ.get("DEST", "/mnt/data/pixels-benchmarks")
CHUNK = 1 << 20
# hf-mirror answers 403 to urllib's default agent (curl passes because it sends one), so the agent is part of the request.
AGENT = "Mozilla/5.0 (pixels-reproduction fetch)"


def api(path: str) -> list[dict]:
    request = urllib.request.Request(f"{BASE}/api/datasets/{path}", headers={"User-Agent": AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def listing(repo: str, folder: str = "") -> list[dict]:
    try:
        return [row for row in api(f"{repo}/tree/main/{folder}" if folder else f"{repo}/tree/main")]
    except Exception as error:  # noqa: BLE001 - an empty directory is not an error worth crashing on
        print(f"  (no listing for {repo}/{folder}: {type(error).__name__})", flush=True)
        return []


def fetch(url: str, target: str, expected: int) -> bool:
    """Resume-friendly download; returns False when the remote end is unavailable."""
    if os.path.exists(target) and os.path.getsize(target) >= expected > 0:
        return True
    os.makedirs(os.path.dirname(target), exist_ok=True)
    part = target + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    for attempt in range(1, 12):
        try:
            headers = {"User-Agent": AGENT}
            if have:
                headers["Range"] = f"bytes={have}-"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response, open(part, "ab" if have else "wb") as sink:
                while True:
                    block = response.read(CHUNK)
                    if not block:
                        break
                    sink.write(block)
                    have += len(block)
            if expected and have < expected:
                raise OSError(f"short read {have}/{expected}")
            os.replace(part, target)
            return True
        except Exception as error:  # noqa: BLE001 - mirrors drop connections; that is the normal case
            print(f"  retry {attempt} for {os.path.basename(target)} at {have/1e6:.0f} MB ({type(error).__name__})", flush=True)
            time.sleep(min(60, 5 * attempt))
    return False


def url_for(repo: str, path: str) -> str:
    return f"{BASE}/datasets/{repo}/resolve/main/{path}"


def stage_dtu(scans: int) -> None:
    repo = "HarrisonPENG/dtu"
    rows = listing(repo, "dtu")
    wanted = [row["path"] for row in rows if row["type"] == "directory"][:scans]
    print(f"DTU: {len(wanted)} scans of {len(rows)}", flush=True)
    for scan in wanted:
        root = os.path.join(DEST, "dtu", os.path.basename(scan))
        if os.path.exists(os.path.join(root, "done")):
            print(f"  have {os.path.basename(scan)}", flush=True)
            continue
        os.makedirs(root, exist_ok=True)
        missing = 0
        for sub in ("cams", "images", "gt_depths"):
            for row in [r for r in listing(repo, f"{scan}/{sub}") if r["type"] == "file"]:
                if not fetch(url_for(repo, row["path"]), os.path.join(root, sub, row["path"].split("/")[-1]),
                             row.get("size") or 0):
                    missing += 1
        ply = os.path.join(root, "scan.ply")
        fetch(url_for(repo, f"{scan}/scan.ply"), ply, next((r.get("size") or 0 for r in listing(repo, scan)
                                                            if r["path"].endswith("scan.ply")), 0))
        if missing == 0:
            open(os.path.join(root, "done"), "w").close()
        print(f"  {os.path.basename(scan)}: {missing} failed files", flush=True)


def stage_hypersim(scenes: int) -> None:
    repo = "HarrisonPENG/hypersim"
    rows = [row for row in listing(repo, "") if row["type"] == "file" and row["path"].endswith(".tar.gz")]
    if scenes:
        rows = rows[:scenes]
    print(f"Hypersim: {len(rows)} scene archives ({sum(r.get('size') or 0 for r in rows)/1e9:.0f} GB)", flush=True)
    for row in rows:
        name = row["path"].split("/")[-1]
        archive = os.path.join(DEST, "hypersim", name)
        marker = archive[:-len(".tar.gz")] + ".extracted"
        if os.path.exists(marker):
            print(f"  have {name}", flush=True)
            continue
        if not fetch(url_for(repo, row["path"]), archive, row.get("size") or 0):
            print(f"  FAILED {name}", flush=True)
            continue
        os.makedirs(os.path.dirname(marker), exist_ok=True)
        try:
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(path=os.path.dirname(archive))
            open(marker, "w").close()
            os.remove(archive)   # scenes are ~600 MB each and there are 451 of them: do not keep both copies
            print(f"  extracted {name}", flush=True)
        except Exception as error:  # noqa: BLE001 - a corrupt archive must not stop the rest
            print(f"  extract failed for {name}: {type(error).__name__}: {error}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("dtu", "hypersim", "both"))
    parser.add_argument("--scans", type=int, default=64, help="DTU scans to take (default: all)")
    parser.add_argument("--scenes", type=int, default=0, help="Hypersim scene cap (0 = the whole release)")
    args = parser.parse_args()
    os.makedirs(DEST, exist_ok=True)
    if args.stage in ("dtu", "both"):
        stage_dtu(args.scans)
    if args.stage in ("hypersim", "both"):
        stage_hypersim(args.scenes)
    print("FETCH_STAGE_DONE", flush=True)


if __name__ == "__main__":
    main()
