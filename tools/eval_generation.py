#!/usr/bin/env python3
"""Text4D-200 / I4D-200 evaluation: two off-axis projections, DINO global / match / F1, CLIP.

The suite is a locked 200-case benchmark; every method (ours and the matched same-latent cascade)
must be run on the same cached DiT latents for the comparison in paper Table 1 to be controlled.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from l4d.eval.baselines import CLAIMED_HEADLINE_GAINS, TABLE_1_REFERENCE, dino_f1_gains
from l4d.eval.projection_metrics import (
    DinoV2Extractor,
    OffAxisProtocol,
    ProjectionEvaluator,
    SurrogateExtractor,
)
from l4d.models.l4ar import L4ARConfig, build_l4ar
from l4d.utils.config import load_config


def load_cases(directory: str) -> list[dict]:
    if not os.path.isdir(directory):
        return []
    cases = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".pt"):
            continue
        payload = torch.load(os.path.join(directory, name), map_location="cpu", weights_only=False)
        cases.append(payload)
    return cases


def synthetic_cases(count: int, latent_shape: tuple[int, ...], seed: int = 0) -> list[dict]:
    generator = torch.Generator().manual_seed(seed)
    cases = []
    for index in range(count):
        cases.append(
            {
                "case_id": f"synthetic-{index:03d}",
                "latent": torch.randn(*latent_shape, generator=generator),
                "reference_rgb": torch.rand(3, 224, 224, generator=generator),
                "text": f"case number {index} rotating in place",
            }
        )
    return cases


def run_ours(model, cases: list[dict], device: str) -> list[dict]:
    points_per_case = []
    with torch.no_grad():
        for case in cases:
            latent = case["latent"].unsqueeze(0).to(device) if case["latent"].dim() == 4 else case["latent"].to(device)
            out = model(latent)
            points_per_case.append((case, out["points"][0].float().cpu(), case.get("reference_rgb"), case.get("text")))
    return points_per_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", dest="eval_cfg", default="configs/eval/text4d200.yaml")
    parser.add_argument("--model", default="configs/model/l4ar_tiny.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--cases", default=None, help="directory of cached generated-latent cases")
    parser.add_argument("--synthetic", type=int, default=0, help="run N synthetic cases (offline protocol check)")
    parser.add_argument("--extractor", default="surrogate", choices=["surrogate", "dinov2"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    eval_cfg = load_config(args.eval_cfg).to_dict()
    model_cfg = load_config(args.model).to_dict()
    model = build_l4ar(L4ARConfig.from_dict(model_cfg["model"])).to(args.device).eval()
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        model.load_state_dict(state.get("model", state), strict=False)

    if args.synthetic:
        spec = model_cfg["model"]["vae"]
        shape = (spec["channels"], 6, 24, 32)
        cases = synthetic_cases(args.synthetic, shape)
    else:
        cases = load_cases(args.cases or eval_cfg["cases_dir"])
        if not cases:
            raise SystemExit(f"no cases found in {args.cases or eval_cfg['cases_dir']} (use --synthetic for an offline run)")

    extractor = SurrogateExtractor(device=args.device) if args.extractor == "surrogate" else DinoV2Extractor(device=args.device)
    evaluator = ProjectionEvaluator(
        extractor, protocol=OffAxisProtocol(image_size=int(eval_cfg.get("image_size", 336)))
    )
    scores = []
    for case, points, reference, text in run_ours(model, cases, args.device):
        entry = evaluator.evaluate_case(points, None, reference, text)
        entry["case_id"] = case.get("case_id", "?")
        scores.append(entry)
    report = evaluator.aggregate(scores)
    row = report.as_row()
    print(json.dumps(row, indent=2))
    reference_row = TABLE_1_REFERENCE.get(eval_cfg.get("reference_row", "Ours (Wan2.1-1.3B)"))
    print("paper reference:", json.dumps(reference_row))
    paper_gains = dino_f1_gains(TABLE_1_REFERENCE)
    print("claimed gains vs matched 4RC cascade:", json.dumps(CLAIMED_HEADLINE_GAINS))
    print("paper-measured gains:", json.dumps({k: round(v, 2) for k, v in paper_gains.items()}))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"row": row, "per_case": scores, "reference": reference_row}, fh, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
