"""Structure / math checks for the Latent-to-4D (arXiv:2608.10744) reproduction framework.

Run: python -m pytest tests -q     (or: python tests/test_reproduction.py)
"""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from l4d.eval.baselines import CLAIMED_HEADLINE_GAINS, TABLE_1_REFERENCE, dino_f1_gains
from l4d.eval.gt_metrics import accuracy_completeness
from l4d.eval.projection_metrics import (
    OffAxisProtocol,
    ProjectionEvaluator,
    SurrogateExtractor,
    dino_set_f1,
    render_off_axis_views,
    render_validity_mask,
)
from l4d.eval.protocol import TABLE_3_VARIANTS
from l4d.eval.residual_probe import null_space_component, width_null_space
from l4d.losses.objectives import LatentTo4DLoss, confidence_weighted
from l4d.models.l4ar import L4ARConfig, VAESpec, build_l4ar
from l4d.models.refinement import FRAME, GLOBAL
from l4d.models.video_interface import (
    CompatibleGenerators,
    LatentProvenance,
    SharedLatentInterface,
    SyntheticVideoVAE,
)
from l4d.utils.geometry import decode_camera_9d, encode_camera_9d, quaternion_to_matrix, unproject_rays

TINY = dict(
    token_dim=64, heads=4, depth=5, tap_after=(1, 4), patch_size=2, align_kernel=(1, 2, 2),
    lora_rank=16, geometry_hidden=32, camera_hidden=32,
)


def model(**overrides):
    return build_l4ar(L4ARConfig(**{**TINY, **overrides}))


def latent(batch=1, channels=16, t=6, h=12, w=16):
    return torch.randn(batch, channels, t, h, w)


# --- Alignment (Eq. 5) ---------------------------------------------------------
def test_alignment_grid_and_token_shape():
    net = model()
    z = latent()
    grid = net.grid_for_latent(z)
    tokens, used = net.alignment(z, grid)
    t, hp, wp = grid
    assert grid == (1 + (6 - 1) * 4, 6, 8)
    assert tokens.shape == (1, t, hp * wp, TINY["token_dim"])
    assert used == grid


def test_alignment_without_local_conv_is_pointwise():
    full = model().alignment.local.conv
    pointwise = model(use_local_conv=False).alignment.local.conv
    assert tuple(full.kernel_size) == (1, 2, 2)
    assert tuple(pointwise.kernel_size) == (1, 1, 1)


def test_alignment_without_grid_changes_token_count():
    z = latent()
    aligned_grid = model().alignment(z, (21, 6, 8))[1]
    native_grid = model(grid_mode="none").alignment(z, (21, 6, 8))[1]
    assert aligned_grid == (21, 6, 8)
    assert native_grid != aligned_grid  # no fixed resampling -> tokens on the native latent grid


# --- Refinement ---------------------------------------------------------------
def test_alternating_scope_pattern_starts_with_frame():
    net = model(depth=7)
    scopes = [block.scope for block in net.refinement.blocks]
    assert scopes[0] == FRAME
    assert scopes[1] == GLOBAL and scopes[2] == FRAME, scopes


def test_attention_token_counts_differ_by_scope():
    net = model()
    z = latent()
    grid = net.grid_for_latent(z)
    tokens, _ = net.alignment(z, grid)
    t, m = grid[0], tokens.shape[2]
    out = net.refinement(tokens, grid)
    assert out.fused.shape == (1, t, m, 2 * TINY["token_dim"])  # frame ++ global at each tap
    assert len(out.levels) == len(TINY["tap_after"])


def test_lora_rank_and_frozen_base_weights():
    net = model()
    ranks = {p.shape[0] for name, p in net.refinement.named_parameters() if name.endswith("lora_a")}
    assert ranks == {TINY["lora_rank"]}
    base = [p for n, p in net.refinement.blocks.named_parameters() if "lora" not in n]
    assert base and all(not p.requires_grad for p in base)


def test_scope_ablations_run():
    for variant, overrides in TABLE_3_VARIANTS.items():
        net = model(**overrides)
        out = net(latent())
        assert torch.isfinite(out["points"]).all(), variant


# --- Decoder (Eq. 6) ----------------------------------------------------------
def test_point_recovery_follows_equation_six():
    directions = F.normalize(torch.randn(2, 3, 4, 5, 3), dim=-1)
    origins = torch.randn(2, 3, 3)
    depth = torch.rand(2, 3, 4, 5) + 0.5
    points = unproject_rays(directions, origins, depth)
    reference = origins[:, :, None, None, :] + depth[..., None] * directions
    assert torch.allclose(points, reference, atol=1e-5)
    assert points.shape == (2, 3, 4, 5, 3)


def test_camera_9d_roundtrip():
    quaternion = F.normalize(torch.randn(4, 4), dim=-1)
    matrix = quaternion_to_matrix(quaternion)
    assert torch.allclose(matrix @ matrix.transpose(-1, -2), torch.eye(3).expand(4, 3, 3), atol=1e-5)
    encoding = torch.cat([torch.randn(4, 3), quaternion, torch.rand(4, 2) + 0.4], dim=-1)
    assert encoding.shape[-1] == 9
    rot, center, fov = decode_camera_9d(encoding)
    again = encode_camera_9d(rot, center, fov)
    rot_again, center_again, fov_again = decode_camera_9d(again)  # quaternion sign is ambiguous
    assert torch.allclose(rot_again, rot, atol=1e-4)
    assert torch.allclose(center_again, center, atol=1e-4)
    assert torch.allclose(fov_again, fov, atol=1e-4)


def test_decoder_output_shapes():
    net = model()
    z = latent()
    out = net(z)
    assert out["depth"].shape[0] == 1 and out["depth"].dim() == 4
    assert out["ray_dirs"].shape[-1] == 3
    assert out["points"].shape == (*out["depth"].shape, 3)
    assert out["camera_encoding"].shape[-1] == 9
    assert torch.allclose(out["ray_dirs"].norm(dim=-1), torch.ones_like(out["depth"]), atol=1e-4)


# --- Staged freezing ----------------------------------------------------------
def test_progressive_activation_order():
    net = model()
    counts = []
    for stage in (1, 2, 3):
        net.set_stage(stage)
        counts.append(sum(p.numel() for p in net.parameters() if p.requires_grad))
    assert counts[0] < counts[1] < counts[2]


# --- Losses (Eq. 7) -----------------------------------------------------------
def test_confidence_weighted_formula():
    error = torch.tensor([[2.0, 0.5]])
    log_conf = torch.tensor([[0.5, -1.0]])
    valid = torch.ones_like(error, dtype=torch.bool)
    expected = (torch.exp(-log_conf) * error + log_conf).mean()
    assert torch.allclose(confidence_weighted(error, log_conf, valid), expected, atol=1e-6)


def test_full_loss_is_finite_and_decomposes():
    net = model()
    z = latent()
    pred = net(z)
    _, t, h, w = pred["depth"].shape
    batch = {
        "gt_depth": torch.rand(1, t, h, w) + 0.5,
        "depth_mask": torch.ones(1, t, h, w, dtype=torch.bool),
        "gt_ray_dirs": F.normalize(torch.randn(1, t, h, w, 3), dim=-1),
        "gt_camera_rotation": pred["camera_rotation"].detach(),
        "gt_camera_centers": pred["camera_centers"].detach(),
        "gt_fov": pred["fov"].detach(),
        "gt_points": pred["points"].detach() + 0.01 * torch.randn_like(pred["points"]),
        "point_mask": torch.ones(1, t, h, w, dtype=torch.bool),
    }
    losses = LatentTo4DLoss()(pred, batch)
    for key in ("loss", "loss_unc", "loss_cam", "loss_geom"):
        assert torch.isfinite(losses[key]), key
    losses["loss"].backward()


# --- Shared-latent interface --------------------------------------------------
def test_compatibility_gate_rejects_foreign_vae():
    spec = VAESpec()
    interface = SharedLatentInterface(spec)
    ours = LatentProvenance(source="generated_dit", generator="Wan2.1-T2V-14B", vae=spec)
    interface.check_compatible(ours, latent())
    foreign = VAESpec(name="cogvideox-vae", checkpoint_id="THUDM/CogVideoX-5b", channels=16)
    try:
        interface.check_compatible(
            LatentProvenance(source="generated_dit", generator="CogVideoX-5B", vae=foreign), latent()
        )
    except ValueError:
        pass
    else:
        raise AssertionError("CogVideoX-5B must not be accepted by the Wan-VAE interface")
    assert "CogVideoX-5B" in CompatibleGenerators().incompatible
    assert "Wan2.2-I2V-A14B" in CompatibleGenerators().all_compatible


def test_channel_mismatch_is_caught():
    interface = SharedLatentInterface(VAESpec())
    try:
        interface.check_compatible(
            LatentProvenance(source="observed_vae", generator="x", vae=VAESpec()), latent(channels=8)
        )
    except ValueError:
        return
    raise AssertionError("wrong latent channel count must raise")


def test_synthetic_vae_produces_expected_latent_shape():
    vae = SyntheticVideoVAE(VAESpec())
    encoded = vae.posterior_mean(torch.rand(1, 3, 21, 64, 96) * 2 - 1)
    assert encoded.shape == (1, 16, 6, 8, 12)  # Tz = 1 + (21-1)/4


# --- Fig. 6 residual probe ----------------------------------------------------
def test_null_space_component_is_invisible_to_grid_align():
    net = model(token_dim=8)  # rank-deficient operator -> non-trivial null space
    basis = width_null_space(net.alignment.local.conv)
    assert basis.shape[1] > 0
    residual = latent()
    component = null_space_component(residual, net.alignment.local.conv, basis)
    conv = net.alignment.local.conv
    after = F.conv3d(component, conv.weight, bias=None, stride=conv.stride)  # bias is not part of the operator
    assert float(after.abs().max()) < 1e-4 * float(residual.abs().max()) + 1e-6


# --- Metrics ------------------------------------------------------------------
def test_dino_set_f1_bounds_and_self_similarity():
    extractor = SurrogateExtractor(dim=64, device="cpu")
    image = torch.rand(3, 56, 56)
    perfect = dino_set_f1(image, image, extractor, threshold=0.99)
    assert 0.0 <= perfect["f1"] <= 100.0
    mismatched = dino_set_f1(image, torch.rand(3, 56, 56), extractor, threshold=0.99)
    assert perfect["f1"] > mismatched["f1"]


def test_off_axis_rendering_and_evaluation_pipeline():
    axis = torch.linspace(0, 4, 64)
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    grid = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
    points = grid.unsqueeze(0).expand(3, -1, -1, -1).contiguous()
    protocol = OffAxisProtocol(image_size=56, patch=14)
    views = render_off_axis_views(points, None, protocol, frame=1)
    assert len(views) == 2 and views[0].shape == (3, 56, 56)
    assert render_validity_mask(views[0], protocol).any()
    evaluator = ProjectionEvaluator(SurrogateExtractor(dim=64, stride=14, device="cpu"), protocol=protocol)
    score = evaluator.evaluate_case(points, None, torch.rand(3, 56, 56), "a chair", frame=1)
    report = evaluator.aggregate([score])
    assert set(report.as_row()) == {"Text CLIP", "CLIP-I", "DINO global", "DINO match", "DINO F1"}


def test_gt_metrics_perfect_match():
    cloud = torch.randn(200, 3)
    scores = accuracy_completeness(cloud, cloud.clone(), threshold=0.05)
    assert scores["accuracy"] < 1e-4 and scores["completeness"] < 1e-4


# --- Paper-number consistency -------------------------------------------------
def test_table_one_gains_match_claims():
    gains = dino_f1_gains(TABLE_1_REFERENCE)
    assert math.isclose(gains["Ours (Wan2.1-14B) vs Wan2.1-14B + 4RC"], 3.45, abs_tol=0.01)
    assert math.isclose(gains["Ours (Wan2.1-1.3B) vs Wan2.1-1.3B + 4RC"], 2.88, abs_tol=0.01)
    assert CLAIMED_HEADLINE_GAINS["i4d200_dino_f1"] == round(
        TABLE_1_REFERENCE["Ours (Wan2.2-I2V-A14B)"]["dino_f1"]
        - TABLE_1_REFERENCE["Wan2.2-I2V-A14B + 4RC"]["dino_f1"],
        2,
    )


def main() -> None:
    failures = []
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print(f"pass  {name}")
            except Exception as error:  # noqa: BLE001 - aggregate failures for a readable summary
                failures.append((name, error))
                print(f"FAIL  {name}: {type(error).__name__}: {error}")
    print(f"\n{len(failures)} failure(s)")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()


# --- Pretrained initialisation ------------------------------------------------
def test_vit_init_fills_the_refinement_blocks(tmp_path=None):
    """With a real local DINOv2 the frozen hierarchy must stop being random; skip if absent."""
    import glob

    from l4d.models.init_from import load_vit_into_refinement

    candidates = glob.glob("/mnt/data/pixels-weights/dinov2-base/model.safetensors")
    if not candidates:
        print("skip  vit init test (no local DINOv2 checkpoint)")
        return
    net = build_l4ar(L4ARConfig(**{**TINY, "token_dim": 768, "heads": 12, "depth": 12}))
    before = net.refinement.blocks[0].norm1.weight.detach().clone()
    report = load_vit_into_refinement(net.refinement, candidates[0], max_blocks=12)
    assert report["copied"] == 12 * 12, report          # qkv/proj/fc1/fc2 x2 + 2 norms per block
    assert report["shape_mismatches"] == [], report["shape_mismatches"]
    assert not torch.allclose(before, net.refinement.blocks[0].norm1.weight.detach())
    for name in ("blocks.0.attn.qkv.weight", "blocks.11.mlp.fc2.bias", "blocks.5.norm2.weight"):
        assert name in dict(net.refinement.named_parameters())
    assert torch.allclose(net.refinement.blocks[0].attn.qkv.base.weight.sum().float(),
                          net.refinement.blocks[0].attn.qkv.base.weight.detach().float().sum())


def test_gt_metrics_subsample_large_clouds_consistently():
    """Clouds past the sampling cap must keep normals aligned with the points actually scored."""
    from l4d.eval.gt_metrics import accuracy_completeness

    torch.manual_seed(0)
    cloud = torch.randn(30000, 3) * 3
    shifted = cloud + 0.01 * torch.randn_like(cloud)
    normals = F.normalize(torch.randn_like(cloud), dim=-1)
    scores = accuracy_completeness(cloud, shifted, threshold=0.2, normal_pred=normals, normal_gt=normals)
    assert scores["accuracy"] < 0.1 and scores["completeness"] < 0.1, scores
    assert scores["normal_consistency"] > 0.95, scores          # same normals both sides => near 1.0
    exact = accuracy_completeness(cloud, cloud, threshold=0.05, normal_pred=normals, normal_gt=normals)
    assert exact["accuracy"] < 1e-4 and exact["normal_consistency"] > 0.999, exact
