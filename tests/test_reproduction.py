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
    params = dict(net.refinement.named_parameters())
    for name in ("blocks.0.attn.qkv.base.weight", "blocks.11.mlp.fc2.base.bias", "blocks.5.norm2.weight"):
        assert name in params, name                       # LoRA wraps each Linear as <name>.base.*
    assert torch.allclose(params["blocks.0.attn.qkv.base.weight"], params["blocks.0.attn.qkv.base.weight"].detach())


def test_gt_metrics_subsample_large_clouds_consistently():
    """Clouds past the sampling cap must keep normals aligned with the points actually scored."""
    from l4d.eval.gt_metrics import accuracy_completeness

    torch.manual_seed(0)
    grid = F.normalize(torch.randn(200, 2, 3), dim=-1) * 3.0          # a point sphere: neighbours share normals
    cloud = grid.reshape(-1, 3).repeat(76, 1)[:30000] + 0.001 * torch.randn(30000, 3)
    shifted = cloud + 0.01 * torch.randn_like(cloud)
    normals = F.normalize(cloud, dim=-1)
    scores = accuracy_completeness(cloud, shifted, threshold=0.2, normal_pred=normals, normal_gt=normals)
    assert scores["accuracy"] < 0.1 and scores["completeness"] < 0.1, scores
    assert scores["normal_consistency"] > 0.95, scores          # same normals both sides => near 1.0
    # the two clouds are subsampled independently, so identical inputs still differ by sampling noise
    exact = accuracy_completeness(cloud, cloud, threshold=0.05, normal_pred=normals, normal_gt=normals)
    assert exact["accuracy"] < 1e-2 and exact["normal_consistency"] > 0.999, exact


# --- Frozen-VAE latent handling ------------------------------------------------
def test_latent_cache_round_trip(tmp_path: str = "/tmp/pixels_latent_cache"):
    import shutil

    from l4d.data.dataset import LatentCache

    shutil.rmtree(tmp_path, ignore_errors=True)
    cache = LatentCache(tmp_path)
    tensor = torch.randn(1, 16, 6, 8, 10)
    cache.save("clip-a", tensor, {"vae": "test"})
    assert torch.equal(cache.load("clip-a"), tensor)
    assert cache.missing(["clip-a", "clip-b"]) == ["clip-b"]


def test_normalisation_uses_the_per_channel_vae_convention():
    spec = VAESpec(latents_mean=(0.0,) * 16, latents_std=(2.0,) * 16)
    interface = SharedLatentInterface(spec)
    latent = torch.ones(2, 16, 3, 4, 5)
    normalised = interface.normalize(latent)
    assert torch.allclose(normalised, torch.full_like(latent, 0.5))
    channel_mean = (0.0,) * 7 + (1.0,) * 8 + (0.0,) * 1
    per_channel = interface.__class__(VAESpec(latents_mean=channel_mean, latents_std=(1.0,) * 16))
    shifted = per_channel.normalize(torch.zeros(1, 16, 2, 2, 2))
    assert float(shifted[0, 7, 0, 0, 0]) == -1.0 and float(shifted[0, 0, 0, 0, 0]) == 0.0


def test_training_step_prefers_cached_latents():
    """A poisoned video plus a raising VAE proves the cached z^obs path is the one used."""
    import importlib.util

    from l4d.models.l4ar import build_l4ar

    net = build_l4ar(L4ARConfig(**TINY))
    spec = importlib.util.spec_from_file_location("train_tool", "tools/train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class ExplodingVAE:
        def posterior_mean(self, video):
            raise AssertionError("the frozen VAE should not run when a cached latent exists")

    net.cfg.vae.latents_mean = (0.0,) * 15 + (1.0,)          # a channel with a known offset
    net.cfg.vae.latents_std = (1.0,) * 16
    cached = torch.randn(1, 16, 6, 12, 16)
    batch = {"latent": cached, "video": torch.rand(1, 3, 21, 96, 128) * 10 - 5}
    latent = module.latent_from_batch(net, ExplodingVAE(), batch, "cpu")
    assert latent.shape == cached.shape
    assert torch.equal(latent[:, :15], cached[:, :15])           # unchanged channels
    assert torch.allclose(latent[:, 15], cached[:, 15] - 1.0)    # (z - mean)/std per channel




def test_source_layernorm_gains_transfer_unchanged(tmp_path: str = "/tmp/pixels_ln_gain"):
    """A well-meaning +1 offset correction would silently rescale every normalised activation."""
    import os
    import shutil

    from safetensors.torch import save_file

    from l4d.models.init_from import load_vit_into_refinement

    net = build_l4ar(L4ARConfig(**{**TINY, "token_dim": 64, "depth": 2}))
    source = {
        "backbone.pretrained.blocks.0.norm1.weight": torch.full((64,), 0.01),
        "backbone.pretrained.blocks.0.norm2.weight": torch.full((64,), 0.5),
        "backbone.pretrained.blocks.1.norm1.weight": torch.ones(64) * 1.02,
    }
    shutil.rmtree(tmp_path, ignore_errors=True)
    os.makedirs(tmp_path, exist_ok=True)
    save_file(source, os.path.join(tmp_path, "model.safetensors"))
    report = load_vit_into_refinement(net.refinement, tmp_path)
    assert torch.allclose(net.refinement.blocks[0].norm1.weight, torch.full((64,), 0.01), atol=1e-6)
    assert torch.allclose(net.refinement.blocks[1].norm1.weight, torch.full((64,), 1.02), atol=1e-6)
    assert report["transfer"] == "raw (no offset correction)"
    assert report["min_mean_gain"] == 0.01 and report["max_mean_gain"] == 1.02, report




def test_cam_dec_camera_head_transfers_by_name(tmp_path: str = "/tmp/pixels_camdec"):
    """The layout exists so 4RC's pretrained camera head loads without a rename table."""
    import os
    import shutil

    from safetensors.torch import save_file

    from l4d.models.l4ar import load_4rc_init

    cfg = L4ARConfig(**{**TINY, "token_dim": 64, "depth": 2, "camera_layout": "cam_dec"})
    net = build_l4ar(cfg)
    dim, width = cfg.token_dim, cfg.token_dim * 2  # attention is d-wide, the heads see 2d taps
    source = {
        "backbone.pretrained.blocks.0.attn.qkv.weight": torch.randn(3 * dim, dim),
        "backbone.pretrained.blocks.1.attn.qkv.weight": torch.randn(3 * dim, dim),
        "cam_dec.backbone.0.weight": torch.randn(width, width),
        "cam_dec.backbone.0.bias": torch.randn(width),
        "cam_dec.backbone.2.weight": torch.randn(width, width),
        "cam_dec.backbone.2.bias": torch.randn(width),
        "cam_dec.fc_t.weight": torch.randn(3, width),
        "cam_dec.fc_t.bias": torch.randn(3),
        "cam_dec.fc_qvec.weight": torch.randn(4, width),
        "cam_dec.fc_qvec.bias": torch.randn(4),
        "cam_dec.fc_fov.0.weight": torch.randn(2, width),
        "cam_dec.fc_fov.0.bias": torch.randn(2),
    }
    shutil.rmtree(tmp_path, ignore_errors=True)
    os.makedirs(tmp_path)
    save_file({key: value.contiguous() for key, value in source.items()}, os.path.join(tmp_path, "model.safetensors"))
    report = load_4rc_init(net, tmp_path)
    assert report["camera_head_copied"] == 10, report
    assert report["copied"] == 2, report  # the two block qkv tensors, not just the head
    head = net.decoder.camera_head.state_dict()
    assert torch.equal(head["fc_qvec.weight"], source["cam_dec.fc_qvec.weight"])
    assert torch.equal(head["backbone.0.weight"], source["cam_dec.backbone.0.weight"])


def test_trainable_only_checkpoint_needs_the_training_init(tmp_path: str = "/tmp/pixels_init_contract"):
    """A trainable-only checkpoint restored on a fresh random backbone answers a different question.

    This is the bug that made the first 4RC-init evaluation read as though nothing had been trained: the
    frozen hierarchy is not in the checkpoint, so restoring heads without rebuilding the backbone they
    were trained against silently invalidates the number.
    """
    import os
    import shutil

    from safetensors.torch import save_file

    from l4d.models.l4ar import apply_pretrained_init, build_initialised_model
    from l4d.utils.checkpoint import load_checkpoint, save_checkpoint

    cfg = {**TINY, "depth": 2, "camera_layout": "cam_dec"}
    dim, width = cfg["token_dim"], cfg["token_dim"] * 2  # attention is d-wide, the heads see 2d taps
    source = {
        "backbone.pretrained.blocks.0.attn.qkv.weight": torch.randn(3 * dim, dim),
        "backbone.pretrained.blocks.1.attn.qkv.weight": torch.randn(3 * dim, dim),
        "cam_dec.backbone.0.weight": torch.randn(width, width),
        "cam_dec.backbone.0.bias": torch.randn(width),
        "cam_dec.backbone.2.weight": torch.randn(width, width),
        "cam_dec.backbone.2.bias": torch.randn(width),
        "cam_dec.fc_t.weight": torch.randn(3, width),
        "cam_dec.fc_t.bias": torch.randn(3),
        "cam_dec.fc_qvec.weight": torch.randn(4, width),
        "cam_dec.fc_qvec.bias": torch.randn(4),
        "cam_dec.fc_fov.0.weight": torch.randn(2, width),
        "cam_dec.fc_fov.0.bias": torch.randn(2),
    }
    shutil.rmtree(tmp_path, ignore_errors=True)
    os.makedirs(tmp_path)
    save_file({key: value.contiguous() for key, value in source.items()}, os.path.join(tmp_path, "model.safetensors"))

    torch.manual_seed(3)
    trained = build_l4ar(L4ARConfig(**cfg))
    assert apply_pretrained_init(trained, {"checkpoint": tmp_path}) == "4RC"
    trained.set_stage(3)
    with torch.no_grad():
        for name, param in trained.named_parameters():
            if "lora" in name or "camera_head" in name or "alignment" in name:
                param.add_(0.02)
    torch.manual_seed(11)
    sample = latent()
    trained.eval()
    expected = trained(sample)["points"]
    save_checkpoint(os.path.join(tmp_path, "stage3.pt"), trained, {}, {}, init_source="4RC")

    # the same seed reproduces the tensors the fixture leaves random, so only the init can differ here
    torch.manual_seed(3)
    restored = build_initialised_model({"model": dict(cfg), "pretrained_init": {"checkpoint": tmp_path}})
    report = load_checkpoint(restored, os.path.join(tmp_path, "stage3.pt"))
    assert not report["frozen_mismatch"], report
    assert torch.allclose(restored(sample)["points"], expected, atol=1e-5)

    torch.manual_seed(3)
    naive = build_l4ar(L4ARConfig(**cfg)).eval()  # the pre-fix evaluation path: heads restored, backbone random
    mismatched = load_checkpoint(naive, os.path.join(tmp_path, "stage3.pt"))
    assert mismatched["frozen_mismatch"], mismatched
    assert not torch.allclose(naive(sample)["points"], expected, atol=1e-5)


def test_checkpoint_load_reports_tensors_the_model_dropped(tmp_path: str = "/tmp/pixels_ckpt_drop"):
    """A trainable-only checkpoint restored into a changed architecture must say so out loud."""
    import os
    import shutil

    from l4d.utils.checkpoint import load_checkpoint, save_checkpoint

    net = build_l4ar(L4ARConfig(**TINY))
    net.set_stage(3)
    shutil.rmtree(tmp_path, ignore_errors=True)
    os.makedirs(tmp_path, exist_ok=True)
    path = os.path.join(tmp_path, "s.pt")
    save_checkpoint(path, net, {}, {})
    renamed = {f"renamed.{key}": value for key, value in torch.load(path, weights_only=False)["model"].items()}
    torch.save({"model": renamed, "model_kind": "trainable-only", "init_source": "random"}, path)
    report = load_checkpoint(build_l4ar(L4ARConfig(**TINY)), path)
    assert report.get("dropped_tensors"), report


if __name__ == "__main__":
    main()
