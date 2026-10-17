"""Architectural contract tests use feature fixtures, not pretend VLM weights."""

import hashlib
import json
import sys
from pathlib import Path
import tempfile
import unittest

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from choice_mem.backbones import FeatureBatch
from choice_mem.checkpoint import (
    CHECKPOINT_SCHEMA, LEGACY_CHECKPOINT_SCHEMA, LEGACY_METHOD, METHOD_FIELDS,
    load_checkpoint, read_checkpoint, save_checkpoint,
)
from choice_mem.config import PolicyConfig
from choice_mem.policy import ChoiceMemoryPolicy


class FeatureFixture(nn.Module):
    vision_dim = 24
    cognition_dim = 32

    def __init__(self):
        super().__init__()
        self.visual_projection = nn.Linear(3, self.vision_dim)
        self.cognition_projection = nn.Linear(3, self.cognition_dim)
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def forward(self, image_batches, instructions):
        batch_size = len(image_batches)
        visual_input = torch.arange(batch_size * 15 * 3, dtype=torch.float32).reshape(batch_size, 15, 3) / 100
        cognition_input = visual_input.mean(dim=1, keepdim=True)
        return FeatureBatch(
            self.visual_projection(visual_input),
            self.cognition_projection(cognition_input),
            ((1, 2, 3), (1, 3, 2), (1, 1, 3)),
        )


class PolicyCheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)

    def create_policy(self, backbone_name, head_name):
        config = PolicyConfig(
            backbone=backbone_name,
            action_head=head_name,
            model_source="feature-fixture",
            horizon=4,
            execute_prefix=2,
            group_size=3,
            per_token_size=16,
            oft_hidden_dim=32,
            mem_length=2,
            dtype="float32",
            cameras=["head_camera", "left_camera", "right_camera"],
            diffusion_model_type="DiT-S",
            mask_padding_loss=True,
            repeated_diffusion_steps=1,
        )
        return ChoiceMemoryPolicy(config, backbone=FeatureFixture())

    def test_four_combinations_shapes_gradients_and_explicit_reset(self):
        for backbone_name in ["memoryvla", "qwen35"]:
            for head_name in ["diffusion", "oft"]:
                with self.subTest(backbone=backbone_name, head=head_name):
                    policy = self.create_policy(backbone_name, head_name)
                    labels = torch.randn(3, 4, 14)
                    valid = torch.ones(3, 4, dtype=torch.bool)
                    valid[-1, -1] = False
                    labels[-1, -1] = float("nan")
                    # Original DiT starts with zero output/perception projections.
                    # Verify real optimization opens these paths, not a modified init.
                    if head_name == "diffusion":
                        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
                        for _warmup_step in range(3):
                            policy.reset()
                            optimizer.zero_grad(set_to_none=True)
                            warmup_loss = policy(
                                [[None] * 3] * 3, ["task"] * 3, labels, valid, ["episode"] * 3, [0, 2, 4]
                            )
                            warmup_loss.backward()
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        policy.reset()
                    loss = policy([[None] * 3] * 3, ["task"] * 3, labels, valid, ["episode"] * 3, [0, 2, 4])
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    for bank_name in ["cog_mem_bank", "per_mem_bank"]:
                        bank = getattr(policy, bank_name)
                        gradient = bank.retrieval_blocks[0].v_proj.weight.grad
                        self.assertIsNotNone(gradient)
                        self.assertGreater(gradient.abs().sum().item(), 0)
                        self.assertLessEqual(len(bank.bank["episode"]), 2)
                    self.assertGreater(policy.per_compr.reduce.weight.grad.abs().sum().item(), 0)
                    policy.eval().reset("episode")
                    with torch.inference_mode():
                        actions = policy.predict([None] * 3, "task", "episode")
                    self.assertEqual(actions.shape, (1, 4, 14))
                    with self.assertRaisesRegex(ValueError, "explicit reset"):
                        policy.predict([None] * 3, "task", "another-episode")
                    policy.reset("another-episode")
                    self.assertFalse(policy.cog_mem_bank.bank)
                    self.assertFalse(policy.per_mem_bank.bank)
                    self.assertEqual(policy.query_timestep, 0)

    def test_stage1_defaults_are_original_memoryvla_method(self):
        config = PolicyConfig()
        self.assertEqual((config.backbone, config.action_head, config.diffusion_model_type),
                         ("memoryvla", "diffusion", "DiT-L"))
        self.assertFalse(config.freeze_backbone)
        self.assertEqual(config.cameras, ["head_camera"])
        self.assertEqual((config.group_size, config.mem_length), (16, 16))
        self.assertEqual(config.history_sampling, "random")
        self.assertTrue(config.image_aug)
        self.assertEqual(config.repeated_diffusion_steps, 4)
        self.assertFalse(config.mask_padding_loss)
        self.assertEqual(config.inference_timestep_mode, "query")
        self.assertEqual(config.model_source, "")
        for invalid_config in (
            {"history_sampling": "balanced"}, {"repeated_diffusion_steps": 0},
            {"repeated_diffusion_steps": True}, {"image_aug": 1},
            {"mask_padding_loss": 0}, {"inference_timestep_mode": "wall-clock"},
        ):
            with self.subTest(invalid_config=invalid_config), self.assertRaises(ValueError):
                PolicyConfig(**invalid_config)

    def test_query_clock_is_separate_from_native_action_timestep(self):
        policy = self.create_policy("memoryvla", "oft").eval()
        for bank in (policy.cog_mem_bank, policy.per_mem_bank):
            bank.mem_length = 8
        with torch.inference_mode():
            for native_timestep in (0, 8, 40):
                policy.predict([None] * 3, "task", "episode", timestep=native_timestep)
        for bank in (policy.cog_mem_bank, policy.per_mem_bank):
            self.assertEqual([time for time, _ in bank.bank["episode"]], [0, 1, 2])
        policy.reset("episode")
        policy.config.inference_timestep_mode = "action"
        with torch.inference_mode():
            for native_timestep in (0, 8, 40):
                policy.predict([None] * 3, "task", "episode", timestep=native_timestep)
        for bank in (policy.cog_mem_bank, policy.per_mem_bank):
            self.assertEqual([time for time, _ in bank.bank["episode"]], [0, 8, 40])

    def test_checkpoint_method_fields_are_required_and_v1_migrates_explicitly(self):
        policy = self.create_policy("qwen35", "oft").eval()
        normalization = {"action": {"q01": [-1.0] * 14, "q99": [1.0] * 14}}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "policy.pt"
            save_checkpoint(checkpoint_path, policy, normalization, {}, step=2)
            payload = read_checkpoint(checkpoint_path)
            self.assertEqual(payload["schema"], CHECKPOINT_SCHEMA)
            for field_name in METHOD_FIELDS:
                incomplete = {**payload, "config": dict(payload["config"])}
                incomplete["config"].pop(field_name)
                incomplete_path = Path(directory) / f"missing-{field_name}.pt"
                torch.save(incomplete, incomplete_path)
                with self.subTest(field_name=field_name), self.assertRaisesRegex(
                    ValueError, "configuration missing fields"
                ):
                    read_checkpoint(incomplete_path)
            legacy = {**payload, "schema": LEGACY_CHECKPOINT_SCHEMA,
                      "config": {key: value for key, value in payload["config"].items() if key not in METHOD_FIELDS}}
            legacy_path = Path(directory) / "legacy.pt"
            torch.save(legacy, legacy_path)
            migrated, migrated_payload = load_checkpoint(legacy_path, backbone=FeatureFixture())
            for field_name, value in LEGACY_METHOD.items():
                self.assertEqual(migrated_payload["config"][field_name], value)
                self.assertEqual(getattr(migrated.config, field_name), value)
            roundtrip_path = Path(directory) / "migrated-v2.pt"
            save_checkpoint(roundtrip_path, migrated, normalization, {}, step=2)
            roundtrip = read_checkpoint(roundtrip_path)
            self.assertEqual(roundtrip["schema"], CHECKPOINT_SCHEMA)
            self.assertEqual(roundtrip["config"], migrated_payload["config"])

    def test_checkpoint_exact_prediction_and_rejects_architecture_mismatch(self):
        policy = self.create_policy("qwen35", "oft").eval()
        normalization = {"action": {"q01": [-1.0] * 14, "q99": [1.0] * 14, "mask": [True] * 14}}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "policy.pt"
            save_checkpoint(checkpoint_path, policy, normalization, {"label_offset": 1}, step=2)
            reloaded, payload = load_checkpoint(checkpoint_path, backbone=FeatureFixture())
            with torch.inference_mode():
                for query_timestep in [0, 2, 4]:
                    original_actions = policy.predict([None] * 3, "task", "episode", timestep=query_timestep)
                    reloaded_actions = reloaded.predict([None] * 3, "task", "episode", timestep=query_timestep)
                    torch.testing.assert_close(original_actions, reloaded_actions, rtol=0, atol=0)
            self.assertEqual(payload["step"], 2)
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                load_checkpoint(checkpoint_path, expected={"action_head": "diffusion"}, backbone=FeatureFixture())
            with self.assertRaises(FileExistsError):
                save_checkpoint(checkpoint_path, policy, normalization, {}, step=3)
            incomplete_payload = dict(payload)
            incomplete_payload["config"] = dict(payload["config"])
            incomplete_payload["config"].pop("execute_prefix")
            incomplete_path = Path(directory) / "incomplete.pt"
            torch.save(incomplete_payload, incomplete_path)
            with self.assertRaisesRegex(ValueError, "configuration missing fields"):
                load_checkpoint(incomplete_path, backbone=FeatureFixture())


@pytest.fixture
def debug_policy(tmp_path, monkeypatch):
    """Keep the real debug DiT-L contract; mock only the expensive head builder."""
    from choice_mem.precision import prepare_training_precision
    from choice_mem.random_init import inspect_random_backbone_assets

    metadata = tmp_path / "llama"
    metadata.mkdir()
    files = {}
    for name, value in {
        "config.json": {"model_type": "llama"},
        "tokenizer_config.json": {"tokenizer_class": "LlamaTokenizerFast"},
        "tokenizer.json": {},
    }.items():
        content = json.dumps(value).encode()
        (metadata / name).write_bytes(content)
        files[name] = hashlib.sha256(content).hexdigest()
    source = tmp_path / "random-assets.json"
    source.write_text(json.dumps({"llama_metadata": {"path": str(metadata), "files": files}}))
    configuration = PolicyConfig(
        backbone_initialization="random-frozen-debug", backbone_initialization_seed=17,
        model_source=str(source), freeze_backbone=True, dtype="bfloat16",
        per_token_size=16, mem_length=2, group_size=3,
        asset_manifest=inspect_random_backbone_assets(source)["asset_manifest"],
    )
    monkeypatch.setattr("choice_mem.policy.DiffusionHead", lambda **kwargs: nn.Linear(32, 14))
    backbone = FeatureFixture().requires_grad_(False).to(torch.bfloat16)
    policy = ChoiceMemoryPolicy(configuration, backbone=backbone)
    return prepare_training_precision(policy).eval()


def test_debug_checkpoint_roundtrip_never_reads_pretrained_weights_and_preserves_storage(debug_policy, tmp_path, monkeypatch):
    normalization = {"action": {"q01": [-1.0] * 14, "q99": [1.0] * 14}}
    protocol = {"requested_configuration": debug_policy.config.to_dict()}
    original_protocol = json.loads(json.dumps(protocol))
    # Make saved frozen values distinct from anything reconstructed from a seed.
    with torch.no_grad():
        debug_policy.backbone.visual_projection.weight.fill_(0.3125)
    checkpoint = tmp_path / "debug.pt"
    save_checkpoint(checkpoint, debug_policy, normalization, protocol, step=2)
    assert protocol == original_protocol
    payload = read_checkpoint(checkpoint)
    assert payload["weight_compatibility"]["reused"].startswith("None;")
    assert "backbone" in payload["weight_compatibility"]["initialized"]
    assert payload["weight_compatibility"]["backbone_initialization_seed"] == 17
    constructions = []

    def construct(configuration, initialize_from_config=False):
        assert initialize_from_config is True
        constructions.append(configuration)
        return FeatureFixture().requires_grad_(False).to(torch.bfloat16)

    monkeypatch.setattr("choice_mem.policy.MemoryVLABackbone", construct)
    original_load = torch.load
    loaded_paths = []

    def only_our_checkpoint(path, *args, **kwargs):
        assert Path(path) == checkpoint, "Loading any pretrained weight is forbidden"
        loaded_paths.append(Path(path))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", only_our_checkpoint)
    with pytest.raises(ValueError, match="explicit allow_debug_checkpoint"):
        load_checkpoint(checkpoint)
    assert not constructions
    restored, _ = load_checkpoint(checkpoint, allow_debug_checkpoint=True)
    assert len(constructions) == 1 and loaded_paths == [checkpoint, checkpoint]
    for name, tensor in restored.state_dict().items():
        torch.testing.assert_close(tensor, debug_policy.state_dict()[name], rtol=0, atol=0)
        if tensor.is_floating_point():
            assert tensor.dtype == (torch.bfloat16 if name.startswith("backbone.") else torch.float32)
    assert not any(parameter.requires_grad for parameter in restored.backbone.parameters())
    with pytest.raises(ValueError, match="configuration mismatch"):
        load_checkpoint(checkpoint, expected={"backbone_initialization_seed": 18}, allow_debug_checkpoint=True)
    (tmp_path / "llama" / "config.json").write_text('{"model_type":"llama","changed":true}')
    with pytest.raises(ValueError, match="SHA256"):
        load_checkpoint(checkpoint, allow_debug_checkpoint=True)
    assert len(constructions) == 1, "Metadata must be checked before any model construction"


@pytest.mark.parametrize("field,value", [
    ("backbone_initialization", "pretrained"), ("backbone_initialization_seed", 18),
    ("base_vlm_id", "different"), ("dtype", "float32"), ("freeze_backbone", False),
])
def test_debug_checkpoint_refuses_protocol_and_requested_configuration_drift(debug_policy, tmp_path, field, value):
    checkpoint = tmp_path / "bad.pt"
    normalization = {"action": {"q01": [-1.0] * 14, "q99": [1.0] * 14}}
    for protocol in ({field: value}, {"requested_configuration": {**debug_policy.config.to_dict(), field: value}}):
        with pytest.raises(ValueError, match=field):
            save_checkpoint(checkpoint, debug_policy, normalization, protocol, step=2)
    assert not checkpoint.exists()


@pytest.mark.parametrize("mutation,reason", [
    (lambda p: p["model"].pop("backbone.visual_projection.weight"), "Missing key"),
    (lambda p: p["model"].update({"backbone.visual_projection.weight": torch.zeros(24, 3)}), "dtype"),
    (lambda p: p["model"].update({"action_head.weight": torch.zeros(14, 32, dtype=torch.bfloat16)}), "dtype"),
    (lambda p: p["model"]["backbone.visual_projection.weight"].fill_(float("nan")), "nonfinite"),
    (lambda p: p["config"]["asset_manifest"].update(checkpoint="a" * 64), "asset_manifest"),
])
def test_debug_load_rejects_incomplete_weights_wrong_precision_and_asset_drift(debug_policy, tmp_path, mutation, reason):
    path = tmp_path / "debug.pt"
    save_checkpoint(path, debug_policy, {"action": {"q01": [-1.0] * 14, "q99": [1.0] * 14}}, {}, step=2)
    payload = read_checkpoint(path)
    mutation(payload)
    torch.save(payload, path)
    with pytest.raises((ValueError, RuntimeError), match=reason):
        load_checkpoint(path, backbone=FeatureFixture().requires_grad_(False), allow_debug_checkpoint=True)


def test_historical_v1_v2_missing_initialization_fields_remain_pretrained(tmp_path):
    policy = PolicyCheckpointTests().create_policy("qwen35", "oft")
    normalization = {"action": {"q01": [-1.0] * 14, "q99": [1.0] * 14}}
    checkpoint = tmp_path / "current.pt"
    save_checkpoint(checkpoint, policy, normalization, {}, step=2)
    current = read_checkpoint(checkpoint)
    for schema in (CHECKPOINT_SCHEMA, LEGACY_CHECKPOINT_SCHEMA):
        payload = {**current, "schema": schema, "config": dict(current["config"]), "protocol": {}}
        payload.pop("weight_compatibility")
        for field in ("backbone_initialization", "backbone_initialization_seed", "base_vlm_id"):
            payload["config"].pop(field)
        if schema == LEGACY_CHECKPOINT_SCHEMA:
            for field in METHOD_FIELDS:
                payload["config"].pop(field)
        path = tmp_path / f"{schema}.pt"
        torch.save(payload, path)
        restored, migrated = load_checkpoint(path, backbone=FeatureFixture())
        assert restored.config.backbone_initialization == "pretrained"
        assert migrated["config"]["backbone_initialization_seed"] == 42
        assert migrated["config"]["base_vlm_id"] == "prism-dinosiglip-224px+7b"


if __name__ == "__main__":
    unittest.main()
