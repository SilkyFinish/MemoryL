"""CPU-only regressions for the distributed full-finetune training contract.

The small named policy is a numerical fixture, not a fake official Qwen model.
No official weights, native dataset, CUDA context, or GPU training is loaded.
"""

import copy
import json
import hashlib
import random
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import train_multitask
from choice_mem.checkpoint import CHECKPOINT_SCHEMA
from choice_mem.config import PolicyConfig


_MODEL_SOURCE = "/unused/local-memoryvla"


@pytest.fixture(autouse=True)
def local_original_assets(tmp_path, monkeypatch):
    source = tmp_path / "original-memoryvla"
    (source / "checkpoints").mkdir(parents=True)
    (source / "config.json").write_text(json.dumps({"model": {"model_id": "original-prismatic"}}))
    (source / "checkpoints/latest-checkpoint.pt").write_bytes(b"metadata fixture; never loaded")
    monkeypatch.setitem(globals(), "_MODEL_SOURCE", str(source))


@pytest.fixture
def random_metadata_manifest(tmp_path):
    metadata = tmp_path / "llama-metadata"
    metadata.mkdir()
    contents = {
        "config.json": json.dumps({"model_type": "llama"}),
        "tokenizer_config.json": json.dumps({"tokenizer_class": "LlamaTokenizerFast"}),
        "tokenizer.json": json.dumps({"fixture": True}),
    }
    for name, content in contents.items():
        (metadata / name).write_text(content)
    manifest = tmp_path / "random-metadata.json"
    manifest.write_text(json.dumps({
        "llama_metadata": {"path": str(metadata), "files": {
            name: hashlib.sha256(content.encode()).hexdigest() for name, content in contents.items()
        }},
        # These deliberately missing pretrained assets must never be required.
        "checkpoint_path": str(tmp_path / "missing-33GB-checkpoint.pt"),
        "local_vision_files": {"vision": str(tmp_path / "missing-vision-weights.bin")},
    }))
    return manifest


class ToyPolicy(nn.Module):
    def __init__(self, stochastic=False, unused=False):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.model = nn.Module()
        self.backbone.model.visual = nn.Module()
        self.backbone.model.visual.blocks = nn.ModuleList([nn.Linear(2, 2)])
        self.backbone.model.language_model = nn.Module()
        self.backbone.model.language_model.layers = nn.ModuleList([nn.Linear(2, 2)])
        self.per_compr = nn.Linear(2, 2)
        self.per_mem_bank = nn.Linear(2, 2)
        self.cog_mem_bank = nn.Linear(2, 2)
        self.action_head = nn.Linear(2, 14)
        if unused:
            self.backbone.model.unused_projection = nn.Linear(2, 2)
        self.config = PolicyConfig(
            model_source="cpu-numerical-fixture",
            backbone="qwen35",
            action_head="diffusion",
            horizon=4,
            execute_prefix=4,
            group_size=16,
            freeze_backbone=False,
        )
        self.stochastic = stochastic
        self.reset_count = 0
        self.bank_empty = True

    def reset(self):
        self.reset_count += 1
        self.bank_empty = True

    def parameter_counts(self):
        return {
            "total": sum(parameter.numel() for parameter in self.parameters()),
            "trainable": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
        }

    def forward(self, images, instructions, actions, valid_mask, episode_ids, timesteps):
        assert self.bank_empty, "State must not leak between chronological groups"
        self.bank_empty = False
        features = self.per_compr(torch.as_tensor(images, dtype=torch.float32))
        visual = self.backbone.model.visual.blocks[0](features)
        language = self.backbone.model.language_model.layers[0](features)
        # A within-group history dependency distinguishes groups from shuffled queries.
        history = visual.cumsum(dim=0) / torch.arange(1, len(images) + 1).unsqueeze(1)
        prediction = self.action_head(self.per_mem_bank(history) + self.cog_mem_bank(language))
        prediction = prediction.unsqueeze(1).expand_as(actions)
        if self.stochastic:
            prediction = (
                prediction + torch.randn_like(prediction) * 0.01 + random.random() * 0.01 + np.random.random() * 0.01
            )
        action_mask = valid_mask.unsqueeze(-1).expand_as(actions)
        return (prediction - actions).square()[action_mask].mean()


class ToyGroups:
    def __init__(self, full_counts=(2, 7), tail_size=3):
        self.groups = []
        for task_index, full_count in enumerate(full_counts):
            record = {"task": f"task_{task_index}/demo_clean", "episode_id": f"episode_{task_index}", "split": "train"}
            for group_number in range(full_count):
                self.groups.append((record, list(range(group_number * 64, (group_number + 1) * 64, 4))))
            if tail_size:
                self.groups.append((record, list(range(full_count * 64, full_count * 64 + tail_size * 4, 4))))

    def __len__(self):
        return len(self.groups)

    def get_group(self, group_index, seed=None):
        return self[group_index]

    def __getitem__(self, group_index):
        record, timesteps = self.groups[group_index]
        features = np.column_stack((np.asarray(timesteps) / 100, np.full(len(timesteps), 0.3))).astype(np.float32)
        return {
            "images": features,
            "instructions": ["fixture"] * len(timesteps),
            "actions": np.full((len(timesteps), 4, 14), 0.15, dtype=np.float32),
            "valid_mask": np.ones((len(timesteps), 4), dtype=bool),
            "episode_ids": [record["episode_id"]] * len(timesteps),
            "timesteps": timesteps,
        }


class ToyRandomGroups:
    """Episode candidate metadata, stateless random queries and numeric augmentation."""

    def __init__(self, episode_counts=(1, 3), short=False):
        self.groups = []
        self.observed_seeds = []
        for task_index, episode_count in enumerate(episode_counts):
            for episode_index in range(episode_count):
                record = {"task": f"task_{task_index}/demo_clean", "episode_id": f"episode_{task_index}_{episode_index}", "split": "train"}
                length = 3 if short and episode_index == 0 else 83 + episode_index
                self.groups.append((record, list(range(length))))

    def get_group(self, index, seed=None):
        self.observed_seeds.append(seed)
        record, candidates = self.groups[index]
        generator = np.random.default_rng(seed)
        queries = sorted(generator.choice(candidates, size=min(len(candidates), 16), replace=False).tolist())
        queries += [queries[-1]] * (16 - len(queries))
        features = np.column_stack((np.asarray(queries) / 100, np.full(16, 0.3))).astype(np.float32)
        features += generator.uniform(-0.01, 0.01, size=features.shape).astype(np.float32)
        return {"images": features, "instructions": ["fixture"] * 16, "actions": np.full((16, 4, 14), 0.15, dtype=np.float32),
                "valid_mask": np.ones((16, 4), dtype=bool), "episode_ids": [record["episode_id"]] * 16, "timesteps": queries}


class RecordingDDP:
    def __init__(self, policy):
        self.policy = policy
        self.inside_no_sync = False
        self.forward_sync_states = []

    @contextmanager
    def no_sync(self):
        assert not self.inside_no_sync
        self.inside_no_sync = True
        try:
            yield
        finally:
            self.inside_no_sync = False

    def __call__(self, *arguments):
        self.forward_sync_states.append(self.inside_no_sync)
        return self.policy(*arguments)


@pytest.fixture(autouse=True)
def keep_cpu_threads_small():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous_threads)


def fixture_arguments(**overrides):
    arguments = train_multitask.build_parser().parse_args(
        [
            "--data-root",
            "/unused/native",
            "--manifest",
            "/unused/manifest.json",
            "--run-dir",
            "/unused/run",
            "--model-path", _MODEL_SOURCE, "--max-steps", "30000",
            "--optimization-recipe", overrides.get("optimization_recipe", "memoryvla"),
            "--backbone", "qwen35" if overrides.get("optimization_recipe") == "qwen-legacy" else "memoryvla",
        ]
    )
    for field_name, field_value in overrides.items():
        setattr(arguments, field_name, field_value)
    return arguments


def test_formal_defaults_and_exact_batch_validation():
    arguments = fixture_arguments()
    train_multitask.validate_arguments(arguments)
    assert (arguments.backbone, arguments.action_head, arguments.unfreeze_backbone) == ("memoryvla", "diffusion", True)
    assert (arguments.group_size, arguments.grad_accumulation, arguments.global_batch_size) == (16, 1, None)
    assert (arguments.max_steps, arguments.save_every, arguments.snapshot_every) == (30000, 1000, 5000)
    assert arguments.cpu_threads == 2
    assert not arguments.gradient_checkpointing
    assert arguments.find_unused_parameters
    assert train_multitask.validate_global_batch(7, 16, 4, 448) == 448
    with pytest.raises(ValueError, match="Exact global batch mismatch"):
        train_multitask.validate_global_batch(8, 16, 4, 448)
    with pytest.raises(ValueError, match="capped preflight"):
        train_multitask.validate_arguments(fixture_arguments(stop_after=3))
    train_multitask.validate_arguments(fixture_arguments(tasks=["one"]))
    with pytest.raises(ValueError, match="distinct"):
        train_multitask.validate_arguments(fixture_arguments(tasks=["one", "one"]))
    with pytest.raises(ValueError, match="explicit --max-steps"):
        train_multitask.validate_arguments(fixture_arguments(max_steps=None))
    assert arguments.history_sampling == "random" and arguments.image_aug
    assert arguments.cameras == ["head_camera"]
    assert (arguments.learning_rate, arguments.weight_decay, arguments.gradient_clip_norm) == (2e-5, 0, 1)
    assert arguments.diffusion_model_type == "DiT-L"
    assert (arguments.backbone_initialization, arguments.backbone_initialization_seed, arguments.base_vlm_id) == (
        "pretrained", 42, "prism-dinosiglip-224px+7b",
    )
    assert (arguments.repeated_diffusion_steps, arguments.mask_padding_loss, arguments.inference_timestep_mode) == (4, False, "query")


@pytest.mark.parametrize("action_head", ["oft", "diffusion"])
@pytest.mark.parametrize("cameras", [["head_camera"], ["head_camera", "right_camera"], ["head_camera", "left_camera", "right_camera"]])
def test_head_and_camera_choices_reach_policy_configuration_independently(action_head, cameras):
    arguments = fixture_arguments(action_head=action_head, cameras=cameras)
    train_multitask.validate_arguments(arguments)
    configuration = train_multitask.build_policy_configuration(arguments, {"protocol": {"image_size": [224, 224]}})
    assert configuration.action_head == action_head
    assert configuration.cameras == cameras
    with pytest.raises(ValueError, match="unique"):
        train_multitask.validate_arguments(fixture_arguments(cameras=["head_camera", "head_camera"]))


def test_full_group_sampler_excludes_tails_and_uses_exact_rank_stream():
    dataset = ToyGroups()
    sampler = train_multitask.FullGroupTaskSampler(dataset, 16, 42)
    assert sampler.counts["task_0"]["full_groups"] == 2
    assert sampler.counts["task_1"]["full_groups"] == 7
    assert sampler.counts["task_0"]["excluded_tail_queries"] == 3
    for completed_steps in [0, 1, 2, 1000]:
        for accumulation_index in range(4):
            for rank in range(7):
                generator = np.random.default_rng(42 + (completed_steps * 4 + accumulation_index) * 7 + rank)
                task_name = sampler.tasks[int(generator.integers(2))]
                task_indices = sampler.indices_by_task[task_name]
                expected_index = task_indices[int(generator.integers(len(task_indices)))]
                sampled_index = sampler.sample_index(completed_steps, accumulation_index, 4, rank, 7)
                assert sampled_index == expected_index
                assert len(dataset.groups[sampled_index][1]) == 16
    # Group-count imbalance must not change the equal task probability.
    sampled_tasks = [
        train_multitask.canonical_task(dataset.groups[sampler.sample_index(step, 0, 4, 0, 7)][0]) for step in range(4000)
    ]
    assert 0.46 < sampled_tasks.count("task_0") / len(sampled_tasks) < 0.54
    with pytest.raises(ValueError, match="unselected tasks"):
        train_multitask.FullGroupTaskSampler(dataset, 16, 42, ["task_0"])
    with pytest.raises(ValueError, match="at least one full"):
        train_multitask.FullGroupTaskSampler(ToyGroups(full_counts=(0, 2)), 16, 42)


def test_cluster_oft_preserves_optimizer_schedule_and_single_group_batch():
    arguments = fixture_arguments(
        optimization_recipe="qwen-legacy", action_head="oft", per_device_batch_size=16, grad_accumulation=1,
        global_batch_size=128, max_steps=55000, gradient_checkpointing=True,
        save_every=5000, snapshot_every=5000,
    )
    train_multitask.validate_arguments(arguments)
    assert train_multitask.validate_global_batch(8, 16, 1, arguments.global_batch_size) == 128
    assert arguments.oft_hidden_dim == 256
    assert arguments.backbone_learning_rate == 1e-5
    assert arguments.learning_rate == 1e-4
    assert arguments.warmup_ratio == 0.05
    assert arguments.weight_decay == 0.005
    with pytest.raises(ValueError, match="Exact global batch mismatch"):
        train_multitask.validate_global_batch(8, 16, 1, 64)
    train_multitask.validate_arguments(fixture_arguments(action_head="oft", unfreeze_backbone=False))


def test_float32_master_parameters_and_disjoint_learning_rates():
    policy = ToyPolicy().bfloat16()
    arguments = fixture_arguments(optimization_recipe="qwen-legacy")
    with pytest.raises(ValueError, match="Float32 master"):
        train_multitask.build_optimizer_groups(policy, arguments)
    policy.float()
    groups = train_multitask.build_optimizer_groups(policy, arguments)
    assert [group["lr"] for group in groups] == [1e-5, 1e-4]
    backbone_ids = {id(parameter) for parameter in groups[0]["params"]}
    head_ids = {id(parameter) for parameter in groups[1]["params"]}
    assert not backbone_ids.intersection(head_ids)
    assert backbone_ids.union(head_ids) == {id(parameter) for parameter in policy.parameters()}
    policy.backbone.model.visual.blocks[0].weight.requires_grad_(False)
    remaining = train_multitask.build_optimizer_groups(policy, arguments)
    assert id(policy.backbone.model.visual.blocks[0].weight) not in {id(p) for group in remaining for p in group["params"]}


def test_schedule_keeps_formal_horizon_across_two_step_preflight_and_resume():
    policy = ToyPolicy()
    arguments = fixture_arguments(stop_after=2, optimization_recipe="qwen-legacy")
    optimizer = torch.optim.AdamW(train_multitask.build_optimizer_groups(policy, arguments))
    scheduler = train_multitask.CosineWarmupSchedule(optimizer, arguments.max_steps, 0.05, [1e-6, 5e-6])
    assert scheduler.warmup_steps == 1500
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5 / 1500)
    scheduler.advance()
    scheduler.advance()
    state = scheduler.state_dict()
    assert state["total_steps"] == 30000
    assert state["completed_steps"] == 2
    restored_optimizer = torch.optim.AdamW(train_multitask.build_optimizer_groups(ToyPolicy(), arguments))
    restored_scheduler = train_multitask.CosineWarmupSchedule(restored_optimizer, 30000, 0.05, [1e-6, 5e-6])
    restored_scheduler.load_state_dict(state)
    assert restored_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    state["completed_steps"] = 1499
    restored_scheduler.load_state_dict(state)
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)
    state["completed_steps"] = 29999
    restored_scheduler.load_state_dict(state)
    assert [group["lr"] for group in restored_optimizer.param_groups] == pytest.approx([1e-6, 5e-6])
    with pytest.raises(ValueError, match="scheduler mismatch"):
        restored_scheduler.load_state_dict({**state, "total_steps": 2})


def test_accumulation_no_sync_encloses_forward_and_resets_every_group():
    torch.manual_seed(12)
    policy = ToyPolicy()
    reference = copy.deepcopy(policy)
    dataset = ToyGroups()
    sampler = train_multitask.FullGroupTaskSampler(dataset, 16, 42)
    arguments = fixture_arguments(gradient_clip_norm=1e6, grad_accumulation=4)
    wrapper = RecordingDDP(policy)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)
    result = train_multitask.perform_optimizer_step(
        wrapper,
        policy,
        optimizer,
        dataset,
        sampler,
        arguments,
        0,
        0,
        1,
        torch.device("cpu"),
        True,
    )
    assert wrapper.forward_sync_states == [True, True, True, False]
    assert policy.reset_count == 8
    assert policy.bank_empty
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    expected_loss = 0.0
    for micro_index in range(4):
        group = dataset[sampler.sample_index(0, micro_index, 4, 0, 1)]
        reference.reset()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = reference(
                group["images"],
                group["instructions"],
                torch.from_numpy(group["actions"]),
                torch.from_numpy(group["valid_mask"]),
                group["episode_ids"],
                group["timesteps"],
            )
        expected_loss += float(loss.detach()) / 4
        (loss / 4).backward()
    reference_optimizer.step()
    assert float(result["loss_tensor"]) == pytest.approx(expected_loss)
    for actual_parameter, expected_parameter in zip(policy.parameters(), reference.parameters()):
        torch.testing.assert_close(actual_parameter, expected_parameter, atol=0, rtol=0)


def test_rng_roundtrip_is_weights_only_safe_and_preserves_all_cpu_generators(tmp_path):
    random.seed(8)
    np.random.seed(9)
    torch.manual_seed(10)
    state = train_multitask.capture_rng_state()
    checkpoint_path = tmp_path / "rng.pt"
    torch.save(state, checkpoint_path)
    expected = (random.random(), np.random.random(4), torch.rand(4))
    restored_state = torch.load(checkpoint_path, weights_only=True)
    train_multitask.restore_rng_state(restored_state)
    assert random.random() == expected[0]
    np.testing.assert_array_equal(np.random.random(4), expected[1])
    torch.testing.assert_close(torch.rand(4), expected[2], atol=0, rtol=0)


@pytest.mark.parametrize("random_frozen", [False, True])
def test_optimizer_scheduler_rng_and_sampler_resume_are_numerically_exact(random_frozen):
    torch.manual_seed(16)
    random.seed(17)
    np.random.seed(18)
    arguments = fixture_arguments(
        grad_accumulation=2,
        max_steps=10,
        backbone_learning_rate=1e-2,
        learning_rate=1e-1,
        minimum_backbone_learning_rate=1e-3,
        minimum_head_learning_rate=5e-3,
    )
    dataset = ToyRandomGroups(short=True)
    sampler = train_multitask.RandomEpisodeSampler(dataset, 16, 42)
    def create_policy():
        from choice_mem.precision import prepare_training_precision

        policy = ToyPolicy(stochastic=True)
        if random_frozen:
            policy.config = PolicyConfig(model_source="cpu-metadata-fixture", backbone_initialization="random-frozen-debug", freeze_backbone=True)
            policy.backbone.requires_grad_(False)
        return prepare_training_precision(policy)

    policy = create_policy()
    frozen_weights = copy.deepcopy(policy.backbone.state_dict()) if random_frozen else None
    optimizer = torch.optim.AdamW(train_multitask.build_optimizer_groups(policy, arguments), weight_decay=0.005)
    scheduler = train_multitask.build_schedule(optimizer, arguments)
    for _update in range(2):
        train_multitask.perform_optimizer_step(
            RecordingDDP(policy),
            policy,
            optimizer,
            dataset,
            sampler,
            arguments,
            scheduler.completed_steps,
            0,
            1,
            torch.device("cpu"),
        )
        scheduler.advance()
    checkpoint = copy.deepcopy(
        {
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "schedule": scheduler.state_dict(),
            "rng": train_multitask.capture_rng_state(),
        }
    )
    uninterrupted_result = train_multitask.perform_optimizer_step(
        RecordingDDP(policy),
        policy,
        optimizer,
        dataset,
        sampler,
        arguments,
        2,
        0,
        1,
        torch.device("cpu"),
    )
    scheduler.advance()
    restored_policy = create_policy()
    restored_optimizer = torch.optim.AdamW(
        train_multitask.build_optimizer_groups(restored_policy, arguments), weight_decay=0.005
    )
    restored_schedule = train_multitask.build_schedule(restored_optimizer, arguments)
    restored_policy.load_state_dict(checkpoint["model"])
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    restored_schedule.load_state_dict(checkpoint["schedule"])
    train_multitask.restore_rng_state(checkpoint["rng"])
    restored_result = train_multitask.perform_optimizer_step(
        RecordingDDP(restored_policy),
        restored_policy,
        restored_optimizer,
        dataset,
        sampler,
        arguments,
        restored_schedule.completed_steps,
        0,
        1,
        torch.device("cpu"),
    )
    restored_schedule.advance()
    torch.testing.assert_close(uninterrupted_result["loss_tensor"], restored_result["loss_tensor"], atol=0, rtol=0)
    for parameter_name, expected_tensor in policy.state_dict().items():
        torch.testing.assert_close(restored_policy.state_dict()[parameter_name], expected_tensor, atol=0, rtol=0)
    assert restored_schedule.state_dict() == scheduler.state_dict()
    if random_frozen:
        for name, weight in policy.backbone.state_dict().items():
            assert weight.dtype == torch.bfloat16
            torch.testing.assert_close(weight, frozen_weights[name], atol=0, rtol=0)
        optimized = {id(parameter) for group in restored_optimizer.param_groups for parameter in group["params"]}
        assert optimized == {id(parameter) for parameter in restored_policy.parameters() if parameter.requires_grad}
        assert all(parameter.dtype == torch.float32 for parameter in restored_policy.parameters() if parameter.requires_grad)


def test_zero_initialized_connected_gradients_differ_from_missing_gradients():
    torch.manual_seed(20)
    policy = ToyPolicy(unused=True)
    probes = train_multitask.capture_weight_probes(policy)
    with torch.no_grad():
        policy.action_head.weight.zero_()
    dataset = ToyGroups()
    sampler = train_multitask.FullGroupTaskSampler(dataset, 16, 42)
    result = train_multitask.perform_optimizer_step(
        RecordingDDP(policy),
        policy,
        torch.optim.AdamW(policy.parameters(), lr=0.01),
        dataset,
        sampler,
        fixture_arguments(),
        0,
        0,
        1,
        torch.device("cpu"),
        True,
    )
    diagnostics = result["gradient_diagnostics"]
    assert diagnostics["backbone_vision"]["missing_gradient_names"] == []
    assert diagnostics["backbone_vision"]["gradient_norm"] == 0
    assert len(diagnostics["backbone_other"]["missing_gradient_names"]) == 2
    changes = train_multitask.compare_weight_probes(policy, probes)
    assert set(changes) == {"backbone_vision", "backbone_language"}
    # Decay can change weights despite zero loss-driven gradients; do not conflate them.
    assert changes["backbone_vision"]["changed_elements"] > 0
    with pytest.warns(RuntimeWarning):
        train_multitask.warn_gradient_diagnostics(diagnostics, changes)


def create_resume_payload():
    configuration = ToyPolicy().config.to_dict()
    protocol = {"optimization": {"grad_accumulation": 4}}
    state = {
        "schema": train_multitask.TRAINING_STATE_SCHEMA,
        "rank_rng_states": [{}, {}],
        "scheduler": {"completed_steps": 2},
        "next_global_micro_index": 16,
        "weight_probes": {"backbone_vision": {}, "backbone_language": {}},
    }
    return {
        "schema": CHECKPOINT_SCHEMA,
        "config": configuration,
        "protocol": protocol,
        "normalization": {"action": {}},
        "optimizer": {},
        "step": 2,
        "training_state": state,
    }


@pytest.mark.parametrize(
    "alteration,error_pattern",
    [
        (lambda payload: payload.pop("optimizer"), "cannot resume"),
        (lambda payload: payload["protocol"]["optimization"].update(grad_accumulation=8), "protocol differs"),
        (lambda payload: payload["config"].update(freeze_backbone=True), "configuration mismatch"),
        (lambda payload: payload["training_state"].update(rank_rng_states=[{}]), "world size"),
        (lambda payload: payload["training_state"].update(next_global_micro_index=15), "sampler position"),
    ],
)
def test_resume_rejects_inference_wrong_protocol_and_wrong_sampler_state(alteration, error_pattern):
    payload = create_resume_payload()
    configuration = copy.deepcopy(payload["config"])
    protocol = copy.deepcopy(payload["protocol"])
    train_multitask.validate_resume_payload(payload, configuration, protocol, payload["normalization"], 2)
    alteration(payload)
    with pytest.raises(ValueError, match=error_pattern):
        train_multitask.validate_resume_payload(payload, configuration, protocol, payload["normalization"], 2)


def test_checkpoint_retention_only_deletes_explicit_owned_files(tmp_path):
    checkpoint_directory = tmp_path / "checkpoints"
    checkpoint_directory.mkdir()
    owned_names = [f"checkpoint-step{step:06d}.pt" for step in [1000, 2000, 3000]]
    for checkpoint_name in owned_names:
        (checkpoint_directory / checkpoint_name).write_bytes(b"fixture")
    foreign_path = checkpoint_directory / "someone-else.pt"
    foreign_path.write_bytes(b"preserve")
    assert train_multitask.prune_owned_checkpoints(tmp_path, owned_names) == owned_names[-2:]
    assert not (checkpoint_directory / owned_names[0]).exists()
    assert foreign_path.read_bytes() == b"preserve"
    with pytest.raises(ValueError, match="not explicitly owned"):
        train_multitask.prune_owned_checkpoints(tmp_path, ["../someone-else.pt"])
    symlink_name = "checkpoint-step004000.pt"
    (checkpoint_directory / symlink_name).symlink_to(foreign_path)
    with pytest.raises(ValueError, match="not explicitly owned"):
        train_multitask.prune_owned_checkpoints(tmp_path, [symlink_name, *owned_names[-2:]])


def test_task_preparation_passes_filter_to_parent_api(monkeypatch):
    observed_arguments = {}

    def prepare_fixture(data_root, manifest_path, *, tasks=None, **keywords):
        observed_arguments.update(tasks=tasks, **keywords)
        return {"fixture": True}

    monkeypatch.setattr(train_multitask, "prepare_manifest", prepare_fixture)
    selected_tasks = [f"task_{index}" for index in range(10)]
    arguments = fixture_arguments(tasks=selected_tasks)
    assert train_multitask.prepare_training_manifest(arguments) == {"fixture": True}
    assert observed_arguments["tasks"] == selected_tasks
    assert observed_arguments["group_size"] == 16


class CheckpointableVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Linear(2, 2)
        self.checkpointing = False

    def forward(self, features):
        if self.checkpointing:
            from torch.utils.checkpoint import checkpoint

            return checkpoint(self.blocks, features, use_reentrant=False)
        return self.blocks(features)


def test_nonreentrant_layer_checkpointing_does_not_repeat_outer_visual_capture():
    model = nn.Module()
    model.visual = CheckpointableVision()
    model.config = SimpleNamespace(use_cache=True, text_config=SimpleNamespace(use_cache=True))
    observed_options = {}

    def enable_fixture(gradient_checkpointing_kwargs):
        observed_options.update(gradient_checkpointing_kwargs)
        model.visual.checkpointing = True

    model.gradient_checkpointing_enable = enable_fixture
    policy = SimpleNamespace(backbone=SimpleNamespace(model=model))
    train_multitask.enable_gradient_checkpointing(policy)
    captured_outputs = []
    hook = model.visual.register_forward_hook(lambda _module, _arguments, output: captured_outputs.append(output))
    try:
        model.visual(torch.ones(4, 2)).square().mean().backward()
    finally:
        hook.remove()
    assert observed_options == {"use_reentrant": False}
    assert not model.config.use_cache
    assert not model.config.text_config.use_cache
    assert len(captured_outputs) == 1
    assert model.visual.blocks.weight.grad is not None


def test_save_routes_full_training_state_through_shared_helper_and_retains_two(tmp_path, monkeypatch):
    import choice_mem.checkpoint as checkpoint_module
    import torch.distributed as distributed

    (tmp_path / "checkpoints").mkdir()
    calls = []
    shared_save_checkpoint = checkpoint_module.save_checkpoint

    def save_fixture(path, policy, normalization, protocol, step, optimizer=None, training_state=None):
        assert not Path(path).exists()
        calls.append({"path": Path(path), "step": step, "optimizer": optimizer, "state": training_state})
        shared_save_checkpoint(
            path, policy, normalization, protocol, step, optimizer=optimizer, training_state=training_state
        )

    monkeypatch.setattr(checkpoint_module, "save_checkpoint", save_fixture)
    monkeypatch.setattr(distributed, "barrier", lambda: None)
    monkeypatch.setattr(
        distributed, "all_gather_object", lambda states, local_state: states.__setitem__(slice(None), [local_state])
    )
    monkeypatch.setattr(train_multitask, "rank_zero_operation", lambda runtime, operation: operation())
    train_multitask.ensure_training_state_support()
    runtime = train_multitask.DistributedRuntime(0, 0, 1, torch.device("cpu"))
    policy = ToyPolicy()
    optimizer = torch.optim.AdamW(train_multitask.build_optimizer_groups(policy, fixture_arguments(optimization_recipe="qwen-legacy")))
    scheduler = train_multitask.CosineWarmupSchedule(optimizer, 30000, 0.05, [1e-6, 5e-6])
    probes = train_multitask.capture_weight_probes(policy)
    manifest = {"normalization": {"action": {"q01": [-1.0] * 14, "q99": [1.0] * 14}}}
    protocol = {"optimization": {"grad_accumulation": 4}}
    owned_names = []
    for step in [2, 1000, 2000]:
        state = scheduler.state_dict()
        state["completed_steps"] = step
        scheduler.load_state_dict(state)
        owned_names, checkpoint_path = train_multitask.save_training_checkpoint(
            tmp_path,
            policy,
            optimizer,
            scheduler,
            manifest,
            protocol,
            runtime,
            owned_names,
            probes,
            {"task_0": 8},
            12.0,
        )
        saved_payload = checkpoint_module.read_checkpoint(checkpoint_path)
        saved_state = saved_payload["training_state"]
        assert saved_payload["schema"] == CHECKPOINT_SCHEMA
        assert all(weight.dtype == torch.float32 for weight in saved_payload["model"].values())
        assert saved_state["scheduler"]["total_steps"] == 30000
        assert saved_state["next_global_micro_index"] == step * 4
        assert len(saved_state["rank_rng_states"]) == 1
    assert len(list((tmp_path / "checkpoints").glob("*.pt"))) == 2
    assert len(calls) == 3
    assert all(call["optimizer"] is optimizer for call in calls)
    assert json.loads((tmp_path / "latest_checkpoint.json").read_text())["step"] == 2000


@pytest.mark.parametrize("random_frozen", [False, True])
def test_preflight_resume_progress_and_final_snapshot_with_cpu_fixture(tmp_path, monkeypatch, random_frozen, random_metadata_manifest):
    import choice_mem.policy as policy_module
    import choice_mem.checkpoint as checkpoint_module
    import torch.distributed as distributed
    import torch.nn.parallel as parallel

    dataset = ToyRandomGroups(episode_counts=(1, 2), short=True)
    records = [record for record, _ in dataset.groups]
    manifest = {
        "episodes": records,
        "manifest_sha256": "a" * 64,
        "data_root": str(tmp_path),
        "split_seed": 42,
        "validation_fraction": 0.1,
        "codec": {"sha256": "fixture"},
        "protocol": {"image_size": [224, 224], "group_size": 16, "execute_prefix": 4, "horizon": 4, "action_dim": 14, "target_offset": 1, "cameras": ["head_camera"]},
        "normalization": {
            "action_dim": 14, "target_offset": 1,
            "training_episode_ids": [record["episode_id"] for record in records],
            "action": {"q01": [-1.0] * 14, "q99": [1.0] * 14},
        },
    }

    def create_fixture(configuration, initialize_from_config=False):
        policy = ToyPolicy(stochastic=True)
        policy.config = configuration
        if configuration.backbone_initialization == "random-frozen-debug":
            from choice_mem.random_init import inspect_random_backbone_assets

            configuration.asset_manifest = inspect_random_backbone_assets(configuration.model_source)["asset_manifest"]
            policy.backbone.requires_grad_(False)
        else:
            source = Path(configuration.model_source)
            configuration.asset_manifest = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in ["config.json", "checkpoints/latest-checkpoint.pt"]}
        configuration.model_revision = configuration.model_revision or "local-sha256"
        return policy

    class LocalDDP(RecordingDDP):
        def __init__(self, policy, **keywords):
            super().__init__(policy)

    def gather_fixture(states, local_state):
        states[0] = local_state

    monkeypatch.setattr(train_multitask, "build_training_dataset", lambda *arguments, **keywords: dataset)
    monkeypatch.setattr(policy_module, "ChoiceMemoryPolicy", create_fixture)
    monkeypatch.setattr(parallel, "DistributedDataParallel", LocalDDP)
    monkeypatch.setattr(train_multitask, "rank_zero_operation", lambda runtime, operation: operation())
    monkeypatch.setattr(distributed, "barrier", lambda: None)
    monkeypatch.setattr(distributed, "all_gather_object", gather_fixture)
    monkeypatch.setattr(distributed, "all_reduce", lambda *arguments, **keywords: None)
    for function_name in ("synchronize", "reset_peak_memory_stats", "manual_seed"):
        monkeypatch.setattr(torch.cuda, function_name, lambda *arguments, **keywords: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *arguments: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *arguments: 0)
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "snapshots").mkdir()
    runtime = train_multitask.DistributedRuntime(0, 0, 1, torch.device("cpu"))
    maximum_steps, stopping_steps = (3, 1) if random_frozen else (4, 2)
    arguments = fixture_arguments(run_dir=tmp_path, global_batch_size=64, grad_accumulation=4, horizon=4,
                                  max_steps=maximum_steps, stop_after=stopping_steps)
    if random_frozen:
        arguments.backbone_initialization = "random-frozen-debug"
        arguments.unfreeze_backbone = False
        arguments.model_path = str(random_metadata_manifest)
    stopped = train_multitask.run_training(arguments, manifest, runtime, tmp_path)
    assert (stopped["status"], stopped["completed_steps"]) == ("stopped", stopping_steps)
    checkpoint_payload = checkpoint_module.read_checkpoint(stopped["checkpoint"])
    assert checkpoint_payload["training_state"]["scheduler"]["total_steps"] == maximum_steps
    assert checkpoint_payload["training_state"]["next_global_micro_index"] == stopping_steps * 4
    assert not (tmp_path / "snapshots" / "final.pt").exists()
    arguments.resume = "latest"
    arguments.stop_after = None
    completed = train_multitask.run_training(arguments, manifest, runtime, tmp_path)
    assert (completed["status"], completed["completed_steps"]) == ("completed", maximum_steps)
    assert completed["evaluation_status"] == "not_run"
    assert completed["gradient_diagnostics_step"] == 3
    assert all(changes["changed_elements"] > 0 for changes in completed["weight_changes"].values())
    snapshot_payload = checkpoint_module.read_checkpoint(completed["final_snapshot"])
    assert "optimizer" not in snapshot_payload
    assert "training_state" not in snapshot_payload
    assert snapshot_payload["step"] == maximum_steps
    from eval_multitask import validate_checkpoint
    gate_options = {"allow_debug_checkpoint": True} if random_frozen else {}
    validate_checkpoint(Path(completed["final_snapshot"]), expected_step=maximum_steps, expected_cameras=("head_camera",), **gate_options)
    validate_checkpoint(Path(completed["checkpoint"]), expected_step=maximum_steps, expected_cameras=("head_camera",), **gate_options)
    if random_frozen:
        assert snapshot_payload["protocol"]["optimization"]["frozen_backbone_dtype"] == "bfloat16"
        for name, weight in snapshot_payload["model"].items():
            assert weight.dtype == (torch.bfloat16 if name.startswith("backbone.") else torch.float32)
        with pytest.raises(ValueError, match="[Dd]ebug|random-frozen-debug"):
            validate_checkpoint(Path(completed["final_snapshot"]), expected_step=maximum_steps, expected_cameras=("head_camera",))
    assert json.loads((tmp_path / "progress.json").read_text())["status"] == "completed"
    original_progress = (tmp_path / "progress.json").read_bytes()
    with pytest.raises(ValueError, match="already reached"):
        train_multitask.run_training(arguments, manifest, runtime, tmp_path)
    assert (tmp_path / "progress.json").read_bytes() == original_progress
    arguments.learning_rate *= 2
    with pytest.raises(ValueError, match="protocol differs"):
        train_multitask.run_training(arguments, manifest, runtime, tmp_path)
    assert (tmp_path / "progress.json").read_bytes() == original_progress


def _run_gloo_rank(rank, rendezvous_path, output_directory):
    """Actual CPU DDP proves accumulation normalization across multiple ranks."""
    import torch.distributed as distributed
    from torch.nn.parallel import DistributedDataParallel

    torch.set_num_threads(2)
    distributed.init_process_group("gloo", init_method=f"file://{rendezvous_path}", rank=rank, world_size=2)
    try:
        torch.manual_seed(31)
        policy = ToyPolicy(unused=True)
        wrapper = DistributedDataParallel(policy, find_unused_parameters=True)
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)
        dataset = ToyRandomGroups(short=True)
        sampler = train_multitask.RandomEpisodeSampler(dataset, 16, 42)
        train_multitask.perform_optimizer_step(
            wrapper,
            policy,
            optimizer,
            dataset,
            sampler,
            fixture_arguments(grad_accumulation=2, gradient_clip_norm=1e6),
            0,
            rank,
            2,
            torch.device("cpu"),
        )
        torch.save(policy.state_dict(), Path(output_directory) / f"rank-{rank}.pt")
    finally:
        distributed.destroy_process_group()


def test_real_two_rank_cpu_ddp_matches_all_rank_microbatch_mean(tmp_path):
    import torch.multiprocessing as multiprocessing

    multiprocessing.spawn(_run_gloo_rank, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True)
    first_rank = torch.load(tmp_path / "rank-0.pt", weights_only=True)
    second_rank = torch.load(tmp_path / "rank-1.pt", weights_only=True)
    torch.manual_seed(31)
    reference = ToyPolicy(unused=True)
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    dataset = ToyRandomGroups(short=True)
    sampler = train_multitask.RandomEpisodeSampler(dataset, 16, 42)
    for rank in range(2):
        for accumulation_index in range(2):
            group = dataset.get_group(sampler.sample_index(0, accumulation_index, 2, rank, 2), seed=train_multitask.microstep_seed(42, 0, accumulation_index, rank))
            reference.reset()
            with torch.autocast("cpu", dtype=torch.bfloat16):
                loss = reference(
                    group["images"],
                    group["instructions"],
                    torch.from_numpy(group["actions"]),
                    torch.from_numpy(group["valid_mask"]),
                    group["episode_ids"],
                    group["timesteps"],
                )
            (loss / 4).backward()
    optimizer.step()
    for parameter_name, reference_tensor in reference.state_dict().items():
        torch.testing.assert_close(first_rank[parameter_name], second_rank[parameter_name], atol=0, rtol=0)
        torch.testing.assert_close(first_rank[parameter_name], reference_tensor, atol=2e-7, rtol=2e-7)


def test_global64_uses_one_original_group_per_rank_without_accumulation():
    class BatchRecordingPolicy(ToyPolicy):
        def __init__(self):
            super().__init__()
            self.observed_batch_sizes = []

        def forward(self, images, instructions, actions, valid_mask, episode_ids, timesteps):
            self.observed_batch_sizes.append(len(images))
            assert len(images) == 16
            assert len(set(episode_ids)) == 1
            assert np.diff(timesteps).tolist() == [4] * 15
            return super().forward(images, instructions, actions, valid_mask, episode_ids, timesteps)

    policy = BatchRecordingPolicy()
    wrapper = RecordingDDP(policy)
    dataset = ToyGroups()
    sampler = train_multitask.FullGroupTaskSampler(dataset, 16, 42)
    arguments = fixture_arguments(per_device_batch_size=16, grad_accumulation=1, global_batch_size=64,
                                  max_steps=55000, diffusion_model_type="DiT-L")
    train_multitask.validate_arguments(arguments)
    assert train_multitask.validate_global_batch(4, 16, 1, 64) == 64
    result = train_multitask.perform_optimizer_step(
        wrapper, policy, torch.optim.SGD(policy.parameters(), lr=0.01), dataset, sampler,
        arguments, 0, 0, 4, torch.device("cpu"),
    )
    assert policy.observed_batch_sizes == [16]
    assert wrapper.forward_sync_states == [False]
    assert sum(result["sampled_task_groups"].values()) == 1
    assert policy.bank_empty
    with pytest.raises(ValueError, match="original 16-query"):
        train_multitask.validate_arguments(fixture_arguments(per_device_batch_size=64))


def test_batched_memory_groups_reset_even_with_repeated_episode_ids():
    from choice_mem.banks import CogMemBank, PerMemBank

    torch.manual_seed(42)
    for bank_class in (CogMemBank, PerMemBank):
        bank = bank_class(dataloader_type="group", group_size=16, token_size=16).train()
        independent_bank = copy.deepcopy(bank)
        tokens = torch.randn(64, 3, 16)
        episode_ids = ["same-episode"] * 64
        timesteps = list(range(0, 64, 4)) * 4
        batched_output = bank.process_batch(tokens, episode_ids, timesteps)
        independent_outputs = []
        for group_start in range(0, 64, 16):
            independent_bank.reset()
            independent_outputs.append(independent_bank.process_batch(
                tokens[group_start:group_start + 16], episode_ids[:16], timesteps[:16],
            ))
        torch.testing.assert_close(batched_output, torch.cat(independent_outputs), rtol=0, atol=0)
        assert len(bank.bank["same-episode"]) == 16



def test_episode_sampler_weights_episodes_not_tasks_and_keeps_short_episodes():
    dataset = ToyRandomGroups(episode_counts=(1, 3), short=True)
    sampler = train_multitask.RandomEpisodeSampler(dataset, 16, 42)
    sampled = [train_multitask.canonical_task(dataset.groups[sampler.sample_index(step, 0, 1, 0, 2)][0]) for step in range(6000)]
    assert 0.22 < sampled.count("task_0") / len(sampled) < 0.28
    assert len(sampler.episodes) == 4
    balanced = train_multitask.RandomEpisodeSampler(dataset, 16, 42, task_balanced=True)
    selected = [train_multitask.canonical_task(dataset.groups[balanced.sample_index(step, 0, 1, 0, 2)][0]) for step in range(4000)]
    assert 0.46 < selected.count("task_0") / len(selected) < 0.54
    group = dataset.get_group(0, seed=23)
    assert group["timesteps"] == [0, 1, 2] + [2] * 13
    train_multitask.validate_loaded_group(group, 16)
    group["timesteps"] = [0, 0] + list(range(1, 15))
    with pytest.raises(ValueError, match="Only repeated final"):
        train_multitask.validate_loaded_group(group, 16)
    group["timesteps"] = list(range(15, -1, -1))
    with pytest.raises(ValueError, match="sorted"):
        train_multitask.validate_loaded_group(group, 16)


def test_original_single_lr_bias_norm_no_decay_constant_schedule_and_freeze_optout():
    policy = ToyPolicy()
    policy.normalization = nn.LayerNorm(2)
    arguments = fixture_arguments(weight_decay=0.1)
    groups = train_multitask.build_optimizer_groups(policy, arguments)
    assert [group["lr"] for group in groups] == [2e-5, 2e-5]
    assert [group["weight_decay"] for group in groups] == [0.1, 0]
    names_by_id = {id(parameter): name for name, parameter in policy.named_parameters()}
    for parameter in groups[1]["params"]:
        assert parameter.ndim <= 1 or "norm" in names_by_id[id(parameter)] or names_by_id[id(parameter)].endswith(".bias")
    policy.backbone.requires_grad_(False)
    frozen_groups = train_multitask.build_optimizer_groups(policy, arguments)
    assert not any(id(parameter) in {id(p) for p in policy.backbone.parameters()} for group in frozen_groups for parameter in group["params"])
    assert train_multitask.capture_weight_probes(policy) == {}
    optimizer = torch.optim.AdamW(frozen_groups)
    schedule = train_multitask.build_schedule(optimizer, arguments)
    for _ in range(3):
        optimizer.step()
        schedule.advance()
    assert [group["lr"] for group in optimizer.param_groups] == [2e-5, 2e-5]
    restored = train_multitask.ConstantSchedule(optimizer, arguments.max_steps)
    restored.load_state_dict(schedule.state_dict())
    assert restored.completed_steps == 3
    with pytest.raises(ValueError, match="scheduler mismatch"):
        restored.load_state_dict({**schedule.state_dict(), "total_steps": 2})


def test_every_microstep_passes_rank_step_and_accumulation_seed_to_history_and_augmentation():
    dataset = ToyRandomGroups(short=True)
    sampler = train_multitask.RandomEpisodeSampler(dataset, 16, 42)
    policy = ToyPolicy()
    arguments = fixture_arguments(grad_accumulation=3)
    train_multitask.perform_optimizer_step(RecordingDDP(policy), policy, torch.optim.SGD(policy.parameters(), lr=0.001),
                                          dataset, sampler, arguments, 19, 1, 2, torch.device("cpu"))
    expected = [train_multitask.microstep_seed(42, 19, index, 1) for index in range(3)]
    assert dataset.observed_seeds == expected
    assert len(set(expected + [train_multitask.microstep_seed(42, 19, index, 0) for index in range(3)])) == 6


def test_memoryvla_checkpointing_dispatches_to_backbone_adapter():
    calls = []
    policy = SimpleNamespace(backbone=SimpleNamespace(enable_gradient_checkpointing=lambda: calls.append(True)))
    train_multitask.enable_gradient_checkpointing(policy)
    assert calls == [True]



@pytest.mark.parametrize("random_frozen", [False, True])
def test_single_process_checkpoint_rng_sampler_resume_matches_uninterrupted_cpu_training(tmp_path, monkeypatch, random_frozen, random_metadata_manifest):
    import train_rmbench
    import choice_mem.policy as policy_module
    from choice_mem.checkpoint import read_checkpoint
    from eval_multitask import validate_checkpoint

    dataset = ToyRandomGroups(episode_counts=(1, 2), short=True)
    records = [record for record, _ in dataset.groups]
    manifest = {
        "episodes": records, "manifest_sha256": "b" * 64, "data_root": str(tmp_path),
        "split_seed": 42, "validation_fraction": 0.1, "codec": {"sha256": "fixture"},
        "protocol": {"image_size": [224, 224], "group_size": 16, "execute_prefix": 4, "horizon": 4,
                     "action_dim": 14, "target_offset": 1, "cameras": ["head_camera"]},
        "normalization": {"action_dim": 14, "target_offset": 1, "training_episode_ids": [record["episode_id"] for record in records],
                          "action": {"q01": [-1.0] * 14, "q99": [1.0] * 14}},
    }

    def fixture_policy(configuration, initialize_from_config=False):
        policy = ToyPolicy(stochastic=True)
        policy.config = configuration
        if configuration.backbone_initialization == "random-frozen-debug":
            from choice_mem.random_init import inspect_random_backbone_assets

            configuration.asset_manifest = inspect_random_backbone_assets(configuration.model_source)["asset_manifest"]
            policy.backbone.requires_grad_(False)
        else:
            source = Path(configuration.model_source)
            configuration.asset_manifest = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in ["config.json", "checkpoints/latest-checkpoint.pt"]}
        configuration.model_revision = configuration.model_revision or "local-sha256"
        return policy

    monkeypatch.setattr(train_rmbench, "build_training_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(policy_module, "ChoiceMemoryPolicy", fixture_policy)
    arguments = fixture_arguments(run_dir=tmp_path / "bounded", device="cpu", max_steps=3, stop_after=1, horizon=4)
    if random_frozen:
        arguments.backbone_initialization = "random-frozen-debug"
        arguments.unfreeze_backbone = False
        arguments.model_path = str(random_metadata_manifest)
    stopped = train_rmbench.run_training(arguments, manifest)
    assert (stopped["status"], stopped["completed_steps"]) == ("stopped", 1)
    assert not (arguments.run_dir / "snapshot-final.pt").exists()
    state = read_checkpoint(stopped["checkpoint"])["training_state"]
    assert state["scheduler"]["total_steps"] == 3
    assert state["next_global_micro_index"] == 1
    arguments.resume = "latest"
    arguments.stop_after = None
    resumed = train_rmbench.run_training(arguments, manifest)
    assert (resumed["status"], resumed["completed_steps"]) == ("completed", 3)
    gate_options = {"allow_debug_checkpoint": True} if random_frozen else {}
    validate_checkpoint(Path(resumed["checkpoint"]), expected_step=3, expected_cameras=("head_camera",), **gate_options)
    validate_checkpoint(Path(resumed["final_snapshot"]), expected_step=3, expected_cameras=("head_camera",), **gate_options)
    resumed_weights = read_checkpoint(resumed["checkpoint"])["model"]
    original_metrics = (arguments.run_dir / "metrics.json").read_bytes()
    with pytest.raises(ValueError, match="already reached"):
        train_rmbench.run_training(arguments, manifest)
    assert (arguments.run_dir / "metrics.json").read_bytes() == original_metrics
    arguments.run_dir = tmp_path / "uninterrupted"
    arguments.resume = None
    uninterrupted = train_rmbench.run_training(arguments, manifest)
    expected_weights = read_checkpoint(uninterrupted["checkpoint"])["model"]
    assert [step["loss"] for step in resumed["steps"]] == [step["loss"] for step in uninterrupted["steps"]]
    for name, weight in expected_weights.items():
        torch.testing.assert_close(resumed_weights[name], weight, atol=0, rtol=0)



def test_distributed_prepare_only_uses_random_augmented_manifest_without_importing_torch(tmp_path, monkeypatch):
    import builtins
    import h5py
    from choice_mem.data import load_image_codec

    root = tmp_path / "native"
    episode = root / "one_task/demo_clean/data/episode0.hdf5"
    episode.parent.mkdir(parents=True)
    actions = np.arange(6 * 14, dtype=np.float64).reshape(6, 14) / 100
    jpeg = load_image_codec().encode_image_bit(np.zeros((8, 10, 3), dtype=np.uint8))
    with h5py.File(episode, "w") as handle:
        handle.create_dataset("joint_action/vector", data=actions)
        for name, values in (("left_arm", actions[:, :6]), ("left_gripper", actions[:, 6]),
                             ("right_arm", actions[:, 7:13]), ("right_gripper", actions[:, 13])):
            handle.create_dataset("joint_action/" + name, data=values)
        handle.create_dataset("language_instruction", data="move the block")
        handle.create_dataset("observation/head_camera/rgb", data=np.asarray([jpeg] * len(actions), dtype=f"S{len(jpeg)}"))
    original_import = builtins.__import__

    def forbid_models(name, *args, **kwargs):
        if name.startswith(("torch", "transformers", "choice_mem.policy", "choice_mem.checkpoint", "choice_mem.backbones")):
            raise AssertionError("Preparation imported model code: " + name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_models)
    path = tmp_path / "new-stage1-manifest.json"
    summary = train_multitask.main(["--prepare-only", "--data-root", str(root), "--manifest", str(path), "--tasks", "one_task"])
    assert summary["train_episodes"] == 1
    protocol = json.loads(path.read_text())["protocol"]
    assert protocol["history_sampling"] == "random" and protocol["image_aug"] is True
    assert protocol["cameras"] == ["head_camera"] and protocol["group_size"] == 16


def test_original_backbone_never_silently_accepts_qwen_or_remote_default_assets(tmp_path):
    import train_rmbench

    for source in (None, "Qwen/Qwen3.5-0.8B", "/missing/original-memoryvla"):
        with pytest.raises(ValueError, match="model-path"):
            train_rmbench.validate_model_source("memoryvla", source)
    qwen = tmp_path / "qwen"
    qwen.mkdir()
    (qwen / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    with pytest.raises(ValueError, match="Original MemoryVLA"):
        train_rmbench.validate_model_source("memoryvla", str(qwen))
    assert train_rmbench.validate_model_source("qwen35", "Qwen/Qwen3.5-0.8B") == "Qwen/Qwen3.5-0.8B"


def test_training_accepts_upstream_vla_checkpoint_without_confusing_it_with_vlm_run(tmp_path):
    import train_rmbench

    run = tmp_path / "vla-run"
    checkpoint_directory = run / "checkpoints"
    checkpoint_directory.mkdir(parents=True)
    checkpoint = checkpoint_directory / "latest-checkpoint.pt"
    checkpoint.write_bytes(b"metadata fixture; validation never loads weights")
    with pytest.raises(ValueError, match="run/config.json"):
        train_rmbench.validate_model_source("memoryvla", str(checkpoint))
    config_path = run / "config.json"
    config_path.write_text(json.dumps({"vla": {"base_vlm": "original-prismatic"}}))
    assert train_rmbench.validate_model_source("memoryvla", str(checkpoint)) == str(checkpoint)
    # A VLA run directory is not the Prismatic VLM run format expected by vla.load.load.
    with pytest.raises(ValueError, match="VLM run needs"):
        train_rmbench.validate_model_source("memoryvla", str(run))
    config_path.write_text(json.dumps({"vla": {}}))
    with pytest.raises(ValueError, match="vla.base_vlm"):
        train_rmbench.validate_model_source("memoryvla", str(checkpoint))
    config_path.write_text(json.dumps({"model": {"model_id": "original-prismatic"}}))
    assert train_rmbench.validate_model_source("memoryvla", str(run)) == str(run)


def test_random_debug_metadata_only_source_and_configuration(random_metadata_manifest, monkeypatch):
    import builtins
    import train_rmbench

    original_import = builtins.__import__

    def forbid_models(name, *args, **kwargs):
        if name.startswith(("torch", "transformers", "choice_mem.policy", "choice_mem.backbones", "choice_mem.checkpoint")):
            raise AssertionError("Metadata-only validation imported model code: " + name)
        return original_import(name, *args, **kwargs)

    arguments = fixture_arguments(
        backbone_initialization="random-frozen-debug", backbone_initialization_seed=123,
        unfreeze_backbone=False, model_path=str(random_metadata_manifest),
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(builtins, "__import__", forbid_models)
        train_multitask.validate_arguments(arguments)
        configuration = train_multitask.build_policy_configuration(arguments, {"protocol": {"image_size": [224, 224]}})
        assert train_rmbench.validate_model_source("memoryvla", str(random_metadata_manifest), "random-frozen-debug") == str(random_metadata_manifest)
    assert configuration.backbone_initialization == "random-frozen-debug"
    assert configuration.backbone_initialization_seed == 123
    assert configuration.base_vlm_id == "prism-dinosiglip-224px+7b"
    assert configuration.freeze_backbone and configuration.dtype == "bfloat16"
    with pytest.raises(ValueError, match="asset manifest"):
        train_rmbench.validate_model_source("memoryvla", str(random_metadata_manifest))


@pytest.mark.parametrize("conflict", [
    {"unfreeze_backbone": True}, {"backbone": "qwen35"}, {"action_head": "oft"},
    {"diffusion_model_type": "DiT-S"}, {"dtype": "float32"}, {"base_vlm_id": "another-vlm"},
    {"backbone_initialization_seed": -1}, {"backbone_initialization_seed": 2**63},
    {"gradient_checkpointing": True},
])
def test_random_debug_conflicts_are_rejected_without_silent_changes(random_metadata_manifest, conflict):
    arguments = fixture_arguments(
        backbone_initialization="random-frozen-debug", unfreeze_backbone=False,
        model_path=str(random_metadata_manifest),
    )
    vars(arguments).update(conflict)
    before = copy.deepcopy(vars(arguments))
    with pytest.raises(ValueError, match="random-frozen-debug|backbone_initialization_seed|conflicts"):
        train_multitask.validate_arguments(arguments)
    assert vars(arguments) == before


def test_frozen_backbone_checkpointing_is_rejected_even_for_pretrained():
    arguments = fixture_arguments(unfreeze_backbone=False, gradient_checkpointing=True)
    with pytest.raises(ValueError, match="conflicts with --freeze-backbone"):
        train_multitask.validate_arguments(arguments)
    policy = ToyPolicy()
    policy.config.freeze_backbone = True
    with pytest.raises(ValueError, match="frozen backbone"):
        train_multitask.enable_gradient_checkpointing(policy)


def test_legacy_pretrained_resume_defaults_only_new_provenance_fields():
    payload = create_resume_payload()
    configuration = copy.deepcopy(payload["config"])
    protocol = copy.deepcopy(payload["protocol"])
    protocol.update(train_multitask.INITIALIZATION_DEFAULTS)
    protocol["requested_configuration"] = copy.deepcopy(configuration)
    payload["protocol"]["requested_configuration"] = copy.deepcopy(configuration)
    for field in train_multitask.INITIALIZATION_DEFAULTS:
        payload["config"].pop(field)
        payload["protocol"]["requested_configuration"].pop(field)
    original = copy.deepcopy(payload)
    train_multitask.validate_resume_payload(payload, configuration, protocol, payload["normalization"], 2)
    assert payload == original
    for field, value in {
        "backbone_initialization": "random-frozen-debug",
        "backbone_initialization_seed": 123,
        "base_vlm_id": "different-vlm",
    }.items():
        changed = copy.deepcopy(configuration)
        changed[field] = value
        with pytest.raises(ValueError, match="configuration mismatch"):
            train_multitask.validate_resume_payload(payload, changed, protocol, payload["normalization"], 2)
        changed_protocol = copy.deepcopy(protocol)
        changed_protocol[field] = value
        with pytest.raises(ValueError, match="protocol differs"):
            train_multitask.validate_resume_payload(payload, configuration, changed_protocol, payload["normalization"], 2)
