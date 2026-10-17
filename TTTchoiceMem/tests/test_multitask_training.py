"""CPU-only regressions for the distributed full-finetune training contract.

The small named policy is a numerical fixture, not a fake official Qwen model.
No official weights, native dataset, CUDA context, or GPU training is loaded.
"""

import copy
import json
import random
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import train_multitask
from choice_mem.config import PolicyConfig


class ToyPolicy(nn.Module):
    def __init__(self, stochastic=False, unused=False):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.model = nn.Module()
        self.backbone.model.visual = nn.Module()
        self.backbone.model.visual.blocks = nn.ModuleList([nn.Linear(2, 2)])
        self.backbone.model.language_model = nn.Module()
        self.backbone.model.language_model.layers = nn.ModuleList([nn.Linear(2, 2)])
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
        features = torch.as_tensor(images, dtype=torch.float32)
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
        ]
    )
    for field_name, field_value in overrides.items():
        setattr(arguments, field_name, field_value)
    return arguments


def test_formal_defaults_and_exact_batch_validation():
    arguments = fixture_arguments()
    train_multitask.validate_arguments(arguments)
    assert (arguments.backbone, arguments.action_head, arguments.unfreeze_backbone) == ("qwen35", "diffusion", True)
    assert (arguments.group_size, arguments.grad_accumulation, arguments.global_batch_size) == (16, 4, 448)
    assert (arguments.max_steps, arguments.save_every, arguments.snapshot_every) == (30000, 1000, 5000)
    assert arguments.cpu_threads == 2
    assert not arguments.gradient_checkpointing
    assert arguments.find_unused_parameters
    assert train_multitask.validate_global_batch(7, 16, 4, 448) == 448
    with pytest.raises(ValueError, match="Exact global batch mismatch"):
        train_multitask.validate_global_batch(8, 16, 4, 448)
    with pytest.raises(ValueError, match="capped preflight"):
        train_multitask.validate_arguments(fixture_arguments(stop_after=3))
    with pytest.raises(ValueError, match="ten distinct"):
        train_multitask.validate_arguments(fixture_arguments(tasks=["one"]))


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
        action_head="oft", per_device_batch_size=16, grad_accumulation=1,
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
    with pytest.raises(ValueError, match="unfrozen"):
        train_multitask.validate_arguments(fixture_arguments(action_head="oft", unfreeze_backbone=False))


def test_float32_master_parameters_and_disjoint_learning_rates():
    policy = ToyPolicy().bfloat16()
    arguments = fixture_arguments()
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
    with pytest.raises(ValueError, match="frozen parameter"):
        train_multitask.build_optimizer_groups(policy, arguments)


def test_schedule_keeps_formal_horizon_across_two_step_preflight_and_resume():
    policy = ToyPolicy()
    arguments = fixture_arguments(stop_after=2)
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
    arguments = fixture_arguments(gradient_clip_norm=1e6)
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


def test_optimizer_scheduler_rng_and_sampler_resume_are_numerically_exact():
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
    dataset = ToyGroups()
    sampler = train_multitask.FullGroupTaskSampler(dataset, 16, 42)
    policy = ToyPolicy(stochastic=True)
    optimizer = torch.optim.AdamW(train_multitask.build_optimizer_groups(policy, arguments), weight_decay=0.005)
    scheduler = train_multitask.CosineWarmupSchedule(optimizer, 10, 0.2, [1e-3, 5e-3])
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
    restored_policy = ToyPolicy(stochastic=True)
    restored_optimizer = torch.optim.AdamW(
        train_multitask.build_optimizer_groups(restored_policy, arguments), weight_decay=0.005
    )
    restored_schedule = train_multitask.CosineWarmupSchedule(restored_optimizer, 10, 0.2, [1e-3, 5e-3])
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
        "schema": "choice-memory-rmbench-v1",
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
    optimizer = torch.optim.AdamW(train_multitask.build_optimizer_groups(policy, fixture_arguments()))
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
        assert saved_payload["schema"] == "choice-memory-rmbench-v1"
        assert all(weight.dtype == torch.float32 for weight in saved_payload["model"].values())
        assert saved_state["scheduler"]["total_steps"] == 30000
        assert saved_state["next_global_micro_index"] == step * 4
        assert len(saved_state["rank_rng_states"]) == 1
    assert len(list((tmp_path / "checkpoints").glob("*.pt"))) == 2
    assert len(calls) == 3
    assert all(call["optimizer"] is optimizer for call in calls)
    assert json.loads((tmp_path / "latest_checkpoint.json").read_text())["step"] == 2000


def test_preflight_resume_progress_and_final_snapshot_with_cpu_fixture(tmp_path, monkeypatch):
    import choice_mem.policy as policy_module
    import choice_mem.checkpoint as checkpoint_module
    import torch.distributed as distributed
    import torch.nn.parallel as parallel

    dataset = ToyGroups(full_counts=(1,) * 10)
    records = [record for record, timesteps in dataset.groups if len(timesteps) == 16]
    manifest = {
        "episodes": records,
        "manifest_sha256": "cpu-fixture-sha",
        "data_root": str(tmp_path),
        "split_seed": 42,
        "validation_fraction": 0.1,
        "codec": {"sha256": "fixture"},
        "protocol": {"image_size": [224, 224], "group_size": 16, "execute_prefix": 4, "horizon": 4},
        "normalization": {
            "training_episode_ids": [record["episode_id"] for record in records],
            "action": {"q01": [-1.0] * 14, "q99": [1.0] * 14},
        },
    }

    def create_fixture(configuration, initialize_from_config=False):
        policy = ToyPolicy(stochastic=True)
        policy.config = configuration
        return policy

    class LocalDDP(RecordingDDP):
        def __init__(self, policy, **keywords):
            super().__init__(policy)

    def gather_fixture(states, local_state):
        states[0] = local_state

    monkeypatch.setattr(train_multitask, "RMBenchSequentialGroups", lambda *arguments, **keywords: dataset)
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
    arguments = fixture_arguments(run_dir=tmp_path, global_batch_size=64, horizon=4, max_steps=4, stop_after=2)
    stopped = train_multitask.run_training(arguments, manifest, runtime, tmp_path)
    assert (stopped["status"], stopped["completed_steps"]) == ("stopped", 2)
    checkpoint_payload = checkpoint_module.read_checkpoint(stopped["checkpoint"])
    assert checkpoint_payload["training_state"]["scheduler"]["total_steps"] == 4
    assert checkpoint_payload["training_state"]["next_global_micro_index"] == 8
    assert not (tmp_path / "snapshots" / "final.pt").exists()
    arguments.resume = "latest"
    arguments.stop_after = None
    completed = train_multitask.run_training(arguments, manifest, runtime, tmp_path)
    assert (completed["status"], completed["completed_steps"]) == ("completed", 4)
    assert completed["evaluation_status"] == "not_run"
    assert completed["gradient_diagnostics_step"] == 3
    assert all(changes["changed_elements"] > 0 for changes in completed["weight_changes"].values())
    snapshot_payload = checkpoint_module.read_checkpoint(completed["final_snapshot"])
    assert "optimizer" not in snapshot_payload
    assert "training_state" not in snapshot_payload
    assert snapshot_payload["step"] == 4
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
        dataset = ToyGroups()
        sampler = train_multitask.FullGroupTaskSampler(dataset, 16, 42)
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
    dataset = ToyGroups()
    sampler = train_multitask.FullGroupTaskSampler(dataset, 16, 42)
    for rank in range(2):
        for accumulation_index in range(2):
            group = dataset[sampler.sample_index(0, accumulation_index, 2, rank, 2)]
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
