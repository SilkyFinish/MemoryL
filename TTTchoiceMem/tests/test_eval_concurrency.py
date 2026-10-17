"""CPU-only scheduling/journal regressions; never start a GPU or simulator."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

import eval_concurrency as concurrency
import eval_multitask as evaluation


TASK_NAMES = (
    "battery_try", "blocks_ranking_try", "cover_blocks", "observe_and_pickup", "place_block_mat",
    "press_button", "put_back_block", "rearrange_blocks", "swap_T", "swap_blocks",
)


@pytest.fixture
def config(tmp_path):
    native_root = tmp_path / "native"
    for directory in ("envs", "scripts", "env_cfg/eval", "env_cfg/task_config"):
        (native_root / directory).mkdir(parents=True)
    (native_root / "env_cfg/eval/all_tasks.yml").write_text(
        "tasks:\n" + "".join(f"  - {task_name}\n" for task_name in TASK_NAMES), encoding="utf-8",
    )
    for source_path in (
        native_root / "scripts/eval_policy_xpolicylab.py",
        native_root / "env_cfg/task_config/demo_clean.yml",
        *(native_root / "envs" / f"{task_name}.py" for task_name in TASK_NAMES),
    ):
        source_path.touch()
    return evaluation.EvaluationConfig(
        tasks=TASK_NAMES, rmbench_root=native_root,
        qwen_python=Path(sys.executable), env_python=Path(sys.executable),
        gpu_ids=tuple(range(8)), workers_per_gpu=3, ports=evaluation.CONCURRENT_PORTS,
    )


def case_events(output_dir, seed, *, skipped=False, success=False):
    events = [
        {"event": "run_started", "requested_episodes": 1, "seed_start": seed,
         "seed_stride": 1, "result_directory": str(output_dir.resolve())},
        {"event": "seed_attempt", "seed": seed},
    ]
    if skipped:
        events.append({"event": "seed_skipped", "seed": seed, "reason": "expert_failed"})
    else:
        events += [
            {"event": "episode_done", "seed": seed, "episode_id": 0, "success": success, "rollout_steps": 4},
            {"event": "run_completed", "completed_episodes": 1, "successes": int(success)},
        ]
    return events


def write_events(output_dir, events):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "episodes.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8",
    )


def result_for(task_name, seed, outcome):
    return {"task_name": task_name, "seed": seed, "outcome": outcome, "problems": []}


def test_default_compatibility_and_24_distinct_gpu_slots(config):
    config.validate()
    slots = concurrency.build_worker_slots(config)
    assert len(slots) == 24
    assert [slot.port for slot in slots] == list(range(18861, 18885))
    assert [slot.gpu_id for slot in slots] == [gpu_id for gpu_id in range(8) for _ in range(3)]
    assert len({slot.name for slot in slots}) == 24
    assert 18844 not in evaluation.ALLOWED_PORTS
    assert evaluation.choose_default_ports(evaluation.DEFAULT_GPU_IDS, 1) == evaluation.DEFAULT_PORTS
    default_config = replace(config, gpu_ids=evaluation.DEFAULT_GPU_IDS, ports=evaluation.DEFAULT_PORTS, workers_per_gpu=1)
    default_config.validate()
    assert default_config.expected_cameras is None
    for workers in (1, 2, 3):
        changed = replace(config, workers_per_gpu=workers, ports=evaluation.CONCURRENT_PORTS[:8 * workers])
        changed.validate()


@pytest.mark.parametrize("workers", [0, 4, True, 1.5])
def test_invalid_worker_counts_are_rejected(config, workers):
    with pytest.raises(ValueError, match="workers_per_gpu"):
        replace(config, workers_per_gpu=workers).validate()


def test_ports_and_camera_contract_are_strict_and_round_trip(config):
    for ports in (config.ports[:-1], (config.ports[0],) * 24, (*config.ports[:-1], 18844)):
        with pytest.raises(ValueError):
            replace(config, ports=ports).validate()
    serialized = config.to_dict()
    serialized.pop("ports")
    serialized["expected_cameras"] = ["right_camera", "head_camera"]
    restored = evaluation.EvaluationConfig.from_dict(serialized)
    assert restored.ports == config.ports
    assert restored.expected_cameras == ("right_camera", "head_camera")
    for cameras in ((), ("front_camera",), ("head_camera", "head_camera"), "head_camera"):
        with pytest.raises(ValueError):
            replace(config, expected_cameras=cameras).validate()


def test_cli_dry_run_prints_24_servers_without_processes(config, tmp_path, capsys):
    arguments = [
        "--checkpoint", str(tmp_path / "unused.pt"), "--eval-root", str(tmp_path / "evaluation"),
        "--rmbench-root", str(config.rmbench_root), "--qwen-python", sys.executable,
        "--env-python", sys.executable, "--gpu-ids", *map(str, range(8)),
        "--workers-per-gpu", "3", "--expected-cameras", "head_camera", "--dry-run",
    ]
    with patch.object(evaluation.subprocess, "Popen") as launch, patch.object(evaluation.subprocess, "run") as command:
        assert evaluation.main(arguments) == 0
    launch.assert_not_called()
    command.assert_not_called()
    output = capsys.readouterr().out
    assert output.count("serve_rmbench.py") == 24
    assert output.count("--native-child") == 24
    assert "--port 18884" in output
    assert "--case-seed 100002" in output
    assert not (tmp_path / "evaluation").exists()
    parsed = evaluation.parse_args(arguments[:-1])
    assert parsed.ports == list(range(18861, 18885))


def test_native_case_seed_boundaries_and_dry_run_never_load_simulator(config, tmp_path, capsys):
    slot = concurrency.build_worker_slots(config)[0]
    command = concurrency.build_candidate_command(config, TASK_NAMES[0], 100000, slot, tmp_path / "case")
    with patch.object(evaluation, "run_native_child") as child, patch.object(evaluation.subprocess, "Popen") as launch:
        assert evaluation.main(command[2:] + ["--dry-run"]) == 0
    child.assert_not_called()
    launch.assert_not_called()
    output = capsys.readouterr().out
    assert "test_num=1 seed=100000 max_seed_attempts=1 instruction_generation_test_num=50" in output
    for seed in (99999, 102500):
        with pytest.raises(SystemExit):
            evaluation.parse_args(command[2:-1] + [str(seed)])
    for seed in (100000, 102499):
        assert evaluation.parse_args(command[2:-1] + [str(seed)]).case_seed == seed
    arguments = evaluation.parse_args(command[2:])
    for seed in (99999, 102500, True):
        arguments.case_seed = seed
        with pytest.raises(ValueError, match="case-seed"):
            evaluation.run_native_child(arguments)
    with pytest.raises(SystemExit):
        evaluation.parse_args(["--case-seed", "100000"])
    with pytest.raises(SystemExit):
        evaluation.parse_args(["--native-child", "--sim-gpu", "0", "--port", "18861"])
    assert not (tmp_path / "case").exists()


def test_suite_forwards_nondefault_camera_head_and_step_before_concurrent_startup(config, tmp_path):
    config = replace(config, expected_cameras=("head_camera",), expected_action_head="oft", expected_training_steps=55000,
                     expected_backbone="memoryvla")
    contract = {"fixture": True, "config": {"horizon": 16, "execute_prefix": 4}}
    expected_summary = {"state": "completed"}
    with patch.object(evaluation, "validate_checkpoint", return_value=contract) as checkpoint_check, \
            patch.object(concurrency, "run_concurrent_suite", return_value=expected_summary) as concurrent_suite, \
            patch.object(evaluation, "check_gpu_utilization") as legacy_gpu_check, \
            patch.object(evaluation.subprocess, "Popen") as launch, \
            patch.object(evaluation.subprocess, "run") as subprocess_run:
        assert evaluation.run_suite(config, tmp_path / "checkpoint.pt", tmp_path / "evaluation") is expected_summary
    checkpoint_check.assert_called_once_with(
        tmp_path / "checkpoint.pt", expected_step=55000, expected_action_head="oft", expected_cameras=("head_camera",),
        expected_backbone="memoryvla",
    )
    concurrent_suite.assert_called_once_with(config, tmp_path / "checkpoint.pt", tmp_path / "evaluation", contract)
    legacy_gpu_check.assert_not_called()
    launch.assert_not_called()
    subprocess_run.assert_not_called()


def test_round_robin_seed_prefix_and_budget_independent_of_completion_order():
    completed_seed_sets = []
    for completion_seed in range(12):
        scheduler = concurrency.EpisodeScheduler(TASK_NAMES, threading.Event())
        candidates = [scheduler.claim() for _ in range(24)]
        assert [task_name for task_name, _ in candidates[:10]] == list(TASK_NAMES)
        random_generator = random.Random(completion_seed)
        while candidates:
            candidate_index = random_generator.randrange(len(candidates))
            task_name, seed = candidates.pop(candidate_index)
            outcome = "skipped" if (seed - evaluation.FIRST_SEED) % 7 == 0 else "completed"
            scheduler.finish(task_name, seed, result_for(task_name, seed, outcome))
            # Refill only after settling a reservation, never oversubscribe.
            candidate = scheduler.claim(wait=False)
            if candidate is not None:
                candidates.append(candidate)
            for budget in scheduler.budgets.values():
                assert budget.completed_episodes + len(budget.inflight) <= 50
        task_completed_seeds = []
        for budget in scheduler.budgets.values():
            assert budget.completed_episodes == 50
            assert not budget.inflight
            attempted_seeds = sorted(result["seed"] for result in budget.results)
            assert attempted_seeds == list(range(100000, budget.next_seed))
            task_completed_seeds.append(sorted(result["seed"] for result in budget.results if result["outcome"] == "completed"))
        completed_seed_sets.append(task_completed_seeds)
    assert all(seed_set == completed_seed_sets[0] for seed_set in completed_seed_sets)


def test_fifty_inflight_reservations_block_speculation_until_a_skip():
    scheduler = concurrency.EpisodeScheduler(("fixture",), threading.Event())
    candidates = [scheduler.claim() for _ in range(50)]
    assert scheduler.claim(wait=False) is None
    scheduler.finish("fixture", 100000, result_for("fixture", 100000, "completed"))
    assert scheduler.claim(wait=False) is None
    scheduler.finish("fixture", 100001, result_for("fixture", 100001, "skipped"))
    assert scheduler.claim(wait=False) == ("fixture", 100050)
    for task_name, seed in candidates[2:] + [("fixture", 100050)]:
        scheduler.finish(task_name, seed, result_for(task_name, seed, "completed"))
    assert scheduler.budgets["fixture"].completed_episodes == 50
    assert not scheduler.budgets["fixture"].inflight
    assert scheduler.claim() is None


def test_task_error_seals_budget_without_replacement_and_drains_existing_candidates():
    scheduler = concurrency.EpisodeScheduler(("fixture",), threading.Event())
    task_name, first_seed = scheduler.claim()
    _, second_seed = scheduler.claim()
    scheduler.finish(task_name, first_seed, result_for(task_name, first_seed, "error"))
    scheduler.finish(task_name, second_seed, result_for(task_name, second_seed, "completed"))
    assert scheduler.claim() is None
    assert scheduler.budgets[task_name].next_seed == 100002
    assert scheduler.budgets[task_name].completed_episodes == 1
    with pytest.raises(ValueError, match="duplicated"):
        scheduler.finish(task_name, second_seed, result_for(task_name, second_seed, "completed"))


def test_pool_exhaustion_and_cancellation_do_not_refill():
    scheduler = concurrency.EpisodeScheduler(("fixture",), threading.Event())
    for seed in range(100000, 102500):
        assert scheduler.claim() == ("fixture", seed)
        scheduler.finish("fixture", seed, result_for("fixture", seed, "skipped"))
    assert scheduler.claim() is None
    assert scheduler.budgets["fixture"].completed_episodes == 0
    scheduler.cancellation.set()
    assert scheduler.claim() is None


def test_candidate_success_failure_skip_and_corruption_are_distinct(tmp_path):
    case_dir = tmp_path / "case"
    for success in (True, False):
        write_events(case_dir, case_events(case_dir, 100000, success=success))
        result = concurrency.summarize_candidate("fixture", 100000, case_dir, 0)
        assert result["outcome"] == "completed"
        assert result["successes"] == int(success)
    write_events(case_dir, case_events(case_dir, 100000, skipped=True))
    assert concurrency.summarize_candidate("fixture", 100000, case_dir, 20)["outcome"] == "skipped"
    assert concurrency.summarize_candidate("fixture", 100000, case_dir, 0)["outcome"] == "error"
    for reason in ([], {}, None, "policy_failed"):
        invalid_skip = case_events(case_dir, 100000, skipped=True)
        invalid_skip[-1]["reason"] = reason
        write_events(case_dir, invalid_skip)
        assert concurrency.summarize_candidate("fixture", 100000, case_dir, 20)["outcome"] == "error"
    baseline = case_events(case_dir, 100000)
    corruptions = [
        baseline + [baseline[2]], [baseline[0], baseline[1], baseline[1], *baseline[2:]],
        baseline[1:], baseline[:-1], list(reversed(baseline)),
        [baseline[0], baseline[1], {**baseline[2], "seed": 102500}, baseline[3]],
        [baseline[0], baseline[1], {**baseline[2], "success": 0}, baseline[3]],
        [baseline[0], baseline[1], {**baseline[2], "episode_id": 1}, baseline[3]],
        [baseline[0], baseline[1], {**baseline[2], "rollout_steps": -1}, baseline[3]],
        [baseline[0], baseline[1], baseline[2], {**baseline[3], "successes": 1}],
        [*baseline[:2], {"event": "episode_error", "seed": 100000, "error": "policy crash"}, *baseline[2:]],
        [*baseline, {"event": "unexpected_event"}],
    ]
    for events in corruptions:
        write_events(case_dir, events)
        assert concurrency.summarize_candidate("fixture", 100000, case_dir, 0)["outcome"] == "error"
    write_events(case_dir, baseline)
    with (case_dir / "episodes.jsonl").open("a", encoding="utf-8") as stream:
        stream.write('["not an event"]\n{bad json}\n{"event": "bad", "seed": NaN}\n')
    result = concurrency.summarize_candidate("fixture", 100000, case_dir, 0)
    assert result["outcome"] == "error"
    assert len(result["problems"]) >= 3


def test_native_single_candidate_preserves_instruction_budget_and_skip_is_proven(config, tmp_path):
    slot = concurrency.build_worker_slots(config)[5]
    case_dir = tmp_path / "case"
    command = concurrency.build_candidate_command(config, TASK_NAMES[0], 100009, slot, case_dir)
    arguments = evaluation.parse_args(command[2:])
    original_instruction = Mock(return_value="native instruction")
    native = SimpleNamespace(build_instruction=original_instruction)

    def native_main(native_arguments):
        assert native_arguments["test_num"] == native_arguments["max_seed_attempts"] == 1
        assert native_arguments["seed"] == 0 and native_arguments["seed_offset"] == 9
        assert not native_arguments["eval_batch"] and native_arguments["num_workers"] == 1
        assert native.build_instruction({}, {}, "unseen", 1) == "native instruction"
        write_events(case_dir, case_events(case_dir, 100009, skipped=True))
        raise RuntimeError("Seed attempt limit reached: completed 0/1")

    native.main = native_main
    with patch.dict(sys.modules, {
        "eval_rmbench": SimpleNamespace(load_native_evaluator=Mock(return_value=native)),
        "test_render": SimpleNamespace(Sapien_TEST=Mock()),
    }), patch.dict(evaluation.os.environ, {}, clear=True):
        assert evaluation.run_native_child(arguments) == 20
        assert evaluation.os.environ["CUDA_VISIBLE_DEVICES"] == "1"
        assert evaluation.os.environ["RMBENCH_RENDER_DEVICE"] == "cuda:0"
        assert evaluation.os.environ["OMP_NUM_THREADS"] == "2"
    original_instruction.assert_called_once_with({}, {}, "unseen", 50)
    assert native.build_instruction is original_instruction
    with pytest.raises(FileExistsError):
        evaluation.run_native_child(arguments)


def test_only_exact_native_skip_exception_is_handled(tmp_path):
    case_dir = tmp_path / "case"
    original_instruction = Mock()
    native = SimpleNamespace(build_instruction=original_instruction)
    native_arguments = {"task_name": "fixture"}
    for exception_message in ("policy failed", "Seed attempt limit reached: completed 0/1"):
        write_events(case_dir, case_events(case_dir, 100000))
        native.main = Mock(side_effect=RuntimeError(exception_message))
        with pytest.raises(RuntimeError, match=exception_message):
            concurrency.invoke_native_candidate(native, native_arguments, 100000, case_dir)
        assert native.build_instruction is original_instruction


def test_gpu_gate_checks_utilization_and_headroom_once_for_all_replicas(config):
    response = SimpleNamespace(stdout="\n".join(f"{gpu_id}, 0, 80000" for gpu_id in range(8)))
    with patch.object(concurrency.subprocess, "run", return_value=response) as command:
        measurements = concurrency.check_gpu_headroom(config)
    command.assert_called_once()
    assert len(measurements["selected_gpus"]) == 8
    for output in ("0, 50, 80000", "0, 0, 16000", "0, 0, nan"):
        with patch.object(concurrency.subprocess, "run", return_value=SimpleNamespace(stdout=output)):
            with pytest.raises(RuntimeError):
                concurrency.check_gpu_headroom(replace(config, gpu_ids=(0,), ports=config.ports[:3]))


@pytest.mark.parametrize("tasks", [TASK_NAMES, (TASK_NAMES[1], TASK_NAMES[3])])
def test_mocked_24_worker_suite_has_persistent_servers_and_exact_selected_episodes(config, tmp_path, tasks):
    config = replace(config, tasks=tasks)
    output_root = tmp_path / "evaluation"
    lock = threading.Lock()
    active_slots = set()
    maximum_concurrency = 0
    servers = []
    first_wave = threading.Barrier(24)
    first_candidate_by_slot = set()

    def launch_server(command, **kwargs):
        assert "serve_rmbench.py" in command[1]
        assert kwargs["env"]["OMP_NUM_THREADS"] == "2"
        process = SimpleNamespace(pid=1000 + len(servers), poll=lambda: None)
        servers.append(process)
        return process

    def native_candidate(config, task_name, seed, slot, eval_root, server, cancellation):
        nonlocal maximum_concurrency
        with lock:
            assert slot.name not in active_slots
            active_slots.add(slot.name)
            maximum_concurrency = max(maximum_concurrency, len(active_slots))
            first_candidate = slot.name not in first_candidate_by_slot
            first_candidate_by_slot.add(slot.name)
        if first_candidate:
            first_wave.wait(timeout=10)
        time.sleep(0.001)
        output_dir = eval_root / task_name / f"seed{seed}"
        skipped = (seed - 100000) % 11 == 0
        write_events(output_dir, case_events(output_dir, seed, skipped=skipped, success=seed % 2 == 0))
        result = concurrency.summarize_candidate(task_name, seed, output_dir, 20 if skipped else 0)
        evaluation.write_atomic_json(output_dir / "summary.json", result)
        with lock:
            active_slots.remove(slot.name)
        return result

    with patch.object(evaluation, "validate_checkpoint", return_value={"config": {"horizon": 16, "execute_prefix": 4}}) as checkpoint_check, \
            patch.object(concurrency, "check_gpu_headroom", return_value={}) as gpu_check, \
            patch.object(evaluation, "check_gpu_utilization") as legacy_gpu_check, \
            patch.object(evaluation, "check_port_free") as ports, \
            patch.object(evaluation, "wait_for_server") as readiness, \
            patch.object(evaluation, "stop_owned_process") as cleanup, \
            patch.object(concurrency.subprocess, "Popen", side_effect=launch_server), \
            patch.object(concurrency, "run_candidate", side_effect=native_candidate):
        summary = evaluation.run_suite(config, tmp_path / "checkpoint.pt", output_root)
    assert summary["state"] == "completed"
    assert summary["requested_episodes"] == summary["completed_episodes"] == len(tasks) * 50
    assert summary["native_official_equal_task_success_rate"] == (
        sum(result["native_official_success_rate"] for result in summary["tasks"]) / len(tasks)
    )
    assert maximum_concurrency == 24
    assert len(servers) == readiness.call_count == cleanup.call_count == ports.call_count == 24
    assert len({call.args[2] for call in readiness.call_args_list}) == 24
    gpu_check.assert_called_once_with(config)
    legacy_gpu_check.assert_not_called()
    checkpoint_check.assert_called_once_with(tmp_path / "checkpoint.pt")
    for result in summary["tasks"]:
        assert len(result["completed_seeds"]) == 50
        assert len(set(result["completed_seeds"])) == 50
        assert result["seed_attempts"] == list(range(100000, 100000 + result["seed_attempt_count"]))
        assert (output_root / result["task_name"] / "episodes.jsonl").is_file()
    with patch.object(evaluation, "validate_checkpoint") as checkpoint_check:
        with pytest.raises(FileExistsError):
            evaluation.run_suite(config, tmp_path / "checkpoint.pt", output_root)
    checkpoint_check.assert_not_called()


def test_server_startup_failure_launches_no_native_candidates_and_cleans_owned(config, tmp_path):
    owned = []

    def launch(command, **kwargs):
        process = SimpleNamespace(pid=4000 + len(owned), poll=lambda: None)
        owned.append(process)
        return process

    with patch.object(evaluation, "validate_checkpoint", return_value={"config": {"horizon": 16, "execute_prefix": 4}}), \
            patch.object(concurrency, "check_gpu_headroom", return_value={}), \
            patch.object(evaluation, "check_port_free"), \
            patch.object(evaluation, "wait_for_server", side_effect=TimeoutError("not ready")), \
            patch.object(evaluation, "stop_owned_process") as cleanup, \
            patch.object(concurrency.subprocess, "Popen", side_effect=launch), \
            patch.object(concurrency, "run_candidate") as candidate:
        summary = evaluation.run_suite(config, tmp_path / "checkpoint.pt", tmp_path / "evaluation")
    assert summary["state"] == "failed" and summary["completed_episodes"] == 0
    assert summary["errors"]
    candidate.assert_not_called()
    assert {call.args[0].pid for call in cleanup.call_args_list if call.args[0] is not None} == {process.pid for process in owned}


def test_candidate_process_uses_case_budget_and_owned_cleanup(config, tmp_path):
    slot = concurrency.build_worker_slots(config)[0]
    task_root = tmp_path / TASK_NAMES[0]
    task_root.mkdir()
    case_dir = task_root / "seed100000"
    server = SimpleNamespace(poll=lambda: None)
    client = SimpleNamespace(pid=5001, poll=lambda: 0, wait=lambda: 0)

    def launch(command, **kwargs):
        assert command[-2:] == ["--case-seed", "100000"]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
        assert kwargs["env"]["RMBENCH_RENDER_DEVICE"] == "cuda:0"
        assert kwargs["start_new_session"] is True
        write_events(case_dir, case_events(case_dir, 100000))
        return client

    with patch.object(concurrency.subprocess, "Popen", side_effect=launch) as launch_mock, \
            patch.object(evaluation, "stop_owned_process") as cleanup:
        result = concurrency.run_candidate(config, TASK_NAMES[0], 100000, slot, tmp_path, server, threading.Event())
    assert result["outcome"] == "completed" and result["successes"] == 0
    launch_mock.assert_called_once()
    cleanup.assert_called_once_with(client)
    case_plan = evaluation.read_json(case_dir / "case_plan.json")
    assert case_plan["test_num"] == case_plan["max_seed_attempts"] == 1
    assert case_plan["instruction_generation_test_num"] == 50


def checkpoint_payload(tmp_path, head, world_size):
    configuration = {
        "backbone": "qwen35", "action_head": head, "freeze_backbone": False,
        "dtype": "bfloat16", "horizon": 16, "execute_prefix": 4, "group_size": 16,
        "model_revision": "2fc06364715b967f1860aea9cf38778875588b17", "mem_length": 16,
        "retrieval_layers": 2, "per_token_size": 256, "image_size": [224, 224],
        "cameras": ["head_camera"], "update_fused": False, "fusion_type": "gate", "consolidate_type": "tome",
        "model_source": str(tmp_path), "asset_manifest": {},
        "diffusion_model_type": "DiT-L" if head == "diffusion" else "inactive-value",
        "ddim_steps": 10 if head == "diffusion" else 999, "cfg_scale": 1.5 if head == "diffusion" else 99,
        "oft_hidden_dim": 256 if head == "oft" else 987,
    }
    return {
        "schema": "choice-memory-rmbench-v1", "step": 55000, "config": configuration,
        "protocol": {"cameras": ["head_camera"], "optimization": {
            "max_steps": 55000, "world_size": world_size, "per_device_batch_size": 16,
            "global_batch_size": world_size * 16, "grad_accumulation": 1, "groups_per_forward": 1,
            "master_dtype": "float32", "autocast_dtype": "bfloat16", "gradient_checkpointing": True,
        }},
        "normalization": {},
        "model": {component + "weight": torch.zeros(1) for component in (
            "backbone.", "action_head.", "per_compr.", "cog_mem_bank.", "per_mem_bank.",
        )},
    }


@pytest.mark.parametrize("head", ["diffusion", "oft"])
@pytest.mark.parametrize("world_size", [4, 8])
def test_checkpoint_heads_and_world_budget_are_decoupled(tmp_path, head, world_size):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture")
    payload = checkpoint_payload(tmp_path, head, world_size)
    with patch("choice_mem.checkpoint.read_checkpoint", return_value=payload):
        evaluation.validate_checkpoint(checkpoint, 55000, head, expected_cameras=("head_camera",))
        with pytest.raises(ValueError, match="expected_cameras"):
            evaluation.validate_checkpoint(checkpoint, 55000, head, expected_cameras=("left_camera",))
        payload["protocol"]["cameras"] = ["left_camera"]
        with pytest.raises(ValueError, match="cameras disagree"):
            evaluation.validate_checkpoint(checkpoint, 55000, head)


def test_checkpoint_still_enforces_active_head_hashes_and_finiteness(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture")
    payload = checkpoint_payload(tmp_path, "diffusion", 8)
    with patch("choice_mem.checkpoint.read_checkpoint", return_value=payload):
        for field_name, bad_value in (("ddim_steps", 2), ("diffusion_model_type", "DiT-S"), ("cfg_scale", 2.0)):
            original = payload["config"][field_name]
            payload["config"][field_name] = bad_value
            with pytest.raises(ValueError, match=field_name):
                evaluation.validate_checkpoint(checkpoint, 55000)
            payload["config"][field_name] = original
        payload["model"]["backbone.weight"] = torch.tensor([float("nan")])
        with pytest.raises(ValueError, match="nonfinite"):
            evaluation.validate_checkpoint(checkpoint, 55000)
        payload["model"]["backbone.weight"] = torch.zeros(1)
        payload["config"]["asset_manifest"] = {"weights": "frozen-hash"}
        with patch("choice_mem.data.hash_file", return_value="changed-hash"):
            with pytest.raises(ValueError, match="source asset changed"):
                evaluation.validate_checkpoint(checkpoint, 55000)


def test_selected_tasks_default_and_positive_training_budget(config):
    assert evaluation.DEFAULT_RMBENCH_ROOT == Path("/mnt/data/dylan/d2/memory/RMBench")
    selected = replace(config, tasks=(TASK_NAMES[3], TASK_NAMES[0]), expected_training_steps=17,
                       expected_action_head="oft", expected_backbone="memoryvla")
    selected.validate()
    assert evaluation.EvaluationConfig.from_dict(selected.to_dict()).tasks == selected.tasks
    serialized = selected.to_dict()
    serialized.pop("tasks")
    assert evaluation.EvaluationConfig.from_dict(serialized).tasks == TASK_NAMES
    serialized["tasks"] = None
    assert evaluation.EvaluationConfig.from_dict(serialized).tasks == TASK_NAMES
    for tasks in ((), (TASK_NAMES[0], TASK_NAMES[0]), ("not_an_official_task",), "battery_try"):
        with pytest.raises(ValueError):
            replace(config, tasks=tasks).validate()
    for step in (0, -1, True, 17.0):
        with pytest.raises(ValueError, match="positive integer"):
            replace(config, expected_training_steps=step).validate()
    with pytest.raises(ValueError, match="backbone"):
        replace(config, expected_backbone="other").validate()


def test_policy_python_alias_and_stage1_subset_dry_run_are_cpu_only(config, tmp_path, capsys):
    arguments = [
        "--checkpoint", str(tmp_path / "unused.pt"), "--eval-root", str(tmp_path / "evaluation"),
        "--rmbench-root", str(config.rmbench_root), "--policy-python", sys.executable,
        "--env-python", sys.executable, "--tasks", TASK_NAMES[3], TASK_NAMES[0],
        "--expected-training-steps", "17", "--expected-backbone", "memoryvla", "--dry-run",
    ]
    assert evaluation.parse_args(arguments).qwen_python == Path(sys.executable)
    legacy_alias = ["--qwen-python" if argument == "--policy-python" else argument for argument in arguments]
    assert vars(evaluation.parse_args(legacy_alias)) == vars(evaluation.parse_args(arguments))
    for step in ("0", "-1", "1.5"):
        invalid = [step if argument == "17" else argument for argument in arguments]
        with pytest.raises(SystemExit):
            evaluation.parse_args(invalid)
    with patch.object(evaluation, "validate_checkpoint") as checkpoint_check, \
            patch.object(evaluation, "run_native_child") as child, \
            patch.object(evaluation.subprocess, "Popen") as launch, \
            patch.object(evaluation.subprocess, "run") as command:
        assert evaluation.main(arguments) == 0
    for operation in (checkpoint_check, child, launch, command):
        operation.assert_not_called()
    output = capsys.readouterr().out
    assert '"total_requested_episodes": 100' in output
    assert output.count("serve_rmbench.py") == output.count("--native-child") == 2
    assert not (tmp_path / "evaluation").exists()


@pytest.fixture
def stage1_payload(tmp_path):
    """A tiny real v2 CPU checkpoint, not a mocked-away schema reader."""
    from choice_mem.config import PolicyConfig
    from choice_mem.data import hash_file

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}", encoding="utf-8")
    configuration = PolicyConfig(
        model_source=str(source), model_revision="local-sha256", group_size=8,
        mem_length=8, per_token_size=32, dtype="float32", freeze_backbone=True,
        asset_manifest={"config.json": hash_file(source / "config.json")},
    ).to_dict()
    normalization = {
        "action": {"q01": [0.0] * 14, "q99": [1.0] * 14},
        "training_episode_ids": ["train-a", "train-b"], "action_dim": 14, "target_offset": 1,
    }
    return {
        "schema": "choice-memory-rmbench-v2", "step": 17, "config": configuration,
        "normalization": normalization,
        "model": {component + "weight": torch.zeros(1) for component in (
            "backbone.", "action_head.", "per_compr.", "cog_mem_bank.", "per_mem_bank.",
        )},
        "protocol": {
            **{field: configuration[field] for field in (
                "cameras", "action_dim", "horizon", "execute_prefix", "group_size", "image_size",
                "history_sampling", "image_aug", "repeated_diffusion_steps", "mask_padding_loss", "inference_timestep_mode",
            )},
            "trainer": "choice-memory-multitask-training-v2", "target_offset": 1,
            "manifest_sha256": "a" * 64,
            "train_episode_ids": ["train-a", "train-b"], "validation_episode_ids": ["validation-a"],
            "requested_configuration": {**configuration, "model_revision": None, "asset_manifest": {}},
            "sampling": {
                "tasks": [TASK_NAMES[0], TASK_NAMES[1]], "history_sampling": "random",
                "per_task": {TASK_NAMES[0]: {"train_episodes": 1, "validation_episodes": 1},
                             TASK_NAMES[1]: {"train_episodes": 1, "validation_episodes": 0}},
            },
            "optimization": {
                "optimizer": "AdamW", "recipe": "memoryvla", "max_steps": 17,
                "world_size": 2, "group_size": 8, "per_device_batch_size": 16,
                "groups_per_forward": 2, "grad_accumulation": 3, "global_batch_size": 96,
                "master_dtype": "float32", "autocast_dtype": "float32",
                "learning_rate": 2e-5, "backbone_learning_rate": 2e-5, "memory_head_learning_rate": 2e-5,
                "weight_decay": 0, "gradient_clip_norm": 1, "warmup_ratio": 0, "schedule": "constant",
                "minimum_backbone_learning_rate": None, "minimum_head_learning_rate": None,
            },
        },
    }


@pytest.fixture
def debug_payload(stage1_payload, tmp_path):
    from choice_mem.random_init import inspect_random_backbone_assets

    metadata = tmp_path / "llama-metadata"
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
    source = tmp_path / "metadata-source.json"
    source.write_text(json.dumps({
        "llama_metadata": {"path": str(metadata), "files": files},
        # These paths deliberately do not exist: none is a dependency of random init.
        "checkpoint_path": str(tmp_path / "must-not-read-33GB.pt"), "checkpoint_sha256": "b" * 64,
        "local_vision_files": {"vision": str(tmp_path / "must-not-read-vision.bin")},
    }))
    configuration = stage1_payload["config"]
    configuration.update(
        backbone_initialization="random-frozen-debug", backbone_initialization_seed=73,
        base_vlm_id="prism-dinosiglip-224px+7b", dtype="bfloat16", freeze_backbone=True,
        model_source=str(source), asset_manifest=inspect_random_backbone_assets(source)["asset_manifest"],
    )
    for name in ("backbone_initialization", "backbone_initialization_seed", "base_vlm_id"):
        stage1_payload["protocol"][name] = configuration[name]
    stage1_payload["protocol"]["requested_configuration"] = {**configuration, "asset_manifest": {}}
    stage1_payload["protocol"]["optimization"].update(autocast_dtype="bfloat16", frozen_backbone_dtype="bfloat16")
    stage1_payload["model"]["backbone.weight"] = torch.zeros(1, dtype=torch.bfloat16)
    return stage1_payload


def test_debug_gate_requires_explicit_opt_in_and_checks_metadata_only(debug_payload, tmp_path):
    checkpoint = tmp_path / "debug.pt"
    torch.save(debug_payload, checkpoint)
    with pytest.raises(ValueError, match="explicit allow_debug_checkpoint"):
        evaluation.validate_checkpoint(checkpoint, 17)
    with patch("choice_mem.data.hash_file", side_effect=AssertionError("No pretrained weight hashing")), \
            patch("choice_mem.checkpoint.ChoiceMemoryPolicy", side_effect=AssertionError("No model loading")), \
            patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("No GPU")):
        # Sixth positional parameter is the new optional opt-in; first five are unchanged.
        contract = evaluation.validate_checkpoint(checkpoint, 17, "diffusion", ("head_camera",), "memoryvla", True)
    assert contract["config"]["backbone_initialization"] == "random-frozen-debug"
    assert "checkpoint" not in contract["config"]["asset_manifest"]
    assert contract["config"]["asset_manifest"]["manifest"] != "b" * 64
    with pytest.raises(ValueError, match="explicit allow_debug_checkpoint"):
        evaluation.validate_checkpoint(checkpoint, 17, allow_debug_checkpoint="true")


@pytest.mark.parametrize("mutation,reason", [
    (lambda p: p["model"].update({"backbone.weight": torch.zeros(1)}), "non-BF16"),
    (lambda p: p["model"].update({"backbone.weight": torch.zeros(1, dtype=torch.float16)}), "non-BF16"),
    (lambda p: p["model"].update({"per_compr.weight": torch.zeros(1, dtype=torch.bfloat16)}), "non-FP32"),
    (lambda p: p["model"].update({"cog_mem_bank.weight": torch.zeros(1, dtype=torch.bfloat16)}), "non-FP32"),
    (lambda p: p["model"].update({"per_mem_bank.weight": torch.zeros(1, dtype=torch.bfloat16)}), "non-FP32"),
    (lambda p: p["model"].update({"action_head.weight": torch.zeros(1, dtype=torch.bfloat16)}), "non-FP32"),
    (lambda p: p["model"]["backbone.weight"].fill_(float("nan")), "nonfinite"),
    (lambda p: p["model"]["action_head.weight"].fill_(float("inf")), "nonfinite"),
    (lambda p: p["model"].pop("backbone.weight"), "component"),
    (lambda p: p.update(step=16), "step"),
    (lambda p: p["protocol"]["optimization"].update(max_steps=18), "max_steps"),
    (lambda p: p["protocol"]["optimization"].update(master_dtype="bfloat16"), "dtype"),
    (lambda p: p["protocol"]["optimization"].pop("frozen_backbone_dtype"), "frozen_backbone_dtype"),
    (lambda p: p["protocol"].update(backbone_initialization_seed=42), "backbone_initialization_seed"),
    (lambda p: p["protocol"].pop("backbone_initialization"), "missing field"),
    (lambda p: p["protocol"]["requested_configuration"].pop("backbone_initialization_seed"), "missing field"),
    (lambda p: p["protocol"]["requested_configuration"].update(dtype="float32"), "dtype"),
    (lambda p: p["protocol"].update(manifest_sha256="bad"), "manifest_sha256"),
    (lambda p: p["normalization"].update(training_episode_ids=["validation-a"]), "normalization"),
    (lambda p: p["config"]["asset_manifest"].update(checkpoint="b" * 64), "asset_manifest"),
])
def test_debug_gate_retains_precision_provenance_and_completed_budget_checks(debug_payload, tmp_path, mutation, reason):
    mutation(debug_payload)
    checkpoint = tmp_path / "bad-debug.pt"
    torch.save(debug_payload, checkpoint)
    with pytest.raises(ValueError, match=reason):
        evaluation.validate_checkpoint(checkpoint, 17, allow_debug_checkpoint=True)


def test_debug_metadata_change_and_resumable_position_still_fail(debug_payload, tmp_path):
    checkpoint = tmp_path / "debug.pt"
    debug_payload["optimizer"] = {"state": {}, "param_groups": []}
    debug_payload["training_state"] = {
        "schema": "choice-memory-multitask-training-v2", "sampler": debug_payload["protocol"]["sampling"],
        "scheduler": {"schedule": "constant", "total_steps": 17, "completed_steps": 17},
        "rank_rng_states": [{}, {}], "next_global_micro_index": 102, "next_global_group_index": 204,
    }
    torch.save(debug_payload, checkpoint)
    assert evaluation.validate_checkpoint(checkpoint, 17, allow_debug_checkpoint=True)["step"] == 17
    for field, value in (("rank_rng_states", [{}]), ("next_global_micro_index", 101), ("next_global_group_index", 203)):
        original = debug_payload["training_state"][field]
        debug_payload["training_state"][field] = value
        torch.save(debug_payload, checkpoint)
        with pytest.raises(ValueError, match="sampler/optimizer"):
            evaluation.validate_checkpoint(checkpoint, 17, allow_debug_checkpoint=True)
        debug_payload["training_state"][field] = original
    torch.save(debug_payload, checkpoint)
    (tmp_path / "llama-metadata" / "tokenizer.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="SHA256"):
        evaluation.validate_checkpoint(checkpoint, 17, allow_debug_checkpoint=True)


def test_official_suite_cannot_opt_in_to_debug_metrics(debug_payload, config, tmp_path):
    checkpoint = tmp_path / "debug.pt"
    torch.save(debug_payload, checkpoint)
    official = replace(config, expected_training_steps=17)
    with patch.object(evaluation, "check_gpu_utilization") as gpu, \
            patch.object(concurrency, "check_gpu_headroom") as headroom, \
            patch.object(evaluation.subprocess, "Popen") as launch:
        with pytest.raises(ValueError, match="explicit allow_debug_checkpoint"):
            evaluation.run_suite(official, checkpoint, tmp_path / "official")
    for operation in (gpu, headroom, launch):
        operation.assert_not_called()
    assert "--allow-debug-checkpoint" not in evaluation.build_server_command(official, checkpoint, official.ports[0])
    assert not (tmp_path / "official").exists()


def test_historical_v2_missing_initialization_fields_keep_strict_fp32_gate(stage1_payload, tmp_path):
    for section in (stage1_payload["config"], stage1_payload["protocol"]["requested_configuration"]):
        for field in ("backbone_initialization", "backbone_initialization_seed", "base_vlm_id"):
            section.pop(field, None)
    checkpoint = tmp_path / "old-v2.pt"
    torch.save(stage1_payload, checkpoint)
    assert evaluation.validate_checkpoint(checkpoint, 17)["config"]["backbone_initialization"] == "pretrained"
    stage1_payload["model"]["backbone.weight"] = torch.zeros(1, dtype=torch.bfloat16)
    torch.save(stage1_payload, checkpoint)
    for opt_in in (False, True):
        with pytest.raises(ValueError, match="non-FP32"):
            evaluation.validate_checkpoint(checkpoint, 17, allow_debug_checkpoint=opt_in)


@pytest.mark.parametrize("backbone", ["memoryvla", "qwen35"])
@pytest.mark.parametrize("head", ["diffusion", "oft"])
@pytest.mark.parametrize("freeze_backbone", [False, True])
def test_stage1_completed_snapshot_accepts_saved_method_without_qwen_budget(
    stage1_payload, tmp_path, backbone, head, freeze_backbone,
):
    payload = stage1_payload
    payload["config"].update(backbone=backbone, action_head=head, freeze_backbone=freeze_backbone)
    payload["protocol"]["requested_configuration"].update(
        backbone=backbone, action_head=head, freeze_backbone=freeze_backbone,
    )
    checkpoint = tmp_path / "final.pt"
    torch.save(payload, checkpoint)
    with patch("choice_mem.checkpoint.ChoiceMemoryPolicy", side_effect=AssertionError("No model loading")), \
            patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("No GPU initialization")), \
            patch.object(evaluation.subprocess, "Popen") as launch:
        contract = evaluation.validate_checkpoint(checkpoint, 17, head, ("head_camera",), backbone)
    assert contract["step"] == 17 and contract["config"]["group_size"] == 8
    assert "training_state" not in payload and "optimizer" not in payload
    launch.assert_not_called()
    other_backbone = "qwen35" if backbone == "memoryvla" else "memoryvla"
    with pytest.raises(ValueError, match="expected_backbone"):
        evaluation.validate_checkpoint(checkpoint, 17, head, expected_backbone=other_backbone)


@pytest.mark.parametrize("mutate,reason", [
    (lambda p: p.update(step=16), "step"),
    (lambda p: p["protocol"]["optimization"].update(max_steps=18), "max_steps"),
    (lambda p: p["protocol"]["optimization"].update(global_batch_size=448), "group/batch"),
    (lambda p: p["protocol"]["optimization"].update(group_size=16), "group/batch"),
    (lambda p: p["protocol"]["optimization"].update(world_size=True), "positive integer"),
    (lambda p: p["protocol"]["optimization"].update(autocast_dtype="bfloat16"), "dtype"),
    (lambda p: p["protocol"]["optimization"].update(learning_rate=float("nan")), "learning_rate"),
    (lambda p: p["protocol"]["optimization"].update(schedule="cosine with linear warmup"), "single-LR"),
    (lambda p: p["protocol"].update(image_aug=False), "image_aug"),
    (lambda p: p["protocol"].update(manifest_sha256="bad-hash"), "manifest_sha256"),
    (lambda p: p["protocol"].update(validation_episode_ids=["train-a"]), "train/validation"),
    (lambda p: p["normalization"].update(training_episode_ids=["validation-a"]), "normalization"),
    (lambda p: p["normalization"]["action"]["q01"].__setitem__(0, float("nan")), "quantiles"),
    (lambda p: p["protocol"]["sampling"].update(tasks=[]), "sampling"),
    (lambda p: p["config"].update(freeze_backbone=False), "requested_configuration"),
    (lambda p: p["model"].pop("per_mem_bank.weight"), "component"),
    (lambda p: p["model"].update({"backbone.weight": torch.tensor([float("inf")])}), "nonfinite"),
    (lambda p: p["model"].update({"action_head.weight": torch.zeros(1, dtype=torch.bfloat16)}), "non-FP32"),
    (lambda p: p["config"].update(asset_manifest={}), "asset_manifest"),
])
def test_stage1_gate_rejects_incomplete_or_inconsistent_payload(stage1_payload, tmp_path, mutate, reason):
    mutate(stage1_payload)
    checkpoint = tmp_path / "invalid.pt"
    torch.save(stage1_payload, checkpoint)
    with pytest.raises(ValueError, match=reason):
        evaluation.validate_checkpoint(checkpoint, 17)


@pytest.mark.parametrize("expected_step", [0, -1, True, 17.0])
def test_checkpoint_expected_budget_is_a_strict_positive_integer(tmp_path, expected_step):
    with pytest.raises(ValueError, match="positive integer"):
        evaluation.validate_checkpoint(tmp_path / "not-read.pt", expected_step)


def test_stage1_resumable_checkpoint_requires_completed_scheduler_and_sampler(stage1_payload, tmp_path):
    payload = stage1_payload
    payload["optimizer"] = {"state": {}, "param_groups": []}
    payload["training_state"] = {
        "schema": "choice-memory-multitask-training-v2", "sampler": payload["protocol"]["sampling"],
        "scheduler": {"schedule": "constant", "total_steps": 17, "completed_steps": 17, "base_rates": [2e-5]},
        "rank_rng_states": [{}, {}], "next_global_micro_index": 102, "next_global_group_index": 204,
    }
    checkpoint = tmp_path / "resumable.pt"
    torch.save(payload, checkpoint)
    assert evaluation.validate_checkpoint(checkpoint, 17)["step"] == 17
    for field_name, bad_value in (("completed_steps", 16), ("total_steps", 18), ("completed_steps", True)):
        original = payload["training_state"]["scheduler"][field_name]
        payload["training_state"]["scheduler"][field_name] = bad_value
        torch.save(payload, checkpoint)
        with pytest.raises(ValueError, match="training_state scheduler"):
            evaluation.validate_checkpoint(checkpoint, 17)
        payload["training_state"]["scheduler"][field_name] = original
    payload["training_state"]["next_global_group_index"] += 1
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="sampler/optimizer"):
        evaluation.validate_checkpoint(checkpoint, 17)


@pytest.mark.parametrize("source_kind", ["pt", "json"])
def test_stage1_memoryvla_file_asset_aliases_and_config_integrity(stage1_payload, tmp_path, source_kind):
    from choice_mem.data import hash_file

    run = tmp_path / "original-run"
    (run / "checkpoints").mkdir(parents=True)
    config_path = run / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    weights = run / "checkpoints/latest-checkpoint.pt"
    weights.write_bytes(b"original source weights; never loaded")
    source = weights
    manifest = {"config.json": hash_file(config_path), "checkpoint": hash_file(weights)}
    if source_kind == "json":
        vision = run / "vision.bin"
        vision.write_bytes(b"vision source; never loaded")
        tokenizer = run / "tokenizer.json"
        tokenizer.write_text("{}", encoding="utf-8")
        source = run / "assets.json"
        source.write_text(json.dumps({
            "checkpoint_path": str(weights), "checkpoint_sha256": hash_file(weights),
            "config.json": {"path": str(config_path), "sha256": hash_file(config_path)},
            "local_vision_files": {"vision-name": str(vision)},
            "llama_metadata": {"path": str(run), "files": {"tokenizer.json": hash_file(tokenizer)}},
        }), encoding="utf-8")
        manifest = {"manifest": hash_file(source), "checkpoint": hash_file(weights),
                    "vision-name": hash_file(vision), "llama/tokenizer.json": hash_file(tokenizer)}
    stage1_payload["config"].update(model_source=str(source), asset_manifest=manifest)
    stage1_payload["protocol"]["requested_configuration"]["model_source"] = str(source)
    checkpoint = tmp_path / "final.pt"
    torch.save(stage1_payload, checkpoint)
    assert evaluation.validate_checkpoint(checkpoint, 17)["step"] == 17
    config_path.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="source asset changed: config.json"):
        evaluation.validate_checkpoint(checkpoint, 17)


@pytest.mark.parametrize("step,head", [(30000, "diffusion"), (55000, "diffusion"), (55000, "oft")])
def test_legacy_v1_keeps_its_frozen_gate(tmp_path, step, head):
    checkpoint = tmp_path / "legacy.pt"
    checkpoint.write_bytes(b"fixture")
    payload = checkpoint_payload(tmp_path, head, 4)
    payload["step"] = step
    with patch("choice_mem.checkpoint.read_checkpoint", return_value=payload):
        assert evaluation.validate_checkpoint(checkpoint, step, head)["step"] == step
        for field_name, bad_value in (("backbone", "memoryvla"), ("freeze_backbone", True), ("group_size", 8)):
            original = payload["config"][field_name]
            payload["config"][field_name] = bad_value
            with pytest.raises(ValueError, match=field_name):
                evaluation.validate_checkpoint(checkpoint, step, head)
            payload["config"][field_name] = original
        payload["step"] = 17
        with pytest.raises(ValueError, match="Legacy v1"):
            evaluation.validate_checkpoint(checkpoint, 17, head)


def test_v2_training_only_horizon_is_generic_but_native_rollout_gate_stays_fixed(stage1_payload, config, tmp_path):
    for section in (stage1_payload["config"], stage1_payload["protocol"], stage1_payload["protocol"]["requested_configuration"]):
        section.update(horizon=4, execute_prefix=2)
    checkpoint = tmp_path / "training-only.pt"
    torch.save(stage1_payload, checkpoint)
    assert evaluation.validate_checkpoint(checkpoint, 17)["config"]["horizon"] == 4
    native_config = replace(config, tasks=TASK_NAMES[:2], expected_training_steps=17)
    with patch.object(evaluation, "check_gpu_utilization") as gpu_check, \
            patch.object(concurrency, "check_gpu_headroom") as headroom_check, \
            patch.object(evaluation.subprocess, "Popen") as launch:
        with pytest.raises(ValueError, match="horizon.*native evaluation"):
            evaluation.run_suite(native_config, checkpoint, tmp_path / "not-launched")
    for operation in (gpu_check, headroom_check, launch):
        operation.assert_not_called()
    assert not (tmp_path / "not-launched").exists()


def test_incomplete_selected_task_coverage_never_publishes_an_official_rate(config, tmp_path):
    selected = replace(config, tasks=TASK_NAMES[:2], workers_per_gpu=1, gpu_ids=(0,), ports=(18861,))

    def run_task(config, checkpoint, task_name, gpu_id, port, eval_root, update_status, cancellation):
        complete = task_name == config.tasks[0]
        return {"coverage_complete": complete, "completed_episodes": 50 if complete else 49,
                "native_official_success_rate": 0.5 if complete else None}

    with patch.object(evaluation, "validate_checkpoint", return_value={"config": {"horizon": 16, "execute_prefix": 4}}), \
            patch.object(evaluation, "check_gpu_utilization"), patch.object(evaluation, "check_port_free"), \
            patch.object(evaluation, "run_task", side_effect=run_task), \
            patch.object(evaluation.subprocess, "Popen") as launch:
        summary = evaluation.run_suite(selected, tmp_path / "final.pt", tmp_path / "subset")
    launch.assert_not_called()
    assert summary["state"] == "failed" and not summary["coverage_complete"]
    assert summary["requested_episodes"] == 100 and summary["completed_episodes"] == 99
    assert summary["native_official_equal_task_success_rate"] is None
