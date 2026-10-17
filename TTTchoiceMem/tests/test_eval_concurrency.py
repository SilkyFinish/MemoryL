"""CPU-only scheduling/journal regressions; never start a GPU or simulator."""

from dataclasses import replace
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
    config = replace(config, expected_cameras=("head_camera",), expected_action_head="oft", expected_training_steps=55000)
    contract = {"fixture": True}
    expected_summary = {"state": "completed"}
    with patch.object(evaluation, "validate_checkpoint", return_value=contract) as checkpoint_check, \
            patch.object(concurrency, "run_concurrent_suite", return_value=expected_summary) as concurrent_suite, \
            patch.object(evaluation, "check_gpu_utilization") as legacy_gpu_check, \
            patch.object(evaluation.subprocess, "Popen") as launch, \
            patch.object(evaluation.subprocess, "run") as subprocess_run:
        assert evaluation.run_suite(config, tmp_path / "checkpoint.pt", tmp_path / "evaluation") is expected_summary
    checkpoint_check.assert_called_once_with(
        tmp_path / "checkpoint.pt", expected_step=55000, expected_action_head="oft", expected_cameras=("head_camera",),
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


def test_mocked_24_worker_suite_has_persistent_servers_and_500_exact_episodes(config, tmp_path):
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

    with patch.object(evaluation, "validate_checkpoint", return_value={}) as checkpoint_check, \
            patch.object(concurrency, "check_gpu_headroom", return_value={}) as gpu_check, \
            patch.object(evaluation, "check_gpu_utilization") as legacy_gpu_check, \
            patch.object(evaluation, "check_port_free") as ports, \
            patch.object(evaluation, "wait_for_server") as readiness, \
            patch.object(evaluation, "stop_owned_process") as cleanup, \
            patch.object(concurrency.subprocess, "Popen", side_effect=launch_server), \
            patch.object(concurrency, "run_candidate", side_effect=native_candidate):
        summary = evaluation.run_suite(config, tmp_path / "checkpoint.pt", output_root)
    assert summary["state"] == "completed"
    assert summary["completed_episodes"] == 500
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

    with patch.object(evaluation, "validate_checkpoint", return_value={}), \
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
        "step": 55000, "config": configuration,
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
