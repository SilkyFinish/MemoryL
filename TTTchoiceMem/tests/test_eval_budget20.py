"""Explicit twenty-episode regressions; all models, GPUs and simulators are mocked."""

from dataclasses import replace
import json
from pathlib import Path
import random
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import eval_concurrency as concurrency
import eval_multitask as evaluation


TASK_NAMES = (
    "battery_try", "blocks_ranking_try", "cover_blocks", "observe_and_pickup", "place_block_mat",
    "press_button", "put_back_block", "rearrange_blocks", "swap_T", "swap_blocks",
)


@pytest.fixture
def config20(tmp_path):
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
        tasks=TASK_NAMES, rmbench_root=native_root, qwen_python=Path(sys.executable),
        env_python=Path(sys.executable), gpu_ids=tuple(range(8)), workers_per_gpu=3,
        ports=evaluation.CONCURRENT_PORTS, episodes=20, max_seed_attempts=1000,
        expected_action_head="oft", expected_training_steps=55000, expected_cameras=("head_camera",),
    )


def build_suite_arguments(config, tmp_path):
    return [
        "--checkpoint", str(tmp_path / "checkpoint.pt"), "--eval-root", str(tmp_path / "evaluation"),
        "--rmbench-root", str(config.rmbench_root), "--qwen-python", sys.executable,
        "--env-python", sys.executable, "--gpu-ids", *map(str, config.gpu_ids),
        "--workers-per-gpu", str(config.workers_per_gpu), "--expected-action-head", "oft",
        "--expected-training-steps", "55000", "--expected-cameras", "head_camera",
    ]


def write_candidate_events(output_dir, seed, *, skipped=False, success=False):
    events = [
        {"event": "run_started", "requested_episodes": 1, "seed_start": seed,
         "seed_stride": 1, "result_directory": str(output_dir.resolve())},
        {"event": "seed_attempt", "seed": seed},
    ]
    if skipped:
        events.append({"event": "seed_skipped", "seed": seed, "reason": "expert_error"})
    else:
        events.extend([
            {"event": "episode_done", "seed": seed, "episode_id": 0, "success": success, "rollout_steps": 4},
            {"event": "run_completed", "completed_episodes": 1, "successes": int(success)},
        ])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "episodes.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8",
    )
    return events


def finish_reservation(scheduler, task_name, seed, outcome):
    scheduler.finish(task_name, seed, {
        "task_name": task_name, "seed": seed, "outcome": outcome, "problems": [],
    })


def test_explicit_budget_derivation_and_legacy_defaults(config20, tmp_path):
    config20.validate()
    serialized = config20.to_dict()
    serialized.pop("max_seed_attempts")
    assert evaluation.EvaluationConfig.from_dict(serialized).max_seed_attempts == 1000
    serialized.pop("episodes")
    restored = evaluation.EvaluationConfig.from_dict(serialized)
    assert restored.episodes == 50 and restored.max_seed_attempts == 2500
    direct_default = evaluation.EvaluationConfig(
        tasks=TASK_NAMES, rmbench_root=config20.rmbench_root,
        qwen_python=config20.qwen_python, env_python=config20.env_python,
    )
    direct_default.validate()
    assert direct_default.episodes == 50 and direct_default.max_seed_attempts == 2500
    assert (evaluation.EPISODES_PER_TASK, evaluation.SEED_POOL_SIZE, evaluation.LAST_SEED) == (50, 2500, 102499)
    scheduler = concurrency.EpisodeScheduler(TASK_NAMES, threading.Event())
    assert (scheduler.episodes, scheduler.max_seed_attempts, scheduler.last_seed) == (50, 2500, 102499)
    arguments = build_suite_arguments(config20, tmp_path)
    legacy_arguments = evaluation.parse_args(arguments)
    assert (legacy_arguments.episodes, legacy_arguments.max_seed_attempts) == (50, 2500)
    explicit_arguments = evaluation.parse_args(arguments + ["--episodes", "20"])
    assert (explicit_arguments.episodes, explicit_arguments.max_seed_attempts) == (20, 1000)
    assert evaluation.parse_args(arguments + ["--episodes", "20", "--max-seed-attempts", "1000"]).max_seed_attempts == 1000


@pytest.mark.parametrize("episodes,max_seed_attempts", [
    (20, 2500), (50, 1000), (1, 50), (True, 50), (20.0, 1000), (20, 1000.0), (20, True), (20, None),
])
def test_dataclass_and_serialized_budget_mismatches_fail_closed(config20, episodes, max_seed_attempts):
    invalid = replace(config20, episodes=episodes, max_seed_attempts=max_seed_attempts)
    with pytest.raises(ValueError):
        invalid.validate()
    with pytest.raises(ValueError):
        evaluation.EvaluationConfig.from_dict(invalid.to_dict())
    if max_seed_attempts is not None:
        with pytest.raises(ValueError):
            concurrency.EpisodeScheduler(TASK_NAMES, threading.Event(), episodes, max_seed_attempts)


def test_cli_rejects_explicit_mismatch_and_propagates_budget_to_dry_run(config20, tmp_path, capsys):
    arguments = build_suite_arguments(config20, tmp_path)
    with pytest.raises(SystemExit):
        evaluation.parse_args(arguments + ["--episodes", "20", "--max-seed-attempts", "2500"])
    with pytest.raises(SystemExit):
        evaluation.parse_args(arguments + ["--episodes", "50", "--max-seed-attempts", "1000"])
    with patch.object(evaluation.subprocess, "Popen") as launch, patch.object(evaluation.subprocess, "run") as command:
        assert evaluation.main(arguments + ["--episodes", "20", "--dry-run"]) == 0
    launch.assert_not_called()
    command.assert_not_called()
    output = capsys.readouterr().out
    assert '"total_requested_episodes": 200' in output
    assert '"instruction_generation_test_num": 20' in output
    assert output.count("serve_rmbench.py") == output.count("--episodes 20") == 24
    assert output.count("--max-seed-attempts 1000") == 24
    assert not (tmp_path / "evaluation").exists()


def test_plan_commands_and_candidate_boundaries_use_exact_pool(config20, tmp_path, capsys):
    plan = evaluation.build_suite_plan(config20, tmp_path / "final.pt")
    assert plan["total_requested_episodes"] == 200
    assert plan["test_num"] == plan["episodes_per_task"] == plan["max_completed_plus_inflight_per_task"] == 20
    assert plan["instruction_generation_test_num"] == 20
    assert plan["native_case_test_num"] == plan["native_case_max_seed_attempts"] == 1
    assert plan["seed_pool_size"] == plan["max_seed_attempts_per_task"] == 1000
    assert plan["candidate_seeds"] == list(range(100000, 101000)) and plan["last_seed"] == 100999
    slot = concurrency.build_worker_slots(config20)[0]
    for seed in (100000, 100999):
        command = concurrency.build_candidate_command(config20, TASK_NAMES[0], seed, slot, tmp_path / "case")
        parsed = evaluation.parse_args(command[2:])
        assert (parsed.episodes, parsed.max_seed_attempts, parsed.case_seed) == (20, 1000, seed)
    for seed in (99999, 101000, 102499, True):
        with pytest.raises(ValueError, match="pool"):
            concurrency.build_candidate_command(config20, TASK_NAMES[0], seed, slot, tmp_path / "case")
    with pytest.raises(SystemExit):
        evaluation.parse_args(command[2:-1] + ["101000"])
    case_dir = tmp_path / "boundary"
    write_candidate_events(case_dir, 101000)
    assert concurrency.summarize_candidate("fixture", 101000, case_dir, 0)["outcome"] == "completed"
    assert concurrency.summarize_candidate("fixture", 101000, case_dir, 0, episodes=20)["outcome"] == "error"
    with patch.object(evaluation, "run_native_child") as child:
        assert evaluation.main(command[2:] + ["--dry-run"]) == 0
    child.assert_not_called()
    assert "instruction_generation_test_num=20" in capsys.readouterr().out


def test_twenty_reservations_never_overshoot_and_skips_exhaust_only_1000_candidates():
    scheduler = concurrency.EpisodeScheduler(("fixture",), threading.Event(), episodes=20)
    reservations = [scheduler.claim() for _ in range(20)]
    assert scheduler.claim(wait=False) is None
    finish_reservation(scheduler, *reservations[0], "completed")
    assert scheduler.claim(wait=False) is None
    finish_reservation(scheduler, *reservations[1], "skipped")
    assert scheduler.claim(wait=False) == ("fixture", 100020)
    for task_name, seed in reservations[2:] + [("fixture", 100020)]:
        finish_reservation(scheduler, task_name, seed, "completed")
        budget = scheduler.budgets[task_name]
        assert budget.completed_episodes + len(budget.inflight) <= 20
    assert scheduler.budgets["fixture"].completed_episodes == 20 and scheduler.claim() is None
    exhausted = concurrency.EpisodeScheduler(("fixture",), threading.Event(), episodes=20)
    for seed in range(100000, 101000):
        assert exhausted.claim() == ("fixture", seed)
        finish_reservation(exhausted, "fixture", seed, "skipped")
    assert exhausted.claim() is None
    assert exhausted.snapshot("fixture")["candidate_count"] == 1000
    assert exhausted.budgets["fixture"].completed_episodes == 0


@pytest.mark.parametrize("completion_seed", [0, 1, 2])
def test_twenty_episode_seed_prefix_is_completion_order_independent(completion_seed):
    scheduler = concurrency.EpisodeScheduler(TASK_NAMES, threading.Event(), episodes=20)
    reservations = [scheduler.claim() for _ in range(24)]
    random_generator = random.Random(completion_seed)
    while reservations:
        task_name, seed = reservations.pop(random_generator.randrange(len(reservations)))
        finish_reservation(scheduler, task_name, seed, "skipped" if (seed - 100000) % 7 == 0 else "completed")
        candidate = scheduler.claim(wait=False)
        if candidate is not None:
            reservations.append(candidate)
        assert all(budget.completed_episodes + len(budget.inflight) <= 20 for budget in scheduler.budgets.values())
    for budget in scheduler.budgets.values():
        assert budget.completed_episodes == 20 and not budget.inflight
        assert sorted(result["seed"] for result in budget.results) == list(range(100000, 100024))


@pytest.mark.parametrize("single_candidate", [False, True])
def test_mocked_native_child_generates_twenty_instruction_variants(config20, tmp_path, single_candidate):
    case_dir = tmp_path / "child"
    command = evaluation.build_native_command(config20, TASK_NAMES[0], 0, 18861, case_dir)
    if single_candidate:
        command += ["--case-seed", "100009"]
    arguments = evaluation.parse_args(command[2:])
    instruction = Mock(return_value="native instruction")
    native = SimpleNamespace(build_instruction=instruction)

    def native_main(native_arguments):
        expected_episodes = 1 if single_candidate else 20
        assert native_arguments["test_num"] == expected_episodes
        assert native_arguments["max_seed_attempts"] == (1 if single_candidate else 1000)
        assert native_arguments["seed_offset"] == (9 if single_candidate else 0)
        assert native.build_instruction({}, {}, "unseen", expected_episodes) == "native instruction"
        if single_candidate:
            write_candidate_events(case_dir, 100009, skipped=True)
            raise RuntimeError("Seed attempt limit reached: completed 0/1")

    native.main = native_main
    render = Mock()
    with patch.dict(sys.modules, {
        "eval_rmbench": SimpleNamespace(load_native_evaluator=Mock(return_value=native)),
        "test_render": SimpleNamespace(Sapien_TEST=render),
    }), patch.dict(evaluation.os.environ, {}, clear=True), patch.object(evaluation.subprocess, "Popen") as launch:
        assert evaluation.run_native_child(arguments) == (20 if single_candidate else None)
    instruction.assert_called_once_with({}, {}, "unseen", 20)
    assert native.build_instruction is instruction
    render.assert_called_once()
    launch.assert_not_called()


def test_candidate_process_records_twenty_budget_without_launching_native(config20, tmp_path):
    slot = concurrency.build_worker_slots(config20)[0]
    (tmp_path / TASK_NAMES[0]).mkdir()
    case_dir = tmp_path / TASK_NAMES[0] / "seed100000"
    client = SimpleNamespace(poll=lambda: 0, wait=lambda: 0)

    def launch(command, **options):
        assert command[command.index("--episodes") + 1] == "20"
        assert command[command.index("--max-seed-attempts") + 1] == "1000"
        assert command[-2:] == ["--case-seed", "100000"]
        write_candidate_events(case_dir, 100000, success=True)
        return client

    with patch.object(concurrency.subprocess, "Popen", side_effect=launch), \
            patch.object(evaluation, "stop_owned_process") as cleanup:
        result = concurrency.run_candidate(
            config20, TASK_NAMES[0], 100000, slot, tmp_path, SimpleNamespace(poll=lambda: None), threading.Event(),
        )
    assert result["outcome"] == "completed"
    cleanup.assert_called_once_with(client)
    plan = evaluation.read_json(case_dir / "case_plan.json")
    assert plan["instruction_generation_test_num"] == plan["requested_episodes_per_task"] == 20
    assert plan["max_seed_attempts_per_task"] == 1000
    assert plan["test_num"] == plan["max_seed_attempts"] == 1


@pytest.mark.parametrize("fatal_error", [
    "RuntimeError: CUDA error: no kernel image is available for execution on the device",
    "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
    "RuntimeError: CUDA error: invalid device function",
])
def test_fatal_native_expert_error_is_infrastructure_failure_and_seals_task(tmp_path, fatal_error):
    scheduler = concurrency.EpisodeScheduler(("fixture",), threading.Event(), episodes=20)
    task_name, seed = scheduler.claim()
    case_dir = tmp_path / "case"
    events = write_candidate_events(case_dir, seed, skipped=True)
    (case_dir / "native.log").write_text(fatal_error + "\nConnectionResetError: [Errno 104] Connection reset\n")
    journal_bytes = (case_dir / "episodes.jsonl").read_bytes()
    result = concurrency.summarize_candidate(task_name, seed, case_dir, 20, episodes=20, max_seed_attempts=1000)
    assert result["outcome"] == "error" and result["infrastructure_failure"]
    assert result["events"] == events and result["skipped_seed_count"] == 1
    assert (case_dir / "episodes.jsonl").read_bytes() == journal_bytes
    scheduler.finish(task_name, seed, result)
    assert scheduler.claim() is None and scheduler.snapshot(task_name)["candidate_count"] == 1
    legacy = evaluation.summarize_task(task_name, case_dir, 20, episodes=20)
    assert legacy["infrastructure_failure"] and not legacy["coverage_complete"]
    native = SimpleNamespace(build_instruction=Mock(), main=Mock(
        side_effect=RuntimeError("Seed attempt limit reached: completed 0/1"),
    ))
    with pytest.raises(RuntimeError, match="Seed attempt limit"):
        concurrency.invoke_native_candidate(native, {"task_name": task_name}, seed, case_dir, episodes=20)


def test_legitimate_expert_skip_stays_distinct_from_cuda_failure(tmp_path):
    case_dir = tmp_path / "case"
    write_candidate_events(case_dir, 100000, skipped=True)
    (case_dir / "native.log").write_text("Expert failed: object unstable\n", encoding="utf-8")
    result = concurrency.summarize_candidate("fixture", 100000, case_dir, 20, episodes=20)
    assert result["outcome"] == "skipped" and not result["infrastructure_failure"]


def write_task_journal(output_dir, episodes):
    output_dir.mkdir(parents=True, exist_ok=True)
    events = [{"event": "run_started", "requested_episodes": episodes, "seed_start": 100000}]
    for episode_index in range(episodes):
        events.extend([
            {"event": "seed_attempt", "seed": 100000 + episode_index},
            {"event": "episode_done", "seed": 100000 + episode_index, "success": episode_index % 2 == 0},
        ])
    events.append({"event": "run_completed", "completed_episodes": episodes, "successes": episodes // 2})
    (output_dir / "episodes.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")


@pytest.mark.parametrize("tasks", [TASK_NAMES, (TASK_NAMES[1], TASK_NAMES[3])])
def test_legacy_task_and_suite_use_actual_twenty_denominator(config20, tmp_path, tasks):
    task_dir = tmp_path / "journal"
    write_task_journal(task_dir, 20)
    result = evaluation.summarize_task("fixture", task_dir, 0, episodes=20, max_seed_attempts=1000)
    assert result["coverage_complete"] and result["native_official_success_rate"] == 0.5
    assert result["requested_episodes"] == 20 and result["requested_seed_attempts"] == 1000
    assert not evaluation.summarize_task("fixture", task_dir, 0)["coverage_complete"]
    legacy_config = replace(config20, tasks=tasks, workers_per_gpu=1, ports=evaluation.CONCURRENT_PORTS[:8])

    def run_task(config, checkpoint, task_name, gpu_id, port, eval_root, update_status, cancellation):
        write_task_journal(eval_root / task_name, config.episodes)
        return evaluation.summarize_task(
            task_name, eval_root / task_name, 0, episodes=config.episodes, max_seed_attempts=config.max_seed_attempts,
        )

    with patch.object(evaluation, "validate_checkpoint", return_value={"config": {"horizon": 16, "execute_prefix": 4}}), \
            patch.object(evaluation, "check_gpu_utilization"), patch.object(evaluation, "check_port_free"), \
            patch.object(evaluation, "run_task", side_effect=run_task), patch.object(evaluation.subprocess, "Popen") as launch:
        summary = evaluation.run_suite(legacy_config, tmp_path / "final.pt", tmp_path / "legacy_suite")
    launch.assert_not_called()
    assert summary["state"] == "completed"
    assert summary["requested_episodes"] == summary["completed_episodes"] == len(tasks) * 20
    assert summary["native_official_equal_task_success_rate"] == 0.5


def test_legacy_running_child_stops_on_fatal_expert_log(config20, tmp_path):
    server = SimpleNamespace(pid=4000, poll=lambda: None)
    client = SimpleNamespace(pid=4001, poll=lambda: None)
    task_dir = tmp_path / TASK_NAMES[0]

    def launch(command, **options):
        if "serve_rmbench.py" in command[1]:
            return server
        options["stdout"].write("RuntimeError: CUDA error: no kernel image is available for execution on the device\n")
        options["stdout"].flush()
        return client

    cancellation = Mock()
    cancellation.wait.return_value = False
    with patch.object(evaluation, "check_gpu_utilization"), patch.object(evaluation, "check_port_free"), \
            patch.object(evaluation, "wait_for_server"), patch.object(evaluation, "stop_owned_process") as cleanup, \
            patch.object(evaluation.subprocess, "Popen", side_effect=launch):
        result = evaluation.run_task(
            config20, tmp_path / "final.pt", TASK_NAMES[0], 0, 18861, tmp_path, Mock(), cancellation,
        )
    assert result["infrastructure_failure"] and not result["coverage_complete"]
    assert result["seed_attempt_count"] == 0 and (task_dir / "native.log").is_file()
    assert [call.args[0] for call in cleanup.call_args_list] == [client, server]


def test_mocked_24_slot_suite_runs_exactly_200_episodes(config20, tmp_path):
    output_root = tmp_path / "evaluation"
    first_wave = threading.Barrier(24)
    lock = threading.Lock()
    active_slots = set()
    first_candidates = set()
    servers = []
    maximum_concurrency = 0

    def launch_server(command, **options):
        assert "serve_rmbench.py" in command[1]
        with lock:
            server = SimpleNamespace(pid=1000 + len(servers), poll=lambda: None)
            servers.append(server)
        return server

    def run_candidate(config, task_name, seed, slot, eval_root, server, cancellation):
        nonlocal maximum_concurrency
        with lock:
            assert slot.name not in active_slots
            active_slots.add(slot.name)
            maximum_concurrency = max(maximum_concurrency, len(active_slots))
            first_candidate = slot.name not in first_candidates
            first_candidates.add(slot.name)
        if first_candidate:
            first_wave.wait(timeout=20)
        case_dir = eval_root / task_name / f"seed{seed}"
        skipped = (seed - 100000) % 11 == 0
        write_candidate_events(case_dir, seed, skipped=skipped, success=seed % 2 == 0)
        result = concurrency.summarize_candidate(
            task_name, seed, case_dir, 20 if skipped else 0,
            episodes=config.episodes, max_seed_attempts=config.max_seed_attempts,
        )
        with lock:
            active_slots.remove(slot.name)
        return result

    with patch.object(evaluation, "validate_checkpoint", return_value={"config": {"horizon": 16, "execute_prefix": 4}}) as checkpoint_check, \
            patch.object(concurrency, "check_gpu_headroom", return_value={}), \
            patch.object(evaluation, "check_port_free"), patch.object(evaluation, "wait_for_server") as readiness, \
            patch.object(evaluation, "stop_owned_process") as cleanup, \
            patch.object(concurrency.subprocess, "Popen", side_effect=launch_server), \
            patch.object(concurrency, "run_candidate", side_effect=run_candidate):
        summary = evaluation.run_suite(config20, tmp_path / "checkpoint.pt", output_root)
    checkpoint_check.assert_called_once_with(
        tmp_path / "checkpoint.pt", expected_step=55000, expected_action_head="oft", expected_cameras=("head_camera",),
    )
    assert summary["state"] == "completed" and summary["coverage_complete"]
    assert summary["requested_episodes"] == summary["completed_episodes"] == 200
    assert len(servers) == readiness.call_count == cleanup.call_count == maximum_concurrency == 24
    assert len(summary["tasks"]) == 10
    for task in summary["tasks"]:
        assert task["requested_episodes"] == task["completed_episodes"] == len(set(task["completed_seeds"])) == 20
        assert task["requested_seed_attempts"] == 1000
        assert task["native_official_success_rate"] == task["successes"] / 20
        assert task["seed_attempts"] == list(range(100000, 100000 + task["seed_attempt_count"]))
        assert (output_root / task["task_name"] / "episodes.jsonl").is_file()
    expected_rate = sum(task["successes"] / 20 for task in summary["tasks"]) / 10
    assert summary["native_official_equal_task_success_rate"] == expected_rate
