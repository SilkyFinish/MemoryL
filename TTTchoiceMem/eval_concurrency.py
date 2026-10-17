"""Bounded episode scheduling with one persistent native policy server per slot.

Only expert/setup skips release a reserved episode without consuming it. Keeping
completed + inflight <= requested episodes prevents speculative policy outcomes:
the final seed set is the same contiguous prefix for every completion order.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
from pathlib import Path
import shlex
import signal
import subprocess
import threading
import time

import eval_multitask as evaluation


SKIPPED_EXIT_CODE = 20
SKIP_REASONS = {"expert_unstable", "expert_error", "expert_failed", "setup_unstable", "setup_error"}


@dataclass(frozen=True)
class WorkerSlot:
    index: int
    gpu_id: int
    replica_index: int
    port: int

    @property
    def name(self) -> str:
        return f"gpu{self.gpu_id}_worker{self.replica_index}"


def build_worker_slots(config: evaluation.EvaluationConfig) -> tuple[WorkerSlot, ...]:
    return tuple(
        WorkerSlot(index, gpu_id, replica_index, config.ports[index])
        for index, (gpu_id, replica_index) in enumerate(
            (gpu_id, replica_index)
            for gpu_id in config.gpu_ids
            for replica_index in range(config.workers_per_gpu)
        )
    )


def build_candidate_command(config, task_name: str, seed: int, slot: WorkerSlot, output_dir: Path) -> list[str]:
    seed_pool_size = evaluation.resolve_seed_attempt_budget(config.episodes, config.max_seed_attempts)
    if type(seed) is not int or not evaluation.FIRST_SEED <= seed < evaluation.FIRST_SEED + seed_pool_size:
        raise ValueError("Candidate seed is outside the frozen pool")
    return evaluation.build_native_command(config, task_name, slot.gpu_id, slot.port, output_dir) + [
        "--case-seed", str(seed),
    ]


def build_replica_server_command(config, checkpoint: Path, slot: WorkerSlot) -> list[str]:
    command = evaluation.build_server_command(config, checkpoint, slot.port) + ["--cpu-threads", "2"]
    if config.expected_cameras is not None:
        command += ["--cameras", *config.expected_cameras]
    return command


def print_concurrent_commands(config, checkpoint: Path, eval_root: Path) -> None:
    for slot in build_worker_slots(config):
        prefix = f"CUDA_VISIBLE_DEVICES={slot.gpu_id} OMP_NUM_THREADS=2 "
        print(prefix + shlex.join(build_replica_server_command(config, checkpoint, slot)))
        task_name = config.tasks[slot.index % len(config.tasks)]
        seed = evaluation.FIRST_SEED + slot.index // len(config.tasks)
        case_dir = eval_root / task_name / f"seed{seed}"
        print(prefix + shlex.join(build_candidate_command(config, task_name, seed, slot, case_dir)))


def check_gpu_headroom(config) -> dict:
    """One prelaunch sample, never gate a later replica on our own GPU activity.

    The configurable 8 GiB/slot default is only a conservative safety floor,
    not a measured model/renderer peak or a promise that replicas will fit.
    """
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15,
    )
    measurements = {}
    for line in result.stdout.splitlines():
        if line.strip():
            gpu_index, utilization, free_memory = (part.strip() for part in line.split(","))
            measurements[int(gpu_index)] = {"utilization_percent": float(utilization), "free_memory_mib": float(free_memory)}
    minimum_memory_mib = config.minimum_free_gpu_gib_per_worker * 1024 * config.workers_per_gpu
    for gpu_id in config.gpu_ids:
        measurement = measurements.get(gpu_id)
        if measurement is None or not 0 <= measurement["utilization_percent"] < 50:
            raise RuntimeError(f"GPU {gpu_id} is missing or not below 50% utilization; refusing to launch")
        if not measurement["free_memory_mib"] >= minimum_memory_mib:
            raise RuntimeError(f"GPU {gpu_id} lacks the configured {minimum_memory_mib / 1024:g} GiB replica safety floor")
    return {
        "selected_gpus": {str(gpu_id): measurements[gpu_id] for gpu_id in config.gpu_ids},
        "minimum_free_gib_per_worker": config.minimum_free_gpu_gib_per_worker,
        "memory_floor_is_not_measured_peak": True,
    }


def read_candidate_events(output_dir: Path) -> tuple[list[dict], list[str]]:
    events = []
    problems = []

    def reject_nonfinite(value):
        raise ValueError(f"Nonfinite journal number: {value}")

    try:
        with (output_dir / "episodes.jsonl").open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line, parse_constant=reject_nonfinite)
                    if not isinstance(event, dict) or not isinstance(event.get("event"), str):
                        raise ValueError("Journal entry must be an event object")
                    events.append(event)
                except ValueError as exception:
                    problems.append(f"Invalid journal entry {line_number}: {exception}")
    except OSError as exception:
        problems.append(f"Unreadable native journal: {exception}")
    return events, problems


def summarize_candidate(task_name: str, seed: int, output_dir: Path, returncode: int | None,
                        error: str | None = None, *, episodes: int = evaluation.EPISODES_PER_TASK,
                        max_seed_attempts: int | None = None) -> dict:
    """Fail closed on malformed journals, including processes that exit zero."""
    seed_pool_size = evaluation.resolve_seed_attempt_budget(episodes, max_seed_attempts)
    events, problems = read_candidate_events(output_dir)
    infrastructure_failures = evaluation.read_native_infrastructure_failures(output_dir)
    problems.extend(f"Native infrastructure failure: {failure}" for failure in infrastructure_failures)
    event_names = [event["event"] for event in events]
    completed = [event for event in events if event["event"] == "episode_done"]
    skipped = [event for event in events if event["event"] == "seed_skipped"]
    attempts = [event for event in events if event["event"] == "seed_attempt"]
    starts = [event for event in events if event["event"] == "run_started"]
    finishes = [event for event in events if event["event"] == "run_completed"]
    if type(seed) is not int or not evaluation.FIRST_SEED <= seed < evaluation.FIRST_SEED + seed_pool_size:
        problems.append("Candidate seed is outside the fixed pool")
    if len(starts) != 1 or any(
        type(starts[0].get(field_name)) is not int or starts[0].get(field_name) != expected_value
        for field_name, expected_value in (("requested_episodes", 1), ("seed_start", seed), ("seed_stride", 1))
    ):
        problems.append("Native run_started disagrees with the single-candidate budget/seed")
    if len(starts) == 1 and starts[0].get("result_directory") != str(output_dir.resolve()):
        problems.append("Native journal result directory disagrees with the case directory")
    if len(attempts) != 1:
        problems.append("Native case must attempt exactly one seed")
    for event in events:
        if event["event"] in {"seed_attempt", "seed_skipped", "episode_done", "episode_error"}:
            if type(event.get("seed")) is not int or event.get("seed") != seed:
                problems.append("Native event has a duplicated, wrong, or invalid candidate seed")
        if event["event"] == "episode_error":
            problems.append("Native policy rollout error; never replaced or counted as a valid outcome")
    for event in completed:
        if type(event.get("success")) is not bool:
            problems.append("Native success is not a boolean")
        if type(event.get("episode_id")) is not int or event["episode_id"] != 0:
            problems.append("Native single-case episode_id must be zero")
        if type(event.get("rollout_steps")) is not int or event["rollout_steps"] < 0:
            problems.append("Native rollout_steps must be a nonnegative integer")
    for event in skipped:
        if not isinstance(event.get("reason"), str) or event["reason"] not in SKIP_REASONS:
            problems.append("Native skip is not an expert/setup skip")

    outcome = "error"
    if event_names == ["run_started", "seed_attempt", "episode_done", "run_completed"]:
        successes = int(completed[0].get("success") is True)
        if (
            type(finishes[0].get("completed_episodes")) is not int or finishes[0]["completed_episodes"] != 1
            or type(finishes[0].get("successes")) is not int or finishes[0]["successes"] != successes
        ):
            problems.append("Native run_completed disagrees with the episode outcome")
        if returncode != 0:
            problems.append(f"Native completed episode did not exit successfully: {returncode}")
        if not problems:
            outcome = "completed"
    elif event_names == ["run_started", "seed_attempt", "seed_skipped"]:
        if returncode != SKIPPED_EXIT_CODE:
            problems.append(f"Native skip lacks its dedicated exit status: {returncode}")
        if not problems:
            outcome = "skipped"
    else:
        problems.append("Native journal boundaries/order/counts are invalid or incomplete")
    if error:
        problems.append(error)
        outcome = "error"
    return {
        "task_name": task_name, "seed": seed, "outcome": outcome,
        "returncode": returncode, "problems": problems, "events": events,
        "infrastructure_failure": bool(infrastructure_failures), "infrastructure_errors": infrastructure_failures,
        "seed_attempt_count": len(attempts), "completed_episodes": len(completed),
        "successes": sum(event.get("success") is True for event in completed),
        "skipped_seed_count": len(skipped), "output_dir": str(output_dir),
    }


def invoke_native_candidate(native_evaluator, native_arguments: dict, seed: int, output_dir: Path,
                            *, episodes: int = evaluation.EPISODES_PER_TASK,
                            max_seed_attempts: int | None = None) -> int:
    seed_pool_size = evaluation.resolve_seed_attempt_budget(episodes, max_seed_attempts)
    original_build_instruction = native_evaluator.build_instruction

    def build_full_budget_instruction(arguments, episode_info, instruction_type, test_num):
        return original_build_instruction(arguments, episode_info, instruction_type, episodes)

    native_evaluator.build_instruction = build_full_budget_instruction
    try:
        try:
            native_evaluator.main(native_arguments)
        except RuntimeError as exception:
            if str(exception) != "Seed attempt limit reached: completed 0/1":
                raise
            result = summarize_candidate(
                native_arguments["task_name"], seed, output_dir, SKIPPED_EXIT_CODE,
                episodes=episodes, max_seed_attempts=seed_pool_size,
            )
            if result["outcome"] != "skipped":
                raise
            return SKIPPED_EXIT_CODE
        return 0
    finally:
        native_evaluator.build_instruction = original_build_instruction


@dataclass
class TaskBudget:
    next_seed: int = evaluation.FIRST_SEED
    completed_episodes: int = 0
    inflight: set[int] = field(default_factory=set)
    results: list[dict] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


class EpisodeScheduler:
    """Thread-safe round-robin reservations; policy errors seal a task."""

    def __init__(self, tasks: tuple[str, ...], cancellation: threading.Event,
                 episodes: int = evaluation.EPISODES_PER_TASK, max_seed_attempts: int | None = None):
        self.max_seed_attempts = evaluation.resolve_seed_attempt_budget(episodes, max_seed_attempts)
        self.episodes = episodes
        self.last_seed = evaluation.FIRST_SEED + self.max_seed_attempts - 1
        self.tasks = tasks
        self.budgets = {task_name: TaskBudget() for task_name in tasks}
        self.cancellation = cancellation
        self.condition = threading.Condition()
        self.next_task_index = 0

    def claim(self, *, wait: bool = True) -> tuple[str, int] | None:
        with self.condition:
            while not self.cancellation.is_set():
                for offset in range(len(self.tasks)):
                    task_index = (self.next_task_index + offset) % len(self.tasks)
                    task_name = self.tasks[task_index]
                    budget = self.budgets[task_name]
                    if (
                        budget.problems or budget.next_seed > self.last_seed
                        or budget.completed_episodes + len(budget.inflight) >= self.episodes
                    ):
                        continue
                    seed = budget.next_seed
                    budget.next_seed += 1
                    budget.inflight.add(seed)
                    self.next_task_index = (task_index + 1) % len(self.tasks)
                    return task_name, seed
                if not wait or not any(budget.inflight for budget in self.budgets.values()):
                    return None
                self.condition.wait(timeout=0.25)
            return None

    def finish(self, task_name: str, seed: int, result: dict) -> None:
        with self.condition:
            budget = self.budgets[task_name]
            if seed not in budget.inflight:
                raise ValueError("Candidate completion was duplicated or never reserved")
            if result.get("seed") != seed or result.get("task_name") != task_name:
                raise ValueError("Candidate completion belongs to a different reservation")
            budget.inflight.remove(seed)
            budget.results.append(result)
            if result["outcome"] == "completed":
                budget.completed_episodes += 1
            elif result["outcome"] != "skipped":
                budget.problems.append(f"Candidate {seed} failed; no replacement: {result['problems']}")
            if budget.completed_episodes + len(budget.inflight) > self.episodes:
                raise AssertionError("Episode reservation budget exceeded")
            self.condition.notify_all()

    def wake(self) -> None:
        with self.condition:
            self.condition.notify_all()

    def snapshot(self, task_name: str) -> dict:
        with self.condition:
            budget = self.budgets[task_name]
            return {
                "completed_episodes": budget.completed_episodes, "inflight_candidates": len(budget.inflight),
                "candidate_count": budget.next_seed - evaluation.FIRST_SEED,
                "problems": list(budget.problems),
            }


def run_candidate(config, task_name: str, seed: int, slot: WorkerSlot, eval_root: Path,
                  server, cancellation: threading.Event) -> dict:
    output_dir = eval_root / task_name / f"seed{seed}"
    output_dir.mkdir(exist_ok=False)
    evaluation.write_atomic_json(output_dir / "case_plan.json", {
        "task_name": task_name, "seed": seed, "seed_offset": seed - evaluation.FIRST_SEED,
        "test_num": 1, "max_seed_attempts": 1, "instruction_generation_test_num": config.episodes,
        "requested_episodes_per_task": config.episodes, "max_seed_attempts_per_task": config.max_seed_attempts,
        "gpu_id": slot.gpu_id, "worker_index": slot.replica_index, "port": slot.port,
        "env_idx": 0, "policy_outcome_retries": 0,
    })
    client = None
    returncode = None
    error = None
    try:
        if cancellation.is_set() or server.poll() is not None:
            raise RuntimeError("Evaluation cancelled or owned policy server exited before the candidate")
        with (output_dir / "native.log").open("x", encoding="utf-8") as native_log:
            client = subprocess.Popen(
                build_candidate_command(config, task_name, seed, slot, output_dir),
                cwd=config.rmbench_root, env=evaluation.build_child_environment(slot.gpu_id, 2),
                stdout=native_log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            while client.poll() is None:
                if cancellation.wait(0.25):
                    raise RuntimeError("Evaluation cancelled; candidate was not retried")
                if server.poll() is not None:
                    raise RuntimeError("Owned policy server exited during the native candidate")
            returncode = client.wait()
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    finally:
        try:
            evaluation.stop_owned_process(client)
        except Exception as exception:
            error = f"{error or ''}; owned candidate cleanup failed: {exception}"
    result = summarize_candidate(
        task_name, seed, output_dir, returncode, error,
        episodes=config.episodes, max_seed_attempts=config.max_seed_attempts,
    )
    evaluation.write_atomic_json(output_dir / "summary.json", result)
    return result


def summarize_concurrent_task(task_name: str, output_dir: Path, budget: TaskBudget,
                              *, episodes: int = evaluation.EPISODES_PER_TASK,
                              max_seed_attempts: int | None = None) -> dict:
    seed_pool_size = evaluation.resolve_seed_attempt_budget(episodes, max_seed_attempts)
    cases = sorted(budget.results, key=lambda result: result["seed"])
    problems = list(budget.problems)
    seeds = [result["seed"] for result in cases]
    reserved_count = budget.next_seed - evaluation.FIRST_SEED
    if seeds != list(range(evaluation.FIRST_SEED, evaluation.FIRST_SEED + reserved_count)):
        problems.append("Candidate summaries do not cover the contiguous reserved seed prefix exactly once")
    if reserved_count > seed_pool_size:
        problems.append("Candidate reservations exceed the fixed seed pool")
    if budget.inflight:
        problems.append(f"Unfinished candidate reservations: {sorted(budget.inflight)}")
    completed = [event for result in cases for event in result["events"] if event["event"] == "episode_done"]
    skipped = [event for result in cases for event in result["events"] if event["event"] == "seed_skipped"]
    attempts = [event.get("seed") for result in cases for event in result["events"] if event["event"] == "seed_attempt"]
    if attempts != seeds:
        problems.append("Native seed_attempt events disagree with the reserved candidate prefix")
    if any(result["outcome"] == "error" for result in cases):
        problems.append("One or more native candidates failed journal/process validation; none were replaced")
    if len(completed) != episodes:
        problems.append(f"Incomplete coverage: {len(completed)}/{episodes} episodes")
    if len(completed) != budget.completed_episodes:
        problems.append("Recorded policy episodes disagree with validated scheduler completions")
    successes = sum(event.get("success") is True for event in completed)
    summary = {
        "task_name": task_name, "requested_episodes": episodes, "completed_episodes": len(completed),
        "requested_seed_attempts": seed_pool_size, "seed_attempt_count": len(attempts),
        "seed_attempts": attempts, "reserved_candidate_seeds": seeds,
        "completed_seeds": [event.get("seed") for event in completed],
        "skipped_seed_count": len(skipped), "skipped_seeds": skipped,
        "successes": successes, "observed_success_rate": successes / len(completed) if completed else None,
        "native_official_success_rate": successes / episodes if not problems else None,
        "infrastructure_failure": any(result.get("infrastructure_failure", False) for result in cases),
        "coverage_complete": not problems, "run_completed": not problems,
        "problems": problems, "cases": cases, "output_dir": str(output_dir),
    }
    # These are copied native events, not synthetic policy outcomes or boundaries.
    with (output_dir / "episodes.jsonl").open("x", encoding="utf-8") as stream:
        for result in cases:
            for event in result["events"]:
                stream.write(json.dumps({**event, "case_seed": result["seed"], "task_name": task_name}) + "\n")
    evaluation.write_atomic_json(output_dir / "summary.json", summary)
    return summary


def run_concurrent_suite(config, checkpoint: Path, eval_root: Path, checkpoint_contract: dict) -> dict:
    if eval_root.exists():
        raise FileExistsError(f"Refusing to reuse evaluation root: {eval_root}")
    headroom = check_gpu_headroom(config)
    slots = build_worker_slots(config)
    for slot in slots:
        evaluation.check_port_free("127.0.0.1", slot.port)
    eval_root.mkdir(parents=True, exist_ok=False)
    (eval_root / "workers").mkdir()
    for task_name in config.tasks:
        (eval_root / task_name).mkdir()
    evaluation.write_atomic_json(eval_root / "plan.json", {
        **evaluation.build_suite_plan(config, checkpoint), "checkpoint_contract": checkpoint_contract,
        "startup_gpu_headroom": headroom,
        "workers": [dict(index=slot.index, gpu_id=slot.gpu_id, replica_index=slot.replica_index, port=slot.port) for slot in slots],
    })
    wall_start = time.time()
    cancellation = threading.Event()
    scheduler = EpisodeScheduler(config.tasks, cancellation, config.episodes, config.max_seed_attempts)
    readiness_barrier = threading.Barrier(len(slots))
    status_lock = threading.Lock()
    status = {
        "state": "running", "started_at": wall_start,
        "tasks": {task_name: {"state": "queued"} for task_name in config.tasks},
        "workers": {slot.name: {"state": "queued", "gpu_id": slot.gpu_id, "port": slot.port} for slot in slots},
    }
    failures = []
    evaluation.write_atomic_json(eval_root / "status.json", status)

    def update_worker(slot: WorkerSlot, changes: dict) -> None:
        with status_lock:
            status["workers"][slot.name].update(changes)
            evaluation.write_atomic_json(eval_root / "status.json", status)

    def update_task(task_name: str) -> None:
        snapshot = scheduler.snapshot(task_name)
        with status_lock:
            status["tasks"][task_name].update(state="evaluating", **snapshot)
            evaluation.write_atomic_json(eval_root / "status.json", status)

    def cancel_suite(signum=None, frame=None) -> None:
        cancellation.set()
        readiness_barrier.abort()
        scheduler.wake()

    def consume_candidates(slot: WorkerSlot) -> None:
        server = None
        worker_error = None
        worker_dir = eval_root / "workers" / slot.name
        try:
            worker_dir.mkdir(exist_ok=False)
            with (worker_dir / "server.log").open("x", encoding="utf-8") as server_log:
                if cancellation.is_set():
                    return
                server = subprocess.Popen(
                    build_replica_server_command(config, checkpoint, slot), cwd=evaluation.PROJECT_ROOT,
                    env=evaluation.build_child_environment(slot.gpu_id, 2), stdout=server_log,
                    stderr=subprocess.STDOUT, start_new_session=True,
                )
                update_worker(slot, {"state": "loading_server", "server_pid": server.pid})
                evaluation.wait_for_server(server, "127.0.0.1", slot.port, config.server_timeout, cancellation)
                update_worker(slot, {"state": "ready"})
                readiness_barrier.wait(timeout=config.server_timeout)
                while not cancellation.is_set():
                    if server.poll() is not None:
                        raise RuntimeError("Persistent owned policy server exited; it was not reloaded")
                    candidate = scheduler.claim()
                    if candidate is None:
                        break
                    task_name, seed = candidate
                    update_worker(slot, {"state": "evaluating", "task_name": task_name, "seed": seed})
                    update_task(task_name)
                    result = run_candidate(config, task_name, seed, slot, eval_root, server, cancellation)
                    scheduler.finish(task_name, seed, result)
                    update_task(task_name)
        except BaseException as exception:
            worker_error = f"{slot.name}: {type(exception).__name__}: {exception}"
            with status_lock:
                failures.append(worker_error)
            cancel_suite()
        finally:
            try:
                evaluation.stop_owned_process(server)
            except Exception as exception:
                worker_error = f"{slot.name}: owned server cleanup failed: {exception}"
                with status_lock:
                    failures.append(worker_error)
                cancel_suite()
            update_worker(slot, {"state": "failed" if worker_error else "stopped", "error": worker_error})

    previous_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for stop_signal in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[stop_signal] = signal.signal(stop_signal, cancel_suite)
    try:
        with ThreadPoolExecutor(max_workers=len(slots), thread_name_prefix="native-episode") as executor:
            futures = [executor.submit(consume_candidates, slot) for slot in slots]
            for future in futures:
                future.result()
    except BaseException as exception:
        failures.append(f"{type(exception).__name__}: {exception}")
        cancel_suite()
    finally:
        for stop_signal, previous_handler in previous_handlers.items():
            signal.signal(stop_signal, previous_handler)
    if cancellation.is_set() and not failures:
        failures.append("Evaluation interrupted; no unfinished candidate was retried")
    ordered_results = [
        summarize_concurrent_task(
            task_name, eval_root / task_name, scheduler.budgets[task_name],
            episodes=config.episodes, max_seed_attempts=config.max_seed_attempts,
        )
        for task_name in config.tasks
    ]
    complete = all(result["coverage_complete"] for result in ordered_results) and not failures
    summary = {
        "state": "completed" if complete else "failed", "coverage_complete": complete,
        "requested_tasks": list(config.tasks), "requested_episodes": len(config.tasks) * config.episodes,
        "completed_episodes": sum(result["completed_episodes"] for result in ordered_results), "tasks": ordered_results,
        "unstarted_tasks": [result["task_name"] for result in ordered_results if not result["cases"]],
        "native_official_equal_task_success_rate": (
            sum(result["native_official_success_rate"] for result in ordered_results) / len(config.tasks) if complete else None
        ),
        "errors": failures, "error": "; ".join(failures) or None,
        "workers_per_gpu": config.workers_per_gpu, "policy_server_count": len(slots),
        "started_at": wall_start, "finished_at": time.time(), "evaluation_wall_seconds": time.time() - wall_start,
    }
    evaluation.write_atomic_json(eval_root / "summary.json", summary)
    status["state"] = summary["state"]
    for result in ordered_results:
        status["tasks"][result["task_name"]].update(
            state="completed" if result["coverage_complete"] else "failed",
            completed_episodes=result["completed_episodes"], problems=result["problems"],
        )
    evaluation.write_atomic_json(eval_root / "status.json", status)
    return summary
