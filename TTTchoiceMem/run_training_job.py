"""Run an existing frozen multi-GPU training plan, then its native evaluation.

Canonical plan keys are job_root, run_dir, eval_root, allowed_gpus,
environment (string-valued child environment overrides), training_command
(an argv list), and evaluation_config (EvaluationConfig fields). No shell
evaluation, implicit budget changes, manifest preparation, or retries occur.
Use --background for a detached, PID/status/log-tracked orchestrator. A later
--resume CHECKPOINT is explicit and reuses the same frozen plan/job directory.

Historical 448/30000 plans remain readable. The 64/55000 recipe uses four
ranks; the 128/55000 recipe uses eight. Either 55000-step recipe can select
diffusion or OFT independently of its batch size.
Both keep one original 16-query group per forward and accumulation one.
Completion gates use the immutable plan's budget, never a historical default.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

from eval_multitask import (
    ALLOWED_GPU_IDS, EPISODES_PER_TASK, EvaluationConfig, GLOBAL_BATCH_SIZE, GROUP_SIZE,
    PROJECT_ROOT, SEED_POOL_SIZE, TARGET_OPTIMIZER_STEPS, TASK_COUNT, build_suite_plan,
    check_gpu_utilization, check_port_free, read_json, require_absolute_path,
    required_accumulation, stop_owned_process, validate_checkpoint, write_atomic_json,
)


def command_option(command: tuple[str, ...] | list[str], *names: str, default=None):
    values = []
    for token_index, token in enumerate(command):
        if token in names:
            if token_index + 1 == len(command) or command[token_index + 1].startswith("--"):
                raise ValueError(f"Missing command value for {token}")
            values.append(command[token_index + 1])
        else:
            for name in names:
                if token.startswith(name + "="):
                    values.append(token[len(name) + 1:])
    if len(values) > 1:
        raise ValueError(f"Duplicate command option: {names}")
    return values[0] if values else default


def training_tasks(command: tuple[str, ...]) -> tuple[str, ...]:
    if command.count("--tasks") != 1:
        raise ValueError("The frozen training command needs exactly one --tasks list")
    task_names = []
    for token in command[command.index("--tasks") + 1:]:
        if token.startswith("--"):
            break
        task_names.append(token)
    return tuple(task_names)


def training_cameras(command: tuple[str, ...]) -> tuple[str, ...]:
    from choice_mem.config import SUPPORTED_CAMERAS, validate_cameras

    option_indices = [index for index, token in enumerate(command) if token == "--cameras" or token.startswith("--cameras=")]
    if not option_indices:
        return SUPPORTED_CAMERAS
    if len(option_indices) != 1:
        raise ValueError("The frozen training command needs at most one --cameras list")
    option_index = option_indices[0]
    option_token = command[option_index]
    camera_names = [option_token.split("=", 1)[1]] if "=" in option_token else []
    for token in command[option_index + 1:]:
        if token.startswith("--"):
            break
        camera_names.append(token)
    return tuple(validate_cameras(camera_names))


def check_training_gpu_memory(plan) -> None:
    if plan.target_optimizer_steps != 55000:
        return
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15,
    )
    free_memory = {
        int(line.split(",")[0]): float(line.split(",")[1])
        for line in result.stdout.splitlines() if line.strip()
    }
    # Both full-finetuning recipes require conservative initialization headroom.
    for gpu_id in plan.allowed_gpus:
        if free_memory.get(gpu_id, 0) < 35 * 1024:
            raise RuntimeError(f"GPU {gpu_id} has insufficient free memory for full finetuning: {free_memory.get(gpu_id)} MiB")


@dataclass(frozen=True)
class TrainingJobPlan:
    job_root: Path
    run_dir: Path
    eval_root: Path
    allowed_gpus: tuple[int, ...]
    environment: dict[str, str]
    training_command: tuple[str, ...]
    evaluation_config: EvaluationConfig
    source_sha256: str

    @property
    def target_optimizer_steps(self) -> int:
        return int(command_option(self.training_command, "--max-steps"))

    @property
    def global_batch_size(self) -> int:
        return int(command_option(self.training_command, "--global-batch-size"))

    @property
    def per_device_batch_size(self) -> int:
        return int(command_option(self.training_command, "--per-device-batch-size", default="16"))

    @property
    def gradient_accumulation(self) -> int:
        return int(command_option(
            self.training_command, "--gradient-accumulation", "--grad-accumulation", "--gradient-accumulation-steps",
        ))

    @classmethod
    def load(cls, plan_path: Path) -> "TrainingJobPlan":
        source_bytes = plan_path.read_bytes()
        payload = json.loads(source_bytes)
        if not isinstance(payload, dict):
            raise ValueError("Frozen plan must be a JSON object")
        plan = cls(
            job_root=require_absolute_path(payload["job_root"], "job_root"),
            run_dir=require_absolute_path(payload["run_dir"], "run_dir"),
            eval_root=require_absolute_path(payload["eval_root"], "eval_root"),
            allowed_gpus=tuple(payload["allowed_gpus"]), environment=dict(payload["environment"]),
            training_command=tuple(payload["training_command"]),
            evaluation_config=EvaluationConfig.from_dict(payload["evaluation_config"]),
            source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        )
        plan.validate()
        return plan

    def validate(self) -> None:
        if not self.allowed_gpus or len(set(self.allowed_gpus)) != len(self.allowed_gpus):
            raise ValueError("allowed_gpus must list unique physical GPU ids")
        if any(type(gpu_id) is not int or gpu_id not in ALLOWED_GPU_IDS for gpu_id in self.allowed_gpus):
            raise ValueError(f"Physical GPU ids must be unique values in {ALLOWED_GPU_IDS}")
        world_size = len(self.allowed_gpus)
        recipe = (self.global_batch_size, self.target_optimizer_steps)
        expected_action_head = command_option(self.training_command, "--action-head", default="diffusion")
        if expected_action_head not in {"diffusion", "oft"}:
            raise ValueError("The frozen action head must be diffusion or oft")
        if recipe == (64, 55000):
            if world_size != 4 or self.per_device_batch_size != 16:
                raise ValueError("The 64/55000 recipe requires four ranks with per-device batch 16")
            expected_accumulation = 1
            expected_save_every = "5000"
        elif recipe == (128, 55000):
            if world_size != 8 or self.per_device_batch_size != 16:
                raise ValueError("The 128/55000 recipe requires eight ranks with per-device batch 16")
            expected_accumulation = 1
            expected_save_every = "5000"
        elif recipe == (GLOBAL_BATCH_SIZE, TARGET_OPTIMIZER_STEPS):
            expected_accumulation = required_accumulation(world_size)
            expected_save_every = "1000"
            if expected_action_head != "diffusion":
                raise ValueError("Historical 448/30000 plans require diffusion")
            if self.per_device_batch_size != GROUP_SIZE:
                raise ValueError("Historical plans require per-device batch 16")
        else:
            raise ValueError("Unsupported frozen budget; authorized recipes are 448/30000, 64/55000 or 128/55000")
        if not all(isinstance(key, str) and isinstance(value, str) for key, value in self.environment.items()):
            raise ValueError("Plan environment must contain string keys and values")
        visible_devices = ",".join(str(gpu_id) for gpu_id in self.allowed_gpus)
        if self.environment.get("CUDA_VISIBLE_DEVICES") != visible_devices:
            raise ValueError(f"Plan must explicitly freeze CUDA_VISIBLE_DEVICES={visible_devices}")
        if not set(self.evaluation_config.gpu_ids).issubset(self.allowed_gpus):
            raise ValueError("Evaluation may use only the frozen allowed GPUs")
        if self.evaluation_config.expected_training_steps != self.target_optimizer_steps:
            raise ValueError("Training and evaluation completion gates must use the same frozen step budget")
        if self.evaluation_config.expected_action_head != expected_action_head:
            raise ValueError("Training and evaluation must freeze the same expected action head")
        expected_cameras = self.evaluation_config.expected_cameras
        if expected_cameras is not None and training_cameras(self.training_command) != expected_cameras:
            raise ValueError("Training and evaluation must freeze the same ordered cameras")
        training_cameras(self.training_command)
        if len(self.training_command) < 3 or any(not isinstance(token, str) or not token for token in self.training_command):
            raise ValueError("training_command must be a nonempty argv list, never a shell command")
        command = self.training_command
        is_torchrun = Path(command[0]).name == "torchrun" or (
            len(command) > 3 and command[1:3] == ("-m", "torch.distributed.run")
        )
        if not is_torchrun or not any(Path(token).name == "train_multitask.py" for token in command):
            raise ValueError("Training must invoke train_multitask.py through torchrun")
        required_options = {
            ("--nproc-per-node", "--nproc_per_node"): str(world_size),
            ("--group-size",): str(GROUP_SIZE), ("--global-batch-size",): str(self.global_batch_size),
            ("--gradient-accumulation", "--grad-accumulation", "--gradient-accumulation-steps"): str(expected_accumulation),
            ("--max-steps",): str(self.target_optimizer_steps),
            ("--save-every",): expected_save_every, ("--snapshot-every",): "5000",
        }
        for option_names, expected_value in required_options.items():
            if command_option(command, *option_names) != expected_value:
                raise ValueError(f"Frozen command requires {option_names[0]}={expected_value}")
        if command_option(command, "--nnodes", default="1") != "1":
            raise ValueError("Only single-node DDP is allowed")
        if "--standalone" in command or command_option(command, "--rdzv-endpoint", "--rdzv_endpoint") is not None:
            raise ValueError("Use an explicit local --master-port, not an uncheckable rendezvous endpoint")
        if command_option(command, "--resume") is not None:
            raise ValueError("Resume must be requested explicitly on run_training_job.py, not baked into the plan")
        if any(command_option(command, option_name) is not None for option_name in ("--stop-after", "--benchmark-updates")):
            raise ValueError("A formal job cannot contain a bounded preflight option")
        if self.target_optimizer_steps == 55000:
            fixed_options = {
                "--model-revision": "2fc06364715b967f1860aea9cf38778875588b17",
                "--horizon": "16", "--execute-prefix": "4",
                "--mem-length": "16", "--retrieval-layers": "2", "--per-token-size": "256",
                "--backbone-learning-rate": "0.00001", "--learning-rate": "0.0001",
                "--minimum-backbone-learning-rate": "0.000001", "--minimum-head-learning-rate": "0.000005",
                "--weight-decay": "0.005", "--warmup-ratio": "0.05", "--gradient-clip-norm": "2.5",
                "--seed": "42", "--validation-fraction": "0.1",
            }
            if expected_action_head == "diffusion":
                fixed_options["--diffusion-model-type"] = "DiT-L"
            else:
                fixed_options["--action-head"] = "oft"
                fixed_options["--oft-hidden-dim"] = "256"
            for option_name, expected_value in fixed_options.items():
                if command_option(command, option_name) != expected_value:
                    raise ValueError(f"The {self.global_batch_size}/55000 recipe requires {option_name}={expected_value}")
        run_directory = command_option(command, "--run-dir")
        if run_directory is None or require_absolute_path(run_directory, "--run-dir") != self.run_dir:
            raise ValueError("Frozen command --run-dir must equal plan run_dir")
        for option_name in ("--data-root", "--manifest", "--model-path"):
            value = command_option(command, option_name)
            if value is None:
                raise ValueError(f"Frozen training command requires {option_name}")
            require_absolute_path(value, option_name)
        if command_option(command, "--model-revision") is None:
            raise ValueError("Freeze an explicit --model-revision")
        if training_tasks(command) != self.evaluation_config.tasks:
            raise ValueError("Training and evaluation must freeze the same ordered ten-task list")
        for field_name, expected_value in (("--backbone", "qwen35"), ("--action-head", expected_action_head)):
            if command_option(command, field_name, default=expected_value) != expected_value:
                raise ValueError(f"This job requires {field_name}={expected_value}")
        if "--freeze-backbone" in command:
            raise ValueError("This job requires full fine-tuning, not a frozen backbone")
        if "--unfreeze-backbone" not in command:
            raise ValueError("Full fine-tuning must be requested explicitly with --unfreeze-backbone")
        # The verified 4-rank allocation peaks near 21.4 GB only with activation
        # checkpointing; without it the step OOMs on the shared GPUs.
        if "--gradient-checkpointing" not in command:
            raise ValueError("This allocation requires --gradient-checkpointing")
        directories = (self.job_root, self.run_dir, self.eval_root)
        if len(set(directories)) != 3 or self.job_root.is_relative_to(self.run_dir) or self.job_root.is_relative_to(self.eval_root):
            raise ValueError("Job root must not be identical to or nested inside training/evaluation outputs")
        if self.run_dir.is_relative_to(self.eval_root) or self.eval_root.is_relative_to(self.run_dir):
            raise ValueError("Training/evaluation directories must be independent")
        self.training_port()

    def training_port(self) -> int:
        host = command_option(self.training_command, "--master-addr", "--master_addr", default="127.0.0.1")
        if host not in {"127.0.0.1", "localhost"}:
            raise ValueError("torchrun must use a local master address")
        port = int(command_option(self.training_command, "--master-port", "--master_port", default="29500"))
        if not 1 <= port <= 65535 or port in self.evaluation_config.ports:
            raise ValueError("torchrun needs a valid, separate master port")
        return port


def build_evaluation_command(plan: TrainingJobPlan, checkpoint: Path) -> list[str]:
    config = plan.evaluation_config
    command = [
        str(config.qwen_python), str(PROJECT_ROOT / "eval_multitask.py"), "--checkpoint", str(checkpoint),
        "--eval-root", str(plan.eval_root), "--rmbench-root", str(config.rmbench_root),
        "--qwen-python", str(config.qwen_python), "--env-python", str(config.env_python),
        "--tasks", *config.tasks, "--gpu-ids", *(str(gpu_id) for gpu_id in config.gpu_ids),
        "--ports", *(str(port) for port in config.ports), "--episodes", str(EPISODES_PER_TASK), "--seed", "0",
        "--instruction-type", "unseen", "--task-config", "demo_clean",
        "--max-seed-attempts", str(SEED_POOL_SIZE),
        "--server-timeout", str(config.server_timeout),
        "--expected-training-steps", str(plan.target_optimizer_steps),
        "--expected-action-head", config.expected_action_head,
    ]
    if config.workers_per_gpu != 1:
        command.extend(["--workers-per-gpu", str(config.workers_per_gpu)])
    if config.expected_cameras is not None:
        command.extend(["--expected-cameras", *config.expected_cameras])
    if config.minimum_free_gpu_gib_per_worker != 8.0:
        command.extend(["--minimum-free-gpu-gib-per-worker", str(config.minimum_free_gpu_gib_per_worker)])
    return command


def resolve_completed_checkpoint(plan: TrainingJobPlan) -> Path:
    progress = read_json(plan.run_dir / "progress.json")
    step_fields = [progress[field_name] for field_name in ("optimizer_steps", "completed_steps", "step") if field_name in progress]
    if not step_fields or any(type(step) is not int or step != plan.target_optimizer_steps for step in step_fields):
        raise ValueError(f"Training progress has not reached exactly {plan.target_optimizer_steps} optimizer steps")
    if progress.get("status", progress.get("state")) != "completed":
        raise ValueError("Training progress must explicitly record completed status")
    # The trainer's own field is final_snapshot; older runs may instead record
    # final_checkpoint or checkpoint. Names are read, never guessed.
    final_checkpoint = (
        progress.get("final_snapshot") or progress.get("final_checkpoint") or progress.get("checkpoint")
    )
    if not isinstance(final_checkpoint, str) or not final_checkpoint:
        raise ValueError("Completed progress must name final_snapshot, final_checkpoint or checkpoint")
    checkpoint = Path(final_checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = plan.run_dir / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_relative_to(plan.run_dir) or not checkpoint.is_file():
        raise ValueError("Final checkpoint must be an existing file inside the frozen training run")
    checkpoint_options = {"expected_step": plan.target_optimizer_steps}
    if plan.evaluation_config.expected_action_head != "diffusion":
        checkpoint_options["expected_action_head"] = plan.evaluation_config.expected_action_head
    if plan.evaluation_config.expected_cameras is not None:
        checkpoint_options["expected_cameras"] = plan.evaluation_config.expected_cameras
    contract = validate_checkpoint(checkpoint, **checkpoint_options)
    if plan.target_optimizer_steps == 55000:
        manifest = read_json(Path(command_option(plan.training_command, "--manifest")))
        protocol = contract["protocol"]
        if contract["normalization"] != manifest["normalization"] or protocol["manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("Final checkpoint data manifest or normalization differs from the frozen local dataset")
        if protocol["train_episode_ids"] != manifest["normalization"]["training_episode_ids"]:
            raise ValueError("Final checkpoint training episode split differs from the frozen local dataset")
    model_revision = command_option(plan.training_command, "--model-revision")
    model_source = command_option(plan.training_command, "--model-path")
    if contract["config"].get("model_revision") != model_revision or contract["config"].get("model_source") != model_source:
        raise ValueError("Final checkpoint model source/revision differs from the frozen training command")
    if plan.target_optimizer_steps == 55000:
        if tuple(contract["config"].get("cameras", ())) != training_cameras(plan.training_command):
            raise ValueError("Final checkpoint cameras differ from the frozen training command")
        optimization = contract["protocol"].get("optimization", {})
        if optimization.get("world_size") != len(plan.allowed_gpus) or optimization.get("global_batch_size") != plan.global_batch_size:
            raise ValueError("Final checkpoint batch/world size differs from the frozen training command")
    return checkpoint


def validate_resume(plan: TrainingJobPlan, resume: Path | None) -> None:
    if plan.eval_root.exists():
        raise FileExistsError(f"Refusing to reuse evaluation output: {plan.eval_root}")
    if resume is None:
        frozen_plan_path = plan.job_root / "plan.json"
        only_frozen_plan = (
            plan.job_root.is_dir() and set(plan.job_root.iterdir()) == {frozen_plan_path}
            and hashlib.sha256(frozen_plan_path.read_bytes()).hexdigest() == plan.source_sha256
        )
        if plan.run_dir.exists() or (plan.job_root.exists() and not only_frozen_plan):
            raise FileExistsError("Job/training outputs exist; a future restart requires explicit --resume CHECKPOINT")
        return
    if not resume.is_file() or not resume.is_relative_to(plan.run_dir):
        raise ValueError("An explicit resume checkpoint must exist inside the frozen run_dir")
    lock_path = plan.job_root / "orchestrator.lock"
    if lock_path.exists():
        with lock_path.open("r", encoding="utf-8") as lock_stream:
            try:
                fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exception:
                raise RuntimeError("The original orchestrator is still running; refusing concurrent resume") from exception
    status = read_json(plan.job_root / "status.json")
    if status.get("plan_sha256") != plan.source_sha256:
        raise ValueError("Refusing to refreeze or alter the original job plan on resume")
    if status.get("state") in {"completed", "evaluating", "evaluation_failed"}:
        raise ValueError("A completed/evaluated job cannot silently retrain or rerun evaluation")
    frozen_plan_bytes = (plan.job_root / "plan.json").read_bytes()
    if hashlib.sha256(frozen_plan_bytes).hexdigest() != plan.source_sha256:
        raise ValueError("Frozen plan copy does not match the explicit resume plan")


def prepare_job(plan: TrainingJobPlan, plan_path: Path, resume: Path | None) -> None:
    validate_resume(plan, resume)
    source_bytes = plan_path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != plan.source_sha256:
        raise ValueError("Plan changed between validation and preparation; refusing to refreeze")
    if resume is None and not plan.job_root.exists():
        plan.job_root.mkdir(parents=True, exist_ok=False)
        # Preserve the user's already frozen JSON bytes; do not synthesize a new plan.
        with (plan.job_root / "plan.json").open("xb") as stream:
            stream.write(source_bytes)
            stream.flush()
            os.fsync(stream.fileno())
    write_atomic_json(plan.job_root / "status.json", {
        "state": "prepared", "plan_sha256": plan.source_sha256, "pid": None,
        "resume": str(resume) if resume is not None else None, "target_optimizer_steps": plan.target_optimizer_steps,
        "allowed_gpus": list(plan.allowed_gpus),
        "world_size": len(plan.allowed_gpus),
        "gradient_accumulation": plan.gradient_accumulation,
        "per_device_batch_size": plan.per_device_batch_size,
        "global_batch_size": plan.global_batch_size,
        "requested_evaluation_episodes": 500,
        "requested_seed_attempts_per_task": SEED_POOL_SIZE,
        "evaluation_started": False,
    })


def run_job(plan: TrainingJobPlan, resume: Path | None = None) -> int:
    """Execute a prepared job once. Tests substitute the external processes."""
    lock_path = plan.job_root / "orchestrator.lock"
    # Kernel-owned locks release even after SIGKILL; stale lockfiles never
    # obstruct an explicitly requested future resume.
    lock_stream = lock_path.open("a", encoding="utf-8")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exception:
        lock_stream.close()
        raise RuntimeError("An orchestrator already owns this job") from exception
    process = None
    status = read_json(plan.job_root / "status.json")
    status.update({"pid": os.getpid(), "started_at": time.time(), "state": "preflight"})
    previous_handlers = {}

    def record_status(state: str, **values) -> None:
        status.update({"state": state, "updated_at": time.time(), **values})
        write_atomic_json(plan.job_root / "status.json", status)

    def interrupt_job(signum, frame):
        raise KeyboardInterrupt(f"Job received signal {signum}")

    for stop_signal in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[stop_signal] = signal.signal(stop_signal, interrupt_job)
    environment = dict(os.environ)
    environment.update(plan.environment)
    environment["PYTHONUNBUFFERED"] = "1"
    command = list(plan.training_command)
    if resume is not None:
        command.extend(["--resume", str(resume)])
    try:
        if plan.eval_root.exists() or (resume is None and plan.run_dir.exists()):
            raise FileExistsError("Outputs appeared after preparation; refusing to overwrite")
        check_gpu_utilization(plan.allowed_gpus)
        check_training_gpu_memory(plan)
        check_port_free("127.0.0.1", plan.training_port())
        with (plan.job_root / "training.log").open("a", encoding="utf-8") as training_log:
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment, stdout=training_log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            record_status("training", training_pid=process.pid, training_command=command, training_started_at=time.time())
            training_returncode = process.wait()
            stop_owned_process(process)
            process = None
        training_finished_at = time.time()
        record_status("checking_training", training_returncode=training_returncode,
                      training_finished_at=training_finished_at,
                      training_wall_seconds=training_finished_at - status["training_started_at"])
        if training_returncode != 0:
            record_status("training_failed", error="Training exited nonzero; evaluation was not launched")
            return 1
        checkpoint = resolve_completed_checkpoint(plan)
        record_status("training_completed", final_snapshot=str(checkpoint),
                      completed_optimizer_steps=plan.target_optimizer_steps)
        evaluation_command = build_evaluation_command(plan, checkpoint)
        with (plan.job_root / "evaluation.log").open("x", encoding="utf-8") as evaluation_log:
            process = subprocess.Popen(evaluation_command, cwd=PROJECT_ROOT, env=environment,
                                       stdout=evaluation_log, stderr=subprocess.STDOUT, start_new_session=True)
            record_status("evaluating", evaluation_pid=process.pid, evaluation_started=True, evaluation_started_at=time.time())
            evaluation_returncode = process.wait()
            stop_owned_process(process)
            process = None
        evaluation_finished_at = time.time()
        record_status("checking_evaluation", evaluation_finished_at=evaluation_finished_at,
                      evaluation_wall_seconds=evaluation_finished_at - status["evaluation_started_at"])
        if evaluation_returncode != 0:
            record_status("evaluation_failed", evaluation_returncode=evaluation_returncode,
                          error="Native suite failed/incomplete; no retry was attempted")
            return 1
        summary = read_json(plan.eval_root / "summary.json")
        expected_episodes = TASK_COUNT * EPISODES_PER_TASK
        if (
            summary.get("state") != "completed" or summary.get("coverage_complete") is not True
            or summary.get("completed_episodes") != expected_episodes
        ):
            raise ValueError(f"Evaluation exit zero without verified complete {expected_episodes}-episode coverage")
        record_status("completed", evaluation_returncode=0, evaluation_summary=str(plan.eval_root / "summary.json"))
        return 0
    except KeyboardInterrupt as exception:
        record_status("interrupted", error=str(exception))
        return 130
    except Exception as exception:
        record_status("failed", error=f"{type(exception).__name__}: {exception}")
        return 1
    finally:
        # Ignore a second terminal signal while reaping our own process group.
        for stop_signal in previous_handlers:
            signal.signal(stop_signal, signal.SIG_IGN)
        try:
            stop_owned_process(process, timeout=120 if status.get("evaluation_started") else 15)
            status["finished_at"] = time.time()
            write_atomic_json(plan.job_root / "status.json", status)
        finally:
            for stop_signal, previous_handler in previous_handlers.items():
                signal.signal(stop_signal, previous_handler)
            lock_stream.close()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="Existing frozen JSON; never created/refrozen implicitly")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--background", action="store_true", help="Detach orchestrator, leaving PID/status/log tracking")
    parser.add_argument("--resume", type=Path, help="Explicit future restart from an existing checkpoint")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    arguments = parse_args(argv)
    plan_path = arguments.plan.expanduser().resolve(strict=True)
    plan = TrainingJobPlan.load(plan_path)
    resume = arguments.resume.expanduser().resolve(strict=True) if arguments.resume is not None else None
    if arguments.dry_run:
        validate_resume(plan, resume)
        command = list(plan.training_command) + (["--resume", str(resume)] if resume is not None else [])
        print(shlex.join(command))
        print(json.dumps({"job_root": str(plan.job_root), "environment": plan.environment,
                          "allowed_gpus": list(plan.allowed_gpus), "world_size": len(plan.allowed_gpus),
                          "gradient_accumulation": plan.gradient_accumulation,
                          "evaluation": build_suite_plan(plan.evaluation_config, Path("<progress.json final_snapshot>")),
                          "evaluation_gate": f"training exit 0 + completed {plan.target_optimizer_steps} steps + valid full final snapshot",
                          "executed": False}, indent=2))
        return 0
    if arguments.worker:
        frozen_plan = plan.job_root / "plan.json"
        if frozen_plan != plan_path or read_json(plan.job_root / "status.json").get("plan_sha256") != plan.source_sha256:
            raise ValueError("Internal worker requires its prepared, unchanged frozen plan")
        return run_job(plan, resume)
    prepare_job(plan, plan_path, resume)
    if not arguments.background:
        return run_job(plan, resume)
    command = [str(plan.evaluation_config.qwen_python), str(Path(__file__).resolve()),
               "--plan", str(plan.job_root / "plan.json"), "--worker"]
    if resume is not None:
        command.extend(["--resume", str(resume)])
    with (plan.job_root / "orchestrator.log").open("a", encoding="utf-8") as orchestrator_log:
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL, stdout=orchestrator_log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    # The worker owns status.json; the launcher never races its phase updates.
    write_atomic_json(plan.job_root / "process.json", {"pid": process.pid, "command": command, "launched_at": time.time()})
    print(json.dumps({"pid": process.pid, "status": str(plan.job_root / "status.json"),
                      "log": str(plan.job_root / "orchestrator.log"), "training_log": str(plan.job_root / "training.log")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
