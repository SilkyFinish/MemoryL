"""Native RMBench evaluation: selected official tasks, twenty or fifty episodes.

Each selected physical GPU owns one to three independent policy/client slots;
the default remains one task-level worker on each of the four GPUs allocated to
training. Multiple slots use bounded episode scheduling, not native batch mode.
Simulator/task/termination/recording code stays native.

Native expert checking runs *before* the policy ever acts. Seeds whose scripted
expert demonstration fails are therefore not episodes and are never counted.
Reaching a true fifty policy episodes per task therefore needs the official
expert-validation pool: 50 episodes * the native 50-seed pool = 2500 candidate
seeds (100000-102499). Skipped seeds are logged and consume pool budget; they
are never replaced by rerunning or discarding a policy success/failure, and an
exhausted pool is reported as incomplete coverage rather than a silent retry.
The explicit twenty-episode budget uses the same rule with 1000 candidates
(100000-100999); the legacy fifty-episode defaults remain unchanged.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import queue
import random
import shlex
import signal
import socket
import subprocess
import tempfile
import threading
import time

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RMBENCH_ROOT = Path("/mnt/data/dylan/d2/memory/RMBench")
EPISODES_PER_TASK = 50
TASK_COUNT = 10
TARGET_OPTIMIZER_STEPS = 30000
GROUP_SIZE = 16
GLOBAL_BATCH_SIZE = 448
FIRST_SEED = 100000
SEED_STRIDE = 1
# Official native default: 50 requested episodes * 50 candidate seeds each.
SEED_POOL_SIZE = EPISODES_PER_TASK * 50
LAST_SEED = FIRST_SEED + SEED_POOL_SIZE - 1
ALLOWED_GPU_IDS = tuple(range(8))
LEGACY_PORTS = tuple(range(18821, 18829))
CONCURRENT_PORTS = tuple(range(18861, 18885))
ALLOWED_PORTS = LEGACY_PORTS + CONCURRENT_PORTS
DEFAULT_GPU_IDS = (2, 3, 4, 6)
DEFAULT_PORTS = (18822, 18823, 18824, 18826)
AVAILABLE_CAMERAS = ("head_camera", "left_camera", "right_camera")


def resolve_seed_attempt_budget(episodes: int, max_seed_attempts: int | None = None) -> int:
    """Allow only authorized episode counts, each with the native 50x pool."""
    if type(episodes) is not int or episodes not in (20, 50):
        raise ValueError("episodes must be an integer equal to 20 or 50")
    required_attempts = episodes * 50
    if max_seed_attempts is not None and (
        type(max_seed_attempts) is not int or max_seed_attempts != required_attempts
    ):
        raise ValueError(f"max_seed_attempts must equal episodes * 50 = {required_attempts}")
    return required_attempts


def positive_integer(value: str) -> int:
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return value


def require_positive_integer(value, field_name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def read_native_infrastructure_failures(output_dir: Path) -> list[str]:
    """Recognize fatal child errors that native expert checking can mask as skips."""
    fatal_patterns = (
        "cuda error", "cuda runtime error", "cuda driver error", "cuda driver version is insufficient",
        "no cuda-capable device", "no cuda gpus are available", "cuda initialization error",
        "no kernel image", "invalid device function", "out of memory", "outofmemoryerror",
        "cublas_status_alloc_failed", "cublas_status_not_initialized", "cublas_status_execution_failed",
        "cublas_status_arch_mismatch", "cudnn_status_alloc_failed", "cudnn_status_execution_failed",
        "cannot allocate memory", "failed to allocate memory", "segmentation fault", "bus error",
    )
    native_log = output_dir / "native.log"
    if not native_log.is_file():
        return []
    failures = []
    with native_log.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if any(pattern in line.casefold() for pattern in fatal_patterns):
                failure = line.strip()
                if failure not in failures:
                    failures.append(failure)
                if len(failures) >= 20:
                    break
    return failures


def choose_default_ports(gpu_ids: tuple[int, ...], workers_per_gpu: int) -> tuple[int, ...]:
    """Keep the old allocation unchanged; new allocations use a separate range."""
    if type(workers_per_gpu) is not int or not 1 <= workers_per_gpu <= 3:
        raise ValueError("workers_per_gpu must be an integer from 1 to 3")
    if gpu_ids == DEFAULT_GPU_IDS and workers_per_gpu == 1:
        return DEFAULT_PORTS
    return CONCURRENT_PORTS[:len(gpu_ids) * workers_per_gpu]


def validate_cameras(cameras: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if not isinstance(cameras, (tuple, list)) or not 1 <= len(cameras) <= 3:
        raise ValueError("Cameras must be an ordered selection of one to three native cameras")
    if any(not isinstance(camera, str) or camera not in AVAILABLE_CAMERAS for camera in cameras):
        raise ValueError(f"Cameras must be selected from {AVAILABLE_CAMERAS}")
    if len(set(cameras)) != len(cameras):
        raise ValueError("Cameras must be ordered and unique")
    return tuple(cameras)


def required_accumulation(world_size: int) -> int:
    """The frozen budget is exact: world_size * group_size * accumulation == 448."""
    if world_size < 1 or GLOBAL_BATCH_SIZE % (world_size * GROUP_SIZE):
        raise ValueError(f"{world_size} ranks cannot reach an exact global batch of {GLOBAL_BATCH_SIZE}")
    return GLOBAL_BATCH_SIZE // (world_size * GROUP_SIZE)


def write_atomic_json(path: Path, payload: dict) -> None:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def read_native_tasks(rmbench_root: Path) -> tuple[str, ...]:
    """Read the audited flat task list without importing simulator dependencies."""
    task_names = []
    in_tasks = False
    with (rmbench_root / "env_cfg" / "eval" / "all_tasks.yml").open(encoding="utf-8") as stream:
        for raw_line in stream:
            content = raw_line.partition("#")[0].strip()
            if content == "tasks:":
                in_tasks = True
                continue
            if not in_tasks or not content:
                continue
            if not content.startswith("- "):
                break
            task_name = content[2:].strip().strip("\"'")
            if not task_name.isidentifier():
                raise ValueError(f"Invalid native task name: {task_name!r}")
            task_names.append(task_name)
    if len(task_names) != TASK_COUNT or len(set(task_names)) != TASK_COUNT:
        raise ValueError("The official all_tasks.yml must contain exactly ten unique tasks")
    return tuple(task_names)


def require_absolute_path(value: str | Path, field_name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{field_name} must be an absolute path: {value}")
    if field_name in {"qwen_python", "env_python"}:
        # Resolving a virtualenv's python symlink loses its pyvenv.cfg and
        # accidentally runs the base interpreter without model/simulator deps.
        return Path(os.path.abspath(path))
    return path.resolve()


def check_gpu_utilization(gpu_ids: tuple[int, ...] | list[int]) -> None:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15,
    )
    utilization = {}
    for line in result.stdout.splitlines():
        if line.strip():
            gpu_index, percentage = (part.strip() for part in line.split(","))
            utilization[int(gpu_index)] = float(percentage)
    for gpu_id in gpu_ids:
        if gpu_id not in utilization or not 0 <= utilization[gpu_id] < 50:
            raise RuntimeError(f"GPU {gpu_id} is missing or not below 50% utilization; refusing to launch")


def check_port_free(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        # No reuse flags: an existing owner (including a wildcard bind) must win.
        listener.bind((host, port))


def stop_owned_process(process: subprocess.Popen | None, timeout: float = 15) -> None:
    """Signal only a session created by us, including native ffmpeg/DDP children."""
    if process is None:
        return
    process_group = process.pid
    for stop_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process_group, stop_signal)
        except ProcessLookupError:
            process.wait()
            return
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            continue
        # A parent can exit while its renderer/ffmpeg children are still alive.
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                os.killpg(process_group, 0)
            except ProcessLookupError:
                return
            time.sleep(0.1)
    process.wait(timeout=timeout)


def wait_for_server(process: subprocess.Popen, host: str, port: int, timeout: float,
                    cancellation: threading.Event | None = None) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cancellation is not None and cancellation.is_set():
            raise RuntimeError("Evaluation cancelled before server readiness")
        if process.poll() is not None:
            raise RuntimeError(f"Owned policy server exited before readiness: {process.returncode}")
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"Policy server did not accept a connection within {timeout}s at {host}:{port}")


@dataclass(frozen=True)
class EvaluationConfig:
    tasks: tuple[str, ...]
    rmbench_root: Path
    qwen_python: Path
    env_python: Path
    gpu_ids: tuple[int, ...] = DEFAULT_GPU_IDS
    ports: tuple[int, ...] = DEFAULT_PORTS
    episodes: int = EPISODES_PER_TASK
    seed: int = 0
    instruction_type: str = "unseen"
    task_config: str = "demo_clean"
    action_type: str = "joint"
    expert_check: bool = True
    max_seed_attempts: int = SEED_POOL_SIZE
    horizon: int = 16
    execute_prefix: int = 4
    server_timeout: float = 120
    expected_training_steps: int = TARGET_OPTIMIZER_STEPS
    expected_action_head: str = "diffusion"
    workers_per_gpu: int = 1
    expected_cameras: tuple[str, ...] | None = None
    minimum_free_gpu_gib_per_worker: float = 8.0
    expected_backbone: str | None = None

    @classmethod
    def from_dict(cls, configuration: dict) -> "EvaluationConfig":
        configuration = dict(configuration)
        if configuration.get("tasks") is None:
            configuration["tasks"] = read_native_tasks(
                require_absolute_path(configuration["rmbench_root"], "rmbench_root"),
            )
        if "max_seed_attempts" not in configuration:
            configuration["max_seed_attempts"] = resolve_seed_attempt_budget(
                configuration.get("episodes", EPISODES_PER_TASK),
            )
        for field_name in ("tasks", "gpu_ids", "ports"):
            if field_name in configuration:
                configuration[field_name] = tuple(configuration[field_name])
        if configuration.get("expected_cameras") is not None:
            configuration["expected_cameras"] = validate_cameras(configuration["expected_cameras"])
        if "ports" not in configuration:
            configuration["ports"] = choose_default_ports(
                configuration.get("gpu_ids", DEFAULT_GPU_IDS), configuration.get("workers_per_gpu", 1),
            )
        for field_name in ("rmbench_root", "qwen_python", "env_python"):
            configuration[field_name] = require_absolute_path(configuration[field_name], field_name)
        config = cls(**configuration)
        config.validate()
        return config

    def validate(self) -> None:
        official_tasks = read_native_tasks(self.rmbench_root)
        if (
            not isinstance(self.tasks, (tuple, list)) or not self.tasks
            or any(not isinstance(task, str) or task not in official_tasks for task in self.tasks)
            or len(set(self.tasks)) != len(self.tasks)
        ):
            raise ValueError("Evaluation tasks must be a nonempty, unique subset of the official ten-task list")
        if type(self.max_seed_attempts) is not int:
            raise ValueError("max_seed_attempts must be an integer equal to episodes * 50")
        resolve_seed_attempt_budget(self.episodes, self.max_seed_attempts)
        required_values = {
            "seed": 0, "instruction_type": "unseen", "task_config": "demo_clean",
            "action_type": "joint", "expert_check": True,
            "horizon": 16, "execute_prefix": 4,
        }
        for field_name, required_value in required_values.items():
            actual_value = getattr(self, field_name)
            if type(actual_value) is not type(required_value) or actual_value != required_value:
                raise ValueError(f"Frozen evaluation requires {field_name}={required_value!r}")
        require_positive_integer(self.expected_training_steps, "expected_training_steps")
        if self.expected_action_head not in {"diffusion", "oft"}:
            raise ValueError("Expected action head must be diffusion or oft")
        if self.expected_backbone is not None and self.expected_backbone not in {"memoryvla", "qwen35"}:
            raise ValueError("Expected backbone must be memoryvla or qwen35")
        if type(self.workers_per_gpu) is not int or not 1 <= self.workers_per_gpu <= 3:
            raise ValueError("workers_per_gpu must be an integer from 1 to 3")
        if self.expected_cameras is not None:
            validate_cameras(self.expected_cameras)
        if (
            type(self.minimum_free_gpu_gib_per_worker) not in (float, int)
            or not math.isfinite(self.minimum_free_gpu_gib_per_worker)
            or self.minimum_free_gpu_gib_per_worker <= 0
        ):
            raise ValueError("minimum_free_gpu_gib_per_worker must be finite and positive")
        if not 1 <= len(self.gpu_ids) <= len(ALLOWED_GPU_IDS) or len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise ValueError("Use unique selected GPUs")
        if any(type(gpu_id) is not int or gpu_id not in ALLOWED_GPU_IDS for gpu_id in self.gpu_ids):
            raise ValueError(f"Physical GPU ids must be unique values in {ALLOWED_GPU_IDS}")
        if len(self.ports) != len(self.gpu_ids) * self.workers_per_gpu or len(set(self.ports)) != len(self.ports):
            raise ValueError("Each evaluation worker requires its own unique policy-server port")
        if any(type(port) is not int or port not in ALLOWED_PORTS for port in self.ports):
            raise ValueError("Use only reserved policy-server ports 18821-18828 or 18861-18884")
        if not 60 <= self.server_timeout <= 120:
            raise ValueError("Server readiness timeout must be between 60 and 120 seconds")
        required_paths = [
            self.qwen_python, self.env_python,
            self.rmbench_root / "scripts" / "eval_policy_xpolicylab.py",
            self.rmbench_root / "env_cfg" / "task_config" / "demo_clean.yml",
            *(self.rmbench_root / "envs" / f"{task_name}.py" for task_name in self.tasks),
        ]
        for required_path in required_paths:
            if not required_path.is_file():
                raise FileNotFoundError(required_path)

    def to_dict(self) -> dict:
        return {field_name: str(value) if isinstance(value, Path) else value for field_name, value in asdict(self).items()}


def _validate_source_assets(configuration: dict, *, legacy: bool = False) -> None:
    """Hash local assets only; random debug depends on metadata, not pretrained weights."""
    if configuration.get("backbone_initialization", "pretrained") == "random-frozen-debug":
        from choice_mem.checkpoint import validate_debug_source_assets

        validate_debug_source_assets(configuration)
        return
    from choice_mem.data import hash_file

    source = require_absolute_path(configuration["model_source"], "model_source")
    manifest = configuration.get("asset_manifest")
    if not isinstance(manifest, dict) or (not manifest and not legacy):
        raise ValueError("Final checkpoint requires a nonempty source asset_manifest")
    assets = {}
    declared_hashes = {}
    if configuration["backbone"] == "memoryvla" and source.is_file():
        if source.suffix == ".pt":
            assets = {"config.json": source.parent.parent / "config.json", "checkpoint": source}
        elif source.suffix == ".json":
            specification = read_json(source)
            assets = {"manifest": source, "checkpoint": Path(specification["checkpoint_path"])}
            assets.update({name: Path(path) for name, path in specification["local_vision_files"].items()})
            metadata = specification["llama_metadata"]
            assets.update({"llama/" + name: Path(metadata["path"]) / name for name in metadata["files"]})
            declared_hashes = {"checkpoint": specification["checkpoint_sha256"], **{
                "llama/" + name: digest for name, digest in metadata["files"].items()
            }}
            source_config = specification["config.json"]
            if hash_file(Path(source_config["path"])) != source_config["sha256"]:
                raise ValueError("Final checkpoint's official source asset changed: config.json")
        else:
            raise ValueError("Unsupported MemoryVLA model_source file")
        if set(assets) != set(manifest):
            raise ValueError("Final checkpoint source asset_manifest aliases are incomplete or unexpected")
    else:
        for name in manifest:
            relative_path = Path(name)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError("Source asset_manifest must use relative asset paths")
            assets[name] = source / relative_path
    for name, path in assets.items():
        expected_hash = manifest[name]
        if not legacy and (
            not isinstance(expected_hash, str) or len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)
        ):
            raise ValueError(f"Invalid source asset SHA256: {name}")
        if hash_file(path) != expected_hash or declared_hashes.get(name, expected_hash) != expected_hash:
            raise ValueError(f"Final checkpoint's official source asset changed: {name}")


def _validate_legacy_checkpoint(payload: dict, expected_step: int) -> None:
    """Keep the authorized v1 Qwen recipes separate from the v2 method contract."""
    if expected_step not in (30000, 55000) or (payload["config"]["action_head"] == "oft" and expected_step != 55000):
        raise ValueError("Legacy v1 requires an authorized 30000/55000 budget (OFT: 55000)")
    for field_name, required_value in {
        "backbone": "qwen35", "freeze_backbone": False, "dtype": "bfloat16", "group_size": 16,
        "horizon": 16, "execute_prefix": 4,
    }.items():
        if payload["config"].get(field_name) != required_value:
            raise ValueError(f"Final checkpoint must have {field_name}={required_value!r}")
    if expected_step != 55000:
        return
    required_model = {
        "model_revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "mem_length": 16, "retrieval_layers": 2, "per_token_size": 256, "image_size": [224, 224],
        "update_fused": False, "fusion_type": "gate", "consolidate_type": "tome",
    }
    if payload["config"]["action_head"] == "diffusion":
        required_model.update(diffusion_model_type="DiT-L", ddim_steps=10, cfg_scale=1.5)
    else:
        required_model["oft_hidden_dim"] = 256
    for field_name, expected_value in required_model.items():
        if payload["config"].get(field_name) != expected_value:
            raise ValueError(f"Final 55000 checkpoint requires {field_name}={expected_value!r}")
    optimization = payload["protocol"].get("optimization", {})
    if (optimization.get("world_size"), optimization.get("global_batch_size")) not in ((4, 64), (8, 128)):
        raise ValueError("Final 55000 checkpoint requires world/global batch 4/64 or 8/128 independently of head")
    for field_name, expected_value in {
        "max_steps": 55000, "per_device_batch_size": 16, "grad_accumulation": 1, "groups_per_forward": 1,
        "master_dtype": "float32", "autocast_dtype": "bfloat16", "gradient_checkpointing": True,
    }.items():
        if optimization.get(field_name) != expected_value:
            raise ValueError(f"Final 55000 checkpoint has wrong optimization contract: {field_name}")
    _validate_source_assets(payload["config"], legacy=True)


def _validate_v2_checkpoint(payload: dict, expected_step: int) -> None:
    configuration, protocol = payload["config"], payload["protocol"]
    for field_name in ("horizon", "execute_prefix"):
        require_positive_integer(configuration[field_name], field_name)
    if configuration["execute_prefix"] > configuration["horizon"]:
        raise ValueError("Checkpoint execute_prefix must not exceed horizon")
    method_fields = ("history_sampling", "image_aug", "repeated_diffusion_steps", "mask_padding_loss", "inference_timestep_mode")
    for field_name in ("action_dim", "horizon", "execute_prefix", "group_size", "image_size", *method_fields):
        if protocol.get(field_name) != configuration.get(field_name) or field_name not in protocol:
            raise ValueError(f"Checkpoint configuration and protocol disagree: {field_name}")
    requested = protocol.get("requested_configuration")
    if isinstance(requested, dict) and configuration.get("backbone_initialization", "pretrained") == "pretrained":
        from choice_mem.checkpoint import INITIALIZATION_DEFAULTS

        requested = {**INITIALIZATION_DEFAULTS, **requested}
    enriched_fields = {"preprocessing", "feature_dimensions", "asset_manifest"}
    if not isinstance(requested, dict) or set(configuration).difference(enriched_fields, requested):
        raise ValueError("Checkpoint lacks its requested_configuration contract")
    for field_name, expected_value in requested.items():
        if field_name in enriched_fields or (field_name == "model_revision" and expected_value is None):
            continue
        if configuration.get(field_name) != expected_value:
            raise ValueError(f"Checkpoint requested_configuration disagrees: {field_name}")
    if configuration["action_dim"] != 14 or protocol.get("target_offset") != 1:
        raise ValueError("Checkpoint must preserve native 14D next-recorded-action alignment")
    digest = protocol.get("manifest_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("Checkpoint requires a valid training manifest_sha256")
    train_ids, validation_ids = protocol.get("train_episode_ids"), protocol.get("validation_episode_ids")
    for field_name, episode_ids in (("train_episode_ids", train_ids), ("validation_episode_ids", validation_ids)):
        if (
            not isinstance(episode_ids, list) or (field_name == "train_episode_ids" and not episode_ids)
            or any(not isinstance(identifier, str) or not identifier for identifier in episode_ids)
            or len(set(episode_ids)) != len(episode_ids)
        ):
            raise ValueError(f"Invalid checkpoint manifest {field_name}")
    normalization = payload["normalization"]
    if set(train_ids).intersection(validation_ids) or normalization.get("training_episode_ids") != train_ids:
        raise ValueError("Checkpoint manifest train/validation IDs or train-only normalization disagree")
    if normalization.get("action_dim") != 14 or normalization.get("target_offset") != 1:
        raise ValueError("Checkpoint normalization action/target contract disagrees")
    statistics = normalization.get("action", normalization)
    lower, upper = statistics.get("q01", []), statistics.get("q99", [])
    if len(lower) != 14 or len(upper) != 14 or any(
        type(value) not in (int, float) or not math.isfinite(value) for value in (*lower, *upper)
    ) or any(high < low for low, high in zip(lower, upper)):
        raise ValueError("Checkpoint normalization quantiles must be finite ordered 14D values")
    sampling = protocol.get("sampling", {})
    tasks, per_task = sampling.get("tasks"), sampling.get("per_task", {})
    if (
        not isinstance(tasks, list) or not tasks or any(not isinstance(task, str) or not task for task in tasks)
        or len(set(tasks)) != len(tasks) or set(per_task) != set(tasks)
    ):
        raise ValueError("Checkpoint sampling requires nonempty distinct tasks and per-task manifest coverage")
    for task in tasks:
        require_positive_integer(per_task[task].get("train_episodes"), f"{task}.train_episodes")
        validation_count = per_task[task].get("validation_episodes")
        if type(validation_count) is not int or validation_count < 0:
            raise ValueError("Invalid per-task validation episode coverage")
    if (
        sum(per_task[task]["train_episodes"] for task in tasks) != len(train_ids)
        or sum(per_task[task]["validation_episodes"] for task in tasks) != len(validation_ids)
        or ("tasks" in protocol and set(protocol["tasks"]) != set(tasks))
        or sampling.get("history_sampling", configuration["history_sampling"]) != configuration["history_sampling"]
    ):
        raise ValueError("Checkpoint sampling and manifest coverage disagree")
    optimization = protocol.get("optimization", {})
    for field_name in ("max_steps", "world_size", "group_size", "per_device_batch_size", "groups_per_forward", "grad_accumulation", "global_batch_size"):
        require_positive_integer(optimization.get(field_name), f"optimization.{field_name}")
    if optimization["max_steps"] != expected_step:
        raise ValueError("Final checkpoint has not completed its optimization.max_steps budget")
    if (
        optimization["group_size"] != configuration["group_size"]
        or optimization["per_device_batch_size"] != optimization["group_size"] * optimization["groups_per_forward"]
        or optimization["global_batch_size"] != optimization["world_size"] * optimization["per_device_batch_size"] * optimization["grad_accumulation"]
    ):
        raise ValueError("Checkpoint optimization group/batch contract disagrees")
    if optimization.get("master_dtype") != "float32" or optimization.get("autocast_dtype") != configuration["dtype"]:
        raise ValueError("Checkpoint optimization master/autocast dtype contract disagrees")
    if configuration.get("backbone_initialization", "pretrained") == "random-frozen-debug" and (
        optimization.get("frozen_backbone_dtype") != "bfloat16"
    ):
        raise ValueError("Debug checkpoint optimization frozen_backbone_dtype must be bfloat16")
    for field_name in ("learning_rate", "backbone_learning_rate", "memory_head_learning_rate", "gradient_clip_norm", "weight_decay", "warmup_ratio"):
        value = optimization.get(field_name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (field_name not in {"weight_decay", "warmup_ratio"} and value == 0):
            raise ValueError(f"Invalid checkpoint optimization {field_name}")
    recipe = optimization.get("recipe")
    if optimization.get("optimizer") != "AdamW" or recipe not in {"memoryvla", "qwen-legacy"}:
        raise ValueError("Unsupported checkpoint optimization recipe/optimizer")
    if recipe == "memoryvla":
        if (
            optimization.get("schedule") != "constant" or optimization["warmup_ratio"] != 0
            or optimization["learning_rate"] != optimization["backbone_learning_rate"]
            or optimization["learning_rate"] != optimization["memory_head_learning_rate"]
            or any(optimization.get(field_name) is not None for field_name in ("minimum_backbone_learning_rate", "minimum_head_learning_rate"))
        ):
            raise ValueError("Checkpoint MemoryVLA single-LR constant optimization contract disagrees")
    elif optimization.get("schedule") != "cosine with linear warmup" or not 0 <= optimization["warmup_ratio"] < 1:
        raise ValueError("Checkpoint legacy cosine optimization contract disagrees")
    state = payload.get("training_state")
    if state is not None:
        scheduler = state.get("scheduler", {})
        if (
            state.get("schema") != "choice-memory-multitask-training-v2" or state["schema"] != protocol.get("trainer")
            or type(scheduler.get("completed_steps")) is not int or scheduler["completed_steps"] != expected_step
            or type(scheduler.get("total_steps")) is not int or scheduler["total_steps"] != expected_step
            or (recipe == "memoryvla" and scheduler.get("schedule") != "constant")
        ):
            raise ValueError("Checkpoint training_state scheduler has not completed the declared training budget")
        micro_index = expected_step * optimization["grad_accumulation"] * optimization["world_size"]
        if (
            state.get("sampler") != sampling or len(state.get("rank_rng_states", [])) != optimization["world_size"]
            or type(state.get("next_global_micro_index")) is not int or state["next_global_micro_index"] != micro_index
            or type(state.get("next_global_group_index")) is not int
            or state["next_global_group_index"] != micro_index * optimization["groups_per_forward"]
            or not isinstance(payload.get("optimizer"), dict)
        ):
            raise ValueError("Checkpoint training_state sampler/optimizer position disagrees")
    elif "optimizer" in payload:
        raise ValueError("Checkpoint optimizer lacks its training_state completion contract")
    _validate_source_assets(configuration)


def validate_checkpoint(
    checkpoint: Path, expected_step: int = TARGET_OPTIMIZER_STEPS, expected_action_head: str = "diffusion",
    expected_cameras: tuple[str, ...] | None = None, expected_backbone: str | None = None,
    allow_debug_checkpoint: bool = False,
) -> dict:
    """Validate a completed native policy on CPU, never instantiate a VLM or simulator.

    v1 retains the frozen Qwen recipes. v2 uses the saved method/data/optimizer
    contract; final inference snapshots need no resumable optimizer/training_state.
    Debug opt-in is deliberately not exposed by the official suite CLI/config.
    """
    require_positive_integer(expected_step, "expected_step")
    if expected_action_head not in {"diffusion", "oft"}:
        raise ValueError("Unsupported expected action head")
    if expected_backbone is not None and expected_backbone not in {"memoryvla", "qwen35"}:
        raise ValueError("Expected backbone must be memoryvla or qwen35")
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ValueError(f"Missing or empty final checkpoint: {checkpoint}")
    from choice_mem.checkpoint import CHECKPOINT_SCHEMA, LEGACY_CHECKPOINT_SCHEMA, read_checkpoint
    import torch

    payload = read_checkpoint(checkpoint)
    if type(payload["step"]) is not int or payload["step"] != expected_step:
        raise ValueError(f"Final checkpoint step must equal {expected_step}")
    configuration = payload["config"]
    debug = configuration.get("backbone_initialization", "pretrained") == "random-frozen-debug"
    if debug and allow_debug_checkpoint is not True:
        raise ValueError("Debug checkpoint requires explicit allow_debug_checkpoint=True; not an official pretrained metric")
    if configuration.get("backbone") not in {"memoryvla", "qwen35"}:
        raise ValueError("Final checkpoint must use the memoryvla or qwen35 backbone")
    if expected_backbone is not None and configuration["backbone"] != expected_backbone:
        raise ValueError("Checkpoint backbone disagrees with expected_backbone")
    if configuration.get("action_head") != expected_action_head:
        raise ValueError(f"Final checkpoint must have action_head={expected_action_head!r}")
    cameras = validate_cameras(configuration.get("cameras"))
    if validate_cameras(payload["protocol"].get("cameras")) != cameras:
        raise ValueError("Checkpoint configuration and protocol cameras disagree")
    if expected_cameras is not None and cameras != validate_cameras(expected_cameras):
        raise ValueError("Checkpoint cameras disagree with expected_cameras")
    if payload["schema"] == LEGACY_CHECKPOINT_SCHEMA:
        _validate_legacy_checkpoint(payload, expected_step)
    elif payload["schema"] == CHECKPOINT_SCHEMA:
        _validate_v2_checkpoint(payload, expected_step)
    else:
        raise ValueError("Unsupported checkpoint schema")
    for component in ("backbone.", "action_head.", "per_compr.", "cog_mem_bank.", "per_mem_bank."):
        if not any(parameter_name.startswith(component) for parameter_name in payload["model"]):
            raise ValueError(f"Final checkpoint lacks full model component: {component}")
    for parameter_name, tensor in payload["model"].items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Final checkpoint model entry is not a tensor: {parameter_name}")
        require_fp32 = payload["schema"] == CHECKPOINT_SCHEMA or expected_step == 55000
        expected_dtype = torch.bfloat16 if debug and parameter_name.startswith("backbone.") else torch.float32
        if (tensor.is_floating_point() or tensor.is_complex()) and (
            not torch.isfinite(tensor).all() or (require_fp32 and tensor.dtype != expected_dtype)
        ):
            requirement = "non-BF16 backbone" if debug and parameter_name.startswith("backbone.") else "non-FP32"
            raise ValueError(f"Final checkpoint has nonfinite/{requirement} model tensor: {parameter_name}")
    return {"step": payload["step"], "config": configuration, "protocol": payload["protocol"],
            "normalization": payload["normalization"]}


def build_suite_plan(config: EvaluationConfig, checkpoint: Path) -> dict:
    seed_pool_size = resolve_seed_attempt_budget(config.episodes, config.max_seed_attempts)
    last_seed = FIRST_SEED + seed_pool_size - 1
    return {
        "schema": "tttchoicemem-native-evaluation-v1", "checkpoint": str(checkpoint),
        "evaluation_config": config.to_dict(), "tasks": list(config.tasks),
        "test_num": config.episodes, "episodes_per_task": config.episodes,
        "total_requested_episodes": len(config.tasks) * config.episodes,
        "seed": 0, "first_seed": FIRST_SEED, "seed_stride": SEED_STRIDE,
        "seed_pool_size": seed_pool_size, "last_seed": last_seed,
        "candidate_seeds": list(range(FIRST_SEED, last_seed + 1)), "max_seed_attempts_per_task": seed_pool_size,
        "expert_check_before_policy": True,
        "expert_skipped_seeds_consume_pool": True, "expert_skipped_seeds_are_not_episodes": True,
        "candidate_pool_refill_or_rerun": False,
        "native_entrypoint": "eval_policy_xpolicylab.main", "expert_check": True,
        "policy_outcome_retries": 0, "incomplete_coverage_is_failure": True,
        "workers_per_gpu": config.workers_per_gpu,
        "policy_server_count": len(config.gpu_ids) * config.workers_per_gpu,
        "scheduling": "task_level" if config.workers_per_gpu == 1 else "bounded_episode_round_robin",
        "max_completed_plus_inflight_per_task": config.episodes,
        "native_case_test_num": config.episodes if config.workers_per_gpu == 1 else 1,
        "native_case_max_seed_attempts": seed_pool_size if config.workers_per_gpu == 1 else 1,
        "instruction_generation_test_num": config.episodes,
        "candidate_case_directory": "{task}/seed{seed}" if config.workers_per_gpu > 1 else "{task}",
    }


def build_server_command(config: EvaluationConfig, checkpoint: Path, port: int) -> list[str]:
    return [
        str(config.qwen_python), str(PROJECT_ROOT / "serve_rmbench.py"), "--checkpoint", str(checkpoint),
        "--rmbench-root", str(config.rmbench_root), "--device", "cuda:0", "--host", "127.0.0.1",
        "--port", str(port), "--protocol", "ws", "--action-type", "joint",
        "--horizon", "16", "--execute-prefix", "4",
    ]


def build_native_command(config: EvaluationConfig, task_name: str, gpu_id: int, port: int, output_dir: Path) -> list[str]:
    seed_pool_size = resolve_seed_attempt_budget(config.episodes, config.max_seed_attempts)
    return [
        str(config.env_python), str(Path(__file__).resolve()), "--native-child",
        "--rmbench-root", str(config.rmbench_root), "--task-name", task_name,
        "--sim-gpu", str(gpu_id), "--port", str(port), "--episodes", str(config.episodes), "--seed", "0",
        "--instruction-type", "unseen", "--task-config", "demo_clean",
        "--max-seed-attempts", str(seed_pool_size),
        "--expected-training-steps", str(config.expected_training_steps),
        "--output-dir", str(output_dir),
    ]


def build_child_environment(gpu_id: int, cpu_threads: int | None = None) -> dict[str, str]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    environment["RMBENCH_RENDER_DEVICE"] = "cuda:0"
    environment["PYTHONUNBUFFERED"] = "1"
    if cpu_threads is not None:
        for field_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[field_name] = str(cpu_threads)
    for field_name in ("NO_PROXY", "no_proxy"):
        hosts = environment.get(field_name, "").split(",") + ["localhost", "127.0.0.1", "::1"]
        environment[field_name] = ",".join(dict.fromkeys(host for host in hosts if host))
    return environment


def summarize_task(task_name: str, output_dir: Path, returncode: int | None, error: str | None = None,
                   *, episodes: int = EPISODES_PER_TASK, max_seed_attempts: int | None = None) -> dict:
    seed_pool_size = resolve_seed_attempt_budget(episodes, max_seed_attempts)
    journal = output_dir / "episodes.jsonl"
    events = []
    problems = []
    if journal.is_file():
        try:
            with journal.open(encoding="utf-8") as stream:
                events = [json.loads(line) for line in stream if line.strip()]
        except (ValueError, OSError) as exception:
            problems.append(f"Unreadable native journal: {exception}")
    else:
        problems.append("Native episode journal is missing")
    completed = [event for event in events if event.get("event") == "episode_done"]
    skipped = [event for event in events if event.get("event") == "seed_skipped"]
    attempts = [event.get("seed") for event in events if event.get("event") == "seed_attempt"]
    completions = [event for event in events if event.get("event") == "run_completed"]
    starts = [event for event in events if event.get("event") == "run_started"]
    completed_seeds = [event.get("seed") for event in completed]
    successes = sum(event.get("success") is True for event in completed)
    if (
        len(starts) != 1 or starts[0].get("requested_episodes") != episodes
        or starts[0].get("seed_start") != FIRST_SEED
    ):
        problems.append("Native run_started does not match the frozen budget/seed")
    # Seeds are consumed from one contiguous fixed pool; experts are checked
    # before the policy starts, so a skipped seed is neither an episode nor a
    # policy outcome, and the pool is never refilled or re-run.
    if attempts != list(range(FIRST_SEED, FIRST_SEED + len(attempts))) or len(attempts) > seed_pool_size:
        problems.append(f"Native seed attempts depart from the fixed {seed_pool_size}-candidate pool")
    if len(set(completed_seeds)) != len(completed_seeds) or any(seed not in attempts for seed in completed_seeds):
        problems.append("Completed seeds are duplicated or were not attempted")
    if any(type(event.get("success")) is not bool for event in completed):
        problems.append("Episode success values are not native booleans")
    if len(completed) != episodes:
        problems.append(f"Incomplete coverage: {len(completed)}/{episodes} episodes")
    if len(completions) != 1 or completions[0].get("completed_episodes") != len(completed):
        problems.append("Native run_completed is missing or disagrees with episode counts")
    elif completions[0].get("successes") != successes:
        problems.append("Native run_completed disagrees with recorded successes")
    if any(event.get("event") == "episode_error" for event in events):
        problems.append("Native policy rollout error; not a valid policy-outcome failure")
    if returncode != 0:
        problems.append(f"Native process did not exit successfully: {returncode}")
    if error:
        problems.append(error)
    infrastructure_failures = read_native_infrastructure_failures(output_dir)
    problems.extend(f"Native infrastructure failure: {failure}" for failure in infrastructure_failures)
    return {
        "task_name": task_name, "requested_episodes": episodes, "completed_episodes": len(completed),
        "requested_seed_attempts": seed_pool_size, "seed_attempt_count": len(attempts),
        "skipped_seed_count": len(skipped),
        "successes": successes, "observed_success_rate": successes / len(completed) if completed else None,
        "native_official_success_rate": successes / episodes if not problems else None,
        "seed_attempts": attempts, "completed_seeds": completed_seeds, "skipped_seeds": skipped,
        "run_completed": bool(completions), "coverage_complete": not problems,
        "infrastructure_failure": bool(infrastructure_failures), "infrastructure_errors": infrastructure_failures,
        "returncode": returncode, "problems": problems, "output_dir": str(output_dir),
    }


def run_task(config: EvaluationConfig, checkpoint: Path, task_name: str, gpu_id: int, port: int,
             eval_root: Path, update_status, cancellation: threading.Event) -> dict:
    output_dir = eval_root / task_name
    output_dir.mkdir(exist_ok=False)
    server = None
    client = None
    returncode = None
    error = None
    try:
        check_gpu_utilization((gpu_id,))
        check_port_free("127.0.0.1", port)
        environment = build_child_environment(gpu_id)
        with (output_dir / "server.log").open("x", encoding="utf-8") as server_log, \
                (output_dir / "native.log").open("x", encoding="utf-8") as native_log:
            server = subprocess.Popen(
                build_server_command(config, checkpoint, port), cwd=PROJECT_ROOT, env=environment,
                stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            try:
                update_status(task_name, {"state": "loading_server", "gpu_id": gpu_id, "port": port, "server_pid": server.pid})
                wait_for_server(server, "127.0.0.1", port, config.server_timeout, cancellation)
                client = subprocess.Popen(
                    build_native_command(config, task_name, gpu_id, port, output_dir),
                    cwd=config.rmbench_root, env=environment, stdout=native_log,
                    stderr=subprocess.STDOUT, start_new_session=True,
                )
                update_status(task_name, {"state": "evaluating", "native_pid": client.pid})
                while client.poll() is None:
                    if cancellation.wait(0.5):
                        raise RuntimeError("Evaluation cancelled; native task was not retried")
                    if server.poll() is not None:
                        raise RuntimeError("Owned policy server exited during native evaluation")
                    infrastructure_failures = read_native_infrastructure_failures(output_dir)
                    if infrastructure_failures:
                        raise RuntimeError(f"Native infrastructure failure: {infrastructure_failures[0]}")
                returncode = client.wait()
            finally:
                try:
                    stop_owned_process(client)
                finally:
                    stop_owned_process(server)
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    result = summarize_task(
        task_name, output_dir, returncode, error, episodes=config.episodes, max_seed_attempts=config.max_seed_attempts,
    )
    write_atomic_json(output_dir / "summary.json", result)
    update_status(task_name, {"state": "completed" if result["coverage_complete"] else "failed", "result": result})
    return result


def run_suite(config: EvaluationConfig, checkpoint: Path, eval_root: Path) -> dict:
    wall_start = time.time()
    config.validate()
    if eval_root.exists():
        raise FileExistsError(f"Refusing to reuse evaluation root: {eval_root}")
    checkpoint_expectations = {}
    if config.expected_training_steps != TARGET_OPTIMIZER_STEPS:
        checkpoint_expectations["expected_step"] = config.expected_training_steps
    if config.expected_action_head != "diffusion":
        checkpoint_expectations["expected_action_head"] = config.expected_action_head
    if config.expected_cameras is not None:
        checkpoint_expectations["expected_cameras"] = config.expected_cameras
    if config.expected_backbone is not None:
        checkpoint_expectations["expected_backbone"] = config.expected_backbone
    checkpoint_contract = validate_checkpoint(checkpoint, **checkpoint_expectations)
    for field_name in ("horizon", "execute_prefix"):
        if checkpoint_contract["config"][field_name] != getattr(config, field_name):
            raise ValueError(f"Checkpoint {field_name} disagrees with the native evaluation contract")
    if config.workers_per_gpu > 1:
        from eval_concurrency import run_concurrent_suite
        return run_concurrent_suite(config, checkpoint, eval_root, checkpoint_contract)
    check_gpu_utilization(config.gpu_ids)
    for port in config.ports:
        check_port_free("127.0.0.1", port)
    eval_root.mkdir(parents=True, exist_ok=False)
    suite_plan = build_suite_plan(config, checkpoint)
    suite_plan["checkpoint_contract"] = checkpoint_contract
    write_atomic_json(eval_root / "plan.json", suite_plan)
    status_lock = threading.Lock()
    status = {"state": "running", "pid": os.getpid(), "tasks": {task: {"state": "pending"} for task in config.tasks}}
    write_atomic_json(eval_root / "status.json", status)
    task_queue: queue.Queue[str] = queue.Queue()
    for task_name in config.tasks:
        task_queue.put(task_name)
    stop_event = threading.Event()
    results: dict[str, dict] = {}

    def update_status(task_name: str, changes: dict) -> None:
        with status_lock:
            status["tasks"][task_name].update(changes)
            write_atomic_json(eval_root / "status.json", status)

    def consume_tasks(gpu_id: int, port: int) -> None:
        while not stop_event.is_set():
            try:
                task_name = task_queue.get_nowait()
            except queue.Empty:
                return
            try:
                result = run_task(config, checkpoint, task_name, gpu_id, port, eval_root, update_status, stop_event)
                with status_lock:
                    results[task_name] = result
                # An incomplete task is never retried, but does not erase the
                # other selected tasks' planned native evaluations.
            except BaseException:
                stop_event.set()
                raise
            finally:
                task_queue.task_done()

    failure = None
    previous_handlers = {}

    def cancel_suite(signum, frame) -> None:
        stop_event.set()

    if threading.current_thread() is threading.main_thread():
        for stop_signal in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[stop_signal] = signal.signal(stop_signal, cancel_suite)
    try:
        with ThreadPoolExecutor(max_workers=len(config.gpu_ids), thread_name_prefix="native-gpu") as executor:
            futures = [executor.submit(consume_tasks, gpu_id, port) for gpu_id, port in zip(config.gpu_ids, config.ports)]
            for future in futures:
                future.result()
    except BaseException as exception:
        stop_event.set()
        failure = f"{type(exception).__name__}: {exception}"
    finally:
        for stop_signal, previous_handler in previous_handlers.items():
            signal.signal(stop_signal, previous_handler)
    if stop_event.is_set() and failure is None:
        failure = "Evaluation interrupted; no unfinished native rollout was retried"
    ordered_results = [results[task_name] for task_name in config.tasks if task_name in results]
    requested_episodes = len(config.tasks) * config.episodes
    complete = len(ordered_results) == len(config.tasks) and all(result["coverage_complete"] for result in ordered_results)
    summary = {
        "state": "completed" if complete and failure is None else "failed", "requested_tasks": list(config.tasks),
        "requested_episodes": requested_episodes,
        "completed_episodes": sum(result["completed_episodes"] for result in ordered_results),
        "coverage_complete": complete, "tasks": ordered_results,
        "unstarted_tasks": [task_name for task_name in config.tasks if task_name not in results],
        "native_official_equal_task_success_rate": (
            sum(result["native_official_success_rate"] for result in ordered_results) / len(config.tasks) if complete else None
        ), "error": failure,
        "started_at": wall_start, "finished_at": time.time(), "evaluation_wall_seconds": time.time() - wall_start,
    }
    write_atomic_json(eval_root / "summary.json", summary)
    status["state"] = summary["state"]
    write_atomic_json(eval_root / "status.json", status)
    return summary


def run_native_child(arguments: argparse.Namespace) -> int | None:
    seed_pool_size = resolve_seed_attempt_budget(arguments.episodes, arguments.max_seed_attempts)
    last_seed = FIRST_SEED + seed_pool_size - 1
    if arguments.case_seed is not None and (
        type(arguments.case_seed) is not int or not FIRST_SEED <= arguments.case_seed <= last_seed
    ):
        raise ValueError(f"case-seed must be an integer in the frozen {FIRST_SEED}-{last_seed} pool")
    if not arguments.task_name or not arguments.task_name.isidentifier() or arguments.output_dir is None:
        raise ValueError("A native child requires one valid task-name and an output-dir")
    if arguments.task_name not in read_native_tasks(arguments.rmbench_root):
        raise ValueError("Task is not in the official all_tasks.yml")
    # The orchestrator reserves the empty task root and its two log files. Never
    # permit an existing journal/results to turn a new child into a rerun.
    output_dir = arguments.output_dir.resolve()
    allowed_files = {"server.log", "native.log"}
    if arguments.case_seed is not None:
        allowed_files.add("case_plan.json")
    if output_dir.exists() and any(path.name not in allowed_files for path in output_dir.iterdir()):
        raise FileExistsError(f"Refusing to reuse native results: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update(build_child_environment(arguments.sim_gpu, 2 if arguments.case_seed is not None else None))
    random.seed(0)
    import numpy as np
    np.random.seed(0)
    from eval_rmbench import load_native_evaluator

    # load_native_evaluator already installs install_timestep_adapter exactly
    # once. The adapter forwards actual take_action_cnt, not query indexes.
    native_evaluator = load_native_evaluator(arguments.rmbench_root)
    from test_render import Sapien_TEST
    Sapien_TEST()
    native_arguments = {
        "bench_name": "RMBench", "task_name": arguments.task_name, "task_config": "demo_clean",
        "policy_name": "TTTchoiceMem", "ckpt_setting": f"policy-step{arguments.expected_training_steps}",
        "env_cfg_type": "aloha_agilex", "action_type": "joint", "host": "127.0.0.1",
        "port": arguments.port, "protocol": "ws", "xpolicylab_root": str(arguments.rmbench_root / "XPolicyLab"),
        "root_dir": str(arguments.rmbench_root), "seed": 0, "seed_offset": 0, "seed_stride": 1,
        # test_num counts *policy* episodes; the larger fixed pool only covers
        # native expert-validation skips that happen before the policy starts.
        "test_num": arguments.episodes, "eval_batch": False, "num_workers": 1,
        "max_seed_attempts": seed_pool_size,
        "expert_check": True, "instruction_type": "unseen", "eval_output_dir": output_dir,
        "episode_log": output_dir / "episodes.jsonl",
    }
    if arguments.case_seed is not None:
        from eval_concurrency import invoke_native_candidate
        native_arguments.update(test_num=1, max_seed_attempts=1, seed_offset=arguments.case_seed - FIRST_SEED)
        return invoke_native_candidate(
            native_evaluator, native_arguments, arguments.case_seed, output_dir,
            episodes=arguments.episodes, max_seed_attempts=seed_pool_size,
        )
    native_evaluator.main(native_arguments)
    return None


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--eval-root", type=Path)
    parser.add_argument("--rmbench-root", type=Path, default=DEFAULT_RMBENCH_ROOT)
    parser.add_argument("--policy-python", "--qwen-python", dest="qwen_python", type=Path,
                        help="Policy-server Python interpreter (legacy alias: --qwen-python)")
    parser.add_argument("--env-python", type=Path)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=list(DEFAULT_GPU_IDS))
    parser.add_argument("--workers-per-gpu", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--ports", nargs="+", type=int)
    parser.add_argument("--expected-cameras", nargs="+", choices=AVAILABLE_CAMERAS)
    parser.add_argument("--minimum-free-gpu-gib-per-worker", type=float, default=8.0)
    parser.add_argument("--episodes", type=int, choices=(20, 50), default=EPISODES_PER_TASK)
    parser.add_argument("--seed", type=int, choices=(0,), default=0)
    parser.add_argument("--instruction-type", choices=("unseen",), default="unseen")
    parser.add_argument("--task-config", choices=("demo_clean",), default="demo_clean")
    # Policy episodes use an exact 50x expert pool; omission derives that pool.
    parser.add_argument("--max-seed-attempts", type=int)
    parser.add_argument("--server-timeout", type=float, default=120)
    parser.add_argument("--expected-training-steps", type=positive_integer, default=TARGET_OPTIMIZER_STEPS,
                        help="Expected completed optimizer budget; must match the checkpoint training contract")
    parser.add_argument("--expected-action-head", choices=("diffusion", "oft"), default="diffusion")
    parser.add_argument("--expected-backbone", choices=("memoryvla", "qwen35"),
                        help="Optional assertion of the saved policy backbone")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--native-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--task-name", help=argparse.SUPPRESS)
    parser.add_argument("--sim-gpu", type=int, choices=ALLOWED_GPU_IDS, help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, choices=ALLOWED_PORTS, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--case-seed", type=int, help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    try:
        arguments.max_seed_attempts = resolve_seed_attempt_budget(arguments.episodes, arguments.max_seed_attempts)
    except ValueError as exception:
        parser.error(str(exception))
    if arguments.ports is None:
        arguments.ports = list(choose_default_ports(tuple(arguments.gpu_ids), arguments.workers_per_gpu))
    arguments.rmbench_root = arguments.rmbench_root.expanduser().resolve()
    if arguments.native_child:
        if arguments.sim_gpu is None or arguments.port is None:
            parser.error("native child requires sim-gpu and port")
        if not arguments.task_name or not arguments.task_name.isidentifier() or arguments.output_dir is None:
            parser.error("native child requires a valid task-name and output-dir")
        last_seed = FIRST_SEED + arguments.max_seed_attempts - 1
        if arguments.case_seed is not None and not FIRST_SEED <= arguments.case_seed <= last_seed:
            parser.error(f"case-seed must be in the frozen {FIRST_SEED}-{last_seed} pool")
    elif arguments.case_seed is not None:
        parser.error("case-seed is reserved for native episode children")
    elif any(getattr(arguments, name) is None for name in ("checkpoint", "eval_root", "qwen_python", "env_python")):
        parser.error("suite requires checkpoint, eval-root, policy-python (or qwen-python) and env-python")
    return arguments


def main(argv=None) -> int:
    arguments = parse_args(argv)
    if arguments.native_child:
        if arguments.dry_run:
            if arguments.case_seed is not None:
                print(
                    f"Native task={arguments.task_name} test_num=1 seed={arguments.case_seed} "
                    f"max_seed_attempts=1 instruction_generation_test_num={arguments.episodes} (not executed)"
                )
                return 0
            print(
                f"Native task={arguments.task_name} test_num={arguments.episodes} "
                f"seeds={FIRST_SEED}-{FIRST_SEED + arguments.max_seed_attempts - 1} "
                f"max_seed_attempts={arguments.max_seed_attempts} (not executed)"
            )
            return 0
        return run_native_child(arguments) or 0
    config = EvaluationConfig.from_dict({
        "tasks": arguments.tasks,
        "rmbench_root": arguments.rmbench_root, "qwen_python": arguments.qwen_python,
        "env_python": arguments.env_python, "gpu_ids": arguments.gpu_ids, "ports": arguments.ports,
        "episodes": arguments.episodes, "max_seed_attempts": arguments.max_seed_attempts,
        "server_timeout": arguments.server_timeout,
        "expected_training_steps": arguments.expected_training_steps,
        "expected_action_head": arguments.expected_action_head,
        "expected_backbone": arguments.expected_backbone,
        "workers_per_gpu": arguments.workers_per_gpu,
        "expected_cameras": arguments.expected_cameras,
        "minimum_free_gpu_gib_per_worker": arguments.minimum_free_gpu_gib_per_worker,
    })
    checkpoint = require_absolute_path(arguments.checkpoint, "checkpoint")
    eval_root = require_absolute_path(arguments.eval_root, "eval_root")
    if eval_root.exists():
        raise FileExistsError(f"Refusing to reuse evaluation root: {eval_root}")
    if arguments.dry_run:
        print(json.dumps(build_suite_plan(config, checkpoint), indent=2))
        if config.workers_per_gpu > 1:
            from eval_concurrency import print_concurrent_commands
            print_concurrent_commands(config, checkpoint, eval_root)
            return 0
        for task_index, task_name in enumerate(config.tasks):
            worker_index = task_index % len(config.gpu_ids)
            print(shlex.join(build_server_command(config, checkpoint, config.ports[worker_index])))
            print(shlex.join(build_native_command(
                config, task_name, config.gpu_ids[worker_index], config.ports[worker_index], eval_root / task_name,
            )))
        return 0
    summary = run_suite(config, checkpoint, eval_root)
    print(json.dumps(summary, indent=2))
    return 0 if summary["state"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
