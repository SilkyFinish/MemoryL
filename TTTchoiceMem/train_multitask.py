"""Bounded, resumable MemoryVLA training on native RMBench tasks with torchrun.

Stage 1 defaults to full MemoryVLA + diffusion DiT-L, one sorted random
16-query episode group per forward, one LR 2e-5, constant AdamW, clip 1.
Episodes, not tasks or fixed-stride chunks, receive equal sampling weight.
Qwen/OFT, sequential history and the old split-LR cosine recipe are explicit
opt-ins. --max-steps is required for training; --stop-after is a preflight cap,
not a new scheduler budget. Preparation never imports torch; no evaluation
or real training is launched implicitly.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import re
import time
import warnings
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from choice_mem.config import debug_precision_metadata
from choice_mem.data import build_training_dataset, prepare_manifest
from train_rmbench import build_parser as build_rmbench_parser
from train_rmbench import (
    build_optimizer_groups, microstep_seed, positive_integer, summarize_manifest,
    validate_model_source, validate_optimization, validate_output_path, write_run_json,
)


TRAINING_STATE_SCHEMA = "choice-memory-multitask-training-v2"
ROLLING_CHECKPOINT_PATTERN = re.compile(r"checkpoint-step[0-9]{6,}\.pt\Z")
# Keep this metadata-only module torch-free; importing checkpoint's matching
# defaults here would import torch during preparation / frozen-plan dry runs.
INITIALIZATION_DEFAULTS = {
    "backbone_initialization": "pretrained",
    "backbone_initialization_seed": 42,
    "base_vlm_id": "prism-dinosiglip-224px+7b",
}


def configuration_with_initialization_defaults(configuration: dict) -> dict:
    """Historical missing provenance means pretrained, never random debug."""
    return {**INITIALIZATION_DEFAULTS, **configuration}


def protocols_match(recorded: dict, requested: dict) -> bool:
    """Only fill the three absent legacy defaults; keep all other resume gates."""
    def normalized(protocol):
        result = configuration_with_initialization_defaults(deepcopy(protocol))
        if isinstance(result.get("requested_configuration"), dict):
            result["requested_configuration"] = configuration_with_initialization_defaults(result["requested_configuration"])
        return result

    return normalized(recorded) == normalized(requested)


def build_parser() -> argparse.ArgumentParser:
    parser = build_rmbench_parser()
    parser.description = __doc__
    parser.set_defaults(cpu_threads=2, device="rankauto")
    parser._option_string_actions["--max-steps"].help = "Explicit full optimizer-step budget; required except with --prepare-only"
    parser._option_string_actions["--run-dir"].help = "New directory, or the same existing directory with --resume"
    parser._option_string_actions["--device"].help = "rankauto maps LOCAL_RANK to an already-visible CUDA device"
    parser._option_string_actions["--learning-rate"].help = "Single LR for all trainable parameters (head LR only in qwen-legacy)"
    parser.add_argument("--global-batch-size", type=positive_integer, default=None, help="Optional exact query budget; otherwise world_size * group_size * accumulation")
    parser.add_argument("--sampling-strategy", choices=["episode", "task-balanced"], default="episode", help="Episode-weighted by default; equal task weighting is an explicit ablation")
    parser.add_argument(
        "--per-device-batch-size", type=positive_integer, default=16,
        help="Queries in one forward/backward; exactly one original 16-query memory group",
    )
    parser.add_argument("--grad-accumulation", "--gradient-accumulation-steps", type=positive_integer, default=1)
    parser.add_argument(
        "--gradient-checkpointing", action="store_true", help="Opt-in nonreentrant Qwen layer checkpointing"
    )
    parser.add_argument(
        "--find-unused-parameters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Required for Turbo full-finetuning's final visual-update branch; conservative for Qwen; zero DiT gradients are not unused",
    )
    parser.add_argument("--save-every", type=positive_integer, default=1000)
    parser.add_argument("--snapshot-every", type=positive_integer, default=5000)
    return parser


def validate_arguments(arguments: argparse.Namespace) -> None:
    from choice_mem.config import validate_cameras

    validate_cameras(arguments.cameras)
    validate_optimization(arguments)
    if (arguments.backbone == "turbo_interaction" and arguments.unfreeze_backbone
            and not arguments.find_unused_parameters):
        # Only fused text feeds Cog; the final fusion's visual-update output
        # is intentionally discarded while raw original vision feeds Per.
        raise ValueError("Turbo full-finetuning requires --find-unused-parameters for the unused final visual-update branch")
    validate_output_path(arguments.manifest, "--manifest")
    if arguments.run_dir is not None:
        validate_output_path(arguments.run_dir, "--run-dir")
    if arguments.per_device_batch_size != arguments.group_size:
        raise ValueError("--per-device-batch-size must equal the single episode group size (default original 16-query group)")
    if arguments.stop_after is not None and arguments.stop_after > 2:
        raise ValueError("--stop-after is a capped preflight: choose 1 or 2 optimizer steps")
    if not arguments.prepare_only:
        if arguments.max_steps is None or arguments.max_steps < 1:
            raise ValueError("Training requires explicit --max-steps; there is no implicit training budget")
        if arguments.run_dir is None:
            raise ValueError("Training requires --run-dir; new runs cannot reuse an existing directory")
        arguments.model_path = validate_model_source(arguments.backbone, arguments.model_path, arguments.backbone_initialization)
    elif arguments.backbone == "turbo_interaction":
        # Preparation audits Turbo's local metadata/manifest without importing torch.
        arguments.model_path = validate_model_source(arguments.backbone, arguments.model_path, arguments.backbone_initialization)
    if arguments.tasks is not None and (not arguments.tasks or len(set(arguments.tasks)) != len(arguments.tasks)):
        raise ValueError("--tasks must specify nonempty distinct canonical native tasks")
    if arguments.prepare_only and arguments.resume is not None:
        raise ValueError("--prepare-only cannot resume training")


def validate_global_batch(world_size: int, group_size: int, accumulation_steps: int, requested_batch: int | None) -> int:
    if requested_batch is None:
        requested_batch = world_size * group_size * accumulation_steps
    if min(world_size, group_size, accumulation_steps, requested_batch) < 1:
        raise ValueError("World size, group size, accumulation and global batch must all be positive")
    actual_batch = world_size * group_size * accumulation_steps
    if actual_batch != requested_batch:
        raise ValueError(
            f"Exact global batch mismatch: {world_size} ranks * {group_size} queries * "
            f"{accumulation_steps} accumulation = {actual_batch}, requested {requested_batch}"
        )
    return actual_batch


def canonical_task(record: dict) -> str:
    task_name = record.get("task_name", record.get("task"))
    if not isinstance(task_name, str) or not task_name:
        raise ValueError("Every manifest episode must identify its task")
    # Older native manifests identify a task as '<task>/demo_clean'.
    return task_name.split("/", 1)[0]


class RandomEpisodeSampler:
    """Uniform episodes with replacement, never implicitly uniform tasks/chunks.

    Random datasets expose one candidate list per episode; candidate count is
    not batch size. get_group performs sorted history sampling with its own
    stateless microstep seed, including last-query repeats for short episodes.
    """

    def __init__(self, dataset, group_size: int, seed: int, tasks=None, *, history_sampling="random", task_balanced=False):
        self.seed = seed
        self.group_size = group_size
        self.history_sampling = history_sampling
        self.task_balanced = task_balanced
        observed = sorted({canonical_task(record) for record, _ in dataset.groups})
        self.tasks = sorted(observed if tasks is None else tasks)
        if not self.tasks or len(set(self.tasks)) != len(self.tasks) or set(observed).difference(self.tasks):
            raise ValueError("Sampler requires distinct selected tasks and no unselected tasks")
        self.indices_by_episode = {}
        self.episodes_by_task = {task: [] for task in self.tasks}
        self.excluded_tail_groups = 0
        for index, (record, candidates) in enumerate(dataset.groups):
            if not candidates or any(later <= earlier for earlier, later in zip(candidates, candidates[1:])):
                raise ValueError("Episode candidates must be nonempty and strictly chronological")
            if history_sampling == "sequential" and len(candidates) != group_size:
                self.excluded_tail_groups += 1
                continue
            episode_id = record["episode_id"]
            task = canonical_task(record)
            if episode_id not in self.indices_by_episode:
                self.indices_by_episode[episode_id] = []
                self.episodes_by_task[task].append(episode_id)
            elif episode_id not in self.episodes_by_task[task]:
                raise ValueError("Episode ids must uniquely identify a task")
            self.indices_by_episode[episode_id].append(index)
        self.episodes = sorted(self.indices_by_episode)
        if not self.episodes or any(not self.episodes_by_task[task] for task in self.tasks):
            raise ValueError("Every selected task requires at least one eligible training episode")

    def sample_index(self, completed_steps, accumulation_index, accumulation_steps, rank, world_size):
        if completed_steps < 0 or not 0 <= accumulation_index < accumulation_steps or not 0 <= rank < world_size:
            raise ValueError("Invalid sampler step, accumulation index, or rank")
        generator = np.random.default_rng(microstep_seed(self.seed, completed_steps, accumulation_index, rank))
        episodes = self.episodes
        if self.task_balanced:
            task = self.tasks[int(generator.integers(len(self.tasks)))]
            episodes = self.episodes_by_task[task]
        episode = episodes[int(generator.integers(len(episodes)))]
        indices = self.indices_by_episode[episode]
        return indices[int(generator.integers(len(indices)))]

    def summary(self, manifest):
        per_task = {task: {"train_episodes": 0, "validation_episodes": 0, "eligible_train_episodes": len(self.episodes_by_task[task])} for task in self.tasks}
        for record in manifest["episodes"]:
            task = canonical_task(record)
            if task not in per_task:
                raise ValueError("Normalization/split manifest contains unselected tasks")
            per_task[task][f"{record['split']}_episodes"] += 1
        return {
            "tasks": self.tasks, "per_task": per_task, "eligible_train_episodes": len(self.episodes),
            "excluded_tail_groups": self.excluded_tail_groups,
            "history_sampling": self.history_sampling,
            "sampling": "uniform task then episode (explicit ablation)" if self.task_balanced else "uniform episode with replacement; no task balancing",
            "rng": "SeedSequence([seed mod 2**64, completed_steps, accumulation_index, rank]); get_group uses the same microstep seed",
        }


class FullGroupTaskSampler:
    """Uniform task, then uniform full group, with replacement and no mutable RNG.

    global_micro_index is the rank-zero position in the interleaved rank stream:
    (completed_steps * accumulation_steps + accumulation_index) * world_size.
    Each rank uses exactly default_rng(train_seed + global_micro_index + rank).
    Task balance is equal sampling probability, not guaranteed per-step quotas.
    Incomplete *query groups* are excluded; valid_mask still handles action tails.
    """

    def __init__(self, dataset, group_size: int, seed: int, tasks: list[str] | None = None):
        self.group_size = group_size
        self.seed = seed
        observed_tasks = sorted({canonical_task(record) for record, _timesteps in dataset.groups})
        self.tasks = sorted(observed_tasks if tasks is None else tasks)
        if not self.tasks or len(set(self.tasks)) != len(self.tasks):
            raise ValueError("Sampler tasks must be nonempty and distinct")
        if set(observed_tasks).difference(self.tasks):
            raise ValueError("Dataset contains unselected tasks; prepare a genuinely filtered manifest")
        self.indices_by_task = {task_name: [] for task_name in self.tasks}
        self.counts = {
            task_name: {"full_groups": 0, "full_queries": 0, "excluded_tail_groups": 0, "excluded_tail_queries": 0}
            for task_name in self.tasks
        }
        for group_index, (record, timesteps) in enumerate(dataset.groups):
            task_name = canonical_task(record)
            if len(timesteps) > group_size or any(later <= earlier for earlier, later in zip(timesteps, timesteps[1:])):
                raise ValueError("Manifest groups must be bounded and strictly chronological")
            if len(timesteps) == group_size:
                self.indices_by_task[task_name].append(group_index)
                self.counts[task_name]["full_groups"] += 1
                self.counts[task_name]["full_queries"] += group_size
            else:
                self.counts[task_name]["excluded_tail_groups"] += 1
                self.counts[task_name]["excluded_tail_queries"] += len(timesteps)
        empty_tasks = [task_name for task_name, indices in self.indices_by_task.items() if not indices]
        if empty_tasks:
            raise ValueError(f"Every selected task requires at least one full training group: {empty_tasks}")

    def sample_index(
        self, completed_steps: int, accumulation_index: int, accumulation_steps: int, rank: int, world_size: int,
    ) -> int:
        if completed_steps < 0 or not 0 <= accumulation_index < accumulation_steps or not 0 <= rank < world_size:
            raise ValueError("Invalid sampler step, accumulation index, or rank")
        global_micro_index = (completed_steps * accumulation_steps + accumulation_index) * world_size
        generator = np.random.default_rng(self.seed + global_micro_index + rank)
        task_name = self.tasks[int(generator.integers(len(self.tasks)))]
        task_indices = self.indices_by_task[task_name]
        return task_indices[int(generator.integers(len(task_indices)))]

    def summary(self, manifest: dict) -> dict:
        episode_counts = {task_name: {"train_episodes": 0, "validation_episodes": 0} for task_name in self.tasks}
        for record in manifest["episodes"]:
            task_name = canonical_task(record)
            if task_name not in episode_counts:
                raise ValueError("Normalization/split manifest contains unselected tasks")
            episode_counts[task_name][f"{record['split']}_episodes"] += 1
        return {
            "tasks": self.tasks,
            "per_task": {task_name: {**episode_counts[task_name], **self.counts[task_name]} for task_name in self.tasks},
            "full_groups": sum(count["full_groups"] for count in self.counts.values()),
            "excluded_tail_groups": sum(count["excluded_tail_groups"] for count in self.counts.values()),
            "excluded_tail_queries": sum(count["excluded_tail_queries"] for count in self.counts.values()),
            "sampling": "uniform task, then uniform full group, independently with replacement",
            "rng": "default_rng(train_seed + (step * accumulation + micro_index) * world_size + rank)",
        }


class ConstantSchedule:
    """Original optimizer LR, retaining an explicit resumable update budget."""

    def __init__(self, optimizer, total_steps):
        if total_steps < 1:
            raise ValueError("Schedule requires a positive explicit budget")
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.base_rates = [float(group["lr"]) for group in optimizer.param_groups]
        self.completed_steps = 0

    def advance(self):
        if self.completed_steps >= self.total_steps:
            raise ValueError("Cannot advance a completed schedule")
        self.completed_steps += 1

    def state_dict(self):
        return {"schedule": "constant", "total_steps": self.total_steps, "base_rates": self.base_rates, "completed_steps": self.completed_steps}

    def load_state_dict(self, state):
        expected = self.state_dict()
        if any(state.get(name) != expected[name] for name in ("schedule", "total_steps", "base_rates")):
            raise ValueError("Resume scheduler mismatch")
        if not isinstance(state.get("completed_steps"), int) or not 0 <= state["completed_steps"] <= self.total_steps:
            raise ValueError("Invalid resumed scheduler step")
        self.completed_steps = state["completed_steps"]
        for group, rate in zip(self.optimizer.param_groups, self.base_rates):
            group["lr"] = rate


def build_schedule(optimizer, arguments):
    if arguments.optimization_recipe == "memoryvla":
        return ConstantSchedule(optimizer, arguments.max_steps)
    minimum_rates = [
        arguments.minimum_backbone_learning_rate if group["name"] == "backbone" else arguments.minimum_head_learning_rate
        for group in optimizer.param_groups
    ]
    return CosineWarmupSchedule(optimizer, arguments.max_steps, arguments.warmup_ratio, minimum_rates)


class CosineWarmupSchedule:
    """Learning rate of the next optimizer update, indexed by completed updates."""

    def __init__(self, optimizer, total_steps: int, warmup_ratio: float, minimum_rates: list[float]):
        if total_steps < 1 or not 0 <= warmup_ratio < 1 or len(minimum_rates) != len(optimizer.param_groups):
            raise ValueError("Invalid cosine schedule specification")
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = min(total_steps - 1, math.ceil(total_steps * warmup_ratio))
        self.base_rates = [float(group["lr"]) for group in optimizer.param_groups]
        self.minimum_rates = list(minimum_rates)
        if any(not 0 < minimum <= base for minimum, base in zip(self.minimum_rates, self.base_rates)):
            raise ValueError("Schedule minimum rates must be positive and no greater than base rates")
        self.completed_steps = 0
        self._apply_rates()

    def _apply_rates(self):
        update_number = min(self.completed_steps + 1, self.total_steps)
        for group, base_rate, minimum_rate in zip(self.optimizer.param_groups, self.base_rates, self.minimum_rates):
            if self.warmup_steps and update_number <= self.warmup_steps:
                learning_rate = base_rate * update_number / self.warmup_steps
            else:
                decay_fraction = (update_number - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                learning_rate = minimum_rate + (base_rate - minimum_rate) * 0.5 * (
                    1 + math.cos(math.pi * decay_fraction)
                )
            group["lr"] = learning_rate

    def advance(self):
        if self.completed_steps >= self.total_steps:
            raise ValueError("Cannot advance a completed schedule")
        self.completed_steps += 1
        self._apply_rates()

    def state_dict(self) -> dict:
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "base_rates": self.base_rates,
            "minimum_rates": self.minimum_rates,
            "completed_steps": self.completed_steps,
        }

    def load_state_dict(self, state: dict):
        expected = self.state_dict()
        for field_name in ("total_steps", "warmup_steps", "base_rates", "minimum_rates"):
            if state.get(field_name) != expected[field_name]:
                raise ValueError(f"Resume scheduler mismatch: {field_name}")
        completed_steps = state.get("completed_steps")
        if not isinstance(completed_steps, int) or not 0 <= completed_steps <= self.total_steps:
            raise ValueError("Invalid resumed scheduler step")
        self.completed_steps = completed_steps
        self._apply_rates()


def _nested_lists(value):
    return [_nested_lists(element) for element in value] if isinstance(value, (tuple, list)) else value


def _nested_tuples(value):
    return tuple(_nested_tuples(element) for element in value) if isinstance(value, list) else value


def capture_rng_state(device=None) -> dict:
    import torch

    uses_cuda = device is not None and torch.device(device).type == "cuda"
    numpy_state = np.random.get_state()
    return {
        "python": _nested_lists(random.getstate()),
        "numpy": [
            numpy_state[0],
            numpy_state[1].tolist(),
            int(numpy_state[2]),
            int(numpy_state[3]),
            float(numpy_state[4]),
        ],
        "torch_cpu": torch.get_rng_state().tolist(),
        "torch_cuda": torch.cuda.get_rng_state(device).tolist() if uses_cuda else None,
    }


def restore_rng_state(state: dict, device=None) -> None:
    import torch

    random.setstate(_nested_tuples(state["python"]))
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), numpy_state[2], numpy_state[3], numpy_state[4])
    )
    torch.set_rng_state(torch.tensor(state["torch_cpu"], dtype=torch.uint8))
    if device is not None and torch.device(device).type == "cuda":
        if state["torch_cuda"] is None:
            raise ValueError("CUDA resume requires the selected rank's CUDA RNG state")
        torch.cuda.set_rng_state(torch.tensor(state["torch_cuda"], dtype=torch.uint8), device)


def enable_gradient_checkpointing(policy) -> None:
    if getattr(getattr(policy, "config", None), "freeze_backbone", False):
        raise ValueError("Gradient checkpointing conflicts with a frozen backbone; omit --gradient-checkpointing")
    enable_backbone = getattr(policy.backbone, "enable_gradient_checkpointing", None)
    if callable(enable_backbone):
        enable_backbone()
        return
    if getattr(getattr(policy, "config", None), "backbone", None) == "turbo_interaction":
        raise ValueError("Turbo gradient checkpointing requires an adapter enable_gradient_checkpointing implementation")
    model = policy.backbone.model
    enable_function = getattr(model, "gradient_checkpointing_enable", None)
    if enable_function is None:
        raise ValueError("The installed Qwen model does not support layer gradient checkpointing")
    # Enable internal transformer blocks only. Never checkpoint the whole visual
    # encoder/backbone: its temporary output capture hook must run exactly once.
    enable_function(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False


TURBO_GRADIENT_MODULES = {
    "bert": "backbone_bert",
    "text_projection": "backbone_text_projection",
    "vision_language_interaction": "backbone_vision_language_interaction",
    "cognition_projection": "backbone_cognition_projection",
    "vision_projection": "backbone_vision_projection",
}


def gradient_module(parameter_name: str) -> str:
    if parameter_name.startswith("backbone."):
        component = parameter_name.split(".", 2)[1]
        if component in TURBO_GRADIENT_MODULES:
            return TURBO_GRADIENT_MODULES[component]
        if ".visual." in parameter_name or ".vision_backbone." in parameter_name:
            return "backbone_vision"
        if ".language_model." in parameter_name or ".llm_backbone." in parameter_name:
            return "backbone_language"
        return "backbone_other"
    return parameter_name.split(".", 1)[0]


def collect_gradient_diagnostics(policy) -> dict:
    import torch

    module_statistics = {}
    for parameter_name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            continue
        module_name = gradient_module(parameter_name)
        statistics = module_statistics.setdefault(
            module_name, {"gradient_squared_norms": [], "parameters": 0, "missing_gradient_names": []}
        )
        statistics["parameters"] += 1
        if parameter.grad is None:
            statistics["missing_gradient_names"].append(parameter_name)
        else:
            parameter_norm = parameter.grad.detach().float().norm()
            statistics["gradient_squared_norms"].append(parameter_norm.square())
    for statistics in module_statistics.values():
        squared_norms = statistics.pop("gradient_squared_norms")
        squared_norm = float(torch.stack(squared_norms).sum().item()) if squared_norms else 0.0
        statistics["gradient_norm"] = math.sqrt(squared_norm)
    return module_statistics


def capture_weight_probes(policy) -> dict:
    candidates = {name: [] for name in ("backbone_vision", "backbone_language", *TURBO_GRADIENT_MODULES.values())}
    for parameter_name, parameter in policy.named_parameters():
        module_name = gradient_module(parameter_name)
        if module_name in candidates and parameter.requires_grad and parameter.ndim >= 2:
            candidates[module_name].append((parameter_name, parameter))
    probes = {}
    for module_name, parameters in candidates.items():
        if not parameters:
            # --freeze-backbone intentionally has no trainable backbone probes.
            if any(parameter.requires_grad for name, parameter in policy.named_parameters() if gradient_module(name) == module_name):
                raise ValueError(f"Cannot audit actual backbone weights: missing {module_name} matrix")
            continue
        # Prefer an encoder-layer projection over a potentially enormous embedding.
        parameter_name, parameter = min(
            parameters, key=lambda entry: (not (".blocks." in entry[0] or ".layers." in entry[0]), entry[1].numel())
        )
        probes[module_name] = {"parameter_name": parameter_name, "initial_weight": parameter.detach().cpu().clone()}
    return probes


def compare_weight_probes(policy, probes: dict) -> dict:
    named_parameters = dict(policy.named_parameters())
    changes = {}
    for module_name, probe in probes.items():
        parameter_name = probe["parameter_name"]
        current_weight = named_parameters[parameter_name].detach().cpu()
        initial_weight = probe["initial_weight"]
        difference = current_weight - initial_weight
        changes[module_name] = {
            "parameter_name": parameter_name,
            "changed_elements": int((current_weight != initial_weight).sum().item()),
            "total_elements": current_weight.numel(),
            "maximum_absolute_change": float(difference.abs().max().item()),
            "change_norm": float(difference.norm().item()),
            "interpretation": "Actual float32 weight changes; decay alone is not evidence of loss-driven learning",
        }
    return changes


def warn_gradient_diagnostics(diagnostics: dict, weight_changes: dict) -> None:
    for module_name, statistics in diagnostics.items():
        if statistics["missing_gradient_names"]:
            warnings.warn(
                f"After the zero-init grace period, {module_name} has "
                f"{len(statistics['missing_gradient_names'])}/{statistics['parameters']} missing gradients; "
                "see gradient_diagnostics in metrics.jsonl",
                RuntimeWarning,
                stacklevel=2,
            )
        if statistics["gradient_norm"] == 0:
            warnings.warn(f"After step 3, {module_name} gradient norm is zero", RuntimeWarning, stacklevel=2)
    for module_name, changes in weight_changes.items():
        if changes["changed_elements"] == 0:
            warnings.warn(
                f"Actual {module_name} weights have not changed after optimizer steps", RuntimeWarning, stacklevel=2
            )


def validate_loaded_group(group: dict, group_size: int, execute_prefix: int | None = None) -> None:
    images = group.get("image_batches", group.get("images"))
    if images is None:
        raise ValueError("Native group must provide image_batches (or images)")
    for field_name in ("instructions", "actions", "valid_mask", "episode_ids", "timesteps"):
        if len(group[field_name]) != group_size:
            raise ValueError(f"Incomplete training group: {field_name}")
    if len(images) != group_size or len(set(group["episode_ids"])) != 1:
        raise ValueError("Every forward must contain one full chronological single-episode group")
    timesteps = group["timesteps"]
    if any(int(value) != value or value < 0 for value in timesteps) or any(later < earlier for earlier, later in zip(timesteps, timesteps[1:])):
        raise ValueError("Query timesteps must be sorted nonnegative integers")
    # Short episodes pad their final query; duplicates anywhere else are not a
    # random history and would incorrectly revisit earlier bank states.
    if any(later == earlier and later != timesteps[-1] for earlier, later in zip(timesteps, timesteps[1:])):
        raise ValueError("Only repeated final queries may pad a short episode group")


def perform_optimizer_step(
    distributed_policy,
    policy,
    optimizer,
    dataset,
    sampler,
    arguments,
    completed_steps: int,
    rank: int,
    world_size: int,
    device,
    diagnostic_step: bool = False,
) -> dict:
    import torch

    optimizer.zero_grad(set_to_none=True)
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    data_latency = 0.0
    sampled_tasks = {task_name: 0 for task_name in sampler.tasks}
    for accumulation_index in range(arguments.grad_accumulation):
        group_index = sampler.sample_index(
            completed_steps, accumulation_index, arguments.grad_accumulation, rank, world_size,
        )
        data_start = time.perf_counter()
        group = dataset.get_group(group_index, seed=microstep_seed(arguments.seed, completed_steps, accumulation_index, rank))
        validate_loaded_group(group, arguments.group_size)
        data_latency += time.perf_counter() - data_start
        sampled_tasks[canonical_task(dataset.groups[group_index][0])] += 1
        images = group.get("image_batches", group.get("images"))
        actions = torch.as_tensor(group["actions"], device=device, dtype=torch.float32)
        valid_mask = torch.as_tensor(group["valid_mask"], device=device, dtype=torch.bool)
        synchronizing_microstep = accumulation_index == arguments.grad_accumulation - 1
        synchronization_context = nullcontext() if synchronizing_microstep else distributed_policy.no_sync()
        policy.reset()
        try:
            # no_sync must enclose forward as well as backward, not just backward.
            with synchronization_context:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=arguments.dtype == "bfloat16"):
                    loss = distributed_policy(
                        images,
                        group["instructions"],
                        actions,
                        valid_mask,
                        group["episode_ids"],
                        group["timesteps"],
                    )
                if loss.ndim != 0 or not torch.isfinite(loss.detach()).item():
                    raise FloatingPointError("Every chronological group must produce a finite scalar loss")
                total_loss += loss.detach().float() / arguments.grad_accumulation
                (loss / arguments.grad_accumulation).backward()
        finally:
            # Memory banks must not carry a graph/state across accumulated groups.
            policy.reset()
    diagnostics = collect_gradient_diagnostics(policy) if diagnostic_step else None
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        arguments.gradient_clip_norm,
        error_if_nonfinite=True,
    )
    optimizer.step()
    return {
        "loss_tensor": total_loss,
        "gradient_norm_before_clip": float(gradient_norm.item()),
        "gradient_diagnostics": diagnostics,
        "data_latency_s": data_latency,
        "sampled_task_groups": sampled_tasks,
    }


@dataclass(frozen=True)
class DistributedRuntime:
    rank: int
    local_rank: int
    world_size: int
    device: object


def initialize_distributed(arguments) -> DistributedRuntime:
    import torch
    import torch.distributed as distributed

    required_environment = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    if any(name not in os.environ for name in required_environment):
        raise ValueError("Launch training with torchrun; RANK, LOCAL_RANK and WORLD_SIZE are required")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    arguments.global_batch_size = validate_global_batch(
        world_size, arguments.per_device_batch_size, arguments.grad_accumulation, arguments.global_batch_size,
    )
    if not 0 <= rank < world_size or not 0 <= local_rank < torch.cuda.device_count():
        raise ValueError("Rank must address a CUDA device already visible to this process")
    if arguments.device not in {"rankauto", "auto", f"cuda:{local_rank}"}:
        raise ValueError("--device must be rankauto/auto or the current LOCAL_RANK's visible CUDA index")
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires visible CUDA devices supporting bfloat16")
    torch.set_num_threads(arguments.cpu_threads)
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if arguments.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The current LOCAL_RANK's visible device does not support bfloat16")
    distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=timedelta(minutes=10))
    return DistributedRuntime(rank, local_rank, world_size, device)


def rank_zero_operation(runtime: DistributedRuntime, operation):
    import torch.distributed as distributed

    message = [None]
    if runtime.rank == 0:
        try:
            message[0] = {"result": operation(), "error": None}
        except Exception as error:
            message[0] = {"result": None, "error": f"{type(error).__name__}: {error}"}
    distributed.broadcast_object_list(message, src=0, device=runtime.device)
    if message[0]["error"] is not None:
        raise RuntimeError(f"Rank-zero operation failed: {message[0]['error']}")
    return message[0]["result"]


def prepare_training_manifest(arguments) -> dict:
    keyword_arguments = dict(
        cameras=list(arguments.cameras),
        horizon=arguments.horizon,
        execute_prefix=arguments.execute_prefix,
        group_size=arguments.group_size,
        seed=arguments.seed,
        validation_fraction=arguments.validation_fraction,
        history_sampling=arguments.history_sampling,
        image_aug=arguments.image_aug,
    )
    if arguments.tasks is not None:
        if "tasks" not in inspect.signature(prepare_manifest).parameters:
            raise RuntimeError("The shared prepare_manifest must support tasks= before filtered training")
        keyword_arguments["tasks"] = arguments.tasks
    return prepare_manifest(arguments.data_root, arguments.manifest, **keyword_arguments)


def build_policy_configuration(arguments, manifest):
    from choice_mem.config import PolicyConfig

    return PolicyConfig(
        backbone=arguments.backbone,
        action_head=arguments.action_head,
        model_source=arguments.model_path,
        model_revision=arguments.model_revision,
        backbone_initialization=arguments.backbone_initialization,
        backbone_initialization_seed=arguments.backbone_initialization_seed,
        base_vlm_id=arguments.base_vlm_id,
        horizon=arguments.horizon,
        action_dim=14,
        execute_prefix=arguments.execute_prefix,
        cameras=list(arguments.cameras),
        image_size=manifest["protocol"]["image_size"],
        per_token_size=arguments.per_token_size,
        mem_length=arguments.mem_length,
        retrieval_layers=arguments.retrieval_layers,
        oft_hidden_dim=arguments.oft_hidden_dim,
        diffusion_model_type=arguments.diffusion_model_type,
        group_size=arguments.group_size,
        history_sampling=arguments.history_sampling,
        image_aug=arguments.image_aug,
        repeated_diffusion_steps=arguments.repeated_diffusion_steps,
        mask_padding_loss=arguments.mask_padding_loss,
        inference_timestep_mode=arguments.inference_timestep_mode,
        freeze_backbone=not arguments.unfreeze_backbone,
        dtype=arguments.dtype,
    )


def build_protocol(arguments, manifest, runtime, sampler, requested_configuration) -> dict:
    requested_configuration = configuration_with_initialization_defaults(requested_configuration)
    return {
        **manifest["protocol"],
        # Match save_checkpoint's Turbo provenance before strict resume compares
        # protocols; leave historical MemoryVLA/Qwen protocol shapes unchanged.
        **({"backbone": "turbo_interaction"} if requested_configuration.get("backbone") == "turbo_interaction" else {}),
        "trainer": TRAINING_STATE_SCHEMA,
        "manifest_path": str(arguments.manifest.expanduser().absolute()),
        "manifest_sha256": manifest["manifest_sha256"],
        "data_root": manifest["data_root"],
        "split_seed": manifest["split_seed"],
        "validation_fraction": manifest["validation_fraction"],
        "train_episode_ids": manifest["normalization"]["training_episode_ids"],
        "validation_episode_ids": [
            record["episode_id"] for record in manifest["episodes"] if record["split"] == "validation"
        ],
        "codec": manifest["codec"],
        "requested_configuration": requested_configuration,
        **{field: requested_configuration.get(field, default) for field, default in INITIALIZATION_DEFAULTS.items()},
        "sampling": sampler.summary(manifest),
        "optimization": {
            "optimizer": "AdamW",
            "recipe": arguments.optimization_recipe,
            "learning_rate": arguments.learning_rate,
            "backbone_learning_rate": arguments.backbone_learning_rate if arguments.optimization_recipe == "qwen-legacy" else arguments.learning_rate,
            "memory_head_learning_rate": arguments.learning_rate,
            "weight_decay": arguments.weight_decay,
            "weight_decay_exclusions": ["bias", "norm", "one-dimensional parameters"] if arguments.optimization_recipe == "memoryvla" else [],
            "schedule": "constant" if arguments.optimization_recipe == "memoryvla" else "cosine with linear warmup",
            "warmup_ratio": arguments.warmup_ratio,
            "minimum_backbone_learning_rate": arguments.minimum_backbone_learning_rate if arguments.optimization_recipe == "qwen-legacy" else None,
            "minimum_head_learning_rate": arguments.minimum_head_learning_rate if arguments.optimization_recipe == "qwen-legacy" else None,
            "gradient_clip_norm": arguments.gradient_clip_norm,
            "max_steps": arguments.max_steps,
            "world_size": runtime.world_size,
            "group_size": arguments.group_size,
            "per_device_batch_size": arguments.per_device_batch_size,
            "groups_per_forward": arguments.per_device_batch_size // arguments.group_size,
            "grad_accumulation": arguments.grad_accumulation,
            "global_batch_size": arguments.global_batch_size,
            "training_seed": arguments.seed,
            "master_dtype": "float32",  # Trainable optimizer masters, not every model parameter.
            **debug_precision_metadata(arguments.backbone_initialization, arguments.dtype),
            "autocast_dtype": arguments.dtype,
            "gradient_checkpointing": arguments.gradient_checkpointing,
            "find_unused_parameters": arguments.find_unused_parameters,
            "torch_cpu_threads": arguments.cpu_threads,
            "hyperparameter_provenance": "Original MemoryVLA single-LR AdamW/constant/clip1 defaults; RMBench joint targets and deployment prefix are interface adaptations" if arguments.optimization_recipe == "memoryvla" else "Explicit legacy Qwen split-LR cosine recipe; not the stage-1 baseline",
        },
        "history_sampling": arguments.history_sampling,
        "image_aug": arguments.image_aug,
        "repeated_diffusion_steps": arguments.repeated_diffusion_steps,
        "mask_padding_loss": arguments.mask_padding_loss,
        "inference_timestep_mode": arguments.inference_timestep_mode,
        "execute_prefix_provenance": "RMBench deployment choice; not the random training history stride",
        "evaluation": {"executed_by_this_trainer": False},
    }


def validate_resume_payload(payload, requested_configuration, protocol, normalization, world_size: int) -> dict:
    from choice_mem.checkpoint import CHECKPOINT_SCHEMA

    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("Resume requires the current shared checkpoint schema (legacy checkpoints must be read/migrated first)")
    if not protocols_match(payload.get("protocol", {}), protocol) or payload.get("normalization") != normalization:
        raise ValueError("Resume data/configuration/optimization protocol differs from this run")
    recorded_configuration = configuration_with_initialization_defaults(payload.get("config", {}))
    enriched_fields = {"preprocessing", "feature_dimensions", "asset_manifest"}
    for field_name, field_value in requested_configuration.items():
        if field_name in enriched_fields or (field_name == "model_revision" and field_value is None):
            continue
        if recorded_configuration.get(field_name) != field_value:
            raise ValueError(f"Resume policy configuration mismatch: {field_name}")
    if "optimizer" not in payload or "training_state" not in payload:
        raise ValueError("Inference snapshots or old smoke checkpoints cannot resume this training")
    state = payload["training_state"]
    if state.get("schema") != TRAINING_STATE_SCHEMA:
        raise ValueError("Missing/unsupported multitask training state")
    if len(state.get("rank_rng_states", [])) != world_size:
        raise ValueError("Resume requires the same world size and every rank's RNG state")
    step = payload["step"]
    if not isinstance(step, int) or step < 0 or state["scheduler"]["completed_steps"] != step:
        raise ValueError("Checkpoint optimizer/scheduler step mismatch")
    expected_micro_index = step * protocol["optimization"]["grad_accumulation"] * world_size
    if state.get("next_global_micro_index") != expected_micro_index:
        raise ValueError("Checkpoint sampler position mismatch")
    groups_per_forward = protocol["optimization"].get("groups_per_forward", 1)
    if groups_per_forward > 1 and state.get("next_global_group_index") != expected_micro_index * groups_per_forward:
        raise ValueError("Checkpoint independent memory-group sampler position mismatch")
    if "weight_probes" not in state:
        raise ValueError("Resumable checkpoint must preserve initial-weight probes (empty when backbone is frozen)")
    required_probes = {"backbone_vision", "backbone_language"}
    if requested_configuration.get("backbone") == "turbo_interaction":
        required_probes = {"backbone_vision", *TURBO_GRADIENT_MODULES.values()}
    if not requested_configuration.get("freeze_backbone", False) and not required_probes.issubset(state["weight_probes"]):
        raise ValueError("Resumable full-finetune checkpoint must preserve actual backbone initial-weight probes")
    if state.get("sampler") != protocol.get("sampling"):
        raise ValueError("Checkpoint sampler definition mismatch")
    return state


def resolve_resume_path(run_directory: Path, resume_argument: str) -> Path:
    if resume_argument == "latest":
        with (run_directory / "latest_checkpoint.json").open(encoding="utf-8") as stream:
            checkpoint_path = Path(json.load(stream)["path"])
    else:
        checkpoint_path = Path(resume_argument).expanduser().absolute()
    if checkpoint_path.is_symlink() or checkpoint_path.resolve().parent != (run_directory / "checkpoints").resolve():
        raise ValueError("Resume must use a resumable checkpoint from the same run directory")
    if not ROLLING_CHECKPOINT_PATTERN.fullmatch(checkpoint_path.name) or not checkpoint_path.is_file():
        raise ValueError("Resume checkpoint is missing or is not an owned rolling checkpoint")
    return checkpoint_path


def prune_owned_checkpoints(run_directory: Path, owned_names: list[str], keep: int = 2) -> list[str]:
    if keep < 1 or len(set(owned_names)) != len(owned_names):
        raise ValueError("Invalid owned checkpoint retention list")
    checkpoint_directory = run_directory / "checkpoints"
    for checkpoint_name in owned_names:
        checkpoint_path = checkpoint_directory / checkpoint_name
        if not ROLLING_CHECKPOINT_PATTERN.fullmatch(checkpoint_name) or checkpoint_path.is_symlink():
            raise ValueError("Refusing to prune a path not explicitly owned by this trainer")
    for checkpoint_name in owned_names[:-keep]:
        (checkpoint_directory / checkpoint_name).unlink(missing_ok=True)
    return owned_names[-keep:]


def ensure_training_state_support():
    from choice_mem.checkpoint import save_checkpoint

    if "training_state" not in inspect.signature(save_checkpoint).parameters:
        raise RuntimeError("Shared save_checkpoint must accept training_state= for full resumability")


def save_training_checkpoint(
    run_directory,
    policy,
    optimizer,
    scheduler,
    manifest,
    protocol,
    runtime,
    owned_names,
    probes,
    sampled_counts,
    previous_wall_seconds,
    last_saved_step=None,
) -> tuple[list[str], Path]:
    import torch.distributed as distributed
    from choice_mem.checkpoint import save_checkpoint

    completed_steps = scheduler.completed_steps
    checkpoint_path = run_directory / "checkpoints" / f"checkpoint-step{completed_steps:06d}.pt"
    if last_saved_step == completed_steps:
        distributed.barrier()
        return owned_names, checkpoint_path
    distributed.barrier()
    rank_rng_states = [None] * runtime.world_size
    distributed.all_gather_object(rank_rng_states, capture_rng_state(runtime.device))
    new_owned_names = [*owned_names, checkpoint_path.name]
    retained_names = new_owned_names[-2:]
    training_state = {
        "schema": TRAINING_STATE_SCHEMA,
        "sampler": protocol.get("sampling"),
        "scheduler": scheduler.state_dict(),
        "rank_rng_states": rank_rng_states,
        "next_global_micro_index": completed_steps * protocol["optimization"]["grad_accumulation"] * runtime.world_size,
        "next_global_group_index": (
            completed_steps * protocol["optimization"]["grad_accumulation"] * runtime.world_size
            * protocol["optimization"].get("groups_per_forward", 1)
        ),
        "owned_rolling_checkpoints": retained_names,
        "weight_probes": probes,
        "sampled_task_groups": sampled_counts,
        "training_wall_seconds": previous_wall_seconds,
    }

    def write_checkpoint():
        save_checkpoint(
            checkpoint_path,
            policy,
            manifest["normalization"],
            protocol,
            completed_steps,
            optimizer=optimizer,
            training_state=training_state,
        )
        write_run_json(run_directory / "latest_checkpoint.json", {"path": str(checkpoint_path), "step": completed_steps})
        prune_owned_checkpoints(run_directory, new_owned_names)

    rank_zero_operation(runtime, write_checkpoint)
    distributed.barrier()
    return retained_names, checkpoint_path


def save_inference_snapshot(run_directory, policy, manifest, protocol, runtime, step, final=False):
    import torch.distributed as distributed
    from choice_mem.checkpoint import save_checkpoint

    snapshot_path = run_directory / "snapshots" / ("final.pt" if final else f"snapshot-step{step:06d}.pt")
    distributed.barrier()
    rank_zero_operation(
        runtime,
        lambda: save_checkpoint(snapshot_path, policy, manifest["normalization"], protocol, step),
    )
    distributed.barrier()
    return snapshot_path


def append_metrics(path: Path, metrics: dict):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(metrics, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def run_training(arguments, manifest, runtime, run_directory: Path) -> dict:
    import torch
    import torch.distributed as distributed
    from torch.nn.parallel import DistributedDataParallel
    from choice_mem.checkpoint import read_checkpoint
    from choice_mem.config import PolicyConfig
    from choice_mem.policy import ChoiceMemoryPolicy
    from choice_mem.precision import prepare_training_precision

    ensure_training_state_support()
    actual_batch = validate_global_batch(
        runtime.world_size, arguments.per_device_batch_size, arguments.grad_accumulation, arguments.global_batch_size
    )
    arguments.global_batch_size = actual_batch
    dataset = build_training_dataset(manifest, split="train", history_sampling=arguments.history_sampling, image_aug=arguments.image_aug)
    sampler = RandomEpisodeSampler(
        dataset, arguments.group_size, arguments.seed, arguments.tasks,
        history_sampling=arguments.history_sampling, task_balanced=arguments.sampling_strategy == "task-balanced",
    )
    configuration = build_policy_configuration(arguments, manifest)
    requested_configuration = configuration.to_dict()
    protocol = build_protocol(arguments, manifest, runtime, sampler, requested_configuration)
    resume_payload = None
    resume_state = None
    owned_names = []
    previous_wall_seconds = 0.0
    sampled_counts = {task_name: 0 for task_name in sampler.tasks}
    last_saved_step = None
    resume_validated = arguments.resume is None
    wall_start = time.perf_counter()
    progress = {
        "status": "running",
        "phase": "initializing",
        "completed_steps": 0,
        "max_steps": arguments.max_steps,
        "actual_global_batch_size": actual_batch,
        "world_size": runtime.world_size,
        "manifest_sha256": manifest["manifest_sha256"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "visible_gpu_mask": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "evaluation_episodes_requested": 0,
        "evaluation_status": "not_run",
    }
    try:
        if arguments.resume is not None:
            resume_path = resolve_resume_path(run_directory, arguments.resume)
            resume_payload = read_checkpoint(resume_path)
            resume_state = validate_resume_payload(
                resume_payload,
                requested_configuration,
                protocol,
                manifest["normalization"],
                runtime.world_size,
            )
            with (run_directory / "protocol.json").open(encoding="utf-8") as stream:
                if not protocols_match(json.load(stream), protocol):
                    raise ValueError("Run directory protocol differs from the resumed checkpoint")
            if resume_payload["step"] >= arguments.max_steps:
                raise ValueError("Checkpoint has already reached the complete training budget")
            resume_validated = True
            configuration = PolicyConfig(**resume_payload["config"])
            progress["completed_steps"] = resume_payload["step"]
            progress["resumed_from"] = str(resume_path)
        elif runtime.rank == 0:
            write_run_json(run_directory / "protocol.json", protocol)
        if runtime.rank == 0:
            write_run_json(run_directory / "progress.json", progress)
            write_run_json(run_directory / "sampling.json", sampler.summary(manifest))
        # Identical initialization on every rank; stochastic training RNG is rank-specific below.
        random.seed(arguments.seed)
        np.random.seed(arguments.seed)
        torch.manual_seed(arguments.seed)
        torch.cuda.manual_seed(arguments.seed)
        policy = ChoiceMemoryPolicy(configuration, initialize_from_config=resume_payload is not None)
        # FP32 trainable masters; random debug keeps its frozen VLM in BF16.
        policy = prepare_training_precision(policy).to(runtime.device).train()
        if resume_payload is not None:
            policy.load_state_dict(resume_payload["model"], strict=True)
        if arguments.gradient_checkpointing:
            enable_gradient_checkpointing(policy)
        optimizer = torch.optim.AdamW(build_optimizer_groups(policy, arguments), weight_decay=arguments.weight_decay)
        scheduler = build_schedule(optimizer, arguments)
        distributed_policy = DistributedDataParallel(
            policy,
            device_ids=[runtime.local_rank],
            output_device=runtime.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=arguments.find_unused_parameters,
        )
        probes = capture_weight_probes(policy) if runtime.rank == 0 else {}
        if resume_state is not None:
            optimizer.load_state_dict(resume_payload["optimizer"])
            scheduler.load_state_dict(resume_state["scheduler"])
            owned_names = resume_state["owned_rolling_checkpoints"]
            sampled_counts = resume_state["sampled_task_groups"]
            previous_wall_seconds = resume_state["training_wall_seconds"]
            probes = resume_state["weight_probes"] if runtime.rank == 0 else {}
            last_saved_step = resume_payload["step"]
            del resume_payload
        if runtime.rank == 0:
            progress["parameter_counts"] = policy.parameter_counts()
            print(
                json.dumps(
                    {
                        "event": "train_start",
                        "config": configuration.to_dict(),
                        "optimization": protocol["optimization"],
                        "sampling": sampler.summary(manifest),
                        "parameter_counts": progress["parameter_counts"],
                    }
                ),
                flush=True,
            )
        # Restore after all initialization/DDP synchronization. Saving and logging consume no policy RNG.
        if resume_state is not None:
            restore_rng_state(resume_state["rank_rng_states"][runtime.rank], runtime.device)
            del resume_state
        else:
            random.seed(arguments.seed + runtime.rank)
            np.random.seed(arguments.seed + runtime.rank)
            torch.manual_seed(arguments.seed + runtime.rank)
            torch.cuda.manual_seed(arguments.seed + runtime.rank)
        torch.cuda.reset_peak_memory_stats(runtime.device)
        starting_step = scheduler.completed_steps
        bounded_updates = arguments.stop_after
        stopping_step = (
            min(arguments.max_steps, starting_step + bounded_updates)
            if bounded_updates
            else arguments.max_steps
        )
        if starting_step >= arguments.max_steps:
            raise ValueError("Checkpoint has already reached the complete training budget")
        while scheduler.completed_steps < stopping_step:
            update_number = scheduler.completed_steps + 1
            diagnostic_step = update_number == 3 or update_number % 10 == 0
            torch.cuda.synchronize(runtime.device)
            step_start = time.perf_counter()
            learning_rates = {group["name"]: float(group["lr"]) for group in optimizer.param_groups}
            result = perform_optimizer_step(
                distributed_policy,
                policy,
                optimizer,
                dataset,
                sampler,
                arguments,
                scheduler.completed_steps,
                runtime.rank,
                runtime.world_size,
                runtime.device,
                diagnostic_step,
            )
            scheduler.advance()
            distributed.all_reduce(result["loss_tensor"], op=distributed.ReduceOp.SUM)
            task_counts_tensor = torch.tensor(
                [result["sampled_task_groups"][task_name] for task_name in sampler.tasks],
                device=runtime.device,
                dtype=torch.int64,
            )
            distributed.all_reduce(task_counts_tensor, op=distributed.ReduceOp.SUM)
            global_task_counts = task_counts_tensor.cpu().tolist()
            for task_name, task_count in zip(sampler.tasks, global_task_counts):
                sampled_counts[task_name] += task_count
            torch.cuda.synchronize(runtime.device)
            local_latency = time.perf_counter() - step_start
            latency_memory = torch.tensor(
                [
                    local_latency,
                    torch.cuda.max_memory_allocated(runtime.device),
                    torch.cuda.max_memory_reserved(runtime.device),
                ],
                dtype=torch.float64,
                device=runtime.device,
            )
            distributed.all_reduce(latency_memory, op=distributed.ReduceOp.MAX)
            maximum_latency, maximum_allocated, maximum_reserved = latency_memory.cpu().tolist()
            elapsed_seconds = time.perf_counter() - wall_start
            average_step_seconds = elapsed_seconds / (scheduler.completed_steps - starting_step)
            progress.update(
                {
                    "status": "running",
                    "phase": "training",
                    "completed_steps": scheduler.completed_steps,
                    "loss": float(result["loss_tensor"].item() / runtime.world_size),
                    "learning_rates": learning_rates,
                    "next_learning_rates": {group["name"]: float(group["lr"]) for group in optimizer.param_groups},
                    "gradient_norm_before_clip": result["gradient_norm_before_clip"],
                    "step_latency_s": maximum_latency,
                    "data_latency_s_rank0": result["data_latency_s"],
                    "queries_per_second": actual_batch / maximum_latency,
                    "eta_seconds": (arguments.max_steps - scheduler.completed_steps) * average_step_seconds,
                    "wall_time_s": previous_wall_seconds + elapsed_seconds,
                    "peak_memory_allocated_bytes_max_rank": int(maximum_allocated),
                    "peak_memory_reserved_bytes_max_rank": int(maximum_reserved),
                    "memory_scope": "PyTorch allocator, maximum across ranks, not system-wide GPU memory",
                    "sampled_task_groups": dict(sampled_counts),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            if runtime.rank == 0 and diagnostic_step:
                progress["gradient_diagnostics"] = result["gradient_diagnostics"]
                progress["gradient_diagnostics_step"] = update_number
                progress["weight_changes"] = compare_weight_probes(policy, probes)
                if update_number == 3:
                    warn_gradient_diagnostics(result["gradient_diagnostics"], progress["weight_changes"])
            if runtime.rank == 0:
                write_run_json(run_directory / "progress.json", progress)
                if update_number == 3 or update_number % 10 == 0 or bounded_updates:
                    append_metrics(run_directory / "metrics.jsonl", {"event": "optimizer_step", **progress})
                    print(json.dumps({"event": "optimizer_step", **progress}), flush=True)
            if update_number % arguments.save_every == 0:
                owned_names, checkpoint_path = save_training_checkpoint(
                    run_directory,
                    policy,
                    optimizer,
                    scheduler,
                    manifest,
                    protocol,
                    runtime,
                    owned_names,
                    probes,
                    sampled_counts,
                    progress["wall_time_s"],
                    last_saved_step,
                )
                last_saved_step = update_number
                progress["checkpoint"] = str(checkpoint_path)
            if update_number % arguments.snapshot_every == 0:
                progress["snapshot"] = str(
                    save_inference_snapshot(
                        run_directory,
                        policy,
                        manifest,
                        protocol,
                        runtime,
                        update_number,
                    )
                )
        owned_names, checkpoint_path = save_training_checkpoint(
            run_directory,
            policy,
            optimizer,
            scheduler,
            manifest,
            protocol,
            runtime,
            owned_names,
            probes,
            sampled_counts,
            progress["wall_time_s"],
            last_saved_step,
        )
        completed = scheduler.completed_steps == arguments.max_steps
        progress.update(
            {
                "status": "completed" if completed else "stopped",
                "phase": "finished",
                "checkpoint": str(checkpoint_path),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "stop_reason": "training_budget_reached" if completed else "bounded_preflight",
            }
        )
        if completed:
            progress["final_snapshot"] = str(
                save_inference_snapshot(
                    run_directory,
                    policy,
                    manifest,
                    protocol,
                    runtime,
                    scheduler.completed_steps,
                    final=True,
                )
            )
        if runtime.rank == 0:
            progress["weight_changes"] = compare_weight_probes(policy, probes)
            write_run_json(run_directory / "progress.json", progress)
            append_metrics(run_directory / "metrics.jsonl", {"event": "train_finished", **progress})
            print(json.dumps({"event": "train_finished", **progress}), flush=True)
        distributed.barrier()
        return progress
    except Exception as error:
        progress.update(
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        if runtime.rank == 0 and resume_validated:
            write_run_json(run_directory / "progress.json", progress)
            append_metrics(run_directory / "metrics.jsonl", {"event": "train_failed", **progress})
        raise


def main(argv: list[str] | None = None) -> dict:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        validate_arguments(arguments)
    except ValueError as error:
        parser.error(str(error))
    if arguments.prepare_only:
        manifest = prepare_training_manifest(arguments)
        summary = summarize_manifest(manifest)
        print(json.dumps(summary), flush=True)
        return summary
    runtime = initialize_distributed(arguments)
    run_directory = arguments.run_dir.expanduser().absolute()
    directory_owned = False
    try:

        def prepare_run_directory():
            if arguments.resume is None:
                run_directory.mkdir(parents=True, exist_ok=False)
                (run_directory / "checkpoints").mkdir()
                (run_directory / "snapshots").mkdir()
            elif not run_directory.is_dir():
                raise FileNotFoundError("--resume requires an existing run directory")

        rank_zero_operation(runtime, prepare_run_directory)
        directory_owned = arguments.resume is None
        import torch.distributed as distributed

        distributed.barrier()
        manifest = rank_zero_operation(runtime, lambda: prepare_training_manifest(arguments))
        if runtime.rank == 0:
            print(json.dumps(summarize_manifest(manifest)), flush=True)
        return run_training(arguments, manifest, runtime, run_directory)
    except Exception as error:
        if directory_owned and runtime.rank == 0:
            progress_path = run_directory / "progress.json"
            progress = {}
            if progress_path.is_file():
                with progress_path.open(encoding="utf-8") as stream:
                    progress = json.load(stream)
            progress.update(
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            write_run_json(progress_path, progress)
        raise
    finally:
        import torch.distributed as distributed

        distributed.destroy_process_group()


if __name__ == "__main__":
    main()
