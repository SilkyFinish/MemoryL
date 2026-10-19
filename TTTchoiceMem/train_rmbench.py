"""Explicitly bounded local RMBench training; the original train.py is untouched.

Use --prepare-only to audit native episodes and freeze the split/normalization
without importing torch, a backbone, a policy, or checkpoint model code.
Training has no implicit duration: --max-steps must be explicitly supplied.
Every step samples a random, sorted episode group and resets both banks.
The stage-1 default is original MemoryVLA + diffusion DiT-L, full finetuning.
--execute-prefix 4 is an RMBench deployment choice, not a history stride.
Labels keep the verified target_offset=1 shift: observation[t] supervises
joint_action/vector[t+1 : t+1+H], so the final recorded RGB frame is unsupervised.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from choice_mem.config import validate_cameras
from choice_mem.data import CAMERAS, build_training_dataset, prepare_manifest


class TrainingArgumentParser(argparse.ArgumentParser):
    """Keep the old optimization recipe opt-in, independent of argument order."""

    def parse_args(self, args=None, namespace=None):
        import sys

        tokens = list(sys.argv[1:] if args is None else args)
        arguments = super().parse_args(tokens, namespace)
        if arguments.optimization_recipe == "qwen-legacy":
            legacy_defaults = {
                "learning_rate": 1e-4, "weight_decay": 0.005,
                "gradient_clip_norm": 2.5, "warmup_ratio": 0.05,
            }
            for name, value in legacy_defaults.items():
                option = "--" + name.replace("_", "-")
                if not any(token == option or token.startswith(option + "=") for token in tokens):
                    setattr(arguments, name, value)
        return arguments


PROJECT_ROOT = Path(__file__).resolve().parent


def validate_output_path(path: Path, name: str) -> Path:
    """Never reuse the original project's outputs through paths or symlinks."""
    resolved = path.expanduser().resolve()
    for ancestor in (resolved, *resolved.parents):
        if ancestor.name == PROJECT_ROOT.name and ancestor != PROJECT_ROOT:
            raise ValueError(f"{name} points into another TTTchoiceMem checkout; choose a new local stage-1 path")
    original_tree = PROJECT_ROOT.parent.parent / "memory"
    if PROJECT_ROOT.parent.name == "memory_l" and resolved.is_relative_to(original_tree):
        raise ValueError(f"{name} may not write into the original memory tree; choose a new stage-1 output")
    return resolved


def validate_model_source(backbone: str, model_path: str | None, backbone_initialization: str = "pretrained") -> str:
    """Validate local assets without torch; debug follows metadata only, never weights."""
    if not model_path:
        raise ValueError("Training requires an explicit --model-path; MemoryVLA never falls back to Qwen")
    if backbone == "turbo_interaction":
        if backbone_initialization not in {"pretrained", "turbo-random-frozen-debug"}:
            raise ValueError("turbo_interaction requires pretrained or turbo-random-frozen-debug initialization")
        from choice_mem.turbo_assets import inspect_turbo_assets

        source = Path(model_path).expanduser()
        if not source.is_absolute():
            raise ValueError("turbo_interaction requires an absolute local --model-path manifest JSON")
        inspect_turbo_assets(str(source), initialization=backbone_initialization)
        return str(source)
    if backbone_initialization == "random-frozen-debug":
        if backbone != "memoryvla":
            raise ValueError("random-frozen-debug requires --backbone memoryvla")
        from choice_mem.random_init import inspect_random_backbone_assets

        source = Path(model_path).expanduser()
        if not source.is_absolute():
            raise ValueError("random-frozen-debug requires an absolute local --model-path metadata manifest JSON")
        inspect_random_backbone_assets(str(source))
        return str(source)
    if backbone_initialization != "pretrained":
        raise ValueError("Unknown backbone initialization")
    if backbone == "qwen35":
        return model_path
    source = Path(model_path).expanduser()
    if not source.is_absolute() or not source.exists():
        raise ValueError("--backbone memoryvla requires an existing absolute local --model-path")
    if source.is_dir():
        config_path = source / "config.json"
        if not config_path.is_file() or not (source / "checkpoints/latest-checkpoint.pt").is_file():
            raise ValueError("Original MemoryVLA assets need config.json and checkpoints/latest-checkpoint.pt")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if "qwen" in str(config.get("model_type", "")).lower() or "model" not in config:
            raise ValueError(
                "A VLM run needs config.json model configuration, not Qwen; "
                "for an original VLA run pass its checkpoints/*.pt or audited manifest"
            )
    elif source.is_file() and source.suffix == ".pt":
        config_path = source.parent.parent / "config.json"
        if not config_path.is_file():
            raise ValueError("Original VLA checkpoint requires run/config.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not config.get("vla", {}).get("base_vlm"):
            raise ValueError("Original VLA checkpoint config.json must contain vla.base_vlm")
    elif source.suffix == ".json":
        asset_manifest = json.loads(source.read_text(encoding="utf-8"))
        if not {"checkpoint_path", "checkpoint_sha256", "config.json", "llama_metadata", "local_vision_files"}.issubset(asset_manifest):
            raise ValueError("--model-path JSON must be an original MemoryVLA local asset manifest, not a Qwen config")
        required_assets = [
            asset_manifest["checkpoint_path"], asset_manifest["config.json"]["path"],
            asset_manifest["llama_metadata"]["path"], *asset_manifest["local_vision_files"].values(),
        ]
        if any(not Path(path).is_absolute() or not Path(path).exists() for path in required_assets):
            raise ValueError("MemoryVLA asset manifest must reference existing absolute local assets")
    else:
        raise ValueError("--model-path must be an original VLM directory, VLA .pt checkpoint, or asset manifest JSON")
    return str(source)


def build_optimizer_groups(policy, arguments: argparse.Namespace) -> list[dict]:
    """Original AdamW: one LR, excluding bias/norm parameters from weight decay."""
    import torch

    groups = {"decay": [], "no_decay": []}
    if arguments.optimization_recipe == "qwen-legacy":
        groups = {"backbone": [], "memory_and_head": []}
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.dtype != torch.float32:
            raise ValueError(f"Float32 master parameters are required before AdamW: {name}")
        if arguments.optimization_recipe == "qwen-legacy":
            group = "backbone" if name.startswith("backbone.") else "memory_and_head"
        else:
            group = "no_decay" if parameter.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower() else "decay"
        groups[group].append(parameter)
    result = []
    for name, parameters in groups.items():
        if parameters:
            result.append({
                "params": parameters, "name": name,
                "lr": arguments.backbone_learning_rate if name == "backbone" else arguments.learning_rate,
                "weight_decay": 0.0 if name == "no_decay" else arguments.weight_decay,
            })
    if not result:
        raise ValueError("Policy has no trainable parameters")
    return result


def microstep_seed(seed: int, completed_steps: int, accumulation_index: int = 0, rank: int = 0) -> int:
    """Stateless history/augmentation seed; checkpoint step is the sampler cursor."""
    if min(completed_steps, accumulation_index, rank) < 0:
        raise ValueError("Sampler step, accumulation index and rank must be nonnegative")
    return int(np.random.SeedSequence([seed % (2**64), completed_steps, accumulation_index, rank]).generate_state(1, dtype=np.uint64)[0])


def validate_optimization(arguments) -> None:
    from choice_mem.config import PolicyConfig

    # Use the same torch-free contract as model construction. In particular,
    # debug requires an explicit --freeze-backbone; never repair conflicts.
    PolicyConfig(
        backbone=arguments.backbone, action_head=arguments.action_head,
        diffusion_model_type=arguments.diffusion_model_type,
        freeze_backbone=not arguments.unfreeze_backbone, dtype=arguments.dtype,
        backbone_initialization=arguments.backbone_initialization,
        backbone_initialization_seed=arguments.backbone_initialization_seed,
        base_vlm_id=arguments.base_vlm_id,
    )
    if not arguments.unfreeze_backbone and getattr(arguments, "gradient_checkpointing", False):
        raise ValueError("--gradient-checkpointing conflicts with --freeze-backbone: the frozen VLM has no training graph to checkpoint")
    if arguments.optimization_recipe == "qwen-legacy" and arguments.backbone != "qwen35":
        raise ValueError("qwen-legacy optimization requires explicit --backbone qwen35")
    positive_fields = ["learning_rate", "gradient_clip_norm"]
    if arguments.optimization_recipe == "qwen-legacy":
        positive_fields += ["backbone_learning_rate", "minimum_backbone_learning_rate", "minimum_head_learning_rate"]
    for name in positive_fields:
        value = getattr(arguments, name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(arguments.weight_decay) or arguments.weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if not np.isfinite(arguments.warmup_ratio) or not 0 <= arguments.warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if arguments.optimization_recipe == "memoryvla" and arguments.warmup_ratio != 0:
        raise ValueError("Original MemoryVLA uses a constant schedule; warmup requires qwen-legacy")
    if arguments.optimization_recipe == "qwen-legacy":
        if arguments.minimum_backbone_learning_rate > arguments.backbone_learning_rate:
            raise ValueError("Minimum backbone learning rate exceeds its initial learning rate")
        if arguments.minimum_head_learning_rate > arguments.learning_rate:
            raise ValueError("Minimum head learning rate exceeds its initial learning rate")
    if arguments.stop_after is not None and arguments.stop_after > 2:
        raise ValueError("--stop-after is a capped preflight: choose 1 or 2 optimizer steps")
    if arguments.prepare_only and arguments.resume is not None:
        raise ValueError("--prepare-only cannot resume training")
    if arguments.group_size < 2 or not 1 <= arguments.execute_prefix <= arguments.horizon:
        raise ValueError("group_size >= 2 and 1 <= execute_prefix <= horizon are required")


def positive_integer(value: str) -> int:
    integer_value = int(value)
    if integer_value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return integer_value


def build_parser() -> argparse.ArgumentParser:
    parser = TrainingArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=["memoryvla", "qwen35", "turbo_interaction"], default="memoryvla")
    parser.add_argument("--action-head", choices=["diffusion", "oft"], default="diffusion")
    parser.add_argument("--model-path", default=None, help="Explicit original MemoryVLA assets, Turbo asset manifest, or Qwen source with --backbone qwen35")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--backbone-initialization", choices=["pretrained", "random-frozen-debug", "turbo-random-frozen-debug"], default="pretrained",
                        help="Distinct original/Turbo debug architectures use metadata only and require explicit --freeze-backbone")
    parser.add_argument("--backbone-initialization-seed", type=int, default=42)
    parser.add_argument("--base-vlm-id", default="prism-dinosiglip-224px+7b")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", help="Explicit native task names under a multi-task data root")
    parser.add_argument(
        "--cameras", nargs="+", choices=CAMERAS, default=["head_camera"],
        help="Ordered one-to-three views; head_camera is the main view. Freeze a separate manifest for each selection.",
    )
    parser.add_argument("--manifest", type=Path, required=True, help="One saved split reused by all four policies")
    parser.add_argument("--run-dir", type=Path, help="New, nonexisting directory; required for training")
    parser.add_argument("--max-steps", type=positive_integer, default=None, help="Required for training; e.g. 3")
    parser.add_argument("--horizon", type=positive_integer, default=16)
    parser.add_argument("--execute-prefix", type=positive_integer, default=4, help="RMBench deployment execution prefix; does not set random history spacing")
    parser.add_argument("--group-size", type=positive_integer, default=16)
    parser.add_argument("--history-sampling", choices=["random", "sequential"], default="random")
    parser.add_argument("--image-aug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repeated-diffusion-steps", type=positive_integer, default=4)
    parser.add_argument("--mask-padding-loss", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--inference-timestep-mode", choices=["query", "action"], default="query")
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument(
        "--cpu-threads",
        type=positive_integer,
        default=4,
        help="torch.set_num_threads value; kept small so CPU image decode/padding cannot oversubscribe a smoke run",
    )
    backbone_mode = parser.add_mutually_exclusive_group()
    backbone_mode.add_argument("--freeze-backbone", dest="unfreeze_backbone", action="store_false", help="Opt out of full finetuning")
    backbone_mode.add_argument("--unfreeze-backbone", dest="unfreeze_backbone", action="store_true")
    parser.set_defaults(unfreeze_backbone=True)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--per-token-size", type=positive_integer, default=256)
    parser.add_argument("--mem-length", type=positive_integer, default=16)
    parser.add_argument("--retrieval-layers", type=positive_integer, default=2)
    parser.add_argument("--oft-hidden-dim", type=positive_integer, default=256)
    parser.add_argument("--diffusion-model-type", default="DiT-L")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", nargs="?", const="latest", default=None, help="Explicit owned checkpoint from the same run; no implicit restart")
    parser.add_argument("--stop-after", type=positive_integer, default=None, help="Capped 1/2-update preflight, preserving explicit --max-steps budget")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--optimization-recipe", choices=["memoryvla", "qwen-legacy"], default="memoryvla", help="Original single-LR constant AdamW; old split-LR cosine Qwen recipe is opt-in")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.0, help="Used only by qwen-legacy cosine schedule")
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5, help="Used only by qwen-legacy")
    parser.add_argument("--minimum-backbone-learning-rate", type=float, default=1e-6, help="Used only by qwen-legacy")
    parser.add_argument("--minimum-head-learning-rate", type=float, default=5e-6, help="Used only by qwen-legacy")
    return parser


def summarize_manifest(manifest: dict) -> dict:
    protocol = manifest["protocol"]
    return {
        "event": "data_prepared",
        "manifest_sha256": manifest["manifest_sha256"],
        "data_root": manifest["data_root"],
        "episode_count": len(manifest["episodes"]),
        "train_episodes": sum(record["split"] == "train" for record in manifest["episodes"]),
        "validation_episodes": sum(record["split"] == "validation" for record in manifest["episodes"]),
        "train_action_frames": manifest["normalization"]["sample_count"],
        "cameras": protocol["cameras"],
        "action_dim": 14,
        "target_offset": protocol["target_offset"],
        "label_alignment": protocol["label_alignment"],
        "gripper_binarization": False,
        "semantic_risks": protocol["semantic_risks"],
    }


def write_run_json(path: Path, payload: dict) -> None:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".partial", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def run_training(arguments: argparse.Namespace, manifest: dict) -> dict:
    # These imports are deliberately below the prepare-only return in main().
    import torch
    from choice_mem.checkpoint import read_checkpoint, save_checkpoint
    from choice_mem.config import PolicyConfig
    from choice_mem.policy import ChoiceMemoryPolicy
    from choice_mem.precision import prepare_training_precision
    from train_multitask import (
        ROLLING_CHECKPOINT_PATTERN, TRAINING_STATE_SCHEMA, RandomEpisodeSampler,
        build_policy_configuration, build_protocol, build_schedule, capture_rng_state,
        capture_weight_probes, restore_rng_state, validate_loaded_group, validate_resume_payload,
        protocols_match,
    )

    device = torch.device(arguments.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only explicit CPU/CUDA devices are supported")
    torch.set_num_threads(arguments.cpu_threads)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; preparation needs no GPU, but this training request does")
        torch.cuda.set_device(device)
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    dataset = build_training_dataset(manifest, split="train", history_sampling=arguments.history_sampling, image_aug=arguments.image_aug)
    sampler = RandomEpisodeSampler(dataset, arguments.group_size, arguments.seed, arguments.tasks, history_sampling=arguments.history_sampling)
    # Reuse the distributed trainer's exact serialized data/method/optimizer
    # contract, with one rank and one episode group per optimizer update.
    contract_arguments = argparse.Namespace(**vars(arguments))
    contract_arguments.grad_accumulation = 1
    contract_arguments.per_device_batch_size = arguments.group_size
    contract_arguments.global_batch_size = arguments.group_size
    contract_arguments.gradient_checkpointing = False
    contract_arguments.find_unused_parameters = False
    configuration = build_policy_configuration(arguments, manifest)
    requested_configuration = configuration.to_dict()
    protocol = build_protocol(contract_arguments, manifest, argparse.Namespace(world_size=1), sampler, requested_configuration)
    protocol["training_engine"] = "single-process"
    run_directory = arguments.run_dir.expanduser().absolute()
    metrics_path = run_directory / "metrics.json"
    resume_payload = None
    resume_state = None
    previous_steps = []
    previous_wall_seconds = 0.0
    owned_names = []
    starting_step = 0
    if arguments.resume is not None:
        if arguments.resume == "latest":
            resume_path = Path(json.loads((run_directory / "latest_checkpoint.json").read_text(encoding="utf-8"))["path"])
        else:
            resume_path = Path(arguments.resume).expanduser().absolute()
        if resume_path.is_symlink() or resume_path.resolve().parent != run_directory.resolve() or not ROLLING_CHECKPOINT_PATTERN.fullmatch(resume_path.name):
            raise ValueError("Resume must use an owned single-process checkpoint from the same run directory")
        resume_payload = read_checkpoint(resume_path)
        resume_state = validate_resume_payload(resume_payload, requested_configuration, protocol, manifest["normalization"], 1)
        if not protocols_match(json.loads((run_directory / "protocol.json").read_text(encoding="utf-8")), protocol):
            raise ValueError("Run directory protocol differs from the resumed checkpoint")
        starting_step = resume_payload["step"]
        if starting_step >= arguments.max_steps:
            raise ValueError("Checkpoint has already reached the complete training budget")
        configuration = PolicyConfig(**resume_payload["config"])
        previous_wall_seconds = resume_state["training_wall_seconds"]
        owned_names = list(resume_state["owned_rolling_checkpoints"])
        if metrics_path.is_file():
            previous_steps = [step for step in json.loads(metrics_path.read_text(encoding="utf-8")).get("steps", []) if step["step"] <= starting_step]
    else:
        run_directory.mkdir(parents=True, exist_ok=False)
        write_run_json(run_directory / "protocol.json", protocol)
    metrics = {
        "status": "initializing", "started_at": datetime.now(timezone.utc).isoformat(),
        "backbone": arguments.backbone, "action_head": arguments.action_head,
        "device": str(device), "dtype": arguments.dtype, "target_offset": manifest["protocol"]["target_offset"],
        "optimization": protocol["optimization"],
        "backbone_initialization": arguments.backbone_initialization,
        "torch_cpu_threads": arguments.cpu_threads, "max_steps": arguments.max_steps,
        "completed_steps": starting_step, "manifest_sha256": manifest["manifest_sha256"], "steps": previous_steps,
        "peak_memory_allocated_bytes": 0 if device.type == "cuda" else None,
        "peak_memory_reserved_bytes": 0 if device.type == "cuda" else None,
        "memory_scope": "PyTorch allocator on selected GPU, not system-wide nvidia-smi usage; CPU is null",
    }
    wall_start = time.perf_counter()
    try:
        policy = prepare_training_precision(
            ChoiceMemoryPolicy(configuration, initialize_from_config=resume_payload is not None)
        ).to(device).train()
        if resume_payload is not None:
            policy.load_state_dict(resume_payload["model"], strict=True)
        metrics["parameter_counts"] = policy.parameter_counts()
        trainable_parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(build_optimizer_groups(policy, arguments), lr=arguments.learning_rate, weight_decay=arguments.weight_decay)
        scheduler = build_schedule(optimizer, arguments)
        probes = capture_weight_probes(policy)
        sampled_counts = {task: 0 for task in sampler.tasks}
        if resume_state is not None:
            optimizer.load_state_dict(resume_payload["optimizer"])
            scheduler.load_state_dict(resume_state["scheduler"])
            probes = resume_state["weight_probes"]
            sampled_counts = dict(resume_state["sampled_task_groups"])
            restore_rng_state(resume_state["rank_rng_states"][0], device)
            del resume_payload, resume_state
        write_run_json(metrics_path, metrics)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        print(json.dumps({"event": "train_start", "config": configuration.to_dict(), "parameter_counts": metrics["parameter_counts"]}), flush=True)
        stopping_step = min(arguments.max_steps, starting_step + arguments.stop_after) if arguments.stop_after else arguments.max_steps
        while scheduler.completed_steps < stopping_step:
            step = scheduler.completed_steps
            data_start = time.perf_counter()
            group_index = sampler.sample_index(step, 0, 1, 0, 1)
            group = dataset.get_group(group_index, seed=microstep_seed(arguments.seed, step))
            validate_loaded_group(group, arguments.group_size)
            data_latency = time.perf_counter() - data_start
            sampled_counts[canonical_task_name(dataset.groups[group_index][0])] += 1
            policy.reset()
            optimizer.zero_grad(set_to_none=True)
            actions = torch.as_tensor(group["actions"], device=device, dtype=torch.float32)
            valid_mask = torch.as_tensor(group["valid_mask"], device=device, dtype=torch.bool)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_start = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=arguments.dtype == "bfloat16"):
                loss = policy(group.get("image_batches", group.get("images")), group["instructions"], actions,
                              valid_mask, group["episode_ids"], group["timesteps"])
            if loss.ndim != 0 or not torch.isfinite(loss).item():
                raise FloatingPointError(f"Expected finite scalar training loss, got {loss.detach().cpu()}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=arguments.gradient_clip_norm, error_if_nonfinite=True)
            optimizer.step()
            scheduler.advance()
            policy.reset()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_latency = time.perf_counter() - step_start
            step_metrics = {
                "step": scheduler.completed_steps, "loss": float(loss.detach().cpu()),
                "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
                "step_latency_s": step_latency, "data_latency_s": data_latency,
                "latency_per_query_s": step_latency / len(group["timesteps"]),
                "episode_id": group["episode_ids"][0], "query_timesteps": group["timesteps"],
                "queries": len(group["timesteps"]), "valid_action_steps": int(group["valid_mask"].sum()),
                "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
            }
            metrics["steps"].append(step_metrics)
            metrics.update(status="training", completed_steps=scheduler.completed_steps,
                           peak_memory_allocated_bytes=step_metrics["peak_memory_allocated_bytes"],
                           peak_memory_reserved_bytes=step_metrics["peak_memory_reserved_bytes"])
            write_run_json(metrics_path, metrics)
            print(json.dumps({"event": "train_step", **step_metrics}), flush=True)
        completed_steps = scheduler.completed_steps
        checkpoint_path = run_directory / f"checkpoint-step{completed_steps:06d}.pt"
        new_owned_names = [*owned_names, checkpoint_path.name]
        # Validate retention before saving or deleting any owned path.
        if len(set(new_owned_names)) != len(new_owned_names) or any(
            not ROLLING_CHECKPOINT_PATTERN.fullmatch(name) or (run_directory / name).is_symlink() for name in new_owned_names
        ):
            raise ValueError("Invalid owned single-process checkpoint retention list")
        training_state = {
            "schema": TRAINING_STATE_SCHEMA, "sampler": protocol["sampling"],
            "scheduler": scheduler.state_dict(), "rank_rng_states": [capture_rng_state(device)],
            "next_global_micro_index": completed_steps, "next_global_group_index": completed_steps,
            "owned_rolling_checkpoints": new_owned_names[-2:], "weight_probes": probes,
            "sampled_task_groups": sampled_counts, "training_wall_seconds": previous_wall_seconds + time.perf_counter() - wall_start,
        }
        save_checkpoint(checkpoint_path, policy, manifest["normalization"], protocol, completed_steps,
                        optimizer=optimizer, training_state=training_state)
        write_run_json(run_directory / "latest_checkpoint.json", {"path": str(checkpoint_path), "step": completed_steps})
        for name in new_owned_names[:-2]:
            (run_directory / name).unlink(missing_ok=True)
        completed = completed_steps == arguments.max_steps
        if completed:
            final_snapshot = run_directory / "snapshot-final.pt"
            save_checkpoint(final_snapshot, policy, manifest["normalization"], protocol, completed_steps)
            metrics["final_snapshot"] = str(final_snapshot)
        metrics.update({
            "status": "completed" if completed else "stopped", "checkpoint": str(checkpoint_path),
            "final_loss": metrics["steps"][-1]["loss"],
            "mean_step_latency_s": float(np.mean([step["step_latency_s"] for step in metrics["steps"]])),
            "wall_time_s": previous_wall_seconds + time.perf_counter() - wall_start,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "stop_reason": "training_budget_reached" if completed else "bounded_preflight",
        })
        write_run_json(metrics_path, metrics)
        print(json.dumps({"event": "train_complete" if completed else "train_stopped", **{name: value for name, value in metrics.items() if name != "steps"}}), flush=True)
        return metrics
    except Exception as error:
        metrics.update(status="failed", error=f"{type(error).__name__}: {error}", wall_time_s=previous_wall_seconds + time.perf_counter() - wall_start)
        write_run_json(metrics_path, metrics)
        print(json.dumps({"event": "train_failed", "completed_steps": metrics["completed_steps"], "error": metrics["error"]}), flush=True)
        raise


def canonical_task_name(record):
    return record.get("task_name", record.get("task", "")).split("/", 1)[0]


def main(argv: list[str] | None = None) -> dict:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    validate_cameras(arguments.cameras)
    try:
        validate_optimization(arguments)
        validate_output_path(arguments.manifest, "--manifest")
        if arguments.run_dir is not None:
            validate_output_path(arguments.run_dir, "--run-dir")
        if arguments.prepare_only and arguments.backbone == "turbo_interaction":
            arguments.model_path = validate_model_source(arguments.backbone, arguments.model_path, arguments.backbone_initialization)
    except ValueError as error:
        parser.error(str(error))
    if not arguments.prepare_only:
        if arguments.max_steps is None:
            parser.error("training requires explicit --max-steps (for a short smoke, use --max-steps 3)")
        if arguments.run_dir is None:
            parser.error("training requires --run-dir pointing to a new directory")
        if arguments.run_dir.exists() and arguments.resume is None:
            parser.error("--run-dir already exists; refusing to overwrite an earlier run without explicit --resume")
        if arguments.resume is not None and not arguments.run_dir.is_dir():
            parser.error("--resume requires the same existing run directory")
        try:
            arguments.model_path = validate_model_source(arguments.backbone, arguments.model_path, arguments.backbone_initialization)
            validate_optimization(arguments)
        except ValueError as error:
            parser.error(str(error))
    manifest = prepare_manifest(
        arguments.data_root,
        arguments.manifest,
        cameras=list(arguments.cameras),
        horizon=arguments.horizon,
        execute_prefix=arguments.execute_prefix,
        group_size=arguments.group_size,
        seed=arguments.seed,
        validation_fraction=arguments.validation_fraction,
        tasks=arguments.tasks,
        history_sampling=arguments.history_sampling,
        image_aug=arguments.image_aug,
    )
    preparation_summary = summarize_manifest(manifest)
    print(json.dumps(preparation_summary), flush=True)
    if arguments.prepare_only:
        return preparation_summary
    return run_training(arguments, manifest)


if __name__ == "__main__":
    main()
