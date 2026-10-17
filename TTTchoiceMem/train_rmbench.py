"""Explicitly bounded local RMBench training; the original train.py is untouched.

Use --prepare-only to audit native episodes and freeze the split/normalization
without importing torch, a backbone, a policy, or checkpoint model code.
Training has no implicit duration: --max-steps must be explicitly supplied.
Every step forwards one chronological episode group and resets both banks.
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
from choice_mem.data import CAMERAS, RMBenchSequentialGroups, prepare_manifest

DEFAULT_QWEN_MODEL = "Qwen/Qwen3.5-0.8B"


def positive_integer(value: str) -> int:
    integer_value = int(value)
    if integer_value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return integer_value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=["memoryvla", "qwen35"], default="qwen35")
    parser.add_argument("--action-head", choices=["diffusion", "oft"], default="oft")
    parser.add_argument("--model-path", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", help="Explicit native task names under a multi-task data root")
    parser.add_argument(
        "--cameras", nargs="+", choices=CAMERAS, default=list(CAMERAS),
        help="Ordered one-to-three views; head_camera is the main view. Freeze a separate manifest for each selection.",
    )
    parser.add_argument("--manifest", type=Path, required=True, help="One saved split reused by all four policies")
    parser.add_argument("--run-dir", type=Path, help="New, nonexisting directory; required for training")
    parser.add_argument("--max-steps", type=positive_integer, default=None, help="Required for training; e.g. 3")
    parser.add_argument("--horizon", type=positive_integer, default=16)
    parser.add_argument("--execute-prefix", type=positive_integer, default=4)
    parser.add_argument("--group-size", type=positive_integer, default=4)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument(
        "--cpu-threads",
        type=positive_integer,
        default=4,
        help="torch.set_num_threads value; kept small so CPU image decode/padding cannot oversubscribe a smoke run",
    )
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--per-token-size", type=positive_integer, default=256)
    parser.add_argument("--mem-length", type=positive_integer, default=16)
    parser.add_argument("--retrieval-layers", type=positive_integer, default=2)
    parser.add_argument("--oft-hidden-dim", type=positive_integer, default=256)
    parser.add_argument("--diffusion-model-type", default="DiT-S")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
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

    from choice_mem.checkpoint import save_checkpoint
    from choice_mem.config import PolicyConfig
    from choice_mem.policy import ChoiceMemoryPolicy

    device = torch.device(arguments.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only explicit CPU/CUDA devices are supported")
    # Native JPEG decode plus horizon padding is CPU heavy; keep the thread pool small.
    torch.set_num_threads(arguments.cpu_threads)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; preparation needs no GPU, but this training request does")
        torch.cuda.set_device(device)
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    dataset = RMBenchSequentialGroups(manifest, split="train")
    if not len(dataset):
        raise ValueError("Manifest contains no train groups")
    run_directory = arguments.run_dir.expanduser().absolute()
    run_directory.mkdir(parents=True, exist_ok=False)
    config = PolicyConfig(
        backbone=arguments.backbone,
        action_head=arguments.action_head,
        model_source=arguments.model_path,
        model_revision=arguments.model_revision,
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
        freeze_backbone=not arguments.unfreeze_backbone,
        dtype=arguments.dtype,
    )
    protocol = {
        **manifest["protocol"],
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
        "optimizer": "AdamW",
        "learning_rate": arguments.learning_rate,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "sampler": "deterministic whole-group permutation per epoch; single episode per forward",
        "training_seed": arguments.seed,
        "torch_cpu_threads": arguments.cpu_threads,
        "policy_architecture": {
            "backbone": arguments.backbone,
            "action_head": arguments.action_head,
            "per_token_size": arguments.per_token_size,
            "mem_length": arguments.mem_length,
            "retrieval_layers": arguments.retrieval_layers,
            "oft_hidden_dim": arguments.oft_hidden_dim,
            "diffusion_model_type": arguments.diffusion_model_type,
            "freeze_backbone": not arguments.unfreeze_backbone,
            "dtype": arguments.dtype,
        },
    }
    metrics = {
        "status": "initializing",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "backbone": arguments.backbone,
        "action_head": arguments.action_head,
        "device": str(device),
        "dtype": arguments.dtype,
        "target_offset": manifest["protocol"]["target_offset"],
        "torch_cpu_threads": arguments.cpu_threads,
        "max_steps": arguments.max_steps,
        "completed_steps": 0,
        "manifest_sha256": manifest["manifest_sha256"],
        "steps": [],
        "peak_memory_allocated_bytes": 0 if device.type == "cuda" else None,
        "peak_memory_reserved_bytes": 0 if device.type == "cuda" else None,
        "memory_scope": "PyTorch allocator on selected GPU, not system-wide nvidia-smi usage; CPU is null",
    }
    metrics_path = run_directory / "metrics.json"
    write_run_json(metrics_path, metrics)
    write_run_json(run_directory / "protocol.json", protocol)
    wall_start = time.perf_counter()
    try:
        policy = ChoiceMemoryPolicy(config).to(device).train()
        metrics["parameter_counts"] = policy.parameter_counts()
        trainable_parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
        if not trainable_parameters:
            raise ValueError("Policy has no trainable parameters")
        optimizer = torch.optim.AdamW(trainable_parameters, lr=arguments.learning_rate, weight_decay=1e-4)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        print(
            json.dumps(
                {"event": "train_start", "config": config.to_dict(), "parameter_counts": metrics["parameter_counts"]}
            ),
            flush=True,
        )
        epoch = 0
        completed_steps = 0
        while completed_steps < arguments.max_steps:
            epoch_iterator = iter(dataset.iter_epoch(epoch, arguments.seed))
            while completed_steps < arguments.max_steps:
                data_start = time.perf_counter()
                try:
                    group = next(epoch_iterator)
                except StopIteration:
                    break
                data_latency = time.perf_counter() - data_start
                policy.reset()
                optimizer.zero_grad(set_to_none=True)
                actions = torch.from_numpy(group["actions"]).to(device=device, dtype=torch.float32)
                valid_mask = torch.from_numpy(group["valid_mask"]).to(device=device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                step_start = time.perf_counter()
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=arguments.dtype == "bfloat16"
                ):
                    loss = policy(
                        group["image_batches"],
                        group["instructions"],
                        actions,
                        valid_mask,
                        group["episode_ids"],
                        group["timesteps"],
                    )
                if loss.ndim != 0 or not torch.isfinite(loss).item():
                    raise FloatingPointError(f"Expected finite scalar training loss, got {loss.detach().cpu()}")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, max_norm=1.0, error_if_nonfinite=True
                )
                optimizer.step()
                policy.reset()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                step_latency = time.perf_counter() - step_start
                completed_steps += 1
                step_metrics = {
                    "step": completed_steps,
                    "epoch": epoch,
                    "loss": float(loss.detach().cpu()),
                    "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
                    "step_latency_s": step_latency,
                    "data_latency_s": data_latency,
                    "latency_per_query_s": step_latency / len(group["timesteps"]),
                    "episode_id": group["episode_ids"][0],
                    "query_timesteps": group["timesteps"],
                    "queries": len(group["timesteps"]),
                    "valid_action_steps": int(group["valid_mask"].sum()),
                    "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else None,
                    "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device)
                    if device.type == "cuda"
                    else None,
                }
                metrics["steps"].append(step_metrics)
                metrics["completed_steps"] = completed_steps
                metrics["status"] = "training"
                metrics["peak_memory_allocated_bytes"] = step_metrics["peak_memory_allocated_bytes"]
                metrics["peak_memory_reserved_bytes"] = step_metrics["peak_memory_reserved_bytes"]
                write_run_json(metrics_path, metrics)
                print(json.dumps({"event": "train_step", **step_metrics}), flush=True)
            epoch += 1
        checkpoint_path = run_directory / f"checkpoint-step{completed_steps:06d}.pt"
        save_checkpoint(
            checkpoint_path, policy, manifest["normalization"], protocol, completed_steps, optimizer=optimizer
        )
        metrics.update(
            {
                "status": "completed",
                "checkpoint": str(checkpoint_path),
                "final_loss": metrics["steps"][-1]["loss"],
                "mean_step_latency_s": float(np.mean([step["step_latency_s"] for step in metrics["steps"]])),
                "wall_time_s": time.perf_counter() - wall_start,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        write_run_json(metrics_path, metrics)
        print(
            json.dumps(
                {"event": "train_complete", **{name: value for name, value in metrics.items() if name != "steps"}}
            ),
            flush=True,
        )
        return metrics
    except Exception as error:
        metrics.update(
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "wall_time_s": time.perf_counter() - wall_start,
            }
        )
        write_run_json(metrics_path, metrics)
        print(
            json.dumps(
                {"event": "train_failed", "completed_steps": metrics["completed_steps"], "error": metrics["error"]}
            ),
            flush=True,
        )
        raise


def main(argv: list[str] | None = None) -> dict:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    validate_cameras(arguments.cameras)
    if not arguments.prepare_only:
        if arguments.max_steps is None:
            parser.error("training requires explicit --max-steps (for a short smoke, use --max-steps 3)")
        if arguments.run_dir is None:
            parser.error("training requires --run-dir pointing to a new directory")
        if arguments.run_dir.exists():
            parser.error("--run-dir already exists; refusing to overwrite an earlier run")
        if arguments.backbone == "memoryvla" and arguments.model_path == DEFAULT_QWEN_MODEL:
            parser.error("--backbone memoryvla requires an explicit original Prismatic/MemoryVLA --model-path")
        if not np.isfinite(arguments.learning_rate) or arguments.learning_rate <= 0:
            parser.error("--learning-rate must be finite and positive")
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
    )
    preparation_summary = summarize_manifest(manifest)
    print(json.dumps(preparation_summary), flush=True)
    if arguments.prepare_only:
        return preparation_summary
    return run_training(arguments, manifest)


if __name__ == "__main__":
    main()
