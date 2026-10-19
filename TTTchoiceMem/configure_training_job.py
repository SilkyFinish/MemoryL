"""Freeze a fresh local stage-1 training plan; never launch training or evaluation.

All training budgets and original/Turbo local assets must be explicit. Historical
Qwen plans are not templates for this route and are never rewritten. --dry-run
prints the exact plan without importing torch or creating any output.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys

from choice_mem.data import write_manifest_once
from eval_multitask import EvaluationConfig, require_absolute_path
from run_training_job import PROJECT_ROOT, TrainingJobPlan
from train_multitask import build_parser as build_training_parser, validate_arguments
from train_rmbench import validate_output_path


# Explicit legacy callers may still inspect this file; the CLI never reads it.
CLUSTER_TEMPLATE = PROJECT_ROOT / "protocols/qwen35_oft_fullft_rmbench10_a8008_global128_20261013_plan.json"


def validate_cluster_gpu_models(gpu_ids: tuple[int, ...], gpu_names: dict[int, str]) -> dict[int, str]:
    """Legacy eight-Ampere allocation check, not a default stage-1 requirement."""
    if len(gpu_ids) != 8 or len(set(gpu_ids)) != 8 or any(type(gpu_id) is not int or gpu_id not in range(8) for gpu_id in gpu_ids):
        raise ValueError("This cluster launcher requires eight unique assigned GPU indices 0-7")
    selected = {}
    for gpu_id in gpu_ids:
        name = gpu_names.get(gpu_id)
        if not isinstance(name, str) or re.search(r"\b(?:A100|A800)\b", name, flags=re.IGNORECASE) is None:
            raise RuntimeError(f"This launcher requires eight assigned A100 or A800 GPUs; index{gpu_id}: {name}")
        selected[gpu_id] = name
    return selected


def replace_command_option(command: list[str], option: str, values: list[str]) -> None:
    if option in command:
        option_index = command.index(option)
        following_index = option_index + 1
        while following_index < len(command) and not command[following_index].startswith("--"):
            following_index += 1
        del command[option_index:following_index]
    if values:
        command.extend([option, *values])


def build_cluster_variant(template: dict, action_head: str, cameras: list[str], workers_per_gpu: int) -> dict:
    """Explicit legacy API only: reject foreign plans rather than relocating them."""
    from choice_mem.config import validate_cameras

    validate_variant(template)
    if action_head not in {"diffusion", "oft"} or workers_per_gpu not in (1, 2, 3):
        raise ValueError("Select diffusion/oft and one, two or three evaluation workers per GPU")
    payload = deepcopy(template)
    selected = validate_cameras(cameras)
    replace_command_option(payload["training_command"], "--action-head", [action_head])
    replace_command_option(payload["training_command"], "--cameras", selected)
    if payload.get("evaluation_config") is not None:
        payload["evaluation_config"].update(expected_action_head=action_head, expected_cameras=selected, workers_per_gpu=workers_per_gpu)
        payload["evaluation_config"]["ports"] = list(range(18861, 18861 + len(payload["allowed_gpus"]) * workers_per_gpu))
    return payload


def validate_variant(payload: dict) -> TrainingJobPlan:
    plan = TrainingJobPlan(
        job_root=require_absolute_path(payload["job_root"], "job_root"),
        run_dir=require_absolute_path(payload["run_dir"], "run_dir"),
        eval_root=require_absolute_path(payload["eval_root"], "eval_root"),
        allowed_gpus=tuple(payload["allowed_gpus"]), environment=dict(payload["environment"]),
        training_command=tuple(payload["training_command"]),
        evaluation_config=EvaluationConfig.from_dict(payload["evaluation_config"]) if payload.get("evaluation_config") is not None else None,
        source_sha256=hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest(),
    )
    plan.validate()
    return plan


def freeze_variant(payload: dict, plan_path: Path | None = None) -> Path:
    plan = validate_variant(payload)
    plan_path = plan_path or PROJECT_ROOT / "protocols" / f"{plan.run_dir.name}_plan.json"
    plan_path = validate_output_path(plan_path, "--plan")
    if any(path.exists() for path in (plan.job_root, plan.run_dir, plan.eval_root)):
        raise FileExistsError("Stage-1 plan outputs already exist; do not overwrite/reuse historical runs")
    serialized = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    try:
        write_manifest_once(plan_path, payload)
    except FileExistsError:
        if plan_path.read_text(encoding="utf-8") != serialized:
            raise ValueError(f"Refusing to replace a different frozen plan: {plan_path}")
    TrainingJobPlan.load(plan_path)
    return plan_path


def build_stage1_plan(arguments) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", arguments.run_name):
        raise ValueError("--run-name must be a new simple filename (letters, numbers, dot, underscore, dash)")
    arguments.run_dir = arguments.run_dir or PROJECT_ROOT / "runs" / arguments.run_name
    arguments.manifest = arguments.manifest or PROJECT_ROOT / "protocols" / f"{arguments.run_name}_data.json"
    arguments.run_dir = arguments.run_dir.expanduser().absolute()
    arguments.manifest = arguments.manifest.expanduser().absolute()
    if arguments.prepare_only or arguments.resume is not None or arguments.stop_after is not None:
        raise ValueError("Freeze a complete formal training plan, not preparation/resume/preflight flags")
    arguments.global_batch_size = arguments.global_batch_size or len(arguments.gpu_ids) * arguments.per_device_batch_size * arguments.grad_accumulation
    validate_arguments(arguments)
    python = arguments.python.expanduser().absolute()
    command = [
        str(python), "-m", "torch.distributed.run", "--nproc-per-node", str(len(arguments.gpu_ids)),
        "--master-addr", "127.0.0.1", "--master-port", str(arguments.master_port), str(PROJECT_ROOT / "train_multitask.py"),
    ]
    fields = [
        "backbone", "action_head", "model_path", "backbone_initialization", "backbone_initialization_seed", "base_vlm_id",
        "data_root", "manifest", "run_dir", "max_steps",
        "horizon", "execute_prefix", "group_size", "per_device_batch_size", "grad_accumulation", "global_batch_size",
        "history_sampling", "repeated_diffusion_steps", "inference_timestep_mode", "diffusion_model_type",
        "dtype", "per_token_size", "mem_length", "retrieval_layers", "oft_hidden_dim", "cpu_threads", "device",
        "optimization_recipe", "learning_rate", "weight_decay", "warmup_ratio", "gradient_clip_norm",
        "sampling_strategy", "save_every", "snapshot_every", "seed", "validation_fraction",
    ]
    if arguments.model_revision is not None:
        fields.append("model_revision")
    if arguments.optimization_recipe == "qwen-legacy":
        fields += ["backbone_learning_rate", "minimum_backbone_learning_rate", "minimum_head_learning_rate"]
    for field in fields:
        value = getattr(arguments, field)
        if isinstance(value, Path):
            value = value.expanduser().absolute()
        command += ["--" + field.replace("_", "-"), str(value)]
    command += ["--cameras", *arguments.cameras]
    if arguments.tasks is not None:
        command += ["--tasks", *arguments.tasks]
    command += ["--unfreeze-backbone" if arguments.unfreeze_backbone else "--freeze-backbone"]
    command += ["--image-aug" if arguments.image_aug else "--no-image-aug"]
    command += ["--mask-padding-loss" if arguments.mask_padding_loss else "--no-mask-padding-loss"]
    command += ["--find-unused-parameters" if arguments.find_unused_parameters else "--no-find-unused-parameters"]
    if arguments.gradient_checkpointing:
        command.append("--gradient-checkpointing")
    payload = {
        "schema": "tttchoicemem-stage1-training-plan-v1",
        "job_root": str((arguments.job_root or PROJECT_ROOT / "runs" / f"job_{arguments.run_name}").expanduser().absolute()),
        "run_dir": str(arguments.run_dir),
        "eval_root": str((arguments.eval_root or PROJECT_ROOT / "runs" / f"eval_{arguments.run_name}").expanduser().absolute()),
        "allowed_gpus": arguments.gpu_ids,
        "environment": {
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, arguments.gpu_ids)), "OMP_NUM_THREADS": str(arguments.cpu_threads),
            "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        },
        "training_command": command,
        "evaluation_config": None,
        "training_budget_definition": f"{len(arguments.gpu_ids)} ranks * {arguments.group_size} queries * accumulation {arguments.grad_accumulation}; {arguments.max_steps} explicit optimizer updates",
        "checkpoint_policy": "Full optimizer/scheduler/per-rank RNG/stateless sampler resume; retain last two rolling checkpoints; final snapshot only at exact max_steps",
        "evaluation_gate": "Training-only plan. Evaluation is a separate explicit operation after exact-budget finite full-checkpoint validation.",
        "execute_prefix_provenance": "RMBench deployment choice, not a history stride or original-algorithm change",
        "safety": "No implicit launch, evaluation, foreign training code, historical plan rewrite or output reuse",
    }
    validate_variant(payload)
    return payload


def build_parser():
    parser = build_training_parser()
    parser.description = __doc__
    parser._option_string_actions["--manifest"].required = False
    parser.add_argument("--run-name", required=True, help="Fresh stage-1 experiment name; never an old output name")
    parser.add_argument("--gpu-ids", type=int, nargs="+", required=True, help="Explicit assigned physical GPU ids")
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="Explicit training interpreter; defaults to this process's sys.executable")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--job-root", type=Path)
    parser.add_argument("--eval-root", type=Path)
    parser.add_argument("--plan", type=Path, help="New immutable local plan path")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        payload = build_stage1_plan(arguments)
        if arguments.dry_run:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(freeze_variant(payload, arguments.plan))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
