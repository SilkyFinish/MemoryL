"""Freeze a fresh cluster variant without changing any existing experiment plan."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shlex

from choice_mem.config import SUPPORTED_CAMERAS, validate_cameras
from choice_mem.data import write_manifest_once
from eval_multitask import EvaluationConfig, require_absolute_path
from run_training_job import PROJECT_ROOT, TrainingJobPlan, build_evaluation_command


CLUSTER_TEMPLATE = PROJECT_ROOT / "protocols/qwen35_oft_fullft_rmbench10_a8008_global128_20261013_plan.json"


def validate_cluster_gpu_models(gpu_ids: tuple[int, ...], gpu_names: dict[int, str]) -> dict[int, str]:
    """Accept the supported Ampere cluster families without weakening allocation checks."""
    if (
        len(gpu_ids) != 8 or len(set(gpu_ids)) != 8
        or any(type(gpu_id) is not int or gpu_id not in range(8) for gpu_id in gpu_ids)
    ):
        raise ValueError("This cluster launcher requires eight unique assigned GPU indices 0-7")
    selected_gpu_names = {}
    for gpu_id in gpu_ids:
        gpu_name = gpu_names.get(gpu_id)
        if not isinstance(gpu_name, str) or re.search(r"\b(?:A100|A800)\b", gpu_name, flags=re.IGNORECASE) is None:
            raise RuntimeError(
                f"This launcher requires eight assigned A100 or A800 GPUs; index{gpu_id}: {gpu_name}"
            )
        selected_gpu_names[gpu_id] = gpu_name
    return selected_gpu_names


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
    selected_cameras = validate_cameras(cameras)
    if action_head not in {"diffusion", "oft"} or type(workers_per_gpu) is not int or workers_per_gpu not in (1, 2, 3):
        raise ValueError("Select diffusion/oft and one, two or three evaluation workers per GPU")
    payload = deepcopy(template)
    camera_tag = "_".join(camera.removesuffix("_camera") for camera in selected_cameras)
    run_tag = f"qwen35_{action_head}_rmbench10_a8008_global128_{camera_tag}_w{workers_per_gpu}_20261013"
    payload["job_root"] = str(PROJECT_ROOT / "runs" / f"job_{run_tag}")
    payload["run_dir"] = str(PROJECT_ROOT / "runs" / run_tag)
    payload["eval_root"] = str(PROJECT_ROOT / "runs" / f"eval_{run_tag}_50")
    command = payload["training_command"]
    replace_command_option(command, "--action-head", [action_head])
    replace_command_option(command, "--cameras", selected_cameras)
    replace_command_option(command, "--run-dir", [payload["run_dir"]])
    replace_command_option(command, "--diffusion-model-type", ["DiT-L"] if action_head == "diffusion" else [])
    replace_command_option(command, "--oft-hidden-dim", ["256"] if action_head == "oft" else [])
    if selected_cameras != list(SUPPORTED_CAMERAS):
        manifest_path = PROJECT_ROOT / "protocols" / f"rmbench10_h16_p4_g16_{camera_tag}_20261013.json"
        replace_command_option(command, "--manifest", [str(manifest_path)])
        payload.pop("data_manifest_internal_sha256", None)
    evaluation = payload["evaluation_config"]
    evaluation["expected_action_head"] = action_head
    evaluation["expected_cameras"] = selected_cameras
    evaluation["workers_per_gpu"] = workers_per_gpu
    evaluation["ports"] = list(range(18861, 18861 + len(evaluation["gpu_ids"]) * workers_per_gpu))
    payload["action_head"] = action_head
    payload["observation_cameras"] = selected_cameras
    payload["evaluation_gate"] = "Exact55000,matching action head,cameras,global128/world8,finite full model and frozen data before500 native episodes"
    payload["model_initialization"] = "Pinned official Qwen with fresh compressor/Banks/selected head;never resume another variant"
    payload["checkpoint_policy"] = "Every5000 full resumable checkpoint(last2) and inference snapshot;final.pt only after55000"
    return payload


def freeze_variant(payload: dict) -> Path:
    validate_variant(payload)
    plan_path = PROJECT_ROOT / "protocols" / f"{Path(payload['run_dir']).name}_plan.json"
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        # Atomic, exclusive publication also prevents readers seeing a partial plan.
        write_manifest_once(plan_path, payload)
    except FileExistsError:
        if plan_path.read_text(encoding="utf-8") != serialized:
            raise ValueError(f"Refusing to replace a different frozen plan: {plan_path}")
    TrainingJobPlan.load(plan_path)
    return plan_path


def validate_variant(payload: dict) -> TrainingJobPlan:
    plan = TrainingJobPlan(
        job_root=require_absolute_path(payload["job_root"], "job_root"),
        run_dir=require_absolute_path(payload["run_dir"], "run_dir"),
        eval_root=require_absolute_path(payload["eval_root"], "eval_root"),
        allowed_gpus=tuple(payload["allowed_gpus"]), environment=dict(payload["environment"]),
        training_command=tuple(payload["training_command"]),
        evaluation_config=EvaluationConfig.from_dict(payload["evaluation_config"]),
        source_sha256=hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest(),
    )
    plan.validate()
    return plan


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-head", choices=("diffusion", "oft"), default="oft")
    parser.add_argument("--cameras", nargs="+", choices=SUPPORTED_CAMERAS, default=["head_camera"])
    parser.add_argument("--workers-per-gpu", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate this variant's final snapshot without training")
    parser.add_argument("--dry-run", action="store_true", help="Print the variant without creating even a plan file")
    arguments = parser.parse_args(argv)
    template = json.loads(CLUSTER_TEMPLATE.read_text(encoding="utf-8"))
    payload = build_cluster_variant(template, arguments.action_head, arguments.cameras, arguments.workers_per_gpu)
    plan = validate_variant(payload)
    if arguments.evaluate_only:
        command = build_evaluation_command(plan, plan.run_dir / "snapshots/final.pt")
        if arguments.dry_run:
            print(shlex.join(command))
        else:
            os.execv(command[0], command)
        return
    if arguments.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(freeze_variant(payload))


if __name__ == "__main__":
    main()
