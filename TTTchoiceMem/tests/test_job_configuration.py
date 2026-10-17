"""Check independent head/view choices without loading weights or starting jobs."""

from copy import deepcopy
import json
from pathlib import Path
import shlex
import subprocess

import pytest

import configure_training_job
from run_training_job import command_option, training_cameras


@pytest.fixture
def cluster_template():
    return json.loads(configure_training_job.CLUSTER_TEMPLATE.read_text(encoding="utf-8"))


@pytest.mark.parametrize("gpu_name", ["NVIDIA A100-SXM4-80GB", "NVIDIA A100-PCIE-40GB", "NVIDIA A800-SXM4-80GB"])
def test_cluster_model_check_accepts_a100_and_a800_without_changing_allocation(gpu_name):
    gpu_names = {gpu_id: gpu_name for gpu_id in range(8)}
    assert configure_training_job.validate_cluster_gpu_models(tuple(range(8)), gpu_names) == gpu_names


def test_cluster_model_check_still_rejects_missing_wrong_family_or_fewer_devices():
    gpu_names = {gpu_id: "NVIDIA A100-SXM4-80GB" for gpu_id in range(8)}
    for unsupported_name in (None, "NVIDIA GeForce RTX 4090", "NVIDIA RTX A1000"):
        changed_gpu_names = {**gpu_names, 7: unsupported_name}
        with pytest.raises(RuntimeError, match="index7"):
            configure_training_job.validate_cluster_gpu_models(tuple(range(8)), changed_gpu_names)
    with pytest.raises(RuntimeError, match="index7"):
        configure_training_job.validate_cluster_gpu_models(tuple(range(8)), {gpu_id: gpu_names[gpu_id] for gpu_id in range(7)})
    for invalid_indices in (tuple(range(4)), (0,) * 8, (0, 1, 2, 3, 4, 5, 6, 8)):
        with pytest.raises(ValueError, match="eight unique assigned"):
            configure_training_job.validate_cluster_gpu_models(invalid_indices, gpu_names)


@pytest.mark.parametrize("action_head", ["oft", "diffusion"])
@pytest.mark.parametrize("world_size", [4, 8])
def test_formal_head_is_independent_of_four_or_eight_rank_budget(cluster_template, action_head, world_size):
    payload = configure_training_job.build_cluster_variant(cluster_template, action_head, ["head_camera"], 3)
    gpu_ids = list(range(world_size))
    payload["allowed_gpus"] = gpu_ids
    payload["environment"]["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    payload["evaluation_config"]["gpu_ids"] = gpu_ids
    payload["evaluation_config"]["ports"] = list(range(18861, 18861 + world_size * 3))
    configure_training_job.replace_command_option(payload["training_command"], "--nproc-per-node", [str(world_size)])
    configure_training_job.replace_command_option(payload["training_command"], "--global-batch-size", [str(world_size * 16)])
    plan = configure_training_job.validate_variant(payload)
    assert plan.evaluation_config.expected_action_head == action_head
    assert training_cameras(plan.training_command) == ("head_camera",)
    assert plan.global_batch_size == world_size * 16
    if action_head == "oft":
        assert command_option(plan.training_command, "--diffusion-model-type") is None
    else:
        assert command_option(plan.training_command, "--diffusion-model-type") == "DiT-L"


def test_variant_does_not_mutate_template_and_rejects_evaluation_view_mismatch(cluster_template):
    original = deepcopy(cluster_template)
    payload = configure_training_job.build_cluster_variant(cluster_template, "oft", ["head_camera", "right_camera"], 2)
    assert cluster_template == original
    plan = configure_training_job.validate_variant(payload)
    assert plan.evaluation_config.workers_per_gpu == 2
    assert len(plan.evaluation_config.ports) == 16
    assert "head_right" in command_option(plan.training_command, "--manifest")
    assert "data_manifest_internal_sha256" not in payload
    payload["evaluation_config"]["expected_cameras"] = ["head_camera"]
    with pytest.raises(ValueError, match="same ordered cameras"):
        configure_training_job.validate_variant(payload)


def test_variant_publication_is_immutable(tmp_path, monkeypatch, cluster_template):
    (tmp_path / "protocols").mkdir()
    monkeypatch.setattr(configure_training_job, "PROJECT_ROOT", tmp_path)
    payload = configure_training_job.build_cluster_variant(cluster_template, "oft", ["head_camera"], 3)
    plan_path = configure_training_job.freeze_variant(payload)
    original_bytes = plan_path.read_bytes()
    assert configure_training_job.freeze_variant(payload) == plan_path
    assert plan_path.read_bytes() == original_bytes
    payload["safety"] = "Different frozen metadata"
    with pytest.raises(ValueError, match="different frozen plan"):
        configure_training_job.freeze_variant(payload)
    assert plan_path.read_bytes() == original_bytes


@pytest.mark.parametrize("action_head,cameras,workers", [
    ("oft", ["head_camera"], 3),
    ("diffusion", ["head_camera", "right_camera"], 2),
    ("oft", ["head_camera", "left_camera", "right_camera"], 1),
])
def test_shell_selectors_print_exact_variant_without_publishing(action_head, cameras, workers):
    project_root = Path(__file__).resolve().parents[1]
    launch_command = [
        "/bin/bash", str(project_root / "script/train/run_qwen35_oft_a8008.sh"),
        "--action-head", action_head, "--cameras", *cameras, "--workers-per-gpu", str(workers), "--dry-run",
    ]
    response = subprocess.run(
        ["/bin/sh", "-c", "exec " + shlex.join(launch_command)],
        check=True, capture_output=True, text=True, timeout=30,
    )
    payload = json.loads(response.stdout)
    plan = configure_training_job.validate_variant(payload)
    assert training_cameras(plan.training_command) == tuple(cameras)
    assert plan.evaluation_config.workers_per_gpu == workers
    assert plan.evaluation_config.expected_action_head == action_head
    assert len(plan.evaluation_config.ports) == 8 * workers
