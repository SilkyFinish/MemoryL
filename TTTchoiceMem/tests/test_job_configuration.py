"""Local immutable stage-1 plans; no weights, CUDA, model or job is loaded."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

import configure_training_job
from run_training_job import TrainingJobPlan, command_option, training_cameras


@pytest.fixture
def original_assets(tmp_path):
    source = tmp_path / "original-memoryvla"
    (source / "checkpoints").mkdir(parents=True)
    (source / "config.json").write_text(json.dumps({"model": {"model_id": "original-prismatic"}}))
    (source / "checkpoints/latest-checkpoint.pt").write_bytes(b"CPU metadata fixture, never loaded")
    return source


def plan_arguments(tmp_path, original_assets, *extra):
    return [
        "--run-name", "stage1-cpu-test", "--model-path", str(original_assets),
        "--data-root", str(tmp_path / "native"), "--manifest", str(tmp_path / "stage1-data.json"),
        "--run-dir", str(tmp_path / "training"), "--job-root", str(tmp_path / "job"),
        "--eval-root", str(tmp_path / "evaluation"), "--plan", str(tmp_path / "stage1-plan.json"),
        "--gpu-ids", "2", "6", "--max-steps", "123", "--tasks", "one_task", *extra,
    ]


def build_payload(tmp_path, original_assets, *extra):
    arguments = configure_training_job.build_parser().parse_args(plan_arguments(tmp_path, original_assets, *extra))
    return configure_training_job.build_stage1_plan(arguments)


def test_stage1_plan_defaults_are_local_original_method_and_explicit_budget(tmp_path, original_assets):
    payload = build_payload(tmp_path, original_assets)
    plan = configure_training_job.validate_variant(payload)
    command = plan.training_command
    assert payload["evaluation_config"] is None
    assert plan.evaluation_config is None
    assert command[0] == sys.executable
    assert str(configure_training_job.PROJECT_ROOT / "train_multitask.py") in command
    assert command_option(command, "--backbone") == "memoryvla"
    assert command_option(command, "--action-head") == "diffusion"
    assert command_option(command, "--diffusion-model-type") == "DiT-L"
    assert command_option(command, "--model-path") == str(original_assets)
    assert command_option(command, "--model-revision") is None
    assert training_cameras(command) == ("head_camera",)
    assert command_option(command, "--history-sampling") == "random"
    assert "--image-aug" in command and "--no-mask-padding-loss" in command
    assert command_option(command, "--repeated-diffusion-steps") == "4"
    assert command_option(command, "--inference-timestep-mode") == "query"
    assert command_option(command, "--optimization-recipe") == "memoryvla"
    assert float(command_option(command, "--learning-rate")) == 2e-5
    assert float(command_option(command, "--weight-decay")) == 0
    assert float(command_option(command, "--gradient-clip-norm")) == 1
    assert command_option(command, "--max-steps") == "123"
    assert (plan.global_batch_size, plan.gradient_accumulation) == (32, 1)
    assert "--unfreeze-backbone" in command and "--gradient-checkpointing" not in command
    assert command_option(command, "--sampling-strategy") == "episode"
    assert not any("/memory/TTTchoiceMem/" in token for token in command[1:])


@pytest.mark.parametrize("action_head,cameras", [("diffusion", ["head_camera"]), ("oft", ["head_camera", "right_camera"])])
def test_explicit_head_camera_freeze_and_legacy_method_options_reach_frozen_command(tmp_path, original_assets, action_head, cameras):
    payload = build_payload(tmp_path, original_assets, "--action-head", action_head, "--cameras", *cameras,
                            "--freeze-backbone", "--history-sampling", "sequential", "--no-image-aug",
                            "--repeated-diffusion-steps", "1", "--mask-padding-loss", "--inference-timestep-mode", "action")
    command = configure_training_job.validate_variant(payload).training_command
    assert "--freeze-backbone" in command and "--unfreeze-backbone" not in command
    assert training_cameras(command) == tuple(cameras)
    assert command_option(command, "--action-head") == action_head
    assert command_option(command, "--history-sampling") == "sequential"
    assert "--no-image-aug" in command and "--mask-padding-loss" in command


def test_qwen_recipe_is_explicit_opt_in_not_required_by_budget(tmp_path, original_assets):
    payload = build_payload(tmp_path, original_assets, "--backbone", "qwen35", "--optimization-recipe", "qwen-legacy")
    plan = configure_training_job.validate_variant(payload)
    assert float(command_option(plan.training_command, "--learning-rate")) == 1e-4
    assert float(command_option(plan.training_command, "--weight-decay")) == 0.005
    assert float(command_option(plan.training_command, "--gradient-clip-norm")) == 2.5
    assert plan.target_optimizer_steps == 123


def test_publication_is_immutable_and_refuses_existing_outputs(tmp_path, original_assets):
    payload = build_payload(tmp_path, original_assets)
    path = tmp_path / "new-stage1-plan.json"
    configure_training_job.freeze_variant(payload, path)
    original = path.read_bytes()
    assert configure_training_job.freeze_variant(payload, path) == path
    assert path.read_bytes() == original
    changed = deepcopy(payload)
    changed["safety"] = "different frozen metadata"
    with pytest.raises(ValueError, match="different frozen plan"):
        configure_training_job.freeze_variant(changed, path)
    assert path.read_bytes() == original
    (tmp_path / "training").mkdir()
    with pytest.raises(FileExistsError, match="outputs already exist"):
        configure_training_job.freeze_variant(payload, path)


def test_historical_external_plan_is_rejected_not_rewritten():
    path = configure_training_job.CLUSTER_TEMPLATE
    original = path.read_bytes()
    with pytest.raises(ValueError, match="another checkout"):
        TrainingJobPlan.load(path)
    with pytest.raises(ValueError, match="another checkout"):
        configure_training_job.build_cluster_variant(json.loads(original), "diffusion", ["head_camera"], 1)
    assert path.read_bytes() == original


@pytest.mark.parametrize("missing", ["--max-steps", "--model-path"])
def test_budget_and_model_assets_are_required_even_in_dry_run(tmp_path, original_assets, missing):
    arguments = plan_arguments(tmp_path, original_assets)
    index = arguments.index(missing)
    del arguments[index:index + 2]
    with pytest.raises(SystemExit):
        configure_training_job.main([*arguments, "--dry-run"])
    assert not (tmp_path / "training").exists()
    assert not (tmp_path / "stage1-plan.json").exists()


def test_shell_dry_run_stays_in_local_checkout_and_does_not_publish_or_launch(tmp_path, original_assets):
    project_root = Path(__file__).resolve().parents[1]
    command = ["/bin/bash", str(project_root / "script/train/run_qwen35_oft_a8008.sh"), "--python", sys.executable,
               *plan_arguments(tmp_path, original_assets), "--dry-run"]
    response = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    payload = json.loads(response.stdout)
    plan = configure_training_job.validate_variant(payload)
    assert plan.evaluation_config is None
    assert plan.training_command[0] == sys.executable
    for path in (tmp_path / "job", tmp_path / "training", tmp_path / "evaluation", tmp_path / "stage1-plan.json"):
        assert not path.exists()


@pytest.mark.parametrize("random_frozen", [False, True])
def test_plan_dry_run_does_not_import_torch_or_model_modules(tmp_path, original_assets, random_frozen):
    project_root = Path(__file__).resolve().parents[1]
    program = """
import importlib.abc
import sys
class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'tensorflow'} or fullname in {'choice_mem.policy', 'choice_mem.backbones', 'choice_mem.checkpoint'}:
            raise AssertionError('Plan construction imported model code: ' + fullname)
sys.meta_path.insert(0, BlockModels())
sys.path.insert(0, sys.argv.pop(1))
import configure_training_job
configure_training_job.main(sys.argv[1:])
assert 'torch' not in sys.modules
"""
    extra = []
    if random_frozen:
        metadata = tmp_path / "llama-metadata"
        metadata.mkdir()
        for name, payload in {
            "config.json": {"model_type": "llama"}, "tokenizer_config.json": {}, "tokenizer.json": {},
        }.items():
            (metadata / name).write_text(json.dumps(payload))
        source = tmp_path / "metadata-manifest.json"
        source.write_text(json.dumps({"llama_metadata": {"path": str(metadata), "files": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in metadata.iterdir()
        }}, "checkpoint_path": "/does-not-exist/33GB.pt", "local_vision_files": {"vision": "/does-not-exist/vision.bin"}}))
        extra = ["--model-path", str(source), "--backbone-initialization", "random-frozen-debug",
                 "--backbone-initialization-seed", "123", "--base-vlm-id", "prism-dinosiglip-224px+7b", "--freeze-backbone"]
    response = subprocess.run([sys.executable, "-I", "-B", "-c", program, str(project_root), *plan_arguments(tmp_path, original_assets, *extra), "--dry-run"],
                              check=True, capture_output=True, text=True, timeout=30)
    payload = json.loads(response.stdout)
    assert payload["evaluation_config"] is None
    command = payload["training_command"]
    assert command_option(command, "--backbone-initialization") == ("random-frozen-debug" if random_frozen else "pretrained")
    assert command_option(command, "--backbone-initialization-seed") == ("123" if random_frozen else "42")
    assert command_option(command, "--base-vlm-id") == "prism-dinosiglip-224px+7b"
    if random_frozen:
        assert "--freeze-backbone" in command and "--unfreeze-backbone" not in command
        assert "--gradient-checkpointing" not in command
        assert command_option(command, "--model-path") == str(source)
        # The exact frozen argv must survive validation/publication/load, with
        # no implied launch or metadata-to-pretrained rewrite.
        path = tmp_path / "published-debug-plan.json"
        configure_training_job.freeze_variant(payload, path)
        assert list(TrainingJobPlan.load(path).training_command) == command
    for directory in ("job", "training", "evaluation"):
        assert not (tmp_path / directory).exists()
    assert not (tmp_path / "stage1-plan.json").exists()


@pytest.mark.parametrize("gpu_name", ["NVIDIA A100-SXM4-80GB", "NVIDIA A100-PCIE-40GB", "NVIDIA A800-SXM4-80GB"])
def test_legacy_gpu_inventory_check_remains_explicit(gpu_name):
    names = {gpu_id: gpu_name for gpu_id in range(8)}
    assert configure_training_job.validate_cluster_gpu_models(tuple(range(8)), names) == names
    with pytest.raises(ValueError, match="eight unique"):
        configure_training_job.validate_cluster_gpu_models((2, 6), names)
