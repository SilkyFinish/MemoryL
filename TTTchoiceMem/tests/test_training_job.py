"""CPU-only process-boundary tests: no model, simulator, GPU or actual job.

The verified allocation is four ranks on GPUs 2,3,4,6 with accumulation 7 and
activation checkpointing; the budget 4*16*7 == 448 and the 30000-step schedule
are frozen. Evaluation therefore asks for 50 *policy* episodes per task over
the official 2500-seed expert-validation pool, never 50 raw attempts.
"""

from contextlib import redirect_stdout
from dataclasses import replace
import io
import hashlib
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_multitask
import run_training_job
import train_multitask


TASK_NAMES = (
    "battery_try", "blocks_ranking_try", "cover_blocks", "observe_and_pickup", "place_block_mat",
    "press_button", "put_back_block", "rearrange_blocks", "swap_T", "swap_blocks",
)
ALLOCATED_GPUS = (2, 3, 4, 6)
ALLOCATED_PORTS = (18822, 18823, 18824, 18826)


class TrainingJobTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.rmbench_root = self.root / "native"
        for directory in ("envs", "scripts", "env_cfg/eval", "env_cfg/task_config"):
            (self.rmbench_root / directory).mkdir(parents=True)
        (self.rmbench_root / "env_cfg/eval/all_tasks.yml").write_text(
            "tasks:\n" + "".join(f"  - {task_name}\n" for task_name in TASK_NAMES), encoding="utf-8",
        )
        for source_path in (
            self.rmbench_root / "scripts/eval_policy_xpolicylab.py",
            self.rmbench_root / "env_cfg/task_config/demo_clean.yml",
            *(self.rmbench_root / "envs" / f"{task_name}.py" for task_name in TASK_NAMES),
        ):
            source_path.touch()
        self.config = eval_multitask.EvaluationConfig(
            tasks=TASK_NAMES, rmbench_root=self.rmbench_root,
            qwen_python=Path(sys.executable), env_python=Path(sys.executable),
        )
        self.run_dir = self.root / "training"
        self.job_root = self.root / "job"
        self.eval_root = self.root / "evaluation"
        self.plan_path = self.root / "frozen.json"
        self.manifest = {
            "manifest_sha256": "a" * 64, "protocol": {"image_size": [224, 224]},
            "normalization": {"training_episode_ids": ["fixture-episode"], "action": {"q01": [-1.0] * 14, "q99": [1.0] * 14}},
        }
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))
        self.plan = self.write_plan()
        self.contract = self.make_contract(self.plan.training_command)

    def build_command(self, gpu_ids=ALLOCATED_GPUS, accumulation=None, extra=(), run_dir=None, **overrides):
        if accumulation is None:
            accumulation = eval_multitask.required_accumulation(len(gpu_ids))
        options = {
            "--group-size": "16", "--global-batch-size": "448", "--max-steps": "30000",
            "--save-every": "1000", "--snapshot-every": "5000",
        }
        options.update(overrides)
        trailing = list(extra) or ["--gradient-checkpointing", "--unfreeze-backbone"]
        return [
            sys.executable, "-m", "torch.distributed.run",
            "--nproc-per-node", str(len(gpu_ids)), "--master-port", "18990",
            str(run_training_job.PROJECT_ROOT / "train_multitask.py"),
            "--backbone", "qwen35", "--optimization-recipe", "qwen-legacy",
            "--history-sampling", "sequential", "--no-image-aug",
            "--repeated-diffusion-steps", "1", "--mask-padding-loss", "--inference-timestep-mode", "action",
            "--cameras", "head_camera", "left_camera", "right_camera",
            "--data-root", str(self.root / "data"), "--manifest", str(self.root / "manifest.json"),
            "--run-dir", str(run_dir or self.run_dir), "--tasks", *TASK_NAMES,
            "--model-path", str(self.root / "model"), "--model-revision", "frozen-revision",
            "--group-size", options["--group-size"], "--global-batch-size", options["--global-batch-size"],
            "--grad-accumulation", str(accumulation),
            "--max-steps", options["--max-steps"], "--save-every", options["--save-every"],
            "--snapshot-every", options["--snapshot-every"], *trailing,
        ]

    def write_plan(self, gpu_ids=ALLOCATED_GPUS, command=None, config=None):
        payload = {
            "job_root": str(self.job_root), "run_dir": str(self.run_dir), "eval_root": str(self.eval_root),
            "allowed_gpus": list(gpu_ids),
            "environment": {"CUDA_VISIBLE_DEVICES": ",".join(str(gpu_id) for gpu_id in gpu_ids)},
            "training_command": command or self.build_command(gpu_ids),
            "evaluation_config": (config or self.config).to_dict(),
        }
        self.plan_path.write_text(json.dumps(payload), encoding="utf-8")
        return run_training_job.TrainingJobPlan.load(self.plan_path)

    def make_contract(self, command):
        command = tuple(command)
        index = next(index for index, token in enumerate(command) if Path(token).name == "train_multitask.py")
        args = train_multitask.build_parser().parse_args(list(command[index + 1:]))
        config = train_multitask.build_policy_configuration(args, self.manifest).to_dict()
        return {"config": config, "normalization": self.manifest["normalization"], "protocol": {
            "manifest_sha256": self.manifest["manifest_sha256"], "train_episode_ids": self.manifest["normalization"]["training_episode_ids"],
            "sampling": {"tasks": sorted(args.tasks)},
            "optimization": {"max_steps": args.max_steps, "world_size": int(run_training_job.command_option(command, "--nproc-per-node")),
                "global_batch_size": args.global_batch_size, "per_device_batch_size": args.per_device_batch_size,
                "group_size": args.group_size, "grad_accumulation": args.grad_accumulation,
                "recipe": args.optimization_recipe, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
                "gradient_clip_norm": args.gradient_clip_norm, "warmup_ratio": args.warmup_ratio,
                "backbone_learning_rate": args.backbone_learning_rate if args.optimization_recipe == "qwen-legacy" else args.learning_rate,
                "memory_head_learning_rate": args.learning_rate,
                "minimum_backbone_learning_rate": args.minimum_backbone_learning_rate if args.optimization_recipe == "qwen-legacy" else None,
                "minimum_head_learning_rate": args.minimum_head_learning_rate if args.optimization_recipe == "qwen-legacy" else None,
                "training_seed": args.seed, "gradient_checkpointing": args.gradient_checkpointing,
                "find_unused_parameters": args.find_unused_parameters,
                "schedule": "cosine with linear warmup" if args.optimization_recipe == "qwen-legacy" else "constant"},
        }}

    def allocation_config(self, gpu_ids):
        """A frozen evaluation config whose GPU/port allocation matches training."""
        ports = tuple(range(18821, 18821 + len(gpu_ids)))
        return replace(self.config, gpu_ids=tuple(gpu_ids), ports=ports)

    def prepare(self):
        run_training_job.prepare_job(self.plan, self.plan_path, None)

    def record_training_completion(self, step=30000, status="completed", field="final_snapshot"):
        self.run_dir.mkdir(exist_ok=True)
        checkpoint = self.run_dir / "arbitrary-final-inference.pt"
        checkpoint.write_bytes(b"checkpoint fixture, never loaded as a model")
        eval_multitask.write_atomic_json(self.run_dir / "progress.json", {
            "step": step, "status": status, field: checkpoint.name,
        })
        return checkpoint

    def record_suite_completion(self, episodes=500):
        self.eval_root.mkdir(exist_ok=True)
        eval_multitask.write_atomic_json(self.eval_root / "summary.json", {
            "state": "completed", "coverage_complete": True, "completed_episodes": episodes,
        })

    # ---------------------------------------------------------------- planning

    def test_verified_four_gpu_allocation_is_the_frozen_default(self):
        self.assertEqual(eval_multitask.DEFAULT_GPU_IDS, ALLOCATED_GPUS)
        self.assertEqual(eval_multitask.DEFAULT_PORTS, ALLOCATED_PORTS)
        self.assertEqual(self.config.gpu_ids, ALLOCATED_GPUS)
        self.assertEqual(self.config.ports, ALLOCATED_PORTS)
        self.assertEqual(self.config.max_seed_attempts, eval_multitask.SEED_POOL_SIZE)
        self.assertEqual(self.config.max_seed_attempts, 2500)
        self.assertEqual(eval_multitask.required_accumulation(4), 7)
        self.assertEqual(len(ALLOCATED_GPUS), 4)
        self.assertEqual(self.plan.allowed_gpus, ALLOCATED_GPUS)
        self.assertEqual(self.plan.environment["CUDA_VISIBLE_DEVICES"], "2,3,4,6")

    def test_required_accumulation_always_reaches_448_exactly(self):
        for world_size in (1, 2, 4, 7):
            with self.subTest(world_size=world_size):
                accumulation = eval_multitask.required_accumulation(world_size)
                self.assertEqual(world_size * eval_multitask.GROUP_SIZE * accumulation, 448)
        self.assertEqual(eval_multitask.required_accumulation(4), 7)
        self.assertEqual(eval_multitask.required_accumulation(7), 4)
        for world_size in (3, 5, 6, 8, 9):
            with self.subTest(world_size=world_size):
                with self.assertRaisesRegex(ValueError, "global batch"):
                    eval_multitask.required_accumulation(world_size)

    def test_plan_accepts_any_compatible_gpu_set_with_its_exact_accumulation(self):
        for gpu_ids in [(2, 6), ALLOCATED_GPUS, (0, 1, 2, 3, 4, 5, 6)]:
            with self.subTest(gpu_ids=gpu_ids):
                plan = self.write_plan(gpu_ids, config=self.allocation_config(gpu_ids))
                self.assertEqual(plan.allowed_gpus, gpu_ids)
                self.assertEqual(
                    plan.environment["CUDA_VISIBLE_DEVICES"], ",".join(str(gpu_id) for gpu_id in gpu_ids),
                )
                self.assertEqual(
                    run_training_job.command_option(plan.training_command, "--grad-accumulation"),
                    str(eval_multitask.required_accumulation(len(gpu_ids))),
                )
        mismatched = self.build_command(ALLOCATED_GPUS, accumulation=8)
        with self.assertRaisesRegex(ValueError, "global batch mismatch"):
            self.write_plan(ALLOCATED_GPUS, command=mismatched).validate()

    def test_world_sizes_that_cannot_reach_448_are_rejected(self):
        for gpu_ids in (tuple(range(8)), (0, 1, 2, 3, 4)):
            with self.subTest(gpu_ids=gpu_ids):
                # Neither 8 nor 5 ranks can divide 448 by 16 exactly, so these
                # sets must be refused rather than silently rescaled.
                world_size = len(gpu_ids)
                self.assertEqual(eval_multitask.GLOBAL_BATCH_SIZE % (world_size * eval_multitask.GROUP_SIZE) != 0, True)
                with self.assertRaisesRegex(ValueError, "global batch"):
                    eval_multitask.required_accumulation(world_size)
                with self.assertRaisesRegex(ValueError, "global batch"):
                    self.write_plan(gpu_ids, command=self.build_command(gpu_ids, accumulation=1), config=self.allocation_config(gpu_ids))

    def test_checkpointing_and_backbone_freeze_are_explicit_options_not_qwen_requirements(self):
        for extra in (["--unfreeze-backbone"], ["--gradient-checkpointing"], ["--freeze-backbone"]):
            plan = self.write_plan(command=self.build_command(extra=extra))
            plan.validate()
        with self.assertRaises(ValueError):
            self.write_plan(command=self.build_command(extra=["--freeze-backbone", "--unfreeze-backbone"]))

    def test_gpu_visibility_and_evaluation_subset_follow_allowed_gpus(self):
        payload = json.loads(self.plan_path.read_text(encoding="utf-8"))
        payload["environment"]["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"
        self.plan_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "CUDA_VISIBLE_DEVICES=2,3,4,6"):
            run_training_job.TrainingJobPlan.load(self.plan_path)
        outside_subset = replace(self.config, gpu_ids=(2, 3, 4, 5), ports=(18822, 18823, 18824, 18825))
        with self.assertRaisesRegex(ValueError, "allowed GPUs"):
            self.write_plan(config=outside_subset).validate()
        subset = replace(self.config, gpu_ids=(2, 6), ports=(18822, 18826))
        self.assertEqual(self.write_plan(config=subset).evaluation_config.gpu_ids, (2, 6))

    def test_plan_budget_must_match_explicit_evaluation_gate_and_exact_batch(self):
        with self.assertRaisesRegex(ValueError, "same frozen step budget"):
            self.write_plan(command=self.build_command(**{"--max-steps": "100"}))
        with self.assertRaisesRegex(ValueError, "global batch mismatch"):
            self.write_plan(command=self.build_command(**{"--global-batch-size": "256"}))
        with self.assertRaisesRegex(ValueError, "group size"):
            self.write_plan(command=self.build_command(**{"--group-size": "8"}))

    def test_suite_plan_and_commands_freeze_500_episodes_over_2500_expert_seeds(self):
        plan = eval_multitask.build_suite_plan(self.config, self.root / "final.pt")
        self.assertEqual(plan["total_requested_episodes"], 500)
        self.assertEqual(plan["test_num"], 50)
        self.assertEqual(plan["candidate_seeds"], list(range(100000, 102500)))
        self.assertEqual(plan["max_seed_attempts_per_task"], 2500)
        self.assertEqual(plan["seed_pool_size"], 2500)
        self.assertTrue(plan["expert_check_before_policy"])
        self.assertTrue(plan["expert_skipped_seeds_are_not_episodes"])
        self.assertFalse(plan["candidate_pool_refill_or_rerun"])
        self.assertEqual(plan["tasks"], list(TASK_NAMES))
        for task_name in TASK_NAMES:
            command = eval_multitask.build_native_command(
                self.config, task_name, 2, 18822, self.eval_root / task_name,
            )
            self.assertEqual(command[command.index("--episodes") + 1], "50")
            self.assertEqual(command[command.index("--instruction-type") + 1], "unseen")
            self.assertEqual(command[command.index("--max-seed-attempts") + 1], "2500")
            self.assertEqual(command[command.index("--sim-gpu") + 1], "2")

    def test_new_global64_recipe_uses_original_dit_l_and_gates_at_55000(self):
        command = self.build_command(
            ALLOCATED_GPUS, accumulation=1,
            **{"--global-batch-size": "64", "--max-steps": "55000", "--save-every": "5000"},
        )
        command[command.index("--model-revision") + 1] = "2fc06364715b967f1860aea9cf38778875588b17"
        fixed_options = {
            "--per-device-batch-size": "16", "--diffusion-model-type": "DiT-L",
            "--horizon": "16", "--execute-prefix": "4", "--mem-length": "16",
            "--retrieval-layers": "2", "--per-token-size": "256",
            "--backbone-learning-rate": "0.00001", "--learning-rate": "0.0001",
            "--minimum-backbone-learning-rate": "0.000001", "--minimum-head-learning-rate": "0.000005",
            "--weight-decay": "0.005", "--warmup-ratio": "0.05", "--gradient-clip-norm": "2.5",
            "--seed": "42", "--validation-fraction": "0.1",
        }
        for option_name, option_value in fixed_options.items():
            command.extend([option_name, option_value])
        config = replace(self.config, expected_training_steps=55000)
        plan = self.write_plan(ALLOCATED_GPUS, command=command, config=config)
        self.assertEqual(plan.global_batch_size, 64)
        self.assertEqual(plan.per_device_batch_size, 16)
        self.assertEqual(plan.gradient_accumulation, 1)
        self.record_training_completion(step=30000)
        with patch.object(run_training_job, "validate_checkpoint") as checkpoint_validator:
            with self.assertRaisesRegex(ValueError, "55000"):
                run_training_job.resolve_completed_checkpoint(plan)
            checkpoint_validator.assert_not_called()
        checkpoint = self.record_training_completion(step=55000)
        self.contract = self.make_contract(command)
        with patch.object(run_training_job, "validate_checkpoint", return_value=self.contract) as checkpoint_validator:
            self.assertEqual(run_training_job.resolve_completed_checkpoint(plan), checkpoint)
            checkpoint_validator.assert_called_once_with(checkpoint, expected_step=55000, expected_action_head="diffusion", expected_cameras=("head_camera", "left_camera", "right_camera"))
        evaluation_command = run_training_job.build_evaluation_command(plan, checkpoint)
        self.assertEqual(evaluation_command[evaluation_command.index("--expected-training-steps") + 1], "55000")
        command[command.index("--diffusion-model-type") + 1] = "DiT-S"
        self.assertEqual(self.write_plan(ALLOCATED_GPUS, command=command, config=config).target_optimizer_steps, 55000)
        command[command.index("--diffusion-model-type") + 1] = "DiT-L"
        command[command.index("--per-device-batch-size") + 1] = "64"
        with self.assertRaisesRegex(ValueError, "group size"):
            self.write_plan(ALLOCATED_GPUS, command=command, config=config)

    def test_frozen_evaluation_protocol_values_are_not_configurable(self):
        for changed in (
            replace(self.config, episodes=1),
            replace(self.config, max_seed_attempts=50),
            replace(self.config, seed=1),
            replace(self.config, tasks=(TASK_NAMES[0], TASK_NAMES[0])),
            replace(self.config, tasks=("unknown_native_task",)),
            replace(self.config, gpu_ids=(2, 2), ports=(18822, 18823)),
            replace(self.config, gpu_ids=(2, 3), ports=(18822, 18822)),
            replace(self.config, gpu_ids=(2, 3), ports=(19000, 19001)),
            replace(self.config, gpu_ids=(9,), ports=(18822,)),
        ):
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    changed.validate()

    def test_cluster_global128_oft_plan_gates_head_and_eight_rank_budget(self):
        command = self.build_command(
            tuple(range(8)), accumulation=1,
            **{"--global-batch-size": "128", "--max-steps": "55000", "--save-every": "5000"},
        )
        command[command.index("--model-revision") + 1] = "2fc06364715b967f1860aea9cf38778875588b17"
        fixed_options = {
            "--action-head": "oft", "--per-device-batch-size": "16", "--oft-hidden-dim": "256",
            "--horizon": "16", "--execute-prefix": "4", "--mem-length": "16",
            "--retrieval-layers": "2", "--per-token-size": "256",
            "--backbone-learning-rate": "0.00001", "--learning-rate": "0.0001",
            "--minimum-backbone-learning-rate": "0.000001", "--minimum-head-learning-rate": "0.000005",
            "--weight-decay": "0.005", "--warmup-ratio": "0.05", "--gradient-clip-norm": "2.5",
            "--seed": "42", "--validation-fraction": "0.1",
        }
        for option_name, option_value in fixed_options.items():
            command.extend([option_name, option_value])
        config = replace(
            self.allocation_config(tuple(range(8))), expected_training_steps=55000, expected_action_head="oft",
        )
        plan = self.write_plan(tuple(range(8)), command=command, config=config)
        self.assertEqual((plan.global_batch_size, plan.per_device_batch_size, plan.gradient_accumulation), (128, 16, 1))
        self.assertEqual(config.ports[-1], 18828)
        checkpoint = self.record_training_completion(step=55000)
        self.contract = self.make_contract(command)
        with patch.object(run_training_job, "validate_checkpoint", return_value=self.contract) as checkpoint_validator:
            self.assertEqual(run_training_job.resolve_completed_checkpoint(plan), checkpoint)
            checkpoint_validator.assert_called_once_with(checkpoint, expected_step=55000, expected_action_head="oft", expected_cameras=("head_camera", "left_camera", "right_camera"))
        evaluation_command = run_training_job.build_evaluation_command(plan, checkpoint)
        self.assertEqual(evaluation_command[evaluation_command.index("--expected-action-head") + 1], "oft")
        self.assertEqual(evaluation_command[evaluation_command.index("--episodes") + 1], "50")
        with self.assertRaisesRegex(ValueError, "same expected action head"):
            self.write_plan(tuple(range(8)), command=command, config=replace(config, expected_action_head="diffusion"))
        action_head_option_index = command.index("--action-head")
        missing_action_head = command[:action_head_option_index] + command[action_head_option_index + 2:]
        with self.assertRaisesRegex(ValueError, "same expected action head"):
            self.write_plan(tuple(range(8)), command=missing_action_head, config=config)
        command[command.index("--action-head") + 1] = "diffusion"
        with self.assertRaisesRegex(ValueError, "same expected action head"):
            self.write_plan(tuple(range(8)), command=command, config=config)

    # ------------------------------------------------------------------ gating

    def test_success_launches_eval_only_after_training_exit_and_snapshot_validation(self):
        self.prepare()
        events = []

        def launch(command, **options):
            if "torch.distributed.run" in command:
                events.append("launch_training")
                self.assertEqual(options["env"]["CUDA_VISIBLE_DEVICES"], "2,3,4,6")
                self.assertTrue(options["start_new_session"])

                def finish_training():
                    events.append("training_exit")
                    self.record_training_completion()
                    return 0

                return SimpleNamespace(pid=111, wait=finish_training)
            events.append("launch_evaluation")
            self.assertEqual(command[command.index("--episodes") + 1], "50")
            self.assertEqual(command[command.index("--max-seed-attempts") + 1], "2500")
            self.assertEqual(command[command.index("--gpu-ids") + 1], "2")
            self.assertEqual(command[command.index("--checkpoint") + 1], str(self.run_dir / "arbitrary-final-inference.pt"))
            self.record_suite_completion()
            return SimpleNamespace(pid=222, wait=lambda: 0)

        def validate(checkpoint, expected_step, **options):
            events.append("validate_checkpoint")
            self.assertTrue(checkpoint.is_file())
            self.assertEqual(expected_step, 30000)
            return self.contract

        with patch.object(run_training_job, "check_gpu_utilization"), \
                patch.object(run_training_job, "check_port_free"), \
                patch.object(run_training_job, "stop_owned_process"), \
                patch.object(run_training_job, "validate_checkpoint", side_effect=validate), \
                patch.object(run_training_job.subprocess, "Popen", side_effect=launch):
            self.assertEqual(run_training_job.run_job(self.plan), 0)
        self.assertEqual(events, ["launch_training", "training_exit", "validate_checkpoint", "launch_evaluation"])
        status = eval_multitask.read_json(self.job_root / "status.json")
        self.assertEqual(status["state"], "completed")
        self.assertTrue(status["evaluation_started"])
        self.assertEqual(status["allowed_gpus"], list(ALLOCATED_GPUS))
        self.assertEqual(status["gradient_accumulation"], 7)
        self.assertEqual(status["requested_seed_attempts_per_task"], 2500)

    def test_failed_training_never_checks_or_launches_evaluation(self):
        self.prepare()
        with patch.object(run_training_job, "check_gpu_utilization"), \
                patch.object(run_training_job, "check_port_free"), \
                patch.object(run_training_job, "stop_owned_process"), \
                patch.object(run_training_job, "resolve_completed_checkpoint") as completion_gate, \
                patch.object(run_training_job.subprocess, "Popen",
                             return_value=SimpleNamespace(pid=111, wait=lambda: 9)) as launch:
            self.assertEqual(run_training_job.run_job(self.plan), 1)
        launch.assert_called_once()
        completion_gate.assert_not_called()
        self.assertFalse(self.eval_root.exists())
        self.assertEqual(eval_multitask.read_json(self.job_root / "status.json")["state"], "training_failed")

    def test_zero_exit_before_budget_never_launches_eval(self):
        self.prepare()

        def finish_training():
            self.record_training_completion(step=2, status="stopped")
            return 0

        with patch.object(run_training_job, "check_gpu_utilization"), \
                patch.object(run_training_job, "check_port_free"), \
                patch.object(run_training_job, "stop_owned_process"), \
                patch.object(run_training_job, "validate_checkpoint") as validate, \
                patch.object(run_training_job.subprocess, "Popen",
                             return_value=SimpleNamespace(pid=111, wait=finish_training)):
            self.assertEqual(run_training_job.run_job(self.plan), 1)
        validate.assert_not_called()
        self.assertFalse(eval_multitask.read_json(self.job_root / "status.json")["evaluation_started"])

    def test_final_snapshot_is_preferred_then_legacy_names_then_missing_blocks_gate(self):
        snapshot = self.record_training_completion()
        legacy = self.run_dir / "legacy-checkpoint.pt"
        legacy.write_bytes(b"older field")
        progress_path = self.run_dir / "progress.json"
        progress = eval_multitask.read_json(progress_path)
        progress["final_checkpoint"] = legacy.name
        progress["checkpoint"] = legacy.name
        eval_multitask.write_atomic_json(progress_path, progress)
        with patch.object(run_training_job, "validate_checkpoint", return_value=self.contract):
            self.assertEqual(run_training_job.resolve_completed_checkpoint(self.plan), snapshot)
            progress.pop("final_snapshot")
            eval_multitask.write_atomic_json(progress_path, progress)
            self.assertEqual(run_training_job.resolve_completed_checkpoint(self.plan), legacy)
            progress.pop("final_checkpoint")
            eval_multitask.write_atomic_json(progress_path, progress)
            self.assertEqual(run_training_job.resolve_completed_checkpoint(self.plan), legacy)
            for name in ("final_snapshot", "final_checkpoint", "checkpoint"):
                progress.pop(name, None)
            eval_multitask.write_atomic_json(progress_path, progress)
            with self.assertRaisesRegex(ValueError, "final_snapshot"):
                run_training_job.resolve_completed_checkpoint(self.plan)

    def test_missing_or_wrong_final_snapshot_blocks_gate(self):
        checkpoint = self.record_training_completion()
        with patch.object(run_training_job, "validate_checkpoint") as validate:
            checkpoint.unlink()
            with self.assertRaisesRegex(ValueError, "existing file"):
                run_training_job.resolve_completed_checkpoint(self.plan)
            validate.assert_not_called()
        self.record_training_completion()
        with patch.object(run_training_job, "validate_checkpoint", return_value={"config": {"model_revision": "wrong"}}):
            with self.assertRaisesRegex((ValueError, KeyError), "source/revision|protocol"):
                run_training_job.resolve_completed_checkpoint(self.plan)

    def test_busy_gpu_or_port_blocks_training_without_retry(self):
        self.prepare()
        for safety_check in ("check_gpu_utilization", "check_port_free"):
            with self.subTest(safety_check=safety_check), \
                    patch.object(run_training_job, "check_gpu_utilization"), \
                    patch.object(run_training_job, "check_port_free"), \
                    patch.object(run_training_job, safety_check, side_effect=RuntimeError("occupied")), \
                    patch.object(run_training_job, "stop_owned_process"), \
                    patch.object(run_training_job.subprocess, "Popen") as launch:
                self.assertEqual(run_training_job.run_job(self.plan), 1)
                launch.assert_not_called()

    def test_dry_run_has_no_subprocesses_or_output_directories(self):
        with patch.object(run_training_job.subprocess, "Popen") as launch, \
                patch.object(run_training_job, "check_gpu_utilization") as gpu_check, \
                redirect_stdout(io.StringIO()) as output:
            self.assertEqual(run_training_job.main(["--plan", str(self.plan_path), "--dry-run", "--background"]), 0)
        launch.assert_not_called()
        gpu_check.assert_not_called()
        printed = output.getvalue()
        self.assertIn('"total_requested_episodes": 500', printed)
        self.assertIn('"max_seed_attempts_per_task": 2500', printed)
        self.assertIn('"gradient_accumulation": 7', printed)
        for output_dir in (self.job_root, self.run_dir, self.eval_root):
            self.assertFalse(output_dir.exists())

    def test_background_launcher_records_owned_pid_without_launching_training_itself(self):
        with patch.object(run_training_job.subprocess, "Popen", return_value=SimpleNamespace(pid=4321)) as launch, \
                patch.object(run_training_job, "run_job") as worker, redirect_stdout(io.StringIO()):
            self.assertEqual(run_training_job.main(["--plan", str(self.plan_path), "--background"]), 0)
        worker.assert_not_called()
        command = launch.call_args.args[0]
        self.assertIn("--worker", command)
        self.assertEqual(command[command.index("--plan") + 1], str(self.job_root / "plan.json"))
        self.assertTrue(launch.call_args.kwargs["start_new_session"])
        self.assertEqual(eval_multitask.read_json(self.job_root / "process.json")["pid"], 4321)
        self.assertEqual((self.job_root / "plan.json").read_bytes(), self.plan_path.read_bytes())
        self.assertFalse(self.run_dir.exists())

    def test_existing_plan_only_job_root_is_accepted_without_refreezing(self):
        self.job_root.mkdir()
        frozen_plan = self.job_root / "plan.json"
        frozen_plan.write_bytes(self.plan_path.read_bytes())
        original_bytes = frozen_plan.read_bytes()
        with redirect_stdout(io.StringIO()):
            self.assertEqual(run_training_job.main(["--plan", str(frozen_plan), "--dry-run"]), 0)
        self.assertEqual(set(self.job_root.iterdir()), {frozen_plan})
        run_training_job.prepare_job(self.plan, frozen_plan, None)
        self.assertEqual(frozen_plan.read_bytes(), original_bytes)
        self.assertTrue((self.job_root / "status.json").is_file())

    def test_refuses_existing_outputs_and_unknown_plan_changes(self):
        self.eval_root.mkdir()
        with self.assertRaises(FileExistsError):
            run_training_job.prepare_job(self.plan, self.plan_path, None)
        with patch.object(eval_multitask, "validate_checkpoint") as validate:
            with self.assertRaises(FileExistsError):
                eval_multitask.run_suite(self.config, self.root / "checkpoint.pt", self.eval_root)
        validate.assert_not_called()
        self.assertFalse(self.job_root.exists())

    def test_explicit_resume_preserves_frozen_plan_and_adds_only_resume(self):
        self.prepare()
        checkpoint = self.record_training_completion(step=1000, status="stopped")
        original_bytes = (self.job_root / "plan.json").read_bytes()
        run_training_job.prepare_job(self.plan, self.plan_path, checkpoint)
        with patch.object(run_training_job, "check_gpu_utilization"), \
                patch.object(run_training_job, "check_port_free"), \
                patch.object(run_training_job, "stop_owned_process"), \
                patch.object(run_training_job.subprocess, "Popen",
                             return_value=SimpleNamespace(pid=111, wait=lambda: 1)) as launch:
            self.assertEqual(run_training_job.run_job(self.plan, checkpoint), 1)
        self.assertEqual(launch.call_args.args[0], list(self.plan.training_command) + ["--resume", str(checkpoint)])
        self.assertEqual((self.job_root / "plan.json").read_bytes(), original_bytes)
        payload = json.loads(self.plan_path.read_text(encoding="utf-8"))
        payload["environment"]["CHANGED"] = "yes"
        self.plan_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "refreeze"):
            run_training_job.validate_resume(run_training_job.TrainingJobPlan.load(self.plan_path), checkpoint)

    def stage1_plan(self, steps=7, random_frozen=False):
        import configure_training_job

        source = self.root / "original-memoryvla"
        (source / "checkpoints").mkdir(parents=True)
        (source / "config.json").write_text(json.dumps({"model": {"model_id": "original-prismatic"}}))
        (source / "checkpoints/latest-checkpoint.pt").write_bytes(b"CPU metadata fixture, never loaded")
        extra = []
        if random_frozen:
            metadata = self.root / "llama-metadata"
            metadata.mkdir()
            for name, payload in {"config.json": {"model_type": "llama"}, "tokenizer_config.json": {}, "tokenizer.json": {}}.items():
                (metadata / name).write_text(json.dumps(payload))
            source = self.root / "metadata-manifest.json"
            source.write_text(json.dumps({"llama_metadata": {"path": str(metadata), "files": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in metadata.iterdir()
            }}}))
            extra = ["--backbone-initialization", "random-frozen-debug", "--backbone-initialization-seed", "123", "--freeze-backbone"]
        arguments = configure_training_job.build_parser().parse_args([
            "--run-name", "new-stage1", "--model-path", str(source), "--max-steps", str(steps),
            "--data-root", str(self.root / "native"), "--manifest", str(self.root / "manifest.json"),
            "--gpu-ids", "2", "6", "--tasks", "press_button",
            "--run-dir", str(self.run_dir), "--job-root", str(self.job_root), "--eval-root", str(self.eval_root), *extra,
        ])
        payload = configure_training_job.build_stage1_plan(arguments)
        self.plan_path.write_text(json.dumps(payload))
        self.plan = run_training_job.TrainingJobPlan.load(self.plan_path)
        self.contract = self.make_contract(self.plan.training_command)
        if random_frozen:
            self.contract["protocol"]["optimization"].update(master_dtype="float32", frozen_backbone_dtype="bfloat16", autocast_dtype="bfloat16")
        return self.plan

    def test_stage1_training_only_completes_once_without_launching_evaluation(self):
        self.stage1_plan()
        self.prepare()

        def finish():
            self.record_training_completion(step=7)
            return 0

        with patch.object(run_training_job, "check_gpu_utilization"), \
                patch.object(run_training_job, "check_port_free"), \
                patch.object(run_training_job, "stop_owned_process"), \
                patch.object(run_training_job, "validate_checkpoint", return_value=self.contract) as gate, \
                patch.object(run_training_job.subprocess, "Popen", return_value=SimpleNamespace(pid=123, wait=finish)) as launch:
            self.assertEqual(run_training_job.run_job(self.plan), 0)
        launch.assert_called_once()
        gate.assert_called_once()
        self.assertEqual(launch.call_args.kwargs["cwd"], run_training_job.PROJECT_ROOT)
        status = eval_multitask.read_json(self.job_root / "status.json")
        self.assertEqual(status["state"], "completed")
        self.assertEqual(status["requested_evaluation_episodes"], 0)
        self.assertFalse(status["evaluation_started"])
        self.assertFalse(self.eval_root.exists())

    def test_debug_plan_uses_explicit_training_only_checkpoint_gate_and_same_budget(self):
        plan = self.stage1_plan(steps=3, random_frozen=True)
        checkpoint = self.record_training_completion(step=3)
        with patch.object(run_training_job, "validate_checkpoint", return_value=self.contract) as gate:
            self.assertEqual(run_training_job.resolve_completed_checkpoint(plan), checkpoint)
            gate.assert_called_once_with(checkpoint, expected_step=3, expected_action_head="diffusion",
                                         expected_cameras=("head_camera",), allow_debug_checkpoint=True)
            self.contract["protocol"]["optimization"]["frozen_backbone_dtype"] = "float32"
            with self.assertRaisesRegex(ValueError, "optimization/batch"):
                run_training_job.resolve_completed_checkpoint(plan)
        self.record_training_completion(step=1, status="stopped")
        with patch.object(run_training_job, "validate_checkpoint") as gate:
            with self.assertRaisesRegex(ValueError, "exactly 3"):
                run_training_job.resolve_completed_checkpoint(plan)
            gate.assert_not_called()
        with self.assertRaisesRegex(ValueError, "training-only"):
            replace(plan, evaluation_config=self.config).validate()

    def test_pretrained_completion_legacy_new_field_defaults_remain_strict(self):
        plan = self.stage1_plan()
        checkpoint = self.record_training_completion(step=7)
        for field in train_multitask.INITIALIZATION_DEFAULTS:
            self.contract["config"].pop(field)
        with patch.object(run_training_job, "validate_checkpoint", return_value=self.contract) as gate:
            self.assertEqual(run_training_job.resolve_completed_checkpoint(plan), checkpoint)
            gate.assert_called_once_with(checkpoint, expected_step=7, expected_action_head="diffusion", expected_cameras=("head_camera",))
            self.contract["config"]["backbone_initialization"] = "random-frozen-debug"
            with self.assertRaisesRegex(ValueError, "configuration"):
                run_training_job.resolve_completed_checkpoint(plan)

    def test_stage1_final_checkpoint_must_match_frozen_method_manifest_and_optimizer(self):
        plan = self.stage1_plan()
        self.record_training_completion(step=7)
        with patch.object(run_training_job, "validate_checkpoint", return_value=self.contract):
            run_training_job.resolve_completed_checkpoint(plan)
            self.contract["config"]["mask_padding_loss"] = True
            with self.assertRaisesRegex(ValueError, "configuration"):
                run_training_job.resolve_completed_checkpoint(plan)
            self.contract["config"]["mask_padding_loss"] = False
            self.contract["protocol"]["optimization"]["global_batch_size"] = 448
            with self.assertRaisesRegex(ValueError, "optimization/batch"):
                run_training_job.resolve_completed_checkpoint(plan)

    def test_old_training_code_and_outputs_are_rejected_without_rewriting_plan(self):
        plan = self.stage1_plan()
        payload = json.loads(self.plan_path.read_text())
        command = payload["training_command"]
        script_index = command.index(str(run_training_job.PROJECT_ROOT / "train_multitask.py"))
        command[script_index] = "/mnt/data/dylan/d2/memory/TTTchoiceMem/train_multitask.py"
        self.plan_path.write_text(json.dumps(payload))
        original = self.plan_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "another checkout"):
            run_training_job.TrainingJobPlan.load(self.plan_path)
        self.assertEqual(self.plan_path.read_bytes(), original)
        command[script_index] = str(run_training_job.PROJECT_ROOT / "train_multitask.py")
        command[command.index("--run-dir") + 1] = "/mnt/data/dylan/d2/memory/TTTchoiceMem/runs/forbidden-stage1"
        payload["run_dir"] = command[command.index("--run-dir") + 1]
        self.plan_path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "another TTTchoiceMem|original memory"):
            run_training_job.TrainingJobPlan.load(self.plan_path)
        for path in (plan.job_root, plan.run_dir, plan.eval_root):
            self.assertFalse(path.exists())

    def test_stage1_job_dry_run_has_no_torch_subprocess_or_outputs(self):
        self.stage1_plan()
        with patch.object(run_training_job.subprocess, "Popen") as launch, \
                patch.object(run_training_job, "check_gpu_utilization") as gpu_check, redirect_stdout(io.StringIO()) as output:
            self.assertEqual(run_training_job.main(["--plan", str(self.plan_path), "--dry-run"]), 0)
        launch.assert_not_called()
        gpu_check.assert_not_called()
        self.assertIn('"evaluation": null', output.getvalue())
        for path in (self.job_root, self.run_dir, self.eval_root):
            self.assertFalse(path.exists())

    # --------------------------------------------------------------- summaries

    def write_native_journal(self, output_dir, episodes=50, skipped_seeds=(), attempts=None, complete=True):
        output_dir.mkdir(parents=True)
        if attempts is None:
            attempts = episodes + len(skipped_seeds)
        events = [{"event": "run_started", "requested_episodes": 50, "seed_start": 100000}]
        completed = 0
        successes = 0
        for offset in range(attempts):
            seed = 100000 + offset
            events.append({"event": "seed_attempt", "seed": seed})
            if seed in skipped_seeds:
                events.append({"event": "seed_skipped", "seed": seed, "reason": "expert_failed"})
                continue
            if completed >= episodes:
                break
            success = completed % 2 == 0
            successes += int(success)
            events.append({"event": "episode_done", "episode_id": completed, "seed": seed,
                           "success": success, "rollout_steps": 4})
            completed += 1
        if complete:
            events.append({"event": "run_completed", "completed_episodes": completed, "successes": successes})
        (output_dir / "episodes.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8",
        )

    def test_expert_skips_are_logged_and_never_replaced_or_counted(self):
        complete_dir = self.root / "complete"
        # Two expert-invalid seeds are consumed from the pool before 50 episodes.
        self.write_native_journal(complete_dir, episodes=50, skipped_seeds={100000, 100005})
        complete = eval_multitask.summarize_task(TASK_NAMES[0], complete_dir, 0)
        self.assertTrue(complete["coverage_complete"])
        self.assertEqual(complete["completed_episodes"], 50)
        self.assertEqual(complete["seed_attempt_count"], 52)
        self.assertEqual(complete["skipped_seed_count"], 2)
        self.assertEqual(complete["requested_seed_attempts"], 2500)
        self.assertEqual(complete["native_official_success_rate"], 0.5)
        self.assertEqual([event["seed"] for event in complete["skipped_seeds"]], [100000, 100005])

    def test_exhausted_pool_reports_incomplete_coverage_without_retry(self):
        exhausted_dir = self.root / "exhausted"
        # 2460 expert-invalid seeds leave only 40 policy episodes in the pool.
        self.write_native_journal(
            exhausted_dir, episodes=50, skipped_seeds=set(range(100000, 102460)), attempts=2500, complete=False,
        )
        exhausted = eval_multitask.summarize_task(TASK_NAMES[0], exhausted_dir, 0)
        self.assertFalse(exhausted["coverage_complete"])
        self.assertEqual(exhausted["completed_episodes"], 40)
        self.assertEqual(exhausted["skipped_seed_count"], 2460)
        self.assertIsNone(exhausted["native_official_success_rate"])
        self.assertTrue(any("Incomplete coverage" in problem for problem in exhausted["problems"]))

    def test_seed_attempts_outside_the_fixed_pool_are_flagged(self):
        outside_dir = self.root / "outside"
        self.write_native_journal(outside_dir, episodes=50)
        journal = outside_dir / "episodes.jsonl"
        events = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines() if line.strip()]
        events[1]["seed"] = 99999
        journal.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
        flagged = eval_multitask.summarize_task(TASK_NAMES[0], outside_dir, 0)
        self.assertFalse(flagged["coverage_complete"])
        self.assertTrue(any("2500-candidate pool" in problem for problem in flagged["problems"]))

    def test_native_child_requests_fifty_policy_episodes_over_2500_expert_seeds(self):
        command = eval_multitask.build_native_command(
            self.config, TASK_NAMES[0], 4, 18824, self.eval_root / TASK_NAMES[0],
        )
        arguments = eval_multitask.parse_args(command[2:])
        native_evaluator = SimpleNamespace(main=Mock())
        loader = Mock(return_value=native_evaluator)
        numpy_fixture = SimpleNamespace(random=SimpleNamespace(seed=Mock()))
        with patch.dict(sys.modules, {
            "eval_rmbench": SimpleNamespace(load_native_evaluator=loader),
            "test_render": SimpleNamespace(Sapien_TEST=Mock()), "numpy": numpy_fixture,
        }), patch.dict(eval_multitask.os.environ, {}, clear=True):
            eval_multitask.run_native_child(arguments)
            self.assertEqual(eval_multitask.os.environ["CUDA_VISIBLE_DEVICES"], "4")
            self.assertEqual(eval_multitask.os.environ["RMBENCH_RENDER_DEVICE"], "cuda:0")
        loader.assert_called_once_with(self.rmbench_root)
        native_arguments = native_evaluator.main.call_args.args[0]
        self.assertEqual(native_arguments["test_num"], 50)
        self.assertEqual(native_arguments["max_seed_attempts"], 2500)
        self.assertEqual(native_arguments["seed"], 0)
        self.assertEqual(native_arguments["seed_stride"], 1)
        self.assertTrue(native_arguments["expert_check"])
        self.assertFalse(native_arguments["eval_batch"])
        (arguments.output_dir / "episodes.jsonl").touch()
        with self.assertRaises(FileExistsError):
            eval_multitask.run_native_child(arguments)

    def test_workers_never_overlap_on_a_gpu_and_aggregate_all_ten_tasks(self):
        config = replace(self.config, gpu_ids=(2, 6), ports=(18822, 18826))
        active_gpus = set()
        completed_tasks = []
        assignment_lock = threading.Lock()

        def native_task(config, checkpoint, task_name, gpu_id, port, eval_root, update_status, cancellation):
            with assignment_lock:
                self.assertNotIn(gpu_id, active_gpus)
                active_gpus.add(gpu_id)
            time.sleep(0.005)
            with assignment_lock:
                active_gpus.remove(gpu_id)
                completed_tasks.append(task_name)
            return {"task_name": task_name, "coverage_complete": True, "completed_episodes": 50,
                    "native_official_success_rate": 0.5}

        with patch.object(eval_multitask, "validate_checkpoint", return_value={"config": {"horizon": 16, "execute_prefix": 4}}), \
                patch.object(eval_multitask, "check_gpu_utilization"), \
                patch.object(eval_multitask, "check_port_free"), \
                patch.object(eval_multitask, "run_task", side_effect=native_task):
            summary = eval_multitask.run_suite(config, self.root / "model.pt", self.eval_root)
        self.assertEqual(sorted(completed_tasks), sorted(TASK_NAMES))
        self.assertEqual(summary["completed_episodes"], 500)
        self.assertEqual(summary["native_official_equal_task_success_rate"], 0.5)
        suite_plan = eval_multitask.read_json(self.eval_root / "plan.json")
        self.assertEqual(suite_plan["test_num"], 50)
        self.assertEqual(suite_plan["max_seed_attempts_per_task"], 2500)
        self.assertEqual(suite_plan["evaluation_config"]["gpu_ids"], [2, 6])

    def test_gpu_gate_rejects_fifty_percent_and_port_gate_preserves_owner(self):
        with patch.object(eval_multitask.subprocess, "run", return_value=SimpleNamespace(stdout="2, 49\n6, 50\n")):
            eval_multitask.check_gpu_utilization((2,))
            with self.assertRaisesRegex(RuntimeError, "GPU 6"):
                eval_multitask.check_gpu_utilization((6,))
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as owner:
            owner.bind(("127.0.0.1", 0))
            owner.listen()
            with self.assertRaises(OSError):
                eval_multitask.check_port_free("127.0.0.1", owner.getsockname()[1])
            self.assertGreater(owner.fileno(), 0)

    def test_server_readiness_failure_cleans_only_owned_server_and_never_launches_native(self):
        self.eval_root.mkdir()
        server = SimpleNamespace(pid=789)
        update_status = Mock()
        with patch.object(eval_multitask, "check_gpu_utilization") as gpu_check, \
                patch.object(eval_multitask, "check_port_free"), \
                patch.object(eval_multitask, "wait_for_server", side_effect=TimeoutError("not ready")), \
                patch.object(eval_multitask, "stop_owned_process") as cleanup, \
                patch.object(eval_multitask.subprocess, "Popen", return_value=server) as launch:
            result = eval_multitask.run_task(self.config, self.root / "final.pt", TASK_NAMES[0], 2, 18822,
                                            self.eval_root, update_status, threading.Event())
        launch.assert_called_once()
        gpu_check.assert_called_once_with((2,))
        self.assertEqual(launch.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "2")
        self.assertEqual(launch.call_args.kwargs["env"]["RMBENCH_RENDER_DEVICE"], "cuda:0")
        cleanup.assert_any_call(server)
        self.assertFalse(result["coverage_complete"])
        self.assertIn("TimeoutError", " ".join(result["problems"]))

    def test_checkpoint_gate_rejects_wrong_step_and_partial_model(self):
        import torch

        checkpoint = self.root / "fixture.pt"
        checkpoint.write_bytes(b"CPU metadata fixture")
        payload = {
            "schema": "choice-memory-rmbench-v1", "step": 30000, "config": {"backbone": "qwen35", "action_head": "diffusion", "freeze_backbone": False,
                                      "dtype": "bfloat16", "horizon": 16, "execute_prefix": 4, "group_size": 16,
                                      "cameras": ["head_camera", "left_camera", "right_camera"]},
            "protocol": {"cameras": ["head_camera", "left_camera", "right_camera"]},
            "normalization": {}, "model": {component + "weight": torch.ones(1) for component in
                ("backbone.", "action_head.", "per_compr.", "cog_mem_bank.", "per_mem_bank.")},
        }
        reader = Mock(return_value=payload)
        with patch.dict(sys.modules, {"choice_mem.checkpoint": SimpleNamespace(CHECKPOINT_SCHEMA="choice-memory-rmbench-v2", LEGACY_CHECKPOINT_SCHEMA="choice-memory-rmbench-v1", read_checkpoint=reader)}):
            self.assertEqual(eval_multitask.validate_checkpoint(checkpoint)["step"], 30000)
            payload["step"] = 29999
            with self.assertRaisesRegex(ValueError, "step"):
                eval_multitask.validate_checkpoint(checkpoint)
            payload["step"] = 30000
            payload["model"].pop("backbone.weight")
            with self.assertRaisesRegex(ValueError, "full model component"):
                eval_multitask.validate_checkpoint(checkpoint)

    def test_final_oft_checkpoint_rejects_diffusion_wrong_batch_and_nonfinite_tensors(self):
        import torch
        from choice_mem.config import PolicyConfig

        checkpoint = self.root / "oft-fixture.pt"
        checkpoint.write_bytes(b"CPU metadata fixture")
        configuration = PolicyConfig(
            backbone="qwen35", model_source=str(self.root / "model"), action_head="oft", freeze_backbone=False, group_size=16,
            model_revision="2fc06364715b967f1860aea9cf38778875588b17",
        ).to_dict()
        optimization = {
            "max_steps": 55000, "world_size": 8, "per_device_batch_size": 16,
            "global_batch_size": 128, "grad_accumulation": 1, "groups_per_forward": 1,
            "master_dtype": "float32", "autocast_dtype": "bfloat16", "gradient_checkpointing": True,
        }
        payload = {
            "schema": "choice-memory-rmbench-v1", "step": 55000, "config": configuration,
            "protocol": {"optimization": optimization, "cameras": configuration["cameras"]},
            "normalization": {}, "model": {component + "weight": torch.ones(1) for component in
                ("backbone.", "action_head.", "per_compr.", "cog_mem_bank.", "per_mem_bank.")},
        }
        with patch.dict(sys.modules, {"choice_mem.checkpoint": SimpleNamespace(CHECKPOINT_SCHEMA="choice-memory-rmbench-v2", LEGACY_CHECKPOINT_SCHEMA="choice-memory-rmbench-v1", read_checkpoint=lambda path: payload)}):
            self.assertEqual(eval_multitask.validate_checkpoint(checkpoint, 55000, "oft")["step"], 55000)
            with self.assertRaisesRegex(ValueError, "action_head"):
                eval_multitask.validate_checkpoint(checkpoint, 55000)
            optimization["global_batch_size"] = 64
            with self.assertRaisesRegex(ValueError, "world/global batch"):
                eval_multitask.validate_checkpoint(checkpoint, 55000, "oft")
            optimization["global_batch_size"] = 128
            payload["model"]["action_head.weight"] = torch.tensor([float("nan")])
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                eval_multitask.validate_checkpoint(checkpoint, 55000, "oft")

    def test_virtualenv_python_symlink_is_not_resolved_to_base_interpreter(self):
        virtualenv_python = self.root / "venv/bin/python"
        virtualenv_python.parent.mkdir(parents=True)
        virtualenv_python.symlink_to(sys.executable)
        configuration = self.config.to_dict()
        configuration["qwen_python"] = str(virtualenv_python)
        parsed = eval_multitask.EvaluationConfig.from_dict(configuration)
        self.assertEqual(parsed.qwen_python, virtualenv_python)
        self.assertNotEqual(parsed.qwen_python, virtualenv_python.resolve())


if __name__ == "__main__":
    unittest.main()
