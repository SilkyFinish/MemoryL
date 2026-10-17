"""CPU fake-policy checks, including the unchanged native RMBench rollout loop.

Pass --network-smoke to also use the real XPolicyLab websocket server/client.
Neither mode loads a trained model, starts a simulator, or uses a GPU.
"""

from __future__ import annotations

import ast
import asyncio
from collections.abc import Mapping, Sequence
from contextlib import redirect_stdout
from copy import deepcopy
import io
import os
from pathlib import Path
import subprocess
import sys
import traceback
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from choice_mem.config import PolicyConfig
from choice_mem.runtime import DEFAULT_RMBENCH_ROOT, ModelWrapper, add_native_paths
from eval_rmbench import build_native_command, install_timestep_adapter, parse_args


NETWORK_SMOKE = "--network-smoke" in sys.argv
if NETWORK_SMOKE:
    sys.argv.remove("--network-smoke")


class FakePolicy:
    """Record query inputs and both memory writes without a backbone or simulator."""

    def __init__(self):
        self.config = PolicyConfig(dtype="float32")
        self.training = True
        self.per_mem_bank = types.SimpleNamespace(bank={})
        self.cog_mem_bank = types.SimpleNamespace(bank={})
        self.reset_calls = []
        self.predict_calls = []
        self.prediction = torch.zeros(1, self.config.horizon, 14)
        self.prediction[:, :, 6] = -0.3
        self.prediction[:, :, 13] = 0.3

    def eval(self):
        self.training = False
        return self

    def reset(self, episode_id=None):
        self.reset_calls.append(episode_id)
        self.per_mem_bank.bank.clear()
        self.cog_mem_bank.bank.clear()

    def predict(self, images, instruction, episode_id, timestep=None):
        if self.training or torch.is_grad_enabled():
            raise AssertionError("Runtime did not put inference in eval/no-grad mode")
        self.predict_calls.append(
            {
                "images": [image.copy() for image in images],
                "instruction": instruction,
                "episode_id": episode_id,
                "timestep": timestep,
            }
        )
        for memory_bank in (self.per_mem_bank, self.cog_mem_bank):
            memory_bank.bank.setdefault(episode_id, []).append(timestep)
        return self.prediction


# Audited checkpoint contract: data.py now uses target_offset=1 ("next captured
# absolute drive target"), keeps both grippers continuous, and masks nothing.
AUDITED_LABEL_ALIGNMENT = (
    "observation[t] -> joint_action/vector[t+1:t+1+H]; each label is the next recorded absolute "
    "drive target, not measured qpos and not a certified next-issued command"
)


def make_payload(policy, *, direct_statistics=False, target_offset=1, gripper_binarization=False):
    lower_quantiles = np.full(14, -2.0, dtype=np.float32)
    upper_quantiles = np.full(14, 2.0, dtype=np.float32)
    lower_quantiles[[6, 13]] = 0.0
    upper_quantiles[[6, 13]] = 1.0
    statistics = {
        "q01": lower_quantiles.tolist(),
        "q99": upper_quantiles.tolist(),
        # Audited normalization keeps all fourteen channels; grippers use the same transform.
        "mask": [True] * 14,
    }
    normalization = statistics if direct_statistics else {"action": statistics, "source": "train episodes only"}
    return {
        "config": policy.config.to_dict(),
        "normalization": normalization,
        "protocol": {
            "action_type": "joint",
            "action_dim": 14,
            "horizon": 16,
            "execute_prefix": 4,
            "cameras": policy.config.cameras,
            "image_size": policy.config.image_size,
            "target_offset": target_offset,
            "label_alignment": AUDITED_LABEL_ALIGNMENT,
            "gripper_binarization": gripper_binarization,
        },
    }


def make_observation(timestep, instruction="put the battery in the box"):
    return {
        "timestep": timestep,
        "instruction": instruction,
        "env_idx": 0,
        "vision": {
            camera_name: {"color": np.full((3, 5, 3), [timestep, camera_index + 40, 201], dtype=np.uint8)}
            for camera_index, camera_name in enumerate(("cam_head", "cam_left_wrist", "cam_right_wrist"))
        },
        "state": {"must_not_enter_model": np.arange(14)},
        "success": True,
        "episode_label": "must_not_enter_model",
    }


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.policy = FakePolicy()
        self.payload = make_payload(self.policy)
        self.wrapper = ModelWrapper(policy=self.policy, payload=self.payload)

    def test_selected_cameras_need_no_unselected_pixels_and_keep_checkpoint_order(self):
        native_names = {"head_camera": "cam_head", "left_camera": "cam_left_wrist", "right_camera": "cam_right_wrist"}
        for selected_cameras in (["head_camera"], ["right_camera", "head_camera"], list(native_names)):
            with self.subTest(cameras=selected_cameras):
                policy = FakePolicy()
                policy.config.cameras = selected_cameras
                wrapper = ModelWrapper(policy=policy, payload=make_payload(policy))
                observation = make_observation(0)
                observation["vision"] = {
                    native_names[camera]: observation["vision"][native_names[camera]] for camera in selected_cameras
                }
                wrapper.update_obs(observation)
                wrapper.get_action()
                images = policy.predict_calls[-1]["images"]
                self.assertEqual(len(images), len(selected_cameras))
                self.assertEqual([int(image[0, 0, 1]) for image in images], [
                    40 + list(native_names).index(camera) for camera in selected_cameras
                ])
                observation["vision"].pop(native_names[selected_cameras[0]])
                with self.assertRaisesRegex(ValueError, "Missing decoded RGB camera"):
                    wrapper.update_obs(observation)

    def test_latest_rgb_observation_is_owned_and_only_get_action_writes_memory(self):
        for timestep in (0, 1, 3):
            self.wrapper.update_obs(make_observation(timestep))
        latest_observation = make_observation(7, instruction="latest instruction")
        self.wrapper.update_obs(latest_observation)
        latest_observation["vision"]["cam_head"]["color"][:] = 0
        self.assertEqual(self.policy.predict_calls, [])
        self.assertEqual(self.policy.per_mem_bank.bank, {})
        self.assertEqual(self.policy.cog_mem_bank.bank, {})
        actions = self.wrapper.get_action()
        self.assertEqual(actions.shape, (4, 14))
        self.assertEqual(len(self.policy.predict_calls), 1)
        recorded = self.policy.predict_calls[0]
        self.assertEqual(recorded["timestep"], 7)
        self.assertEqual(recorded["instruction"], "latest instruction")
        for camera_index, image in enumerate(recorded["images"]):
            np.testing.assert_array_equal(image[0, 0], [7, camera_index + 40, 201])
        for memory_bank in (self.policy.per_mem_bank, self.policy.cog_mem_bank):
            self.assertEqual(memory_bank.bank, {self.wrapper.episode_id: [7]})
        with self.assertRaisesRegex(RuntimeError, "update_obs"):
            self.wrapper.get_action()
        self.assertEqual(len(self.policy.predict_calls), 1)

    def test_reset_clears_both_banks_pending_pixels_and_old_timestep(self):
        self.wrapper.update_obs(make_observation(12))
        self.wrapper.get_action()
        previous_episode_id = self.wrapper.episode_id
        self.wrapper.update_obs(make_observation(15))
        self.wrapper.reset()
        self.assertNotEqual(self.wrapper.episode_id, previous_episode_id)
        self.assertEqual(self.policy.reset_calls[-1], self.wrapper.episode_id)
        self.assertEqual(self.policy.per_mem_bank.bank, {})
        self.assertEqual(self.policy.cog_mem_bank.bank, {})
        with self.assertRaisesRegex(RuntimeError, "update_obs"):
            self.wrapper.get_action()
        self.wrapper.update_obs(make_observation(0, instruction="new episode"))
        self.wrapper.get_action()
        self.assertEqual(self.policy.predict_calls[-1]["timestep"], 0)
        self.assertEqual(self.policy.per_mem_bank.bank, {self.wrapper.episode_id: [0]})

    def test_train_quantile_inverse_clipping_mask_and_both_continuous_grippers(self):
        for direct_statistics in (False, True):
            with self.subTest(direct_statistics=direct_statistics):
                payload = make_payload(self.policy, direct_statistics=direct_statistics)
                statistics = payload["normalization"] if direct_statistics else payload["normalization"]["action"]
                statistics["q01"][10] = 0.42
                statistics["q99"][10] = 0.42
                statistics["mask"][5] = False  # Synthetic pass-through channel; the audited mask is all True.
                wrapper = ModelWrapper(policy=self.policy, payload=payload)
                self.policy.prediction[0, 0, :3] = torch.tensor([-3.0, 0.0, 3.0])
                self.policy.prediction[0, 0, 5] = 0.25
                self.policy.prediction[0, 1, 6] = 0.0
                self.policy.prediction[0, 1, 13] = -0.2
                # Nonexecuted slots must not leak into the returned chunk.
                self.policy.prediction[0, 4:, :] = 0.99
                wrapper.update_obs(make_observation(0))
                actions = wrapper.get_action()
                np.testing.assert_allclose(actions[0, :3], [-2.0, 0.0, 2.0])
                np.testing.assert_allclose(actions[0, [5, 6, 10, 13]], [0.25, 0.35, 0.42, 0.65], atol=1e-7)
                np.testing.assert_allclose(actions[1, [6, 13]], [0.5, 0.4], atol=1e-7)
                self.assertEqual(actions.dtype, np.float32)
                self.assertEqual(len(actions), 4)

    def test_checkpoint_loader_is_used_with_assertions_not_architecture_overrides(self):
        with patch("choice_mem.checkpoint.load_checkpoint", return_value=(self.policy, self.payload)) as loader:
            wrapper = ModelWrapper("trained.pt", "cpu", expected={"execute_prefix": 4})
        loader.assert_called_once_with("trained.pt", "cpu", expected={"execute_prefix": 4})
        self.assertIs(wrapper.policy, self.policy)
        self.assertEqual(wrapper.protocol["target_offset"], 1)
        self.assertEqual(wrapper.normalization["source"], "train episodes only")
        for expected in ({"horizon": 8}, {"execute_prefix": 8}, {"cameras": ["left_camera"]}):
            with self.subTest(expected=expected), self.assertRaisesRegex(ValueError, "configuration mismatch"):
                ModelWrapper(policy=self.policy, payload=self.payload, expected=expected)
        with self.assertRaisesRegex(ValueError, "action_type=joint"):
            ModelWrapper(policy=self.policy, payload=self.payload, action_type="ee")
        with self.assertRaisesRegex(ValueError, "protocol mismatch"):
            ModelWrapper(
                policy=self.policy, payload=self.payload, expected_protocol={"label_alignment": "next_state_qpos"}
            )
        with self.assertRaisesRegex(ValueError, "protocol mismatch"):
            ModelWrapper(policy=self.policy, payload=self.payload, expected_protocol={"target_offset": 0})
        invalid_payload = deepcopy(self.payload)
        invalid_payload["protocol"]["horizon"] = 8
        with self.assertRaisesRegex(ValueError, "protocol/configuration mismatch"):
            ModelWrapper(policy=self.policy, payload=invalid_payload)
        invalid_payload = deepcopy(self.payload)
        invalid_payload["protocol"]["channel_order"] = ["right_arm_joint_0"] * 14
        with self.assertRaisesRegex(ValueError, "action channel order"):
            ModelWrapper(policy=self.policy, payload=invalid_payload)
        invalid_payload = deepcopy(self.payload)
        invalid_payload["protocol"]["gripper_indices"] = [12, 13]
        with self.assertRaisesRegex(ValueError, "gripper indices"):
            ModelWrapper(policy=self.policy, payload=invalid_payload)

    def test_debug_runtime_requires_and_forwards_explicit_opt_in(self):
        policy = FakePolicy()
        policy.config = PolicyConfig(backbone_initialization="random-frozen-debug", freeze_backbone=True)
        payload = make_payload(policy)
        for field in ("backbone_initialization", "backbone_initialization_seed", "base_vlm_id"):
            payload["protocol"][field] = getattr(policy.config, field)
        with self.assertRaisesRegex(ValueError, "explicit allow_debug_checkpoint"):
            ModelWrapper(policy=policy, payload=payload)
        with self.assertRaisesRegex(ValueError, "explicit allow_debug_checkpoint"):
            ModelWrapper(policy=policy, payload=payload, allow_debug_checkpoint="true")
        with patch("choice_mem.checkpoint.load_checkpoint", return_value=(policy, payload)) as loader:
            wrapper = ModelWrapper("debug.pt", "cpu", allow_debug_checkpoint=True)
        loader.assert_called_once_with("debug.pt", "cpu", expected=None, allow_debug_checkpoint=True)
        self.assertEqual(wrapper.protocol_summary()["backbone_initialization"], "random-frozen-debug")
        wrapper.update_obs(make_observation(0))
        actions = wrapper.get_action()
        self.assertEqual(actions.shape, (4, 14))
        np.testing.assert_allclose(actions[0, [6, 13]], [0.35, 0.65], atol=1e-7)

    def test_server_explicit_debug_flag_is_forwarded_without_a_completion_gate(self):
        import serve_rmbench as serving

        self.assertFalse(serving.parse_args(["--checkpoint", "debug.pt"]).allow_debug_checkpoint)
        args = ["--checkpoint", "debug.pt", "--allow-debug-checkpoint"]
        self.assertTrue(serving.parse_args(args).allow_debug_checkpoint)
        server = Mock()
        native = types.SimpleNamespace(PolicyServer=Mock(return_value=server), PolicyServerConfig=Mock())
        model = Mock(protocol_summary=Mock(return_value=self.wrapper.protocol_summary()))
        with patch.object(serving, "ModelWrapper", return_value=model) as wrapper, \
                patch.object(serving, "add_native_paths") as paths, \
                patch.object(serving.asyncio, "run") as run, \
                patch.dict(sys.modules, {"client_server.ws.model_server": native}), \
                patch("eval_multitask.validate_checkpoint", side_effect=AssertionError("Not the completion gate")), \
                patch.object(torch.cuda, "set_device", side_effect=AssertionError("CPU test only")), \
                redirect_stdout(io.StringIO()):
            serving.main(args)
        self.assertTrue(wrapper.call_args.kwargs["allow_debug_checkpoint"])
        self.assertEqual(wrapper.call_args.args, (Path("debug.pt"), "cpu"))
        native.PolicyServerConfig.assert_called_once_with(host="127.0.0.1", port=18820)
        paths.assert_called_once()
        run.assert_called_once_with(server.serve_forever.return_value)

    def test_audited_target_offset_is_honored_and_never_reapplied_at_eval(self):
        observed = {}
        for target_offset in (0, 1):
            policy = FakePolicy()
            wrapper = ModelWrapper(policy=policy, payload=make_payload(policy, target_offset=target_offset))
            self.assertEqual(wrapper.target_offset, target_offset)
            self.assertFalse(wrapper.gripper_binarization)
            self.assertIn("t+1:t+1+H", wrapper.label_alignment)
            wrapper.update_obs(make_observation(4))
            observed[target_offset] = (
                wrapper.get_action(),
                policy.predict_calls[-1]["timestep"],
                wrapper.protocol_summary(),
            )
        # The offset shapes training labels only: the observation row index stays the memory
        # key verbatim and the executed prefix is byte-identical for either offset.
        self.assertEqual(observed[0][1], observed[1][1])
        self.assertEqual(observed[0][1], 4)
        np.testing.assert_array_equal(observed[0][0], observed[1][0])
        self.assertEqual(observed[1][2]["target_offset"], 1)
        self.assertEqual(observed[1][2]["label_alignment"], AUDITED_LABEL_ALIGNMENT)
        # Continuous grippers are clipped to the native command range and never rounded to 0/1.
        gripper_columns = observed[1][0][:, [6, 13]]
        self.assertTrue(np.all((gripper_columns > 0.0) & (gripper_columns < 1.0)))

    def test_checkpoint_declared_gripper_contract_is_rejected_not_silently_ignored(self):
        with self.assertRaisesRegex(NotImplementedError, "gripper_binarization=true"):
            ModelWrapper(policy=self.policy, payload=make_payload(self.policy, gripper_binarization=True))
        for invalid_offset in (-1, True, "1"):
            with self.subTest(target_offset=invalid_offset), self.assertRaisesRegex(ValueError, "target_offset"):
                ModelWrapper(policy=self.policy, payload=make_payload(self.policy, target_offset=invalid_offset))

    def test_actual_timestep_is_required_and_stale_queries_do_not_write_again(self):
        observation = make_observation(0)
        del observation["timestep"]
        observation["step"] = 99  # Native websocket step counts inference calls, not action steps.
        with self.assertRaisesRegex(ValueError, "actual.*timestep"):
            self.wrapper.update_obs(observation)
        observation["additional_info"] = {"timestep": 5}
        self.wrapper.update_obs(observation)
        self.wrapper.get_action()
        self.wrapper.update_obs(make_observation(5))
        with self.assertRaisesRegex(ValueError, "newer actual"):
            self.wrapper.get_action()
        with self.assertRaisesRegex(ValueError, "backwards"):
            self.wrapper.update_obs(make_observation(4))
        self.assertEqual(len(self.policy.predict_calls), 1)

    def test_failed_prediction_cannot_be_replayed_into_banks(self):
        self.policy.prediction = torch.zeros(1, 3, 14)
        self.wrapper.update_obs(make_observation(0))
        with self.assertRaisesRegex(ValueError, "prediction"):
            self.wrapper.get_action()
        with self.assertRaisesRegex(RuntimeError, "update_obs"):
            self.wrapper.get_action()
        self.wrapper.update_obs(make_observation(0))
        with self.assertRaisesRegex(ValueError, "newer actual"):
            self.wrapper.get_action()
        self.assertEqual(self.policy.per_mem_bank.bank, {self.wrapper.episode_id: [0]})

    def test_diffusion_rng_is_reproducible_from_native_case_not_scheduling(self):
        self.policy.config.action_head = "diffusion"
        first_case = {"task_name": "press_button", "seed": 100000, "action_type": "joint"}
        self.wrapper.prepare_case(first_case)
        self.wrapper.reset()
        first_noise = torch.randn(1, 16, 14)
        self.wrapper.prepare_case({"task_name": "swap_T", "seed": 100001, "action_type": "joint"})
        self.wrapper.reset()
        other_noise = torch.randn(1, 16, 14)
        self.wrapper.prepare_case(first_case)
        self.wrapper.reset()
        repeated_noise = torch.randn(1, 16, 14)
        torch.testing.assert_close(first_noise, repeated_noise, rtol=0, atol=0)
        self.assertFalse(torch.equal(first_noise, other_noise))


def load_cpu_native_rollout():
    """Execute native function definitions, excluding all simulator imports."""
    source_path = DEFAULT_RMBENCH_ROOT / "scripts" / "eval_policy_xpolicylab.py"
    source_tree = ast.parse(source_path.read_text(encoding="utf-8"))
    definitions = [definition for definition in source_tree.body if isinstance(definition, ast.FunctionDef)]
    camera_assignment = next(
        definition
        for definition in source_tree.body
        if isinstance(definition, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "CAMERA_NAME_MAP" for target in definition.targets)
    )
    native_evaluator = types.ModuleType("cpu_native_rollout")
    native_evaluator.__dict__.update(
        {
            "np": np,
            "Mapping": Mapping,
            "Sequence": Sequence,
            "Path": Path,
            "subprocess": subprocess,
            "traceback": traceback,
            "UnStableError": type("FakeUnstableError", (Exception,), {}),
            "generate_episode_descriptions": lambda *_arguments: [{"seen": ["put the battery in the box"]}],
        }
    )
    future_import = ast.parse("from __future__ import annotations").body[0]
    isolated_tree = ast.Module(body=[future_import, camera_assignment, *definitions], type_ignores=[])
    exec(compile(isolated_tree, str(source_path), "exec"), native_evaluator.__dict__)
    install_timestep_adapter(native_evaluator)
    return native_evaluator


class FakeNativeTask:
    def __init__(self, *, success_after, step_limit):
        self.success_after = success_after
        self.step_lim = step_limit
        self.render_freq = 0
        self.eval_video_path = None
        self.setup_calls = []
        self.executed_actions = []
        self.close_calls = []

    def setup_demo(self, **arguments):
        self.setup_calls.append(arguments)
        self.take_action_cnt = 0
        self.eval_success = False
        self.plan_success = True

    def play_once(self):
        return {"info": {}}

    def check_success(self):
        return True

    def close_env(self, clear_cache=False):
        self.close_calls.append(clear_cache)

    def set_instruction(self, instruction):
        self.instruction = instruction

    def get_instruction(self):
        return self.instruction

    def get_obs(self):
        return {
            "observation": {
                camera_name: {"rgb": np.full((3, 5, 3), self.take_action_cnt, dtype=np.uint8)}
                for camera_name in ("head_camera", "left_camera", "right_camera")
            },
            "joint_action": {
                "left_arm": np.zeros(6),
                "left_gripper": 0.5,
                "right_arm": np.zeros(6),
                "right_gripper": 0.5,
            },
        }

    def take_action(self, action, action_type):
        if action_type != "qpos" or action.shape != (14,):
            raise AssertionError("Native deployment changed the action semantics")
        self.executed_actions.append(action.copy())
        self.take_action_cnt += 1
        if self.success_after is not None and self.take_action_cnt >= self.success_after:
            self.eval_success = True


class LocalNativeClient:
    _robotwin_protocol = "ws"

    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.calls = []

    def call(self, func_name=None, obs=None):
        self.calls.append((func_name, obs))
        if func_name in {"prepare_case", "trial_end"}:
            return None
        method = getattr(self.wrapper, func_name)
        return method() if obs is None else method(obs)


class NativeDeploymentTests(unittest.TestCase):
    def test_single_episode_uses_native_loop_and_its_success_or_step_limit(self):
        for success_after, expected_steps, expected_successes in ((5, 5, 1), (None, 6, 0)):
            with self.subTest(success_after=success_after):
                native_evaluator = load_cpu_native_rollout()
                policy = FakePolicy()
                wrapper = ModelWrapper(policy=policy, payload=make_payload(policy))
                client = LocalNativeClient(wrapper)
                task_env = FakeNativeTask(success_after=success_after, step_limit=6)
                task_arguments = {
                    "task_name": "battery_try",
                    "policy_name": "TTTchoiceMem",
                    "task_config": "demo_clean",
                    "ckpt_setting": "fake-policy-only",
                    "render_freq": 0,
                    "clear_cache_freq": 5,
                }
                with redirect_stdout(io.StringIO()):
                    _, successes = native_evaluator.eval_remote_policy(
                        "battery_try",
                        task_env,
                        task_arguments,
                        client,
                        100000,
                        usr_args={"max_seed_attempts": 1},
                        test_num=1,
                        instruction_type="seen",
                    )
                self.assertEqual(task_env.test_num, 1)
                self.assertEqual(successes, expected_successes)
                self.assertEqual(len(task_env.executed_actions), expected_steps)
                self.assertEqual(len(task_env.setup_calls), 2)  # Native expert check, then same-seed rollout.
                self.assertEqual([call["timestep"] for call in policy.predict_calls], [0, 4])
                self.assertEqual(len([name for name, _observation in client.calls if name == "reset"]), 1)
                for memory_bank in (policy.per_mem_bank, policy.cog_mem_bank):
                    self.assertEqual(memory_bank.bank, {wrapper.episode_id: [0, 4]})
                for action in task_env.executed_actions:
                    np.testing.assert_allclose(action[[6, 13]], [0.35, 0.65], atol=1e-7)

    def test_launcher_runs_one_native_child_with_benchmark_paths_and_no_batch(self):
        arguments = parse_args(["--rmbench-root", str(DEFAULT_RMBENCH_ROOT), "--task-name", "battery_try"])
        command = build_native_command(arguments)
        self.assertIn("--native-child", command)
        self.assertEqual(command[command.index("--episodes") + 1], "1")
        with patch("eval_rmbench.subprocess.run") as native_subprocess:
            from eval_rmbench import main

            with redirect_stdout(io.StringIO()):
                main(["--rmbench-root", str(DEFAULT_RMBENCH_ROOT), "--task-name", "battery_try"])
        self.assertEqual(native_subprocess.call_args.kwargs, {"cwd": DEFAULT_RMBENCH_ROOT, "check": True})
        launched_command = native_subprocess.call_args.args[0]
        self.assertEqual(launched_command[launched_command.index("--episodes") + 1], "1")
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--episodes", "2"])

    def test_native_client_treats_policy_name_as_a_label_without_an_adapter_package(self):
        native_source = (DEFAULT_RMBENCH_ROOT / "scripts" / "eval_policy_xpolicylab.py").read_text(encoding="utf-8")
        # The shared eval entrypoint only labels results with policy_name; it resolves the
        # policy over the socket and never imports XPolicyLab.policy.<name>.model.
        self.assertNotIn("XPolicyLab.policy.", native_source)
        self.assertIn('importlib.import_module(f"envs.{task_name}")', native_source)
        self.assertFalse((DEFAULT_RMBENCH_ROOT / "XPolicyLab" / "policy" / "TTTchoiceMem").exists())
        # Our two files therefore reference the shared communication server, not a policy package.
        self.assertIn("client_server.ws.model_server", (PROJECT_ROOT / "serve_rmbench.py").read_text(encoding="utf-8"))
        self.assertNotIn("XPolicyLab.policy", (PROJECT_ROOT / "eval_rmbench.py").read_text(encoding="utf-8"))
        self.assertNotIn("XPolicyLab.policy", (PROJECT_ROOT / "serve_rmbench.py").read_text(encoding="utf-8"))


@unittest.skipUnless(NETWORK_SMOKE, "opt in with --network-smoke; this remains a CPU fake-policy check")
class NativeSocketSmokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_native_server_client_reset_observe_query_and_reset(self):
        add_native_paths(DEFAULT_RMBENCH_ROOT)
        from client_server.ws.model_client import WsModelClient
        from client_server.ws.model_server import PolicyServer, PolicyServerConfig

        policy = FakePolicy()
        wrapper = ModelWrapper(policy=policy, payload=make_payload(policy))
        server = PolicyServer(wrapper, PolicyServerConfig(host="127.0.0.1", port=0))
        await server.start()

        def exchange_requests():
            with WsModelClient(
                url=server.url,
                evaluation_id="cpu-fake-policy-smoke",
                trial_id="one-fake-episode",
                action_case_id="press-button-native-seed100000",
                connect_timeout_s=2,
                handshake_timeout_s=2,
                request_timeout_s=2,
                max_connect_attempts=1,
            ) as client:
                case_result = client.call(
                    func_name="prepare_case",
                    obs={"task_name": "press_button", "seed": 100000, "action_type": "joint"},
                )
                if case_result["inference_seed"] != wrapper.inference_seed:
                    raise AssertionError("Native transport did not apply the checkpoint-derived inference seed")
                client.call(func_name="reset")
                for timestep in (0, 1, 2):
                    client.call(func_name="update_obs", obs=make_observation(timestep))
                actions = client.call(func_name="get_action")
                client.call(func_name="reset")
                return np.asarray(actions)

        try:
            with patch.dict(os.environ, {"NO_PROXY": "localhost,127.0.0.1,::1", "no_proxy": "localhost,127.0.0.1,::1"}):
                actions = await asyncio.to_thread(exchange_requests)
        finally:
            await server.stop()
        self.assertEqual(actions.shape, (4, 14))
        self.assertEqual(wrapper.case_metadata["seed"], 100000)
        self.assertEqual(wrapper.case_metadata["task_name"], "press_button")
        np.testing.assert_allclose(actions[0, [6, 13]], [0.35, 0.65], atol=1e-7)
        self.assertEqual([call["timestep"] for call in policy.predict_calls], [2])
        np.testing.assert_array_equal(policy.predict_calls[0]["images"][0][0, 0], [2, 40, 201])
        self.assertEqual(policy.per_mem_bank.bank, {})
        self.assertEqual(policy.cog_mem_bank.bank, {})
        self.assertIsNone(wrapper.pending_observation)


if __name__ == "__main__":
    unittest.main(verbosity=2)
