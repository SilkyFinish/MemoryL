"""Single-episode XPolicyLab RPC adapter; only inference touches memory banks."""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Mapping
from copy import deepcopy
from numbers import Integral
from pathlib import Path

import numpy as np
import torch

from .config import validate_cameras

DEFAULT_RMBENCH_ROOT = Path("/mnt/data/dylan/d2/memory/RMBench")
# Audited in RMBench/scripts/eval_policy_xpolicylab.py:CAMERA_NAME_MAP.
NATIVE_CAMERA_NAMES = {
    "head_camera": "cam_head",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}


def add_native_paths(rmbench_root: str | Path) -> Path:
    """Make the existing communication package importable without installing policies."""
    root = Path(rmbench_root).expanduser().resolve(strict=True)
    if not (root / "XPolicyLab" / "client_server" / "ws" / "model_server.py").is_file():
        raise FileNotFoundError(f"Missing native XPolicyLab communication server under {root}")
    for directory in (root, root / "XPolicyLab"):
        directory_name = str(directory)
        if directory_name not in sys.path:
            sys.path.insert(0, directory_name)
    return root


class ModelWrapper:
    """Native reset/update_obs/get_action interface for one client and one episode.

    Injecting ``policy`` and ``payload`` is for CPU contract tests, not a model
    fallback. Production always loads both through choice_mem.checkpoint.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "cpu",
        expected: dict | None = None,
        action_type: str = "joint",
        *,
        expected_protocol: dict | None = None,
        policy=None,
        payload: dict | None = None,
        allow_debug_checkpoint: bool = False,
    ):
        if action_type != "joint":
            raise ValueError("TTTchoiceMem requires action_type=joint (native ALOHA absolute joints)")
        if policy is None:
            if checkpoint_path is None or payload is not None:
                raise ValueError("Pass a checkpoint, or inject both policy and payload for a CPU test")
            from .checkpoint import load_checkpoint

            debug_opt_in = {"allow_debug_checkpoint": True} if allow_debug_checkpoint is True else {}
            policy, payload = load_checkpoint(checkpoint_path, device, expected=expected, **debug_opt_in)
        elif checkpoint_path is not None or payload is None:
            raise ValueError("Injected policies require payload and must not claim a real checkpoint")
        self.policy = policy
        self.config = policy.config
        self.protocol = deepcopy(payload["protocol"])
        self.normalization = deepcopy(payload["normalization"])
        if not isinstance(self.protocol, Mapping) or not isinstance(self.normalization, Mapping):
            raise ValueError("Checkpoint protocol and normalization must be mappings")
        configuration = self.config.to_dict() if hasattr(self.config, "to_dict") else vars(self.config)
        if configuration.get("backbone_initialization", "pretrained") == "random-frozen-debug":
            if allow_debug_checkpoint is not True:
                raise ValueError("Debug checkpoint requires explicit allow_debug_checkpoint=True")
            from .checkpoint import validate_initialization_contract

            validate_initialization_contract(configuration, self.protocol, payload.get("weight_compatibility"))
        for field_name, requested_value in (expected or {}).items():
            if field_name not in configuration or configuration[field_name] != requested_value:
                raise ValueError(f"Checkpoint configuration mismatch: {field_name}={requested_value!r}")
        for field_name, requested_value in (expected_protocol or {}).items():
            if self.protocol.get(field_name) != requested_value:
                raise ValueError(f"Checkpoint protocol mismatch: {field_name}={requested_value!r}")
        self._validate_contract(configuration)
        statistics = self.normalization.get("action", self.normalization)
        self.lower_quantiles = np.asarray(statistics.get("q01"), dtype=np.float32)
        self.upper_quantiles = np.asarray(statistics.get("q99"), dtype=np.float32)
        self.normalization_mask = np.asarray(statistics.get("mask", [True] * 14))
        if self.lower_quantiles.shape != (14,) or self.upper_quantiles.shape != (14,):
            raise ValueError("Checkpoint action quantiles must each contain 14 channels")
        if self.normalization_mask.shape != (14,) or self.normalization_mask.dtype != np.bool_:
            raise ValueError("Checkpoint action mask must contain 14 booleans")
        if not np.isfinite(self.lower_quantiles).all() or not np.isfinite(self.upper_quantiles).all():
            raise ValueError("Checkpoint action quantiles must be finite")
        if np.any(self.upper_quantiles < self.lower_quantiles):
            raise ValueError("Checkpoint q99 must be greater than or equal to q01")
        self.policy.eval()
        self.episode_number = 0
        self.inference_seed = int(self.protocol.get("training_seed", 42))
        self.case_metadata = None
        self.reset()

    def _validate_contract(self, configuration: dict) -> None:
        if self.config.action_dim != 14:
            raise ValueError("Checkpoint must have the native ALOHA 14D joint action")
        if not 1 <= self.config.execute_prefix <= self.config.horizon:
            raise ValueError("Invalid checkpoint execution prefix")
        validate_cameras(self.config.cameras)
        if not isinstance(self.protocol, Mapping):
            raise ValueError("Checkpoint protocol must be a mapping")
        for field_name in ("action_dim", "horizon", "execute_prefix", "cameras", "image_size"):
            if field_name in self.protocol and self.protocol[field_name] != configuration[field_name]:
                raise ValueError(f"Checkpoint protocol/configuration mismatch: {field_name}")
        if "action_type" in self.protocol and self.protocol["action_type"] != "joint":
            raise ValueError("Checkpoint protocol must use joint actions")
        native_action_order = (
            [f"left_arm_joint_{index}" for index in range(6)]
            + ["left_gripper"]
            + [f"right_arm_joint_{index}" for index in range(6)]
            + ["right_gripper"]
        )
        if "channel_order" in self.protocol and self.protocol["channel_order"] != native_action_order:
            raise ValueError("Checkpoint action channel order differs from native ALOHA joints/grippers")
        if "gripper_indices" in self.protocol and self.protocol["gripper_indices"] != [6, 13]:
            raise ValueError("Checkpoint gripper indices must be the native channels 6 and 13")
        # Honor whatever the checkpoint declares: record the trained label offset
        # and gripper contract instead of assuming a shift or a rounding rule.
        self.target_offset = self.protocol.get("target_offset")
        if self.target_offset is not None and (
            isinstance(self.target_offset, bool)
            or not isinstance(self.target_offset, Integral)
            or self.target_offset < 0
        ):
            raise ValueError("Checkpoint target_offset must be a nonnegative integer when present")
        self.label_alignment = self.protocol.get("label_alignment")
        self.gripper_binarization = bool(self.protocol.get("gripper_binarization", False))
        if self.gripper_binarization:
            # The checkpoint's own contract wins; this adapter only implements the continuous one.
            raise NotImplementedError(
                "Checkpoint declares gripper_binarization=true, which this runtime does not implement"
            )
        # The offset/label alignment shape training labels only. At eval the checkpoint's own
        # horizon is read back, so no local shift is applied to query timesteps or predictions.

    def protocol_summary(self) -> dict:
        """Report the honored checkpoint contract for server logs and audits."""
        return {
            "target_offset": self.target_offset,
            "label_alignment": self.label_alignment,
            "gripper_binarization": self.gripper_binarization,
            "horizon": self.config.horizon,
            "execute_prefix": self.config.execute_prefix,
            "cameras": list(self.config.cameras),
            "action_dim": self.config.action_dim,
            "backbone_initialization": getattr(self.config, "backbone_initialization", "pretrained"),
            "backbone_initialization_seed": getattr(self.config, "backbone_initialization_seed", 42),
        }

    def reset(self, episode_id=None) -> None:
        self.episode_number += 1
        self.episode_id = self.episode_number if episode_id is None else episode_id
        self.pending_observation = None
        self.last_observation_timestep = None
        self.last_query_timestep = None
        self.query_count = 0
        self.policy.reset(self.episode_id)
        if self.config.action_head == "diffusion":
            # Seed from the native episode identity, not task scheduling order.
            # Only the server's isolated visible devices are affected.
            torch.manual_seed(self.inference_seed)

    def prepare_case(self, case_metadata: Mapping) -> dict:
        if not isinstance(case_metadata, Mapping):
            raise ValueError("Native case metadata must be a mapping")
        seed = case_metadata.get("seed")
        task_name = case_metadata.get("task_name")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0 or not isinstance(task_name, str):
            raise ValueError("Native case requires task_name and a nonnegative seed")
        if case_metadata.get("action_type", "joint") != "joint":
            raise ValueError("Native case action_type must be joint")
        self.case_metadata = dict(case_metadata)
        identity = f"{self.protocol.get('training_seed', 42)}:{task_name}:{seed}"
        self.inference_seed = int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest()[:4], "little")
        return {"inference_seed": self.inference_seed, "task_name": task_name, "seed": int(seed)}

    def update_obs(self, obs: Mapping) -> None:
        if not isinstance(obs, Mapping):
            raise TypeError("update_obs expects a decoded XPolicyLab observation mapping")
        if obs.get("env_idx", 0) != 0:
            raise ValueError("This adapter supports only one native episode (env_idx=0)")
        additional_info = obs.get("additional_info", {})
        if not isinstance(additional_info, Mapping):
            raise ValueError("Observation additional_info must be a mapping")
        timestep = obs.get("timestep", additional_info.get("timestep"))
        if isinstance(timestep, bool) or not isinstance(timestep, Integral) or timestep < 0:
            raise ValueError("Observation requires the actual nonnegative RMBench action timestep; use eval_rmbench.py")
        timestep = int(timestep)
        if self.last_observation_timestep is not None and timestep < self.last_observation_timestep:
            raise ValueError("Observation timestep moved backwards; reset for a new episode")
        instruction = obs.get("instruction")
        if instruction is None and obs.get("instructions"):
            instruction = obs["instructions"][0]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Observation requires a nonempty task instruction")
        vision = obs.get("vision")
        if not isinstance(vision, Mapping):
            raise ValueError("Observation requires decoded obs['vision'] camera arrays")
        images = []
        for camera_name in self.config.cameras:
            native_name = NATIVE_CAMERA_NAMES[camera_name]
            camera = vision.get(native_name, vision.get(camera_name))
            if not isinstance(camera, Mapping) or "color" not in camera:
                raise ValueError(f"Missing decoded RGB camera {native_name!r}")
            image = np.asarray(camera["color"])
            if image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 1 or image.dtype != np.uint8:
                raise ValueError(
                    f"Camera {native_name!r} must be a uint8 HWC RGB array, got {image.shape}/{image.dtype}"
                )
            # Own the latest pixels, without decoding, channel swapping, or writing banks.
            images.append(np.array(image, copy=True, order="C"))
        self.pending_observation = (images, instruction, timestep)
        self.last_observation_timestep = timestep

    def get_action(self) -> np.ndarray:
        if self.pending_observation is None:
            raise RuntimeError("Call update_obs after reset and before each get_action")
        images, instruction, timestep = self.pending_observation
        if self.last_query_timestep is not None and timestep <= self.last_query_timestep:
            raise ValueError("A query requires a newer actual observation timestep")
        # Consume before inference: retrying a failed prediction cannot write history twice.
        self.pending_observation = None
        self.last_query_timestep = timestep
        self.query_count += 1
        with torch.inference_mode():
            prediction = self.policy.predict(images, instruction, self.episode_id, timestep=timestep)
        if not isinstance(prediction, torch.Tensor) or tuple(prediction.shape) != (1, self.config.horizon, 14):
            raise ValueError(
                f"Expected policy prediction [1,{self.config.horizon},14], got {getattr(prediction, 'shape', None)}"
            )
        normalized_actions = prediction[0, : self.config.execute_prefix].detach().float().cpu().numpy()
        if not np.isfinite(normalized_actions).all():
            raise ValueError("Policy returned nonfinite actions")
        normalized_actions = np.clip(normalized_actions, -1.0, 1.0)
        quantile_range = self.upper_quantiles - self.lower_quantiles
        restored_actions = 0.5 * (normalized_actions + 1.0) * quantile_range + self.lower_quantiles
        restored_actions[:, quantile_range <= 1e-8] = self.lower_quantiles[quantile_range <= 1e-8]
        actions = np.where(self.normalization_mask, restored_actions, normalized_actions).astype(np.float32)
        actions[:, [6, 13]] = np.clip(actions[:, [6, 13]], 0.0, 1.0)
        return actions
