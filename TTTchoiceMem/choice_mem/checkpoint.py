"""Strict checkpoints: weights plus the complete action/observation contract."""

import platform
from pathlib import Path

import torch

from .config import PolicyConfig
from .policy import ChoiceMemoryPolicy

CHECKPOINT_SCHEMA = "choice-memory-rmbench-v1"


def save_checkpoint(path, policy, normalization, protocol, step, optimizer=None, training_state=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing checkpoint: {path}")
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "config": policy.config.to_dict(),
        "model": policy.state_dict(),
        "normalization": normalization,
        "protocol": protocol,
        "step": step,
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__)},
        "weight_compatibility": {
            "reused": "Selected pretrained VLM backbone only; no old action weights are imported",
            "initialized": ["per_compr", "per_mem_bank", "cog_mem_bank", "action_head"],
            "incompatible": "Original 7D diffusion / EventVLA action heads and banks after representation changes",
        },
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if training_state is not None:
        payload["training_state"] = training_state
    temporary_path = path.with_suffix(path.suffix + ".partial")
    if temporary_path.exists():
        raise FileExistsError(f"Previous incomplete checkpoint exists: {temporary_path}")
    torch.save(payload, temporary_path)
    temporary_path.rename(path)


def read_checkpoint(path):
    # Load only checkpoints produced by this code, never arbitrary downloaded pickle objects.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(
            "Not a choice-memory checkpoint; original action weights require explicit conversion/retraining"
        )
    for required_key in ["config", "model", "normalization", "protocol", "step"]:
        if required_key not in payload:
            raise ValueError(f"Checkpoint missing required field: {required_key}")
    required_configuration = {
        "backbone",
        "action_head",
        "model_source",
        "model_revision",
        "preprocessing",
        "feature_dimensions",
        "horizon",
        "action_dim",
        "execute_prefix",
        "cameras",
        "image_size",
        "per_token_size",
        "mem_length",
        "retrieval_layers",
        "use_timestep_pe",
        "fusion_type",
        "consolidate_type",
        "update_fused",
        "group_size",
        "oft_hidden_dim",
        "diffusion_model_type",
        "ddim_steps",
        "cfg_scale",
        "freeze_backbone",
        "dtype",
        "use_cache",
        "asset_manifest",
    }
    missing_configuration = required_configuration.difference(payload["config"])
    if missing_configuration:
        raise ValueError(f"Checkpoint configuration missing fields: {sorted(missing_configuration)}")
    normalization = payload["normalization"]
    statistics = normalization.get("action", normalization)
    if len(statistics.get("q01", [])) != 14 or len(statistics.get("q99", [])) != 14:
        raise ValueError("Checkpoint must contain train-only 14D action normalization")
    PolicyConfig(**payload["config"])
    return payload


def load_checkpoint(path, device="cpu", expected=None, backbone=None):
    payload = read_checkpoint(path)
    config = PolicyConfig(**payload["config"])
    for field_name, expected_value in (expected or {}).items():
        if field_name not in payload["config"] or payload["config"][field_name] != expected_value:
            raise ValueError(f"Checkpoint configuration mismatch: {field_name}={expected_value!r}")
    policy = ChoiceMemoryPolicy(config, backbone=backbone, initialize_from_config=True)
    policy.load_state_dict(payload["model"], strict=True)
    policy.to(device).eval()
    policy.reset()
    return policy, payload
