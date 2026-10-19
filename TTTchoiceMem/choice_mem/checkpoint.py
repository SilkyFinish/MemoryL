"""Strict checkpoints: weights plus the complete action/observation contract."""

from copy import deepcopy
import platform
from pathlib import Path

import torch

from .config import PolicyConfig, debug_parameter_dtype, debug_precision_metadata, is_debug_initialization
from .policy import ChoiceMemoryPolicy

CHECKPOINT_SCHEMA = "choice-memory-rmbench-v2"
LEGACY_CHECKPOINT_SCHEMA = "choice-memory-rmbench-v1"
METHOD_FIELDS = {
    "history_sampling", "image_aug", "repeated_diffusion_steps",
    "mask_padding_loss", "inference_timestep_mode",
}
LEGACY_METHOD = {
    "history_sampling": "sequential",
    "image_aug": False,
    "repeated_diffusion_steps": 1,
    "mask_padding_loss": True,
    "inference_timestep_mode": "action",
}


# Missing fields in historical v1/v2 snapshots mean the original pretrained
# route, never permission to reinterpret those weights as random initialization.
INITIALIZATION_DEFAULTS = {
    "backbone_initialization": "pretrained",
    "backbone_initialization_seed": 42,
    "base_vlm_id": "prism-dinosiglip-224px+7b",
}


def validate_initialization_contract(configuration, protocol, weight_compatibility=None):
    """Check new provenance without requiring new fields in historical recipes."""
    initialization = configuration.get("backbone_initialization", "pretrained")
    debug = is_debug_initialization(initialization)
    fields = (*INITIALIZATION_DEFAULTS, "dtype", "freeze_backbone")
    if configuration.get("backbone") == "turbo_interaction":
        fields += ("backbone",)
    for field_name in fields:
        expected = configuration.get(field_name, INITIALIZATION_DEFAULTS.get(field_name))
        if field_name in protocol:
            if type(protocol[field_name]) is not type(expected) or protocol[field_name] != expected:
                raise ValueError(f"Checkpoint protocol/configuration mismatch: {field_name}")
        elif debug and field_name in INITIALIZATION_DEFAULTS:
            raise ValueError(f"Debug checkpoint protocol missing field: {field_name}")
        requested = protocol.get("requested_configuration")
        if isinstance(requested, dict):
            if field_name in requested:
                if type(requested[field_name]) is not type(expected) or requested[field_name] != expected:
                    raise ValueError(f"Checkpoint requested_configuration disagrees: {field_name}")
            elif debug:
                raise ValueError(f"Debug checkpoint requested_configuration missing field: {field_name}")
        if isinstance(weight_compatibility, dict) and field_name in weight_compatibility:
            if type(weight_compatibility[field_name]) is not type(expected) or weight_compatibility[field_name] != expected:
                raise ValueError(f"Checkpoint weight_compatibility disagrees: {field_name}")
    if debug:
        precision = debug_precision_metadata(initialization, configuration["dtype"])
        optimization = protocol.get("optimization")
        if optimization is not None:
            if not isinstance(optimization, dict):
                raise ValueError("Debug checkpoint optimization must be a mapping")
            for field, expected in precision.items():
                if optimization.get(field) != expected:
                    raise ValueError(f"Debug checkpoint optimization dtype disagrees: {field}")
        if isinstance(weight_compatibility, dict):
            compatibility_precision_fields = precision if initialization == "turbo-random-frozen-debug" else ("master_dtype", "frozen_backbone_dtype")
            for field in compatibility_precision_fields:
                if field in weight_compatibility and weight_compatibility[field] != precision[field]:
                    raise ValueError(f"Checkpoint weight_compatibility disagrees: {field}")
    return debug


def validate_turbo_source_assets(configuration):
    """Audit Turbo's own asset contract, never the original Llama/VLA sources."""
    from .turbo_assets import inspect_turbo_assets

    inspected = inspect_turbo_assets(
        configuration["model_source"], initialization=configuration.get("backbone_initialization", "pretrained")
    )
    if not configuration.get("asset_manifest") or configuration["asset_manifest"] != inspected["asset_manifest"]:
        raise ValueError("Turbo checkpoint source asset_manifest changed or is incomplete")


def validate_debug_source_assets(configuration):
    """Validate debug metadata only, using separate original and Turbo inspectors."""
    if configuration.get("backbone_initialization") == "turbo-random-frozen-debug":
        validate_turbo_source_assets(configuration)
        return
    from .random_init import inspect_random_backbone_assets

    inspected = inspect_random_backbone_assets(configuration["model_source"])
    if not configuration.get("asset_manifest") or configuration["asset_manifest"] != inspected["asset_manifest"]:
        raise ValueError("Debug checkpoint metadata source asset_manifest changed or is incomplete")


def _validate_debug_model_state(model, *, initialization="random-frozen-debug", check_finite=True):
    for name, tensor in model.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Debug checkpoint model entry is not a tensor: {name}")
        if tensor.is_floating_point() or tensor.is_complex():
            expected_dtype = getattr(torch, debug_parameter_dtype(name, initialization))
            if tensor.dtype != expected_dtype or (check_finite and not torch.isfinite(tensor).all()):
                raise ValueError(f"Debug checkpoint has nonfinite/incorrect dtype model tensor: {name}")


def save_checkpoint(path, policy, normalization, protocol, step, optimizer=None, training_state=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing checkpoint: {path}")
    configuration = policy.config.to_dict()
    protocol = deepcopy(protocol)
    # Keep the caller's protocol immutable, fill only absent provenance, and
    # refuse contradictory declarations rather than silently overwriting them.
    for field_name, default in INITIALIZATION_DEFAULTS.items():
        value = configuration.get(field_name, default)
        protocol.setdefault(field_name, value)
        if isinstance(protocol.get("requested_configuration"), dict):
            protocol["requested_configuration"].setdefault(field_name, value)
    turbo = configuration.get("backbone") == "turbo_interaction"
    if turbo:
        protocol.setdefault("backbone", configuration["backbone"])
    debug = validate_initialization_contract(configuration, protocol)
    if debug:
        validate_debug_source_assets(configuration)
    elif turbo:
        validate_turbo_source_assets(configuration)
    compatibility = {
        **{field: configuration.get(field, default) for field, default in INITIALIZATION_DEFAULTS.items()},
        "dtype": configuration["dtype"],
        "freeze_backbone": configuration["freeze_backbone"],
        "reused": (
            "None; the frozen VLM backbone is randomly initialized, not pretrained"
            if debug else "Selected pretrained VLM backbone only; no old action weights are imported"
        ),
        "initialized": (["backbone"] if debug else []) + ["per_compr", "per_mem_bank", "cog_mem_bank", "action_head"],
        "incompatible": "Original 7D diffusion / EventVLA action heads and banks after representation changes",
    }
    if turbo:
        compatibility.update(
            backbone="turbo_interaction",
            reused=(
                "None; frozen Turbo vision/BERT/interaction weights are randomly initialized, not pretrained"
                if debug else "Audited original vision encoders, BERT and GroundingDINO fusion/text/feat_map subset; no Llama or old action weights"
            ),
            initialized=(
                ["backbone"] if debug else ["backbone.cognition_projection", "backbone.vision_projection"]
            ) + ["per_compr", "per_mem_bank", "cog_mem_bank", "action_head"],
            incompatible="Original MemoryVLA/Llama checkpoints and action heads; Turbo requires complete matching architecture weights",
        )
    if debug:
        precision = debug_precision_metadata(configuration["backbone_initialization"], configuration["dtype"])
        if not turbo:
            precision.pop("autocast_dtype")  # Preserve the original debug provenance exactly.
        compatibility.update(precision)
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "config": configuration,
        "model": policy.state_dict(),
        "normalization": normalization,
        "protocol": protocol,
        "step": step,
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__)},
        "weight_compatibility": compatibility,
    }
    if debug:
        _validate_debug_model_state(payload["model"], initialization=configuration["backbone_initialization"], check_finite=False)
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
    if payload.get("schema") not in {CHECKPOINT_SCHEMA, LEGACY_CHECKPOINT_SCHEMA}:
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
    if payload["schema"] == CHECKPOINT_SCHEMA:
        required_configuration.update(METHOD_FIELDS)
    missing_configuration = required_configuration.difference(payload["config"])
    if missing_configuration:
        raise ValueError(f"Checkpoint configuration missing fields: {sorted(missing_configuration)}")
    normalization = payload["normalization"]
    statistics = normalization.get("action", normalization)
    if len(statistics.get("q01", [])) != 14 or len(statistics.get("q99", [])) != 14:
        raise ValueError("Checkpoint must contain train-only 14D action normalization")
    if payload["schema"] == LEGACY_CHECKPOINT_SCHEMA:
        # Old Qwen/OFT and diffusion snapshots must retain their trained method,
        # rather than silently adopting the restored MemoryVLA defaults.
        payload["config"] = {**LEGACY_METHOD, **payload["config"]}
    payload["config"] = {**INITIALIZATION_DEFAULTS, **payload["config"]}
    PolicyConfig(**payload["config"])
    validate_initialization_contract(payload["config"], payload["protocol"], payload.get("weight_compatibility"))
    return payload


def load_checkpoint(path, device="cpu", expected=None, backbone=None, allow_debug_checkpoint=False):
    payload = read_checkpoint(path)
    config = PolicyConfig(**payload["config"])
    debug = is_debug_initialization(config.backbone_initialization)
    if debug and allow_debug_checkpoint is not True:
        raise ValueError("Debug checkpoint requires explicit allow_debug_checkpoint=True")
    for field_name, expected_value in (expected or {}).items():
        if field_name not in payload["config"] or payload["config"][field_name] != expected_value:
            raise ValueError(f"Checkpoint configuration mismatch: {field_name}={expected_value!r}")
    if debug:
        # Check before constructing even an injected backbone; no original
        # checkpoint/vision weights are assets of this mode.
        validate_debug_source_assets(payload["config"])
        _validate_debug_model_state(payload["model"], initialization=config.backbone_initialization)
    elif config.backbone == "turbo_interaction":
        # Audit provenance, but initialize_from_config constructs architecture only;
        # every actual weight is restored strictly from this full checkpoint.
        validate_turbo_source_assets(payload["config"])
    policy = ChoiceMemoryPolicy(config, backbone=backbone, initialize_from_config=True)
    if debug:
        from .precision import prepare_training_precision

        prepare_training_precision(policy)
    policy.load_state_dict(payload["model"], strict=True)
    policy.to(device).eval()
    policy.reset()
    return policy, payload
