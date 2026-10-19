"""Serializable MemoryVLA method and native RMBench observation contract."""

from dataclasses import asdict, dataclass, field


SUPPORTED_CAMERAS = ("head_camera", "left_camera", "right_camera")
DEBUG_INITIALIZATIONS = frozenset({"random-frozen-debug", "turbo-random-frozen-debug"})


def is_debug_initialization(initialization):
    return initialization in DEBUG_INITIALIZATIONS


def debug_parameter_dtype(name, initialization):
    """Torch-free storage contract; computation precision is recorded separately."""
    if not is_debug_initialization(initialization):
        raise ValueError("A debug initialization is required for the debug storage contract")
    if initialization == "turbo-random-frozen-debug" and name.startswith("backbone.vision_language_interaction."):
        return "float32"
    return "bfloat16" if name.startswith("backbone.") else "float32"


def debug_precision_metadata(initialization, dtype="bfloat16"):
    if not is_debug_initialization(initialization):
        return {}
    metadata = {
        "master_dtype": "float32", "autocast_dtype": dtype,
        "frozen_backbone_dtype": "bfloat16",
    }
    if initialization == "turbo-random-frozen-debug":
        metadata.update({
            "frozen_backbone_dtype": "mixed:bfloat16,float32",
            "frozen_interaction_dtype": "float32", "interaction_compute_dtype": "float32",
        })
    return metadata


def validate_cameras(cameras) -> list[str]:
    """Keep the selected view order explicit and reject silently repeated views."""
    if isinstance(cameras, str):
        raise ValueError("cameras must be an ordered list, not a single string")
    selected_cameras = list(cameras)
    if not 1 <= len(selected_cameras) <= len(SUPPORTED_CAMERAS):
        raise ValueError("Select between one and three RMBench cameras")
    if len(set(selected_cameras)) != len(selected_cameras):
        raise ValueError("Camera names must be nonempty and unique")
    if any(camera not in SUPPORTED_CAMERAS for camera in selected_cameras):
        raise ValueError(f"Camera names must come from {SUPPORTED_CAMERAS}")
    return selected_cameras


@dataclass
class PolicyConfig:
    backbone: str = "memoryvla"
    action_head: str = "diffusion"
    # Original VLM assets must be selected explicitly, never a Qwen fallback.
    model_source: str = ""
    model_revision: str | None = None
    backbone_initialization: str = "pretrained"
    backbone_initialization_seed: int = 42
    base_vlm_id: str = "prism-dinosiglip-224px+7b"
    horizon: int = 16
    action_dim: int = 14
    execute_prefix: int = 4
    cameras: list[str] = field(default_factory=lambda: ["head_camera"])
    image_size: list[int] = field(default_factory=lambda: [224, 224])
    per_token_size: int = 256
    mem_length: int = 16
    retrieval_layers: int = 2
    use_timestep_pe: bool = True
    fusion_type: str = "gate"
    consolidate_type: str = "tome"
    update_fused: bool = False
    group_size: int = 16
    history_sampling: str = "random"
    image_aug: bool = True
    oft_hidden_dim: int = 256
    diffusion_model_type: str = "DiT-L"
    repeated_diffusion_steps: int = 4
    mask_padding_loss: bool = False
    inference_timestep_mode: str = "query"
    ddim_steps: int = 10
    cfg_scale: float = 1.5
    freeze_backbone: bool = False
    dtype: str = "bfloat16"
    use_cache: bool = False
    preprocessing: dict = field(default_factory=dict)
    feature_dimensions: dict = field(default_factory=dict)
    asset_manifest: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.backbone_initialization not in {"pretrained", *DEBUG_INITIALIZATIONS}:
            raise ValueError("backbone_initialization must be pretrained, random-frozen-debug or turbo-random-frozen-debug")
        if type(self.backbone_initialization_seed) is not int or not 0 <= self.backbone_initialization_seed < 2**63:
            raise ValueError("backbone_initialization_seed must be an integer in [0, 2**63)")
        if self.backbone_initialization == "random-frozen-debug":
            required = {
                "backbone": "memoryvla",
                "action_head": "diffusion",
                "diffusion_model_type": "DiT-L",
                "freeze_backbone": True,
                "dtype": "bfloat16",
                "base_vlm_id": "prism-dinosiglip-224px+7b",
            }
            for name, value in required.items():
                if getattr(self, name) != value or (name == "freeze_backbone" and self.freeze_backbone is not True):
                    raise ValueError(f"random-frozen-debug requires {name}={value!r}")
        if self.backbone_initialization == "turbo-random-frozen-debug":
            required = {
                "backbone": "turbo_interaction", "freeze_backbone": True, "dtype": "bfloat16",
            }
            for name, value in required.items():
                if type(getattr(self, name)) is not type(value) or getattr(self, name) != value:
                    raise ValueError(f"turbo-random-frozen-debug requires {name}={value!r}")
        if self.backbone not in {"memoryvla", "qwen35", "turbo_interaction"}:
            raise ValueError("backbone must be memoryvla, qwen35 or turbo_interaction")
        if self.backbone == "turbo_interaction":
            # This identifier selects only the original vision architecture; it
            # must not cause the replacement path to instantiate Llama or its projector.
            required = {
                "base_vlm_id": "prism-dinosiglip-224px+7b", "action_head": "diffusion",
                "diffusion_model_type": "DiT-L", "image_size": [224, 224], "per_token_size": 256,
            }
            for name, value in required.items():
                if getattr(self, name) != value:
                    raise ValueError(f"turbo_interaction minimal replacement requires {name}={value!r}")
        if self.action_head not in {"diffusion", "oft"}:
            raise ValueError("action_head must be diffusion or oft")
        if self.action_dim != 14:
            raise ValueError("This RMBench adapter requires the native ALOHA 14D absolute joint action")
        if not 1 <= self.execute_prefix <= self.horizon:
            raise ValueError("execute_prefix must be between 1 and horizon")
        if self.use_cache:
            raise ValueError("Cross-query backbone caching is not supported; use_cache must be false")
        if self.mem_length < 1 or self.retrieval_layers < 1 or self.group_size < 2:
            raise ValueError("Positive memory capacity/layers and a group_size >= 2 are required")
        if self.history_sampling not in {"random", "sequential"}:
            raise ValueError("history_sampling must be random or sequential")
        if type(self.repeated_diffusion_steps) is not int or self.repeated_diffusion_steps < 1:
            raise ValueError("repeated_diffusion_steps must be a positive integer")
        if type(self.mask_padding_loss) is not bool or type(self.image_aug) is not bool:
            raise ValueError("mask_padding_loss and image_aug must be booleans")
        if self.inference_timestep_mode not in {"query", "action"}:
            raise ValueError("inference_timestep_mode must be query or action")
        self.cameras = validate_cameras(self.cameras)
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError("image_size must contain height and width")
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("dtype must be float32 or bfloat16")

    def to_dict(self):
        return asdict(self)
