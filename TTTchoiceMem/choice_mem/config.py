"""Serializable architecture and observation contract for all four policies."""

from dataclasses import asdict, dataclass, field


SUPPORTED_CAMERAS = ("head_camera", "left_camera", "right_camera")


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
    backbone: str = "qwen35"
    action_head: str = "oft"
    model_source: str = "Qwen/Qwen3.5-0.8B"
    model_revision: str | None = None
    horizon: int = 16
    action_dim: int = 14
    execute_prefix: int = 4
    cameras: list[str] = field(default_factory=lambda: ["head_camera", "left_camera", "right_camera"])
    image_size: list[int] = field(default_factory=lambda: [224, 224])
    per_token_size: int = 256
    mem_length: int = 16
    retrieval_layers: int = 2
    use_timestep_pe: bool = True
    fusion_type: str = "gate"
    consolidate_type: str = "tome"
    update_fused: bool = False
    group_size: int = 4
    oft_hidden_dim: int = 256
    diffusion_model_type: str = "DiT-S"
    ddim_steps: int = 10
    cfg_scale: float = 1.5
    freeze_backbone: bool = True
    dtype: str = "bfloat16"
    use_cache: bool = False
    preprocessing: dict = field(default_factory=dict)
    feature_dimensions: dict = field(default_factory=dict)
    asset_manifest: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.backbone not in {"memoryvla", "qwen35"}:
            raise ValueError("backbone must be memoryvla or qwen35")
        if self.action_head not in {"diffusion", "oft"}:
            raise ValueError("action_head must be diffusion or oft")
        if self.action_dim != 14:
            raise ValueError("This RMBench adapter requires the native ALOHA 14D absolute joint action")
        if not 1 <= self.execute_prefix <= self.horizon:
            raise ValueError("execute_prefix must be between 1 and horizon")
        if self.use_cache:
            raise ValueError("Cross-query backbone caching is not supported; use_cache must be false")
        if self.mem_length < 1 or self.retrieval_layers < 1 or self.group_size < 2:
            raise ValueError("Positive memory capacity/layers and a sequential group_size >= 2 are required")
        self.cameras = validate_cameras(self.cameras)
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError("image_size must contain height and width")
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("dtype must be float32 or bfloat16")

    def to_dict(self):
        return asdict(self)
