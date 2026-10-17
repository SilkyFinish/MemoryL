# MLPResNet adapted from EventVLA (2025 eventvla community, Jinhui YE),
# MIT-licensed MLP_ActionHeader.py, itself adapted from OpenVLA-OFT.
"""Action heads consuming post-retrieval cognition and perception tokens."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class MLPResNetBlock(nn.Module):
    """EventVLA's pre-normalized LayerNorm/Linear/ReLU residual block."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.ffn(features) + features


class MLPResNet(nn.Module):
    """Preserve the structure and parameter names of EventVLA MLPResNet."""

    def __init__(self, num_blocks: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList([MLPResNetBlock(dim=hidden_dim) for _ in range(num_blocks)])
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.relu(self.fc1(self.layer_norm1(features)))
        for residual_block in self.mlp_resnet_blocks:
            features = residual_block(features)
        return self.fc2(self.layer_norm2(features))


def _expand_valid_mask(actions: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if valid_mask.shape == actions.shape[:2]:
        valid_mask = valid_mask.unsqueeze(-1).expand_as(actions)
    elif valid_mask.shape != actions.shape:
        raise ValueError(
            f"valid_mask must have shape {tuple(actions.shape[:2])} or {tuple(actions.shape)}, "
            f"got {tuple(valid_mask.shape)}"
        )
    return valid_mask.to(device=actions.device, dtype=torch.bool)


def _validate_actions(actions: torch.Tensor, batch_size: int, horizon: int, action_dim: int):
    expected_shape = (batch_size, horizon, action_dim)
    if tuple(actions.shape) != expected_shape:
        raise ValueError(f"actions must have shape {expected_shape}, got {tuple(actions.shape)}")


def _validate_condition_tokens(
    cognition: torch.Tensor, perception: torch.Tensor, cognition_dim: int, perception_dim: int
):
    for name, tokens, feature_dim in (
        ("cognition", cognition, cognition_dim),
        ("perception", perception, perception_dim),
    ):
        if tokens.ndim != 3 or tokens.shape[-1] != feature_dim or tokens.shape[1] < 1:
            raise ValueError(f"{name} must be nonempty [batch, tokens, {feature_dim}], got {tuple(tokens.shape)}")
    if cognition.shape[0] != perception.shape[0]:
        raise ValueError("cognition and perception must have the same batch size")


class OFTHead(nn.Module):
    """Learned horizon queries read each bank separately, then predict with L1.

    The learned slot residual keeps future slots distinct even when a bank has
    only one token (in that case attention alone cannot distinguish queries).
    """

    def __init__(self, cognition_dim: int, perception_dim: int, horizon: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        if min(cognition_dim, perception_dim, horizon, action_dim, hidden_dim) < 1:
            raise ValueError("Head dimensions and horizon must be positive")
        self.cognition_dim = cognition_dim
        self.perception_dim = perception_dim
        self.horizon = horizon
        self.action_dim = action_dim
        self.horizon_slots = nn.Embedding(horizon, hidden_dim)
        self.cognition_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=1, kdim=cognition_dim, vdim=cognition_dim, batch_first=True
        )
        self.perception_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=1, kdim=perception_dim, vdim=perception_dim, batch_first=True
        )
        self.slot_fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.model = MLPResNet(num_blocks=2, input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=action_dim)

    def predict(self, cognition: torch.Tensor, perception: torch.Tensor) -> torch.Tensor:
        _validate_condition_tokens(cognition, perception, self.cognition_dim, self.perception_dim)
        batch_size = cognition.shape[0]
        horizon_queries = self.horizon_slots.weight.unsqueeze(0).expand(batch_size, -1, -1)
        cognition_readout, _ = self.cognition_attention(horizon_queries, cognition, cognition, need_weights=False)
        perception_readout, _ = self.perception_attention(horizon_queries, perception, perception, need_weights=False)
        fused_slots = self.slot_fusion(torch.cat([cognition_readout, perception_readout], dim=-1)) + horizon_queries
        return self.model(fused_slots)

    def forward(self, cognition: torch.Tensor, perception: torch.Tensor) -> torch.Tensor:
        return self.predict(cognition, perception)

    def loss(
        self,
        actions: torch.Tensor,
        cognition: torch.Tensor,
        perception: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _validate_actions(actions, cognition.shape[0], self.horizon, self.action_dim)
        predicted_actions = self.predict(cognition, perception)
        if valid_mask is None:
            return (predicted_actions - actions).abs().mean()
        action_mask = _expand_valid_mask(actions, valid_mask)
        # Mask before arithmetic: NaN padding must not enter the backward graph.
        safe_actions = torch.where(action_mask, actions, torch.zeros_like(actions))
        action_difference = torch.where(action_mask, predicted_actions - safe_actions, 0.0)
        return action_difference.abs().sum() / action_mask.sum().clamp_min(1)


class DiffusionHead(nn.Module):
    """Wrap the unchanged ActionModel, including its original CFG/DDIM path.

    ActionModel conditions on one cognition token, as in MemoryVLA; perception
    may contain any nonempty token sequence. Imports of DiT/timm are deferred
    until this head is constructed, so OFT and memory-bank imports stay light.
    """

    def __init__(
        self,
        cognition_dim: int,
        perception_dim: int,
        horizon: int,
        action_dim: int,
        model_type: str = "DiT-S",
        ddim_steps: int = 10,
        cfg_scale: float = 1.5,
    ):
        super().__init__()
        if min(cognition_dim, perception_dim, horizon, action_dim, ddim_steps) < 1:
            raise ValueError("Head dimensions, horizon, and ddim_steps must be positive")
        from action_model.action_model import ActionModel

        self.cognition_dim = cognition_dim
        self.perception_dim = perception_dim
        self.horizon = horizon
        self.action_dim = action_dim
        self.ddim_steps = ddim_steps
        self.cfg_scale = cfg_scale
        self.action_model = ActionModel(
            token_size=cognition_dim,
            model_type=model_type,
            in_channels=action_dim,
            future_action_window_size=horizon - 1,
            use_per_attn=True,
            per_token_size=perception_dim,
        )

    def _validate_conditions(self, cognition: torch.Tensor, perception: torch.Tensor):
        _validate_condition_tokens(cognition, perception, self.cognition_dim, self.perception_dim)
        if cognition.shape[1] != 1:
            raise ValueError("The original diffusion ActionModel requires exactly one cognition token per sample")

    def loss(
        self,
        actions: torch.Tensor,
        cognition: torch.Tensor,
        perception: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._validate_conditions(cognition, perception)
        _validate_actions(actions, cognition.shape[0], self.horizon, self.action_dim)
        if valid_mask is None:
            return self.action_model.loss(actions, cognition, perception)

        action_mask = _expand_valid_mask(actions, valid_mask)
        safe_actions = torch.where(action_mask, actions, torch.zeros_like(actions))
        # Preserve the original RNG order and noise-prediction objective.
        noise = torch.randn_like(safe_actions)
        diffusion_timesteps = torch.randint(
            0, self.action_model.diffusion.num_timesteps, (safe_actions.size(0),), device=safe_actions.device
        )
        noised_actions = self.action_model.diffusion.q_sample(safe_actions, diffusion_timesteps, noise)
        predicted_noise = self.action_model.net(noised_actions, diffusion_timesteps, cognition, per_token=perception)
        assert predicted_noise.shape == noise.shape == safe_actions.shape
        noise_difference = torch.where(action_mask, predicted_noise - noise, 0.0)
        return noise_difference.square().sum() / action_mask.sum().clamp_min(1)

    @torch.inference_mode()
    def predict(self, cognition: torch.Tensor, perception: torch.Tensor) -> torch.Tensor:
        self._validate_conditions(cognition, perception)
        batch_size = cognition.shape[0]
        model_dtype = next(self.action_model.net.parameters()).dtype
        noise = torch.randn(batch_size, self.horizon, self.action_dim, device=cognition.device).to(model_dtype)
        using_cfg = self.cfg_scale > 1.0
        if using_cfg:
            noise = torch.cat([noise, noise], dim=0)
            unconditional_tokens = self.action_model.net.z_embedder.uncondition.unsqueeze(0)
            unconditional_tokens = unconditional_tokens.expand(batch_size, *unconditional_tokens.shape[1:])
            condition_tokens = torch.cat([cognition, unconditional_tokens], dim=0)
            model_kwargs = {
                "z": condition_tokens,
                "cfg_scale": self.cfg_scale,
                "per_token": perception.repeat(2, 1, 1),
            }
            sample_function = self.action_model.net.forward_with_cfg
        else:
            model_kwargs = {"z": cognition, "per_token": perception}
            sample_function = self.action_model.net.forward

        if self.action_model.ddim_diffusion is None:
            self.action_model.create_ddim(ddim_step=self.ddim_steps)
        samples = self.action_model.ddim_diffusion.ddim_sample_loop(
            sample_function,
            noise.shape,
            noise,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=cognition.device,
            eta=0.0,
        )
        return samples.chunk(2, dim=0)[0] if using_cfg else samples

    def forward(self, cognition: torch.Tensor, perception: torch.Tensor) -> torch.Tensor:
        return self.predict(cognition, perception)
