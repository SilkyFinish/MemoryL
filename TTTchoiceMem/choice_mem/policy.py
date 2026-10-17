"""Shared explicit-memory policy independent of representation/action family."""

from torch import nn

from .backbones import MemoryVLABackbone, Qwen35Backbone
from .banks import CogMemBank, PerMemBank
from .compression import GridBottleneckSE
from .config import PolicyConfig
from .heads import DiffusionHead, OFTHead


class ChoiceMemoryPolicy(nn.Module):
    def __init__(self, config: PolicyConfig, backbone=None, initialize_from_config=False):
        super().__init__()
        self.config = config
        if backbone is None:
            backbone_class = Qwen35Backbone if config.backbone == "qwen35" else MemoryVLABackbone
            backbone = backbone_class(config, initialize_from_config=initialize_from_config)
        self.backbone = backbone
        dimensions = {"vision": backbone.vision_dim, "cognition": backbone.cognition_dim}
        if config.feature_dimensions and config.feature_dimensions != dimensions:
            raise ValueError("Actual backbone dimensions differ from checkpoint")
        config.feature_dimensions = dimensions
        self.per_compr = GridBottleneckSE(backbone.vision_dim, config.per_token_size * 2, config.per_token_size)
        memory_arguments = dict(
            dataloader_type="group",
            group_size=config.group_size,
            mem_length=config.mem_length,
            retrieval_layers=config.retrieval_layers,
            use_timestep_pe=config.use_timestep_pe,
            fusion_type=config.fusion_type,
            consolidate_type=config.consolidate_type,
            update_fused=config.update_fused,
        )
        self.cog_mem_bank = CogMemBank(token_size=backbone.cognition_dim, **memory_arguments)
        self.per_mem_bank = PerMemBank(token_size=config.per_token_size, **memory_arguments)
        head_arguments = dict(
            cognition_dim=backbone.cognition_dim,
            perception_dim=config.per_token_size,
            horizon=config.horizon,
            action_dim=config.action_dim,
        )
        self.action_head = (
            OFTHead(**head_arguments, hidden_dim=config.oft_hidden_dim)
            if config.action_head == "oft"
            else DiffusionHead(
                **head_arguments,
                model_type=config.diffusion_model_type,
                ddim_steps=config.ddim_steps,
                cfg_scale=config.cfg_scale,
            )
        )
        self.episode_id = None
        self.query_timestep = 0
        self.camera_layout = None

    def reset(self, episode_id=None):
        self.cog_mem_bank.reset()
        self.per_mem_bank.reset()
        self.backbone.reset()
        self.episode_id = episode_id
        self.query_timestep = 0
        self.camera_layout = None

    def read_features(self, image_batches, instructions, episode_ids, timesteps):
        if not (len(image_batches) == len(instructions) == len(episode_ids) == len(timesteps)):
            raise ValueError("Images, instructions, episode IDs and timesteps must have matching batch lengths")
        features = self.backbone(image_batches, instructions)
        if self.camera_layout is not None and features.camera_grids != self.camera_layout:
            raise ValueError("Camera token grid changed inside an episode; reset or use fixed preprocessing")
        self.camera_layout = features.camera_grids
        parameter_dtype = next(self.per_compr.parameters()).dtype
        perception = self.per_compr(features.visual_tokens.to(parameter_dtype), features.camera_grids)
        cognition = features.cognition.to(next(self.cog_mem_bank.parameters()).dtype)
        cognition = self.cog_mem_bank.process_batch(cognition, episode_ids, timesteps)
        perception = self.per_mem_bank.process_batch(perception, episode_ids, timesteps)
        return cognition, perception

    def forward(self, image_batches, instructions, actions, valid_mask, episode_ids, timesteps):
        if tuple(actions.shape[1:]) != (self.config.horizon, self.config.action_dim):
            raise ValueError("Action labels do not match checkpoint horizon/action_dim")
        cognition, perception = self.read_features(image_batches, instructions, episode_ids, timesteps)
        return self.action_head.loss(actions, cognition, perception, valid_mask)

    def predict(self, images, instruction, episode_id, timestep=None):
        if self.training:
            raise RuntimeError("Call eval() before episode inference")
        if self.episode_id is None:
            self.reset(episode_id)
        elif self.episode_id != episode_id:
            raise ValueError("A new episode requires an explicit reset; refusing history leakage")
        timestep = self.query_timestep * self.config.execute_prefix if timestep is None else timestep
        cognition, perception = self.read_features([images], [instruction], [episode_id], [timestep])
        self.query_timestep += 1
        return self.action_head.predict(cognition, perception)

    def parameter_counts(self):
        return {
            "total": sum(parameter.numel() for parameter in self.parameters()),
            "trainable": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
            "backbone": sum(parameter.numel() for parameter in self.backbone.parameters()),
            "memory_and_head": sum(
                parameter.numel() for name, parameter in self.named_parameters() if not name.startswith("backbone.")
            ),
        }
