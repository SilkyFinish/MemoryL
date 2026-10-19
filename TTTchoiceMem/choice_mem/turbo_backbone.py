"""VL-only replacement: original visual patches, BERT/Turbo cognition, no Llama."""

import hashlib
import json
from pathlib import Path
import random
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from .backbones import FeatureBatch, prepare_images
from .config import debug_parameter_dtype
from .turbo_assets import inspect_turbo_assets
from .turbo_modules import TurboTextEncoder, VisionLanguageInteraction, masked_mean_cognition


VISION_BACKBONE_ID = "dinosiglip-vit-so-224px"
IMAGE_RESIZE_STRATEGY = "resize-naive"
TURBO_SOURCE_COMMIT = "6727c875666f8d5dda8d8cca0043da200738fe73"


def build_original_vision(assets, *, pretrained):
    """Reuse the original factory/transform, loading only audited local vision files."""
    from prismatic.models.materialize import get_vision_backbone_and_transform

    if not pretrained:
        return get_vision_backbone_and_transform(
            VISION_BACKBONE_ID, IMAGE_RESIZE_STRATEGY, pretrained=False
        )[0]
    import timm

    original_factory, original_load = timm.create_model, torch.load
    paths = {name: str(Path(path).resolve(strict=True)) for name, path in assets["vision_files"].items()}
    allowed_paths = set(paths.values())

    def local_factory(model_name, *arguments, **keywords):
        if model_name not in paths:
            raise ValueError(f"Unexpected model in original-vision factory: {model_name}")
        if keywords.get("pretrained") is not True:
            raise ValueError("Pretrained original vision must explicitly load its audited weights")
        keywords["pretrained_cfg_overlay"] = {"file": paths[model_name]}
        return original_factory(model_name, *arguments, **keywords)

    def safe_local_load(path, *arguments, **keywords):
        if not isinstance(path, (str, Path)) or str(Path(path).resolve(strict=True)) not in allowed_paths:
            raise ValueError("Vision initialization attempted to read an unaudited weight file")
        from .turbo_initialization import validate_torch_weights_archive

        validate_torch_weights_archive(path)
        keywords["weights_only"] = True
        return original_load(path, *arguments, **keywords)

    # Keep timm's original model-specific filtering (including OpenCLIP SigLIP),
    # without allowing its legacy pickle loader to execute checkpoint objects.
    with patch("timm.create_model", local_factory), patch("torch.load", safe_local_load):
        return get_vision_backbone_and_transform(
            VISION_BACKBONE_ID, IMAGE_RESIZE_STRATEGY, pretrained=True
        )[0]


class TurboInteractionBackbone(TurboTextEncoder):
    """Keep Per tokens untouched; replace only the instruction-conditioned branch."""

    def __init__(self, config, initialize_from_config=False):
        config.__post_init__()
        if config.backbone != "turbo_interaction":
            raise ValueError("TurboInteractionBackbone requires backbone=turbo_interaction")
        assets = inspect_turbo_assets(config.model_source, config.backbone_initialization)
        if config.asset_manifest and config.asset_manifest != assets["asset_manifest"]:
            raise ValueError("Turbo assets differ from recorded source SHA256")
        debug = config.backbone_initialization == "turbo-random-frozen-debug"
        recipe = {
            **assets["recipe"], "mode": config.backbone_initialization,
            "seed": config.backbone_initialization_seed, "source_commit": TURBO_SOURCE_COMMIT,
            "vision_backbone_id": VISION_BACKBONE_ID, "image_resize_strategy": IMAGE_RESIZE_STRATEGY,
            "text_backbone": "bert-base-uncased",
            "text_projection_initialization": "random-linear" if debug else "groundingdino-feat-map",
            "interaction_initialization": "random" if debug else "groundingdino-encoder-subset-v1",
            "vision_projection_initialization": "random-linear",
            "vision_initialization_order": "first-with-independent-cpu-seed",
            "cognition_projection_initialization": "random-linear",
            "storage": "frozen-bf16-except-fp32-interaction" if debug else "float32-master",
        }
        revision_bytes = json.dumps({"recipe": recipe, "assets": assets["asset_manifest"]}, sort_keys=True).encode()
        revision = "turbo-vl-only-v1-sha256:" + hashlib.sha256(revision_bytes).hexdigest()
        if config.model_revision is not None and config.model_revision != revision:
            raise ValueError("Turbo initialization recipe differs from recorded model_revision")

        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            # No CUDA seeding or Llama/PrismaticVLM construction. Actual checkpoint
            # state, not this seed, is authoritative when resuming/deploying.
            with torch.random.fork_rng(devices=[]), torch.device("cpu"):
                torch.random.default_generator.manual_seed(config.backbone_initialization_seed)
                random.seed(config.backbone_initialization_seed)
                np.random.seed(config.backbone_initialization_seed % 2**32)
                # Match the original random-debug visual initialization before
                # consuming RNG for the replacement text/interaction modules.
                vision = build_original_vision(assets, pretrained=not (debug or initialize_from_config))
                super().__init__(
                    assets["bert_path"], random_init=debug,
                    initialize_from_config=initialize_from_config,
                )
                self.vision_backbone = vision
                self.vision_dim = int(self.vision_backbone.embed_dim)
                self.cognition_dim = 4096
                if self.vision_dim != 2176:
                    raise ValueError("Turbo VL-only replacement requires original 2176-wide DINO+SigLIP patches")
                self.vision_projection = nn.Linear(self.vision_dim, 256)
                self.vision_language_interaction = VisionLanguageInteraction()
                self.cognition_projection = nn.Linear(256, self.cognition_dim)
                self.initialization_report = {"mode": "random-config-only" if debug else "checkpoint-architecture-only"}
                if not debug and not initialize_from_config:
                    from .turbo_initialization import load_interaction_initialization

                    self.initialization_report = load_interaction_initialization(
                        self.vision_language_interaction, self.text_projection, assets["interaction_path"]
                    )
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        self.policy_config = config
        if debug:
            # Never round the fixed FP32 interaction profile through BF16.
            for name, child in self.named_children():
                child.to(dtype=getattr(torch, debug_parameter_dtype("backbone." + name + ".", config.backbone_initialization)))
        if config.freeze_backbone:
            self.requires_grad_(False)
        self.train(self.training)
        dimensions = {"vision": self.vision_dim, "cognition": self.cognition_dim}
        if config.feature_dimensions and config.feature_dimensions != dimensions:
            raise ValueError("Actual Turbo backbone dimensions differ from checkpoint")
        preprocessing = {
            "backend": "memoryvla-original-vision+turbo-vl-interaction",
            "shared_resize": config.image_size, "image_transform": repr(self.vision_backbone.image_transform),
            "prompt": "raw-instruction", "use_cache": False, "backbone_initialization": recipe,
        }
        if config.preprocessing and config.preprocessing != preprocessing:
            raise ValueError("Turbo preprocessing/profile differs from checkpoint")
        config.feature_dimensions = dimensions
        config.asset_manifest = assets["asset_manifest"]
        config.model_revision = revision
        config.preprocessing = preprocessing

    def train(self, mode=True):
        super().train(False if getattr(self, "policy_config", None) and self.policy_config.freeze_backbone else mode)
        return self

    def reset(self):
        # No token/vision/history cache: the policy's two banks own query history.
        pass

    def enable_gradient_checkpointing(self):
        if self.policy_config.freeze_backbone:
            raise ValueError("Gradient checkpointing conflicts with a frozen Turbo backbone")
        self.bert.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.bert.config.use_cache = False

    def forward(self, image_batches, instructions):
        if not image_batches or len(image_batches) != len(instructions):
            raise ValueError("Turbo images and instructions require matching nonempty batch lengths")
        images = prepare_images(image_batches, self.policy_config)
        device = next(self.vision_backbone.parameters()).device
        vision_dtype = next(self.vision_backbone.parameters()).dtype
        camera_grids, visual_batches = None, []
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.training and not self.policy_config.freeze_backbone):
            text, padding_mask, allowed_mask, content_mask = super().forward(instructions, device)
            for cameras in images:
                camera_features, current_grids = [], []
                for image in cameras:
                    pixels = self.vision_backbone.image_transform(image)
                    pixels = (
                        {name: value.unsqueeze(0).to(device=device, dtype=vision_dtype) for name, value in pixels.items()}
                        if isinstance(pixels, dict) else pixels.unsqueeze(0).to(device=device, dtype=vision_dtype)
                    )
                    features = self.vision_backbone(pixels)
                    height, width = self.vision_backbone.siglip_featurizer.patch_embed.grid_size
                    if features.ndim != 3 or features.shape[1:] != (height * width, self.vision_dim):
                        raise ValueError("Original visual tokens do not match their actual dense patch grid")
                    if tuple(self.vision_backbone.dino_featurizer.patch_embed.grid_size) != (height, width):
                        raise ValueError("DINO and SigLIP camera grids differ")
                    camera_features.append(features)
                    current_grids.append((1, height, width))
                layout = tuple(current_grids)
                if camera_grids is not None and camera_grids != layout:
                    raise ValueError("Inconsistent original camera grids")
                camera_grids = layout
                visual_batches.append(torch.cat(camera_features, dim=1))
            raw_visual = torch.cat(visual_batches, dim=0)
            projected = self.vision_projection(raw_visual.to(self.vision_projection.weight.dtype))
            _, fused_text = self.vision_language_interaction(projected, text, allowed_mask, padding_mask)
            pooled = masked_mean_cognition(fused_text, content_mask)
            cognition = self.cognition_projection(pooled.to(self.cognition_projection.weight.dtype))
        return FeatureBatch(raw_visual, cognition, camera_grids)
