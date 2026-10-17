"""Current-query features, without a second implicit temporal memory."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from .config import PolicyConfig


@dataclass
class FeatureBatch:
    visual_tokens: torch.Tensor
    cognition: torch.Tensor
    camera_grids: tuple[tuple[int, int, int], ...]


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_images(image_batches, config):
    """Use the same RGB/camera geometry protocol for either backbone."""
    height, width = config.image_size
    prepared = []
    for image_batch in image_batches:
        if len(image_batch) != len(config.cameras):
            raise ValueError("Every query must contain the checkpoint's ordered cameras")
        cameras = []
        for image in image_batch:
            image = (image if isinstance(image, Image.Image) else Image.fromarray(image)).convert("RGB")
            if image.size != (width, height):
                image = image.resize((width, height), resample=Image.Resampling.BILINEAR)
            cameras.append(image)
        prepared.append(cameras)
    return prepared


class Qwen35Backbone(nn.Module):
    def __init__(self, config: PolicyConfig, initialize_from_config=False):
        super().__init__()
        import transformers
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig, AutoProcessor, Qwen3_5Model

        self.policy_config = config
        source_path = Path(config.model_source)
        if not source_path.is_dir():
            source_path = Path(
                snapshot_download(
                    config.model_source,
                    revision=config.model_revision,
                    allow_patterns=["*.json", "*.jinja", "*.safetensors", "*.txt", "*.model"],
                )
            )
            config.model_revision = source_path.name
        elif config.model_revision is None:
            config.model_revision = "local-sha256"

        source_path = source_path.resolve()
        self.processor = AutoProcessor.from_pretrained(source_path, local_files_only=True, trust_remote_code=False)
        backbone_config = AutoConfig.from_pretrained(source_path, local_files_only=True, trust_remote_code=False)
        if backbone_config.model_type != "qwen3_5":
            raise ValueError("qwen35 requires a multimodal Qwen3.5 checkpoint, not a text-only Qwen model")
        backbone_config.text_config.use_cache = False
        model_dtype = getattr(torch, config.dtype)
        if initialize_from_config:
            self.model = Qwen3_5Model(backbone_config).to(dtype=model_dtype)
        else:
            self.model = Qwen3_5Model.from_pretrained(
                source_path,
                config=backbone_config,
                local_files_only=True,
                dtype=model_dtype,
                attn_implementation="sdpa",
            )
        self.vision_dim = int(backbone_config.vision_config.out_hidden_size)
        self.cognition_dim = int(backbone_config.text_config.hidden_size)
        self.merge_size = int(backbone_config.vision_config.spatial_merge_size)
        self.model.requires_grad_(not config.freeze_backbone)
        self.model.eval() if config.freeze_backbone else self.model.train()

        manifest = {
            str(asset.relative_to(source_path)): hash_file(asset)
            for asset in sorted(source_path.iterdir())
            if asset.suffix in {".json", ".jinja", ".safetensors", ".txt", ".model"} and asset.is_file()
        }
        if config.asset_manifest and config.asset_manifest != manifest:
            raise ValueError("Qwen source assets differ from checkpoint SHA256 manifest")
        config.asset_manifest = manifest
        preprocessing = {
            "backend": "qwen35-official-processor",
            "transformers": transformers.__version__,
            "processor": self.processor.to_dict() if hasattr(self.processor, "to_dict") else {},
            "image_processor": self.processor.image_processor.to_dict(),
            "shared_resize": {
                "height": config.image_size[0],
                "width": config.image_size[1],
                "interpolation": "bilinear",
            },
            "prompt": "camera-names-images-task-action-query; no assistant generation",
            "use_cache": False,
            "vision_features": "own-vision-encoder-merged-pooler-output",
        }
        # Convert enums/array values into a stable checkpoint-safe JSON representation.
        preprocessing = json.loads(json.dumps(preprocessing, default=str))
        if config.preprocessing and config.preprocessing != preprocessing:
            raise ValueError("Qwen processor/runtime preprocessing differs from checkpoint")
        config.preprocessing = preprocessing

    def train(self, mode=True):
        super().train(mode)
        if self.policy_config.freeze_backbone:
            self.model.eval()
        return self

    def reset(self):
        self.model.rope_deltas = None

    def forward(self, image_batches, instructions):
        image_batches = prepare_images(image_batches, self.policy_config)
        conversations = []
        for images, instruction in zip(image_batches, instructions):
            content = []
            for camera_name, image in zip(self.policy_config.cameras, images):
                content.extend([{"type": "text", "text": camera_name + ":"}, {"type": "image", "image": image}])
            content.append({"type": "text", "text": f"Task: {instruction}\nWhat action should the robot take next?"})
            conversations.append([{"role": "user", "content": content}])
        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )
        device = next(self.model.parameters()).device
        inputs = {name: value.to(device) if torch.is_tensor(value) else value for name, value in inputs.items()}
        grids = inputs["image_grid_thw"].detach().cpu().tolist()
        camera_count = len(self.policy_config.cameras)
        layouts = [
            tuple(tuple(int(value) for value in grid) for grid in grids[start : start + camera_count])
            for start in range(0, len(grids), camera_count)
        ]
        if len(layouts) != len(image_batches) or any(layout != layouts[0] for layout in layouts):
            raise ValueError("Queries must have identical per-camera processor grids for token-merge memory")
        merged_grids = tuple(
            (temporal, height // self.merge_size, width // self.merge_size) for temporal, height, width in layouts[0]
        )
        captured_visual = []

        def capture_visual(_module, _arguments, output):
            captured_visual.append(output.pooler_output)

        hook = self.model.visual.register_forward_hook(capture_visual)
        self.reset()
        try:
            with torch.set_grad_enabled(self.training and not self.policy_config.freeze_backbone):
                output = self.model(**inputs, use_cache=False, past_key_values=None, return_dict=True)
        finally:
            hook.remove()
            self.reset()
        if len(captured_visual) != 1:
            raise RuntimeError("Expected exactly one call to Qwen's own vision encoder")
        visual = captured_visual[0]
        if isinstance(visual, (tuple, list)):
            visual = torch.cat(visual, dim=0)
        tokens_per_query = sum(temporal * height * width for temporal, height, width in merged_grids)
        visual = visual.reshape(len(image_batches), tokens_per_query, self.vision_dim)
        positions = torch.arange(inputs["attention_mask"].shape[1], device=device)
        last_positions = (
            positions.expand_as(inputs["attention_mask"])
            .masked_fill(~inputs["attention_mask"].bool(), -1)
            .max(dim=1)
            .values
        )
        cognition = output.last_hidden_state[torch.arange(len(image_batches), device=device), last_positions].unsqueeze(
            1
        )
        return FeatureBatch(visual, cognition, merged_grids)


class MemoryVLABackbone(nn.Module):
    """Adapt the original DINO+SigLIP/Prismatic model, loading no Qwen modules."""

    def __init__(self, config: PolicyConfig, initialize_from_config=False):
        super().__init__()
        self.policy_config = config
        if not config.model_source or str(config.model_source).startswith("Qwen/"):
            raise ValueError("MemoryVLA requires an explicit original VLM run, VLA checkpoint, or audited asset manifest")
        random_debug = config.backbone_initialization == "random-frozen-debug"
        if random_debug:
            self._initialize_random_debug(config)
            return
        source_path = Path(config.model_source).expanduser()
        if source_path.is_absolute() and not source_path.exists():
            raise FileNotFoundError(f"Original MemoryVLA assets do not exist: {source_path}")
        if source_path.is_file() and source_path.suffix == ".json":
            self.vlm, manifest = self.load_from_asset_manifest(source_path)
        elif source_path.is_file() and source_path.suffix == ".pt":
            self.vlm, manifest = self.load_from_vla_checkpoint(source_path)
        else:
            from vla.load import load

            self.vlm = load(config.model_source, load_for_training=False)
            manifest = {}
            if source_path.is_dir():
                for relative_path in ["config.json", "checkpoints/latest-checkpoint.pt"]:
                    manifest[relative_path] = hash_file(source_path / relative_path)
        self.vlm.freeze_backbones("align" if config.freeze_backbone else "full-finetune")
        if config.freeze_backbone:
            self.vlm.requires_grad_(False)
        self.vlm.to(dtype=getattr(torch, config.dtype))
        self.vision_dim = int(self.vlm.vision_backbone.embed_dim)
        self.cognition_dim = int(self.vlm.llm_backbone.embed_dim)
        if config.asset_manifest and config.asset_manifest != manifest:
            raise ValueError("MemoryVLA base checkpoint differs from recorded source SHA256")
        config.asset_manifest = manifest
        config.model_revision = config.model_revision or "local-sha256"
        preprocessing = {
            "backend": "memoryvla-original-image-transform",
            "shared_resize": config.image_size,
            "image_transform": repr(self.vlm.vision_backbone.image_transform),
            "prompt": "original-prismatic-human-action-question-plus-29871-2",
            "use_cache": False,
        }
        if config.preprocessing and config.preprocessing != preprocessing:
            raise ValueError("MemoryVLA preprocessing differs from checkpoint")
        config.preprocessing = preprocessing

    def _initialize_random_debug(self, config):
        """Build the original registry architecture using audited metadata, not weights."""
        # Recheck the strict recipe even if a caller mutated a dataclass after creation.
        config.__post_init__()
        from .random_init import inspect_random_backbone_assets

        assets = inspect_random_backbone_assets(config.model_source)
        manifest = assets["asset_manifest"]
        if config.asset_manifest and config.asset_manifest != manifest:
            raise ValueError("Random MemoryVLA metadata differs from recorded source SHA256")
        llama_config = assets["llama_config"]
        original_shape = {
            "hidden_size": 4096, "intermediate_size": 11008,
            "num_hidden_layers": 32, "num_attention_heads": 32, "vocab_size": 32000,
        }
        if any(llama_config.get(key) != value for key, value in original_shape.items()):
            raise ValueError("random-frozen-debug requires original Llama-2 7B config dimensions")
        if llama_config.get("num_key_value_heads", 32) != 32:
            raise ValueError("random-frozen-debug requires original Llama-2 7B attention heads")

        # Imports and all model allocation happen only after the metadata audit.
        import random
        from unittest.mock import patch

        import numpy as np

        from prismatic.conf import ModelConfig
        from prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
        from prismatic.models.vlms.prismatic import PrismaticVLM

        model_config = ModelConfig.get_choice_class(config.base_vlm_id)()
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            # Constructors run on CPU. PrismaticVLM calls torch.manual_seed for
            # its original projector rule; restrict that call to the CPU generator
            # so it cannot reseed CUDA or enqueue a lazy CUDA seed as a side effect.
            # fork_rng restores the caller's CPU state even if construction fails.
            with torch.random.fork_rng(devices=[]), torch.device("cpu"), patch(
                "torch.manual_seed", torch.random.default_generator.manual_seed
            ):
                torch.manual_seed(config.backbone_initialization_seed)
                random.seed(config.backbone_initialization_seed)
                np.random.seed(config.backbone_initialization_seed % 2**32)
                vision, _ = get_vision_backbone_and_transform(
                    model_config.vision_backbone_id, model_config.image_resize_strategy, pretrained=False
                )
                vision.to(dtype=torch.bfloat16)
                language, _ = get_llm_backbone_and_tokenizer(
                    model_config.llm_backbone_id,
                    llm_max_length=model_config.llm_max_length,
                    inference_mode=True,
                    hf_hub_path=assets["metadata_path"],
                    local_files_only=True,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                    use_cache=False,
                )
                self.vlm = PrismaticVLM(
                    model_config.model_id, vision, language, arch_specifier=model_config.arch_specifier
                )
                self.vlm.to(dtype=torch.bfloat16)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        self.vlm.requires_grad_(False)
        self.vlm.eval()
        self.vision_dim = int(self.vlm.vision_backbone.embed_dim)
        self.cognition_dim = int(self.vlm.llm_backbone.embed_dim)
        dimensions = {"vision": self.vision_dim, "cognition": self.cognition_dim}
        if config.feature_dimensions and config.feature_dimensions != dimensions:
            raise ValueError("Actual backbone dimensions differ from checkpoint")
        recipe = {
            "mode": "random-frozen-debug", "recipe_version": 1,
            "base_vlm_id": config.base_vlm_id, "seed": config.backbone_initialization_seed,
            "vision_backbone_id": model_config.vision_backbone_id,
            "llm_backbone_id": model_config.llm_backbone_id,
            "image_resize_strategy": model_config.image_resize_strategy,
            "arch_specifier": model_config.arch_specifier, "llm_max_length": model_config.llm_max_length,
            "vision_pretrained": False, "llm_initialization": "local-config-only",
            "weights_loaded": False, "dtype": "bfloat16", "attention": "sdpa", "use_cache": False,
            "projector_seed_rule": "torch.manual_seed(vision_backbone.embed_dim)",
            "projector_seed": self.vision_dim, "torch_version": str(torch.__version__),
        }
        revision_payload = json.dumps({"recipe": recipe, "assets": manifest}, sort_keys=True).encode()
        revision = "random-frozen-debug-v1-sha256:" + hashlib.sha256(revision_payload).hexdigest()
        if config.model_revision is not None and config.model_revision != revision:
            raise ValueError("Random MemoryVLA recipe differs from recorded model_revision")
        preprocessing = {
            "backend": "memoryvla-original-image-transform",
            "shared_resize": config.image_size,
            "image_transform": repr(self.vlm.vision_backbone.image_transform),
            "prompt": "original-prismatic-human-action-question-plus-29871-2",
            "use_cache": False,
            "backbone_initialization": recipe,
        }
        if config.preprocessing and config.preprocessing != preprocessing:
            raise ValueError("MemoryVLA preprocessing differs from checkpoint")
        config.asset_manifest = manifest
        config.model_revision = revision
        config.preprocessing = preprocessing
        config.feature_dimensions = dimensions

    @staticmethod
    def load_from_vla_checkpoint(checkpoint_path):
        """Reuse only VLM modules from an upstream VLA run, on CPU.

        Its 7D action head, banks and action statistics are intentionally not
        loaded: the RMBench policy initializes its own 14D modules and statistics.
        Use the audited manifest path for fully local vision/tokenizer assets.
        """
        configuration_path = checkpoint_path.parent.parent / "config.json"
        if not configuration_path.is_file():
            raise FileNotFoundError(f"Original VLA checkpoint requires run/config.json: {configuration_path}")
        configuration = json.loads(configuration_path.read_text())
        if "base_vlm" not in configuration.get("vla", {}):
            raise ValueError("Original VLA config.json must contain vla.base_vlm")
        from prismatic.conf import ModelConfig
        from prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
        from prismatic.models.vlms.prismatic import PrismaticVLM

        model_config = ModelConfig.get_choice_class(configuration["vla"]["base_vlm"])()
        vision, _ = get_vision_backbone_and_transform(
            model_config.vision_backbone_id, model_config.image_resize_strategy
        )
        language, _ = get_llm_backbone_and_tokenizer(
            model_config.llm_backbone_id, llm_max_length=model_config.llm_max_length, inference_mode=True
        )
        vlm = PrismaticVLM(model_config.model_id, vision, language, arch_specifier=model_config.arch_specifier)
        state = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)["model"]
        if not {"projector", "llm_backbone"}.issubset(state):
            raise ValueError("Original checkpoint lacks the required VLM modules")
        for name in ("projector", "llm_backbone", "vision_backbone"):
            if name in state:
                getattr(vlm, name).load_state_dict(state[name], strict=True)
        return vlm, {"config.json": hash_file(configuration_path), "checkpoint": hash_file(checkpoint_path)}

    @staticmethod
    def load_from_asset_manifest(manifest_path):
        """Reuse locally audited VLM weights, never import its old banks/head."""
        import tensorflow as tensorflow

        tensorflow.config.set_visible_devices([], "GPU")
        import timm

        from prismatic.conf import ModelConfig
        from prismatic.models.backbones.llm.llama2 import LLAMA2_MODELS
        from prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
        from prismatic.models.vlms.prismatic import PrismaticVLM

        source_manifest = json.loads(manifest_path.read_text())
        checkpoint_path = Path(source_manifest["checkpoint_path"])
        recorded_hashes = {"manifest": hash_file(manifest_path), "checkpoint": hash_file(checkpoint_path)}
        if recorded_hashes["checkpoint"] != source_manifest["checkpoint_sha256"]:
            raise ValueError("Original MemoryVLA source checkpoint SHA256 mismatch")
        source_config_path = Path(source_manifest["config.json"]["path"])
        if hash_file(source_config_path) != source_manifest["config.json"]["sha256"]:
            raise ValueError("Original MemoryVLA source config SHA256 mismatch")
        source_config = json.loads(source_config_path.read_text())
        model_config = ModelConfig.get_choice_class(source_config["vla"]["base_vlm"])()
        previous_llama_source = LLAMA2_MODELS[model_config.llm_backbone_id]["hf_hub_path"]
        original_factory = timm.create_model

        def create_local_visual(model_name, *arguments, **keywords):
            if model_name in source_manifest["local_vision_files"] and keywords.get("pretrained"):
                keywords["pretrained_cfg_overlay"] = {"file": source_manifest["local_vision_files"][model_name]}
            return original_factory(model_name, *arguments, **keywords)

        LLAMA2_MODELS[model_config.llm_backbone_id]["hf_hub_path"] = source_manifest["llama_metadata"]["path"]
        timm.create_model = create_local_visual
        try:
            vision, _transform = get_vision_backbone_and_transform(
                model_config.vision_backbone_id,
                model_config.image_resize_strategy,
            )
            language, _tokenizer = get_llm_backbone_and_tokenizer(
                model_config.llm_backbone_id,
                llm_max_length=model_config.llm_max_length,
                inference_mode=True,
            )
            vlm = PrismaticVLM(
                model_config.model_id,
                vision,
                language,
                arch_specifier=model_config.arch_specifier,
            )
            state = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)["model"]
            for module_name in ("projector", "llm_backbone", "vision_backbone"):
                if module_name in state:
                    getattr(vlm, module_name).load_state_dict(state[module_name], strict=True)
            if not {"projector", "llm_backbone"}.issubset(state):
                raise ValueError("Source checkpoint lacks original VLM weights")
        finally:
            LLAMA2_MODELS[model_config.llm_backbone_id]["hf_hub_path"] = previous_llama_source
            timm.create_model = original_factory
        for model_name, asset_path in source_manifest["local_vision_files"].items():
            recorded_hashes[model_name] = hash_file(Path(asset_path))
        for file_name, expected_hash in source_manifest["llama_metadata"]["files"].items():
            recorded_hashes["llama/" + file_name] = hash_file(
                Path(source_manifest["llama_metadata"]["path"]) / file_name
            )
            if recorded_hashes["llama/" + file_name] != expected_hash:
                raise ValueError(f"Original Llama metadata SHA256 mismatch: {file_name}")
        return vlm, recorded_hashes

    def train(self, mode=True):
        super().train(mode)
        if self.policy_config.freeze_backbone:
            self.vlm.eval()
        return self

    def reset(self):
        self.vlm.vision_feats = None

    def enable_gradient_checkpointing(self):
        self.vlm.llm_backbone.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def forward(self, image_batches, instructions):
        image_batches = prepare_images(image_batches, self.policy_config)
        device = next(self.vlm.parameters()).device
        model_dtype = next(self.vlm.parameters()).dtype
        visual_batches, cognition_batches = [], []
        grid_layout = None
        for images, instruction in zip(image_batches, instructions):
            builder = self.vlm.get_prompt_builder()
            builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
            input_ids = self.vlm.llm_backbone.tokenizer(builder.get_prompt(), return_tensors="pt").input_ids.to(device)
            input_ids = torch.cat([input_ids, input_ids.new_tensor([[29871, 2]])], dim=1)
            camera_features, camera_grids = [], []
            with torch.set_grad_enabled(self.training and not self.policy_config.freeze_backbone):
                for image in images:
                    pixels = self.vlm.vision_backbone.image_transform(image)
                    pixels = (
                        {name: value.unsqueeze(0).to(device, dtype=model_dtype) for name, value in pixels.items()}
                        if isinstance(pixels, dict)
                        else pixels.unsqueeze(0).to(device, dtype=model_dtype)
                    )
                    features = self.vlm.vision_backbone(pixels)
                    camera_features.append(features)
                    patch_embed = self.vlm.vision_backbone.siglip_featurizer.patch_embed
                    height, width = patch_embed.grid_size
                    if features.shape[1] != height * width:
                        raise ValueError("Original visual tokens do not match the actual patch grid")
                    camera_grids.append((1, height, width))
                visual = torch.cat(camera_features, dim=1)
                projected = self.vlm.projector(visual)
                text = self.vlm.llm_backbone.embed_input_ids(input_ids)
                multimodal = torch.cat([text[:, :1], projected, text[:, 1:]], dim=1)
                output = self.vlm.llm_backbone(
                    inputs_embeds=multimodal,
                    attention_mask=torch.ones(multimodal.shape[:2], device=device, dtype=torch.bool),
                    use_cache=False,
                    past_key_values=None,
                    output_hidden_states=True,
                    return_dict=True,
                )
            visual_batches.append(visual)
            cognition_batches.append(output.hidden_states[-1][:, -1:].contiguous())
            current_layout = tuple(camera_grids)
            if grid_layout is not None and current_layout != grid_layout:
                raise ValueError("Inconsistent camera grids")
            grid_layout = current_layout
        self.reset()
        return FeatureBatch(torch.cat(visual_batches), torch.cat(cognition_batches), grid_layout)
