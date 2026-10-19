"""Strict GroundingDINO initialization for the VL interaction subset only.

This is not a Turbo policy checkpoint loader. BERT, vision, detection, policy,
readout and memory weights are deliberately outside this initialization map.
No executable reference implementation is imported and no unsafe pickle or
TorchScript loading fallback is permitted.
"""

from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import os
import re
import stat
import zipfile

import torch


MAPPING_VERSION = "groundingdino-interaction-subset-v1"
SOURCE_FORMAT = "groundingdino-model-state-dict"
_PREFIXES = (
    ("transformer.encoder.fusion_layers.", "fusion_layers."),
    ("transformer.encoder.text_layers.", "text_layers."),
)
_MAX_CHECKPOINT_BYTES = 8 * 1024**3


@contextmanager
def _open_weights_archive(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_CHECKPOINT_BYTES:
            raise ValueError("Weight source must be a bounded regular file")
        if not zipfile.is_zipfile(stream):
            raise ValueError("Weight source requires a torch.save weights archive; nonzip legacy format is unsupported")
        with zipfile.ZipFile(stream) as archive:
            names = archive.namelist()
            roots = {name.split("/", 1)[0] for name in names}
            if len(roots) != 1:
                raise ValueError("Weight source has an invalid weights archive layout")
            root = next(iter(roots))
            relative = [name[len(root) + 1:] for name in names]
            allowed = re.compile(r"(?:data\.pkl|version|byteorder|\.data/serialization_id|data/[0-9]+)\Z")
            if (len(set(names)) != len(names) or "data.pkl" not in relative
                    or "version" not in relative
                    or any(allowed.fullmatch(name) is None for name in relative)):
                raise ValueError("Weight source must contain weights only, not compiled code")
        stream.seek(0)
        yield stream


def validate_torch_weights_archive(path):
    """Guard only: inspect a bounded local torch.save archive, never unpickle.

    Torch 2.4 can auto-dispatch TorchScript to jit.load despite weights_only=True.
    Unknown archive entries (including code/constants) and nonzip legacy formats
    are therefore rejected explicitly. This validates layout, not tensor contents
    or provenance; callers must still use weights_only=True and validate tensors.
    """
    with _open_weights_archive(path):
        pass


def _load_weights(path):
    # Validate and deserialize the same descriptor, avoiding a path replacement
    # between the archive check and torch.load. There is no unsafe fallback.
    with _open_weights_archive(path) as stream:
        return torch.load(stream, map_location="cpu", weights_only=True)


def _source_state(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("GroundingDINO source requires payload['model'] state dictionary")
    source = {}
    stripped = 0
    for key, value in payload["model"].items():
        if not isinstance(key, str):
            raise ValueError("GroundingDINO source state keys must be strings")
        normalized = key[7:] if key.startswith("module.") else key
        if normalized in source:
            raise ValueError(f"GroundingDINO source key collision after module. stripping: {normalized}")
        stripped += normalized != key
        source[normalized] = value
    return source, stripped


def _validate_tensor(key, value, target):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Selected GroundingDINO value is not a tensor: {key}")
    if not value.is_floating_point() or value.is_complex() or value.layout != torch.strided:
        raise ValueError(f"Selected GroundingDINO tensor requires a real floating strided dtype: {key}")
    if (not isinstance(target, torch.Tensor) or not target.is_floating_point()
            or target.is_complex() or target.layout != torch.strided or target.is_meta):
        raise ValueError(f"Initialization target requires a real floating materialized tensor: {key}")
    if value.shape != target.shape:
        raise ValueError(f"Selected GroundingDINO shape mismatch: {key}: {tuple(value.shape)} != {tuple(target.shape)}")
    if value.dtype != target.dtype:
        raise ValueError(f"Selected GroundingDINO dtype mismatch: {key}: {value.dtype} != {target.dtype}")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"Selected GroundingDINO tensor is nonfinite: {key}")


def _validate_keys(name, mapped, target):
    missing = sorted(target.keys() - mapped.keys())
    unexpected = sorted(mapped.keys() - target.keys())
    if missing or unexpected:
        raise ValueError(f"GroundingDINO {name} key mismatch: missing={missing}, unexpected={unexpected}")


def load_interaction_initialization(interaction, text_projection, path):
    """Validate all selected weights, then strictly load both target modules.

    ``interaction.state_dict()`` must contain only ``fusion_layers.*`` and
    ``text_layers.*``; ``text_projection`` must expose ``weight`` and ``bias``.
    Source suffixes are mapped exactly, never inferred from shapes or profiles.
    All selected keys, shapes, exact floating dtypes and finiteness are checked
    for *both* modules before either is mutated. Use float32 targets for the
    public float32 GroundingDINO source; precision conversion is a later step.
    Returns JSON-serializable subset initialization facts, not policy provenance.
    """
    source, stripped = _source_state(_load_weights(path))
    interaction_target = interaction.state_dict()
    projection_target = text_projection.state_dict()
    if (not interaction_target
            or any(not key.startswith(("fusion_layers.", "text_layers.")) for key in interaction_target)
            or not all(any(key.startswith(prefix) for key in interaction_target)
                       for prefix in ("fusion_layers.", "text_layers."))):
        raise ValueError("Interaction target must expose only nonempty fusion_layers and text_layers state")
    if set(projection_target) != {"weight", "bias"}:
        raise ValueError("Text projection target must expose exactly weight and bias")

    mapped_interaction, mapped_projection = {}, {}
    selected_counts = Counter()
    ignored_counts = Counter({"bert": 0, "vision_backbone": 0, "other_detection_or_unrelated": 0})
    for key, value in source.items():
        for source_prefix, target_prefix in _PREFIXES:
            if key.startswith(source_prefix):
                mapped_interaction[target_prefix + key[len(source_prefix):]] = value
                selected_counts[target_prefix[:-1]] += 1
                break
        else:
            if key.startswith("feat_map."):
                # Retain even unknown suffixes, so matching-prefix extras fail
                # strict validation rather than disappearing into ignored keys.
                mapped_projection[key[len("feat_map."):]] = value
                selected_counts["text_projection"] += 1
            else:
                group = ("bert" if key.startswith("bert.") else "vision_backbone"
                         if key.startswith("backbone.") else "other_detection_or_unrelated")
                ignored_counts[group] += 1

    _validate_keys("interaction", mapped_interaction, interaction_target)
    _validate_keys("text_projection", mapped_projection, projection_target)
    for key, value in mapped_interaction.items():
        _validate_tensor("interaction." + key, value, interaction_target[key])
    for key, value in mapped_projection.items():
        _validate_tensor("text_projection." + key, value, projection_target[key])

    # Do not manufacture a full target state filled with its initial values:
    # every tensor in these strict loads actually came from the source file.
    interaction.load_state_dict(mapped_interaction, strict=True)
    text_projection.load_state_dict(mapped_projection, strict=True)
    layer_counts = {
        name: len({key.split(".")[1] for key in mapped_interaction if key.startswith(name + ".")})
        for name in ("fusion_layers", "text_layers")
    }
    return {
        "mapping_version": MAPPING_VERSION,
        "source_format": SOURCE_FORMAT,
        "source_path": str(Path(path).absolute()),
        "scope": "interaction-and-text-projection-only",
        "loaded_module_count": 2,
        "loaded_layer_counts": layer_counts,
        "loaded_source_key_counts": dict(selected_counts),
        "loaded_source_key_count": len(mapped_interaction) + len(mapped_projection),
        "source_key_count": len(source),
        "module_prefix_stripped_count": stripped,
        "intentionally_ignored_source_key_count": sum(ignored_counts.values()),
        "intentionally_ignored_source_key_counts": dict(ignored_counts),
        "bert_loaded_from_source": False,
        "strict": True,
    }
