"""Torch-free local asset contract for the optional VL-only replacement.

This audits data, not executable reference source. It does not implement or load
TurboVLA. Debug follows BERT metadata only: vision, interaction and BERT weight
paths are not opened, hashed, or even statted in that mode.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import stat

from .random_init import _json_object, _read_small_file


ASSET_SCHEMA = "memoryvla-turbo-vl-assets-v1"
VISION_MODELS = frozenset({
    "vit_large_patch14_reg4_dinov2.lvd142m", "vit_so400m_patch14_siglip_224",
})
_METADATA_NAMES = frozenset({
    "config.json", "tokenizer_config.json", "tokenizer.json", "vocab.txt",
    "special_tokens_map.json", "added_tokens.json",
})
_WEIGHT_NAMES = frozenset({"model.safetensors", "pytorch_model.bin"})
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_MAX_METADATA_BYTES = 16 * 1024 * 1024
_MAX_TOTAL_METADATA_BYTES = 32 * 1024 * 1024

# The explicit adapter profile, not a claim of numerical equivalence to
# release checkpoints whose attention/residual/padding recipe is undocumented.
TURBO_INTERACTION_RECIPE = {
    "version": 1, "vision": "original-dinov2-siglip-raw-patches",
    "hidden_dim": 256, "configured_heads": 8, "effective_heads": 4,
    "layers": 6, "enhancer_inner_dim": 1024, "attention_backend": "manual",
    "residual_style": "normalized", "padding_strategy": "key_padding_mask",
    "interaction_compute_dtype": "float32", "fusion_droppath": 0.1,
    "text_max_length": 256, "text_input": "raw-instruction",
    "text_mask": "delimiter-subsentence-padding-safe-batch-major-heads",
    "cognition_readout": "masked-mean-excluding-padding-specials-and-delimiters",
    "cognition_dim": 4096, "use_cache": False,
}


def _digest(value, name):
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"Turbo asset requires a SHA256 digest: {name}")
    return value.lower()


def _absolute_path(value, name):
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError(f"Turbo asset requires an absolute local path: {name}")
    return Path(value)


def _hash_weight(record, name):
    if not isinstance(record, dict):
        raise ValueError(f"Turbo weight requires path and sha256: {name}")
    path = _absolute_path(record.get("path"), name)
    expected = _digest(record.get("sha256"), name)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    digest = hashlib.sha256()
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 8 * 1024**3:
            raise ValueError(f"Turbo weight must be a bounded regular file: {name}")
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"Turbo asset SHA256 mismatch: {name}")
    return str(path), expected


def inspect_turbo_assets(model_source, initialization="pretrained"):
    """Validate metadata before any model import; return only audited local paths.

    Manifest: ``schema``, ``bert: {path, files: {basename: sha256}}``;
    pretrained additionally requires ``vision: {timm_name: {path, sha256}}``
    and ``interaction: {path, sha256}``. The interaction file is a candidate
    initialization source, NOT a verified compatible model: tensor mapping,
    shapes, provenance and the implementation still require separate checks.
    """
    if initialization not in {"pretrained", "turbo-random-frozen-debug"}:
        raise ValueError("Turbo assets require pretrained or turbo-random-frozen-debug")
    debug = initialization == "turbo-random-frozen-debug"
    manifest_path = _absolute_path(model_source, "model_source")
    if manifest_path.suffix != ".json":
        raise ValueError("Turbo model_source must be a local manifest JSON")
    manifest_bytes = _read_small_file(manifest_path, 1024 * 1024)
    source = _json_object(manifest_bytes, manifest_path)
    if source.get("schema") != ASSET_SCHEMA:
        raise ValueError(f"Turbo asset manifest requires schema={ASSET_SCHEMA!r}")
    bert = source.get("bert")
    if not isinstance(bert, dict) or not isinstance(bert.get("files"), dict):
        raise ValueError("Turbo manifest requires bert.path and bert.files")
    bert_path = _absolute_path(bert.get("path"), "bert.path").resolve(strict=True)
    if not bert_path.is_dir():
        raise ValueError("bert.path must be a local directory")
    files = bert["files"]
    for name, expected in files.items():
        if name not in _METADATA_NAMES | _WEIGHT_NAMES:
            raise ValueError(f"Non-whitelisted BERT asset: {name!r}")
        _digest(expected, "bert/" + name)
    if not {"config.json", "tokenizer_config.json"}.issubset(files):
        raise ValueError("BERT metadata requires config.json and tokenizer_config.json")
    if not {"tokenizer.json", "vocab.txt"}.intersection(files):
        raise ValueError("BERT metadata requires tokenizer.json or vocab.txt")
    for name in ("chat_template.jinja", "chat_template.json", "chat_templates"):
        if os.path.lexists(bert_path / name):
            raise ValueError(f"Unsupported unaudited BERT sidecar: {name}")
    for name in _METADATA_NAMES - files.keys():
        if os.path.lexists(bert_path / name):
            raise ValueError(f"Undeclared BERT metadata file: {name}")
    hashes = {"manifest": hashlib.sha256(manifest_bytes).hexdigest()}
    metadata = {}
    total_bytes = 0
    for name in sorted(_METADATA_NAMES & files.keys()):
        content = _read_small_file(bert_path / name, _MAX_METADATA_BYTES)
        total_bytes += len(content)
        if total_bytes > _MAX_TOTAL_METADATA_BYTES:
            raise ValueError("BERT metadata exceeds total size limit")
        digest = hashlib.sha256(content).hexdigest()
        if digest != files[name].lower():
            raise ValueError(f"BERT metadata SHA256 mismatch: {name}")
        hashes["bert/" + name] = digest
        if name.endswith(".json"):
            metadata[name] = _json_object(content, bert_path / name)
    config = metadata["config.json"]
    expected_shape = {
        "model_type": "bert", "hidden_size": 768, "intermediate_size": 3072,
        "num_hidden_layers": 12, "num_attention_heads": 12, "vocab_size": 30522,
        "max_position_embeddings": 512, "type_vocab_size": 2, "pad_token_id": 0,
    }
    if any(type(config.get(key)) is not type(value) or config.get(key) != value
           for key, value in expected_shape.items()):
        raise ValueError("Turbo VL-only recipe requires BERT-base-uncased dimensions")
    if config.get("auto_map") or config.get("position_embedding_type", "absolute") != "absolute":
        raise ValueError("BERT requires standard local absolute-position architecture")
    tokenizer = metadata["tokenizer_config.json"]
    if tokenizer.get("tokenizer_class", "BertTokenizer") not in {"BertTokenizer", "BertTokenizerFast"}:
        raise ValueError("Turbo requires the standard BERT tokenizer")
    if tokenizer.get("auto_map") or tokenizer.get("do_lower_case", True) is not True:
        raise ValueError("Turbo requires local BERT uncased tokenization without custom code")
    for key in ("tokenizer_file", "vocab_file", "merges_file", "special_tokens_map_file", "added_tokens_file"):
        if tokenizer.get(key) is not None:
            raise ValueError(f"Unaudited BERT tokenizer file override: {key}")
    if tokenizer.get("fast_tokenizer_files"):
        raise ValueError("Versioned tokenizer overrides are not supported")

    vision_files, interaction_path = {}, None
    if not debug:
        # Only formal initialization follows weight paths. No reference code is
        # imported here, and hashing a candidate does not establish compatibility.
        for name in _WEIGHT_NAMES - files.keys():
            if os.path.lexists(bert_path / name):
                raise ValueError(f"Undeclared BERT weight file: {name}")
        weight_names = _WEIGHT_NAMES & files.keys()
        if not weight_names:
            raise ValueError("Pretrained Turbo requires declared local BERT weights")
        for name in sorted(weight_names):
            _, hashes["bert/" + name] = _hash_weight(
                {"path": str(bert_path / name), "sha256": files[name]}, "bert/" + name
            )
        vision = source.get("vision")
        if not isinstance(vision, dict) or set(vision) != VISION_MODELS:
            raise ValueError("Pretrained Turbo requires exactly the original DINOv2 and SigLIP vision assets")
        for name in sorted(VISION_MODELS):
            vision_files[name], hashes["vision/" + name] = _hash_weight(vision[name], name)
        interaction_path, hashes["interaction"] = _hash_weight(source.get("interaction"), "interaction")
    return {
        "asset_manifest": hashes, "bert_path": str(bert_path), "bert_config": config,
        "vision_files": vision_files, "interaction_path": interaction_path,
        "recipe": dict(TURBO_INTERACTION_RECIPE), "weights_audited": not debug,
    }
