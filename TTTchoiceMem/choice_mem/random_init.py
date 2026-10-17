"""Torch-free audit of the *metadata only* random-backbone recipe.

A full pretrained manifest is accepted, but only ``llama_metadata.path/files``
are followed. In particular, checkpoint and vision-weight paths are never even
statted. This module must remain importable by source-validation/eval tools
without importing torch, transformers, or the model package.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import stat


_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_METADATA_BYTES = 16 * 1024 * 1024
_MAX_TOTAL_METADATA_BYTES = 32 * 1024 * 1024
_METADATA_NAMES = frozenset({
    "config.json", "tokenizer_config.json", "tokenizer.json", "tokenizer.model",
    "special_tokens_map.json", "added_tokens.json",
})
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")


def _read_small_file(path, limit):
    """Reject links, devices and oversized inputs before reading their contents."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"Metadata must be a regular file: {path}")
        if info.st_size > limit:
            raise ValueError(f"Metadata file is too large: {path}")
        content = stream.read(limit + 1)
        if len(content) > limit:
            raise ValueError(f"Metadata file is too large: {path}")
        return content


def _json_object(content, path):
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError(f"Metadata JSON must contain an object: {path}")
    return value


def inspect_random_backbone_assets(model_source) -> dict:
    """Audit local Llama config/tokenizer files before constructing any model.

    ``asset_manifest`` contains only real SHA256 digests of the source manifest
    and declared, whitelisted metadata. Required files are config.json,
    tokenizer_config.json and at least one of tokenizer.json/tokenizer.model.
    Known optional tokenizer files must also be declared if present, so a local
    HF loader cannot consume unaudited sidecars. No weight path is accessed.
    """
    if not model_source:
        raise ValueError("random-frozen-debug requires a local metadata manifest JSON")
    manifest_path = Path(model_source).expanduser().absolute()
    if manifest_path.suffix != ".json":
        raise ValueError("random-frozen-debug model_source must be a local metadata manifest JSON")
    manifest_bytes = _read_small_file(manifest_path, _MAX_MANIFEST_BYTES)
    manifest = _json_object(manifest_bytes, manifest_path)
    metadata = manifest.get("llama_metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("path"), str) or not metadata["path"]:
        raise ValueError("Manifest requires llama_metadata.path and llama_metadata.files")
    files = metadata.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Manifest requires a nonempty llama_metadata.files SHA256 mapping")
    # Audit every name/digest before opening *any* metadata. Exact basenames
    # exclude absolute paths, traversal, Python modules and all weight formats.
    for name, expected in files.items():
        if name not in _METADATA_NAMES:
            raise ValueError(f"Non-whitelisted Llama metadata file: {name!r}")
        if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
            raise ValueError(f"Llama metadata requires a SHA256 digest: {name}")
    if not {"config.json", "tokenizer_config.json"}.issubset(files):
        raise ValueError("Llama metadata requires config.json and tokenizer_config.json")
    if not {"tokenizer.json", "tokenizer.model"}.intersection(files):
        raise ValueError("Llama metadata requires tokenizer.json or tokenizer.model")
    metadata_path = Path(metadata["path"]).expanduser()
    if not metadata_path.is_absolute():
        metadata_path = manifest_path.parent / metadata_path
    metadata_path = metadata_path.resolve(strict=True)
    if not metadata_path.is_dir():
        raise ValueError("llama_metadata.path must be a local directory")
    for name in ("chat_template.jinja", "chat_template.json", "chat_templates"):
        if os.path.lexists(metadata_path / name):
            raise ValueError(f"Unsupported unaudited tokenizer sidecar: {name}")
    for name in _METADATA_NAMES - files.keys():
        # Only probe the fixed metadata allowlist, never enumerate weight files.
        if os.path.lexists(metadata_path / name):
            raise ValueError(f"Undeclared Llama metadata file: {name}")
    hashes = {"manifest": hashlib.sha256(manifest_bytes).hexdigest()}
    json_metadata = {}
    total_bytes = 0
    for name, expected in sorted(files.items()):
        path = metadata_path / name
        content = _read_small_file(path, _MAX_METADATA_BYTES)
        total_bytes += len(content)
        if total_bytes > _MAX_TOTAL_METADATA_BYTES:
            raise ValueError("Llama metadata exceeds the total size limit")
        digest = hashlib.sha256(content).hexdigest()
        if digest != expected.lower():
            raise ValueError(f"Llama metadata SHA256 mismatch: {name}")
        hashes["llama/" + name] = digest
        if name.endswith(".json"):
            json_metadata[name] = _json_object(content, path)
    llama_config = json_metadata["config.json"]
    if llama_config.get("model_type") != "llama":
        raise ValueError("random-frozen-debug requires a Llama config")
    tokenizer_config = json_metadata["tokenizer_config.json"]
    if tokenizer_config.get("tokenizer_class", "LlamaTokenizer") not in {"LlamaTokenizer", "LlamaTokenizerFast"}:
        raise ValueError("random-frozen-debug requires the original Llama tokenizer")
    # Tokenizer init kwargs must not redirect the loader outside audited files.
    for key in ("tokenizer_file", "vocab_file", "merges_file", "special_tokens_map_file", "added_tokens_file"):
        if tokenizer_config.get(key) is not None:
            raise ValueError(f"Unaudited tokenizer file override: {key}")
    if tokenizer_config.get("fast_tokenizer_files"):
        raise ValueError("Versioned tokenizer file overrides are not supported by the metadata audit")
    return {"metadata_path": str(metadata_path), "asset_manifest": hashes, "llama_config": llama_config}
