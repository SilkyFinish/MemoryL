"""Native local RMBench episodes with an immutable, architecture-neutral split.

The audited local files use ``joint_action/vector`` in the order left arm (6),
left gripper, right arm (6), right gripper. RMBench's get_*_arm_jointState reads
joint.get_drive_target(), not measured articulation qpos; the grippers are
continuous normalized command values.

Labels use the verified local one-frame shift, observation[t] ->
joint_action/vector[t + 1 : t + 1 + horizon], matching the local TurboVLA
conversion contract. Each target row is therefore the next recorded absolute
drive target, which is NOT a certified next-issued command: the recorded
sequence is a captured drive target stream, and per-file provenance of what the
controller committed between two frames is unavailable. The final recorded RGB
frame is never a query because it has no following target row, so it is
unsupervised. Targets are never differenced or gripper-binarized.

The local image codec is essential: old, unmarked JPEG buffers have reversed
JPEG-standard channels, whereas XPL-RGB1-marked buffers are standard RGB.
Always call RMBench's read-only codec, never blindly decode with PIL.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import tempfile
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
from PIL import Image

from .config import SUPPORTED_CAMERAS, validate_cameras

TARGET_OFFSET = 1
MANIFEST_SCHEMA = "choice-memory-native-rmbench-v2"
ACTION_KEY = "joint_action/vector"
ACTION_DIM = 14
CAMERAS = SUPPORTED_CAMERAS
CHANNEL_ORDER = (
    [f"left_arm_joint_{index}" for index in range(6)]
    + ["left_gripper"]
    + [f"right_arm_joint_{index}" for index in range(6)]
    + ["right_gripper"]
)
_CODEC_CANDIDATES = (
    Path(__file__).resolve().parents[2] / "RMBench/data/decode_image_bit.py",
    Path(__file__).resolve().parents[3] / "memory/RMBench/data/decode_image_bit.py",
)
CODEC_PATH = next((path for path in _CODEC_CANDIDATES if path.is_file()), _CODEC_CANDIDATES[0])
HISTORY_SAMPLING_MODES = {"sequential", "random"}
# Configuration from the original MemoryVLA vla/datasets/datasets.py, not a
# claim of numerical equivalence with its TensorFlow/dlimp implementation.
IMAGE_AUGMENTATION = {
    "random_resized_crop": {"scale": [0.9, 0.9], "ratio": [1.0, 1.0]},
    "random_brightness": [0.2],
    "random_contrast": [0.8, 1.2],
    "random_saturation": [0.8, 1.2],
    "random_hue": [0.05],
    "augment_order": [
        "random_resized_crop", "random_brightness", "random_contrast", "random_saturation", "random_hue",
    ],
}
SEMANTIC_RISKS = [
    "Native labels are captured absolute drive targets, not measured qpos, and are not a certified "
    "next-issued command. With target_offset=1 a target row is the next recorded absolute drive target; "
    "closed-loop validation is still required.",
    "Historical HDF5 files lack per-frame collector commit provenance, so the exact command issued "
    "between two recorded frames cannot be reconstructed from the file alone.",
    "New XPolicyLab state/action HDF5 shifts targets by one frame and reorders packed channels; "
    "it is intentionally not accepted as this native format.",
    "The final recorded RGB frame has no following target row and is therefore never supervised; "
    "the last usable query timestep is episode_length - 2 for a one-step horizon.",
]


def hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_manifest(manifest: dict) -> str:
    content = {name: value for name, value in manifest.items() if name != "manifest_sha256"}
    serialized = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def resolve_codec_path(codec_path: str | Path | None = None, *, recorded_path: str | None = None) -> Path:
    """Prefer an explicit path, then RMBENCH_CODEC_PATH, then read-only local copies.

    A supplied path/environment override must exist: a typo must not silently
    select a different codec. The copied memory_l checkout has no sibling
    RMBench; its existing memory/RMBench codec is a supported fallback.
    """
    override = codec_path if codec_path is not None else os.environ.get("RMBENCH_CODEC_PATH")
    if override is not None:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Explicit native RMBench RGB codec does not exist: {path}")
        return path.resolve()
    candidates = [CODEC_PATH]
    if recorded_path:
        candidates.append(Path(recorded_path).expanduser())
    candidates.extend(_CODEC_CANDIDATES)
    for path in candidates:
        if path.is_file():
            return path.resolve()
    searched = ", ".join(str(path) for path in dict.fromkeys(candidates))
    raise FileNotFoundError(
        f"The audited local RMBench RGB codec is required; searched {searched}. "
        "Pass codec_path or set RMBENCH_CODEC_PATH to the existing decode_image_bit.py; do not copy/replace the codec."
    )


@lru_cache(maxsize=8)
def _load_image_codec(codec_path: Path, sha256: str):
    # Key the cache by both path and content, so an edited codec cannot reuse a
    # previously loaded module after manifest hash validation.
    specification = importlib.util.spec_from_file_location("choice_mem_native_rgb_codec", codec_path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load the local RGB codec: {codec_path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def load_image_codec(codec_path: str | Path | None = None):
    """Load the existing codec without modifying its files or sys.path."""
    path = resolve_codec_path(codec_path)
    return _load_image_codec(path, hash_file(path))


# Preserve the former cached function's management API, while resolving any
# environment override anew on each call rather than caching a stale None key.
load_image_codec.cache_clear = _load_image_codec.cache_clear
load_image_codec.cache_info = _load_image_codec.cache_info


def decode_rgb_frame(frame, codec_path: str | Path | None = None) -> Image.Image:
    rgb = np.asarray(load_image_codec(codec_path).decode_image_bit(frame))
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Native camera frame must decode to uint8 HWC RGB, got {rgb.shape}/{rgb.dtype}")
    return Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")


def image_augmentation_metadata(enabled: bool) -> dict:
    """Describe the training-only, configuration-compatible CPU augmentation."""
    return {
        "enabled": enabled,
        "train_only": True,
        "validation_enabled": False,
        "source": "MemoryVLA vla/datasets/datasets.py image_augment_kwargs",
        "parameters": deepcopy(IMAGE_AUGMENTATION),
        "implementation": "Pillow bilinear crop/resize; NumPy color arithmetic; OpenCV float HSV",
        "equivalence": "same augmentation configuration, NOT bitwise TensorFlow/dlimp equivalent",
        "pipeline": "native RGB decode -> resize to image_size -> train-only augmentation; output size unchanged",
        "randomness": "NumPy default_rng; get_group seed controls sampling and per-query/per-camera augmentation",
        "crop_coordinates": "dlimp normalized coordinates: sqrt(0.9) of each image axis; random floating offsets",
        "color_clipping": "clip to [0, 1] after each color operation, as in dlimp augment_image",
    }


def augment_rgb_image(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Apply the original crop/color parameter ranges, without importing TF/torch.

    Brightness is additive in [0, 1], contrast uses per-channel spatial means,
    saturation is an HSV multiplier, and hue is a fraction of a full turn.
    Interpolation, RNG and rounding differ from TensorFlow/dlimp by design.
    """
    import cv2

    width, height = image.size
    # dlimp's ratio is in normalized box coordinates, not pixel aspect ratio:
    # ratio=1 retains sqrt(scale) of each axis, including nonsquare inputs.
    crop_height, crop_width = height * math.sqrt(0.9), width * math.sqrt(0.9)
    top = float(rng.uniform(0.0, height - crop_height))
    left = float(rng.uniform(0.0, width - crop_width))
    image = image.resize(
        (width, height), resample=Image.Resampling.BILINEAR,
        box=(left, top, left + crop_width, top + crop_height),
    )
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    rgb = np.clip(rgb + rng.uniform(-0.2, 0.2), 0.0, 1.0)
    mean = rgb.mean(axis=(0, 1), keepdims=True)
    rgb = np.clip((rgb - mean) * rng.uniform(0.8, 1.2) + mean, 0.0, 1.0)
    hsv = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.8, 1.2), 0.0, 1.0)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-0.05, 0.05) * 360.0) % 360.0
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return Image.fromarray(np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")


def discover_episodes(data_root: str | Path) -> list[Path]:
    """Follow local directory symlinks, pruning cycles and duplicate targets."""
    root = Path(data_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    episode_paths = []
    visited_directories = set()
    visited_files = set()
    for directory, directory_names, file_names in os.walk(root, followlinks=True):
        directory_path = Path(directory)
        resolved_directory = directory_path.resolve()
        if resolved_directory in visited_directories:
            directory_names[:] = []
            continue
        visited_directories.add(resolved_directory)
        directory_names[:] = sorted(name for name in directory_names if not name.startswith("."))
        for file_name in sorted(file_names):
            if not (file_name.startswith("episode") and file_name.endswith(".hdf5")):
                continue
            episode_path = directory_path / file_name
            resolved_file = episode_path.resolve(strict=True)
            if resolved_file not in visited_files:
                visited_files.add(resolved_file)
                episode_paths.append(episode_path)
    if not episode_paths:
        raise FileNotFoundError(
            f"No native episode*.hdf5 files below {root}. EventVLA's LeRobot template paths "
            "are not the audited local dataset; pass RMBench/data or a task's demo_clean directory."
        )
    return sorted(episode_paths, key=lambda episode_path: episode_path.relative_to(root).as_posix())


def select_episode_paths(root: Path, tasks: list[str] | None = None) -> list[Path]:
    """Explicit task selection excludes data-only tasks without native environments."""
    if tasks is None:
        return discover_episodes(root)
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("Selected task names must be nonempty and unique")
    episode_paths = []
    for task_name in tasks:
        if not task_name.isidentifier():
            raise ValueError(f"Invalid native task name: {task_name}")
        task_root = root / task_name / "demo_clean"
        if not task_root.is_dir():
            raise FileNotFoundError(f"Missing selected native demo_clean task data: {task_root}")
        episode_paths.extend(discover_episodes(task_root))
    return sorted(episode_paths, key=lambda episode_path: episode_path.relative_to(root).as_posix())


def read_instruction(handle: h5py.File, episode_path: Path) -> tuple[str, dict]:
    if "language_instruction" in handle:
        instruction_value = handle["language_instruction"][()]
        if isinstance(instruction_value, bytes):
            instruction_value = instruction_value.decode("utf-8")
        if isinstance(instruction_value, str) and instruction_value.strip():
            return instruction_value.strip(), {"kind": "hdf5", "key": "language_instruction"}
    episode_directory = episode_path.parent.parent if episode_path.parent.name == "data" else episode_path.parent
    instruction_path = episode_directory / "instructions" / f"{episode_path.stem}.json"
    if instruction_path.is_file():
        with instruction_path.open(encoding="utf-8") as stream:
            annotation = json.load(stream)
        seen_instructions = annotation.get("seen", [])
        if seen_instructions and isinstance(seen_instructions[0], str) and seen_instructions[0].strip():
            return seen_instructions[0].strip(), {
                "kind": "sidecar-seen-first",
                "path": str(instruction_path.resolve()),
                "sha256": hash_file(instruction_path),
            }
    raise ValueError(f"No nonempty native instruction or seen[0] instruction sidecar for {episode_path}")


def read_native_actions(handle: h5py.File, episode_path: Path) -> np.ndarray:
    if ACTION_KEY not in handle:
        raise ValueError(f"{episode_path}: expected native {ACTION_KEY}; converted state/action is not supported")
    actions = np.asarray(handle[ACTION_KEY][:], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or len(actions) < 1:
        raise ValueError(f"{episode_path}: ALOHA absolute actions must have nonempty shape [T, 14], got {actions.shape}")
    if len(actions) <= TARGET_OFFSET:
        raise ValueError(
            f"{episode_path}: target_offset={TARGET_OFFSET} needs at least {TARGET_OFFSET + 1} recorded rows "
            f"to form one supervised target, got {len(actions)}"
        )
    if not np.isfinite(actions).all():
        raise ValueError(f"{episode_path}: action labels contain nonfinite values")
    component_keys = ["left_arm", "left_gripper", "right_arm", "right_gripper"]
    if all(f"joint_action/{name}" in handle for name in component_keys):
        components = [np.asarray(handle[f"joint_action/{name}"][:]) for name in component_keys]
        components = [component.reshape(len(actions), -1) for component in components]
        if [component.shape[1] for component in components] != [6, 1, 6, 1]:
            raise ValueError(f"{episode_path}: native joint components do not have dimensions 6/1/6/1")
        if not np.allclose(actions, np.concatenate(components, axis=1), rtol=0, atol=1e-7):
            raise ValueError(f"{episode_path}: vector channel order differs from left6/gripper/right6/gripper")
    return actions


def inspect_episode(
    episode_path: Path, root: Path, cameras: list[str], codec_path: str | Path | None = None
) -> tuple[dict, np.ndarray]:
    with h5py.File(episode_path, "r") as handle:
        actions = read_native_actions(handle, episode_path)
        instruction, instruction_source = read_instruction(handle, episode_path)
        camera_metadata = {}
        for camera in cameras:
            camera_key = f"observation/{camera}/rgb"
            if camera_key not in handle:
                raise ValueError(f"{episode_path}: missing required camera {camera_key}")
            frames = handle[camera_key]
            if frames.shape[0] != len(actions):
                raise ValueError(f"{episode_path}: {camera} frame count differs from action labels")
            first_image = decode_rgb_frame(frames[0], codec_path)
            last_image = decode_rgb_frame(frames[len(actions) - 1], codec_path)
            if first_image.size != last_image.size:
                raise ValueError(f"{episode_path}: {camera} geometry changes between first and last frames")
            camera_metadata[camera] = {
                "key": camera_key,
                "storage_shape": list(frames.shape),
                "dtype": str(frames.dtype),
                "decoded_size_hw": [first_image.height, first_image.width],
            }
    file_statistics = episode_path.stat()
    relative_path = episode_path.relative_to(root).as_posix()
    task_directory = episode_path.parent.parent if episode_path.parent.name == "data" else episode_path.parent
    task_name = task_directory.relative_to(root).as_posix() if task_directory.is_relative_to(root) else "."
    record = {
        "episode_id": relative_path,
        "relative_path": relative_path,
        "resolved_path": str(episode_path.resolve()),
        "task": task_name,
        "length": len(actions),
        "file_size": file_statistics.st_size,
        "mtime_ns": file_statistics.st_mtime_ns,
        "sha256": hash_file(episode_path),
        "instruction": instruction,
        "instruction_source": instruction_source,
        "cameras": camera_metadata,
    }
    return record, actions


def assign_episode_splits(records: list[dict], seed: int, validation_fraction: float) -> None:
    records_by_task = defaultdict(list)
    for record in records:
        records_by_task[record["task"]].append(record)
    for task_records in records_by_task.values():
        ranked_records = sorted(
            task_records,
            key=lambda record: hashlib.sha256(f"{seed}:{record['episode_id']}".encode("utf-8")).hexdigest(),
        )
        validation_count = 0
        if validation_fraction > 0 and len(ranked_records) > 1:
            validation_count = min(len(ranked_records) - 1, max(1, int(len(ranked_records) * validation_fraction)))
        for record_index, record in enumerate(ranked_records):
            record["split"] = "validation" if record_index < validation_count else "train"


def calculate_normalization(training_actions: list[np.ndarray], training_episode_ids: list[str]) -> dict:
    actions = np.concatenate(training_actions, axis=0)
    lower_quantiles, upper_quantiles = np.quantile(actions, [0.01, 0.99], axis=0, method="linear")
    return {
        "action": {
            "q01": lower_quantiles.tolist(),
            "q99": upper_quantiles.tolist(),
            "mask": [True] * ACTION_DIM,
            "min": actions.min(axis=0).tolist(),
            "max": actions.max(axis=0).tolist(),
        },
        "source": (
            "train episodes only; every real target row joint_action/vector[1:] used once each; "
            "the initial drive-target row has no preceding observation and is excluded; "
            "no horizon repetitions and no padding rows"
        ),
        "training_episode_ids": training_episode_ids,
        "sample_count": len(actions),
        "action_dim": ACTION_DIM,
        "target_offset": TARGET_OFFSET,
        "target_rows": "joint_action/vector[1:] per train episode; rows used once each",
        "quantile_method": "linear",
        "clip_range": [-1.0, 1.0],
        "constant_channel_rule": "q99-q01 <= 1e-8 maps to zero; inverse maps to q01; mask stays true",
        "gripper_rule": "continuous channels 6 and 13; same quantile transform as the twelve arm channels",
    }


def write_manifest_once(path: Path, manifest: dict) -> None:
    """Publish a complete JSON atomically, without ever replacing another manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".partial", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(manifest, stream, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def validate_manifest(
    manifest: dict,
    root: Path,
    requested_protocol: dict,
    seed: int,
    validation_fraction: float,
    codec_path: str | Path | None = None,
) -> None:
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("manifest_sha256") != hash_manifest(manifest):
        raise ValueError("Manifest schema/integrity mismatch; refusing to regenerate an existing split")
    if manifest.get("data_root") != str(root):
        raise ValueError("Manifest data root differs; use the exact same local dataset for all four combinations")
    saved_protocol = dict(manifest.get("protocol", {}))
    requested_protocol = dict(requested_protocol)
    # Missing fields mean the exact legacy sequential/no-augmentation protocol.
    # Compare copies only: neither the saved dictionary nor its digest changes.
    for protocol in (saved_protocol, requested_protocol):
        protocol.setdefault("history_sampling", "sequential")
        protocol.setdefault("image_aug", False)
    if saved_protocol != requested_protocol:
        raise ValueError(
            "Manifest observation/action/group protocol differs (including history_sampling/image_aug); "
            "do not overwrite it. Legacy manifests mean sequential/no augmentation; "
            "use a new manifest path for a different training protocol."
        )
    if manifest.get("split_seed") != seed or manifest.get("validation_fraction") != validation_fraction:
        raise ValueError("Manifest split parameters differ; the saved split cannot be resampled")
    current_paths = [
        path.relative_to(root).as_posix() for path in select_episode_paths(root, requested_protocol.get("tasks"))
    ]
    if current_paths != [record["relative_path"] for record in manifest["episodes"]]:
        raise ValueError("Dataset episode inventory changed after manifest creation")
    codec_path = resolve_codec_path(codec_path, recorded_path=manifest["codec"].get("path"))
    if manifest["codec"]["sha256"] != hash_file(codec_path):
        raise ValueError(f"Native RGB codec changed after manifest creation: {codec_path}")
    for record in manifest["episodes"]:
        episode_path = root / record["relative_path"]
        file_statistics = episode_path.stat()
        if (
            str(episode_path.resolve()) != record["resolved_path"]
            or file_statistics.st_size != record["file_size"]
            or file_statistics.st_mtime_ns != record["mtime_ns"]
            or hash_file(episode_path) != record["sha256"]
        ):
            raise ValueError(f"Dataset episode changed after manifest creation: {episode_path}")
        instruction_source = record["instruction_source"]
        if instruction_source["kind"] == "sidecar-seen-first":
            if hash_file(instruction_source["path"]) != instruction_source["sha256"]:
                raise ValueError(f"Instruction sidecar changed: {instruction_source['path']}")


def prepare_manifest(
    data_root: str | Path,
    manifest_path: str | Path,
    *,
    cameras: list[str] | None = None,
    horizon: int = 16,
    execute_prefix: int = 4,
    group_size: int = 4,
    image_size: tuple[int, int] | list[int] = (224, 224),
    seed: int = 42,
    validation_fraction: float = 0.1,
    tasks: list[str] | None = None,
    history_sampling: str | None = None,
    image_aug: bool | None = None,
    codec_path: str | Path | None = None,
) -> dict:
    """Freeze native data, splits, labels, and optional training recipe metadata.

    Omitted history_sampling/image_aug preserve legacy manifest bytes and mean
    sequential/no augmentation. New training should explicitly request random
    and True; an existing incompatible manifest is never rewritten or rehashed.
    """
    if history_sampling is not None and history_sampling not in HISTORY_SAMPLING_MODES:
        raise ValueError("history_sampling must be random or sequential")
    if image_aug is not None and not isinstance(image_aug, bool):
        raise ValueError("image_aug must be a bool")
    if horizon < 1 or not 1 <= execute_prefix <= horizon or group_size < 2:
        raise ValueError("Require horizon >= execute_prefix >= 1 and group_size >= 2")
    if not 0 <= validation_fraction < 1 or not math.isfinite(validation_fraction):
        raise ValueError("validation_fraction must be finite and in [0, 1)")
    if len(image_size) != 2 or min(image_size) < 1:
        raise ValueError("image_size must be positive [height, width]")
    cameras = validate_cameras(CAMERAS if cameras is None else cameras)
    root = Path(data_root).expanduser().resolve(strict=True)
    path = Path(manifest_path).expanduser().absolute()
    protocol = {
        "format": "native-rmbench-hdf5",
        "action_key": ACTION_KEY,
        "action_dim": ACTION_DIM,
        "action_space": "ALOHA14 absolute drive-target joints and continuous normalized grippers",
        "channel_order": CHANNEL_ORDER,
        "gripper_indices": [6, 13],
        "gripper_binarization": False,
        "target_offset": TARGET_OFFSET,
        "label_alignment": (
            "observation[t] -> joint_action/vector[t+1:t+1+H]; each label is the next recorded absolute "
            "drive target, not measured qpos and not a certified next-issued command"
        ),
        "horizon": horizon,
        "execute_prefix": execute_prefix,
        "group_size": group_size,
        "query_schedule": (
            "0, execute_prefix, 2*execute_prefix, ... <= episode_length - 2 (strictly below episode_length - 1, "
            "so every query has at least one following target row); consecutive disjoint groups"
        ),
        "bank_reset": "before every group; all queries within a group belong to one episode",
        "valid_count_rule": "valid_count = min(horizon, episode_length - (t + 1)) against real target rows",
        "padding": (
            "repeat the final real target row; valid_mask false for every padded horizon timestep; "
            "the final recorded RGB frame has no target row and is never supervised"
        ),
        "cameras": cameras,
        "image_size": list(image_size),
        "resize_interpolation": "bilinear",
        "image_decoder": "RMBench decode_image_bit; legacy-unmarked/XPL-RGB1 auto-detection; output RGB",
        "semantic_risks": SEMANTIC_RISKS,
    }
    if history_sampling is not None:
        protocol["history_sampling"] = history_sampling
    if history_sampling == "random":
        protocol["query_schedule"] = (
            "one group per episode; uniformly sample group_size query indices without replacement from "
            "0..episode_length - 2, then sort chronologically; independent of execute_prefix"
        )
        protocol["query_padding"] = (
            "if fewer valid queries than group_size, retain every valid query once and repeat the final "
            "valid query until group_size; do not query the unsupervised final recorded RGB frame"
        )
    if image_aug is not None:
        protocol["image_aug"] = image_aug
    if image_aug:
        protocol["image_augmentation"] = image_augmentation_metadata(True)
    if tasks is not None:
        protocol["tasks"] = list(tasks)
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        validate_manifest(manifest, root, protocol, seed, validation_fraction, codec_path)
        return manifest
    codec_path = resolve_codec_path(codec_path)
    records = []
    actions_by_episode = {}
    for episode_path in select_episode_paths(root, tasks):
        record, actions = inspect_episode(episode_path, root, cameras, codec_path)
        records.append(record)
        actions_by_episode[record["episode_id"]] = actions
    assign_episode_splits(records, seed, validation_fraction)
    training_episode_ids = [record["episode_id"] for record in records if record["split"] == "train"]
    normalization = calculate_normalization(
        [actions_by_episode[episode_id][TARGET_OFFSET:] for episode_id in training_episode_ids],
        training_episode_ids,
    )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "data_root": str(root),
        "split_seed": seed,
        "validation_fraction": validation_fraction,
        "split_algorithm": "per-task SHA256(seed:episode_id) rank",
        "episodes": records,
        "normalization": normalization,
        "protocol": protocol,
        "codec": {"path": str(codec_path), "sha256": hash_file(codec_path)},
    }
    manifest["manifest_sha256"] = hash_manifest(manifest)
    try:
        write_manifest_once(path, manifest)
    except FileExistsError:
        return prepare_manifest(
            root,
            path,
            cameras=cameras,
            horizon=horizon,
            execute_prefix=execute_prefix,
            group_size=group_size,
            image_size=image_size,
            seed=seed,
            validation_fraction=validation_fraction,
            tasks=tasks,
            history_sampling=history_sampling,
            image_aug=image_aug,
            codec_path=codec_path,
        )
    return manifest


def normalize_actions(actions: np.ndarray, normalization: dict) -> np.ndarray:
    statistics = normalization.get("action", normalization)
    lower_quantiles = np.asarray(statistics["q01"], dtype=np.float64)
    upper_quantiles = np.asarray(statistics["q99"], dtype=np.float64)
    if lower_quantiles.shape != (ACTION_DIM,) or upper_quantiles.shape != (ACTION_DIM,):
        raise ValueError("Normalization must contain exactly fourteen channels")
    if statistics.get("mask") != [True] * ACTION_DIM:
        raise ValueError("All fourteen channels, including both continuous grippers, must be normalized")
    ranges = upper_quantiles - lower_quantiles
    safe_ranges = np.where(ranges > 1e-8, ranges, 1.0)
    normalized = 2.0 * (np.asarray(actions, dtype=np.float64) - lower_quantiles) / safe_ranges - 1.0
    normalized = np.where(ranges > 1e-8, normalized, 0.0)
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def denormalize_actions(actions: np.ndarray, normalization: dict) -> np.ndarray:
    statistics = normalization.get("action", normalization)
    lower_quantiles = np.asarray(statistics["q01"], dtype=np.float64)
    ranges = np.asarray(statistics["q99"], dtype=np.float64) - lower_quantiles
    restored = (np.asarray(actions, dtype=np.float64) + 1.0) * 0.5 * ranges + lower_quantiles
    return np.where(ranges > 1e-8, restored, lower_quantiles).astype(np.float32)


class _RMBenchGroups:
    """Shared native RGB, shifted absolute labels and training-only augmentation."""

    history_sampling: str

    def __init__(
        self,
        manifest: dict,
        split: str,
        *,
        image_aug: bool,
        codec_path: str | Path | None = None,
    ):
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        if not isinstance(image_aug, bool):
            raise ValueError("image_aug must be a bool")
        if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("manifest_sha256") != hash_manifest(manifest):
            raise ValueError("Dataset requires an intact prepared manifest")
        self.manifest = manifest
        self.normalization = manifest["normalization"]
        self.root = Path(manifest["data_root"])
        self.protocol = manifest["protocol"]
        saved_sampling = self.protocol.get("history_sampling", "sequential")
        saved_aug = self.protocol.get("image_aug", False)
        if saved_sampling != self.history_sampling or (split == "train" and saved_aug != image_aug):
            raise ValueError(
                f"Dataset history_sampling/image_aug protocol differs: manifest={saved_sampling}/{saved_aug}, "
                f"requested={self.history_sampling}/{image_aug}. Legacy manifests mean sequential/no augmentation; "
                "prepare a new manifest explicitly rather than changing its digest."
            )
        if saved_aug and self.protocol.get("image_augmentation") != image_augmentation_metadata(True):
            raise ValueError("Manifest image augmentation metadata differs from this implementation")
        self.split = split
        self.image_aug = image_aug and split == "train"
        self.augmentation_metadata = image_augmentation_metadata(self.image_aug)
        self.codec_path = resolve_codec_path(codec_path, recorded_path=manifest["codec"].get("path"))
        if hash_file(self.codec_path) != manifest["codec"]["sha256"]:
            raise ValueError(f"Native RGB codec changed after manifest creation: {self.codec_path}")
        self.groups = []
        for record in manifest["episodes"]:
            if record["split"] != split:
                continue
            valid_query_count = record["length"] - self.protocol["target_offset"]
            if valid_query_count < 1:
                raise ValueError(f"Episode has no supervised query: {record['episode_id']}")
            if self.history_sampling == "random":
                # Sampler metadata, NOT the sampled group: one entry per episode
                # exposes every candidate, independent of rollout execute_prefix.
                self.groups.append((record, list(range(valid_query_count))))
            else:
                query_timesteps = list(range(0, valid_query_count, self.protocol["execute_prefix"]))
                for start in range(0, len(query_timesteps), self.protocol["group_size"]):
                    self.groups.append((record, query_timesteps[start : start + self.protocol["group_size"]]))

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, group_index: int) -> dict:
        return self.get_group(group_index)

    def get_group(self, index: int, seed: int | None = None) -> dict:
        record, timesteps = self.groups[index]
        return self._read_group(record, timesteps, np.random.default_rng(seed))

    def _read_group(self, record: dict, timesteps: list[int], rng: np.random.Generator) -> dict:
        episode_path = self.root / record["relative_path"]
        file_statistics = episode_path.stat()
        if (
            str(episode_path.resolve()) != record["resolved_path"]
            or file_statistics.st_size != record["file_size"]
            or file_statistics.st_mtime_ns != record["mtime_ns"]
        ):
            raise ValueError(f"Episode changed during training: {episode_path}")
        horizon = self.protocol["horizon"]
        raw_actions = np.empty((len(timesteps), horizon, ACTION_DIM), dtype=np.float32)
        valid_mask = np.zeros((len(timesteps), horizon), dtype=bool)
        image_batches = []
        with h5py.File(episode_path, "r") as handle:
            for query_index, timestep in enumerate(timesteps):
                target_start = timestep + self.protocol["target_offset"]
                valid_count = min(horizon, record["length"] - target_start)
                if valid_count < 1:
                    raise ValueError(f"{episode_path}: query {timestep} has no following target row")
                action_chunk = np.asarray(
                    handle[ACTION_KEY][target_start : target_start + valid_count], dtype=np.float32
                )
                raw_actions[query_index, :valid_count] = action_chunk
                raw_actions[query_index, valid_count:] = action_chunk[-1]
                valid_mask[query_index, :valid_count] = True
                images = []
                for camera in self.protocol["cameras"]:
                    image = decode_rgb_frame(handle[f"observation/{camera}/rgb"][timestep], self.codec_path)
                    expected_size = record["cameras"][camera]["decoded_size_hw"]
                    if [image.height, image.width] != expected_size:
                        raise ValueError(f"{episode_path}: {camera} frame geometry changed at {timestep}")
                    height, width = self.protocol["image_size"]
                    if image.size != (width, height):
                        image = image.resize((width, height), resample=Image.Resampling.BILINEAR)
                    if self.image_aug:
                        image = augment_rgb_image(image, rng)
                    images.append(image)
                image_batches.append(images)
        return {
            "image_batches": image_batches,
            "instructions": [record["instruction"]] * len(timesteps),
            "actions": normalize_actions(raw_actions, self.normalization),
            "raw_actions": raw_actions,
            "valid_mask": valid_mask,
            "episode_ids": [record["episode_id"]] * len(timesteps),
            "timesteps": timesteps,
            "history_sampling": self.history_sampling,
            "image_aug": self.image_aug,
            "image_augmentation": self.augmentation_metadata,
        }

    def iter_epoch(self, epoch: int, seed: int) -> Iterator[dict]:
        # Keep the legacy group's exact shuffle order. A deterministic per-group
        # seed also supports explicit augmentation in sequential compatibility mode.
        group_indices = np.random.default_rng(seed + epoch).permutation(len(self))
        for group_index in group_indices:
            group_seed = np.random.SeedSequence([seed, epoch, int(group_index)]).generate_state(1, dtype=np.uint64)[0]
            yield self.get_group(int(group_index), seed=group_seed)


class RMBenchSequentialGroups(_RMBenchGroups):
    """Legacy compatibility: disjoint chronological execute_prefix-spaced groups.

    Defaults remain unaugmented, and a short final group is NOT padded to the
    configured group size. Query t labels remain vector[t+1:t+1+horizon].
    """

    history_sampling = "sequential"

    def __init__(
        self, manifest: dict, split: str = "train", *, image_aug: bool = False, codec_path: str | Path | None = None
    ):
        super().__init__(manifest, split, image_aug=image_aug, codec_path=codec_path)


class RMBenchRandomGroups(_RMBenchGroups):
    """Original MemoryVLA episode grouping: uniform sample, then chronological sort.

    One dataset item is one episode. ``groups[index]`` contains the episode and
    its complete legal query population for sampler inspection, not one fixed
    random draw. get_group(index, seed) deterministically controls both the draw
    and its RGB augmentation. An unseeded __getitem__ produces a fresh draw.
    """

    history_sampling = "random"

    def __init__(
        self, manifest: dict, split: str = "train", *, image_aug: bool = True, codec_path: str | Path | None = None
    ):
        super().__init__(manifest, split, image_aug=image_aug, codec_path=codec_path)

    def get_group(self, index: int, seed: int | None = None) -> dict:
        record, candidates = self.groups[index]
        rng = np.random.default_rng(seed)
        group_size = self.protocol["group_size"]
        if len(candidates) < group_size:
            timesteps = candidates + [candidates[-1]] * (group_size - len(candidates))
        else:
            timesteps = sorted(rng.choice(candidates, size=group_size, replace=False).tolist())
        return self._read_group(record, timesteps, rng)

    def iter_epoch(self, epoch: int, seed: int) -> Iterator[dict]:
        # Use independent deterministic streams for episode order and each draw.
        # Neither order nor valid query population depends on execute_prefix.
        order_rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
        for group_index in order_rng.permutation(len(self)):
            group_seed = np.random.SeedSequence([seed, epoch, int(group_index)]).generate_state(1, dtype=np.uint64)[0]
            yield self.get_group(int(group_index), seed=group_seed)


def build_training_dataset(
    manifest: dict,
    split: str = "train",
    history_sampling: str = "random",
    image_aug: bool = True,
    *,
    codec_path: str | Path | None = None,
) -> RMBenchRandomGroups | RMBenchSequentialGroups:
    """Build the explicit manifest-matching recipe; validation never augments RGB.

    The default restores random episode grouping and upstream-configured image
    augmentation. Legacy callers must request sequential/False explicitly; an
    old manifest is never silently reinterpreted as a new training protocol.
    """
    if history_sampling not in HISTORY_SAMPLING_MODES:
        raise ValueError("history_sampling must be random or sequential")
    dataset_type = RMBenchRandomGroups if history_sampling == "random" else RMBenchSequentialGroups
    return dataset_type(manifest, split, image_aug=image_aug, codec_path=codec_path)
