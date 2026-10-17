"""Focused regressions for native camera decoding, labels, split leakage and grouping."""

import builtins
import hashlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

import train_rmbench
from choice_mem.config import PolicyConfig
from choice_mem.data import (
    CAMERAS,
    TARGET_OFFSET,
    RMBenchSequentialGroups,
    decode_rgb_frame,
    denormalize_actions,
    discover_episodes,
    load_image_codec,
    normalize_actions,
    prepare_manifest,
)


def create_native_episode(
    path: Path, *, length: int = 18, offset: float = 0.0, instruction: str | None = "move the block"
):
    path.parent.mkdir(parents=True, exist_ok=True)
    actions = np.arange(length, dtype=np.float64)[:, None] * 0.04 + np.arange(14)[None, :] * 0.2 + offset
    actions[:, 6] = np.linspace(0.12, 0.88, length)
    actions[:, 13] = np.linspace(0.81, 0.17, length)
    codec = load_image_codec()
    with h5py.File(path, "w") as handle:
        handle.create_dataset("joint_action/vector", data=actions)
        for name, component in (
            ("left_arm", actions[:, :6]),
            ("left_gripper", actions[:, 6]),
            ("right_arm", actions[:, 7:13]),
            ("right_gripper", actions[:, 13]),
        ):
            handle.create_dataset(f"joint_action/{name}", data=component)
        if instruction is not None:
            handle.create_dataset("language_instruction", data=instruction)
        for camera_index, camera in enumerate(CAMERAS):
            buffers = []
            for timestep in range(length):
                image = np.full((8, 10, 3), 8, dtype=np.uint8)
                image[:, :, camera_index] = 128 + timestep
                buffers.append(codec.encode_image_bit(image, quality=100))
            maximum_size = max(len(buffer) for buffer in buffers)
            handle.create_dataset(f"observation/{camera}/rgb", data=np.asarray(buffers, dtype=f"S{maximum_size}"))
    return actions


def test_legacy_and_marked_jpeg_both_decode_to_native_rgb():
    rgb = np.zeros((12, 16, 3), dtype=np.uint8)
    rgb[:, :, 0] = 231
    rgb[:, :, 1] = 41
    rgb[:, :, 2] = 13
    standard_buffer = load_image_codec().encode_image_bit(rgb, quality=100)
    succeeded, legacy_buffer = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 100])
    assert succeeded
    for buffer in (standard_buffer, legacy_buffer, np.bytes_(standard_buffer)):
        decoded = np.asarray(decode_rgb_frame(buffer))
        assert decoded.shape == rgb.shape
        np.testing.assert_allclose(decoded.astype(np.int16), rgb.astype(np.int16), atol=2, rtol=0)


def test_target_offset_matches_verified_local_conversion_contract(tmp_path):
    root = tmp_path / "native"
    create_native_episode(root / "task/demo_clean/data/episode0.hdf5")
    manifest = prepare_manifest(root, tmp_path / "manifest.json", image_size=(8, 10))
    protocol = manifest["protocol"]
    assert protocol["target_offset"] == 1 == TARGET_OFFSET
    assert "t+1:t+1+H" in protocol["label_alignment"]
    assert "not measured qpos" in protocol["label_alignment"]
    assert "not a certified next-issued command" in protocol["label_alignment"]
    assert "episode_length - 2" in protocol["query_schedule"]
    assert "episode_length - (t + 1)" in protocol["valid_count_rule"]
    assert "never supervised" in protocol["padding"]
    assert protocol["gripper_binarization"] is False
    assert any("never supervised" in risk for risk in protocol["semantic_risks"])
    assert all(key in protocol for key in ("target_offset", "label_alignment", "valid_count_rule"))


def test_explicit_tasks_exclude_data_only_tasks_from_split_and_statistics(tmp_path):
    root = tmp_path / "native"
    for task_name, offset in [("press_button", 0.0), ("swap_T", 1.0), ("missing_env", 10000.0)]:
        for episode_index in range(2):
            create_native_episode(root / task_name / "demo_clean/data" / f"episode{episode_index}.hdf5", offset=offset)
    manifest_path = tmp_path / "selected.json"
    selected_tasks = ["press_button", "swap_T"]
    manifest = prepare_manifest(root, manifest_path, tasks=selected_tasks)
    assert len(manifest["episodes"]) == 4
    assert manifest["protocol"]["tasks"] == selected_tasks
    assert all("missing_env" not in record["episode_id"] for record in manifest["episodes"])
    assert max(manifest["normalization"]["action"]["q99"]) < 10
    assert prepare_manifest(root, manifest_path, tasks=selected_tasks) == manifest
    with pytest.raises(ValueError, match="protocol differs"):
        prepare_manifest(root, manifest_path, tasks=["press_button"])


def test_split_is_immutable_and_quantiles_use_only_real_target_rows(tmp_path):
    root = tmp_path / "native"
    episode_ids = [f"task/demo_clean/data/episode{index}.hdf5" for index in range(4)]
    heldout_episode_id = min(episode_ids, key=lambda episode_id: hashlib.sha256(f"42:{episode_id}".encode()).hexdigest())
    actions_by_episode = {}
    for episode_index, episode_id in enumerate(episode_ids):
        offset = 10000.0 if episode_id == heldout_episode_id else episode_index * 0.3
        actions_by_episode[episode_id] = create_native_episode(root / episode_id, offset=offset)
    path = tmp_path / "manifest.json"
    manifest = prepare_manifest(root, path, validation_fraction=0.25)
    validation_ids = [record["episode_id"] for record in manifest["episodes"] if record["split"] == "validation"]
    assert validation_ids == [heldout_episode_id]
    train_ids = manifest["normalization"]["training_episode_ids"]
    assert heldout_episode_id not in train_ids
    # Targets are the following recorded rows only; the unused initial row must not enter the quantiles.
    training_rows = np.concatenate([actions_by_episode[episode_id][TARGET_OFFSET:] for episode_id in train_ids])
    expected_quantiles = np.quantile(training_rows, [0.01, 0.99], axis=0, method="linear")
    statistics = manifest["normalization"]["action"]
    np.testing.assert_allclose(statistics["q01"], expected_quantiles[0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(statistics["q99"], expected_quantiles[1], rtol=0, atol=1e-12)
    assert statistics["mask"] == [True] * 14
    assert manifest["normalization"]["sample_count"] == len(training_rows)
    assert manifest["normalization"]["sample_count"] == sum(
        len(actions) - TARGET_OFFSET for actions in (actions_by_episode[episode_id] for episode_id in train_ids)
    )
    assert manifest["normalization"]["target_rows"].startswith("joint_action/vector[1:]")
    assert statistics["q99"][0] < 10
    original_bytes = path.read_bytes()
    assert prepare_manifest(root, path, validation_fraction=0.25) == manifest
    assert path.read_bytes() == original_bytes
    for changed_arguments in ({"seed": 43}, {"horizon": 8}, {"execute_prefix": 2}, {"group_size": 3}):
        with pytest.raises(ValueError, match="differ"):
            prepare_manifest(root, path, validation_fraction=0.25, **changed_arguments)
        assert path.read_bytes() == original_bytes


def test_shifted_targets_padding_and_sequential_groups_have_no_episode_leakage(tmp_path):
    root = tmp_path / "native"
    first_actions = create_native_episode(root / "task_a/demo_clean/data/episode0.hdf5")
    create_native_episode(root / "task_b/demo_clean/data/episode0.hdf5", offset=5)
    manifest = prepare_manifest(root, tmp_path / "manifest.json", image_size=(8, 10))
    dataset = RMBenchSequentialGroups(manifest)
    assert len(dataset) == 4
    first_group = dataset[0]
    assert first_group["timesteps"] == [0, 4, 8, 12]
    assert set(first_group["episode_ids"]) == {"task_a/demo_clean/data/episode0.hdf5"}
    # The initial recorded row has no preceding observation, so it is never a supervised target.
    np.testing.assert_allclose(first_group["raw_actions"][0, 0], first_actions[TARGET_OFFSET])
    assert not np.allclose(first_group["raw_actions"][0, 0], first_actions[0])
    for query_index, timestep in enumerate(first_group["timesteps"]):
        target_start = timestep + TARGET_OFFSET
        valid_count = min(16, 18 - target_start)
        np.testing.assert_allclose(
            first_group["raw_actions"][query_index, :valid_count],
            first_actions[target_start : target_start + valid_count],
        )
        assert first_group["valid_mask"][query_index].tolist() == [True] * valid_count + [False] * (16 - valid_count)
        images = first_group["image_batches"][query_index]
        assert all(image.mode == "RGB" and image.size == (10, 8) for image in images)
        assert [int(np.asarray(image)[0, 0].argmax()) for image in images] == [0, 1, 2]
        assert abs(int(np.asarray(images[0])[0, 0, 0]) - (128 + timestep)) <= 2
    final_group = dataset[1]
    assert final_group["timesteps"] == [16]
    np.testing.assert_allclose(final_group["raw_actions"][0, :1], first_actions[17:18])
    # Only one real target row follows query 16, so the remaining 15 slots repeat it and stay masked out.
    np.testing.assert_allclose(final_group["raw_actions"][0, 1:], np.repeat(first_actions[-1:], 15, axis=0))
    assert final_group["valid_mask"].sum() == 1
    assert not final_group["valid_mask"][0, 1:].any()
    assert first_group["actions"].dtype == np.float32
    assert first_group["valid_mask"].dtype == bool
    for group in dataset.iter_epoch(epoch=0, seed=99):
        assert len(set(group["episode_ids"])) == 1
        # The final recorded RGB frame (index length - 1) must never be a query.
        assert all(timestep <= 16 for timestep in group["timesteps"])
        assert all(
            later - earlier == 4 for earlier, later in zip(group["timesteps"], group["timesteps"][1:], strict=False)
        )
    all_train_timesteps = [timestep for _record, group_timesteps in dataset.groups for timestep in group_timesteps]
    assert 17 not in all_train_timesteps
    first_order = [(group["episode_ids"][0], group["timesteps"]) for group in dataset.iter_epoch(2, 99)]
    repeated_order = [(group["episode_ids"][0], group["timesteps"]) for group in dataset.iter_epoch(2, 99)]
    assert first_order == repeated_order


@pytest.mark.parametrize("selected_cameras", [
    ["head_camera"], ["head_camera", "right_camera"], ["right_camera", "head_camera", "left_camera"],
])
def test_selected_views_decode_only_requested_images_and_preserve_split(tmp_path, selected_cameras):
    root = tmp_path / "native"
    for episode_index in range(2):
        create_native_episode(root / "task/demo_clean/data" / f"episode{episode_index}.hdf5")
    full_manifest = prepare_manifest(root, tmp_path / "three_views.json", image_size=(8, 10))
    selected_manifest = prepare_manifest(
        root, tmp_path / "selected_views.json", cameras=selected_cameras, image_size=(8, 10),
    )
    assert selected_manifest["normalization"] == full_manifest["normalization"]
    assert [(record["episode_id"], record["split"]) for record in selected_manifest["episodes"]] == [
        (record["episode_id"], record["split"]) for record in full_manifest["episodes"]
    ]
    group = RMBenchSequentialGroups(selected_manifest)[0]
    assert selected_manifest["protocol"]["cameras"] == selected_cameras
    for images in group["image_batches"]:
        assert len(images) == len(selected_cameras)
        assert [int(np.asarray(image)[0, 0].argmax()) for image in images] == [
            CAMERAS.index(camera) for camera in selected_cameras
        ]
    with pytest.raises(ValueError, match="protocol differs"):
        prepare_manifest(root, tmp_path / "three_views.json", cameras=selected_cameras, image_size=(8, 10))


def test_short_episodes_keep_only_queries_that_have_a_following_target(tmp_path):
    root = tmp_path / "native"
    create_native_episode(root / "task/demo_clean/data/episode0.hdf5", length=2)
    manifest = prepare_manifest(root, tmp_path / "manifest.json", image_size=(8, 10))
    dataset = RMBenchSequentialGroups(manifest)
    assert len(dataset) == 1
    group = dataset[0]
    assert group["timesteps"] == [0]
    assert group["valid_mask"].tolist() == [[True] + [False] * 15]
    np.testing.assert_allclose(group["raw_actions"][0, 0], group["raw_actions"][0, -1])


def test_normalization_preserves_continuous_grippers_and_constant_channel_mask():
    normalization = {"action": {"q01": [-2.0] * 14, "q99": [2.0] * 14, "mask": [True] * 14}}
    normalization["action"]["q01"][3] = 0.7
    normalization["action"]["q99"][3] = 0.7
    actions = np.zeros((3, 14), dtype=np.float64)
    actions[:, 3] = 0.7
    actions[:, 6] = [0.2, 0.45, 0.8]
    actions[:, 13] = [0.75, 0.35, 0.15]
    normalized = normalize_actions(actions, normalization)
    np.testing.assert_allclose(normalized[:, 6], [0.1, 0.225, 0.4], atol=1e-7)
    np.testing.assert_allclose(normalized[:, 13], [0.375, 0.175, 0.075], atol=1e-7)
    assert (normalized[:, 3] == 0).all()
    assert normalization["action"]["mask"] == [True] * 14
    np.testing.assert_allclose(denormalize_actions(normalized, normalization), actions, atol=1e-7)
    actions[0, 0] = 100
    assert normalize_actions(actions, normalization)[0, 0] == 1


@pytest.mark.parametrize(
    "problem, expected_error",
    [
        ("missing_camera", "missing required camera"),
        ("seven_dimensions", "shape \\[T, 14\\]"),
        ("channel_reorder", "channel order"),
        ("nonfinite", "nonfinite"),
        ("single_target_row", "target_offset=1"),
    ],
)
def test_preparation_rejects_incompatible_native_data(tmp_path, problem, expected_error):
    root = tmp_path / "native"
    path = root / "task/demo_clean/data/episode0.hdf5"
    actions = create_native_episode(path)
    if problem == "single_target_row":
        # Only one recorded row: with target_offset=1 there is no following row to supervise.
        path.unlink()
        create_native_episode(path, length=1)
        with pytest.raises(ValueError, match=expected_error):
            prepare_manifest(root, tmp_path / "manifest.json")
        assert not (tmp_path / "manifest.json").exists()
        return
    with h5py.File(path, "r+") as handle:
        if problem == "missing_camera":
            del handle["observation/right_camera/rgb"]
        elif problem == "seven_dimensions":
            del handle["joint_action/vector"]
            handle.create_dataset("joint_action/vector", data=actions[:, :7])
        elif problem == "channel_reorder":
            handle["joint_action/vector"][:] = np.roll(actions, 1, axis=1)
        else:
            handle["joint_action/vector"][0, 0] = np.nan
    with pytest.raises(ValueError, match=expected_error):
        prepare_manifest(root, tmp_path / "manifest.json")
    assert not (tmp_path / "manifest.json").exists()


def test_sidecar_seen_instruction_and_changed_data_fail_closed(tmp_path):
    root = tmp_path / "native"
    path = root / "task/demo_clean/data/episode0.hdf5"
    create_native_episode(path, instruction=None)
    instruction_path = path.parent.parent / "instructions/episode0.json"
    instruction_path.parent.mkdir()
    instruction_path.write_text(json.dumps({"seen": ["seen task"], "unseen": ["not a training instruction"]}))
    manifest_path = tmp_path / "manifest.json"
    manifest = prepare_manifest(root, manifest_path)
    assert manifest["episodes"][0]["instruction"] == "seen task"
    original_manifest = manifest_path.read_bytes()
    instruction_path.write_text(json.dumps({"seen": ["changed instruction"]}))
    with pytest.raises(ValueError, match="sidecar changed"):
        prepare_manifest(root, manifest_path)
    assert manifest_path.read_bytes() == original_manifest
    instruction_path.write_text(json.dumps({"seen": ["seen task"], "unseen": ["not a training instruction"]}))
    with h5py.File(path, "r+") as handle:
        handle["joint_action/vector"][0, 0] = 3
    with pytest.raises(ValueError, match="episode changed"):
        prepare_manifest(root, manifest_path)
    assert manifest_path.read_bytes() == original_manifest


def test_symlink_roots_and_cycles_do_not_duplicate_episode_ids(tmp_path):
    real_root = tmp_path / "real"
    create_native_episode(real_root / "demo_clean/data/episode0.hdf5")
    linked_root = tmp_path / "links"
    linked_root.mkdir()
    (linked_root / "task").symlink_to(real_root, target_is_directory=True)
    (real_root / "loop").symlink_to(linked_root, target_is_directory=True)
    paths = discover_episodes(linked_root)
    assert [path.relative_to(linked_root).as_posix() for path in paths] == ["task/demo_clean/data/episode0.hdf5"]
    manifest = prepare_manifest(linked_root, tmp_path / "manifest.json")
    assert len(manifest["episodes"]) == 1
    direct_manifest = prepare_manifest(real_root / "demo_clean/data", tmp_path / "direct-manifest.json")
    assert direct_manifest["episodes"][0]["episode_id"] == "episode0.hdf5"
    assert direct_manifest["episodes"][0]["task"] == "."


def test_prepare_only_never_imports_policy_model_or_torch(tmp_path, monkeypatch, capsys):
    root = tmp_path / "native"
    create_native_episode(root / "task/demo_clean/data/episode0.hdf5")
    original_import = builtins.__import__

    def forbid_model_imports(name, *arguments, **keywords):
        if name.startswith(
            ("torch", "transformers", "choice_mem.policy", "choice_mem.checkpoint", "choice_mem.backbones")
        ):
            raise AssertionError(f"prepare-only imported model code: {name}")
        return original_import(name, *arguments, **keywords)

    monkeypatch.setattr(builtins, "__import__", forbid_model_imports)
    summary = train_rmbench.main(
        [
            "--prepare-only",
            "--data-root",
            str(root),
            "--manifest",
            str(tmp_path / "manifest.json"),
        ]
    )
    assert summary["event"] == "data_prepared"
    assert summary["train_episodes"] == 1
    assert summary["target_offset"] == TARGET_OFFSET
    assert "t+1:t+1+H" in summary["label_alignment"]
    assert json.loads(capsys.readouterr().out)["manifest_sha256"] == summary["manifest_sha256"]


def test_exposed_architecture_flags_keep_policy_config_defaults():
    parser = train_rmbench.build_parser()
    defaults = parser.parse_args(["--data-root", "data", "--manifest", "manifest.json"])
    policy_defaults = PolicyConfig()
    assert defaults.cpu_threads == 4
    assert (
        defaults.per_token_size,
        defaults.mem_length,
        defaults.retrieval_layers,
        defaults.oft_hidden_dim,
        defaults.diffusion_model_type,
    ) == (
        policy_defaults.per_token_size,
        policy_defaults.mem_length,
        policy_defaults.retrieval_layers,
        policy_defaults.oft_hidden_dim,
        policy_defaults.diffusion_model_type,
    )
    overridden = parser.parse_args(
        [
            "--data-root",
            "data",
            "--manifest",
            "manifest.json",
            "--per-token-size",
            "128",
            "--mem-length",
            "4",
            "--retrieval-layers",
            "1",
            "--oft-hidden-dim",
            "64",
            "--diffusion-model-type",
            "DiT-B",
            "--cpu-threads",
            "1",
            "--horizon",
            "8",
            "--execute-prefix",
            "2",
            "--group-size",
            "2",
        ]
    )
    assert overridden.cpu_threads == 1
    assert (
        overridden.per_token_size,
        overridden.mem_length,
        overridden.retrieval_layers,
        overridden.oft_hidden_dim,
        overridden.diffusion_model_type,
    ) == (128, 4, 1, 64, "DiT-B")


def test_training_requires_explicit_step_budget_before_data_access(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        train_rmbench.main(
            [
                "--data-root",
                str(tmp_path / "missing-data"),
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--run-dir",
                str(tmp_path / "run"),
            ]
        )
    assert error.value.code == 2
    assert "explicit --max-steps" in capsys.readouterr().err
    assert not (tmp_path / "manifest.json").exists()
    assert not (tmp_path / "run").exists()
