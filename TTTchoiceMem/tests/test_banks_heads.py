"""CPU-only regression checks; original memory classes are loaded through AST."""

from __future__ import annotations

import ast
import itertools
import math
from pathlib import Path
import sys
from typing import Optional
import unittest

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from choice_mem.banks import CogMemBank, CrossTransformerBlock, PerMemBank, TimestepEmbedder
from choice_mem.compression import GridAwareBottleneckSE, GridBottleneckSE
from choice_mem.heads import DiffusionHead, OFTHead


def extract_original_classes(source_path: Path, class_names: set[str], namespace: Optional[dict] = None) -> dict:
    """Execute only selected definitions, never heavy VLA/backbone imports."""
    source_tree = ast.parse(source_path.read_text())
    class_definitions = [
        definition
        for definition in source_tree.body
        if isinstance(definition, ast.ClassDef) and definition.name in class_names
    ]
    if {definition.name for definition in class_definitions} != class_names:
        raise AssertionError(f"Missing requested original classes in {source_path}")
    original_namespace = {
        "torch": torch,
        "nn": nn,
        "F": functional,
        "math": math,
        "np": np,
        "Optional": Optional,
    }
    if namespace is not None:
        original_namespace.update(namespace)
    isolated_module = ast.Module(body=class_definitions, type_ignores=[])
    exec(compile(isolated_module, str(source_path), "exec"), original_namespace)
    return original_namespace


ORIGINAL_MEMORY = extract_original_classes(
    PROJECT_ROOT / "vla/memory_vla.py",
    {"TimestepEmbedder", "CrossTransformerBlock", "BottleneckSE", "GateFusion", "CogMemBank", "PerMemBank"},
)
PREVIOUS_THREAD_COUNT = None


def setUpModule():
    global PREVIOUS_THREAD_COUNT
    PREVIOUS_THREAD_COUNT = torch.get_num_threads()
    torch.set_num_threads(1)


def tearDownModule():
    torch.set_num_threads(PREVIOUS_THREAD_COUNT)


class MemoryCompatibilityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(104)

    def assert_banks_equal(self, original, extracted):
        self.assertEqual(original.eid_stream, extracted.eid_stream)
        self.assertEqual(set(original.bank), set(extracted.bank))
        for episode_id, original_entries in original.bank.items():
            extracted_entries = extracted.bank[episode_id]
            self.assertEqual(len(original_entries), len(extracted_entries))
            for (original_timestep, original_features), (extracted_timestep, extracted_features) in zip(
                original_entries, extracted_entries
            ):
                self.assertEqual(original_timestep, extracted_timestep)
                torch.testing.assert_close(original_features, extracted_features, rtol=0, atol=0)
                self.assertFalse(extracted_features.requires_grad)

    def test_timestep_encoding_and_state_dict_match_original(self):
        for frequency_dimension in (6, 7):
            with self.subTest(frequency_dimension=frequency_dimension):
                original = ORIGINAL_MEMORY["TimestepEmbedder"](16, frequency_dimension)
                extracted = TimestepEmbedder(16, frequency_dimension)
                extracted.load_state_dict(original.state_dict(), strict=True)
                timesteps = torch.tensor([0.0, 3.5, 70.0])
                torch.testing.assert_close(original(timesteps), extracted(timesteps), rtol=0, atol=0)

    def test_stream_group_capacity_fusion_and_updates_match_source(self):
        combinations = itertools.product(
            (CogMemBank, PerMemBank),
            ("stream", "group"),
            ("fifo", "tome"),
            ("gate", "add"),
            (False, True),
            (False, True),
        )
        for bank_class, dataloader_type, consolidation_type, fusion_type, update_fused, use_timestep_pe in combinations:
            with self.subTest(
                bank=bank_class.__name__,
                loader=dataloader_type,
                consolidation=consolidation_type,
                fusion=fusion_type,
                update_fused=update_fused,
                use_timestep_pe=use_timestep_pe,
            ):
                bank_options = {
                    "dataloader_type": dataloader_type,
                    "group_size": 3,
                    "token_size": 16,
                    "mem_length": 2,
                    "retrieval_layers": 2,
                    "use_timestep_pe": use_timestep_pe,
                    "fusion_type": fusion_type,
                    "consolidate_type": consolidation_type,
                    "update_fused": update_fused,
                }
                torch.manual_seed(31)
                original = ORIGINAL_MEMORY[bank_class.__name__](**bank_options)
                torch.manual_seed(31)
                extracted = bank_class(**bank_options)
                self.assertEqual(set(original.state_dict()), set(extracted.state_dict()))
                for parameter_name, original_parameter in original.state_dict().items():
                    torch.testing.assert_close(
                        original_parameter, extracted.state_dict()[parameter_name], rtol=0, atol=0
                    )
                extracted.load_state_dict(original.state_dict(), strict=True)
                for batch_index in range(3):
                    tokens = torch.randn(6, 3, 16)
                    episode_ids = np.array([11, 11, 11, 22, 22, 22])
                    timesteps = np.array([0, 1, 2, 0, 1, 2]) + batch_index * 3
                    original_output = original.process_batch(tokens, episode_ids, timesteps)
                    extracted_output = extracted.process_batch(tokens, episode_ids, timesteps)
                    torch.testing.assert_close(original_output, extracted_output, rtol=0, atol=0)
                    self.assert_banks_equal(original, extracted)
                    self.assertTrue(all(len(entries) <= 2 for entries in extracted.bank.values()))
                original.eval()
                extracted.eval()
                eval_tokens = torch.randn(2, 3, 16)
                torch.testing.assert_close(
                    original.process_batch(eval_tokens, [33, 44], [0, 0]),
                    extracted.process_batch(eval_tokens, [33, 44], [0, 0]),
                    rtol=0,
                    atol=0,
                )
                self.assert_banks_equal(original, extracted)
                extracted.clear_episode(33)
                self.assertNotIn(33, extracted.bank)
                extracted.reset()
                self.assertEqual(extracted.bank, {})
                self.assertIsNone(extracted.eid_stream)

    def test_recording_is_detached_and_preserves_forward_and_gradients(self):
        original = ORIGINAL_MEMORY["CrossTransformerBlock"](16)
        extracted = CrossTransformerBlock(16)
        extracted.load_state_dict(original.state_dict(), strict=True)
        query = torch.randn(1, 3, 16, requires_grad=True)
        key = torch.randn(1, 5, 16, requires_grad=True)
        value = torch.randn(1, 5, 16, requires_grad=True)
        original_output = original(query, key, value)
        disabled_output = extracted(query, key, value)
        torch.testing.assert_close(original_output, disabled_output, rtol=0, atol=0)
        records = []
        extracted.attention_recorder = records.append
        recorded_output = extracted(query, key, value, {"episode_id": 9})
        torch.testing.assert_close(original_output, recorded_output, rtol=0, atol=0)
        loss_weights = torch.randn_like(original_output)
        original_gradients = torch.autograd.grad((original_output * loss_weights).sum(), (query, key, value))
        recorded_gradients = torch.autograd.grad((recorded_output * loss_weights).sum(), (query, key, value))
        for original_gradient, recorded_gradient in zip(original_gradients, recorded_gradients):
            torch.testing.assert_close(original_gradient, recorded_gradient, rtol=0, atol=0)
        self.assertEqual(len(records), 1)
        record = records[0]
        for name in ("projected_query", "projected_key", "projected_value", "softmax", "raw_readout"):
            self.assertFalse(record[name].requires_grad)
            self.assertIsNone(record[name].grad_fn)
        torch.testing.assert_close(record["softmax"] @ record["projected_value"], record["raw_readout"])
        self.assertEqual(record["bank_metadata"], {"episode_id": 9})
        extracted.attention_recorder = None
        extracted(query, key, value)
        self.assertEqual(len(records), 1)

    def test_variable_history_uses_each_entry_token_count_for_timestep(self):
        for bank_class in (CogMemBank, PerMemBank):
            with self.subTest(bank=bank_class.__name__):
                records = []
                bank = bank_class("stream", 1, 16, mem_length=3, retrieval_layers=1, attention_recorder=records.append)
                first_history = torch.randn(2, 16)
                second_history = torch.randn(4, 16)
                bank._memory_consolidate(7, first_history, 1)
                bank._memory_consolidate(7, second_history, 5)
                output = bank.process_batch(torch.randn(1, 3, 16), [7], [8])
                self.assertEqual(tuple(output.shape), (1, 3, 16))
                metadata = records[0]["bank_metadata"]
                self.assertEqual(metadata["history_token_counts"], (2, 4))
                self.assertEqual(metadata["history_timesteps"], (1, 5))
                self.assertEqual(metadata["timestep"], 8)
                self.assertEqual(metadata["layer_index"], 0)
                history_features = torch.cat([first_history, second_history]).unsqueeze(0)
                timestep_embeddings = bank.timestep_encoder(torch.tensor([1, 5]))
                repeated_embeddings = torch.cat(
                    [timestep_embeddings[0].expand(2, -1), timestep_embeddings[1].expand(4, -1)]
                ).unsqueeze(0)
                torch.testing.assert_close(
                    records[0]["projected_key"], bank.retrieval_blocks[0].k_proj(history_features + repeated_embeddings)
                )
                torch.testing.assert_close(
                    records[0]["projected_value"], bank.retrieval_blocks[0].v_proj(history_features)
                )

    def test_token_merge_skips_incompatible_neighbors_and_reports_no_candidates(self):
        bank = CogMemBank("stream", 1, 16, mem_length=2, consolidate_type="tome")
        first_features = torch.randn(2, 16)
        second_features = torch.randn(3, 16)
        third_features = torch.randn(3, 16)
        for timestep, features in enumerate((first_features, second_features, third_features)):
            bank._memory_consolidate(4, features, timestep)
        self.assertEqual([timestep for timestep, _ in bank.bank[4]], [0, 1])
        torch.testing.assert_close(bank.bank[4][1][1], 0.5 * (second_features + third_features))
        bank.reset()
        bank._memory_consolidate(4, torch.randn(2, 16), 0)
        bank._memory_consolidate(4, torch.randn(3, 16), 1)
        with self.assertRaisesRegex(ValueError, "no adjacent entries have equal token shapes"):
            bank._memory_consolidate(4, torch.randn(4, 16), 2)


class OFTHeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.head = OFTHead(16, 12, horizon=4, action_dim=3, hidden_dim=24)
        self.cognition = torch.randn(2, 1, 16)
        self.perception = torch.randn(2, 3, 12)

    def test_slots_remain_distinct_with_singleton_memories(self):
        predictions = self.head.predict(self.cognition, self.perception[:, :1])
        self.assertEqual(tuple(predictions.shape), (2, 4, 3))
        for first_slot, second_slot in itertools.combinations(range(4), 2):
            self.assertFalse(torch.allclose(predictions[:, first_slot], predictions[:, second_slot]))

    def test_both_mask_shapes_exclude_nan_padding_with_finite_gradients(self):
        masks = (
            torch.tensor([[True, True, False, False], [True, False, True, False]]),
            torch.rand(2, 4, 3) > 0.4,
        )
        for valid_mask in masks:
            with self.subTest(mask_shape=tuple(valid_mask.shape)):
                expanded_mask = valid_mask.unsqueeze(-1).expand(2, 4, 3) if valid_mask.ndim == 2 else valid_mask
                padded_actions = torch.randn(2, 4, 3).masked_fill(~expanded_mask, float("nan")).requires_grad_()
                predictions = self.head.predict(self.cognition, self.perception)
                expected_loss = (predictions[expanded_mask] - padded_actions[expanded_mask]).abs().mean()
                masked_loss = self.head.loss(padded_actions, self.cognition, self.perception, valid_mask)
                torch.testing.assert_close(masked_loss, expected_loss)
                self.head.zero_grad(set_to_none=True)
                masked_loss.backward()
                self.assertTrue(torch.isfinite(padded_actions.grad).all())
                self.assertTrue((padded_actions.grad[~expanded_mask] == 0).all())
                for parameter in self.head.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_all_valid_and_all_invalid_masks(self):
        actions = torch.randn(2, 4, 3)
        original_loss = self.head.loss(actions, self.cognition, self.perception)
        full_mask = torch.ones(2, 4, dtype=torch.bool)
        torch.testing.assert_close(self.head.loss(actions, self.cognition, self.perception, full_mask), original_loss)
        padded_actions = torch.full_like(actions, float("nan"))
        empty_loss = self.head.loss(padded_actions, self.cognition, self.perception, ~full_mask)
        self.assertEqual(empty_loss.item(), 0.0)
        empty_loss.backward()
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in self.head.parameters()))
        with self.assertRaisesRegex(ValueError, "valid_mask must have shape"):
            self.head.loss(actions, self.cognition, self.perception, torch.ones(2))

    def test_oft_loss_backpropagates_through_both_history_reads(self):
        cognition_bank = CogMemBank("stream", 1, 16, mem_length=4, retrieval_layers=2)
        perception_bank = PerMemBank("stream", 1, 12, mem_length=4, retrieval_layers=2)
        cognition_history = torch.randn(2, 1, 16, requires_grad=True)
        perception_history = torch.randn(2, 3, 12, requires_grad=True)
        cognition_bank.process_batch(cognition_history, [8, 8], [0, 1])
        perception_bank.process_batch(perception_history, [8, 8], [0, 1])
        current_cognition = torch.randn(1, 1, 16, requires_grad=True)
        current_perception = torch.randn(1, 3, 12, requires_grad=True)
        fused_cognition = cognition_bank.process_batch(current_cognition, [8], [2])
        fused_perception = perception_bank.process_batch(current_perception, [8], [2])
        loss = self.head.loss(torch.randn(1, 4, 3), fused_cognition, fused_perception, torch.ones(1, 4))
        loss.backward()
        for bank in (cognition_bank, perception_bank):
            for retrieval_block in bank.retrieval_blocks:
                for projection in (retrieval_block.q_proj, retrieval_block.k_proj, retrieval_block.v_proj):
                    self.assertIsNotNone(projection.weight.grad)
                    self.assertTrue(torch.isfinite(projection.weight.grad).all())
                    self.assertGreater(projection.weight.grad.abs().sum().item(), 0)
            self.assertGreater(bank.gate_fusion_blocks.proj.weight.grad.abs().sum().item(), 0)
            self.assertGreater(bank.timestep_encoder.mlp[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(current_cognition.grad.abs().sum().item(), 0)
        self.assertGreater(current_perception.grad.abs().sum().item(), 0)
        self.assertIsNone(cognition_history.grad)
        self.assertIsNone(perception_history.grad)


class CompressionTests(unittest.TestCase):
    def test_square_single_camera_matches_original_checkpoint(self):
        original = ORIGINAL_MEMORY["BottleneckSE"](20, 32, 16)
        adapter = GridAwareBottleneckSE(input_dim=20, per_dim=16)
        adapter.load_state_dict(original.state_dict(), strict=True)
        tokens = torch.randn(2, 9, 20)
        torch.testing.assert_close(original(tokens), adapter(tokens, [(1, 3, 3)]), rtol=0, atol=0)
        self.assertEqual(adapter.C_mid, 32)

    def test_camera_and_time_grids_are_independent_and_non_square(self):
        adapter = GridAwareBottleneckSE(input_dim=7, per_dim=8)
        tokens = torch.randn(2, 18, 7, requires_grad=True)
        compressed = adapter(tokens, [(2, 2, 3), (1, 3, 2)])
        separate_frames = [adapter(tokens[:, start : start + 6], [(1, 2, 3)]) for start in (0, 6)]
        separate_frames.append(adapter(tokens[:, 12:], [(1, 3, 2)]))
        torch.testing.assert_close(compressed, torch.cat(separate_frames, dim=1))
        changed_camera = torch.cat([tokens[:, :12], tokens[:, 12:] + 100], dim=1)
        torch.testing.assert_close(compressed[:, :12], adapter(changed_camera, [(2, 2, 3), (1, 3, 2)])[:, :12])
        compressed.square().sum().backward()
        self.assertTrue(torch.isfinite(tokens.grad).all())
        with self.assertRaisesRegex(ValueError, "account for"):
            adapter(tokens, [(1, 2, 3)])
        with self.assertRaisesRegex(ValueError, "must be positive"):
            adapter(tokens, [(0, 3, 6)])

    def test_policy_facing_grid_bottleneck_se_takes_channel_dimensions(self):
        original = ORIGINAL_MEMORY["BottleneckSE"](20, 32, 16)
        policy_adapter = GridBottleneckSE(20, 32, 16)
        policy_adapter.load_state_dict(original.state_dict(), strict=True)
        self.assertEqual((policy_adapter.C_in, policy_adapter.C_mid, policy_adapter.C_out), (20, 32, 16))
        square_tokens = torch.randn(2, 9, 20)
        torch.testing.assert_close(original(square_tokens), policy_adapter(square_tokens, [(1, 3, 3)]))
        # The policy-facing signature also accepts a two-camera non-square layout.
        non_square_tokens = torch.randn(3, 18, 20)
        non_square_output = policy_adapter(tokens=non_square_tokens, camera_grids=[(1, 3, 3), (1, 3, 3)])
        self.assertEqual(tuple(non_square_output.shape), (3, 18, 16))


class DiffusionHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from action_model import create_diffusion
        from action_model import gaussian_diffusion
        from action_model.action_model import DiT_models

        original_action = extract_original_classes(
            PROJECT_ROOT / "action_model/action_model.py",
            {"ActionModel"},
            {"create_diffusion": create_diffusion, "gd": gaussian_diffusion, "DiT_models": DiT_models},
        )
        cls.original_action_class = original_action["ActionModel"]
        torch.manual_seed(27)
        cls.head = DiffusionHead(16, 12, horizon=4, action_dim=3, ddim_steps=2)
        # A nonzero final projection makes parity sensitive to model predictions.
        nn.init.normal_(cls.head.action_model.net.final_layer.linear.weight, std=0.02)

    def setUp(self):
        torch.manual_seed(19)
        self.cognition = torch.randn(2, 1, 16)
        self.perception = torch.randn(2, 3, 12)
        self.actions = torch.randn(2, 4, 3)
        self.head.train()
        self.head.repeated_diffusion_steps = 1
        self.head.mask_padding_loss = True
        self.head.zero_grad(set_to_none=True)

    def test_restored_repeat4_unmasked_loss_and_rng_match_original(self):
        self.head.repeated_diffusion_steps = 4
        self.head.mask_padding_loss = False
        initial_rng_state = torch.get_rng_state()
        with torch.no_grad():
            original_loss = self.original_action_class.loss(
                self.head.action_model,
                self.actions.repeat(4, 1, 1),
                self.cognition.repeat(4, 1, 1),
                self.perception.repeat(4, 1, 1),
            )
            original_final_rng_state = torch.get_rng_state()
            torch.set_rng_state(initial_rng_state)
            # False tail flags are data metadata, not part of the original objective.
            wrapped_loss = self.head.loss(
                self.actions, self.cognition, self.perception, torch.zeros(2, 4, dtype=torch.bool)
            )
        torch.testing.assert_close(original_loss, wrapped_loss, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), original_final_rng_state))
        with self.assertRaisesRegex(ValueError, "finite absolute-action padding"):
            self.head.loss(self.actions * float("nan"), self.cognition, self.perception)

    def test_fixed_rng_original_loss_and_full_masks_are_equivalent(self):
        initial_rng_state = torch.get_rng_state()
        with torch.no_grad():
            original_loss = self.original_action_class.loss(
                self.head.action_model, self.actions, self.cognition, self.perception
            )
            original_final_rng_state = torch.get_rng_state()
            for valid_mask in (None, torch.ones(2, 4), torch.ones(2, 4, 3)):
                with self.subTest(mask_shape=None if valid_mask is None else tuple(valid_mask.shape)):
                    torch.set_rng_state(initial_rng_state)
                    wrapped_loss = self.head.loss(self.actions, self.cognition, self.perception, valid_mask)
                    torch.testing.assert_close(original_loss, wrapped_loss, rtol=0, atol=0)
                    self.assertTrue(torch.equal(torch.get_rng_state(), original_final_rng_state))

    def test_masked_nan_padding_matches_zero_padding_and_has_finite_gradients(self):
        for valid_mask in (torch.rand(2, 4) > 0.5, torch.rand(2, 4, 3) > 0.5):
            with self.subTest(mask_shape=tuple(valid_mask.shape)):
                expanded_mask = valid_mask.unsqueeze(-1).expand_as(self.actions) if valid_mask.ndim == 2 else valid_mask
                padded_actions = self.actions.masked_fill(~expanded_mask, float("nan")).requires_grad_()
                zero_padded_actions = self.actions.masked_fill(~expanded_mask, 0)
                initial_rng_state = torch.get_rng_state()
                masked_loss = self.head.loss(padded_actions, self.cognition, self.perception, valid_mask)
                torch.set_rng_state(initial_rng_state)
                zero_padded_loss = self.head.loss(zero_padded_actions, self.cognition, self.perception, valid_mask)
                torch.testing.assert_close(masked_loss, zero_padded_loss, rtol=0, atol=0)
                self.head.zero_grad(set_to_none=True)
                masked_loss.backward()
                self.assertTrue(torch.isfinite(padded_actions.grad).all())
                self.assertTrue((padded_actions.grad[~expanded_mask] == 0).all())
                for parameter in self.head.parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_empty_mask_returns_differentiable_zero(self):
        loss = self.head.loss(
            torch.full_like(self.actions, float("nan")), self.cognition, self.perception, torch.zeros(2, 4)
        )
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        for parameter in self.head.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_cfg_ddim_samples_match_original_sampling_code(self):
        self.head.eval()
        for cfg_scale in (1.0, 1.5):
            with self.subTest(cfg_scale=cfg_scale), torch.inference_mode():
                self.head.cfg_scale = cfg_scale
                initial_rng_state = torch.get_rng_state()
                predicted_actions = self.head.predict(self.cognition, self.perception)
                torch.set_rng_state(initial_rng_state)
                noise = torch.randn(2, 4, 3)
                if cfg_scale > 1.0:
                    noise = torch.cat([noise, noise])
                    unconditional = self.head.action_model.net.z_embedder.uncondition.unsqueeze(0).expand(2, -1, -1)
                    sample_function = self.head.action_model.net.forward_with_cfg
                    model_kwargs = {
                        "z": torch.cat([self.cognition, unconditional]),
                        "cfg_scale": cfg_scale,
                        "per_token": self.perception.repeat(2, 1, 1),
                    }
                else:
                    sample_function = self.head.action_model.net.forward
                    model_kwargs = {"z": self.cognition, "per_token": self.perception}
                original_samples = self.head.action_model.ddim_diffusion.ddim_sample_loop(
                    sample_function,
                    noise.shape,
                    noise,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    progress=False,
                    device=self.cognition.device,
                    eta=0.0,
                )
                expected_actions = original_samples.chunk(2, dim=0)[0] if cfg_scale > 1.0 else original_samples
                torch.testing.assert_close(predicted_actions, expected_actions, rtol=0, atol=0)
                self.assertEqual(tuple(predicted_actions.shape), (2, 4, 3))


if __name__ == "__main__":
    unittest.main()
