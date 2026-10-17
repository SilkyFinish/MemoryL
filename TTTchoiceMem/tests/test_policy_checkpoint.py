"""Architectural contract tests use feature fixtures, not pretend VLM weights."""

import sys
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from choice_mem.backbones import FeatureBatch
from choice_mem.checkpoint import load_checkpoint, save_checkpoint
from choice_mem.config import PolicyConfig
from choice_mem.policy import ChoiceMemoryPolicy


class FeatureFixture(nn.Module):
    vision_dim = 24
    cognition_dim = 32

    def __init__(self):
        super().__init__()
        self.visual_projection = nn.Linear(3, self.vision_dim)
        self.cognition_projection = nn.Linear(3, self.cognition_dim)
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def forward(self, image_batches, instructions):
        batch_size = len(image_batches)
        visual_input = torch.arange(batch_size * 15 * 3, dtype=torch.float32).reshape(batch_size, 15, 3) / 100
        cognition_input = visual_input.mean(dim=1, keepdim=True)
        return FeatureBatch(
            self.visual_projection(visual_input),
            self.cognition_projection(cognition_input),
            ((1, 2, 3), (1, 3, 2), (1, 1, 3)),
        )


class PolicyCheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)

    def create_policy(self, backbone_name, head_name):
        config = PolicyConfig(
            backbone=backbone_name,
            action_head=head_name,
            model_source="feature-fixture",
            horizon=4,
            execute_prefix=2,
            group_size=3,
            per_token_size=16,
            oft_hidden_dim=32,
            mem_length=2,
            dtype="float32",
        )
        return ChoiceMemoryPolicy(config, backbone=FeatureFixture())

    def test_four_combinations_shapes_gradients_and_explicit_reset(self):
        for backbone_name in ["memoryvla", "qwen35"]:
            for head_name in ["diffusion", "oft"]:
                with self.subTest(backbone=backbone_name, head=head_name):
                    policy = self.create_policy(backbone_name, head_name)
                    labels = torch.randn(3, 4, 14)
                    valid = torch.ones(3, 4, dtype=torch.bool)
                    valid[-1, -1] = False
                    labels[-1, -1] = float("nan")
                    # Original DiT starts with zero output/perception projections.
                    # Verify real optimization opens these paths, not a modified init.
                    if head_name == "diffusion":
                        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
                        for _warmup_step in range(3):
                            policy.reset()
                            optimizer.zero_grad(set_to_none=True)
                            warmup_loss = policy(
                                [[None] * 3] * 3, ["task"] * 3, labels, valid, ["episode"] * 3, [0, 2, 4]
                            )
                            warmup_loss.backward()
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        policy.reset()
                    loss = policy([[None] * 3] * 3, ["task"] * 3, labels, valid, ["episode"] * 3, [0, 2, 4])
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    for bank_name in ["cog_mem_bank", "per_mem_bank"]:
                        bank = getattr(policy, bank_name)
                        gradient = bank.retrieval_blocks[0].v_proj.weight.grad
                        self.assertIsNotNone(gradient)
                        self.assertGreater(gradient.abs().sum().item(), 0)
                        self.assertLessEqual(len(bank.bank["episode"]), 2)
                    self.assertGreater(policy.per_compr.reduce.weight.grad.abs().sum().item(), 0)
                    policy.eval().reset("episode")
                    with torch.inference_mode():
                        actions = policy.predict([None] * 3, "task", "episode")
                    self.assertEqual(actions.shape, (1, 4, 14))
                    with self.assertRaisesRegex(ValueError, "explicit reset"):
                        policy.predict([None] * 3, "task", "another-episode")
                    policy.reset("another-episode")
                    self.assertFalse(policy.cog_mem_bank.bank)
                    self.assertFalse(policy.per_mem_bank.bank)
                    self.assertEqual(policy.query_timestep, 0)

    def test_checkpoint_exact_prediction_and_rejects_architecture_mismatch(self):
        policy = self.create_policy("qwen35", "oft").eval()
        normalization = {"action": {"q01": [-1.0] * 14, "q99": [1.0] * 14, "mask": [True] * 14}}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "policy.pt"
            save_checkpoint(checkpoint_path, policy, normalization, {"label_offset": 1}, step=2)
            reloaded, payload = load_checkpoint(checkpoint_path, backbone=FeatureFixture())
            with torch.inference_mode():
                for query_timestep in [0, 2, 4]:
                    original_actions = policy.predict([None] * 3, "task", "episode", timestep=query_timestep)
                    reloaded_actions = reloaded.predict([None] * 3, "task", "episode", timestep=query_timestep)
                    torch.testing.assert_close(original_actions, reloaded_actions, rtol=0, atol=0)
            self.assertEqual(payload["step"], 2)
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                load_checkpoint(checkpoint_path, expected={"action_head": "diffusion"}, backbone=FeatureFixture())
            with self.assertRaises(FileExistsError):
                save_checkpoint(checkpoint_path, policy, normalization, {}, step=3)
            incomplete_payload = dict(payload)
            incomplete_payload["config"] = dict(payload["config"])
            incomplete_payload["config"].pop("execute_prefix")
            incomplete_path = Path(directory) / "incomplete.pt"
            torch.save(incomplete_payload, incomplete_path)
            with self.assertRaisesRegex(ValueError, "configuration missing fields"):
                load_checkpoint(incomplete_path, backbone=FeatureFixture())


if __name__ == "__main__":
    unittest.main()
