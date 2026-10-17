"""Standalone MemoryVLA banks, with checkpoint-compatible parameter names."""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Sequence

import torch
import torch.nn.functional as functional
from torch import nn

AttentionRecorder = Callable[[dict[str, Any]], None]


class TimestepEmbedder(nn.Module):
    """Embed scalar timesteps using the original sinusoidal encoding and MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half_dimension = dim // 2
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half_dimension, dtype=torch.float32) / half_dimension
        ).to(timesteps.device)
        phase = timesteps[:, None].float() * frequencies[None]
        embedding = torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        parameter = next(self.mlp.parameters())
        timesteps = timesteps.to(parameter.device)
        frequency_embedding = self.timestep_embedding(timesteps, self.frequency_embedding_size).to(parameter.dtype)
        return self.mlp(frequency_embedding)


class CrossTransformerBlock(nn.Module):
    """Original single-head memory read; recording is opt-in and stateless.

    A recorder receives one dictionary per read. All tensors are detached, and
    the block retains no records. The callback owns any storage or CPU copies.
    The recorded softmax is reconstructed from the same projected Q/K; the
    raw readout is the actual SDPA result before either residual/normalization.
    """

    def __init__(self, feature_dim: int, attention_recorder: Optional[AttentionRecorder] = None):
        super().__init__()
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.attn_norm = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Linear(feature_dim * 4, feature_dim),
        )
        self.ffn_norm = nn.LayerNorm(feature_dim)
        self.attention_recorder = attention_recorder

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bank_metadata: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        projected_query = self.q_proj(query)
        projected_key = self.k_proj(key)
        projected_value = self.v_proj(value)
        raw_readout = functional.scaled_dot_product_attention(
            projected_query, projected_key, projected_value, dropout_p=0.0, is_causal=False
        )

        if self.attention_recorder is not None:
            with torch.no_grad():
                attention_scores = projected_query.detach() @ projected_key.detach().transpose(-2, -1)
                attention_weights = torch.softmax(attention_scores / math.sqrt(projected_query.shape[-1]), dim=-1)
                self.attention_recorder(
                    {
                        "projected_query": projected_query.detach(),
                        "projected_key": projected_key.detach(),
                        "projected_value": projected_value.detach(),
                        "softmax": attention_weights.detach(),
                        "raw_readout": raw_readout.detach(),
                        "bank_metadata": {} if bank_metadata is None else dict(bank_metadata),
                    }
                )

        attended_tokens = self.attn_norm(query + raw_readout)
        feedforward_tokens = self.ffn(attended_tokens)
        return self.ffn_norm(attended_tokens + feedforward_tokens)


class GateFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim * 2, dim)
        nn.init.normal_(self.proj.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.proj.bias, mean=0.0, std=1e-3)

    def forward(self, working_tokens: torch.Tensor, retrieved_tokens: torch.Tensor) -> torch.Tensor:
        fusion_scale = torch.sigmoid(self.proj(torch.cat([working_tokens, retrieved_tokens], dim=-1)))
        return fusion_scale * working_tokens + (1 - fusion_scale) * retrieved_tokens


class CogMemBank(nn.Module):
    """Original detached episode memory, supporting variable history lengths.

    State-dict paths, stream/group episode clearing, timestep keys, fusion, and
    consolidation match MemoryVLA. Stored features intentionally do not keep
    an autograd graph; retrieval parameters and the current query remain live.
    """

    def __init__(
        self,
        dataloader_type: str,
        group_size: int,
        token_size: int,
        mem_length: int = 16,
        retrieval_layers: int = 2,
        use_timestep_pe: bool = True,
        fusion_type: str = "gate",
        consolidate_type: str = "tome",
        update_fused: bool = False,
        attention_recorder: Optional[AttentionRecorder] = None,
    ):
        super().__init__()
        assert dataloader_type in ("stream", "group")
        assert fusion_type in ("gate", "add")
        assert consolidate_type in ("fifo", "tome")
        if mem_length < 1 or group_size < 1:
            raise ValueError("mem_length and group_size must both be positive")

        self.dataloader_type = dataloader_type
        self.group_size = group_size
        self.token_size = token_size
        self.mem_length = mem_length
        self.retrieval_layers = retrieval_layers
        self.use_timestep_pe = use_timestep_pe
        self.fusion_type = fusion_type
        self.consolidate_type = consolidate_type
        self.update_fused = update_fused

        self.retrieval_blocks = nn.ModuleList(
            [CrossTransformerBlock(self.token_size, attention_recorder) for _ in range(self.retrieval_layers)]
        )
        if self.fusion_type == "gate":
            self.gate_fusion_blocks = GateFusion(self.token_size)
        if self.use_timestep_pe:
            self.timestep_encoder = TimestepEmbedder(self.token_size, frequency_embedding_size=self.token_size // 4)
        else:
            self.timestep_encoder = None
        self.reset()

    def reset(self):
        self.bank = {}
        self.eid_stream = None

    def clear_episode(self, episode_id):
        self.bank.pop(episode_id, None)

    def set_attention_recorder(self, recorder: Optional[AttentionRecorder]):
        for retrieval_block in self.retrieval_blocks:
            retrieval_block.attention_recorder = recorder

    @torch.no_grad()
    def _consolidate_with_token_merge(self, episode_id):
        episode_bank = self.bank.get(episode_id, [])
        if len(episode_bank) < 2:
            return

        candidate_indices = []
        similarities = []
        for entry_index in range(len(episode_bank) - 1):
            first_features = episode_bank[entry_index][1]
            second_features = episode_bank[entry_index + 1][1]
            if first_features.shape != second_features.shape:
                continue
            first_rows = first_features.flatten(1) if first_features.dim() > 1 else first_features.unsqueeze(0)
            second_rows = second_features.flatten(1) if second_features.dim() > 1 else second_features.unsqueeze(0)
            candidate_indices.append(entry_index)
            similarities.append(functional.cosine_similarity(first_rows, second_rows, dim=1).mean().item())

        if not candidate_indices:
            entry_shapes = [tuple(features.shape) for _, features in episode_bank]
            raise ValueError(
                f"Cannot consolidate episode {episode_id!r}: no adjacent entries have equal token shapes; "
                f"history shapes are {entry_shapes}. Use fixed token layouts or FIFO consolidation."
            )

        # Keep the original float32 argmax and first-match tie breaking.
        merge_index = candidate_indices[int(torch.tensor(similarities).argmax().item())]
        first_timestep, first_features = episode_bank[merge_index]
        _, second_features = episode_bank[merge_index + 1]
        merged_features = 0.5 * (first_features + second_features)
        episode_bank[merge_index] = (first_timestep, merged_features.detach().clone())
        episode_bank.pop(merge_index + 1)

    @torch.no_grad()
    def _memory_consolidate(self, episode_id, feat: torch.Tensor, timestep: Optional[torch.Tensor]):
        if episode_id not in self.bank:
            self.bank[episode_id] = []
        self.bank[episode_id].append((timestep, feat.detach().clone()))

        while len(self.bank[episode_id]) > self.mem_length:
            if self.consolidate_type == "fifo":
                self.bank[episode_id] = self.bank[episode_id][-self.mem_length :]
            elif self.consolidate_type == "tome":
                self._consolidate_with_token_merge(episode_id)
            else:
                raise NotImplementedError

    def process_batch(
        self,
        tokens: torch.Tensor,
        episode_ids: Sequence,
        timesteps: Optional[Sequence],
    ) -> torch.Tensor:
        assert episode_ids is not None, "episode_ids must be provided during training"
        if self.use_timestep_pe:
            assert timesteps is not None, "timesteps must be provided during training"

        batch_size, _, _ = tokens.shape
        outputs = []
        if self.training:
            if self.dataloader_type == "group":
                self.bank.clear()
                self.eid_stream = None
            elif self.dataloader_type == "stream":
                first_episode_id = episode_ids[0]
                if self.eid_stream is not None and self.eid_stream != first_episode_id:
                    self.clear_episode(self.eid_stream)
                self.eid_stream = first_episode_id

        for batch_index in range(batch_size):
            episode_id = episode_ids[batch_index]
            if self.training:
                if self.dataloader_type == "group" and batch_index > 0 and batch_index % self.group_size == 0:
                    previous_group_episode_id = episode_ids[batch_index - self.group_size]
                    self.clear_episode(previous_group_episode_id)
                if self.dataloader_type == "stream" and batch_index > 0:
                    previous_episode_id = episode_ids[batch_index - 1]
                    if episode_id != previous_episode_id:
                        self.clear_episode(previous_episode_id)
                        self.eid_stream = episode_id

            working_memory = tokens[batch_index].unsqueeze(0)
            history = self.bank.get(episode_id, [])
            if history:
                history_features = [features for _, features in history]
                episode_memory = torch.cat(history_features, dim=0).unsqueeze(0)
                history_token_counts = [features.shape[0] for features in history_features]

                if self.use_timestep_pe:
                    history_timesteps = torch.tensor([timestep for timestep, _ in history]).to(working_memory.device)
                    timestep_embeddings = self.timestep_encoder(history_timesteps).unsqueeze(0)
                    entry_repeats = torch.tensor(history_token_counts, device=working_memory.device)
                    timestep_embeddings = timestep_embeddings.repeat_interleave(entry_repeats, dim=1)
                else:
                    timestep_embeddings = torch.zeros_like(episode_memory)

                retrieved_memory = working_memory
                for layer_index, retrieval_block in enumerate(self.retrieval_blocks):
                    bank_metadata = None
                    if retrieval_block.attention_recorder is not None:
                        bank_metadata = {
                            "bank_type": type(self).__name__,
                            "episode_id": episode_id,
                            "timestep": None if timesteps is None else self._metadata_scalar(timesteps[batch_index]),
                            "history_timesteps": tuple(self._metadata_scalar(timestep) for timestep, _ in history),
                            "history_token_counts": tuple(history_token_counts),
                            "layer_index": layer_index,
                        }
                    retrieved_memory = retrieval_block(
                        retrieved_memory, episode_memory + timestep_embeddings, episode_memory, bank_metadata
                    )
            else:
                retrieved_memory = working_memory

            if self.fusion_type == "add":
                fused_features = (working_memory + retrieved_memory) * 0.5
            elif self.fusion_type == "gate":
                fused_features = self.gate_fusion_blocks(working_memory, retrieved_memory)
            outputs.append(fused_features)

            current_timestep = timesteps[batch_index] if self.use_timestep_pe else None
            memory_features = fused_features.squeeze(0) if self.update_fused else tokens[batch_index]
            self._memory_consolidate(episode_id, memory_features, current_timestep)

        return torch.cat(outputs, dim=0)

    @staticmethod
    def _metadata_scalar(value):
        return value.detach().item() if isinstance(value, torch.Tensor) else value


class PerMemBank(CogMemBank):
    """Perception bank uses exactly the same operations as the cognition bank."""

    def __init__(
        self,
        dataloader_type: str,
        group_size: int,
        token_size: int,
        mem_length: int = 16,
        retrieval_layers: int = 2,
        use_timestep_pe: bool = True,
        fusion_type: str = "gate",
        consolidate_type: str = "tome",
        update_fused: bool = False,
        attention_recorder: Optional[AttentionRecorder] = None,
    ):
        super().__init__(
            dataloader_type=dataloader_type,
            group_size=group_size,
            token_size=token_size,
            mem_length=mem_length,
            retrieval_layers=retrieval_layers,
            use_timestep_pe=use_timestep_pe,
            fusion_type=fusion_type,
            consolidate_type=consolidate_type,
            update_fused=update_fused,
            attention_recorder=attention_recorder,
        )
