"""Spatial SE compression with explicit, independent camera grids."""

from __future__ import annotations

import operator
from typing import Sequence

import torch
from torch import nn


class BottleneckSE(nn.Module):
    """Checkpoint-compatible SE bottleneck, without guessing spatial layouts.

    ``camera_grids`` contains one ``(time, height, width)`` tuple per camera in
    token-concatenation order. Each camera frame is compressed independently;
    neither cameras nor time are folded into an artificial square image.
    """

    def __init__(self, C_in: int, C_mid: int, C_out: int):
        super().__init__()
        if min(C_in, C_mid, C_out) < 1:
            raise ValueError("SE channel dimensions must be positive")
        self.C_in = C_in
        self.C_mid = C_mid
        self.C_out = C_out
        self.reduce = nn.Conv2d(C_in, C_mid, 1, bias=False)
        self.act = nn.ReLU(inplace=True)
        excitation_dim = max(1, C_mid // 16)
        self.excite = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C_mid, excitation_dim, 1),
            nn.ReLU(),
            nn.Conv2d(excitation_dim, C_mid, 1),
            nn.Sigmoid(),
        )
        self.expand = nn.Conv2d(C_mid, C_out, 1, bias=False)

    def forward(self, tokens: torch.Tensor, camera_grids: Sequence[tuple[int, int, int]]) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.C_in:
            raise ValueError(f"tokens must have shape [batch, tokens, {self.C_in}], got {tuple(tokens.shape)}")
        validated_grids = []
        for camera_index, grid_size in enumerate(camera_grids):
            if len(grid_size) != 3:
                raise ValueError(f"Camera {camera_index} grid must be (time, height, width), got {grid_size!r}")
            try:
                grid_dimensions = tuple(operator.index(dimension) for dimension in grid_size)
            except TypeError as error:
                raise ValueError(f"Camera {camera_index} grid dimensions must be integers, got {grid_size!r}") from error
            if min(grid_dimensions) < 1:
                raise ValueError(f"Camera {camera_index} grid dimensions must be positive, got {grid_size!r}")
            validated_grids.append(grid_dimensions)

        expected_token_count = sum(time_steps * height * width for time_steps, height, width in validated_grids)
        if not validated_grids or expected_token_count != tokens.shape[1]:
            raise ValueError(
                f"Camera grids account for {expected_token_count} tokens, but input has {tokens.shape[1]}; "
                "provide an explicit grid for every camera"
            )

        batch_size = tokens.shape[0]
        token_offset = 0
        compressed_cameras = []
        for time_steps, height, width in validated_grids:
            camera_token_count = time_steps * height * width
            camera_tokens = tokens[:, token_offset : token_offset + camera_token_count]
            camera_frames = camera_tokens.reshape(batch_size * time_steps, height, width, self.C_in)
            camera_frames = camera_frames.permute(0, 3, 1, 2)
            reduced_frames = self.act(self.reduce(camera_frames))
            excitation_weights = self.excite(reduced_frames)
            compressed_frames = self.expand(reduced_frames * excitation_weights)
            compressed_tokens = compressed_frames.permute(0, 2, 3, 1).reshape(batch_size, camera_token_count, self.C_out)
            compressed_cameras.append(compressed_tokens)
            token_offset += camera_token_count
        return torch.cat(compressed_cameras, dim=1)


class GridAwareBottleneckSE(BottleneckSE):
    """Construct the original bottleneck width from dynamic backbone dimensions."""

    def __init__(self, input_dim: int, per_dim: int):
        super().__init__(C_in=input_dim, C_mid=per_dim * 2, C_out=per_dim)
        self.input_dim = input_dim
        self.per_dim = per_dim


class GridBottleneckSE(BottleneckSE):
    """Policy-facing name: ``GridBottleneckSE(C_in, C_mid, C_out)``.

    Identical math and parameter paths to the extracted original, plus the
    explicit ``forward(tokens, camera_grids)`` camera-aware signature.
    """

    def __init__(self, C_in: int, C_mid: int, C_out: int):
        super().__init__(C_in=C_in, C_mid=C_mid, C_out=C_out)
