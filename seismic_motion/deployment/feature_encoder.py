"""First-priority fixed-shape CoTracker encoder export boundary."""

from __future__ import annotations

import torch


class FeatureEncoderExportWrapper(torch.nn.Module):
    """Include input normalization and feature L2 normalization in the boundary."""

    def __init__(self, encoder: torch.nn.Module, epsilon: float = 1e-12) -> None:
        super().__init__()
        self.encoder = encoder
        self.epsilon = float(epsilon)

    def forward(self, rgb_0_255: torch.Tensor) -> torch.Tensor:
        normalized_rgb = 2 * (rgb_0_255 / 255.0) - 1.0
        features = self.encoder(normalized_rgb)
        denominator = torch.sqrt(
            torch.clamp_min(
                torch.sum(torch.square(features), dim=1, keepdim=True), self.epsilon
            )
        )
        return features / denominator
