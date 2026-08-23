"""Numerically checkable candidate for CoTracker's high-risk Einsum."""

from __future__ import annotations

import torch


def correlation_einsum(correlation_features: torch.Tensor, track_support: torch.Tensor) -> torch.Tensor:
    """Reference: [B,T,N,H,W,C] × [B,N,I,J,C] -> [B,T,N,H,W,I,J]."""

    return torch.einsum(
        "btnhwc,bnijc->btnhwij", correlation_features, track_support
    )


def correlation_batched_matmul(
    correlation_features: torch.Tensor, track_support: torch.Tensor
) -> torch.Tensor:
    """Equivalent reshape + MatMul candidate without torch.einsum."""

    batch, time, points, height, width, channels = correlation_features.shape
    support_batch, support_points, support_h, support_w, support_channels = track_support.shape
    if (support_batch, support_points, support_channels) != (batch, points, channels):
        raise ValueError("correlation and support tensor dimensions do not match")
    left = correlation_features.permute(0, 2, 1, 3, 4, 5).reshape(
        batch, points, time * height * width, channels
    )
    right = track_support.reshape(
        batch, points, support_h * support_w, channels
    ).transpose(-1, -2)
    product = torch.matmul(left, right)
    return product.reshape(
        batch, points, time, height, width, support_h, support_w
    ).permute(0, 2, 1, 3, 4, 5, 6)

