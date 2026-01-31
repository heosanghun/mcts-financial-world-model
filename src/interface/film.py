"""
Phase 4: FiLM — z → γ, β (Feature-wise Linear Modulation)
논문: out = γ*x + β, γ·β ∈ [-5, 5], Low Entropy → γ≈1, High Entropy → γ→0.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class FiLMGenerator(nn.Module):
    """국면 벡터 z → 변조 파라미터 γ, β (채널별)."""

    def __init__(self, z_dim: int = 32, num_channels: int = 64, clip: float = 5.0):
        super().__init__()
        self.z_dim = z_dim
        self.num_channels = num_channels
        self.clip = clip
        self.fc_gamma = nn.Linear(z_dim, num_channels)
        self.fc_beta = nn.Linear(z_dim, num_channels)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """z: (B, z_dim) → γ, β: (B, num_channels)."""
        gamma = torch.tanh(self.fc_gamma(z)) * self.clip
        beta = torch.tanh(self.fc_beta(z)) * self.clip
        return gamma, beta


def apply_film(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """FiLM 적용: γ * x + β. x: (..., C), γ, β: (..., C) or (C,)."""
    return gamma * x + beta
