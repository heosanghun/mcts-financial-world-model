"""
Phase 5: System 1 — Mamba (Selective SSM)
논문: Δ_t = Softplus(Linear(x_t)), Recurrent O(1), Layer 4, State 16, Dim 64.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class MambaBlock(nn.Module):
    """선택적 SSM 블록: Δ_t = Softplus(Linear(x)), recurrent step O(1)."""

    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.delta_proj = nn.Linear(d_model, d_model)
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        self.out_proj = nn.Linear(d_model, d_model)
        self.A = nn.Parameter(torch.randn(d_model, d_state) * 0.01)

    def forward_step(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Recurrent step: x (B, d_model), h (B, d_state) → out (B, d_model), h_new."""
        B = x.size(0)
        device = x.device
        if h is None:
            h = torch.zeros(B, self.d_state, device=device, dtype=x.dtype)
        delta = torch.nn.functional.softplus(self.delta_proj(x))
        B_t = self.B_proj(x)
        C_t = self.C_proj(x)
        # h_new = (1 - delta * A) * h + delta * B_t * x (단순화)
        h = h * (1 - delta.mean(1, keepdim=True)) + delta.mean(1, keepdim=True) * B_t * torch.sigmoid(self.A.mean(0))
        y = (h * C_t).sum(1, keepdim=True).expand(-1, self.d_model) * x
        return self.out_proj(y), h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) → (B, T, d_model). Sequential step (training)."""
        B, T, D = x.shape
        out = []
        h = None
        for t in range(T):
            y, h = self.forward_step(x[:, t, :], h)
            out.append(y)
        return torch.stack(out, dim=1)


class MambaFiLM(nn.Module):
    """Mamba + FiLM: γ, β로 채널별 변조 후 Mamba. 논문: System 1 실행부."""

    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 16,
        num_layers: int = 4,
        num_actions: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.blocks = nn.ModuleList([MambaBlock(d_model, d_state) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, num_actions)
        self.num_actions = num_actions

    def forward(
        self,
        x: torch.Tensor,
        gamma: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x: (B, T, d_model), γ, β: (B, d_model) or None → (B, num_actions)."""
        for block in self.blocks:
            if gamma is not None and beta is not None:
                x = gamma.unsqueeze(1) * x + beta.unsqueeze(1)
            x = block(x) + x
        out = x[:, -1, :]
        return self.fc_out(out)
