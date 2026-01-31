"""
Phase 5: System 1 — 실시간 주문 집행
논문: Mamba(γ, β) + CNN(LOB) → 매수/매도/관망 신호.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .encoder import MultiModalMarketEncoder
from .mamba import MambaFiLM


class System1Executor(nn.Module):
    """System 1: 다중 모달 인코더 → MambaFiLM(γ, β) → 3-class (매수/매도/관망)."""

    def __init__(
        self,
        encoder: Optional[MultiModalMarketEncoder] = None,
        mamba: Optional[MambaFiLM] = None,
        model_dim: int = 64,
        num_actions: int = 3,
    ):
        super().__init__()
        self.encoder = encoder or MultiModalMarketEncoder(model_dim=model_dim)
        self.model_dim = self.encoder.model_dim
        self.mamba = mamba or MambaFiLM(d_model=model_dim, num_actions=num_actions)
        self.num_actions = num_actions

    def forward(
        self,
        lob: torch.Tensor,
        tick: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        gamma: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """lob (B,T,10,2), tick (B,T,2), context (B,768), γ, β (B, d_model) → logits (B, 3)."""
        h = self.encoder(lob, tick, context)
        h = h.unsqueeze(1)
        return self.mamba(h, gamma=gamma, beta=beta)

    def act(
        self,
        lob: torch.Tensor,
        tick: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        gamma: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """행동 선택: logits, action (0=매도, 1=관망, 2=매수)."""
        logits = self.forward(lob, tick, context, gamma, beta)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = torch.distributions.Categorical(logits=logits).sample()
        return logits, action
