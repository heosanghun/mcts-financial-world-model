"""
Phase 5: System 1 — 다중 모달 시장 인코더
논문: LOB CNN 128-dim, Tick 1D-Conv 128-dim, Context 128-dim, Fusion.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class ResBlock2d(nn.Module):
    """ResNet 스타일 잔차 블록: 3x1 Conv -> BN -> ReLU -> 3x1 Conv, + skip."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + x)


class LOBEncoder(nn.Module):
    """LOB 이미지 (T, 10, 2) → 3×1 Conv + ResNet 잔차 블록 + GAP → 128-dim (논문 §3.1.1)."""

    def __init__(self, in_channels: int = 2, out_dim: int = 128, num_res_blocks: int = 2):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(*[ResBlock2d(32) for _ in range(num_res_blocks)])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 10, 2) → (B, out_dim)."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.dim() == 4 and x.size(-1) == 2:
            x = x.permute(0, 3, 1, 2)
        h = self.conv_in(x)
        h = self.res_blocks(h)
        h = self.pool(h)
        return self.fc(h.flatten(1))


class TickEncoder(nn.Module):
    """Tick 시계열 (B, T, 2) → 1D-Conv → 128-dim."""

    def __init__(self, in_dim: int = 2, out_dim: int = 128, kernel: int = 5):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, 64, kernel)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, in_dim) → (B, out_dim)."""
        x = x.permute(0, 2, 1)
        h = torch.relu(self.conv(x))
        h = self.pool(h).flatten(1)
        return self.fc(h)


class ContextEncoder(nn.Module):
    """시맨틱 컨텍스트 768-dim → Linear + Attention → 128-dim."""

    def __init__(self, in_dim: int = 768, out_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.attn = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_dim) or (B, 1, in_dim) → (B, out_dim)."""
        if x.dim() == 3:
            x = x.squeeze(1)
        h = torch.relu(self.proj(x))
        return h


class MultiModalMarketEncoder(nn.Module):
    """다중 모달 인코더: LOB + Tick + Context → Fusion → 단일 벡터 (model_dim)."""

    def __init__(
        self,
        lob_out: int = 128,
        tick_out: int = 128,
        context_in: int = 768,
        context_out: int = 128,
        model_dim: int = 64,
    ):
        super().__init__()
        self.lob_enc = LOBEncoder(out_dim=lob_out)
        self.tick_enc = TickEncoder(out_dim=tick_out)
        self.ctx_enc = ContextEncoder(in_dim=context_in, out_dim=context_out)
        self.fusion = nn.Sequential(
            nn.Linear(lob_out + tick_out + context_out, model_dim * 2),
            nn.ReLU(),
            nn.Linear(model_dim * 2, model_dim),
        )
        self.model_dim = model_dim

    def forward(
        self,
        lob: torch.Tensor,
        tick: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """lob: (B,T,10,2), tick: (B,T,2), context: (B,768) or None → (B, model_dim)."""
        h_lob = self.lob_enc(lob)
        h_tick = self.tick_enc(tick)
        if context is not None:
            h_ctx = self.ctx_enc(context)
        else:
            h_ctx = torch.zeros(h_lob.size(0), self.ctx_enc.proj.out_features, device=lob.device, dtype=lob.dtype)
        h = torch.cat([h_lob, h_tick, h_ctx], dim=1)
        return self.fusion(h)
