"""
Phase 3: System 2 — 국면 벡터 z 생성
HGNN readout + MCTS 경로 압축(LSTM/GRU) → z 32-dim.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Optional


class RegimeVectorBuilder(nn.Module):
    """국면 벡터 z: h_graph + h_path → z (32-dim)."""

    def __init__(
        self,
        hgnn_out: int = 32,
        path_seq_dim: int = 4,
        path_hidden: int = 32,
        z_dim: int = 32,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.gru = nn.GRU(path_seq_dim, path_hidden, batch_first=True)
        self.fc_graph = nn.Linear(hgnn_out, z_dim // 2)
        self.fc_path = nn.Linear(path_hidden, z_dim // 2)
        self.fc_z = nn.Linear(z_dim, z_dim)

    def forward(
        self,
        h_graph: torch.Tensor,
        path_sequence: torch.Tensor,
    ) -> torch.Tensor:
        """
        h_graph: (H,) 또는 (B, H)
        path_sequence: (B, T, path_seq_dim) — MCTS 행동 시퀀스 (예: 포트폴리오 변화).
        반환: z (B, z_dim) 또는 (z_dim,)
        """
        if h_graph.dim() == 1:
            h_graph = h_graph.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        B = h_graph.size(0)
        _, h_path = self.gru(path_sequence)
        h_path = h_path.squeeze(0)
        a = self.fc_graph(h_graph)
        b = self.fc_path(h_path)
        z = torch.relu(torch.cat([a, b], dim=-1))
        z = self.fc_z(z)
        if squeeze:
            z = z.squeeze(0)
        return z
