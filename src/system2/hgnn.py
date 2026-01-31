"""
Phase 3: System 2 — HGNN (하이퍼그래프 신경망)
논문: 2 layers, 12 hyperedge groups, 노드 임베딩 → readout.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


class HGNN(nn.Module):
    """하이퍼그래프 신경망: 12 하이퍼엣지 그룹 + 2-layer 메시지 전달 → readout (논문 §4.4)."""

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 32,
        num_layers: int = 2,
        num_hyperedge_groups: int = 12,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.num_hyperedge_groups = num_hyperedge_groups
        self.fc_in = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(num_layers - 1)
        ])
        self.fc_out = nn.Linear(hidden_dim, out_dim)

    def _hyperedge_incidence(self, N: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """논문: 12 하이퍼엣지 그룹. 노드 i → 그룹 i % 12. (N, 12) incidence."""
        E = self.num_hyperedge_groups
        H = torch.zeros(N, E, device=device, dtype=dtype)
        for i in range(N):
            H[i, i % E] = 1.0
        return H

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (N, node_dim), adj: (N, N) 인접 행렬 (가중치 포함).
        12 하이퍼엣지 그룹 기반 메시지 전달 + 인접 보조.
        반환: (out_dim,) 그래프 readout.
        """
        N = x.size(0)
        h = torch.relu(self.fc_in(x))
        H = self._hyperedge_incidence(N, x.device, x.dtype)
        E = H.size(1)
        for i, layer in enumerate(self.layers):
            m_adj = torch.mm(adj / (adj.sum(dim=1, keepdim=True) + 1e-8), h)
            D_v_inv = 1.0 / (H.sum(dim=1, keepdim=True) + 1e-8)
            D_e_inv = 1.0 / (H.sum(dim=0, keepdim=True) + 1e-8)
            h_he = torch.mm(H.t(), h * D_v_inv)
            m_he = torch.mm(H, h_he * D_e_inv.t())
            m = 0.5 * m_adj + 0.5 * m_he
            h = torch.relu(layer(torch.cat([h, m], dim=1)))
        h_graph = h.mean(dim=0)
        return self.fc_out(h_graph)

    @staticmethod
    def graph_to_adj(G: nx.DiGraph, node_list: List[str]) -> np.ndarray:
        """네트워크X 그래프 → numpy 인접 행렬 (node_list 순서)."""
        n = len(node_list)
        idx = {u: i for i, u in enumerate(node_list)}
        adj = np.zeros((n, n), dtype=np.float32)
        for u, v in G.edges():
            if u in idx and v in idx:
                adj[idx[u], idx[v]] = G[u][v].get("weight", 1.0)
        return adj
