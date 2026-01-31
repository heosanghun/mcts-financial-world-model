"""
Phase 3: 반사실적 스트레스 테스트 — 노드 교란 ε, 엣지 재가중, HGNN 전파
논문 §4.2.1: 구조적 스트레스 테스트, 희소 경로 탐색.
"""
from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple


def apply_shock_to_nodes(
    node_features: np.ndarray,
    source_indices: List[int],
    epsilon: float = 0.5,
) -> np.ndarray:
    """소스 노드(거시 등)에 충격 ε 가산. node_features: (N, D)."""
    x = np.asarray(node_features, dtype=np.float32).copy()
    for i in source_indices:
        if 0 <= i < x.shape[0]:
            x[i] = x[i] + epsilon * (np.random.randn(x.shape[1]).astype(np.float32) * 0.1 + 1.0)
    return x


def reweight_edges_stress(
    adj: np.ndarray,
    risk_path_multiplier: float = 1.5,
) -> np.ndarray:
    """잠재 위험 경로 활성화: 엣지 재가중 (스트레스 시 강화)."""
    adj = np.asarray(adj, dtype=np.float32).copy()
    adj = adj * risk_path_multiplier
    return np.clip(adj, 0.0, 10.0)


def run_stress_propagation(
    adj: np.ndarray,
    node_features: np.ndarray,
    source_indices: List[int],
    epsilon: float = 0.5,
    steps: int = 2,
) -> np.ndarray:
    """노드 충격 ε 적용 → 엣지 재가중 → 위상 전파 후 노드 특징 반환 (단순 전파)."""
    x = apply_shock_to_nodes(node_features, source_indices, epsilon)
    adj_s = reweight_edges_stress(adj)
    for _ in range(steps):
        x = x + (adj_s / (adj_s.sum(axis=1, keepdims=True) + 1e-8)) @ x
    return x.astype(np.float32)
