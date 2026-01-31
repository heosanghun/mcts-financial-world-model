"""
Phase 7: 정성적 분석 (XAI)
논문: 위기 전이 경로 시각화, FiLM 파라미터 히트맵.
"""
from __future__ import annotations

import numpy as np
from typing import Optional, List, Dict, Any

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def visualize_crisis_path(
    graph_adj: np.ndarray,
    node_labels: Optional[List[str]] = None,
    activated_edges: Optional[List[tuple[int, int]]] = None,
    save_path: Optional[str] = None,
) -> None:
    """위기 전이 경로 시각화: 하이퍼그래프 상 위험 전파."""
    if not _HAS_MPL:
        return
    try:
        import networkx as nx
        G = nx.DiGraph()
        n = graph_adj.shape[0]
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(n):
                if graph_adj[i, j] > 0:
                    G.add_edge(i, j, weight=graph_adj[i, j])
        pos = nx.spring_layout(G, seed=42) if G.number_of_edges() > 0 else {i: (i % 3, i // 3) for i in range(n)}
        labels = (node_labels if node_labels and len(node_labels) >= n else None) or [str(i) for i in range(n)]
        nx.draw(G, pos, with_labels=True, labels=dict(zip(range(n), labels[:n])))
        if save_path:
            plt.savefig(save_path)
            plt.close()
    except Exception:
        pass


def film_heatmap(
    gamma_history: np.ndarray,
    beta_history: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> None:
    """FiLM 파라미터 (γ, β) 변화 추이 히트맵."""
    if not _HAS_MPL or gamma_history.size == 0:
        return
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        if gamma_history.ndim == 1:
            gamma_history = gamma_history[:, np.newaxis]
        if beta_history.ndim == 1:
            beta_history = beta_history[:, np.newaxis]
        ax1.imshow(gamma_history.T, aspect="auto")
        ax1.set_ylabel("gamma")
        ax2.imshow(beta_history.T, aspect="auto")
        ax2.set_ylabel("beta")
        if save_path:
            plt.savefig(save_path)
            plt.close()
    except Exception:
        pass
