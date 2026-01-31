"""
Phase 6: Slow Loop / Fast Loop 통합
논문: Slow 1h~4h (그래프, MCTS, z), Fast ~1ms (Mamba + FiLM).
"""
from __future__ import annotations

import torch
import numpy as np
from typing import Optional, Tuple, Dict, Any

# Lazy imports to avoid circular deps
def run_slow_loop_step(
    series_dict: Dict[str, np.ndarray],
    hgnn: Any,
    regime_builder: Any,
    mcts: Any,
    film_gen: Any,
    policy_buffer: Any,
    device: str = "cpu",
    use_stress: bool = True,
    stress_epsilon: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """System 2: 그래프 구축 → (선택) 반사실 스트레스 → HGNN → MCTS → z → FiLM → Policy Buffer write."""
    from ..system2.graph_build import build_hybrid_graph
    from ..system2.hgnn import HGNN
    from ..system2.stress_test import run_stress_propagation
    node_list = list(series_dict.keys())
    import os
    G = build_hybrid_graph(
        series_dict, maxlag=5, p_threshold=0.05,
        llm_verify=True, llm_rule_based=True,
        llm_use_openai=bool(os.environ.get("OPENAI_API_KEY", "").strip()),
    )
    adj = HGNN.graph_to_adj(G, node_list)
    n = len(node_list)
    node_dim = 5
    x = np.zeros((n, node_dim), dtype=np.float32)
    for i, (k, v) in enumerate(series_dict.items()):
        arr = np.asarray(v).ravel()
        x[i, 0] = np.mean(arr)
        x[i, 1] = np.std(arr) if len(arr) > 1 else 0.0
        x[i, 2] = arr[-1] if len(arr) else 0.0
        x[i, 3] = np.min(arr)
        x[i, 4] = np.max(arr)
    if use_stress and n > 0:
        source_indices = [i for i in range(n) if "_macro" in str(node_list[i]) or i >= max(0, n - 3)]
        if not source_indices:
            source_indices = [0]
        x = run_stress_propagation(adj, x, source_indices, epsilon=stress_epsilon, steps=2)
    x_t = torch.from_numpy(x).float().to(device)
    adj_t = torch.from_numpy(adj).float().to(device)
    with torch.no_grad():
        h_graph = hgnn(x_t, adj_t)
    state = np.array([0.5, 0.0])
    _, path, _ = mcts.run(state)
    T_path = 20
    path_seq_dim = 4
    path_arr = np.zeros((1, T_path, path_seq_dim), dtype=np.float32)
    for t, a in enumerate(path[:T_path]):
        path_arr[0, t, 0] = float(a) / max(1, mcts.n_actions - 1)
        path_arr[0, t, 1] = 1.0
    path_t = torch.from_numpy(path_arr).float().to(device)
    with torch.no_grad():
        z = regime_builder(h_graph, path_t)
        gamma, beta = film_gen(z)
    policy_buffer.write(gamma.cpu().numpy(), beta.cpu().numpy())
    return gamma.cpu().numpy(), beta.cpu().numpy()


def run_fast_loop_step(
    model: Any,
    lob: torch.Tensor,
    tick: torch.Tensor,
    policy_buffer: Any,
    context: Optional[torch.Tensor] = None,
    device: str = "cpu",
) -> Tuple[torch.Tensor, int]:
    """System 1: Policy Buffer read (γ, β) → MambaFiLM → 매수/매도/관망."""
    gamma, beta = policy_buffer.read()
    gamma_t = torch.from_numpy(np.atleast_1d(gamma)).float().to(device)
    beta_t = torch.from_numpy(np.atleast_1d(beta)).float().to(device)
    d = getattr(model, "model_dim", gamma_t.numel())
    if gamma_t.numel() == 1:
        gamma_t = gamma_t.expand(d)
        beta_t = beta_t.expand(d)
    if lob.dim() == 3:
        lob = lob.unsqueeze(0)
    if tick.dim() == 2:
        tick = tick.unsqueeze(0)
    if gamma_t.dim() == 1:
        gamma_t = gamma_t.unsqueeze(0)
        beta_t = beta_t.unsqueeze(0)
    logits = model(lob, tick, context, gamma=gamma_t, beta=beta_t)
    action = logits.argmax(dim=-1).item()
    return logits, action
