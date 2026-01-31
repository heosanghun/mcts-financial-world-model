"""
Phase 3: System 2 — 하이브리드 인과 그래프 (Granger + TE, LLM 검증)
논문: Granger p<0.05 directed edge, TE 확장, LLM Accept/Reject/Hold.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from typing import List, Tuple, Optional, Dict, Any

try:
    from statsmodels.tsa.stattools import grangercausalitytests
    _HAS_GRANGER = True
except ImportError:
    _HAS_GRANGER = False


def _granger_pair(x: np.ndarray, y: np.ndarray, maxlag: int, p_threshold: float) -> bool:
    """Granger causality x -> y. p < p_threshold 이면 True."""
    if not _HAS_GRANGER:
        return np.corrcoef(x, y)[0, 1] ** 2 > 0.01  # fallback
    try:
        data = np.column_stack([y, x])
        res = grangercausalitytests(data, maxlag=maxlag, verbose=0)
        for lag in range(1, maxlag + 1):
            p = res[lag][0]["ssr_ftest"][1]
            if p < p_threshold:
                return True
        return False
    except Exception:
        return False


def _transfer_entropy_simple(x: np.ndarray, y: np.ndarray, k: int = 1, bins: int = 5) -> float:
    """간단한 TE 추정: 상관 기반 근사 (실제 TE는 이산화 엔트로피)."""
    n = min(len(x), len(y))
    if n <= k + 1:
        return 0.0
    xk, yk = x[k:], y[:-k] if k else y[:n]
    if len(xk) != len(yk):
        m = min(len(xk), len(yk))
        xk, yk = xk[:m], yk[:m]
    r = np.corrcoef(xk, yk)[0, 1]
    if np.isnan(r):
        return 0.0
    return float(max(0.0, r ** 2))


class GrangerTEBuilder:
    """통계적 뼈대: Granger + TE 엣지 후보."""
    def __init__(self, maxlag: int = 5, p_threshold: float = 0.05, te_bins: int = 5):
        self.maxlag = maxlag
        self.p_threshold = p_threshold
        self.te_bins = te_bins

    def fit(self, series_dict: Dict[str, np.ndarray]) -> nx.DiGraph:
        """series_dict: node_id -> (T,) 시계열. 반환: directed graph."""
        nodes = list(series_dict.keys())
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        for i, ni in enumerate(nodes):
            for j, nj in enumerate(nodes):
                if i == j:
                    continue
                xi = np.asarray(series_dict[ni]).ravel()
                xj = np.asarray(series_dict[nj]).ravel()
                n = min(len(xi), len(xj))
                xi, xj = xi[:n], xj[:n]
                if _granger_pair(xi, xj, self.maxlag, self.p_threshold):
                    te = _transfer_entropy_simple(xi, xj, k=1, bins=self.te_bins)
                    G.add_edge(ni, nj, weight=float(te) + 0.01)
        return G


def llm_verify_edges_placeholder(edges: List[Tuple[str, str, float]], context: str = "") -> List[Tuple[str, str, float]]:
    """LLM 검증 플레이스홀더: Accept만 통과 (모든 후보 유지)."""
    return [(u, v, w) for u, v, w in edges]


def llm_verify_edges_rule_based(
    edges: List[Tuple[str, str, float]],
    accept_threshold: float = 0.1,
    reject_threshold: float = 0.01,
) -> List[Tuple[str, str, float]]:
    """규칙 기반 LLM 검증: Accept(weight>=accept), Reject(weight<reject), Hold(보정 가중치)."""
    out = []
    for u, v, w in edges:
        if w < reject_threshold:
            continue
        if w >= accept_threshold:
            out.append((u, v, w))
        else:
            out.append((u, v, (w + accept_threshold) * 0.5))
    return out


def llm_verify_edges_openai(
    edges: List[Tuple[str, str, float]],
    context: str = "financial time series causal graph",
    model: str = "gpt-4o-mini",
) -> List[Tuple[str, str, float]]:
    """
    OpenAI API 기반 LLM 검증: Accept / Reject / Hold + Confidence Score.
    환경변수 OPENAI_API_KEY 사용. 미설정 시 규칙 기반으로 fallback.
    """
    import os
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or len(edges) == 0:
        return llm_verify_edges_rule_based(edges)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except Exception:
        return llm_verify_edges_rule_based(edges)
    out = []
    # 배치로 묶어서 API 호출 (엣지 많을 때 비용·속도 고려)
    batch_size = 10
    for b in range(0, len(edges), batch_size):
        batch = edges[b : b + batch_size]
        edge_desc = "; ".join([f"({u}->{v}, w={w:.3f})" for u, v, w in batch])
        prompt = (
            f"Context: {context}. You are verifying directed edges for a causal graph. "
            "For each edge below, respond with exactly one word per edge in order: Accept, Reject, or Hold. "
            "Accept = economically plausible causal link. Reject = spurious. Hold = uncertain. "
            "Knowledge cutoff: you have no information after the data date. "
            f"Edges: {edge_desc}. Reply with only the words separated by commas, e.g. Accept, Reject, Hold."
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
            )
            text = (resp.choices[0].message.content or "").strip().lower()
            labels = [x.strip() for x in text.replace(".", ",").split(",") if x.strip()]
        except Exception:
            labels = []
        for i, (u, v, w) in enumerate(batch):
            if i < len(labels):
                lab = labels[i][:6]
                if "reject" in lab:
                    continue
                if "hold" in lab:
                    w = (w + 0.1) * 0.5
            out.append((u, v, w))
    return out if out else llm_verify_edges_rule_based(edges)


def build_hybrid_graph(
    series_dict: Dict[str, np.ndarray],
    maxlag: int = 5,
    p_threshold: float = 0.05,
    llm_verify: bool = False,
    llm_rule_based: bool = True,
    llm_use_openai: bool = True,
) -> nx.DiGraph:
    """하이브리드 그래프: Granger+TE → (선택) LLM 검증(Accept/Reject/Hold). OpenAI 사용 시 llm_use_openai=True 및 OPENAI_API_KEY 설정."""
    builder = GrangerTEBuilder(maxlag=maxlag, p_threshold=p_threshold)
    G = builder.fit(series_dict)
    if llm_verify:
        edges = [(u, v, G[u][v].get("weight", 1.0)) for u, v in G.edges()]
        if llm_use_openai and __import__("os").environ.get("OPENAI_API_KEY", "").strip():
            verified = llm_verify_edges_openai(edges)
        elif llm_rule_based:
            verified = llm_verify_edges_rule_based(edges)
        else:
            verified = llm_verify_edges_placeholder(edges)
        G = nx.DiGraph()
        G.add_nodes_from(list(series_dict.keys()))
        for u, v, w in verified:
            G.add_edge(u, v, weight=w)
    return G
