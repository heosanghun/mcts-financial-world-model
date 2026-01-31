"""
Phase 3: System 2 — MCTS (몬테카를로 트리 탐색)
논문: UCB 선택, Horizon 20, 보상 = Return - λ_risk*Penalty - MDD/Turnover/윤리.
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass


@dataclass
class MCTSNode:
    state: np.ndarray  # 포트폴리오 상태 (예: 현금 비중 등)
    parent: Optional["MCTSNode"] = None
    action: Optional[int] = None
    children: List["MCTSNode"] = None
    visits: int = 0
    value: float = 0.0
    reward: float = 0.0

    def __post_init__(self):
        if self.children is None:
            self.children = []


class MCTSPlanner:
    """그래프 기반 MCTS: 반사실적 시나리오 rollout, 최적 생존 경로."""

    def __init__(
        self,
        n_actions: int = 5,
        horizon: int = 20,
        n_simulations: int = 500,
        c_ucb: float = 1.4,
        gamma: float = 0.99,
        lambda_risk: float = 0.5,
        mdd_threshold: float = 0.03,
        var_percentile: float = 5.0,
        ethical_mdd_critical: float = 0.10,
    ):
        self.n_actions = n_actions
        self.horizon = horizon
        self.n_simulations = n_simulations
        self.c_ucb = c_ucb
        self.gamma = gamma
        self.lambda_risk = lambda_risk
        self.mdd_threshold = mdd_threshold
        self.var_percentile = var_percentile
        self.ethical_mdd_critical = ethical_mdd_critical

    def _reward(
        self,
        returns: np.ndarray,
        risk_penalty: float = 0.0,
        mdd: float = 0.0,
        turnover: float = 0.0,
        action: Optional[int] = None,
    ) -> float:
        """보상 = Return - λ_risk*Penalty - MDD - Turnover - VaR/ES 초과 페널티. 규제 위반 시 −∞."""
        if action is not None and mdd >= self.ethical_mdd_critical and action >= self.n_actions - 1:
            return -np.inf
        R = np.sum(returns)
        pen = risk_penalty + max(0, mdd - self.mdd_threshold) * 10.0 + turnover * 0.01
        var_val = np.percentile(returns, self.var_percentile)
        if var_val < -0.05:
            pen += np.exp(-var_val - 0.05)
        return R - self.lambda_risk * pen

    def _ucb_select(self, node: MCTSNode) -> MCTSNode:
        """UCB로 자식 선택."""
        if not node.children:
            return node
        best = None
        best_score = -np.inf
        for c in node.children:
            if c.visits == 0:
                return c
            ucb = c.value / (c.visits + 1e-8) + self.c_ucb * np.sqrt(np.log(node.visits + 1) / (c.visits + 1e-8))
            if ucb > best_score:
                best_score = ucb
                best = c
        return best or node.children[0]

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """다음 포트폴리오 상태 노드 추가."""
        for a in range(self.n_actions):
            # 상태 전이: 단순화 (현금 비중 +a/10 등)
            new_state = node.state + np.array([0.1 * (a - self.n_actions // 2), 0.0])
            new_state = np.clip(new_state, 0.0, 1.0)
            child = MCTSNode(state=new_state, parent=node, action=a)
            node.children.append(child)
        return node.children[0]

    def _simulate(self, state: np.ndarray) -> Tuple[float, np.ndarray]:
        """반사실적 위기 시나리오 rollout, Horizon 20, Discount."""
        returns = np.zeros(self.horizon, dtype=np.float32)
        s = state.copy()
        for t in range(self.horizon):
            # 랜덤 수익률 시뮬레이션 (스트레스: 음의 편향)
            r = np.random.randn() * 0.02 - 0.005
            returns[t] = r
            s = s * (1 + r)
        mdd = 0.0
        peak = 1.0
        cum = 1.0
        for r in returns:
            cum *= 1 + r
            peak = max(peak, cum)
            mdd = max(mdd, (peak - cum) / peak)
        discount = np.array([self.gamma ** t for t in range(self.horizon)])
        R = self._reward(returns * discount, mdd=mdd, action=None)
        return R, returns

    def _backup(self, node: MCTSNode, value: float) -> None:
        """보상으로 V(s,a) 업데이트."""
        while node is not None:
            node.visits += 1
            node.value += value
            value *= self.gamma
            node = node.parent

    def run(
        self,
        initial_state: np.ndarray,
    ) -> Tuple[np.ndarray, List[int], float]:
        """
        MCTS 실행. 반환: (최종 상태, 행동 시퀀스, 평균 가치).
        """
        root = MCTSNode(state=initial_state)
        for _ in range(self.n_simulations):
            node = root
            # Selection
            while node.children and all(c.visits > 0 for c in node.children):
                node = self._ucb_select(node)
            # Expansion
            if node.visits > 0 and len(node.children) < self.n_actions:
                node = self._expand(node)
            # Simulation
            R, _ = self._simulate(node.state)
            # Backpropagation
            self._backup(node, R)
        # 최적 경로: 가장 많이 방문한 자식 따라가기
        path = []
        node = root
        while node.children:
            best = max(node.children, key=lambda c: c.visits)
            path.append(best.action)
            node = best
        return node.state, path, root.value / (root.visits + 1e-8)
