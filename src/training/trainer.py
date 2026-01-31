"""
Phase 6: 학습 — AdamW 5e-4, Cosine, Batch 256, 보상(Turnover/MDD/윤리).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class Trainer:
    """System 1 + FiLM 학습: 보상 = Return - λ_risk*Penalty - MDD - Turnover."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        lambda_risk: float = 0.5,
        mdd_threshold: float = 0.03,
        turnover_penalty: float = 0.01,
        ethical_mdd_threshold: float = 0.10,
        device: Optional[str] = None,
    ):
        self.model = model.to(device or "cpu")
        self.device = device or "cpu"
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=1000)
        self.batch_size = batch_size
        self.lambda_risk = lambda_risk
        self.mdd_threshold = mdd_threshold
        self.turnover_penalty = turnover_penalty
        self.ethical_mdd_threshold = ethical_mdd_threshold
        self.criterion = nn.CrossEntropyLoss()

    def train_step(
        self,
        lob: torch.Tensor,
        tick: torch.Tensor,
        context: Optional[torch.Tensor],
        gamma: torch.Tensor,
        beta: torch.Tensor,
        action: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
        mdd: Optional[torch.Tensor] = None,
        turnover: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """한 스텝 학습. 규제 위반(위기 시 공격 행동) 시 윤리 페널티 −∞ 대신 큰 가중치."""
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(lob, tick, context, gamma=gamma, beta=beta)
        loss = self.criterion(logits, action)
        if returns is not None and mdd is not None:
            R = returns.mean()
            pen = torch.clamp(mdd - self.mdd_threshold, min=0.0).mean() * 10.0
            if turnover is not None:
                pen = pen + turnover.mean() * 0.01
            loss = loss - R + self.lambda_risk * pen
        if mdd is not None and mdd.numel() == action.numel():
            ethical = (mdd.squeeze() > self.ethical_mdd_threshold) & (action == 2)
            if ethical.any():
                loss = loss + (logits[ethical, 2] - logits[ethical, 1]).clamp(min=0).mean() * 100.0
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        return {"loss": loss.item()}
