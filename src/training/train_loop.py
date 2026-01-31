"""
P1: 학습 루프 — 에폭·배치 루프, 라벨(수익률 방향 기반)
논문: AdamW 5e-4, Cosine, Batch 256, 보상(Turnover/MDD).
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Optional, Dict, Any, Tuple
from pathlib import Path


def make_labels_from_returns(
    returns: np.ndarray,
    threshold: float = 0.0,
) -> np.ndarray:
    """다음 수익률 방향 → 라벨: 0=매도, 1=관망, 2=매수. threshold 근처는 관망."""
    labels = np.ones(len(returns), dtype=np.int64)
    labels[returns > threshold] = 2
    labels[returns < -threshold] = 0
    return labels


def run_training_loop(
    model: Any,
    trainer: Any,
    lob_img: np.ndarray,
    ret: np.ndarray,
    vol_z: np.ndarray,
    gamma_batch: Optional[np.ndarray] = None,
    beta_batch: Optional[np.ndarray] = None,
    lookback: int = 60,
    batch_size: int = 256,
    epochs: int = 5,
    device: str = "cpu",
    out_dir: Optional[Path] = None,
) -> Dict[str, list]:
    """
    학습 루프: 배치 샘플링 → 라벨(수익률 방향) → train_step 반복.
    gamma_batch, beta_batch: (T,) 또는 (T, d) — 없으면 1, 0 사용.
    """
    T = min(len(ret) - lookback - 1, lob_img.shape[0] - lookback - 1)
    if T <= 0:
        return {"loss": []}
    labels = make_labels_from_returns(ret[lookback + 1 : lookback + T + 1])
    indices = np.arange(lookback, lookback + T)
    d_model = getattr(model, "model_dim", 64)
    losses = []
    for epoch in range(epochs):
        perm = np.random.permutation(len(indices))
        epoch_loss = []
        for start in range(0, len(perm), batch_size):
            idx = perm[start : start + batch_size]
            if len(idx) < 2:
                continue
            base_idx = indices[idx]
            lob_list = []
            tick_list = []
            for i in base_idx:
                lob_list.append(lob_img[i - lookback : i + 1])
                tick_list.append(np.column_stack([ret[i - lookback : i + 1], vol_z[i - lookback : i + 1]]))
            lob_b = torch.from_numpy(np.stack(lob_list)).float().to(device)
            tick_b = torch.from_numpy(np.stack(tick_list)).float().to(device)
            pos = np.clip(idx, 0, len(labels) - 1)
            action_b = torch.from_numpy(labels[pos]).long().to(device)
            B = lob_b.size(0)
            if gamma_batch is not None and beta_batch is not None:
                g = torch.from_numpy(np.atleast_2d(gamma_batch)).float().to(device)
                b = torch.from_numpy(np.atleast_2d(beta_batch)).float().to(device)
                if g.dim() == 1 or g.size(0) == 1:
                    g = g.expand(B, d_model)
                    b = b.expand(B, d_model)
                else:
                    g = g[:B].expand(B, d_model)
                    b = b[:B].expand(B, d_model)
            else:
                g = torch.ones(B, d_model, device=device, dtype=torch.float32)
                b = torch.zeros(B, d_model, device=device, dtype=torch.float32)
            out = trainer.train_step(lob_b, tick_b, None, g, b, action_b)
            epoch_loss.append(out["loss"])
        losses.extend(epoch_loss)
        if out_dir and (epoch + 1) % max(1, epochs // 3) == 0:
            torch.save(model.state_dict(), out_dir / f"checkpoint_epoch_{epoch+1}.pt")
    return {"loss": losses}
