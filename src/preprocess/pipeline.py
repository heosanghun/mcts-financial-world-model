"""
Phase 2: 전처리 — Log Return, 롤링 Z-Score, LOB 이미지화
논문 표 3-1: OHLC Log Return·Min-Max, LOB Z-Score, Look-ahead 방지.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def log_return(prices: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """가격 → 로그 수익률. 정상성 확보."""
    p = np.asarray(prices, dtype=np.float64)
    r = np.diff(np.log(p + eps))
    return np.concatenate([[0.0], r]).astype(np.float32)


def rolling_zscore(
    x: np.ndarray,
    window: int = 60,
    min_periods: Optional[int] = None,
) -> np.ndarray:
    """롤링 Z-Score. Look-ahead 방지 (과거 윈도우만 사용)."""
    x = np.asarray(x, dtype=np.float64)
    if min_periods is None:
        min_periods = max(1, window // 2)
    n = len(x)
    out = np.zeros_like(x, dtype=np.float32)
    out[:min_periods] = 0.0
    for i in range(min_periods, n):
        start = max(0, i - window)
        seg = x[start : i + 1]
        m, s = seg.mean(), seg.std()
        if s < 1e-10:
            out[i] = 0.0
        else:
            out[i] = (x[i] - m) / s
    return out


def lob_to_image(lob: np.ndarray) -> np.ndarray:
    """
    LOB (T, 10, 4) → 이미지 (T, 10, 2) — 2채널: 매수잔량, 매도잔량.
    논문: CNN 입력용 (T×Price×Volume → 10×2).
    """
    lob = np.asarray(lob, dtype=np.float32)
    if lob.ndim == 2:
        lob = lob[np.newaxis, :, :]
    # (..., 10, 4) -> bid_size, ask_size
    bid_size = lob[..., 1]   # (T, 10)
    ask_size = lob[..., 3]
    img = np.stack([bid_size, ask_size], axis=-1)  # (T, 10, 2)
    return img


def preprocess_ohlc(
    close: np.ndarray,
    volume: np.ndarray,
    rolling_window: int = 60,
    use_minmax: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """OHLC 전처리: Log Return + 롤링 Z-Score + (선택) Min-Max. 논문 표 3-1: Market Data Log Return·Min-Max. 반환: (returns, volume_z). 수익률은 PnL용으로 실제 로그수익률 유지, 거래량에 Min-Max 적용."""
    ret = log_return(close)
    vol_z = rolling_zscore(volume, window=rolling_window)
    if use_minmax:
        vol_z = minmax_scale(vol_z)
    return ret, vol_z


def preprocess_lob(
    lob: np.ndarray,
    rolling_window: int = 60,
) -> np.ndarray:
    """LOB 전처리: 채널별 롤링 Z-Score 후 이미지화. (T, 10, 4) → (T, 10, 2)."""
    T, L, C = lob.shape
    out = np.zeros_like(lob, dtype=np.float32)
    for c in range(C):
        col = lob[:, :, c].ravel()
        out[:, :, c] = rolling_zscore(col, window=rolling_window).reshape(T, L)
    return lob_to_image(out)


def minmax_scale(x: np.ndarray, axis: Optional[int] = None, eps: float = 1e-8) -> np.ndarray:
    """Min-Max 스케일 [0,1]. 논문: Market Data 정규화."""
    x = np.asarray(x, dtype=np.float64)
    if axis is not None:
        mn = x.min(axis=axis, keepdims=True)
        mx = x.max(axis=axis, keepdims=True)
    else:
        mn, mx = x.min(), x.max()
    return ((x - mn) / (mx - mn + eps)).astype(np.float32)
