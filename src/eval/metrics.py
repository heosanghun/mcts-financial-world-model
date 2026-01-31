"""
Phase 7: 성능 평가 — 논문 표 6-2~6-5
CAGR, Alpha, Sharpe, Sortino, MDD, Win Rate, Profit Factor, Latency, Slippage.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Optional, List, Any


def cagr(returns: np.ndarray, periods_per_year: float = 252.0) -> float:
    """연평균 성장률 (CAGR)."""
    if len(returns) == 0:
        return 0.0
    total = np.prod(1.0 + returns) - 1.0
    n_years = len(returns) / periods_per_year
    if n_years <= 0:
        return 0.0
    return (1.0 + total) ** (1.0 / n_years) - 1.0


def alpha(
    returns: np.ndarray,
    benchmark_returns: np.ndarray,
    risk_free: float = 0.0,
) -> float:
    """알파: 시장 대비 초과 수익률 (간단 버전)."""
    r = np.mean(returns) - risk_free
    b = np.mean(benchmark_returns) - risk_free
    return float(r - b)


def sharpe_ratio(
    returns: np.ndarray,
    risk_free: float = 0.0,
    periods_per_year: float = 252.0,
) -> float:
    """샤프 지수 (연율화)."""
    if len(returns) < 2:
        return 0.0
    excess = np.mean(returns) - risk_free / periods_per_year
    std = np.std(returns)
    if std < 1e-10:
        return 0.0
    return float(excess / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: np.ndarray,
    risk_free: float = 0.0,
    periods_per_year: float = 252.0,
) -> float:
    """소르티노 지수 (하방 변동성만)."""
    if len(returns) < 2:
        return 0.0
    excess = np.mean(returns) - risk_free / periods_per_year
    downside = returns[returns < 0]
    if len(downside) == 0:
        return 10.0
    std_d = np.std(downside)
    if std_d < 1e-10:
        return 0.0
    return float(excess / std_d * np.sqrt(periods_per_year))


def max_drawdown(cumulative_returns: np.ndarray) -> float:
    """최대 낙폭 (MDD). wealth = 1 + cumulative, peak = running max(wealth), dd = (peak - wealth)/peak."""
    if len(cumulative_returns) == 0:
        return 0.0
    wealth = 1.0 + np.asarray(cumulative_returns, dtype=np.float64)
    peak = np.maximum.accumulate(wealth)
    dd = (peak - wealth) / (peak + 1e-10)
    return float(np.max(dd))


def win_rate(returns: np.ndarray) -> float:
    """승률: 양수 수익률 비중."""
    if len(returns) == 0:
        return 0.0
    return float(np.mean(returns > 0))


def profit_factor(returns: np.ndarray) -> float:
    """손익비: 총 이익 / 총 손실."""
    gains = returns[returns > 0].sum()
    losses = np.abs(returns[returns < 0].sum())
    if losses < 1e-10:
        return 10.0
    return float(gains / losses)


def annualized_vol(returns: np.ndarray, periods_per_year: float = 252.0) -> float:
    """연율화 변동성 (논문 표 6-2 Vol. Ann. %)."""
    if len(returns) < 2:
        return 0.0
    return float(np.std(returns) * np.sqrt(periods_per_year))


def calmar_ratio(
    returns: np.ndarray,
    periods_per_year: float = 252.0,
) -> float:
    """Calmar Ratio = CAGR / |MDD| (논문 안정성 지표)."""
    c = cagr(returns, periods_per_year)
    cum = np.cumprod(1.0 + returns) - 1.0
    mdd = max_drawdown(cum)
    if mdd <= 0:
        return 0.0
    return float(c / mdd)


def compute_all_metrics(
    returns: np.ndarray,
    benchmark_returns: Optional[np.ndarray] = None,
    risk_free: float = 0.0,
    periods_per_year: float = 252.0,
) -> Dict[str, float]:
    """논문 정량 지표 일괄 계산 (표 6-2 형식)."""
    cum = np.cumprod(1.0 + returns) - 1.0
    if benchmark_returns is None:
        benchmark_returns = np.zeros_like(returns)
    mdd = max_drawdown(cum)
    c = cagr(returns, periods_per_year)
    return {
        "cagr": c,
        "alpha": alpha(returns, benchmark_returns, risk_free),
        "vol_ann": annualized_vol(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, risk_free, periods_per_year),
        "sortino": sortino_ratio(returns, risk_free, periods_per_year),
        "mdd": mdd,
        "calmar": calmar_ratio(returns, periods_per_year),
        "win_rate": win_rate(returns),
        "profit_factor": profit_factor(returns),
        "cumulative_return": float(cum[-1]) if len(cum) > 0 else 0.0,
    }


def regime_labels(
    returns: np.ndarray,
    bull_thresh: float = 0.0005,
    bear_thresh: float = -0.0005,
) -> np.ndarray:
    """
    시장 국면 라벨: 1=상승장(Bull), 0=횡보장(Sideways), -1=하락장(Bear).
    논문 표 6-4: 상승장/횡보장/하락장별 Win Rate, P/L Ratio.
    """
    out = np.zeros(len(returns), dtype=np.int32)
    out[returns > bull_thresh] = 1
    out[returns < bear_thresh] = -1
    return out


def regime_metrics(
    strategy_returns: np.ndarray,
    market_returns: np.ndarray,
    bull_thresh: float = 0.0005,
    bear_thresh: float = -0.0005,
) -> Dict[str, Dict[str, float]]:
    """
    시장 국면별 성과 (논문 표 6-4).
    반환: {"bull": {win_rate, profit_factor, ...}, "sideways": ..., "bear": ...}
    """
    reg = regime_labels(market_returns, bull_thresh, bear_thresh)
    n = len(strategy_returns)
    if len(reg) != n:
        reg = reg[:n] if len(reg) > n else np.pad(reg, (0, n - len(reg)), constant_values=0)
    out = {}
    for name, mask_val in [("bull", 1), ("sideways", 0), ("bear", -1)]:
        mask = reg == mask_val
        if mask.sum() < 2:
            out[name] = {"win_rate": 0.0, "profit_factor": 0.0, "n_obs": 0}
            continue
        ret = strategy_returns[mask]
        out[name] = {
            "win_rate": win_rate(ret),
            "profit_factor": profit_factor(ret),
            "n_obs": int(mask.sum()),
        }
    return out


# --- P2: Latency 측정 (호출측에서 타이밍 수집 후 전달) ---
def latency_stats(timings_ms: np.ndarray) -> Dict[str, float]:
    """추론 지연(ms) 배열 → mean, std, p50, p99 (밀리초)."""
    if len(timings_ms) == 0:
        return {"latency_mean_ms": 0.0, "latency_std_ms": 0.0, "latency_p50_ms": 0.0, "latency_p99_ms": 0.0}
    return {
        "latency_mean_ms": float(np.mean(timings_ms)),
        "latency_std_ms": float(np.std(timings_ms)),
        "latency_p50_ms": float(np.percentile(timings_ms, 50)),
        "latency_p99_ms": float(np.percentile(timings_ms, 99)),
    }


# --- P3: Slippage 민감도 ---
def apply_slippage(returns: np.ndarray, position_changes: np.ndarray, bps: float) -> np.ndarray:
    """거래 비용 적용: position_changes 절대값 * (bps/10000). returns와 동일 길이."""
    cost = np.abs(position_changes) * (bps / 10000.0)
    return returns - cost


def slippage_sensitivity(
    strategy_returns: np.ndarray,
    position_changes: np.ndarray,
    bps_list: Optional[List[float]] = None,
    benchmark_returns: Optional[np.ndarray] = None,
    periods_per_year: float = 252.0,
) -> Dict[float, Dict[str, float]]:
    """여러 슬리피지(bps)에서 수익률·지표 재계산. bps_list 예: [0, 5, 10, 20]."""
    if bps_list is None:
        bps_list = [0.0, 5.0, 10.0, 20.0, 50.0]
    out = {}
    for bps in bps_list:
        net_ret = apply_slippage(strategy_returns, position_changes, bps)
        out[bps] = compute_all_metrics(net_ret, benchmark_returns, periods_per_year=periods_per_year)
    return out
