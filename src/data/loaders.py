"""
Phase 1: 데이터 수집·로딩 파이프라인
논문: S&P 500, BTC/USDT, 2000-2024, 틱·LOB·거시·뉴스
공개 데이터: yfinance OHLC + 거시; 바이낸스 API OHLC/LOB; LOB 과거는 합성.
"""
from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime

try:
    import yfinance as yf
except ImportError:
    yf = None

# 바이낸스 REST (공개 시세는 API 키 불필요; 키는 .env BINANCE_API_KEY, BINANCE_SECRET_KEY)
BINANCE_BASE = "https://api.binance.com/api/v3"
BINANCE_INTERVAL_MAP = {"1d": "1d", "1h": "1h", "1m": "1m", "5m": "5m"}


def get_date_range(start: str = "2000-01-01", end: str = "2024-12-31") -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(start), pd.Timestamp(end)


def _binance_symbol(yf_symbol: str) -> Optional[str]:
    """yfinance 심볼 -> 바이낸스 심볼. 지원: BTC-USD -> BTCUSDT 등."""
    m = {"BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT", "BNB-USD": "BNBUSDT", "SOL-USD": "SOLUSDT"}
    return m.get(yf_symbol) or (yf_symbol.replace("-", "") + "USDT" if "-" in yf_symbol else None)


def load_ohlc_binance(
    symbol: str,
    start: str = "2000-01-01",
    end: str = "2024-12-31",
    interval: str = "1d",
    limit_per_request: int = 1000,
) -> Optional[pd.DataFrame]:
    """
    바이낸스 API로 OHLCV 조회. symbol은 바이낸스 형식 (예: BTCUSDT).
    반환: open, high, low, close, volume, index=datetime.
    """
    bn_interval = BINANCE_INTERVAL_MAP.get(interval, "1d")
    start_ts = int(pd.Timestamp(start).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end).timestamp() * 1000)
    try:
        import urllib.request
        import json
    except ImportError:
        return None
    rows = []
    while start_ts < end_ts:
        url = f"{BINANCE_BASE}/klines?symbol={symbol}&interval={bn_interval}&startTime={start_ts}&endTime={end_ts}&limit={limit_per_request}"
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read().decode())
        except Exception:
            break
        if not data:
            break
        for k in data:
            # [open_time, open, high, low, close, volume, ...]
            t_ms = int(k[0])
            rows.append({
                "open": float(k[1]), "high": float(k[2]), "low": float(k[3]),
                "close": float(k[4]), "volume": float(k[5]),
                "date": pd.Timestamp(t_ms, unit="ms"),
            })
        start_ts = int(data[-1][0]) + 1
        if len(data) < limit_per_request:
            break
        time.sleep(0.2)
    if not rows:
        return None
    df = pd.DataFrame(rows).drop_duplicates(subset=["date"]).sort_values("date")
    df = df.set_index("date")[["open", "high", "low", "close", "volume"]]
    return df


def load_lob_binance_snapshot(symbol: str = "BTCUSDT", limit: int = 10) -> Optional[np.ndarray]:
    """
    바이낸스 현재 호가창 1스냅샷. 반환: (1, limit, 4) = bid_price, bid_size, ask_price, ask_size.
    과거 시계열이 필요하면 load_lob_synthetic 사용.
    """
    try:
        import urllib.request
        import json
    except ImportError:
        return None
    url = f"{BINANCE_BASE}/depth?symbol={symbol}&limit={limit}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            d = json.loads(resp.read().decode())
    except Exception:
        return None
    bids = [[float(p), float(q)] for p, q in d.get("bids", [])[:limit]]
    asks = [[float(p), float(q)] for p, q in d.get("asks", [])[:limit]]
    n = min(len(bids), len(asks), limit)
    if n == 0:
        return None
    lob = np.zeros((1, limit, 4), dtype=np.float32)
    for i in range(n):
        lob[0, i, 0] = bids[i][0]
        lob[0, i, 1] = bids[i][1]
        lob[0, i, 2] = asks[i][0]
        lob[0, i, 3] = asks[i][1]
    return lob


def collect_lob_binance_series(
    symbol: str = "BTCUSDT",
    limit: int = 10,
    duration_sec: int = 120,
    interval_ms: int = 100,
) -> Optional[np.ndarray]:
    """
    논문 일치용: 바이낸스 실시간 LOB를 interval_ms 간격으로 duration_sec 동안 수집.
    반환: (T, limit, 4) 실거래소 100ms 스냅샷 시계열.
    """
    interval_sec = max(0.05, interval_ms / 1000.0)
    target_n = max(60, int(duration_sec / interval_sec))
    rows = []
    t0 = time.perf_counter()
    while len(rows) < target_n and (time.perf_counter() - t0) < duration_sec + 5:
        snap = load_lob_binance_snapshot(symbol=symbol, limit=limit)
        if snap is not None and snap.shape[1] == limit:
            rows.append(snap[0])
        time.sleep(interval_sec)
    if len(rows) < 60:
        return None
    lob = np.stack(rows, axis=0).astype(np.float32)
    return lob.reshape(len(lob), limit, 4)


def load_ohlc(
    symbols: list[str],
    start: str = "2000-01-01",
    end: str = "2024-12-31",
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """OHLCV 로드. symbols 예: ['^GSPC', 'BTC-USD']."""
    if yf is None:
        raise ImportError("yfinance required: pip install yfinance")
    start_ts, end_ts = get_date_range(start, end)
    out = {}
    for sym in symbols:
        try:
            t = yf.Ticker(sym)
            df = t.history(start=start_ts, end=end_ts, interval=interval, auto_adjust=True)
        except Exception:
            continue
        if df.empty:
            continue
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        out[sym] = df
    return out


def load_macro(
    tickers: list[str],
    start: str = "2000-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """거시 지표 로드. tickers 예: ['^VIX', 'DX-Y.NYB', '^TNX']."""
    if yf is None:
        raise ImportError("yfinance required: pip install yfinance")
    start_ts, end_ts = get_date_range(start, end)
    dfs = []
    for t in tickers:
        try:
            df = yf.download(t, start=start_ts, end=end_ts, progress=False, auto_adjust=True)
            if df.empty:
                continue
            close = df["Close"] if "Close" in df.columns else df.iloc[:, -1]
            close.name = t
            dfs.append(close)
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    macro = pd.concat(dfs, axis=1).dropna(how="all").ffill().bfill()
    return macro


def load_lob_synthetic(
    price_series: np.ndarray,
    volume_series: np.ndarray,
    n_levels: int = 10,
    n_snapshots: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    OHLC/Volume 기반 합성 LOB 스냅샷.
    반환: (T, n_levels, 4) — 4 = bid_price, bid_size, ask_price, ask_size (논문 LOB 10단계).
    """
    rng = np.random.default_rng(seed)
    T = len(price_series)
    if n_snapshots is not None:
        idx = np.linspace(0, T - 1, n_snapshots, dtype=int)
        price_series = price_series[idx]
        volume_series = volume_series[idx]
        T = len(price_series)
    # 스프레드 비율 (가격 대비)
    spread_ratio = 0.0005 + 0.0002 * rng.random(T)
    lob = np.zeros((T, n_levels, 4))
    for t in range(T):
        mid = float(price_series[t])
        vol = max(1e-6, float(volume_series[t]))
        spread = mid * spread_ratio[t]
        for i in range(n_levels):
            lob[t, i, 0] = mid - spread * (i + 1)  # bid price
            lob[t, i, 1] = vol * (0.5 + 0.5 * rng.random()) / (i + 1)  # bid size
            lob[t, i, 2] = mid + spread * (i + 1)  # ask price
            lob[t, i, 3] = vol * (0.5 + 0.5 * rng.random()) / (i + 1)  # ask size
    return lob.astype(np.float32)


def load_lob_from_file(
    path: str | Path,
    n_levels: int = 10,
    align_length: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    논문 §2.1 LOB와 일치: 실거래소 LOB 파일 로드. (T, n_levels, 4) = bid_p, bid_s, ask_p, ask_s.
    .npy (T, 10, 4) 또는 CSV 40컬럼: bid_p1..bid_p10, bid_s1..bid_s10, ask_p1..ask_p10, ask_s1..ask_s10.
    align_length 지정 시 길이 맞춤.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        if path.suffix.lower() == ".npy":
            lob = np.load(path).astype(np.float32)
        else:
            df = pd.read_csv(path)
            lob = np.zeros((len(df), n_levels, 4), dtype=np.float32)
            for i in range(n_levels):
                for prefix, col in [("bid_p", 0), ("bid_s", 1), ("ask_p", 2), ("ask_s", 3)]:
                    for key in [f"{prefix}{i+1}", f"{prefix}_{i+1}"]:
                        if key in df.columns:
                            lob[:, i, col] = np.asarray(df[key], dtype=np.float32)
                            break
            if lob[:, :, :].max() == 0 and lob[:, :, :].min() == 0:
                return None
        if align_length is not None and lob.shape[0] != align_length:
            if lob.shape[0] >= align_length:
                lob = lob[:align_length]
            else:
                pad = np.zeros((align_length - lob.shape[0], lob.shape[1], lob.shape[2]), dtype=np.float32)
                lob = np.concatenate([lob, pad], axis=0)
        return lob
    except Exception:
        return None


def tick_from_ohlc(
    ohlc: pd.DataFrame,
    ticks_per_day: int = 390,
) -> tuple[np.ndarray, np.ndarray]:
    """OHLC를 틱 수준으로 보간 (체결가·체결량 시퀀스). 논문: 틱 단위 입력용."""
    n = len(ohlc) * ticks_per_day
    close = ohlc["close"].values
    volume = ohlc["volume"].values
    t_idx = np.linspace(0, len(ohlc) - 1, n)
    price = np.interp(t_idx, np.arange(len(ohlc)), close)
    vol = np.interp(t_idx, np.arange(len(ohlc)), volume) / max(ticks_per_day, 1)
    return price.astype(np.float32), np.maximum(vol, 1e-6).astype(np.float32)


def tick_from_1m(
    ohlc_1m: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    논문과 일치: 1분봉 실데이터를 틱 대용으로 사용. (close, volume) 그대로 반환.
    반환: (prices, volumes) shape (T_1m,).
    """
    close = ohlc_1m["close"].values.astype(np.float32)
    volume = ohlc_1m["volume"].values.astype(np.float32)
    return close, np.maximum(volume, 1e-6)


# --- 논문 표 6-1 기술통계 기반 합성 데이터 (데이터 부재 시) ---
# 표 6-1: S&P 500 2000-2024 mean 0.03%, std 1.21%, skew -0.98, kurt 12.45, MDD 56.8%
#         BTC 2017-2024 mean 0.18%, std 4.85%, skew -0.45, kurt 18.72, MDD 77.3%
#         VIX mean 19.54, std 8.32; US10Y 3.21%, 1.15%
def _thesis_synthetic_returns(
    n_days: int,
    mean_daily_pct: float = 0.03,
    std_daily_pct: float = 1.21,
    skew: float = -0.98,
    seed: Optional[int] = None,
) -> np.ndarray:
    """논문 표 6-1 수치에 맞춘 일간 수익률 생성 (로그수익률 근사, % 단위 → 소수)."""
    rng = np.random.default_rng(seed)
    mean = mean_daily_pct / 100.0
    std = std_daily_pct / 100.0
    # 왜도 반영: 음수면 하락 꼬리 두껍게 (skew < 0 → 왼쪽 꼬리 두꺼움)
    try:
        from scipy import stats as scipy_stats
        u = rng.uniform(0.001, 0.999, n_days)
        r = scipy_stats.skewnorm.ppf(u, skew, loc=mean, scale=max(std, 1e-6))
    except Exception:
        z = rng.standard_normal(n_days)
        if skew < 0:
            z = np.where(z < 0, z * 1.5, z * 0.7)  # 하락 꼬리 두껍게
        r = mean + std * z
    return np.clip(r.astype(np.float64), -0.20, 0.20)


def generate_thesis_synthetic_ohlc(
    start: str = "2000-01-01",
    end: str = "2024-12-31",
    asset: str = "SP500",
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    논문 표 6-1 기술통계에 맞춘 합성 OHLCV.
    asset: 'SP500' | 'BTC' (기간·통계 다름).
    """
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    days = (end_ts - start_ts).days + 1
    if asset == "BTC":
        # BTC 2017-2024: mean 0.18%, std 4.85%, skew -0.45
        ret = _thesis_synthetic_returns(days, 0.18, 4.85, -0.45, seed)
    else:
        # S&P 500: mean 0.03%, std 1.21%, skew -0.98
        ret = _thesis_synthetic_returns(days, 0.03, 1.21, -0.98, seed)
    close = 100.0 * np.exp(np.cumsum(ret))
    high = close * (1.0 + np.abs(rng.standard_normal(days)) * 0.005)
    low = close * (1.0 - np.abs(rng.standard_normal(days)) * 0.005)
    open_ = np.roll(close, 1)
    open_[0] = 100.0
    volume = np.maximum(1e6 * (1.0 + 0.5 * rng.standard_normal(days)), 1e5)
    idx = pd.date_range(start=start_ts, periods=days, freq="D")
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume
    }, index=idx)


def load_ohlc_thesis_aligned(
    symbols: list[str],
    start: str = "2000-01-01",
    end: str = "2024-12-31",
    interval: str = "1d",
    fallback_synthetic: bool = True,
    seed: Optional[int] = None,
    use_binance: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    논문 데이터 구간·자산에 맞춰 OHLC 로드.
    use_binance=True 이면 암호화폐(BTC-USD 등)는 바이낸스 API 우선, 그 다음 yfinance, 실패 시 표 6-1 합성.
    """
    out: dict[str, pd.DataFrame] = {}
    start_ts, end_ts = get_date_range(start, end)

    for sym in symbols:
        # 1) 암호화폐: 바이낸스 우선 (BINANCE_API_KEY 없어도 공개 시세는 조회 가능)
        bn_sym = _binance_symbol(sym)
        if use_binance and bn_sym:
            df_bn = load_ohlc_binance(bn_sym, start=start, end=end, interval=interval)
            if df_bn is not None and not df_bn.empty and len(df_bn) >= 252:
                out[sym] = df_bn.copy()
                continue
        # 2) yfinance
        if yf is not None:
            try:
                t = yf.Ticker(sym)
                df = t.history(start=start_ts, end=end_ts, interval=interval, auto_adjust=True)
                if df is not None and not df.empty and len(df) >= 252:
                    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
                    df = df[["open", "high", "low", "close", "volume"]].dropna()
                    out[sym] = df
                    continue
            except Exception:
                pass
        # 3) 합성 fallback (이 심볼에 대해 아무 소스도 없을 때만)
        if fallback_synthetic and sym not in out:
            if sym == "BTC-USD":
                btc_start = "2017-01-01" if start < "2017-01-01" else start
                out["BTC-USD"] = generate_thesis_synthetic_ohlc(btc_start, end, "BTC", (seed or 0) + 1)
            elif sym == "^GSPC":
                out["^GSPC"] = generate_thesis_synthetic_ohlc(start, end, "SP500", seed)
            else:
                out[sym] = generate_thesis_synthetic_ohlc(start, end, "SP500", seed)
    if not out and fallback_synthetic:
        if "^GSPC" in symbols or not symbols:
            out["^GSPC"] = generate_thesis_synthetic_ohlc(start, end, "SP500", seed)
        if "BTC-USD" in symbols and "BTC-USD" not in out:
            btc_start = "2017-01-01" if start < "2017-01-01" else start
            out["BTC-USD"] = generate_thesis_synthetic_ohlc(btc_start, end, "BTC", (seed or 0) + 1)
        if not out and symbols:
            out[symbols[0]] = generate_thesis_synthetic_ohlc(start, end, "SP500", seed)
    return out


def load_macro_thesis_aligned(
    tickers: list[str],
    start: str = "2000-01-01",
    end: str = "2024-12-31",
    fallback_synthetic: bool = True,
    seed: Optional[int] = None,
    target_n_cols: int = 15,
) -> pd.DataFrame:
    """
    거시 15종 논문 기준 로드. 항상 15컬럼 반환(부족 시 합성 패딩).
    """
    macro = pd.DataFrame()
    if yf is not None:
        macro = load_macro(tickers[: target_n_cols], start, end)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    days = (end_ts - start_ts).days + 1
    idx = pd.date_range(start=start_ts, periods=days, freq="D")
    rng = np.random.default_rng(seed)
    if macro.empty or len(macro) < 100:
        if fallback_synthetic:
            macro = pd.DataFrame(index=idx)
            for j in range(target_n_cols):
                v = 3.0 + 1.0 * np.cumsum(rng.standard_normal(days) * 0.01)
                macro[f"_macro_{j}"] = np.clip(v, 0.5, 8.0)
            macro.index.name = "Date"
    if not macro.empty and macro.shape[1] < target_n_cols:
        for j in range(macro.shape[1], target_n_cols):
            v = 3.0 + 1.0 * np.cumsum(rng.standard_normal(len(macro)) * 0.01)
            macro[f"_macro_{j}"] = np.clip(v, 0.5, 8.0)
    if not macro.empty:
        macro = macro.reindex(idx).ffill().bfill().fillna(0)
    return macro
