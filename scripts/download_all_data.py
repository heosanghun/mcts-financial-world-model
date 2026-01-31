"""
5개 데이터 항목을 로컬 폴더에 다운로드·저장.
폴더: data/ohlc, data/macro, data/lob, data/tick, data/news_semantic
"""
from __future__ import annotations

import os
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# .env 로드
_env = ROOT / ".env"
if _env.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env)
    except ImportError:
        pass

# 폴더 생성
DATA_ROOT = ROOT / "data"
DIRS = {
    "ohlc": DATA_ROOT / "ohlc",
    "macro": DATA_ROOT / "macro",
    "lob": DATA_ROOT / "lob",
    "tick": DATA_ROOT / "tick",
    "news_semantic": DATA_ROOT / "news_semantic",
}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)


def _safe_filename(sym: str) -> str:
    return sym.replace("^", "").replace("/", "-").replace(" ", "_")


def main():
    with open(ROOT / "configs" / "default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    start = cfg["data"]["start_date"]
    end = cfg["data"]["end_date"]
    symbols = cfg["data"]["assets"].get("equity", ["^GSPC"]) + cfg["data"]["assets"].get("crypto", ["BTC-USD"])
    macro_tickers = cfg["data"].get("macro_tickers", [])[:15]
    lob_levels = cfg["data"].get("lob_levels", 10)
    context_dim = cfg["preprocess"].get("context_dim", 768)
    seed = 42

    print("1/5 OHLC 다운로드 ...")
    from src.data.loaders import (
        load_ohlc,
        load_ohlc_binance,
        load_ohlc_thesis_aligned,
        load_macro_thesis_aligned,
        load_lob_synthetic,
        load_lob_binance_snapshot,
        tick_from_ohlc,
        _binance_symbol,
    )
    # 실제 소스 판별: real only 로드 후 비교
    real_ohlc = {}
    for sym in symbols:
        bn = _binance_symbol(sym)
        if bn:
            df_bn = load_ohlc_binance(bn, start=start, end=end, interval="1d")
            if df_bn is not None and len(df_bn) >= 252:
                real_ohlc[sym] = ("binance", df_bn)
                continue
        raw = load_ohlc([sym], start=start, end=end, interval="1d")
        if sym in raw and raw[sym] is not None and len(raw[sym]) >= 252:
            real_ohlc[sym] = ("yfinance", raw[sym])
    ohlc_dict = load_ohlc_thesis_aligned(symbols, start=start, end=end, fallback_synthetic=True, seed=seed)
    if not ohlc_dict:
        from src.data.loaders import generate_thesis_synthetic_ohlc
        ohlc_dict = {"^GSPC": generate_thesis_synthetic_ohlc(start, end, "SP500", seed)}
    for sym, df in ohlc_dict.items():
        path = DIRS["ohlc"] / f"{_safe_filename(sym)}.csv"
        df.to_csv(path, encoding="utf-8")
        source = "synthetic"
        if sym in real_ohlc:
            src_name, real_df = real_ohlc[sym]
            if abs(len(real_df) - len(df)) <= 50:
                source = src_name
        (DIRS["ohlc"] / f"{_safe_filename(sym)}_source.txt").write_text(source, encoding="utf-8")
        print(f"   저장: {path} ({len(df)}행) [소스: {source}]")

    print("2/5 거시 15종 다운로드 ...")
    macro_df = load_macro_thesis_aligned(macro_tickers, start=start, end=end, target_n_cols=15, seed=seed)
    path = DIRS["macro"] / "macro_15.csv"
    macro_df.to_csv(path, encoding="utf-8")
    synth_cols = [c for c in macro_df.columns if str(c).startswith("_macro_")]
    macro_source = "synthetic" if len(synth_cols) == macro_df.shape[1] else ("real_with_synthetic_padding" if synth_cols else "real")
    (DIRS["macro"] / "source.txt").write_text(macro_source, encoding="utf-8")
    print(f"   저장: {path} ({len(macro_df)}행, {macro_df.shape[1]}컬럼) [소스: {macro_source}]")

    print("3/5 LOB 생성 및 저장 ...")
    key = list(ohlc_dict.keys())[0]
    df = ohlc_dict[key]
    close = np.asarray(df["close"].values, dtype=np.float64)
    volume = np.asarray(df["volume"].values, dtype=np.float64)
    lob_source = "synthetic"
    lob_path_cfg = cfg["data"].get("lob_path")
    # 논문 일치: lob_path 없으면 바이낸스 실시간 LOB 수집(100ms 간격) 후 사용
    if not (lob_path_cfg and Path(lob_path_cfg).exists()):
        from src.data.loaders import collect_lob_binance_series
        bn_for_lob = _binance_symbol(key) or next((_binance_symbol(s) for s in ohlc_dict if _binance_symbol(s)), None)
        if bn_for_lob:
            real_lob = collect_lob_binance_series(bn_for_lob, limit=lob_levels, duration_sec=60, interval_ms=100)
            if real_lob is not None and len(real_lob) >= 60:
                real_path = DIRS["lob"] / "lob_binance_real.npy"
                np.save(real_path, real_lob)
                lob_path_cfg = str(real_path)
    if lob_path_cfg and Path(lob_path_cfg).exists():
        from src.data.loaders import load_lob_from_file
        lob = load_lob_from_file(lob_path_cfg, n_levels=lob_levels, align_length=len(close))
        if lob is not None:
            np.save(DIRS["lob"] / "lob.npy", lob)
            (DIRS["lob"] / "lob_shape.txt").write_text(f"shape={lob.shape}, source=real_file", encoding="utf-8")
            lob_source = "real_file"
    if lob_source == "synthetic":
        lob = load_lob_synthetic(close, volume, n_levels=lob_levels, n_snapshots=len(close), seed=seed)
        bn_sym = _binance_symbol(key)
        bn_lob = None
        if bn_sym:
            bn_lob = load_lob_binance_snapshot(symbol=bn_sym, limit=lob_levels)
            if bn_lob is not None and bn_lob.shape[1] == lob.shape[1]:
                lob[-1:] = bn_lob[:, : lob.shape[1], :]
        np.save(DIRS["lob"] / "lob.npy", lob)
        (DIRS["lob"] / "lob_shape.txt").write_text(f"shape={lob.shape}, symbol={key}", encoding="utf-8")
        lob_source = "synthetic_binance_tail" if bn_sym and bn_lob is not None else "synthetic"
    (DIRS["lob"] / "source.txt").write_text(lob_source, encoding="utf-8")
    print(f"   저장: {DIRS['lob'] / 'lob.npy'} [소스: {lob_source}]")

    print("4/5 틱 생성 및 저장 ...")
    tick_source_cfg = cfg["data"].get("tick_source", "interpolated")
    price_tick, vol_tick = None, None
    if tick_source_cfg == "1m_real":
        from src.data.loaders import load_ohlc_binance, tick_from_1m
        # 논문 일치: 1m 수집은 최근 7일로 제한해 타임아웃 방지 (실제 1분봉 = 일치)
        end_ts = pd.Timestamp(end)
        start_1m = (end_ts - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        for sym in list(ohlc_dict.keys()):
            bn_sym = _binance_symbol(sym)
            if bn_sym:
                ohlc_1m = load_ohlc_binance(bn_sym, start=start_1m, end=end, interval="1m")
                if ohlc_1m is not None and len(ohlc_1m) >= 60:
                    price_tick, vol_tick = tick_from_1m(ohlc_1m)
                    max_rows = min(1_000_000, len(price_tick))
                    price_tick, vol_tick = price_tick[:max_rows], vol_tick[:max_rows]
                    break
    if price_tick is None:
        ticks_per_day = 390
        price_tick, vol_tick = tick_from_ohlc(df, ticks_per_day=ticks_per_day)
        tick_source = "interpolated_from_ohlc"
    else:
        tick_source = "1m_real"
    np.save(DIRS["tick"] / "tick_prices.npy", price_tick)
    np.save(DIRS["tick"] / "tick_volumes.npy", vol_tick)
    (DIRS["tick"] / "tick_shape.txt").write_text(f"prices={price_tick.shape}, volumes={vol_tick.shape}", encoding="utf-8")
    (DIRS["tick"] / "source.txt").write_text(tick_source, encoding="utf-8")
    n_sample = min(100_000, len(price_tick))
    pd.DataFrame({"price": price_tick[:n_sample], "volume": vol_tick[:n_sample]}).to_csv(
        DIRS["tick"] / "tick_sample.csv", index=False, encoding="utf-8"
    )
    print(f"   저장: tick_prices.npy, tick_volumes.npy, tick_sample.csv [소스: {tick_source}]")

    print("5/5 뉴스/시맨틱 컨텍스트 생성 및 저장 ...")
    context_array = np.zeros((len(close), context_dim), dtype=np.float32)
    news_source_cfg = cfg["data"].get("news_source", "openai")
    dates = df.index.astype(str).tolist()[: len(close)]
    news_source = "zeros"
    if news_source_cfg == "finbert":
        from src.preprocess.embedding import embed_context_finbert
        news_file_path = cfg["data"].get("news_file")
        if news_file_path and Path(news_file_path).exists():
            ndf = pd.read_csv(news_file_path)
            date_col = "date" if "date" in ndf.columns else ndf.columns[0]
            text_col = "headline" if "headline" in ndf.columns else (ndf.columns[1] if len(ndf.columns) > 1 else ndf.columns[0])
            target_dates = [str(d)[:10] for d in df.index[: len(close)]]
            by_date = ndf.set_index(ndf[date_col].astype(str).str[:10])[text_col]
            texts = [str(by_date.get(d, "Market context"))[:256] for d in target_dates]
            context_array = embed_context_finbert(texts, out_dim=context_dim).astype(np.float32)
        else:
            context_array = embed_context_finbert(dates, out_dim=context_dim).astype(np.float32)
        if context_array.shape[0] >= len(close) and context_array.shape[1] == context_dim:
            context_array = context_array[: len(close)]
            news_source = "finbert"
    if news_source == "zeros" and os.environ.get("OPENAI_API_KEY", "").strip():
        from src.preprocess.embedding import embed_context_openai
        openai_ctx = embed_context_openai(dates, out_dim=context_dim)
        if openai_ctx.shape[0] == len(close) and openai_ctx.shape[1] == context_dim:
            context_array = openai_ctx.astype(np.float32)
            news_source = "openai"
    if news_source == "zeros" and not macro_df.empty:
        from src.preprocess.embedding import embed_macro
        idx = df.index
        macro_aligned = macro_df.reindex(idx).ffill().bfill().fillna(0)
        if len(macro_aligned) == len(close):
            macro_embed = embed_macro(macro_aligned.values.astype(np.float32), out_dim=cfg["preprocess"].get("macro_dim", 15))
            if macro_embed.shape[0] == len(close):
                if macro_embed.shape[1] < context_dim:
                    proj = np.random.randn(macro_embed.shape[1], context_dim).astype(np.float32) * 0.1
                    context_array = (macro_embed @ proj).astype(np.float32)
                else:
                    context_array = macro_embed[:, :context_dim].astype(np.float32)
                news_source = "macro_embed"
    np.save(DIRS["news_semantic"] / "context_768.npy", context_array)
    pd.DataFrame({"date": dates}).to_csv(DIRS["news_semantic"] / "dates.csv", index=False, encoding="utf-8")
    (DIRS["news_semantic"] / "source.txt").write_text(news_source, encoding="utf-8")
    print(f"   저장: context_768.npy {context_array.shape}, dates.csv [소스: {news_source}]")

    print("\n완료. 로컬 데이터 경로:")
    for name, dirpath in DIRS.items():
        files = list(dirpath.iterdir())
        print(f"   {dirpath.relative_to(ROOT)}: {[f.name for f in files]}")


if __name__ == "__main__":
    main()
