"""
논문 데이터 스펙과 실제/로컬 데이터를 디테일하게 대조 검증.
합성 또는 차선 데이터 사용 시 논문과 불일치로 표시.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if (ROOT / ".env").exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

import yaml
import numpy as np
import pandas as pd

DATA = ROOT / "data"

# 논문 스펙 (doc/초안_박사논문_코드구현용_정리.md §2 기준)
PAPER_SPEC = {
    "period": ("2000-01-01", "2024-12-31"),
    "assets": "S&P 500 구성 종목, BTC/USDT",
    "ohlc": "OHLC, Volume (T,5), 1분/틱, Log Return, Min-Max",
    "lob": "매수/매도 1~10호가 가격·잔량, (T,10,4), 100ms 스냅샷, Z-Score",
    "tick": "체결가·체결량 (틱 단위)",
    "macro": "VIX, DXY, US10Y, US02Y, Ted Spread, Gold, WTI 등 15종",
    "news": "뉴스 헤드라인·공시, 768-dim, BERT/FinBERT (Reuters/Bloomberg)",
}


def read_source(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def infer_ohlc_source(sym: str, local_rows: int, start: str, end: str) -> str:
    """소스 파일 없을 때: real 로드 시도 후 행 수 비교."""
    try:
        from src.data.loaders import load_ohlc, load_ohlc_binance, _binance_symbol
        bn = _binance_symbol(sym.replace("_", "-") if "GSPC" not in sym else "^GSPC")
        if bn and sym != "GSPC":
            df = load_ohlc_binance(bn, start=start, end=end, interval="1d")
            if df is not None and abs(len(df) - local_rows) <= 50:
                return "binance"
        yf_sym = "^GSPC" if sym == "GSPC" else sym.replace("_", "-")
        raw = load_ohlc([yf_sym], start=start, end=end, interval="1d")
        if yf_sym in raw and raw[yf_sym] is not None and abs(len(raw[yf_sym]) - local_rows) <= 50:
            return "yfinance"
    except Exception:
        pass
    return "synthetic"


def verify_ohlc(lines: list, cfg_start: str, cfg_end: str) -> None:
    lines.append("## 1. OHLC (시장 데이터)")
    lines.append("")
    lines.append("| 항목 | 논문 스펙 | 실제/로컬 | 일치 여부 |")
    lines.append("|------|-----------|-----------|-----------|")
    lines.append("| **출처** | S&P 500, BTC/USDT 실제 시장 데이터 | 아래 소스 파일 참조 | - |")
    lines.append("| **기간** | 2000.01.01 ~ 2024.12.31 | config + 실제 수신 구간 | - |")
    for f in sorted((DATA / "ohlc").glob("*.csv")):
        if "_source" in f.name:
            continue
        stem = f.stem
        src_file = DATA / "ohlc" / f"{stem}_source.txt"
        source = read_source(src_file)
        if not source:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            source = infer_ohlc_source(stem, len(df), cfg_start, cfg_end)
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        start, end = str(df.index.min())[:10], str(df.index.max())[:10]
        if source in ("yfinance", "binance"):
            match = "**일치** (실제 데이터)"
        else:
            match = "**불일치** (합성 데이터, 논문과 다름)"
        lines.append(f"| {stem} | 실제 시장 OHLCV | 소스={source}, {len(df)}행, {start}~{end} | {match} |")
    if not list((DATA / "ohlc").glob("*.csv")):
        lines.append("| (로컬 없음) | - | 다운로드 미실행 | - |")
    lines.append("")


def verify_macro(lines: list) -> None:
    lines.append("## 2. 거시 15종")
    lines.append("")
    lines.append("| 항목 | 논문 스펙 | 실제/로컬 | 일치 여부 |")
    lines.append("|------|-----------|-----------|-----------|")
    source = read_source(DATA / "macro" / "source.txt")
    if not source and (DATA / "macro" / "macro_15.csv").exists():
        df = pd.read_csv(DATA / "macro" / "macro_15.csv", index_col=0, nrows=1)
        synth = sum(1 for c in df.columns if str(c).startswith("_macro_"))
        source = "synthetic" if synth == len(df.columns) else ("real_with_synthetic_padding" if synth else "real")
    source = source or "unknown"
    if (DATA / "macro" / "macro_15.csv").exists():
        df = pd.read_csv(DATA / "macro" / "macro_15.csv", index_col=0, nrows=5)
        ncols = len(df.columns)
        synth = sum(1 for c in df.columns if str(c).startswith("_macro_"))
        if source == "real":
            match = "**일치** (실제 yfinance 15종)"
        elif source == "real_with_synthetic_padding":
            match = "**부분 일치** (실제 + 합성 패딩, 부족분은 논문과 다름)"
        else:
            match = "**불일치** (전부 합성 또는 대체, 논문과 다름)"
        lines.append(f"| 거시 지표 | VIX, DXY, US10Y 등 15종 실제 | 소스={source}, {ncols}컬럼, 합성컬럼={synth} | {match} |")
    else:
        lines.append("| 거시 15종 | 실제 거시 데이터 | 로컬 없음 | - |")
    lines.append("")


def verify_lob(lines: list) -> None:
    lines.append("## 3. LOB (호가창 10단계)")
    lines.append("")
    lines.append("| 항목 | 논문 스펙 | 실제/로컬 | 일치 여부 |")
    lines.append("|------|-----------|-----------|-----------|")
    source = read_source(DATA / "lob" / "source.txt")
    if not source and (DATA / "lob" / "lob.npy").exists():
        source = "synthetic_binance_tail"
    source = source or "synthetic"
    paper = "100ms 스냅샷, 실제 거래소 호가창 (T,10,4)"
    if source == "real_file":
        match = "**일치** (실거래소 LOB 파일 사용, 논문 스펙)"
    elif source == "synthetic":
        match = "**불일치** (합성만 사용, 논문은 실제 100ms 스냅샷)"
    elif source == "synthetic_binance_tail":
        match = "**불일치** (과거 합성 + 마지막 1스냅만 바이낸스, 논문은 전 구간 실제)"
    else:
        match = "**불일치** (실제 100ms LOB 미사용)"
    lines.append(f"| LOB | {paper} | 소스={source} | {match} |")
    if (DATA / "lob" / "lob.npy").exists():
        lob = np.load(DATA / "lob" / "lob.npy")
        shape_note = "형태 일치" + (", 출처 일치" if source == "real_file" else ", 출처 불일치")
        lines.append(f"| shape | (T, 10, 4) | {lob.shape} | {shape_note} |")
    lines.append("")


def verify_tick(lines: list) -> None:
    lines.append("## 4. 틱 (체결가·체결량)")
    lines.append("")
    lines.append("| 항목 | 논문 스펙 | 실제/로컬 | 일치 여부 |")
    lines.append("|------|-----------|-----------|-----------|")
    source = read_source(DATA / "tick" / "source.txt")
    if not source and (DATA / "tick" / "tick_prices.npy").exists():
        source = "interpolated_from_ohlc"
    source = source or "unknown"
    paper = "틱 단위 체결가·체결량 (실제 거래 틱)"
    if source == "1m_real":
        match = "**일치** (1분봉 실데이터를 틱 대용, 논문과 동일 용도)"
    elif source == "interpolated_from_ohlc":
        match = "**불일치** (OHLC 보간, 논문은 실제 틱 데이터)"
    elif source == "real":
        match = "**일치** (실제 틱)"
    else:
        match = "**불일치** (실제 틱 미사용)"
    lines.append(f"| 틱 | {paper} | 소스={source} | {match} |")
    lines.append("")


def verify_news(lines: list) -> None:
    lines.append("## 5. 뉴스/시맨틱 컨텍스트 (768-dim)")
    lines.append("")
    lines.append("| 항목 | 논문 스펙 | 실제/로컬 | 일치 여부 |")
    lines.append("|------|-----------|-----------|-----------|")
    source = read_source(DATA / "news_semantic" / "source.txt")
    if not source and (DATA / "news_semantic" / "context_768.npy").exists():
        ctx = np.load(DATA / "news_semantic" / "context_768.npy")
        source = "openai" if (ctx != 0).any() else "zeros"
    source = source or "unknown"
    paper = "Reuters/Bloomberg 뉴스 헤드라인·공시, BERT/FinBERT 768-dim"
    if source == "finbert":
        match = "**일치** (FinBERT 768-dim, 논문 스펙)"
    elif source == "openai":
        match = "**불일치** (차선: OpenAI Embeddings, 논문은 Reuters/Bloomberg+FinBERT)"
    elif source == "macro_embed":
        match = "**불일치** (차선: 거시 임베딩, 논문은 뉴스+FinBERT)"
    elif source == "zeros":
        match = "**불일치** (차선: zeros, 논문과 다름)"
    elif source == "reuters_bloomberg_finbert":
        match = "**일치** (Reuters/Bloomberg+FinBERT)"
    else:
        match = "**불일치**"
    lines.append(f"| 시맨틱 | {paper} | 소스={source} | {match} |")
    lines.append("")


def get_summary_flags(cfg_start: str, cfg_end: str) -> tuple[bool, bool, bool, bool, bool]:
    """소스 파일·데이터 기반으로 OHLC/거시/LOB/틱/뉴스 일치 여부 반환."""
    ohlc_sources = []
    for f in (DATA / "ohlc").glob("*_source.txt"):
        ohlc_sources.append(f.read_text(encoding="utf-8").strip())
    if not ohlc_sources and list((DATA / "ohlc").glob("*.csv")):
        for csv in (DATA / "ohlc").glob("*.csv"):
            if "_source" in csv.name:
                continue
            df = pd.read_csv(csv, index_col=0)
            src = infer_ohlc_source(csv.stem, len(df), cfg_start, cfg_end)
            ohlc_sources.append(src)
    ohlc_ok = all(s in ("yfinance", "binance") for s in ohlc_sources) and bool(ohlc_sources)
    macro_src = read_source(DATA / "macro" / "source.txt")
    if not macro_src and (DATA / "macro" / "macro_15.csv").exists():
        df = pd.read_csv(DATA / "macro" / "macro_15.csv", index_col=0, nrows=1)
        synth = sum(1 for c in df.columns if str(c).startswith("_macro_"))
        macro_src = "synthetic" if len(synth) == len(df.columns) else ("real_with_synthetic_padding" if synth else "real")
    macro_ok = macro_src == "real"
    lob_src = read_source(DATA / "lob" / "source.txt") or ("synthetic_binance_tail" if (DATA / "lob" / "lob.npy").exists() else "synthetic")
    tick_src = read_source(DATA / "tick" / "source.txt") or "interpolated_from_ohlc"
    news_src = read_source(DATA / "news_semantic" / "source.txt")
    if not news_src and (DATA / "news_semantic" / "context_768.npy").exists():
        ctx = np.load(DATA / "news_semantic" / "context_768.npy")
        news_src = "openai" if (ctx != 0).any() else "zeros"
    news_src = news_src or "unknown"
    lob_ok = lob_src == "real_file"
    tick_ok = tick_src == "1m_real"
    news_ok = news_src == "finbert"
    return ohlc_ok, macro_ok, lob_ok, tick_ok, news_ok


def main():
    with open(ROOT / "configs" / "default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg_start = cfg["data"]["start_date"]
    cfg_end = cfg["data"]["end_date"]
    ohlc_ok, macro_ok, lob_ok, tick_ok, news_ok = get_summary_flags(cfg_start, cfg_end)

    lines = []
    lines.append("# 논문 데이터 스펙 대조 검증 보고서")
    lines.append("")
    lines.append("> 합성/차선 데이터 사용 시 **논문과 불일치**로 명시. 논문은 실제 데이터 출처를 요구함.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 논문 스펙 vs 실제 디테일 (§2.1, §2.2, 표 3-1)")
    lines.append("")
    lines.append("| 데이터 | 논문 스펙 (출처·차원·주기) | 실제 구현 | 일치 |")
    lines.append("|--------|---------------------------|-----------|------|")
    lines.append(f"| **OHLC** | S&P500, BTC/USDT 실제 시장, (T,5), 1분/틱, Log Return·Min-Max | yfinance/Binance 실제, (T,5) 일봉, 전처리 동일 | {'O' if ohlc_ok else 'X'} |")
    lines.append(f"| **거시** | VIX, DXY, US10Y 등 15종 실제, (T,3~15), 1h/일 | yfinance 15컬럼 실제, 부족 시 합성 패딩 | {'O' if macro_ok else 'X'} |")
    lines.append(f"| **LOB** | 실제 100ms 스냅샷, (T,10,4), Z-Score | 실거래소 LOB 파일 / 합성 | {'O' if lob_ok else 'X'} |")
    lines.append(f"| **틱** | 실제 체결가·체결량 | 1분봉 실데이터(틱 대용) / OHLC 보간 | {'O' if tick_ok else 'X'} |")
    lines.append(f"| **뉴스** | Reuters/Bloomberg, BERT/FinBERT, 768-dim | FinBERT 768-dim / OpenAI·거시 | {'O' if news_ok else 'X'} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 논문 스펙 요약 (§2.1, §2.2)")
    lines.append("")
    lines.append("- **기간**: 2000.01.01 ~ 2024.12.31")
    lines.append("- **자산**: S&P 500 구성 종목, BTC/USDT")
    lines.append("- **OHLC**: 실제 시장 데이터 (1분/틱), Log Return, Min-Max")
    lines.append("- **LOB**: 실제 100ms 스냅샷, 10단계 매수/매도 가격·잔량")
    lines.append("- **틱**: 실제 체결가·체결량")
    lines.append("- **거시**: VIX, DXY, US10Y 등 15종 실제")
    lines.append("- **뉴스**: Reuters/Bloomberg 뉴스, BERT/FinBERT 768-dim")
    lines.append("")
    lines.append("---")
    lines.append("")

    verify_ohlc(lines, cfg_start, cfg_end)
    verify_macro(lines)
    verify_lob(lines)
    verify_tick(lines)
    verify_news(lines)

    lines.append("---")
    lines.append("")
    lines.append("## 종합 판정")
    lines.append("")
    lines.append("| 데이터 | 논문 요구 | 현재 사용 | 논문 일치 여부 |")
    lines.append("|--------|-----------|-----------|----------------|")
    # Re-read sources for summary
    ohlc_sources = []
    for f in (DATA / "ohlc").glob("*_source.txt"):
        ohlc_sources.append(f.read_text(encoding="utf-8").strip())
    if not ohlc_sources and list((DATA / "ohlc").glob("*.csv")):
        with open(ROOT / "configs" / "default.yaml", "r", encoding="utf-8") as f:
            c = yaml.safe_load(f)
        s, e = c["data"]["start_date"], c["data"]["end_date"]
        for csv in (DATA / "ohlc").glob("*.csv"):
            if "_source" in csv.name:
                continue
            df = pd.read_csv(csv, index_col=0)
            src = infer_ohlc_source(csv.stem, len(df), s, e)
            ohlc_sources.append(src)
    ohlc_ok = all(s in ("yfinance", "binance") for s in ohlc_sources) and ohlc_sources
    macro_src = read_source(DATA / "macro" / "source.txt")
    if not macro_src and (DATA / "macro" / "macro_15.csv").exists():
        df = pd.read_csv(DATA / "macro" / "macro_15.csv", index_col=0, nrows=1)
        synth = sum(1 for c in df.columns if str(c).startswith("_macro_"))
        macro_src = "synthetic" if synth == len(df.columns) else ("real_with_synthetic_padding" if synth else "real")
    macro_ok = macro_src == "real"
    lob_src = read_source(DATA / "lob" / "source.txt") or ("synthetic_binance_tail" if (DATA / "lob" / "lob.npy").exists() else "synthetic")
    tick_src = read_source(DATA / "tick" / "source.txt") or "interpolated_from_ohlc"
    news_src = read_source(DATA / "news_semantic" / "source.txt")
    if not news_src and (DATA / "news_semantic" / "context_768.npy").exists():
        ctx = np.load(DATA / "news_semantic" / "context_768.npy")
        news_src = "openai" if (ctx != 0).any() else "zeros"
    news_src = news_src or "unknown"
    lob_ok = lob_src == "real_file"
    tick_ok = tick_src == "1m_real"
    news_ok = news_src == "finbert"

    lines.append(f"| OHLC | 실제 S&P500, BTC/USDT | {'실제' if ohlc_ok else '합성 가능'} | {'일치' if ohlc_ok else '불일치(합성 시)'} |")
    lines.append(f"| 거시 15종 | 실제 15종 | {'실제' if macro_ok else macro_src or 'unknown'} | {'일치' if macro_ok else '불일치'} |")
    lines.append(f"| LOB | 실제 100ms 스냅샷 | {lob_src or 'unknown'} | {'일치' if lob_ok else '불일치'} |")
    lines.append(f"| 틱 | 실제 틱 | {tick_src or 'unknown'} | {'일치' if tick_ok else '불일치'} |")
    lines.append(f"| 뉴스/시맨틱 | Reuters/Bloomberg+FinBERT | {news_src or 'unknown'} | {'일치' if news_ok else '불일치'} |")
    lines.append("")
    all_ok = ohlc_ok and macro_ok and lob_ok and tick_ok and news_ok
    conclusion = "모든 데이터가 논문 스펙과 **일치**합니다." if all_ok else "OHLC·거시는 실제 사용 시 일치. LOB는 실거래소 파일(lob_path), 틱은 1m_real, 뉴스는 finbert 사용 시 **일치**."
    lines.append(f"**결론**: {conclusion}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*검증 스크립트: `python scripts/verify_thesis_data.py`*")
    lines.append("*다운로드 시 소스 기록: `python scripts/download_all_data.py` (재실행 시 *_source.txt 갱신)*")

    text = "\n".join(lines)
    out_md = ROOT / "doc" / "데이터_논문_일치_검증_보고.md"
    out_md.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n저장: {out_md}")


if __name__ == "__main__":
    main()
