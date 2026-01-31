"""
로컬 데이터(data/) 5개 항목의 기간(시작일·종료일·행 수) 점검.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

DATA = ROOT / "data"
report = []


def check_ohlc():
    report.append("[1] OHLC (data/ohlc/)")
    for f in sorted((DATA / "ohlc").glob("*.csv")):
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        start = df.index.min()
        end = df.index.max()
        report.append(f"    {f.name}: {len(df)}행, {start} ~ {end}")
    report.append("")


def check_macro():
    report.append("[2] 거시 15종 (data/macro/)")
    f = DATA / "macro" / "macro_15.csv"
    if f.exists():
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        start = df.index.min()
        end = df.index.max()
        report.append(f"    macro_15.csv: {len(df)}행, {df.shape[1]}컬럼, {start} ~ {end}")
    else:
        report.append("    macro_15.csv 없음")
    report.append("")


def check_lob():
    report.append("[3] LOB (data/lob/)")
    npy = DATA / "lob" / "lob.npy"
    txt = DATA / "lob" / "lob_shape.txt"
    if npy.exists():
        lob = np.load(npy)
        report.append(f"    lob.npy: shape {lob.shape} (T={lob.shape[0]} 스냅샷)")
    if txt.exists():
        report.append(f"    lob_shape.txt: {txt.read_text(encoding='utf-8').strip()}")
    report.append("    (LOB는 날짜 컬럼 없음. OHLC 일수와 동일 T로 정렬됨)")
    report.append("")


def check_tick():
    report.append("[4] 틱 (data/tick/)")
    p = DATA / "tick" / "tick_prices.npy"
    v = DATA / "tick" / "tick_volumes.npy"
    txt = DATA / "tick" / "tick_shape.txt"
    if p.exists() and v.exists():
        pr = np.load(p)
        vol = np.load(v)
        report.append(f"    tick_prices.npy: {pr.shape[0]} 틱")
        report.append(f"    tick_volumes.npy: {vol.shape[0]} 틱")
    if txt.exists():
        report.append(f"    tick_shape.txt: {txt.read_text(encoding='utf-8').strip()}")
    report.append("    (틱은 OHLC 일수 x 390 ticks/day 보간. 날짜 컬럼 없음)")
    report.append("")


def check_news_semantic():
    report.append("[5] 뉴스/시맨틱 (data/news_semantic/)")
    dates_file = DATA / "news_semantic" / "dates.csv"
    ctx_file = DATA / "news_semantic" / "context_768.npy"
    if dates_file.exists():
        df = pd.read_csv(dates_file)
        dates = pd.to_datetime(df["date"], utc=True)
        report.append(f"    dates.csv: {len(dates)}행, {dates.min()} ~ {dates.max()}")
    if ctx_file.exists():
        ctx = np.load(ctx_file)
        report.append(f"    context_768.npy: shape {ctx.shape}")
    report.append("")


def main():
    report.append("=" * 60)
    report.append("로컬 데이터 기간 점검 (data/)")
    report.append("=" * 60)
    report.append("")
    check_ohlc()
    check_macro()
    check_lob()
    check_tick()
    check_news_semantic()
    report.append("=" * 60)
    report.append("요약: 데이터 기간")
    report.append("=" * 60)
    # 요약 표
    ohlc_dir = DATA / "ohlc"
    if (DATA / "ohlc" / "GSPC.csv").exists():
        df = pd.read_csv(DATA / "ohlc" / "GSPC.csv", index_col=0, parse_dates=True)
        gspc_start, gspc_end = df.index.min(), df.index.max()
    else:
        gspc_start = gspc_end = "-"
    if (DATA / "macro" / "macro_15.csv").exists():
        df = pd.read_csv(DATA / "macro" / "macro_15.csv", index_col=0, parse_dates=True)
        macro_start, macro_end = df.index.min(), df.index.max()
    else:
        macro_start = macro_end = "-"
    if (DATA / "news_semantic" / "dates.csv").exists():
        df = pd.read_csv(DATA / "news_semantic" / "dates.csv")
        d = pd.to_datetime(df["date"], utc=True)
        sem_start, sem_end = d.min(), d.max()
    else:
        sem_start = sem_end = "-"
    report.append("  항목            | 행 수(또는 shape) | 시작일       | 종료일")
    report.append("  ----------------|-------------------|--------------|-------------")
    for f in sorted((DATA / "ohlc").glob("*.csv")):
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        report.append(f"  OHLC {f.stem:12} | {len(df):>6}행          | {str(df.index.min())[:10]:12} | {str(df.index.max())[:10]}")
    report.append(f"  거시 15종        | macro_15.csv      | {str(macro_start)[:10]:12} | {str(macro_end)[:10]}")
    report.append("  LOB             | (T,10,4) OHLC T와 동일 | -            | -")
    report.append("  틱              | T*390 (OHLC T일)  | -            | -")
    report.append(f"  뉴스/시맨틱     | dates.csv 행 수   | {str(sem_start)[:10]:12} | {str(sem_end)[:10]}")
    report.append("")
    text = "\n".join(report)
    print(text)
    out = ROOT / "outputs" / "local_data_period_check.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
