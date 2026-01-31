"""
로컬 data/ 폴더 전체 검증: 파일 존재, 로드 가능 여부, shape/행 수, 소스·논문 일치.
결과를 doc/로컬_데이터_검증_보고.md 및 outputs/local_data_validation.txt 에 저장.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"
OUT_TXT = ROOT / "outputs" / "local_data_validation.txt"
OUT_MD = ROOT / "doc" / "로컬_데이터_검증_보고.md"

import numpy as np
import pandas as pd


def read_source(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def validate() -> list[str]:
    lines = []
    errors = []

    def log(s: str = ""):
        lines.append(s)

    log("=" * 60)
    log("로컬 데이터 전체 검증 (data/)")
    log("=" * 60)
    log("")

    # 1. OHLC
    log("[1] OHLC (data/ohlc/)")
    ohlc_dir = DATA / "ohlc"
    if not ohlc_dir.exists():
        log("    폴더 없음")
        errors.append("ohlc 폴더 없음")
    else:
        for csv in sorted(ohlc_dir.glob("*.csv")):
            if "_source" in csv.name:
                continue
            try:
                df = pd.read_csv(csv, index_col=0, parse_dates=True)
                rows = len(df)
                start = str(df.index.min())[:10]
                end = str(df.index.max())[:10]
                src = read_source(ohlc_dir / f"{csv.stem}_source.txt") or "unknown"
                log(f"    {csv.name}: {rows}행, {start} ~ {end}, 소스={src}")
                if df.shape[1] < 5:
                    errors.append(f"{csv.name}: 컬럼 수 {df.shape[1]} (OHLC+Volume 5 필요)")
            except Exception as e:
                log(f"    {csv.name}: 로드 실패 - {e}")
                errors.append(f"OHLC {csv.name}: {e}")
        if not list(ohlc_dir.glob("*.csv")):
            log("    CSV 파일 없음")
            errors.append("OHLC CSV 없음")
    log("")

    # 2. Macro
    log("[2] 거시 15종 (data/macro/)")
    macro_csv = DATA / "macro" / "macro_15.csv"
    macro_src = read_source(DATA / "macro" / "source.txt")
    if not macro_csv.exists():
        log("    macro_15.csv 없음")
        errors.append("macro_15.csv 없음")
    else:
        try:
            df = pd.read_csv(macro_csv, index_col=0, parse_dates=True)
            rows, cols = df.shape
            start = str(df.index.min())[:10]
            end = str(df.index.max())[:10]
            log(f"    macro_15.csv: {rows}행, {cols}컬럼, {start} ~ {end}, 소스={macro_src or 'unknown'}")
            if cols != 15:
                log(f"    경고: 컬럼 수 {cols} (15 종 목)")
        except Exception as e:
            log(f"    로드 실패: {e}")
            errors.append(f"macro: {e}")
    log("")

    # 3. LOB
    log("[3] LOB (data/lob/)")
    lob_npy = DATA / "lob" / "lob.npy"
    lob_src = read_source(DATA / "lob" / "source.txt")
    if not lob_npy.exists():
        log("    lob.npy 없음")
        errors.append("lob.npy 없음")
    else:
        try:
            lob = np.load(lob_npy)
            log(f"    lob.npy: shape={lob.shape}, 소스={lob_src or 'unknown'}")
            if lob.ndim != 3 or lob.shape[1] != 10 or lob.shape[2] != 4:
                errors.append(f"LOB shape {lob.shape} (기대: (T, 10, 4))")
        except Exception as e:
            log(f"    로드 실패: {e}")
            errors.append(f"LOB: {e}")
    log("")

    # 4. Tick
    log("[4] 틱 (data/tick/)")
    tick_p = DATA / "tick" / "tick_prices.npy"
    tick_v = DATA / "tick" / "tick_volumes.npy"
    tick_src = read_source(DATA / "tick" / "source.txt")
    if not tick_p.exists() or not tick_v.exists():
        log("    tick_prices.npy 또는 tick_volumes.npy 없음")
        errors.append("tick npy 없음")
    else:
        try:
            p = np.load(tick_p)
            v = np.load(tick_v)
            log(f"    tick_prices.npy: shape={p.shape}, tick_volumes.npy: shape={v.shape}, 소스={tick_src or 'unknown'}")
            if len(p) != len(v):
                errors.append(f"tick prices/volumes 길이 불일치: {len(p)} vs {len(v)}")
        except Exception as e:
            log(f"    로드 실패: {e}")
            errors.append(f"tick: {e}")
    log("")

    # 5. News/Semantic
    log("[5] 뉴스/시맨틱 (data/news_semantic/)")
    ctx_npy = DATA / "news_semantic" / "context_768.npy"
    news_src = read_source(DATA / "news_semantic" / "source.txt")
    if not ctx_npy.exists():
        log("    context_768.npy 없음")
        errors.append("context_768.npy 없음")
    else:
        try:
            ctx = np.load(ctx_npy)
            log(f"    context_768.npy: shape={ctx.shape}, 소스={news_src or 'unknown'}")
            if ctx.ndim != 2 or ctx.shape[1] != 768:
                errors.append(f"context shape {ctx.shape} (기대: (T, 768))")
        except Exception as e:
            log(f"    로드 실패: {e}")
            errors.append(f"news_semantic: {e}")
    log("")

    # 종합
    log("=" * 60)
    log("종합: 논문 일치 여부 (소스 기준)")
    log("=" * 60)
    ohlc_files = [f for f in (ohlc_dir.glob("*.csv") if ohlc_dir.exists() else []) if "_source" not in f.name]
    ohlc_ok = bool(ohlc_files) and all(
        read_source(ohlc_dir / f"{f.stem}_source.txt") in ("yfinance", "binance") for f in ohlc_files
    )
    macro_ok = macro_src == "real"
    lob_ok = lob_src == "real_file"
    tick_ok = tick_src == "1m_real"
    news_ok = news_src == "finbert"
    log(f"  OHLC:        {'일치' if ohlc_ok else '불일치'} (소스: yfinance/binance)")
    log(f"  거시 15종:   {'일치' if macro_ok else '불일치'} (소스: real)")
    log(f"  LOB:         {'일치' if lob_ok else '불일치'} (소스: real_file)")
    log(f"  틱:          {'일치' if tick_ok else '불일치'} (소스: 1m_real)")
    log(f"  뉴스/시맨틱: {'일치' if news_ok else '불일치'} (소스: finbert)")
    log("")
    if errors:
        log("검증 경고/오류:")
        for e in errors:
            log(f"  - {e}")
    else:
        log("파일 로드 및 형태 검증: 오류 없음.")
    log("")
    return lines, errors


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    (ROOT / "doc").mkdir(parents=True, exist_ok=True)

    lines, errors = validate()
    text = "\n".join(lines)

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n저장: {OUT_TXT}")

    # Markdown 보고서
    md = [
        "# 로컬 데이터 검증 보고서",
        "",
        "> `python scripts/validate_local_data.py` 실행 결과.",
        "",
        "---",
        "",
        "## 검증 요약",
        "",
        "| 항목 | 폴더 | 주요 파일 | 검증 |",
        "|------|------|-----------|------|",
    ]
    md.append("| OHLC | data/ohlc/ | *.csv, *_source.txt | 로드·행 수·소스 확인 |")
    md.append("| 거시 15종 | data/macro/ | macro_15.csv, source.txt | 로드·15컬럼·소스 확인 |")
    md.append("| LOB | data/lob/ | lob.npy, source.txt | shape (T,10,4)·소스 확인 |")
    md.append("| 틱 | data/tick/ | tick_prices.npy, tick_volumes.npy, source.txt | shape·소스 확인 |")
    md.append("| 뉴스/시맨틱 | data/news_semantic/ | context_768.npy, dates.csv, source.txt | shape (T,768)·소스 확인 |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 상세 로그")
    md.append("")
    md.append("```")
    md.extend(lines)
    md.append("```")
    md.append("")
    md.append("---")
    md.append("")
    md.append("*검증 스크립트: `python scripts/validate_local_data.py`*")
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"보고서: {OUT_MD}")


if __name__ == "__main__":
    main()
