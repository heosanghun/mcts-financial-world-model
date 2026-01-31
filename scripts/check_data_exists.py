"""
논문에서 요구하는 데이터가 실제로 존재하는지 점검.
- OHLC: yfinance 실조회 → 실패 시 합성 여부
- 거시 15종: yfinance 실조회 → 컬럼 수·합성 패딩 여부
- LOB: 코드상 합성만 지원 (실제 100ms 스냅샷 없음)
- 틱: OHLC 보간 (실제 틱 데이터 없음)
- 뉴스/시맨틱: OpenAI API 또는 거시 임베딩 (Reuters/Bloomberg 원시 데이터 없음)
"""
from __future__ import annotations

import os
import sys
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

import yaml
import numpy as np
import pandas as pd

def load_config():
    with open(ROOT / "configs" / "default.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    start = cfg["data"]["start_date"]
    end = cfg["data"]["end_date"]
    symbols = cfg["data"]["assets"].get("equity", []) + cfg["data"]["assets"].get("crypto", [])
    macro_tickers = cfg["data"].get("macro_tickers", [])[:15]

    report = []
    report.append("=" * 60)
    report.append("논문 대비 데이터 존재 여부 점검")
    report.append("=" * 60)
    report.append("")

    # --- 1) OHLC (실제 vs 합성) ---
    report.append("[1] OHLC (시장 데이터)")
    report.append("    논문: S&P 500, BTC/USDT, 2000-2024, OHLC+Volume (T,5)")
    from src.data.loaders import load_ohlc, load_ohlc_thesis_aligned, get_date_range

    raw_ohlc = {}
    try:
        raw_ohlc = load_ohlc(symbols, start, end, interval="1d")
        for sym, df in raw_ohlc.items():
            if df is not None and not df.empty:
                report.append(f"    [yfinance만] {sym}: {len(df)}행, {df.index.min()} ~ {df.index.max()}")
            else:
                report.append(f"    [yfinance만] {sym}: 없음")
        if not raw_ohlc:
            report.append("    [yfinance만] 로드 실패(네트워크/심볼 확인)")
    except Exception as e:
        report.append(f"    [yfinance만] 오류: {e}")

    thesis_ohlc = load_ohlc_thesis_aligned(symbols, start, end, fallback_synthetic=True)
    ohlc_all_synthetic = True
    for sym, df in thesis_ohlc.items():
        n = len(df)
        from_yf = sym in raw_ohlc and raw_ohlc[sym] is not None and not raw_ohlc[sym].empty and abs(len(raw_ohlc[sym]) - n) <= 10
        if from_yf:
            ohlc_all_synthetic = False
        report.append(f"    [파이프라인] {sym}: {n}행 -> {'실제(yfinance)' if from_yf else '합성'}")
    report.append(f"    판정: OHLC 전부 합성? {'예' if ohlc_all_synthetic else '아니오 (yfinance 실제 사용)'}")
    report.append("")

    # --- 2) 거시 15종 ---
    report.append("[2] 거시 지표 (15종)")
    report.append("    논문: VIX, DXY, US10Y, US02Y, Ted Spread, Gold, WTI 등 15종")
    from src.data.loaders import load_macro, load_macro_thesis_aligned

    macro_raw = pd.DataFrame()
    try:
        macro_raw = load_macro(macro_tickers, start, end)
        if not macro_raw.empty:
            report.append(f"    [yfinance만] {macro_raw.shape[1]}컬럼, {len(macro_raw)}행")
            report.append(f"    컬럼: {list(macro_raw.columns)}")
        else:
            report.append("    [yfinance만] 로드 없음")
    except Exception as e:
        report.append(f"    [yfinance만] 오류: {e}")

    macro_thesis = load_macro_thesis_aligned(macro_tickers, start, end, target_n_cols=15)
    report.append(f"    [파이프라인] {macro_thesis.shape[1]}컬럼, {len(macro_thesis)}행")
    # 합성 컬럼: _macro_0, _macro_1 ... (yfinance 컬럼은 티커명)
    synth_cols = [c for c in macro_thesis.columns if str(c).startswith("_macro_")]
    real_cols = [c for c in macro_thesis.columns if not str(c).startswith("_macro_")]
    macro_all_synthetic = len(real_cols) == 0
    report.append(f"    실제 컬럼 수: {len(real_cols)}, 합성 패딩 컬럼 수: {len(synth_cols)}")
    report.append(f"    판정: 거시 전부 합성? {'예' if macro_all_synthetic else '아니오 (yfinance 실제 사용)'}")
    report.append("")

    # --- 3) LOB ---
    report.append("[3] LOB (호가창 10단계)")
    report.append("    논문: 매수/매도 1~10호가 가격·잔량, 100ms 스냅샷")
    report.append("    [실제] 프로젝트 내 LOB 원시 데이터 파일 없음 (data/ 폴더 없음, CSV 없음)")
    report.append("    [구현] load_lob_synthetic() 사용 - OHLC/Volume 기반 합성 (T, 10, 4)")
    report.append("    판정: LOB 전부 합성? 예 (실제 100ms 스냅샷 없음)")
    report.append("")

    # --- 4) 틱 ---
    report.append("[4] 틱 (체결가·체결량)")
    report.append("    논문: 틱 단위 체결가·체결량")
    report.append("    [실제] 프로젝트 내 틱 원시 데이터 없음")
    report.append("    [구현] tick_from_ohlc() - 일봉 OHLC를 틱 수준으로 보간")
    report.append("    판정: 틱 전부 합성(보간)? 예 (실제 틱 데이터 없음)")
    report.append("")

    # --- 5) 뉴스/시맨틱 컨텍스트 ---
    report.append("[5] 뉴스/시맨틱 컨텍스트 (768-dim)")
    report.append("    논문: Reuters/Bloomberg 뉴스 헤드라인/공시 -> BERT/FinBERT 768-dim")
    report.append("    [실제] Reuters/Bloomberg 원시 뉴스 파일 없음")
    has_openai = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    if has_openai:
        report.append("    [구현] embed_context_openai() - OpenAI Embeddings API로 시맨틱 벡터 생성 (날짜 기반)")
    else:
        report.append("    [구현] OPENAI_API_KEY 미설정 -> 거시 임베딩/zeros fallback")
    report.append("    판정: 뉴스 원시 없음 - API/거시 대체 (Reuters/Bloomberg 원시 없음)")
    report.append("")

    # --- 6) 로컬 데이터 폴더/파일 ---
    report.append("[6] 로컬 데이터 파일")
    data_dir = ROOT / "data"
    if data_dir.exists():
        files = list(data_dir.rglob("*"))
        report.append(f"    data/ 존재: {len(files)}개 파일")
        for f in files[:20]:
            report.append(f"      - {f.relative_to(ROOT)}")
    else:
        report.append("    data/ 폴더 없음 - 모든 시계열은 yfinance 또는 코드 내 합성")
    report.append("")

    # --- 요약: 데이터가 모두 합성인지 ---
    report.append("=" * 60)
    report.append("요약: '데이터 실제존재가 모두 합성인지' 판정")
    report.append("=" * 60)
    report.append("  - OHLC: 전부 합성? " + ("예" if ohlc_all_synthetic else "아니오 (yfinance 실제 사용)"))
    report.append("  - 거시 15종: 전부 합성? " + ("예" if macro_all_synthetic else "아니오 (yfinance 실제 사용)"))
    report.append("  - LOB: 전부 합성? 예")
    report.append("  - 틱: 전부 합성(보간)? 예")
    report.append("  - 뉴스: 원시 없음, API/거시 대체")
    all_synthetic = ohlc_all_synthetic and macro_all_synthetic
    report.append("")
    report.append("  >>> 전체 데이터가 모두 합성인가? " + ("예" if all_synthetic else "아니오. OHLC와 거시는 yfinance 실제 데이터 사용."))
    report.append("")

    text = "\n".join(report)
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "data_existence_check.txt"
    out_file.write_text(text, encoding="utf-8")
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("utf-8", errors="replace").decode("utf-8"))
    print("\n보고 저장: " + str(out_file))

if __name__ == "__main__":
    main()
