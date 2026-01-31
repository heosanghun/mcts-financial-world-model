"""
데이터 품질 검증: 실제 데이터 다운로드 성공 여부 vs 합성/차선 대체 여부 검토.
- 각 항목의 소스 기록(*_source.txt)을 읽어 실제(real) vs 합성·대체(synthetic/substitute) 판정.
- 합성·대체 사용 시 "실제 다운로드 실패로 임의 대체되었는지" 명시.
- OHLC: 합성 데이터 통계적 특징(논문 표 6-1)과 비교해 합성 대체 여부 이중 확인.
결과: doc/데이터_품질_검증_보고.md, outputs/data_quality_verification.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"
OUT_TXT = ROOT / "outputs" / "data_quality_verification.txt"
OUT_MD = ROOT / "doc" / "데이터_품질_검증_보고.md"

import numpy as np
import pandas as pd


# 논문 표 6-1 합성 데이터 통계 (일간 수익률 %): SP500 mean 0.03%, std 1.21%; BTC mean 0.18%, std 4.85%
THESIS_SYNTHETIC_MEAN_STD = {
    "SP500": (0.03, 1.21),   # mean%, std%
    "BTC": (0.18, 4.85),
}


def read_source(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def ohlc_synthetic_likelihood(csv_path: Path, symbol_hint: str) -> tuple[str, float]:
    """
    OHLC CSV의 일간 수익률 통계가 논문 합성 데이터(표 6-1)와 유사하면 합성 가능성 점수 반환.
    반환: ("real_likely" | "synthetic_likely" | "unknown", 0~1 점수).
    """
    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if len(df) < 252 or "close" not in df.columns:
            return "unknown", 0.0
        close = df["close"].astype(float)
        ret = close.pct_change().dropna() * 100  # %
        mean_pct = float(ret.mean())
        std_pct = float(ret.std())
        if std_pct < 1e-6:
            return "synthetic_likely", 0.9
        ref_mean, ref_std = THESIS_SYNTHETIC_MEAN_STD.get(
            "BTC" if "BTC" in symbol_hint else "SP500", (0.03, 1.21)
        )
        # 합성은 mean≈ref_mean, std≈ref_std. 실제 데이터는 편차가 클 수 있음.
        mean_diff = abs(mean_pct - ref_mean)
        std_diff = abs(std_pct - ref_std) / max(ref_std, 0.01)
        # 매우 가깝면 합성 가능성 높음 (임계값: mean_diff<0.02, std_ratio 0.1 이내)
        if mean_diff < 0.02 and std_diff < 0.15:
            return "synthetic_likely", min(0.95, 0.5 + (0.15 - std_diff))
        return "real_likely", min(0.95, std_diff * 0.3)
    except Exception:
        return "unknown", 0.0


def run_verification() -> tuple[list[str], list[dict]]:
    lines = []
    details = []

    def log(s: str = ""):
        lines.append(s)

    log("=" * 70)
    log("데이터 품질 검증: 실제 데이터 vs 합성·차선 대체 여부")
    log("=" * 70)
    log("")
    log("1. 검증 목적: 실제 데이터 다운로드 실패 시 합성/차선으로 임의 대체되지 않았는지 검토")
    log("")

    # --- OHLC ---
    log("[1] OHLC (data/ohlc/)")
    ohlc_dir = DATA / "ohlc"
    ohlc_all_real = True
    for csv in sorted(ohlc_dir.glob("*.csv")) if ohlc_dir.exists() else []:
        if "_source" in csv.name:
            continue
        src = read_source(ohlc_dir / f"{csv.stem}_source.txt")
        is_real = src in ("yfinance", "binance")
        if not is_real:
            ohlc_all_real = False
        stat_tag, score = ohlc_synthetic_likelihood(csv, csv.stem)
        if not is_real:
            log(f"    {csv.name}: 소스={src} → **합성·대체 사용** (실제 다운로드 실패로 대체된 가능성)")
        else:
            log(f"    {csv.name}: 소스={src} → 실제 데이터 사용")
        if stat_tag == "synthetic_likely" and is_real:
            log(f"       주의: 수익률 통계가 논문 합성과 유사 (합성 혼동 가능성 낮음, 점수 {score:.2f})")
        elif stat_tag == "synthetic_likely" and not is_real:
            log(f"       확인: 수익률 통계가 논문 합성과 유사 → 합성 데이터로 판정 일치 (점수 {score:.2f})")
        details.append({"category": "OHLC", "file": csv.name, "source": src, "is_real": is_real})
    if not list(ohlc_dir.glob("*.csv")) if ohlc_dir.exists() else True:
        log("    CSV 없음")
        ohlc_all_real = False
    log("")

    # --- Macro ---
    log("[2] 거시 15종 (data/macro/)")
    macro_src = read_source(DATA / "macro" / "source.txt")
    macro_real = macro_src == "real"
    if macro_src == "synthetic":
        log("    소스=synthetic → **전부 합성** (실제 다운로드 실패로 대체된 가능성)")
    elif macro_src == "real_with_synthetic_padding":
        log("    소스=real_with_synthetic_padding → 일부 합성 컬럼으로 보완 (일부 실패로 대체)")
    else:
        log("    소스=real → 실제 데이터 사용")
    details.append({"category": "macro", "source": macro_src, "is_real": macro_real})
    log("")

    # --- LOB ---
    log("[3] LOB (data/lob/)")
    lob_src = read_source(DATA / "lob" / "source.txt")
    lob_real = lob_src == "real_file"
    if lob_src in ("synthetic", "synthetic_binance_tail"):
        log(f"    소스={lob_src} → **합성 사용** (실거래소 LOB 미확보로 대체된 가능성)")
    else:
        log("    소스=real_file → 실거래소 LOB 파일 사용")
    details.append({"category": "LOB", "source": lob_src, "is_real": lob_real})
    log("")

    # --- Tick ---
    log("[4] 틱 (data/tick/)")
    tick_src = read_source(DATA / "tick" / "source.txt")
    tick_real = tick_src == "1m_real"
    if tick_src == "interpolated_from_ohlc":
        log("    소스=interpolated_from_ohlc → **일봉 OHLC 보간** (실제 1m/틱 미확보로 대체된 가능성)")
    else:
        log("    소스=1m_real → 1분봉 실데이터(틱 대용) 사용")
    details.append({"category": "tick", "source": tick_src, "is_real": tick_real})
    log("")

    # --- News ---
    log("[5] 뉴스/시맨틱 (data/news_semantic/)")
    news_src = read_source(DATA / "news_semantic" / "source.txt")
    news_real = news_src == "finbert"
    if news_src in ("openai", "macro_embed", "zeros"):
        log(f"    소스={news_src} → **차선 사용** (FinBERT/뉴스 미사용으로 대체된 가능성)")
    else:
        log("    소스=finbert → FinBERT 768-dim 사용 (논문 스펙)")
    details.append({"category": "news", "source": news_src, "is_real": news_real})
    log("")

    # --- 종합 ---
    log("=" * 70)
    log("종합: 합성·대체 여부 (실제 다운로드 실패로 임의 대체되었는지)")
    log("=" * 70)
    any_substitute = not (ohlc_all_real and macro_real and lob_real and tick_real and news_real)
    if any_substitute:
        log("")
        log("  ⚠ 다음 항목이 실제가 아닌 합성·차선으로 대체되어 있습니다:")
        if not ohlc_all_real:
            log("    - OHLC: 일부 또는 전부 소스=synthetic (실제 수집 실패 시 합성 사용)")
        if not macro_real:
            log("    - 거시: synthetic 또는 real_with_synthetic_padding")
        if not lob_real:
            log("    - LOB: synthetic / synthetic_binance_tail (실거래소 LOB 미사용)")
        if not tick_real:
            log("    - 틱: interpolated_from_ohlc (1분봉 실데이터 미사용)")
        if not news_real:
            log("    - 뉴스/시맨틱: openai / macro_embed / zeros (FinBERT 미사용)")
    else:
        log("")
        log("  [OK] 모든 항목이 실제(또는 논문 스펙) 소스를 사용 중입니다.")
        log("    합성·차선으로 임의 대체된 항목 없음.")
    log("")
    return lines, details


def main():
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    (ROOT / "doc").mkdir(parents=True, exist_ok=True)

    lines, details = run_verification()
    text = "\n".join(lines)

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text(text, encoding="utf-8")
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("utf-8", errors="replace").decode("utf-8"))
    print(f"\n저장: {OUT_TXT}")

    # Markdown 보고서
    all_real = all(d.get("is_real", False) for d in details if d.get("category") != "OHLC") and all(
        d.get("is_real", False) for d in details if d.get("category") == "OHLC"
    )
    # OHLC는 여러 파일이 있으므로 details에서 OHLC인 것들이 모두 is_real인지 확인
    ohlc_details = [d for d in details if d.get("category") == "OHLC"]
    macro_detail = next((d for d in details if d.get("category") == "macro"), {})
    lob_detail = next((d for d in details if d.get("category") == "LOB"), {})
    tick_detail = next((d for d in details if d.get("category") == "tick"), {})
    news_detail = next((d for d in details if d.get("category") == "news"), {})

    all_real = (
        all(d.get("is_real", False) for d in ohlc_details) if ohlc_details else False
    ) and macro_detail.get("is_real", False) and lob_detail.get("is_real", False) and tick_detail.get("is_real", False) and news_detail.get("is_real", False)

    md = [
        "# 데이터 품질 검증 보고서",
        "",
        "> 실제 데이터 다운로드 실패 시 **합성·차선으로 임의 대체되지 않았는지** 검토 결과.",
        "",
        "---",
        "",
        "## 검증 목적",
        "",
        "- 각 데이터 항목의 **소스 기록**(`*_source.txt`)을 읽어 **실제(real)** vs **합성·대체(synthetic/substitute)** 판정.",
        "- 합성·대체 사용 시, 실제 다운로드 실패로 **임의 대체된 가능성**을 명시.",
        "- OHLC: 논문 표 6-1 합성 데이터 통계와 비교해 합성 대체 여부 **이중 확인**.",
        "",
        "---",
        "",
        "## 항목별 판정",
        "",
        "| 항목 | 소스 | 실제 사용 여부 | 비고 |",
        "|------|------|----------------|------|",
    ]
    for d in details:
        cat = d.get("category", "")
        src = d.get("source", "unknown")
        is_r = d.get("is_real", False)
        note = "실제 데이터 사용" if is_r else "**합성·차선 대체** (실제 미사용/실패 가능)"
        if cat == "OHLC":
            f = d.get("file", "")
            md.append(f"| OHLC ({f}) | {src} | {'예' if is_r else '아니오'} | {note} |")
        else:
            md.append(f"| {cat} | {src} | {'예' if is_r else '아니오'} | {note} |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 종합 결론")
    md.append("")
    if all_real:
        md.append("**✓ 모든 항목이 실제(또는 논문 스펙) 소스를 사용 중입니다.**")
        md.append("")
        md.append("합성·차선으로 임의 대체된 항목 없음.")
    else:
        md.append("**⚠ 일부 항목이 합성·차선으로 대체되어 있습니다.**")
        md.append("")
        md.append("해당 항목은 실제 다운로드/수집 실패 시 합성 또는 차선 데이터로 대체된 가능성이 있습니다. ")
        md.append("재다운로드 또는 설정(lob_path, tick_source, news_source 등) 확인 후 `python scripts/download_all_data.py` 재실행을 권장합니다.")
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
    md.append("*검증: `python scripts/verify_data_quality.py`*")
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"보고서: {OUT_MD}")


if __name__ == "__main__":
    main()
