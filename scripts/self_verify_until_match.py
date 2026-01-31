"""
논문 데이터 일치 시까지 다운로드 → 검증을 반복하는 셀프검증 루프.
목표: doc/데이터_논문_일치_검증_보고.md 종합 판정에서 전 항목 '일치'가 될 때까지
      데이터 확보·다운로드를 반복하고, 완료 후에도 최종 검증을 재실행해 일치를 확인.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "doc" / "데이터_논문_일치_검증_보고.md"
CONFIG = ROOT / "configs" / "default.yaml"
MAX_ITERATIONS = 15
VERIFY_AFTER_MATCH_ROUNDS = 2  # 일치 도달 후에도 검증을 추가로 반복할 횟수


def ensure_config_thesis_aligned() -> bool:
    """config에 tick_source=1m_real, news_source=finbert 있도록 보정."""
    if not CONFIG.exists():
        return False
    import re
    text = CONFIG.read_text(encoding="utf-8")
    changed = False
    if "tick_source:" in text and "1m_real" not in text:
        text = re.sub(r"tick_source:\s*\S+", "tick_source: \"1m_real\"", text, count=1)
        changed = True
    if "news_source:" in text and "finbert" not in text:
        text = re.sub(r"news_source:\s*\S+", "news_source: \"finbert\"", text, count=1)
        changed = True
    if changed:
        CONFIG.write_text(text, encoding="utf-8")
    return True


def run_download() -> bool:
    """download_all_data.py 실행. 성공 시 True."""
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "download_all_data.py")],
        cwd=str(ROOT),
        timeout=600,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print("download_all_data.py stderr:", r.stderr[:500] if r.stderr else "")
    return r.returncode == 0


def run_verify() -> bool:
    """verify_thesis_data.py 실행. 성공 시 True."""
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_thesis_data.py")],
        cwd=str(ROOT),
        timeout=120,
        capture_output=True,
        text=True,
    )
    return r.returncode == 0 and REPORT.exists()


def report_has_all_match() -> tuple[bool, str]:
    """
    보고서의 종합 판정 구간에서 '불일치'가 하나도 없고,
    결론에 '모든 데이터가 논문 스펙과 **일치**'가 있으면 True.
    """
    if not REPORT.exists():
        return False, "보고서 없음"
    text = REPORT.read_text(encoding="utf-8")
    if "모든 데이터가 논문 스펙과 **일치**합니다" in text:
        return True, "전 항목 일치"
    in_summary = False
    for line in text.splitlines():
        if "## 종합 판정" in line:
            in_summary = True
            continue
        if in_summary and line.strip().startswith("**결론**"):
            break
        if in_summary and "불일치" in line and "|" in line:
            return False, f"불일치 항목 존재: {line.strip()}"
    return False, "종합 판정에서 불일치 없음이지만 결론 문구 미도달"


def main() -> None:
    ensure_config_thesis_aligned()
    print("=== 논문 데이터 일치 시까지 셀프검증 루프 시작 ===")
    iteration = 0
    match_rounds_left = 0

    while iteration < MAX_ITERATIONS:
        iteration += 1
        print(f"\n--- 라운드 {iteration}: 다운로드 실행 ---")
        if not run_download():
            print("다운로드 실패, 재시도 ...")
            continue
        print("다운로드 완료. 검증 실행 ...")
        if not run_verify():
            print("검증 스크립트 실패")
            continue
        all_ok, msg = report_has_all_match()
        if all_ok:
            if match_rounds_left == 0:
                match_rounds_left = VERIFY_AFTER_MATCH_ROUNDS
                print(f"\n*** 일치 도달: {msg}. 추가 셀프검증 {VERIFY_AFTER_MATCH_ROUNDS}회 실행 ***")
            match_rounds_left -= 1
            if match_rounds_left <= 0:
                print("\n=== 최종 목표 달성: 데이터가 논문 내용과 일치합니다. ===")
                print("보고서:", REPORT)
                return
            print(f"  추가 검증 남은 횟수: {match_rounds_left}")
            continue
        print(f"검증 결과: {msg}")
        if "불일치" in msg:
            print("다음 라운드에서 재다운로드·재검증 ...")

    print("\n=== 최대 반복 횟수 도달. 보고서를 확인하세요. ===")
    if REPORT.exists():
        print(REPORT.read_text(encoding="utf-8")[-2000:])


if __name__ == "__main__":
    main()
