#!/usr/bin/env python
"""
Phase 8: 실험 스크립트 — 설정으로 기간·자산 지정, 표 6-2~6-5 형식 결과 출력
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

if __name__ == "__main__":
    from src.run import main
    main()
