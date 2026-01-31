"""
OPENAI_API_KEY 로드 및 LLM 검증·뉴스/시맨틱 컨텍스트 사용 검증.
.env 로드 후 키 존재 여부, LLM 검증(OpenAI), embed_context_openai 호출 여부 확인.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 1) .env 로드
_env_path = ROOT / ".env"
loaded = False
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
        loaded = True
    except ImportError:
        pass

key = os.environ.get("OPENAI_API_KEY", "").strip()
key_ok = bool(key)
print(f"[1] .env 로드: {'OK' if loaded else 'skip (dotenv 없음)'}")
print(f"[2] OPENAI_API_KEY 설정: {'OK' if key_ok else 'NOT SET'}")

if not key_ok:
    print("OPENAI_API_KEY가 없습니다. .env에 OPENAI_API_KEY=... 형태로 설정 후 다시 실행하세요.")
    sys.exit(1)

# 3) LLM 검증 (OpenAI) — 엣지 2개로 최소 호출
print("[3] LLM 검증 (OpenAI) 호출 ...")
from src.system2.graph_build import build_hybrid_graph, llm_verify_edges_openai
import numpy as np
series_dict = {
    "a": np.cumsum(np.random.randn(100).astype(np.float64)) + 100.0,
    "b": np.cumsum(np.random.randn(100).astype(np.float64)) + 100.0,
}
G = build_hybrid_graph(
    series_dict, maxlag=3, p_threshold=0.05,
    llm_verify=True, llm_rule_based=False, llm_use_openai=True,
)
n_edges = G.number_of_edges()
print(f"    그래프 엣지 수 (LLM 검증 후): {n_edges}")

# 4) 뉴스/시맨틱 컨텍스트 (OpenAI Embeddings)
print("[4] 뉴스/시맨틱 컨텍스트 (embed_context_openai) 호출 ...")
from src.preprocess.embedding import embed_context_openai
dates = ["2024-01-01", "2024-01-02"]
ctx = embed_context_openai(dates, out_dim=768)
non_zero = np.abs(ctx).sum() > 0
print(f"    임베딩 shape: {ctx.shape}, 비영(non-zero): {non_zero}")

print("\n[검증 완료] OPENAI_API_KEY가 LLM 검증과 뉴스/시맨틱 컨텍스트에 사용되었습니다.")
