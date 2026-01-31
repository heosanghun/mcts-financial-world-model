"""
Phase 2: 텍스트·거시 임베딩 — FinBERT 768-dim, 거시 벡터화·투영
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Optional, List, Union

# Optional FinBERT
try:
    from transformers import AutoModel, AutoTokenizer
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


def embed_macro(
    macro_df_or_array: Union[np.ndarray, "pd.DataFrame"],
    out_dim: Optional[int] = None,
) -> np.ndarray:
    """거시 지표 벡터화·동일 차원 투영. (T, n_indicators) → (T, out_dim) or (T, n_indicators)."""
    if hasattr(macro_df_or_array, "values"):
        x = np.asarray(macro_df_or_array.values, dtype=np.float32)
    else:
        x = np.asarray(macro_df_or_array, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    # 차분 + 정규화 (논문: Differencing, 정규화)
    x = np.diff(x, axis=0, prepend=x[0:1])
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
    if out_dim is not None and x.shape[1] != out_dim:
        # 간단 투영: 랜덤 행렬 (학습 시 학습됨)
        proj = np.random.randn(x.shape[1], out_dim).astype(np.float32) * 0.1
        x = x @ proj
    return x.astype(np.float32)


def embed_context_openai(
    dates_or_texts: List[str],
    out_dim: int = 768,
    model: str = "text-embedding-3-small",
) -> np.ndarray:
    """
    OpenAI API 기반 시맨틱 컨텍스트 (논문: 뉴스·거시 → 768-dim).
    환경변수 OPENAI_API_KEY 사용. 미설정 시 zeros 반환.
    """
    import os
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or not dates_or_texts:
        return np.zeros((len(dates_or_texts) if dates_or_texts else 0, out_dim), dtype=np.float32)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except Exception:
        return np.zeros((len(dates_or_texts), out_dim), dtype=np.float32)
    texts = [f"Market risk and regime context for date or period: {d}" for d in dates_or_texts]
    out = []
    batch_size = 50
    for b in range(0, len(texts), batch_size):
        batch = texts[b : b + batch_size]
        try:
            resp = client.embeddings.create(model=model, input=batch)
            for i, e in enumerate(resp.data):
                idx = min(i, len(e.embedding) - 1)
                vec = e.embedding[:out_dim] if len(e.embedding) >= out_dim else e.embedding + [0.0] * (out_dim - len(e.embedding))
                out.append(np.array(vec, dtype=np.float32))
        except Exception:
            out.extend([np.zeros(out_dim, dtype=np.float32)] * len(batch))
    arr = np.stack(out) if out else np.zeros((len(dates_or_texts), out_dim), dtype=np.float32)
    if arr.shape[0] < len(dates_or_texts):
        pad = np.zeros((len(dates_or_texts) - arr.shape[0], out_dim), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    return arr.astype(np.float32)


def embed_context_finbert(
    dates_or_texts: List[str],
    out_dim: int = 768,
    model_name: str = "ProsusAI/finbert",
    max_length: int = 128,
) -> np.ndarray:
    """
    논문 §2.2와 일치: BERT 기반 (Fine-tuned) = FinBERT 768-dim.
    dates_or_texts: 날짜 문자열이면 "Market date {d}"로, 텍스트면 그대로 FinBERT 임베딩.
    """
    if not dates_or_texts:
        return np.zeros((0, out_dim), dtype=np.float32)
    texts = []
    for x in dates_or_texts:
        if x and (len(x) > 20 or "-" in x[:10] or "/" in x[:10]):
            texts.append(f"Market date and regime context: {x[:64]}")
        else:
            texts.append(str(x)[:64] if x else "Market context")
    emb = embed_text_finbert(texts, model_name=model_name, max_length=max_length)
    if emb.shape[1] > out_dim:
        emb = emb[:, :out_dim]
    elif emb.shape[1] < out_dim:
        pad = np.zeros((emb.shape[0], out_dim - emb.shape[1]), dtype=np.float32)
        emb = np.concatenate([emb, pad], axis=1)
    return emb.astype(np.float32)


def embed_text_finbert(
    texts: List[str],
    model_name: str = "ProsusAI/finbert",
    max_length: int = 128,
    device: Optional[str] = None,
) -> np.ndarray:
    """뉴스 헤드라인 → FinBERT 768-dim. 없으면 랜덤 768-dim (플레이스홀더)."""
    if not _HAS_TRANSFORMERS:
        return np.random.randn(len(texts), 768).astype(np.float32) * 0.01
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model = model.to(device).eval()
    except Exception:
        return np.random.randn(len(texts), 768).astype(np.float32) * 0.01
    out = []
    for t in texts:
        inp = tokenizer(t[:512], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            h = model(**inp).last_hidden_state[:, 0, :].cpu().numpy()
        out.append(h.squeeze(0))
    return np.stack(out).astype(np.float32)
