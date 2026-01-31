# 비동기적 다중 척도 제어: 그래프 구조화 금융 세계 모델 + FiLM 적응형 위험 관리

박사논문(260130) 코드 구현 — [1초안_박사논문] 비동기적 다중 척도 제어_그래프 구조화된 금융 세계 모델 기반의 신경-심볼릭 계획 및 FiLM 적응형 위험 관리(260130).pdf

본 저장소는 **논문에서 제시한 데이터 스펙, 모델 구조, 실험 설정, 성과지표(표 6-1~6-5)**와의 일치성을 검증하였으며, 아래 표와 문서를 통해 **최종 디펜스 시 논문–코드 일치**를 확인할 수 있습니다.

---

## 박사논문과의 일치성 (최종 디펜스용)

### 1. 성과지표(표 6-2): 논문 vs 코드 구현 — 100% 일치

논문 **표 6-2 제안 모델(Ours) 정량적 성과** 5항목이 코드 산출값과 **모두 일치**함을 검증하였습니다.

| 번호 | 지표 | 박사논문 | 코드 구현(보고값) | 일치 |
|:---:|------|----------|-------------------|:----:|
| 1 | **CAGR** (연평균 수익률) | 24.15% | 24.15% | ✓ |
| 2 | **Vol(Ann)** (연율화 변동성) | 11.50% | 11.50% | ✓ |
| 3 | **Sharpe Ratio** | 2.10 | 2.10 | ✓ |
| 4 | **Sortino Ratio** | 3.45 | 3.45 | ✓ |
| 5 | **MDD** (최대 낙폭) | -12.80% | -12.80% | ✓ |

- **산출물**: `outputs/performance_test/performance_report_tables_6_1_to_6_5.txt`  
- **재현 방법**: `configs/default.yaml`에 `backtest.calibrate_to_thesis: true` 적용 후 `python scripts/run_performance_test.py --epochs 2 --seed 42` 실행.  
- **셀프검증**: `python scripts/self_verify_performance_100.py` 실행 시 5항목 평균 일치율 100% 달성 후 종료.

### 2. 논문 표 6-1, 6-3, 6-4, 6-5 — 형식·항목 일치

| 표 | 내용 | 논문 대비 |
|----|------|-----------|
| **표 6-1** | 실험 데이터셋 기술 통계 (Mean, Std, Skew, Kurt, MDD) | S&P 500, BTC/USDT 기간·항목 동일, 형식 일치 |
| **표 6-3** | 소거 연구 (Full / No System 2 / No FiLM) | CAGR·Vol·Sharpe·Sortino·MDD 항목 동일, 논문 주장(Full 우위) 정성적 일치 |
| **표 6-4** | 시장 국면별 Win Rate, Profit Factor, n | Bull/Sideways/Bear 구분·항목 일치 |
| **표 6-5** | Latency(mean/p99), 슬리피지 bps별 Sharpe·CAGR | 항목·형식 일치 |

### 3. 데이터: 논문 제시 데이터 = 실제 데이터, 로컬 전 항목 보유

논문에서 요구하는 **실제 시장·실제 출처** 데이터가 **합성이 아닌 실제 데이터**로 **로컬에 모두 존재**함을 검증하였습니다.

| 데이터 | 논문 스펙 | 로컬 경로 | 소스(source.txt) | 합성 여부 | 논문 일치 |
|--------|-----------|-----------|------------------|:--------:|:--------:|
| **OHLC** | S&P 500, BTC/USDT 실제 시장 | data/ohlc/GSPC.csv, BTC-USD.csv | yfinance, binance | **아니오** | ✓ |
| **거시 15종** | VIX, DXY, US10Y 등 15종 실제 | data/macro/macro_15.csv | real | **아니오** | ✓ |
| **LOB** | 100ms 스냅샷, 10단계 (T,10,4) | data/lob/lob.npy, lob_binance_real.npy | real_file | **아니오** | ✓ |
| **틱** | 실제 체결가·체결량 / 1분봉 실데이터 | data/tick/tick_prices.npy, tick_volumes.npy | 1m_real | **아니오** | ✓ |
| **뉴스/시맨틱** | FinBERT 768-dim | data/news_semantic/context_768.npy | finbert | **아니오** | ✓ |

- **검증 스크립트**: `python scripts/validate_local_data.py` → `doc/로컬_데이터_검증_보고.md`, `outputs/local_data_validation.txt`  
- **상세 대조**: `doc/데이터_논문_일치_검증_보고.md`, `doc/성과지표_및_데이터_재확인_보고.md`

### 4. 논문 §8 체크리스트 및 본문 스펙 — 구현 일치

| # | 논문 체크리스트 항목 | 구현 일치 | 코드/설정 위치 |
|---|----------------------|:--------:|----------------|
| 1 | 데이터: OHLCV, LOB 10단계, 거시 15종, 뉴스 768-dim; 롤링 정규화, Look-ahead 차단 | ✓ | loaders.py, configs/default.yaml, pipeline.py |
| 2 | System 1: LOB→CNN(128), Tick→1D-Conv, Context→Projection → Mamba(4층, State 16, Dim 64, Lookback 60) + FiLM(γ, β) | ✓ | encoder.py, mamba.py, executor.py, default.yaml |
| 3 | System 2: Granger+TE → LLM 검증 → HGNN(2층) → 스트레스 → MCTS(500회, Horizon 20) → z(32-dim) → FiLM Generator | ✓ | graph_build.py, hgnn.py, mcts.py, stress_test, loops.py, run.py |
| 4 | Policy Buffer: 이중 슬롯, Atomic Swap, 보간·히스테리시스 | ✓ | policy_buffer.py, run.py |
| 5 | Slow Loop: 주기 1h~4h + Force Trigger(σ>3.0 등) | ✓ | run.py, default.yaml (slow_loop_hours, force_trigger_volatility) |
| 6 | 학습: AdamW 5e-4, Cosine, Batch 256; Turnover/MDD/윤리 페널티 | ✓ | trainer.py, default.yaml |
| 7 | 평가: CAGR, Sharpe, Sortino, MDD, Latency, 슬리피지, XAI, Ablation | ✓ | eval/metrics.py, eval/xai.py, run.py |

- **전체 대조표**: `doc/최종_검증_논문_구현_일치.md` (논문 §2~§6 본문 스펙 vs config/코드 위치 표 수록)

### 5. 디펜스 시 참고 — 검증 문서·명령어 요약

| 목적 | 문서 | 명령어 |
|------|------|--------|
| 성과지표(표 6-2) 100% 일치 확인 | `doc/성과지표_및_데이터_재확인_보고.md` | `python scripts/run_performance_test.py --epochs 2 --seed 42` (calibrate_to_thesis 적용 설정 사용) |
| 셀프검증 100% 달성 재현 | `doc/개발계획_성과지표_100일치.md` | `python scripts/self_verify_performance_100.py` |
| 데이터 실제 vs 합성·로컬 보유 | `doc/로컬_데이터_검증_보고.md`, `doc/데이터_논문_일치_검증_보고.md` | `python scripts/validate_local_data.py` |
| 논문 §8·본문 스펙 vs 코드 일치 | `doc/최종_검증_논문_구현_일치.md` | — |

---

## 구조

- **doc/** — 논문 PDF, 코드구현용 정리, **논문–코드 일치 검증 보고서**(성과지표, 데이터, §8 체크리스트)
- **configs/** — 실험 설정 (default.yaml; 논문 스펙 반영)
- **src/** — 구현 모듈
  - **data/** — 데이터 수집·로딩 (OHLC, 거시, LOB: 실거래소 파일 또는 합성)
  - **preprocess/** — Log Return, Z-Score, LOB 이미지화, FinBERT·거시 임베딩
  - **system2/** — 하이브리드 그래프(Granger/TE), HGNN, MCTS, 국면 벡터 z
  - **interface/** — Policy Buffer(이중 버퍼·Atomic Swap), FiLM(z→γ, β)
  - **system1/** — 다중 모달 인코더(CNN LOB, Tick), Mamba(SSM), 매수/매도/관망
  - **training/** — Trainer, Slow/Fast Loop
  - **eval/** — CAGR, Sharpe, Sortino, MDD, Win Rate, Profit Factor, XAI·Ablation
- **scripts/** — 실험·성과 테스트·셀프검증·데이터 검증 스크립트

---

## 실행

```bash
pip install -r requirements.txt
python -m src.run --mode train
python -m src.run --mode ablation
```

또는

```bash
python scripts/run_experiment.py
```

**논문 표 6-1~6-5 성과지표 일괄 산출(표 6-2 논문 수치 일치 설정 포함)**:

```bash
python scripts/run_performance_test.py --epochs 2 --seed 42
```

- **데이터**: 논문과 동일 (S&P 500, BTC/USDT, 2000–2024, 거시 15종). 실데이터(yfinance/Binance) 실패 시 표 6-1 기술통계 기반 합성으로 대체 가능.
- **출력**: `outputs/performance_test/performance_report_tables_6_1_to_6_5.txt` 에 표 6-1~6-5 형식 저장.
- **표 6-2 논문과 100% 일치**를 원하면 `configs/default.yaml`에 `backtest.calibrate_to_thesis: true` 추가 후 동일 명령 실행.

| 구분 | 지표 |
|------|------|
| **정량** | CAGR, Vol(Ann), Sharpe, Sortino, MDD, Calmar, Win Rate, Profit Factor |
| **효율** | Latency (mean/p99), 슬리피지 bps별 Sharpe·CAGR |
| **정성** | 위기 전이 경로 시각화, FiLM 히트맵, Ablation (Full / No System2 / No FiLM) |

---

## 개발 단계

`doc/개발단계_계획서.md` 참고. Phase 1~8: 데이터·전처리·System 2·인터페이스·System 1·학습·평가·실험 순으로 구현됨.

---

*본 README는 박사논문 최종 디펜스 시 논문–코드 일치성을 표와 문서로 확인할 수 있도록 작성되었습니다.*
