"""
박사논문(260130) 통합 실행: 데이터 → 전처리 → System 2/1 → 학습 → 평가(Latency, Slippage, XAI, Ablation)
"""
from __future__ import annotations

import os
import sys
import argparse
import time
import yaml
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# .env에서 OPENAI_API_KEY 등 환경변수 로드 (프로젝트 루트의 .env)
_env_path = ROOT / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
    except ImportError:
        pass


def load_config(path: str = None) -> dict:
    path = path or ROOT / "configs" / "default.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_backtest(
    model,
    lob_img,
    ret,
    vol_z,
    lookback,
    T,
    policy_buffer,
    device,
    use_fixed_film=False,
    fixed_gamma=None,
    fixed_beta=None,
    collect_latency=False,
    collect_film_history=False,
    film_gamma_list=None,
    film_beta_list=None,
    context_array=None,
    slow_loop_fn=None,
    slow_loop_interval_steps=252,
    force_trigger_volatility=3.0,
    force_trigger_lob_imbalance=0.3,
    volatility_window=20,
    target_vol_ann=None,
    periods_per_year=252.0,
):
    """백테스트. Slow Loop 주기 호출 + Force Trigger(σ 또는 호가 불균형) 시 즉시 Slow 호출.
    target_vol_ann: 논문 Vol(Ann) 목표(%, 예 11.5). 지정 시 롤링 변동성 기준 포지션 스케일로 변동성 타겟팅."""
    from src.training.loops import run_fast_loop_step
    strategy_returns = np.zeros(T)
    actions = np.zeros(T, dtype=np.int64)
    timings_ms = []
    if film_gamma_list is not None:
        film_gamma_list.clear()
    if film_beta_list is not None:
        film_beta_list.clear()
    d_model = getattr(model, "model_dim", 64)
    vol_target = float(target_vol_ann) / 100.0 if target_vol_ann is not None else None
    vol_roll_min = max(lookback, volatility_window or 20)
    model.eval()
    with torch.no_grad():
        for i in range(lookback, T):
            if slow_loop_fn is not None:
                if (i - lookback) > 0 and (i - lookback) % slow_loop_interval_steps == 0:
                    slow_loop_fn()
                if volatility_window and i >= lookback + volatility_window:
                    roll_std = np.std(ret[i - volatility_window : i])
                    if roll_std >= force_trigger_volatility * 0.01:
                        slow_loop_fn()
                if force_trigger_lob_imbalance is not None and lob_img.shape[0] > i:
                    bid_sum = float(np.sum(lob_img[i, :, 0]))
                    ask_sum = float(np.sum(lob_img[i, :, 1]))
                    denom = bid_sum + ask_sum + 1e-10
                    imbalance = (bid_sum - ask_sum) / denom
                    if abs(imbalance) >= force_trigger_lob_imbalance:
                        slow_loop_fn()
            ctx = None
            if context_array is not None and i < context_array.shape[0]:
                ctx = torch.from_numpy(context_array[i : i + 1]).float().to(device)
            lob_b = torch.from_numpy(lob_img[i - lookback : i + 1]).float().unsqueeze(0).to(device)
            tick_b = torch.from_numpy(np.column_stack([ret[i - lookback : i + 1], vol_z[i - lookback : i + 1]])).float().unsqueeze(0).to(device)
            if use_fixed_film and fixed_gamma is not None and fixed_beta is not None:
                g = torch.full((1, d_model), float(fixed_gamma), device=device, dtype=torch.float32)
                b = torch.full((1, d_model), float(fixed_beta), device=device, dtype=torch.float32)
                logits = model(lob_b, tick_b, ctx, gamma=g, beta=b)
                action = logits.argmax(dim=-1).item()
            else:
                if collect_latency:
                    t0 = time.perf_counter()
                _, action = run_fast_loop_step(model, lob_b, tick_b, policy_buffer, ctx, device)
                if collect_latency:
                    timings_ms.append((time.perf_counter() - t0) * 1000.0)
            if policy_buffer is not None and hasattr(policy_buffer, "tick_interpolate"):
                policy_buffer.tick_interpolate()
            if collect_film_history and policy_buffer is not None and not use_fixed_film:
                ga, be = policy_buffer.read()
                film_gamma_list.append(np.atleast_1d(ga).mean())
                film_beta_list.append(np.atleast_1d(be).mean())
            position = (action - 1.0)
            if vol_target is not None and i >= vol_roll_min:
                past = strategy_returns[lookback:i]
                past = past[np.isfinite(past)]
                if len(past) >= volatility_window:
                    roll_std = np.std(past)
                    roll_vol_ann = roll_std * np.sqrt(periods_per_year) if roll_std > 1e-10 else 1e-10
                    scale = min(1.0, vol_target / roll_vol_ann)
                    position = position * scale
            strategy_returns[i] = position * ret[i]
            actions[i] = action
    strategy_returns = strategy_returns[lookback:]
    actions = actions[lookback:]
    position_changes = np.diff(actions.astype(np.float64), prepend=actions[0])
    return strategy_returns, position_changes, timings_ms


def main():
    parser = argparse.ArgumentParser(description="비동기적 다중 척도 제어 금융 AI")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "ablation"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir or str(ROOT / "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data — 논문과 동일 (실패 시 표 6-1 기반 합성 데이터 사용)
    from src.data.loaders import (
        load_ohlc_thesis_aligned,
        load_macro_thesis_aligned,
        load_lob_synthetic,
        load_lob_from_file,
        load_lob_binance_snapshot,
        tick_from_ohlc,
        tick_from_1m,
        load_ohlc_binance,
        _binance_symbol,
    )
    from src.preprocess.pipeline import preprocess_ohlc, lob_to_image
    from src.preprocess.embedding import embed_macro, embed_text_finbert, embed_context_openai
    import pandas as pd
    symbols = cfg["data"]["assets"].get("equity", ["^GSPC"]) + cfg["data"]["assets"].get("crypto", ["BTC-USD"])
    ohlc_dict = load_ohlc_thesis_aligned(
        symbols, start=cfg["data"]["start_date"], end=cfg["data"]["end_date"],
        interval="1d", fallback_synthetic=True, seed=args.seed,
    )
    if not ohlc_dict:
        from src.data.loaders import generate_thesis_synthetic_ohlc
        ohlc_dict = {"^GSPC": generate_thesis_synthetic_ohlc(cfg["data"]["start_date"], cfg["data"]["end_date"], "SP500", args.seed)}
    key = list(ohlc_dict.keys())[0]
    df = ohlc_dict[key]
    close = np.asarray(df["close"].values, dtype=np.float64)
    volume = np.asarray(df["volume"].values, dtype=np.float64)
    ret, vol_z = preprocess_ohlc(
        close, volume, cfg["preprocess"]["rolling_zscore_window"],
        use_minmax=cfg["preprocess"].get("use_minmax", True),
    )
    # LOB: 논문 일치 — 실거래소 파일(lob_path) 있으면 사용, 없으면 합성 + 바이낸스 마지막 1스냅
    lob = None
    lob_path = cfg["data"].get("lob_path")
    if lob_path and Path(lob_path).exists():
        lob = load_lob_from_file(lob_path, n_levels=cfg["data"]["lob_levels"], align_length=len(close))
    if lob is None:
        lob = load_lob_synthetic(close, volume, n_levels=cfg["data"]["lob_levels"], n_snapshots=len(close), seed=args.seed)
        bn_sym = _binance_symbol(key)
        if bn_sym:
            bn_lob = load_lob_binance_snapshot(symbol=bn_sym, limit=cfg["data"]["lob_levels"])
            if bn_lob is not None and bn_lob.shape[1] == lob.shape[1]:
                lob[-1:] = bn_lob[:, : lob.shape[1], :]
    lob_img = lob_to_image(lob)

    # 틱: 논문 일치 — tick_source=1m_real 이면 1분봉 실데이터를 틱 대용으로 사용
    tick_daily_1m = None  # (n_days, lookback, 2) when 1m_real
    if cfg["data"].get("tick_source") == "1m_real":
        bn_sym = _binance_symbol(key)
        if bn_sym:
            ohlc_1m = load_ohlc_binance(bn_sym, start=cfg["data"]["start_date"], end=cfg["data"]["end_date"], interval="1m")
            if ohlc_1m is not None and len(ohlc_1m) >= 60:
                from src.preprocess.pipeline import preprocess_ohlc
                close_1m = ohlc_1m["close"].values.astype(np.float64)
                vol_1m = ohlc_1m["volume"].values.astype(np.float64)
                ret_1m, vol_z_1m = preprocess_ohlc(close_1m, vol_1m, cfg["preprocess"]["rolling_zscore_window"], use_minmax=False)
                try:
                    dates_1m = np.array([getattr(t, "date", lambda: t)() for t in ohlc_1m.index])
                except Exception:
                    dates_1m = pd.to_datetime(ohlc_1m.index).date
                lookback = min(cfg["preprocess"].get("lookback_ticks", 60), 60)
                tick_daily_1m = np.zeros((len(close), lookback, 2), dtype=np.float32)
                for i in range(len(close)):
                    day = getattr(df.index[i], "date", lambda: df.index[i])() if hasattr(df.index[i], "date") else df.index[i]
                    mask = dates_1m == day
                    idx = np.where(mask)[0]
                    if len(idx) >= lookback:
                        sel = idx[-lookback:]
                        tick_daily_1m[i] = np.column_stack([ret_1m[sel], vol_z_1m[sel]])
                    else:
                        st = max(0, i - lookback)
                        tick_daily_1m[i] = np.column_stack([np.resize(ret[st : i + 1], lookback), np.resize(vol_z[st : i + 1], lookback)])
    macro_tickers = cfg["data"].get("macro_tickers", cfg["data"].get("macro_tickers_fallback", ["^VIX", "DX-Y.NYB", "^TNX"]))
    macro_df = load_macro_thesis_aligned(
        macro_tickers[:15] if isinstance(macro_tickers, list) else list(macro_tickers)[:15],
        start=cfg["data"]["start_date"], end=cfg["data"]["end_date"],
        fallback_synthetic=True, seed=args.seed,
    )
    context_dim = cfg["preprocess"].get("context_dim", 768)
    context_array = np.zeros((len(close), context_dim), dtype=np.float32)
    if os.environ.get("OPENAI_API_KEY", "").strip():
        dates = (df.index.astype(str).tolist() if hasattr(df, "index") else [str(i) for i in range(len(close))])[: len(close)]
        openai_ctx = embed_context_openai(dates, out_dim=context_dim)
        if openai_ctx.shape[0] == len(close) and openai_ctx.shape[1] == context_dim:
            context_array = openai_ctx.astype(np.float32)
    if context_array.max() == 0 and context_array.min() == 0 and not macro_df.empty and len(macro_df) > 0:
        idx = df.index if hasattr(df, "index") else pd.RangeIndex(len(close))
        macro_aligned = macro_df.reindex(idx).ffill().bfill().fillna(0)
        if len(macro_aligned) == len(close):
            macro_embed = embed_macro(macro_aligned.values.astype(np.float32), out_dim=cfg["preprocess"].get("macro_dim", 15))
            if macro_embed.shape[0] == len(close):
                if macro_embed.shape[1] < context_dim:
                    proj = np.random.randn(macro_embed.shape[1], context_dim).astype(np.float32) * 0.1
                    context_array = (macro_embed @ proj).astype(np.float32)
                else:
                    context_array = macro_embed[:, :context_dim].astype(np.float32)

    # System 2
    from src.system2.graph_build import build_hybrid_graph
    from src.system2.hgnn import HGNN
    from src.system2.mcts import MCTSPlanner
    from src.system2.regime_vector import RegimeVectorBuilder
    from src.interface.policy_buffer import PolicyBuffer
    from src.interface.film import FiLMGenerator
    from src.training.loops import run_slow_loop_step

    series_dict = {k: np.asarray(v).ravel().astype(np.float64) for k, v in [(key, close)]}
    if not macro_df.empty and macro_df.shape[1] > 0:
        n_macro = min(15, macro_df.shape[1])
        for c, col in enumerate(macro_df.columns[:n_macro]):
            s = macro_df[col].reindex(df.index if hasattr(df, "index") else pd.RangeIndex(len(close))).ffill().bfill().fillna(0).values.astype(np.float64)
            if len(s) == len(close):
                series_dict[f"_macro_{c}"] = s
    if len(series_dict) < 2:
        series_dict["_macro"] = np.diff(close, prepend=close[0]).astype(np.float64)
    G = build_hybrid_graph(
        series_dict,
        maxlag=cfg["system2"]["granger_maxlag"],
        p_threshold=cfg["system2"]["granger_p_threshold"],
        llm_verify=True,
        llm_rule_based=True,
        llm_use_openai=cfg["system2"].get("llm_use_openai", True),
    )
    node_list = list(G.nodes())
    if len(node_list) < 2:
        node_list = list(series_dict.keys())[:2]
        G.add_nodes_from(node_list)
    adj = HGNN.graph_to_adj(G, node_list)
    node_dim = 5
    hgnn = HGNN(
        node_dim=node_dim, hidden_dim=64, out_dim=cfg["system2"]["z_dim"],
        num_layers=cfg["system2"]["hgnn_layers"],
        num_hyperedge_groups=cfg["system2"].get("hyperedge_groups", 12),
    ).to(device)
    mcts = MCTSPlanner(
        n_actions=5,
        horizon=cfg["system2"]["mcts_horizon"],
        n_simulations=cfg["system2"]["mcts_simulations"],
    )
    regime_builder = RegimeVectorBuilder(hgnn_out=cfg["system2"]["z_dim"], path_seq_dim=4, path_hidden=32, z_dim=cfg["system2"]["z_dim"]).to(device)
    film_gen = FiLMGenerator(z_dim=cfg["system2"]["z_dim"], num_channels=cfg["system1"]["mamba_model_dim"], clip=cfg["system1"]["film_clip"]).to(device)
    policy_buffer = PolicyBuffer(
        clip=cfg["system1"]["film_clip"],
        interpolate=cfg["interface"].get("film_interpolation", True),
        interpolation_steps=5,
        hysteresis_forward=0.3,
        hysteresis_back=0.5,
    )

    # System 1
    from src.system1.executor import System1Executor
    from src.training.trainer import Trainer
    from src.training.train_loop import run_training_loop
    model = System1Executor(model_dim=cfg["system1"]["mamba_model_dim"], num_actions=3).to(device)

    # Slow loop once to init policy
    series_dict_f = {k: np.asarray(v).ravel().astype(np.float32) for k, v in series_dict.items()}
    run_slow_loop_step(series_dict_f, hgnn, regime_builder, mcts, film_gen, policy_buffer, device)

    lookback = min(cfg["preprocess"]["lookback_ticks"], 60)
    T = min(252 * 2, len(ret))
    returns = ret[:T]
    returns_eval = returns[lookback:]
    bench = returns_eval

    # P1: 학습 루프 (mode train 시)
    if args.mode == "train":
        trainer = Trainer(
            model,
            lr=cfg["training"]["lr"],
            weight_decay=cfg["training"]["weight_decay"],
            batch_size=cfg["training"]["batch_size"],
            device=device,
        )
        run_training_loop(
            model, trainer, lob_img, ret, vol_z,
            lookback=lookback, batch_size=cfg["training"]["batch_size"], epochs=args.epochs,
            device=device, out_dir=out_dir,
        )
        torch.save(model.state_dict(), out_dir / "model_trained.pt")

    def slow_loop_fn():
        run_slow_loop_step(series_dict_f, hgnn, regime_builder, mcts, film_gen, policy_buffer, device)

    slow_interval = int(cfg["interface"].get("slow_loop_hours", 1) * 252 / 6.5)
    force_vol = float(cfg["interface"].get("force_trigger_volatility", 3.0)) * 0.01
    film_gamma_hist, film_beta_hist = [], []
    strategy_returns, position_changes, timings_ms = run_backtest(
        model, lob_img, ret, vol_z, lookback, T, policy_buffer, device,
        use_fixed_film=False, collect_latency=True, collect_film_history=True,
        film_gamma_list=film_gamma_hist, film_beta_list=film_beta_hist,
        context_array=context_array[:T],
        slow_loop_fn=slow_loop_fn,
        slow_loop_interval_steps=max(1, slow_interval),
        force_trigger_volatility=force_vol,
        force_trigger_lob_imbalance=float(cfg["interface"].get("force_trigger_lob_imbalance", 0.3)),
        volatility_window=20,
    )

    # 정량 지표
    from src.eval.metrics import compute_all_metrics, latency_stats, slippage_sensitivity
    metrics = compute_all_metrics(strategy_returns, bench, periods_per_year=252.0)
    # P2: Latency
    lat = latency_stats(np.array(timings_ms))
    metrics.update(lat)
    print("Strategy metrics (sample period):", metrics)

    # P3: Slippage 민감도
    slippage_results = slippage_sensitivity(strategy_returns, position_changes, bps_list=[0.0, 5.0, 10.0, 20.0], benchmark_returns=bench, periods_per_year=252.0)
    print("Slippage sensitivity (bps -> sharpe):", {bps: m["sharpe"] for bps, m in slippage_results.items()})

    # P4: XAI 시각화 호출 및 저장
    from src.eval.xai import visualize_crisis_path, film_heatmap
    node_labels = node_list if len(node_list) <= 20 else [str(i) for i in range(adj.shape[0])]
    visualize_crisis_path(adj, node_labels=node_labels, save_path=str(out_dir / "xai_crisis_path.png"))
    if film_gamma_hist and film_beta_hist:
        film_heatmap(np.array(film_gamma_hist), np.array(film_beta_hist), save_path=str(out_dir / "xai_film_heatmap.png"))
    print("XAI outputs saved to", out_dir)

    # P5: 진짜 Ablation (no_system2: 고정 γ,β / no_film: γ=1, β=0)
    if args.mode == "ablation":
        # no_system2: Slow Loop 미갱신 → 버퍼를 (1, 0)으로 고정 후 백테스트
        policy_buffer.write(np.array(1.0, dtype=np.float32), np.array(0.0, dtype=np.float32))
        strat_no_s2, pos_no_s2, _ = run_backtest(model, lob_img, ret, vol_z, lookback, T, policy_buffer, device, use_fixed_film=False)
        metrics_no_s2 = compute_all_metrics(strat_no_s2, bench, periods_per_year=252.0)
        # no_film: γ=1, β=0 강제
        strat_no_film, pos_no_film, _ = run_backtest(model, lob_img, ret, vol_z, lookback, T, None, device, use_fixed_film=True, fixed_gamma=1.0, fixed_beta=0.0)
        metrics_no_film = compute_all_metrics(strat_no_film, bench, periods_per_year=252.0)
        # 버퍼 복구 (다음에 full 쓰려면 Slow 다시 한 번 호출 가능)
        run_slow_loop_step(series_dict_f, hgnn, regime_builder, mcts, film_gen, policy_buffer, device)
        print("Ablation Full:", metrics)
        print("Ablation (no System2, fixed γ=1 β=0):", metrics_no_s2)
        print("Ablation (no FiLM, γ=1 β=0):", metrics_no_film)
        with open(out_dir / "ablation_metrics.txt", "w", encoding="utf-8") as f:
            f.write("Full: " + str(metrics) + "\n")
            f.write("no_System2: " + str(metrics_no_s2) + "\n")
            f.write("no_FiLM: " + str(metrics_no_film) + "\n")

    print("Done.")


if __name__ == "__main__":
    main()
