"""
논문용 데이터 개요 차트 생성 (선택).
OHLC 시계열, 일일 수익률 분포, 거시 지표, LOB 스냅샷 예시.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

DATA = ROOT / "data"
OUT = ROOT / "outputs" / "figures"


def main():
    if not _HAS_MPL:
        print("matplotlib 없음. pip install matplotlib")
        return
    OUT.mkdir(parents=True, exist_ok=True)

    # 1) OHLC 종가 시계열 (GSPC)
    ohlc_path = DATA / "ohlc" / "GSPC.csv"
    if ohlc_path.exists():
        df = pd.read_csv(ohlc_path, index_col=0, parse_dates=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df.index, df["close"], linewidth=0.8)
        ax.set_title("OHLC Close (GSPC) - Data Overview")
        ax.set_ylabel("Close")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "chart_ohlc_close.png", dpi=150)
        plt.close()
        print("저장: chart_ohlc_close.png")

    # 2) 일일 수익률 분포 (히스토그램)
    if ohlc_path.exists():
        df = pd.read_csv(ohlc_path, index_col=0, parse_dates=True)
        ret = df["close"].pct_change().dropna()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(ret, bins=80, density=True, alpha=0.7, edgecolor="black", linewidth=0.3)
        ax.axvline(ret.mean(), color="red", linestyle="--", label=f"Mean={ret.mean():.4f}")
        ax.set_title("Daily Return Distribution (GSPC)")
        ax.set_xlabel("Return")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "chart_return_histogram.png", dpi=150)
        plt.close()
        print("저장: chart_return_histogram.png")

    # 3) 거시 지표 시계열 (VIX, 1~2개만)
    macro_path = DATA / "macro" / "macro_15.csv"
    if macro_path.exists():
        df = pd.read_csv(macro_path, index_col=0, parse_dates=True)
        cols = [c for c in df.columns if "VIX" in str(c) or c == df.columns[0]][:2]
        if cols:
            fig, ax = plt.subplots(figsize=(10, 3))
            for c in cols:
                ax.plot(df.index, df[c], label=c, linewidth=0.7)
            ax.set_title("Macro Indicators (Sample) - Data Overview")
            ax.set_ylabel("Level")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(OUT / "chart_macro_sample.png", dpi=150)
            plt.close()
            print("저장: chart_macro_sample.png")

    # 4) LOB 1스냅샷 예시 (막대그래프)
    lob_path = DATA / "lob" / "lob.npy"
    if lob_path.exists():
        lob = np.load(lob_path)
        t_idx = min(0, lob.shape[0] - 1)
        snap = lob[t_idx]  # (10, 4): bid_p, bid_s, ask_p, ask_s
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(snap.shape[0])
        ax.bar(x - 0.2, snap[:, 1], width=0.35, label="Bid size")
        ax.bar(x + 0.2, snap[:, 3], width=0.35, label="Ask size")
        ax.set_title("LOB Snapshot Example (Bid/Ask Size)")
        ax.set_xlabel("Level")
        ax.set_ylabel("Size")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(OUT / "chart_lob_snapshot.png", dpi=150)
        plt.close()
        print("저장: chart_lob_snapshot.png")

    print(f"\n차트 저장 경로: {OUT}")


if __name__ == "__main__":
    main()
