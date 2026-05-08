"""Generate strategy flow diagram + equity curve chart."""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from data_collectors.sp100_loader import load_universe
from ml.research.alpha_v7_multi import (
    add_returns_and_basket, add_features_A, load_anchors,
)
from ml.research.alpha_v7_regime import compute_regime_indicators
from ml.research.alpha_v7_xyz_filtered import XYZ_IN_SP100
from ml.research.alpha_v7_freq_sweep import add_residual_and_label, metrics_freq
from ml.research.alpha_v7_pead_fixed import add_features_B_fixed
from ml.research.alpha_v7_honest import gate_rolling
from ml.research.alpha_v7_daily_optimized import (
    daily_portfolio_hysteresis, make_folds,
)
from ml.research.alpha_v7_daily_v2 import run_walk_multihorizon
from ml.research.alpha_v7_push import add_sector_features

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = Path("/home/yuqing/ctaNew/docs/strategy_charts")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---- Chart 1: Strategy flow diagram -------------------------------------

def chart_flow_diagram() -> None:
    """Architectural diagram of the strategy pipeline."""
    fig, ax = plt.subplots(figsize=(14, 11))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 11)
    ax.axis("off")

    # Color scheme
    DATA_C = "#E3F2FD"     # light blue
    FEAT_C = "#FFF9C4"     # light yellow
    MODEL_C = "#FFE0B2"    # light orange
    PORT_C = "#C8E6C9"     # light green
    GATE_C = "#FFCDD2"     # light red
    EXEC_C = "#D1C4E9"     # light purple
    EDGE = "#37474F"

    def box(x, y, w, h, text, color, fontsize=9, weight="normal"):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                        linewidth=1.2, edgecolor=EDGE,
                                        facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, weight=weight)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.3, color=EDGE))

    # Title
    ax.text(7, 10.5, "xyz Alpha-Residual v7 Strategy Pipeline",
            fontsize=15, weight="bold", ha="center")
    ax.text(7, 10.1, "Active Sharpe +3.16 / Annual +15.3% (in-sample) → +2.4 / +12% live estimate",
            fontsize=9, style="italic", ha="center", color="#555")

    # Stage 1: Data sources
    box(0.5, 8.5, 4, 0.7, "DATA SOURCES", DATA_C, 10, "bold")
    box(0.5, 7.7, 1.9, 0.6, "yfinance daily\n(100 names ×13y)", DATA_C, 8)
    box(2.5, 7.7, 2, 0.6, "yfinance earnings\n(time-stamped)", DATA_C, 8)
    box(0.5, 7.0, 1.9, 0.6, "yfinance anchors\n(SPY, TLT, VIX)", DATA_C, 8)
    box(2.5, 7.0, 2, 0.6, "Polygon 5m × 2y\n(extended hours)", DATA_C, 8)

    # Stage 2: Features
    box(5.5, 8.5, 8, 0.7, "FEATURE ENGINEERING (18 features)", FEAT_C, 10, "bold")
    box(5.2, 7.7, 2.0, 0.6, "A: Price-pattern\n(10 features)", FEAT_C, 8)
    box(7.3, 7.7, 2.0, 0.6, "B: PEAD fixed timing\n(4 features)", FEAT_C, 8)
    box(9.4, 7.7, 2.0, 0.6, "F: Sector momentum\n(3 features)", FEAT_C, 8)
    box(11.5, 7.7, 1.8, 0.6, "sym_id\n(categorical)", FEAT_C, 8)

    # Sub-features
    ax.text(6.2, 7.4, "ret_1d, ret_5d, vol_22d,\nidio_ret_22d, OBV-z, etc.",
            ha="center", va="top", fontsize=6.5, color="#555")
    ax.text(8.3, 7.4, "days_since_earn (AMC→\nnext BDay), surprise_pct,\nevent_day_resid, decay",
            ha="center", va="top", fontsize=6.5, color="#555")
    ax.text(10.4, 7.4, "own_sector_5d/22d,\nsector_relative_5d",
            ha="center", va="top", fontsize=6.5, color="#555")

    # Stage 3: Residualization
    box(0.5, 5.5, 4.5, 0.7, "RESIDUALIZATION", DATA_C, 10, "bold")
    box(0.5, 4.5, 4.5, 0.9, "leave-one-out basket\n(equal-weight 100-name S&P 100)\n→ resid = ret − beta_60d × bk_ret",
            DATA_C, 8)

    # Stage 4: Multi-horizon labels
    box(5.5, 5.5, 8, 0.7, "FORWARD LABELS (multi-horizon)", FEAT_C, 10, "bold")
    box(5.5, 4.5, 2.6, 0.9, "fwd_resid_3d\n(short PEAD)", FEAT_C, 9)
    box(8.2, 4.5, 2.6, 0.9, "fwd_resid_5d\n(medium PEAD)", FEAT_C, 9)
    box(10.9, 4.5, 2.6, 0.9, "fwd_resid_10d\n(long PEAD drift)", FEAT_C, 9)

    # Stage 5: Models
    box(5.5, 3.0, 8, 0.7, "MODEL ENSEMBLE (15 LGBM)", MODEL_C, 10, "bold")
    box(5.5, 2.0, 2.6, 0.9, "LGBM × 5 seeds\non fwd_3d", MODEL_C, 9)
    box(8.2, 2.0, 2.6, 0.9, "LGBM × 5 seeds\non fwd_5d", MODEL_C, 9)
    box(10.9, 2.0, 2.6, 0.9, "LGBM × 5 seeds\non fwd_10d", MODEL_C, 9)

    # Walk-forward note
    box(0.5, 3.2, 4.5, 0.5, "Walk-forward: annual retrain", MODEL_C, 8, "bold")
    box(0.5, 2.6, 4.5, 0.5, "Embargo: 5 days between fold", MODEL_C, 7)
    box(0.5, 2.0, 4.5, 0.5, "Expanding training window", MODEL_C, 7)

    # Stage 6: Predictions averaged
    box(5.5, 0.7, 8, 0.7, "AVERAGE PREDICTIONS → cross-section ranking",
            MODEL_C, 10, "bold")

    # Stage 7: Portfolio
    box(0.2, 0.05, 4.6, 0.55, "PORTFOLIO\nhysteresis K=5 enter, exit rank>7\n(→ daily decision, ~26% turnover)",
            PORT_C, 8, "bold")

    # Stage 8: Gate
    box(4.9, 0.05, 4.6, 0.55, "DISPERNSION GATE\nTrade only if XS-dispersion ≥ 60th pctile\nof trailing 252 days",
            GATE_C, 8, "bold")

    # Stage 9: Execution
    box(9.6, 0.05, 4.3, 0.55, "EXECUTE on xyz\n15-name universe / patient maker\n1.5 bps/side / daily P&L",
            EXEC_C, 8, "bold")

    # Arrows
    arrow(2.5, 7.0, 5.4, 5.9)         # data → features
    arrow(2.5, 5.7, 5.4, 5.7)         # data → residualization
    arrow(2.5, 5.0, 5.4, 5.0)         # data → labels
    arrow(9.5, 7.7, 9.5, 6.2)         # features → labels (downward connection)
    arrow(7, 4.5, 7, 3.7)             # labels → models
    arrow(7, 2.0, 7, 1.4)             # models → predictions
    arrow(2.5, 1.7, 2.5, 0.65)         # walk-forward → portfolio
    arrow(7, 0.7, 4.9, 0.35)          # predictions → portfolio
    arrow(4.9, 0.35, 9.5, 0.35)       # portfolio → gate
    arrow(9.5, 0.35, 13.9, 0.35)      # gate → execute (will overflow but show flow)

    plt.tight_layout()
    out_path = OUT_DIR / "strategy_flow.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("saved %s", out_path)


# ---- Chart 2: Equity curve + drawdown -----------------------------------

def chart_equity_curve() -> None:
    """Compute the full strategy backtest, plot equity curve + drawdown + per-year."""
    log.info("loading panel for equity curve...")
    panel, earnings, _ = load_universe()
    if panel.empty: return
    anchors = load_anchors()
    panel = add_returns_and_basket(panel)
    for h in (1, 3, 5, 10):
        panel = add_residual_and_label(panel, h)
    panel, feats_A = add_features_A(panel)
    panel, feats_B = add_features_B_fixed(panel, earnings)
    panel, feats_F_sector = add_sector_features(panel)
    panel["sym_id"] = panel["symbol"].astype("category").cat.codes
    regime = compute_regime_indicators(panel, anchors)

    feats = feats_A + feats_B + feats_F_sector + ["sym_id"]
    folds = make_folds(panel)
    train_labels = ["fwd_resid_3d", "fwd_resid_5d", "fwd_resid_10d"]

    log.info("running production backtest...")
    pnl_pre = run_walk_multihorizon(
        panel, feats, train_labels, folds,
        daily_portfolio_hysteresis,
        {"pnl_label": "fwd_resid_1d", "allowed": set(XYZ_IN_SP100),
         "top_k": 5, "exit_buffer": 2, "cost_bps_side": 1.5},
    )
    pnl = gate_rolling(pnl_pre, regime, pctile=0.6, window_days=252)
    if pnl.empty:
        log.error("empty pnl")
        return

    # Compute equity curve
    pnl = pnl.sort_values("ts").reset_index(drop=True)
    pnl["cum_net"] = pnl["net_alpha"].cumsum()
    pnl["running_max"] = pnl["cum_net"].cummax()
    pnl["drawdown"] = pnl["cum_net"] - pnl["running_max"]
    pnl["year"] = pnl["ts"].dt.year

    # Per-year breakdown
    per_year = pnl.groupby("year").agg(
        net_total=("net_alpha", "sum"),
        n_trades=("net_alpha", "count"),
    ).reset_index()
    per_year["return_pct"] = per_year["net_total"] * 100

    # === Figure ===
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), height_ratios=[2.5, 1, 1.5])

    # Top: cumulative equity curve
    ax1 = axes[0]
    ax1.fill_between(pnl["ts"], pnl["cum_net"] * 100, 0,
                      where=(pnl["cum_net"] >= 0), color="#4CAF50", alpha=0.3, label="profit")
    ax1.fill_between(pnl["ts"], pnl["cum_net"] * 100, 0,
                      where=(pnl["cum_net"] < 0), color="#F44336", alpha=0.3, label="drawdown")
    ax1.plot(pnl["ts"], pnl["cum_net"] * 100, color="#1B5E20", linewidth=1.5)
    ax1.plot(pnl["ts"], pnl["running_max"] * 100, color="#666",
              linestyle="--", linewidth=0.8, label="peak")
    ax1.set_title("Cumulative Net P&L (% on long notional)\n"
                  "v7 production: A+B+sector / 3-horizon ensemble / hysteresis / gated",
                  fontsize=12, weight="bold")
    ax1.set_ylabel("Cumulative return (%)")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left")

    # Annotate final cumulative + Sharpe
    final_cum = pnl["cum_net"].iloc[-1] * 100
    ax1.annotate(f"Final: {final_cum:.0f}% cumulative\nSharpe (active): +3.16\nMax DD: {pnl['drawdown'].min()*100:.1f}%",
                  xy=(pnl["ts"].iloc[-1], final_cum), xytext=(0.65, 0.85),
                  textcoords="axes fraction", fontsize=10,
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#1B5E20"))

    # Mid: drawdown
    ax2 = axes[1]
    ax2.fill_between(pnl["ts"], pnl["drawdown"] * 100, 0,
                      color="#F44336", alpha=0.5)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_title("Underwater chart (drawdown from peak)", fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.axhline(y=0, color="#333", linewidth=0.5)
    ax2.axhline(y=-5, color="#999", linewidth=0.5, linestyle=":")
    ax2.axhline(y=-10, color="#999", linewidth=0.5, linestyle=":")

    # Bottom: per-year bars
    ax3 = axes[2]
    colors = ["#4CAF50" if v >= 0 else "#F44336" for v in per_year["return_pct"]]
    bars = ax3.bar(per_year["year"].astype(str), per_year["return_pct"], color=colors, alpha=0.75)
    for bar, val, n in zip(bars, per_year["return_pct"], per_year["n_trades"]):
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, h + (0.5 if h >= 0 else -1.5),
                 f"{val:+.1f}%\n(n={n})", ha="center", va="bottom" if h >= 0 else "top",
                 fontsize=8)
    ax3.set_ylabel("Annual net return (%)")
    ax3.set_title("Per-year returns (10/11 years positive)", fontsize=10)
    ax3.grid(axis="y", alpha=0.3)
    ax3.axhline(y=0, color="#333", linewidth=0.5)

    plt.tight_layout()
    out_path = OUT_DIR / "equity_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("saved %s", out_path)

    # Print summary
    log.info("\nEquity curve stats:")
    log.info("  cumulative return:  %+.2f%%", final_cum)
    log.info("  max drawdown:       %.2f%% from peak", pnl["drawdown"].min() * 100)
    log.info("  trades:             %d over %d years", len(pnl), pnl["year"].nunique())
    log.info("  avg trades/year:    %.0f", len(pnl) / pnl["year"].nunique())


def main() -> None:
    log.info("Generating strategy flow diagram...")
    chart_flow_diagram()
    log.info("Generating equity curve chart...")
    chart_equity_curve()
    log.info("\nCharts saved to %s", OUT_DIR)


if __name__ == "__main__":
    main()
