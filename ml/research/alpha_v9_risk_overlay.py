"""Risk overlays composed with conviction_gate p=0.30 (validated winner).

Tests whether market-state filters add Sharpe over the gate-only baseline:

  A. Drawdown brake: pause when rolling 22-cycle (~4d) realized cumulative
     net is below threshold; resume once recovered.

  B. High-vol gate: skip when basket realized vol (24h trailing) is in
     top 30% of trailing 252-cycle distribution. Hypothesis: high-vol
     regimes wash out cross-sectional alpha.

  C. Low-vol gate: skip bottom 30%. Hypothesis: low-vol = no opportunity.

  D. BTC-extreme gate: skip when |BTC log return last 24h| is in top 30%.
     BTC dominance regime; alts move together and residual signal suppressed.

All built on top of conviction_gate p=0.30. Multi-OOS paired vs gate-only.
"""
from __future__ import annotations
import json
import sys
import time
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import (
    XS_FEATURE_COLS_V6_CLEAN, XS_RANK_SOURCES,
    add_basket_features, add_engineered_flow_features, add_xs_rank_features,
    build_basket, build_kline_features, list_universe, make_xs_alpha_labels,
)
from ml.research.alpha_v4_xs_1d import (
    ENSEMBLE_SEEDS, _multi_oos_splits, _slice, _train,
)
from ml.research.alpha_v4_xs import block_bootstrap_ci

HORIZON = 48
TOP_K = 7
COST_PER_LEG = 4.5
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
GATE_LOOKBACK = 252
GATE_PCTILE_CONV = 0.30
DD_LOOKBACK = 22       # ~4 days at h=48 cadence
DD_THRESHOLD_BPS = -50
VOL_TRAIL = 288        # 1 day in 5min bars
VOL_PCTILE_HI = 0.70
VOL_PCTILE_LO = 0.30
BTC_MOVE_PCTILE = 0.70
NEW_SYMBOLS = {"ETCUSDT", "HBARUSDT", "ICPUSDT", "LDOUSDT", "TRBUSDT",
                "AAVEUSDT", "MKRUSDT", "AXSUSDT", "GMXUSDT",
                "1000PEPEUSDT", "1000SHIBUSDT", "TONUSDT", "ORDIUSDT", "WIFUSDT"}
OUT_DIR = REPO / "outputs/h48_risk_overlay"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sharpe_est = lambda x: x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR) if x.std() > 0 else 0


def build_panel_with_market_state():
    """Wide panel + basket realized vol + |BTC 24h return| at each bar."""
    universe_full = sorted(list_universe(min_days=200))
    orig25 = sorted([s for s in universe_full if s not in NEW_SYMBOLS])
    print(f"Building panel for {len(orig25)} ORIG25 syms…")
    feats = {s: build_kline_features(s) for s in orig25}
    closes = pd.DataFrame({s: feats[s]["close"] for s in orig25}).sort_index()
    basket_ret, basket_close = build_basket(closes)
    btc_close = closes["BTCUSDT"].copy()

    sym_to_id = {s: i for i, s in enumerate(orig25)}
    enriched = {}
    for s in orig25:
        f = feats[s].reindex(closes.index)
        f = add_basket_features(f, basket_close, basket_ret)
        f = add_engineered_flow_features(f)
        f["sym_id"] = sym_to_id[s]
        if f.index.tz is None:
            f.index = f.index.tz_localize("UTC")
        enriched[s] = f
    labels = make_xs_alpha_labels(enriched, basket_close, HORIZON)

    rank_cols = [c for c in XS_FEATURE_COLS_V6_CLEAN if c.endswith("_xs_rank")]
    src_cols = list({src for src, dst in XS_RANK_SOURCES.items() if dst in rank_cols})
    needed = list(set(list(XS_FEATURE_COLS_V6_CLEAN)
                        + ["sym_id", "autocorr_pctile_7d", "beta_short_vs_bk"]
                        + src_cols) - set(rank_cols))
    frames = []
    for s, f in enriched.items():
        avail = [c for c in needed if c in f.columns]
        df = f[avail].join(labels[s], how="inner")
        df["symbol"] = s
        df = df.reset_index().rename(columns={"index": "open_time"})
        for c in df.select_dtypes("float64").columns:
            df[c] = df[c].astype("float32")
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True, sort=False)
    panel = add_xs_rank_features(panel, sources=XS_RANK_SOURCES)
    for c in rank_cols:
        if c in panel.columns:
            panel[c] = panel[c].astype("float32")
    panel = panel.dropna(subset=list(XS_FEATURE_COLS_V6_CLEAN)
                            + ["autocorr_pctile_7d", "demeaned_target", "return_pct"])

    # --- Market-state series (per-bar, broadcast) ---
    bk_ret_1bar = basket_close.pct_change()
    bk_vol_24h = (bk_ret_1bar.rolling(VOL_TRAIL, min_periods=VOL_TRAIL // 4).std()
                   * np.sqrt(VOL_TRAIL))
    btc_ret_24h = (btc_close.pct_change(VOL_TRAIL).abs()).rename("btc_abs_ret_24h")
    market = pd.DataFrame({"bk_vol_24h": bk_vol_24h, "btc_abs_ret_24h": btc_ret_24h})
    if market.index.tz is None:
        market.index = market.index.tz_localize("UTC")
    market = market.reset_index().rename(columns={"index": "open_time"})
    panel = panel.merge(market, on="open_time", how="left")

    print(f"  panel: {len(panel):,} rows  bars: {panel['open_time'].nunique():,}")
    print(f"  bk_vol_24h non-null: {panel['bk_vol_24h'].notna().sum():,}")
    print(f"  btc_abs_ret_24h non-null: {panel['btc_abs_ret_24h'].notna().sum():,}")
    return panel


def _bn_scales(top_g, bot_g):
    beta_L = top_g["beta_short_vs_bk"].mean()
    beta_S = bot_g["beta_short_vs_bk"].mean()
    if beta_L < 0.1 or beta_S < 0.1 or (beta_L + beta_S) < 0.3:
        return 1.0, 1.0
    denom = beta_L + beta_S
    return (float(np.clip(2.0 * beta_S / denom, 0.5, 1.5)),
            float(np.clip(2.0 * beta_L / denom, 0.5, 1.5)))


def evaluate_with_risk(
    test: pd.DataFrame, yt_pred: np.ndarray, *,
    use_conv_gate: bool = True,
    use_dd_brake: bool = False,
    use_hi_vol_gate: bool = False,
    use_lo_vol_gate: bool = False,
    use_btc_move_gate: bool = False,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    cols = ["open_time", "symbol", "return_pct", "alpha_realized",
            "basket_fwd", "beta_short_vs_bk", "bk_vol_24h", "btc_abs_ret_24h"]
    df = test[cols].copy()
    df["pred"] = yt_pred
    times = sorted(df["open_time"].unique())
    keep_times = set(times[::HORIZON])
    df = df[df["open_time"].isin(keep_times)]

    bars = []
    prev_long_w: dict[str, float] = {}
    prev_short_w: dict[str, float] = {}
    conv_history: deque = deque(maxlen=GATE_LOOKBACK)
    vol_history: deque = deque(maxlen=GATE_LOOKBACK)
    btc_history: deque = deque(maxlen=GATE_LOOKBACK)
    pnl_history: deque = deque(maxlen=DD_LOOKBACK)

    for t, g in df.groupby("open_time"):
        if len(g) < 2 * top_k + 1:
            continue
        sorted_g = g.sort_values("pred")
        bot = sorted_g.head(top_k)
        top = sorted_g.tail(top_k)
        dispersion = top["pred"].mean() - bot["pred"].mean()
        bk_vol_now = g["bk_vol_24h"].iloc[0]
        btc_move_now = g["btc_abs_ret_24h"].iloc[0]

        skip = False
        # Conv gate
        if use_conv_gate and len(conv_history) >= 30:
            thr = np.quantile(list(conv_history), GATE_PCTILE_CONV)
            if dispersion < thr:
                skip = True
        conv_history.append(dispersion)

        # Vol gates (PIT trailing percentile of basket vol)
        if not pd.isna(bk_vol_now):
            if use_hi_vol_gate and len(vol_history) >= 30:
                thr = np.quantile(list(vol_history), VOL_PCTILE_HI)
                if bk_vol_now > thr:
                    skip = True
            if use_lo_vol_gate and len(vol_history) >= 30:
                thr = np.quantile(list(vol_history), VOL_PCTILE_LO)
                if bk_vol_now < thr:
                    skip = True
            vol_history.append(bk_vol_now)

        # BTC-move gate (PIT trailing percentile)
        if not pd.isna(btc_move_now):
            if use_btc_move_gate and len(btc_history) >= 30:
                thr = np.quantile(list(btc_history), BTC_MOVE_PCTILE)
                if btc_move_now > thr:
                    skip = True
            btc_history.append(btc_move_now)

        # Drawdown brake (rolling sum of recent cycle nets)
        if use_dd_brake and len(pnl_history) >= 5:
            recent = sum(pnl_history)
            if recent < DD_THRESHOLD_BPS:
                skip = True

        if skip:
            bars.append({"time": t, "spread_ret_bps": 0.0, "long_turnover": 0.0,
                         "short_turnover": 0.0, "cost_bps": 0.0, "net_bps": 0.0,
                         "skipped": 1, "bk_vol": bk_vol_now,
                         "btc_move": btc_move_now, "dispersion": dispersion})
            pnl_history.append(0.0)
            continue

        scale_L, scale_S = _bn_scales(top, bot)
        n_l = len(top); n_s = len(bot)
        long_w = {s: scale_L / n_l for s in top["symbol"]}
        short_w = {s: scale_S / n_s for s in bot["symbol"]}
        long_ret = scale_L * top["return_pct"].mean()
        short_ret = scale_S * bot["return_pct"].mean()
        spread_ret = long_ret - short_ret

        if not prev_long_w:
            long_to, short_to = scale_L, scale_S
        else:
            all_l = set(long_w) | set(prev_long_w)
            long_to = sum(abs(long_w.get(s, 0) - prev_long_w.get(s, 0)) for s in all_l)
            all_s = set(short_w) | set(prev_short_w)
            short_to = sum(abs(short_w.get(s, 0) - prev_short_w.get(s, 0)) for s in all_s)
        bar_cost_bps = COST_PER_LEG * (long_to + short_to)
        net_bps = (spread_ret * 1e4) - bar_cost_bps

        bars.append({"time": t, "spread_ret_bps": spread_ret * 1e4,
                     "long_turnover": long_to, "short_turnover": short_to,
                     "cost_bps": bar_cost_bps, "net_bps": net_bps, "skipped": 0,
                     "bk_vol": bk_vol_now, "btc_move": btc_move_now,
                     "dispersion": dispersion})
        prev_long_w, prev_short_w = long_w, short_w
        pnl_history.append(net_bps)

    return pd.DataFrame(bars)


def main():
    panel = build_panel_with_market_state()
    folds = _multi_oos_splits(panel)
    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    print(f"Multi-OOS folds: {len(folds)}")

    variants = [
        ("sharp",                False, False, False, False, False),
        ("conv_gate",             True, False, False, False, False),  # validated
        ("conv_gate_dd",          True,  True, False, False, False),
        ("conv_gate_hi_vol",      True, False,  True, False, False),
        ("conv_gate_lo_vol",      True, False, False,  True, False),
        ("conv_gate_btc_move",    True, False, False, False,  True),
        ("conv_gate_dd_hivol",    True,  True,  True, False, False),
        ("all_overlays",          True,  True,  True, False,  True),
    ]
    cycles: dict[str, list] = {v[0]: [] for v in variants}

    for fold in folds:
        t0 = time.time()
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200:
            continue
        avail = [c for c in v6_clean if c in panel.columns]
        Xt = tr[avail].to_numpy(dtype=np.float32)
        yt_ = tr["demeaned_target"].to_numpy(dtype=np.float32)
        Xc = ca[avail].to_numpy(dtype=np.float32)
        yc_ = ca["demeaned_target"].to_numpy(dtype=np.float32)
        models = [_train(Xt, yt_, Xc, yc_, seed=seed) for seed in ENSEMBLE_SEEDS]
        Xtest = test[avail].to_numpy(dtype=np.float32)
        yt_pred = np.mean([m.predict(Xtest, num_iteration=m.best_iteration)
                            for m in models], axis=0)

        for name, conv, dd, hv, lv, btc in variants:
            df = evaluate_with_risk(
                test, yt_pred,
                use_conv_gate=conv, use_dd_brake=dd,
                use_hi_vol_gate=hv, use_lo_vol_gate=lv, use_btc_move_gate=btc,
            )
            for _, r in df.iterrows():
                cycles[name].append({
                    "fold": fold["fid"], "time": r["time"],
                    "gross": r["spread_ret_bps"], "cost": r["cost_bps"],
                    "net": r["net_bps"], "long_turn": r["long_turnover"],
                    "skipped": r["skipped"],
                })
        print(f"  fold {fold['fid']}: {time.time() - t0:.0f}s")

    print("\n" + "=" * 130)
    print(f"RISK OVERLAY SWEEP (h={HORIZON} K={TOP_K} ORIG25, β-neutral, "
          f"{COST_PER_LEG} bps/leg, post-fix cost)")
    print("=" * 130)
    print(f"  {'variant':<26} {'n_cyc':>5} {'%trade':>7} {'gross':>7} {'cost':>6} "
          f"{'net':>7} {'L_turn':>7} {'Sharpe':>7} {'95% CI':>15} "
          f"{'Δsharp_Sh':>10} {'Δgate_Sh':>10}")

    base_arr = np.array([r["net"] for r in cycles["sharp"]])
    base_sh = sharpe_est(base_arr)
    gate_recs = pd.DataFrame(cycles["conv_gate"])

    summary = {}
    for name, *_ in variants:
        df = pd.DataFrame(cycles[name])
        if df.empty: continue
        traded = df[df["skipped"] == 0]
        pct_trade = 100 * len(traded) / len(df)
        sh, lo, hi = block_bootstrap_ci(df["net"].values, statistic=sharpe_est,
                                          block_size=7, n_boot=2000)

        m_sh = pd.DataFrame(cycles["sharp"])[["fold", "time", "net"]].rename(
            columns={"net": "base"}).merge(
            df[["fold", "time", "net"]], on=["fold", "time"], how="inner")
        d_sharp = sharpe_est((m_sh["net"] - m_sh["base"]).to_numpy())
        m_g = gate_recs[["fold", "time", "net"]].rename(columns={"net": "base_g"}).merge(
            df[["fold", "time", "net"]], on=["fold", "time"], how="inner")
        d_gate = sharpe_est((m_g["net"] - m_g["base_g"]).to_numpy())

        print(f"  {name:<26} {len(df):>5d} {pct_trade:>6.1f}% "
              f"{traded['gross'].mean() if len(traded) > 0 else 0:>+6.2f}  "
              f"{traded['cost'].mean() if len(traded) > 0 else 0:>5.2f}  "
              f"{df['net'].mean():>+6.2f}  "
              f"{traded['long_turn'].mean() if len(traded) > 0 else 0:>6.0%}  "
              f"{sh:>+6.2f}  [{lo:>+5.2f},{hi:>+5.2f}]  "
              f"{d_sharp:>+9.2f}  {d_gate:>+9.2f}")
        summary[name] = {
            "n_cycles": int(len(df)), "pct_trade": float(pct_trade),
            "net_overall": float(df["net"].mean()),
            "sharpe": float(sh), "ci": [float(lo), float(hi)],
            "delta_sharpe_vs_sharp": float(d_sharp),
            "delta_sharpe_vs_convgate": float(d_gate),
        }

    with open(OUT_DIR / "alpha_v9_risk_overlay_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    for name, *_ in variants:
        pd.DataFrame(cycles[name]).to_csv(OUT_DIR / f"{name}_cycles.csv", index=False)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
