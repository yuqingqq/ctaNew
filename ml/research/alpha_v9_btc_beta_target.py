"""Proper β-adjusted BTC-residual target.

Constructs target the same way as v6_clean's basket version, but with BTC
as reference asset:

    beta_to_btc[t,s] = rolling cov(my_ret, btc_ret) / var(btc_ret), shifted 1
    alpha_vs_btc[t,s] = my_fwd - beta_to_btc * btc_fwd       # β-orthogonal
    target = (alpha - rmean) / rstd                            # z-scored

Trades the model with β-weighted BTC hedge:
  - Long top-K alts at scale_L/K each
  - Short BTC at scale_L × mean(beta_to_btc[top-K]) / K        # dollar-β hedged
  - Captured: mean(alpha_vs_btc[top-K]) — pure alpha, BTC-direction-neutral

Configs:
  A. baseline_basket — current production (basket target, cross-sectional)
  B. btc_beta_xs    — β-adjusted BTC target, cross-sectional book (long top-K, short bot-K alts)
  C. btc_beta_long_btc — β-adjusted BTC target, long top-K alts + β×BTC short
  D. btc_beta_short_btc — symmetric short side: short bot-K alts + β×BTC long
"""
from __future__ import annotations
import sys, time, warnings
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import (
    XS_FEATURE_COLS_V6_CLEAN, list_universe, build_kline_features,
)
from ml.research.alpha_v4_xs_1d import (
    ENSEMBLE_SEEDS, _multi_oos_splits, _slice, _train,
)
from ml.research.alpha_v4_xs import block_bootstrap_ci
from ml.research.alpha_v8_h48_audit import build_wide_panel
from ml.research.alpha_v9_pred_momentum_stack import evaluate_stacked

HORIZON = 48
TOP_K = 7
COST_PER_LEG = 4.5
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
GATE_LOOKBACK = 252
GATE_PCTILE = 0.30
PM_M = 2
PM_BAND = 1.0
BTC_SYMBOL = "BTCUSDT"
BETA_WINDOW = 288   # 1d, matches features_ml.cross_sectional.BETA_WINDOW
OUT_DIR = REPO / "outputs/btc_beta_target"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def add_btc_beta_target(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute beta_to_btc + alpha_vs_btc + z-scored target.

    Mirrors features_ml.cross_sectional.add_basket_features β computation
    (rolling 288-bar cov/var, shift(1) for point-in-time).
    """
    print("  Computing beta_to_btc per symbol from raw kline data...")
    syms = sorted(panel["symbol"].unique())
    btc_feats = build_kline_features(BTC_SYMBOL)
    if btc_feats.empty:
        raise RuntimeError("BTCUSDT features missing; cannot compute beta-to-btc")
    btc_close = btc_feats["close"].copy()
    btc_close.index = pd.to_datetime(btc_close.index, utc=True)
    btc_ret = btc_close.pct_change()
    btc_fwd_series = btc_close.pct_change(HORIZON).shift(-HORIZON)

    # Per-symbol: compute rolling β to BTC + align to panel's open_time
    rows = []
    for s in syms:
        f = build_kline_features(s)
        if f.empty: continue
        my_close = f["close"].copy()
        my_close.index = pd.to_datetime(my_close.index, utc=True)
        my_ret = my_close.pct_change()
        # Align both ret series on shared timestamps
        df = pd.DataFrame({"my_ret": my_ret, "btc_ret": btc_ret, "btc_fwd": btc_fwd_series})
        df = df.dropna(subset=["my_ret", "btc_ret"])
        if len(df) < BETA_WINDOW + 10: continue
        cov = (df["my_ret"] * df["btc_ret"]).rolling(BETA_WINDOW).mean() - \
              df["my_ret"].rolling(BETA_WINDOW).mean() * df["btc_ret"].rolling(BETA_WINDOW).mean()
        var = df["btc_ret"].rolling(BETA_WINDOW).var().replace(0, np.nan)
        beta = (cov / var).clip(-5, 5)
        beta_pit = beta.shift(1)   # point-in-time
        # Stack with symbol + timestamp
        sub = pd.DataFrame({
            "open_time": df.index,
            "symbol": s,
            "beta_to_btc": beta_pit.values,
            "btc_fwd": df["btc_fwd"].values,
        })
        rows.append(sub)
    btc_betas = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    print(f"    {len(btc_betas):,} (sym, time) β rows")

    # Merge into panel by (symbol, open_time)
    panel = panel.copy()
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    btc_betas["open_time"] = pd.to_datetime(btc_betas["open_time"], utc=True)
    panel = panel.merge(btc_betas, on=["symbol", "open_time"], how="left")

    # β-adjusted residual: alpha = return - beta × btc_fwd
    panel["alpha_vs_btc_realized"] = panel["return_pct"] - panel["beta_to_btc"] * panel["btc_fwd"]

    # Z-score per-symbol with same convention as basket target (expanding mean shifted by horizon, 7d rolling std)
    panel = panel.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    g = panel.groupby("symbol")["alpha_vs_btc_realized"]
    rmean = g.transform(lambda s: s.expanding(min_periods=288).mean().shift(HORIZON))
    rstd = g.transform(lambda s: s.rolling(288 * 7, min_periods=288).std().shift(HORIZON))
    panel["btc_beta_target"] = (panel["alpha_vs_btc_realized"] - rmean) / rstd.replace(0, np.nan)
    return panel


def evaluate_btc_beta_hedge(
    test: pd.DataFrame, yt: np.ndarray, *,
    side: str,   # "long" or "short"
    use_conv_gate: bool = True, use_pm_gate: bool = True,
    top_k: int = TOP_K, cost_bps_per_leg: float = COST_PER_LEG,
    sample_every: int = HORIZON,
) -> pd.DataFrame:
    """β-weighted BTC hedge evaluator.

    side="long":  long top-K alts, short BTC at mean(beta_to_btc) × notional
    side="short": short bot-K alts, long BTC at mean(beta_to_btc) × notional
    """
    assert side in ("long", "short")
    cols = ["open_time", "symbol", "return_pct", "alpha_realized",
            "basket_fwd", "beta_short_vs_bk", "beta_to_btc", "btc_fwd"]
    df = test[cols].copy()
    df["pred"] = yt
    times = sorted(df["open_time"].unique())
    if not times: return pd.DataFrame()
    if sample_every > 1:
        keep_times = set(times[::sample_every])
        df = df[df["open_time"].isin(keep_times)]

    band_k = max(top_k, int(round(PM_BAND * top_k)))
    history = []
    dispersion_history = deque(maxlen=GATE_LOOKBACK)
    cur_alts = set()
    prev_alt_w, prev_btc_w = {}, 0.0
    prev_alt_scale, prev_beta_hedge = 1.0, 1.0

    bars = []
    for t, g in df.groupby("open_time"):
        if len(g) < 2 * top_k + 1: continue
        # Drop rows without beta_to_btc / btc_fwd (edge of data)
        g = g.dropna(subset=["beta_to_btc", "btc_fwd"])
        if g.empty or len(g) < 2 * top_k + 1: continue
        btc_row = g[g["symbol"] == BTC_SYMBOL]
        # If BTC isn't in this bar (rare), skip
        if btc_row.empty: continue

        sym_arr = g["symbol"].to_numpy()
        pred_arr = g["pred"].to_numpy()
        idx_top = np.argpartition(-pred_arr, top_k - 1)[:top_k]
        idx_bot = np.argpartition(pred_arr, top_k - 1)[:top_k]

        # Conv gate
        dispersion = float(pred_arr[idx_top].mean() - pred_arr[idx_bot].mean())
        skip = False
        if use_conv_gate and len(dispersion_history) >= 30:
            thr = float(np.quantile(list(dispersion_history), GATE_PCTILE))
            if dispersion < thr: skip = True
        dispersion_history.append(dispersion)

        # PM history
        bk = min(band_k, len(g))
        idx_top_band = np.argpartition(-pred_arr, bk - 1)[:bk] if bk < len(g) else np.arange(len(g))
        idx_bot_band = np.argpartition(pred_arr, bk - 1)[:bk] if bk < len(g) else np.arange(len(g))
        history.append({"long": set(sym_arr[idx_top_band]), "short": set(sym_arr[idx_bot_band])})
        if len(history) > PM_M: history = history[-PM_M:]

        if skip:
            # Hold-through MtM
            if cur_alts:
                alt_g_h = g[g["symbol"].isin(cur_alts)]
                if not alt_g_h.empty:
                    n_h = len(alt_g_h)
                    gross_alt_h = prev_alt_scale * n_h / top_k
                    alt_ret_h = gross_alt_h * alt_g_h["return_pct"].mean()
                    btc_ret_h = prev_beta_hedge * float(btc_row["return_pct"].iloc[0])
                    if side == "long":
                        spread_ret_h = alt_ret_h - btc_ret_h
                    else:
                        spread_ret_h = btc_ret_h - alt_ret_h
                    bars.append({"time": t, "spread_ret_bps": spread_ret_h * 1e4,
                                 "long_turnover": 0.0, "short_turnover": 0.0,
                                 "cost_bps": 0.0, "net_bps": spread_ret_h * 1e4,
                                 "n_alts": n_h, "skipped": 1, "beta_hedge": prev_beta_hedge})
                    continue
            bars.append({"time": t, "spread_ret_bps": 0.0,
                         "long_turnover": 0.0, "short_turnover": 0.0,
                         "cost_bps": 0.0, "net_bps": 0.0,
                         "n_alts": 0, "skipped": 1, "beta_hedge": 0.0})
            continue

        # Active side candidates
        if side == "long":
            cand_alts = set(sym_arr[idx_top])
        else:
            cand_alts = set(sym_arr[idx_bot])
        cand_alts.discard(BTC_SYMBOL)

        # PM gate
        if use_pm_gate:
            new_alts = cur_alts & cand_alts
            if len(history) >= PM_M:
                past = [h["long" if side == "long" else "short"] for h in history[-PM_M:][:PM_M-1]]
                for s in cand_alts - cur_alts:
                    if all(s in p for p in past): new_alts.add(s)
            else:
                new_alts |= cand_alts
            if len(new_alts) > top_k:
                if side == "long":
                    ranked = sorted(new_alts, key=lambda s: -pred_arr[sym_arr == s][0])[:top_k]
                else:
                    ranked = sorted(new_alts, key=lambda s: pred_arr[sym_arr == s][0])[:top_k]
                new_alts = set(ranked)
        else:
            new_alts = cand_alts

        if not new_alts:
            bars.append({"time": t, "spread_ret_bps": 0.0,
                         "long_turnover": 0.0, "short_turnover": 0.0,
                         "cost_bps": 0.0, "net_bps": 0.0,
                         "n_alts": 0, "skipped": 0, "beta_hedge": 0.0})
            continue

        alt_g = g[g["symbol"].isin(new_alts)]
        if alt_g.empty: continue

        # β-weighted BTC hedge: BTC notional = scale_L × mean(beta_to_btc[alt_leg])
        scale_alt = 1.0
        beta_hedge = float(alt_g["beta_to_btc"].mean())
        if not np.isfinite(beta_hedge) or beta_hedge < 0.1 or beta_hedge > 3.0:
            beta_hedge = 1.0   # safe fallback if β is missing/extreme

        # Per-name weights (signed by side)
        gross_alt = scale_alt * len(new_alts) / top_k
        alt_avg_ret = alt_g["return_pct"].mean()
        btc_ret = float(btc_row["return_pct"].iloc[0])

        if side == "long":
            spread_ret = (gross_alt * alt_avg_ret) - (beta_hedge * btc_ret)
            new_alt_w = {s: scale_alt / top_k for s in new_alts}
            new_btc_w = -beta_hedge   # short BTC
        else:
            spread_ret = (beta_hedge * btc_ret) - (gross_alt * alt_avg_ret)
            new_alt_w = {s: -scale_alt / top_k for s in new_alts}
            new_btc_w = +beta_hedge   # long BTC

        # Turnover (alt + BTC)
        if not prev_alt_w and prev_btc_w == 0:
            alt_to = sum(abs(v) for v in new_alt_w.values())
            btc_to = abs(new_btc_w)
        else:
            all_alts = set(new_alt_w) | set(prev_alt_w)
            alt_to = sum(abs(new_alt_w.get(s, 0) - prev_alt_w.get(s, 0)) for s in all_alts)
            btc_to = abs(new_btc_w - prev_btc_w)
        cost_bps = cost_bps_per_leg * (alt_to + btc_to)

        bars.append({"time": t, "spread_ret_bps": spread_ret * 1e4,
                     "long_turnover": alt_to, "short_turnover": btc_to,
                     "cost_bps": cost_bps, "net_bps": spread_ret * 1e4 - cost_bps,
                     "n_alts": len(new_alts), "skipped": 0, "beta_hedge": beta_hedge})

        cur_alts = set(new_alts)
        prev_alt_w, prev_btc_w = new_alt_w, new_btc_w
        prev_alt_scale, prev_beta_hedge = scale_alt, beta_hedge

    return pd.DataFrame(bars)


def _sharpe(x):
    x = np.asarray(x)
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR)) if x.std() > 0 else 0.0


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def main():
    print("Building basket panel...")
    panel = build_wide_panel()
    print(f"  panel: {len(panel):,} rows")

    print("\nAdding β-adjusted BTC target...")
    panel = add_btc_beta_target(panel)
    n_with_btc_target = panel["btc_beta_target"].notna().sum()
    n_with_beta = panel["beta_to_btc"].notna().sum()
    print(f"  rows with non-NaN btc_beta_target: {n_with_btc_target:,}")
    print(f"  rows with non-NaN beta_to_btc:     {n_with_beta:,}")

    folds = _multi_oos_splits(panel)
    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    avail_feats = [c for c in v6_clean if c in panel.columns]
    print(f"\nFolds: {len(folds)}, features: {len(avail_feats)}")

    cycles = {"A_basket_xs": [], "B_btc_beta_xs": [],
              "C_btc_beta_long_btc": [], "D_btc_beta_short_btc": []}

    for fold in folds:
        t0 = time.time()
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200: continue

        Xt = tr[avail_feats].to_numpy(np.float32)
        Xc = ca[avail_feats].to_numpy(np.float32)
        Xtest = test[avail_feats].to_numpy(np.float32)

        # === A. basket-target (production reference) ===
        yt_basket = tr["demeaned_target"].to_numpy(np.float32)
        yc_basket = ca["demeaned_target"].to_numpy(np.float32)
        models_basket = [_train(Xt, yt_basket, Xc, yc_basket, seed=s) for s in ENSEMBLE_SEEDS]
        pred_basket = np.mean([m.predict(Xtest, num_iteration=m.best_iteration) for m in models_basket], axis=0)

        # === B/C/D. β-adjusted BTC-target ===
        yt_btc_raw = tr["btc_beta_target"].to_numpy(np.float32)
        yc_btc_raw = ca["btc_beta_target"].to_numpy(np.float32)
        mask_t = ~np.isnan(yt_btc_raw); mask_c = ~np.isnan(yc_btc_raw)
        if mask_t.sum() < 1000 or mask_c.sum() < 200:
            print(f"  fold {fold['fid']}: BTC-target too sparse"); continue
        models_btc = [_train(Xt[mask_t], yt_btc_raw[mask_t], Xc[mask_c], yc_btc_raw[mask_c], seed=s)
                      for s in ENSEMBLE_SEEDS]
        pred_btc = np.mean([m.predict(Xtest, num_iteration=m.best_iteration) for m in models_btc], axis=0)

        # A: basket production cross-sectional
        df_A = evaluate_stacked(test, pred_basket, use_conv_gate=True, use_pm_gate=True)
        for _, r in df_A.iterrows():
            cycles["A_basket_xs"].append({"fold": fold["fid"], "time": r["time"],
                                            "net": r["net_bps"], "cost": r["cost_bps"],
                                            "skipped": r["skipped"]})

        # B: BTC target cross-sectional (control for target only)
        df_B = evaluate_stacked(test, pred_btc, use_conv_gate=True, use_pm_gate=True)
        for _, r in df_B.iterrows():
            cycles["B_btc_beta_xs"].append({"fold": fold["fid"], "time": r["time"],
                                             "net": r["net_bps"], "cost": r["cost_bps"],
                                             "skipped": r["skipped"]})

        # C: BTC target + β-weighted BTC hedge (long top alts)
        df_C = evaluate_btc_beta_hedge(test, pred_btc, side="long",
                                         use_conv_gate=True, use_pm_gate=True)
        for _, r in df_C.iterrows():
            cycles["C_btc_beta_long_btc"].append({"fold": fold["fid"], "time": r["time"],
                                                    "net": r["net_bps"], "cost": r["cost_bps"],
                                                    "skipped": r["skipped"]})

        # D: BTC target + β-weighted BTC hedge (short bot alts)
        df_D = evaluate_btc_beta_hedge(test, pred_btc, side="short",
                                         use_conv_gate=True, use_pm_gate=True)
        for _, r in df_D.iterrows():
            cycles["D_btc_beta_short_btc"].append({"fold": fold["fid"], "time": r["time"],
                                                     "net": r["net_bps"], "cost": r["cost_bps"],
                                                     "skipped": r["skipped"]})

        line = f"  fold {fold['fid']:>2}: "
        for label in cycles:
            df_t = pd.DataFrame(cycles[label])
            if df_t.empty or "fold" not in df_t.columns:
                line += f"{label[:14]}=n/a "
                continue
            df_t = df_t[df_t["fold"] == fold["fid"]]
            if df_t.empty:
                line += f"{label[:14]}=n/a "
                continue
            n = df_t["net"].to_numpy()
            line += f"{label[:14]}={n.mean():+.2f}({_sharpe(n):+.1f}) "
        print(line + f"({time.time()-t0:.0f}s)")

    print("\n" + "=" * 110)
    print("β-ADJUSTED BTC-TARGET TEST  (proper β-orthogonal residual; multi-OOS, conv+PM)")
    print("=" * 110)
    print(f"  {'config':<24}  {'n':>4}  {'mean_net':>9}  {'mean_cost':>10}  "
          f"{'Sharpe':>7}  {'CI_lo':>7}  {'CI_hi':>7}  {'max_DD':>8}")
    rows = []
    for label in cycles:
        df_v = pd.DataFrame(cycles[label])
        if df_v.empty: continue
        net = df_v["net"].to_numpy()
        sh, ci_lo, ci_hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
        max_dd = _max_dd(net)
        cost_avg = df_v["cost"].mean()
        rows.append({"config": label, "n": len(net), "mean_net": net.mean(),
                     "mean_cost": cost_avg, "sharpe": sh, "ci_lo": ci_lo, "ci_hi": ci_hi,
                     "max_dd": max_dd})
        print(f"  {label:<24}  {len(net):>4}  {net.mean():>+9.2f}  {cost_avg:>+10.2f}  "
              f"{sh:>+7.2f}  {ci_lo:>+7.2f}  {ci_hi:>+7.2f}  {max_dd:>+8.0f}")

    pd.DataFrame(rows).to_csv(OUT_DIR / "summary.csv", index=False)
    for label, c in cycles.items():
        if c: pd.DataFrame(c).to_csv(OUT_DIR / f"{label}_cycles.csv", index=False)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
