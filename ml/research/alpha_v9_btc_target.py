"""BTC-residual target: retrain on alpha-vs-BTC, test BTC-hedge variants.

Framing: instead of predicting alpha-vs-basket and trading top-K vs bot-K,
predict alpha-vs-BTC and trade top-K alts vs BTC. This makes BTC the
*natural* counterparty (matches what the model is predicting), eliminating
the (basket − BTC) drift noise that broke the prior BTC-hedge test.

Three configs:
  A. baseline_basket_xs — current production (basket target, cross-sectional)
  B. btc_target_xs       — BTC target, cross-sectional book (alts only)
                           [isolates the target-change effect]
  C. btc_target_long_btc — BTC target, long top-K alts + short BTC
                           [the proposal: BTC hedge with right target]
  D. btc_target_short_btc — BTC target, short bot-K alts + long BTC
                            [symmetric: short-side proposal]

Cost model: 4.5 bps/leg uniform (research convention; understates BTC's
real liquidity advantage).
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

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
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
OUT_DIR = REPO / "outputs/btc_target"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def add_btc_residual_target(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `btc_fwd` and `alpha_vs_btc_target` columns to panel.

    btc_fwd[t] = BTC's forward return at time t (h-bar window)
    alpha_vs_btc_realized[t,sym] = return_pct[t,sym] - btc_fwd[t]
    alpha_vs_btc_target[t,sym] = z-scored alpha_vs_btc_realized
                                  (same standardization as `demeaned_target`)
    """
    panel = panel.copy()
    btc_panel = panel[panel["symbol"] == BTC_SYMBOL][["open_time", "return_pct"]]
    btc_panel = btc_panel.rename(columns={"return_pct": "btc_fwd"})
    panel = panel.merge(btc_panel, on="open_time", how="left")
    panel["alpha_vs_btc_realized"] = panel["return_pct"] - panel["btc_fwd"]
    # Z-score target: per-symbol expanding mean + 7-day rolling std (matches
    # demeaned_target convention from features_ml.cross_sectional.make_xs_alpha_labels).
    # Using groupby + cumulative stats here for simplicity.
    panel = panel.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    g = panel.groupby("symbol")["alpha_vs_btc_realized"]
    rmean = g.transform(lambda s: s.expanding(min_periods=288).mean().shift(48))
    rstd = g.transform(lambda s: s.rolling(288*7, min_periods=288).std().shift(48))
    panel["alpha_vs_btc_target"] = (panel["alpha_vs_btc_realized"] - rmean) / rstd.replace(0, np.nan)
    return panel


def evaluate_btc_hedge_one_sided(
    test: pd.DataFrame, yt: np.ndarray, *,
    side: str,  # "long" (long top-K alts, short BTC) | "short" (short bot-K alts, long BTC)
    use_conv_gate: bool = True, use_pm_gate: bool = True,
    top_k: int = TOP_K, cost_bps_per_leg: float = COST_PER_LEG,
    sample_every: int = HORIZON,
) -> pd.DataFrame:
    """One-sided BTC-hedge evaluator. Trade only the active side; hedge with BTC.
    β-neutral via BTC scaling (BTC β=1, so btc_scale = alts_scale × mean(β_alts)).
    """
    assert side in ("long", "short")
    cols = ["open_time", "symbol", "return_pct", "alpha_realized",
            "basket_fwd", "beta_short_vs_bk"]
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
    prev_alt_w = {}
    prev_btc_w = 0.0
    prev_alt_scale, prev_btc_scale = 1.0, 1.0

    bars = []
    for t, g in df.groupby("open_time"):
        if len(g) < 2 * top_k + 1: continue
        btc_row = g[g["symbol"] == BTC_SYMBOL]
        if btc_row.empty: continue
        sym_arr = g["symbol"].to_numpy()
        pred_arr = g["pred"].to_numpy()

        # Conv gate (using all-symbol predictions; same as evaluate_stacked)
        idx_top = np.argpartition(-pred_arr, top_k - 1)[:top_k]
        idx_bot = np.argpartition(pred_arr, top_k - 1)[:top_k]
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
            # Hold-through: compute MtM on prior alts vs prior BTC
            if cur_alts:
                alt_g_h = g[g["symbol"].isin(cur_alts)]
                if not alt_g_h.empty:
                    n_h = len(alt_g_h)
                    gross_alt_h = prev_alt_scale * n_h / top_k
                    alt_ret_h = gross_alt_h * alt_g_h["return_pct"].mean()
                    btc_ret_h = prev_btc_scale * float(btc_row["return_pct"].iloc[0])
                    if side == "long":
                        spread_ret_h = alt_ret_h - btc_ret_h
                    else:
                        spread_ret_h = btc_ret_h - alt_ret_h
                    bars.append({
                        "time": t, "spread_ret_bps": spread_ret_h * 1e4,
                        "long_turnover": 0.0, "short_turnover": 0.0,
                        "cost_bps": 0.0, "net_bps": spread_ret_h * 1e4,
                        "n_alts": n_h, "skipped": 1,
                    })
                    continue
            bars.append({"time": t, "spread_ret_bps": 0.0,
                         "long_turnover": 0.0, "short_turnover": 0.0,
                         "cost_bps": 0.0, "net_bps": 0.0,
                         "n_alts": 0, "skipped": 1})
            continue

        # Identify alt candidates
        if side == "long":
            cand_alts = set(sym_arr[idx_top])
        else:
            cand_alts = set(sym_arr[idx_bot])
        cand_alts.discard(BTC_SYMBOL)  # don't trade BTC as an alt name

        # PM gate filter on alt names
        if use_pm_gate:
            new_alts = cur_alts & cand_alts
            if len(history) >= PM_M:
                if side == "long":
                    past_band = [h["long"] for h in history[-PM_M:][:PM_M-1]]
                else:
                    past_band = [h["short"] for h in history[-PM_M:][:PM_M-1]]
                for s in cand_alts - cur_alts:
                    if all(s in p for p in past_band): new_alts.add(s)
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
            # PM-empty: hold prior
            bars.append({"time": t, "spread_ret_bps": 0.0,
                         "long_turnover": 0.0, "short_turnover": 0.0,
                         "cost_bps": 0.0, "net_bps": 0.0,
                         "n_alts": 0, "skipped": 0})
            continue

        alt_g = g[g["symbol"].isin(new_alts)]
        if alt_g.empty: continue

        # β-neutral sizing: BTC notional matches alt-leg dollar β.
        # (β_BTC = 1 by feature def for BTC; β_alt is the average of held alts.)
        beta_alts = float(alt_g["beta_short_vs_bk"].mean())
        beta_btc = float(btc_row["beta_short_vs_bk"].iloc[0])
        if beta_alts < 0.1 or beta_btc < 0.1:
            alt_scale, btc_scale = 1.0, 1.0
        else:
            alt_scale = 1.0
            btc_scale = float(np.clip(beta_alts / beta_btc, 0.5, 1.5))

        # Per-alt weight
        alt_w = {s: alt_scale / top_k for s in new_alts}
        btc_w = btc_scale  # signed by side: + for long (asset held), see below

        # Returns
        alt_avg_ret = alt_g["return_pct"].mean()
        btc_ret = float(btc_row["return_pct"].iloc[0])
        gross_alt = alt_scale * len(new_alts) / top_k
        if side == "long":
            spread_ret = (gross_alt * alt_avg_ret) - (btc_scale * btc_ret)
            new_alt_w = alt_w
            new_btc_w = -btc_scale  # short BTC
        else:
            spread_ret = (btc_scale * btc_ret) - (gross_alt * alt_avg_ret)
            new_alt_w = {s: -w for s, w in alt_w.items()}  # negative for short alts
            new_btc_w = +btc_scale  # long BTC

        # Turnover (alt leg + BTC)
        if not prev_alt_w and prev_btc_w == 0:
            alt_to = sum(abs(v) for v in new_alt_w.values())
            btc_to = abs(new_btc_w)
        else:
            all_alts = set(new_alt_w) | set(prev_alt_w)
            alt_to = sum(abs(new_alt_w.get(s, 0) - prev_alt_w.get(s, 0)) for s in all_alts)
            btc_to = abs(new_btc_w - prev_btc_w)
        cost_bps = cost_bps_per_leg * (alt_to + btc_to)

        bars.append({
            "time": t, "spread_ret_bps": spread_ret * 1e4,
            "long_turnover": alt_to, "short_turnover": btc_to,
            "cost_bps": cost_bps, "net_bps": spread_ret * 1e4 - cost_bps,
            "n_alts": len(new_alts), "skipped": 0,
        })

        cur_alts = set(new_alts)
        prev_alt_w = new_alt_w
        prev_btc_w = new_btc_w
        prev_alt_scale, prev_btc_scale = alt_scale, btc_scale

    return pd.DataFrame(bars)


def _sharpe(x):
    x = np.asarray(x)
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR)) if x.std() > 0 else 0.0


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def main():
    print("Building panel + adding BTC residual target...")
    panel = build_wide_panel()
    panel = add_btc_residual_target(panel)
    # IMPORTANT: don't drop NaN target rows here. BTC's alpha-vs-BTC is 0 with
    # std=0 → NaN target; dropping BTC rows breaks both (a) the BTC-hedge
    # evaluator (can't find BTC) and (b) cross-sectional config A (ranks
    # without BTC = different universe). Instead, keep all rows, mask NaN
    # only during the BTC-target training step.
    print(f"  Panel rows: {len(panel):,} ({panel['alpha_vs_btc_target'].notna().sum():,} non-NaN btc target)")

    folds = _multi_oos_splits(panel)
    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    avail_feats = [c for c in v6_clean if c in panel.columns]

    cycles = {
        "A_basket_xs":     [],   # current production
        "B_btc_target_xs": [],   # btc target, cross-sectional
        "C_btc_long_btc":  [],   # btc target, long top-K alts + short BTC
        "D_btc_short_btc": [],   # btc target, short bot-K alts + long BTC
    }

    for fold in folds:
        t0 = time.time()
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200: continue

        Xt = tr[avail_feats].to_numpy(np.float32)
        Xc = ca[avail_feats].to_numpy(np.float32)
        Xtest = test[avail_feats].to_numpy(np.float32)

        # === Train basket-target model (config A) ===
        yt_basket = tr["demeaned_target"].to_numpy(np.float32)
        yc_basket = ca["demeaned_target"].to_numpy(np.float32)
        models_basket = [_train(Xt, yt_basket, Xc, yc_basket, seed=s) for s in ENSEMBLE_SEEDS]
        pred_basket = np.mean([m.predict(Xtest, num_iteration=m.best_iteration) for m in models_basket], axis=0)

        # === Train btc-target model (configs B/C/D) ===
        yt_btc = tr["alpha_vs_btc_target"].to_numpy(np.float32)
        yc_btc = ca["alpha_vs_btc_target"].to_numpy(np.float32)
        # Drop NaN rows (early bars where rolling std hasn't built up)
        mask_t = ~np.isnan(yt_btc); mask_c = ~np.isnan(yc_btc)
        if mask_t.sum() < 1000 or mask_c.sum() < 200:
            print(f"  fold {fold['fid']}: btc-target has insufficient non-NaN ({mask_t.sum()} train)"); continue
        models_btc = [_train(Xt[mask_t], yt_btc[mask_t], Xc[mask_c], yc_btc[mask_c], seed=s)
                      for s in ENSEMBLE_SEEDS]
        pred_btc = np.mean([m.predict(Xtest, num_iteration=m.best_iteration) for m in models_btc], axis=0)

        # === Config A: basket target + cross-sectional (current production) ===
        df_A = evaluate_stacked(test, pred_basket, use_conv_gate=True, use_pm_gate=True)
        for _, r in df_A.iterrows():
            cycles["A_basket_xs"].append({
                "fold": fold["fid"], "time": r["time"], "net": r["net_bps"],
                "cost": r["cost_bps"], "skipped": r["skipped"],
            })

        # === Config B: btc target + cross-sectional ===
        df_B = evaluate_stacked(test, pred_btc, use_conv_gate=True, use_pm_gate=True)
        for _, r in df_B.iterrows():
            cycles["B_btc_target_xs"].append({
                "fold": fold["fid"], "time": r["time"], "net": r["net_bps"],
                "cost": r["cost_bps"], "skipped": r["skipped"],
            })

        # === Config C: btc target + BTC hedge (long alts) ===
        df_C = evaluate_btc_hedge_one_sided(test, pred_btc, side="long",
                                              use_conv_gate=True, use_pm_gate=True)
        for _, r in df_C.iterrows():
            cycles["C_btc_long_btc"].append({
                "fold": fold["fid"], "time": r["time"], "net": r["net_bps"],
                "cost": r["cost_bps"], "skipped": r["skipped"],
            })

        # === Config D: btc target + BTC hedge (short alts) ===
        df_D = evaluate_btc_hedge_one_sided(test, pred_btc, side="short",
                                              use_conv_gate=True, use_pm_gate=True)
        for _, r in df_D.iterrows():
            cycles["D_btc_short_btc"].append({
                "fold": fold["fid"], "time": r["time"], "net": r["net_bps"],
                "cost": r["cost_bps"], "skipped": r["skipped"],
            })

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
    print("BTC-RESIDUAL TARGET TEST  (multi-OOS, conv+PM, 4.5 bps/leg)")
    print("=" * 110)
    print(f"  {'config':<22}  {'n':>4}  {'mean_net':>9}  {'mean_cost':>10}  "
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
        print(f"  {label:<22}  {len(net):>4}  {net.mean():>+9.2f}  {cost_avg:>+10.2f}  "
              f"{sh:>+7.2f}  {ci_lo:>+7.2f}  {ci_hi:>+7.2f}  {max_dd:>+8.0f}")

    pd.DataFrame(rows).to_csv(OUT_DIR / "summary.csv", index=False)
    for label, c in cycles.items():
        if c: pd.DataFrame(c).to_csv(OUT_DIR / f"{label}_cycles.csv", index=False)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
