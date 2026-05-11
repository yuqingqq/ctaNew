"""ICP / loser-symbols analysis: what's special about names the model fails on?

Compares feature/realized characteristics of consistent loser vs winner symbols.

Loser cohort: ICP, ORDI, HBAR, TAO, AAVE, TIA, ENA, RUNE
Winner cohort: VVV, WIF, AVAX, WLD, AXS, LTC, NEAR, LINK

Cross-axes:
  1. Realized return distribution per side (long vs short picks)
  2. Per-symbol IC (pred-vs-realized rank corr)
  3. Volatility regime (atr_pct)
  4. BTC β stability (beta_to_btc_change_5d)
  5. Funding rate / funding state
  6. Volume / liquidity proxies
  7. Feature value distribution (top-loaded features in WINNER_21)

Output:
  - Per-symbol summary of feature distributions
  - Side decomposition: was the model bad at long picks, short picks, or both?
  - Pattern detection: do losers cluster by feature characteristics?
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice, _train

PANEL_PATH = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
ATTRIBUTION_PATH = REPO / "outputs/vBTC_universe_filter/per_symbol_attribution.csv"
OUT_DIR = REPO / "outputs/vBTC_loser_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
SEEDS = (42, 1337, 7, 19, 2718)
MIN_OBS_PER_SYM = 100
TARGET_N = 15
K = 4
ALL_FOLDS = list(range(10))
OOS_FOLDS = list(range(1, 10))

V6_CLEAN_28 = list(XS_FEATURE_COLS_V6_CLEAN)
ALL_DROPS = [
    "return_1d_xs_rank", "bk_ret_48b", "volume_ma_50",
    "ema_slope_20_1h", "ema_slope_20_1h_xs_rank",
    "vwap_zscore_xs_rank", "vwap_zscore",
    "atr_pct_xs_rank", "dom_z_7d_vs_bk", "obv_z_1d_xs_rank",
    "obv_signal", "price_volume_corr_10",
    "hour_cos", "hour_sin",
]
FUNDING_LEAN = ["funding_rate", "funding_rate_z_7d"]
ADD_CROSS_BTC = ["corr_to_btc_1d", "idio_vol_to_btc_1h", "beta_to_btc_change_5d"]
ADD_MORE_FUNDING = ["funding_rate_1d_change", "funding_streak_pos"]
WINNER_21 = [f for f in V6_CLEAN_28 if f not in ALL_DROPS] + FUNDING_LEAN + ADD_CROSS_BTC + ADD_MORE_FUNDING

LOSERS = ["ICPUSDT", "ORDIUSDT", "HBARUSDT", "TAOUSDT", "AAVEUSDT", "TIAUSDT", "ENAUSDT", "RUNEUSDT"]
WINNERS = ["VVVUSDT", "WIFUSDT", "AVAXUSDT", "WLDUSDT", "AXSUSDT", "LTCUSDT", "NEARUSDT", "LINKUSDT"]


def _sharpe(x):
    x = np.asarray(x)
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR)) if x.std() > 0 else 0.0


def train_fold(panel, fold, feat_set):
    train, cal, test = _slice(panel, fold)
    tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
    ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
    if len(tr) < 1000 or len(ca) < 200: return None, None
    Xt = tr[feat_set].to_numpy(np.float32)
    Xc = ca[feat_set].to_numpy(np.float32)
    Xtest = test[feat_set].to_numpy(np.float32)
    yt = tr["target_A"].to_numpy(np.float32)
    yc = ca["target_A"].to_numpy(np.float32)
    mask_t = ~np.isnan(yt); mask_c = ~np.isnan(yc)
    if mask_t.sum() < 1000 or mask_c.sum() < 200: return None, None
    preds = []
    for s in SEEDS:
        m = _train(Xt[mask_t], yt[mask_t], Xc[mask_c], yc[mask_c], seed=s)
        preds.append(m.predict(Xtest, num_iteration=m.best_iteration))
    return test.copy(), np.mean(preds, axis=0)


def main():
    print(f"Loading panel...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    feat_set = [f for f in WINNER_21 if f in panel.columns]
    folds_all = _multi_oos_splits(panel)

    # Train + collect predictions per symbol
    print(f"\n=== Train all 10 folds ===", flush=True)
    all_preds = []
    for fid in ALL_FOLDS:
        if fid >= len(folds_all): continue
        t0 = time.time()
        td, p = train_fold(panel, folds_all[fid], feat_set)
        if td is None: continue
        df = td[["symbol", "open_time", "alpha_A", "return_pct"] + feat_set].copy()
        df["pred"] = p; df["fold"] = fid
        all_preds.append(df)
        print(f"  fold {fid}: ({time.time()-t0:.0f}s)", flush=True)
    apd = pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])
    oos_pred = apd[apd["fold"].isin(OOS_FOLDS)].copy()

    cohort_syms = LOSERS + WINNERS
    df_oos = oos_pred[oos_pred["symbol"].isin(cohort_syms)].copy()
    print(f"\n  OOS rows for cohort: {len(df_oos):,}", flush=True)

    # === Analysis 1: per-symbol IC + realized stats ===
    print(f"\n=== Per-symbol IC and realized stats (OOS only) ===", flush=True)
    print(f"  {'symbol':<12} {'cohort':<6} {'n':>5}  {'pred_mean':>9}  {'real_mean':>9}  "
          f"{'IC':>6}  {'realized_sharpe':>15}  {'std_real':>8}", flush=True)
    sym_stats = []
    for sym in cohort_syms:
        g = df_oos[df_oos["symbol"] == sym].dropna(subset=["alpha_A"])
        if len(g) < 50: continue
        cohort = "loser" if sym in LOSERS else "winner"
        ic = g["pred"].rank().corr(g["alpha_A"].rank())
        pred_mean = g["pred"].mean()
        real_mean = g["alpha_A"].mean()  # using alpha_A as residual realized
        ret_mean = g["return_pct"].mean()
        ret_std = g["return_pct"].std()
        ret_sharpe = ret_mean / ret_std * np.sqrt(CYCLES_PER_YEAR / HORIZON) if ret_std > 0 else 0
        # std of return in bps
        std_real_bps = g["return_pct"].std() * 1e4
        sym_stats.append({"symbol": sym, "cohort": cohort, "n": len(g),
                            "pred_mean": pred_mean, "real_mean": real_mean,
                            "IC": ic, "ret_sharpe": ret_sharpe, "std_real_bps": std_real_bps,
                            "ret_mean_bps": ret_mean * 1e4})
        print(f"  {sym:<12} {cohort:<6} {len(g):>5}  {pred_mean:>+9.4f}  {real_mean:>+9.4f}  "
              f"{ic:>+6.3f}  {ret_sharpe:>+15.2f}  {std_real_bps:>8.1f}", flush=True)

    df_sym = pd.DataFrame(sym_stats)

    # === Analysis 2: feature distribution comparison ===
    print(f"\n=== Feature value distribution: losers vs winners (means) ===", flush=True)
    print(f"  {'feature':<32}  {'loser_mean':>11}  {'winner_mean':>11}  {'ratio':>7}", flush=True)
    feature_diffs = []
    for f in feat_set:
        if f not in df_oos.columns: continue
        loser_vals = df_oos.loc[df_oos["symbol"].isin(LOSERS), f].dropna()
        winner_vals = df_oos.loc[df_oos["symbol"].isin(WINNERS), f].dropna()
        if len(loser_vals) == 0 or len(winner_vals) == 0: continue
        l_mean = loser_vals.mean()
        w_mean = winner_vals.mean()
        ratio = (l_mean / w_mean) if abs(w_mean) > 1e-9 else (l_mean - w_mean)
        feature_diffs.append({"feature": f, "loser_mean": l_mean, "winner_mean": w_mean,
                                "abs_diff": abs(l_mean - w_mean),
                                "loser_std": loser_vals.std(), "winner_std": winner_vals.std()})
    df_feat = pd.DataFrame(feature_diffs).sort_values("abs_diff", ascending=False)
    for _, r in df_feat.head(15).iterrows():
        ratio_str = f"{r['loser_mean'] / r['winner_mean']:>+7.2f}" if abs(r['winner_mean']) > 1e-9 else "    inf"
        print(f"  {r['feature']:<32}  {r['loser_mean']:>+11.4f}  {r['winner_mean']:>+11.4f}  {ratio_str}",
              flush=True)

    # === Analysis 3: prediction extremity ===
    print(f"\n=== Prediction-extremity comparison ===", flush=True)
    print(f"  cohort  abs_pred_mean  std_pred  pred_top5%  pred_bot5%", flush=True)
    for cohort_label, syms in [("loser", LOSERS), ("winner", WINNERS)]:
        g = df_oos[df_oos["symbol"].isin(syms)]
        if g.empty: continue
        abs_pred = g["pred"].abs().mean()
        std_pred = g["pred"].std()
        p95 = g["pred"].quantile(0.95)
        p5 = g["pred"].quantile(0.05)
        print(f"  {cohort_label:<7}  {abs_pred:>13.4f}  {std_pred:>8.4f}  "
              f"{p95:>+10.4f}  {p5:>+10.4f}", flush=True)

    # === Analysis 4: realized return when picked ===
    # Use top-K=4 universe to determine "picked" status approximately:
    # pick = pred is in top 4 or bottom 4 of all symbols at that time
    print(f"\n=== When picked: directional accuracy ===", flush=True)
    # Need to recompute pick status per cycle. For each open_time, mark top-4 / bot-4 of cohort+others
    print(f"  Computing pick-side and accuracy...", flush=True)
    df_picks_records = []
    for t, g in oos_pred.groupby("open_time"):
        if len(g) < 8: continue
        pred = g["pred"].to_numpy()
        idx_top = np.argpartition(-pred, K - 1)[:K]
        idx_bot = np.argpartition(pred, K - 1)[:K]
        for i in idx_top:
            df_picks_records.append({"time": t, "symbol": g.iloc[i]["symbol"],
                                       "side": "long", "pred": pred[i],
                                       "realized_pct": g.iloc[i]["return_pct"]})
        for i in idx_bot:
            df_picks_records.append({"time": t, "symbol": g.iloc[i]["symbol"],
                                       "side": "short", "pred": pred[i],
                                       "realized_pct": g.iloc[i]["return_pct"]})
    df_picks = pd.DataFrame(df_picks_records)
    print(f"  Total picks (across cycles): {len(df_picks)}", flush=True)
    print(f"  {'symbol':<12} {'side':<6} {'n':>5}  {'mean_real_bps':>13}  {'std_bps':>8}  "
          f"{'win_rate':>9}", flush=True)
    sym_side_stats = []
    for sym in cohort_syms:
        for side in ["long", "short"]:
            sub = df_picks[(df_picks["symbol"] == sym) & (df_picks["side"] == side)]
            if len(sub) < 5: continue
            real_bps = sub["realized_pct"].to_numpy() * 1e4
            mean = real_bps.mean()
            std = real_bps.std()
            # Win = ret > 0 for long, ret < 0 for short
            if side == "long":
                wins = (real_bps > 0).mean()
            else:
                wins = (real_bps < 0).mean()
            cohort = "L" if sym in LOSERS else "W"
            sym_side_stats.append({"symbol": sym, "cohort": cohort, "side": side,
                                     "n": len(sub), "mean_bps": mean, "std_bps": std,
                                     "win_rate": wins})
            print(f"  [{cohort}] {sym:<10} {side:<6} {len(sub):>5}  {mean:>+13.0f}  {std:>8.0f}  "
                  f"{wins:>8.1%}", flush=True)

    df_side = pd.DataFrame(sym_side_stats)

    # === Save outputs ===
    df_sym.to_csv(OUT_DIR / "per_symbol_oos_stats.csv", index=False)
    df_feat.to_csv(OUT_DIR / "feature_distribution_diff.csv", index=False)
    df_side.to_csv(OUT_DIR / "per_symbol_side_stats.csv", index=False)

    # === Synthesize ===
    print(f"\n{'=' * 90}", flush=True)
    print(f"SYNTHESIS", flush=True)
    print(f"{'=' * 90}", flush=True)
    losers_df = df_sym[df_sym["cohort"] == "loser"]
    winners_df = df_sym[df_sym["cohort"] == "winner"]
    print(f"  Loser cohort: mean IC = {losers_df['IC'].mean():+.3f}, "
          f"mean ret_sharpe = {losers_df['ret_sharpe'].mean():+.2f}, "
          f"mean std_real_bps = {losers_df['std_real_bps'].mean():.1f}", flush=True)
    print(f"  Winner cohort: mean IC = {winners_df['IC'].mean():+.3f}, "
          f"mean ret_sharpe = {winners_df['ret_sharpe'].mean():+.2f}, "
          f"mean std_real_bps = {winners_df['std_real_bps'].mean():.1f}", flush=True)

    # Side analysis
    losers_side = df_side[df_side["cohort"] == "L"]
    winners_side = df_side[df_side["cohort"] == "W"]
    if len(losers_side) > 0:
        long_wins_L = losers_side[losers_side["side"] == "long"]["win_rate"].mean()
        short_wins_L = losers_side[losers_side["side"] == "short"]["win_rate"].mean()
        print(f"\n  Loser cohort win-rate: long picks {long_wins_L:.1%}, short picks {short_wins_L:.1%}",
              flush=True)
    if len(winners_side) > 0:
        long_wins_W = winners_side[winners_side["side"] == "long"]["win_rate"].mean()
        short_wins_W = winners_side[winners_side["side"] == "short"]["win_rate"].mean()
        print(f"  Winner cohort win-rate: long picks {long_wins_W:.1%}, short picks {short_wins_W:.1%}",
              flush=True)

    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
