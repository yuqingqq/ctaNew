"""3-way architecture comparison: universal vs per-symbol vs hybrid.

All three use:
  - Target: target_beta_btc (z-scored β-residual against BTC)
  - 51-panel data
  - Same LGBM hyperparameters, same SEEDS, same folds

Architectures:
  1. Universal model — single LGBM trained on all 51 symbols pooled, features = WINNER_BTC (25)
     (already trained — predictions at outputs/vBTC_audit_panel_btc_only/all_predictions.parquet)
  2. Per-symbol model — 51 separate LGBMs, each on one symbol's data, features = WINNER_BTC_PERSYM (22)
     (drops listing_age_days, log_quote_volume_90d, residual_vol_90d_own_pctile which are
      near-constants within a single symbol)
  3. Hybrid — 51 separate LGBMs with universal_pred added as a feature (23 features)
     (universal_pred is the universal model's prediction, included PIT)

For each variant: per-cycle IC + V3.1 β-hedged Sharpe + end-equity on $100.
"""
from __future__ import annotations
import sys, time, importlib.util, warnings
from pathlib import Path
from collections import deque, defaultdict
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice, _train
spec = importlib.util.spec_from_file_location("psl", REPO / "scripts/phase_ah_sleeve.py")
psl = importlib.util.module_from_spec(spec); spec.loader.exec_module(psl)
from ml.research.alpha_v4_xs import block_bootstrap_ci

PANEL_BTC = REPO / "outputs/vBTC_features_btc_only/panel_btc_only_clean.parquet"
APD_UNI = REPO / "outputs/vBTC_audit_panel_btc_only/all_predictions.parquet"
OUT_DIR = REPO / "outputs/vBTC_3way_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = (42, 1337, 7, 19, 2718)
ALL_FOLDS = list(range(10))
OOS_FOLDS = list(range(1, 10))
HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
CAPITAL = 100.0

WINNER_BTC = [
    "idio_ret_to_btc_12b", "idio_ret_to_btc_48b", "idio_ret_to_btc_288b",
    "dom_btc_z_1d", "dom_btc_change_48b", "dom_btc_change_288b",
    "beta_to_btc", "beta_to_btc_change_5d", "corr_to_btc_1d", "corr_to_btc_change_3d",
    "idio_vol_to_btc_1h", "idio_vol_to_btc_1d", "idio_vol_ratio_to_btc",
    "btc_ret_48b", "btc_realized_vol_1d", "btc_realized_vol_30d",
    "atr_pct", "obv_z_1d", "vwap_slope_96",
    "funding_rate", "funding_rate_z_7d", "funding_rate_1d_change",
    "listing_age_days", "log_quote_volume_90d", "residual_vol_90d_own_pctile",
]
PER_SYM_DROPS = {"listing_age_days", "log_quote_volume_90d", "residual_vol_90d_own_pctile"}
WINNER_BTC_PERSYM = [f for f in WINNER_BTC if f not in PER_SYM_DROPS]


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(psl.CYCLES_PER_YEAR))


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def folds_positive(df_v):
    return sum(1 for _, g in df_v.groupby("fold") if _sharpe(g["net_pnl_bps"]) > 0)


def train_per_symbol_bank(panel, folds_all, feat_set, label, extra_feat_col=None):
    """Train one LGBM per symbol with given features. Return concatenated predictions."""
    print(f"\n--- Training {label} ({len(feat_set)} features) ---", flush=True)
    symbols = sorted(panel["symbol"].unique())
    n_sym = len(symbols)
    all_preds = []
    t_start = time.time()
    for i, sym in enumerate(symbols):
        sym_t0 = time.time()
        sym_preds = []
        for fid in ALL_FOLDS:
            if fid >= len(folds_all): continue
            train, cal, test = _slice(panel, folds_all[fid])
            tr = train[(train["symbol"] == sym) & (train["autocorr_pctile_7d"] >= THRESHOLD)]
            ca = cal[(cal["symbol"] == sym) & (cal["autocorr_pctile_7d"] >= THRESHOLD)]
            test_r = test[test["symbol"] == sym].copy()
            if len(tr) < 500 or len(ca) < 100 or len(test_r) < 50: continue
            cols_use = feat_set + ([extra_feat_col] if extra_feat_col else [])
            Xt = tr[cols_use].to_numpy(np.float32)
            Xc = ca[cols_use].to_numpy(np.float32)
            Xtest = test_r[cols_use].to_numpy(np.float32)
            yt = tr["target_beta_btc"].to_numpy(np.float32)
            yc = ca["target_beta_btc"].to_numpy(np.float32)
            mt = ~np.isnan(yt) & ~np.any(np.isnan(Xt), axis=1)
            mc = ~np.isnan(yc) & ~np.any(np.isnan(Xc), axis=1)
            if mt.sum() < 500 or mc.sum() < 100: continue
            preds = []
            for s in SEEDS:
                m = _train(Xt[mt], yt[mt], Xc[mc], yc[mc], seed=s)
                preds.append(m.predict(Xtest, num_iteration=m.best_iteration))
            avg = np.mean(preds, axis=0)
            sym_preds.append((test_r[["symbol", "open_time", "alpha_beta", "return_pct"]].copy(),
                              avg, fid))
        if sym_preds:
            for test_df, avg, fid in sym_preds:
                test_df["pred"] = avg
                test_df["fold"] = fid
                all_preds.append(test_df)
        if (i+1) % 10 == 0 or i == n_sym-1:
            elapsed = time.time() - t_start
            eta = elapsed/(i+1) * (n_sym - i - 1)
            print(f"  {i+1}/{n_sym} symbols done ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)",
                  flush=True)
    print(f"  total: {time.time()-t_start:.0f}s", flush=True)
    apd = pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])
    return apd


def per_cycle_ic(apd):
    apd_o = apd.dropna(subset=["alpha_A" if "alpha_A" in apd.columns else "alpha_beta", "pred"])
    if "alpha_A" in apd_o.columns:
        target_col = "alpha_A"
    else:
        target_col = "alpha_beta"
    cyc_ic = apd_o.groupby("open_time").apply(
        lambda g: g["pred"].rank().corr(g[target_col].rank()) if len(g) >= 5 else np.nan
    ).dropna()
    return float(cyc_ic.mean()), len(cyc_ic)


def aggregate_alpha(records, alpha_wide):
    sleeve_queue = deque(maxlen=psl.N_SLEEVES)
    prev_weights = {}
    rows = []
    for _, rec in records.iterrows():
        t = rec["time"]; fold = rec["fold"]
        if rec["traded"]:
            sleeve_queue.append({"entry_time": t, "longs": list(rec["long_basket"]),
                                  "shorts": list(rec["short_basket"])})
        else:
            sleeve_queue.append({"entry_time": t, "longs": [], "shorts": []})
        target_weights = defaultdict(float)
        sleeve_weight = 1.0 / psl.N_SLEEVES
        for sleeve in sleeve_queue:
            n_long = len(sleeve["longs"]); n_short = len(sleeve["shorts"])
            if n_long == 0 or n_short == 0: continue
            for s in sleeve["longs"]:
                target_weights[s] += sleeve_weight * (1.0 / n_long)
            for s in sleeve["shorts"]:
                target_weights[s] -= sleeve_weight * (1.0 / n_short)
        gross = 0.0
        if t in alpha_wide.index:
            alphas = alpha_wide.loc[t]
            for sym, w in prev_weights.items():
                if sym in alphas.index and not pd.isna(alphas[sym]):
                    gross += w * alphas[sym] * 1e4
        all_syms = set(target_weights.keys()) | set(prev_weights.keys())
        abs_delta = sum(abs(target_weights.get(s, 0.0) - prev_weights.get(s, 0.0)) for s in all_syms)
        cost = abs_delta * psl.COST_PER_UNIT_ABS_DELTA
        rows.append({"time": t, "fold": fold,
                      "gross_pnl_bps": gross, "cost_bps": cost,
                      "net_pnl_bps": gross - cost, "turnover": abs_delta})
        prev_weights = dict(target_weights)
    return pd.DataFrame(rows)


def run_v31_beta_hedged(apd, panel_syms, listings):
    apd2 = apd.copy()
    apd2["open_time"] = pd.to_datetime(apd2["open_time"], utc=True)
    if "exit_time" not in apd2.columns:
        apd2["exit_time"] = apd2["open_time"] + pd.Timedelta(minutes=HORIZON * 5)
    apd2["exit_time"] = pd.to_datetime(apd2["exit_time"], utc=True)
    if "alpha_A" not in apd2.columns and "alpha_beta" in apd2.columns:
        apd2 = apd2.rename(columns={"alpha_beta": "alpha_A"})
    def elig_pit(b):
        if isinstance(b, pd.Timestamp): ts = b
        else: ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=psl.MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}
    target_t = sorted(apd2[apd2["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::psl.HORIZON_ENTRY]
    universe = psl.build_rolling_ic_universe(apd2, sampled_t, psl.TOP_N, elig_pit)
    records = psl.run_production_protocol_save_sleeves(apd2, universe)
    alpha_wide = apd2.pivot_table(index="open_time", columns="symbol",
                                    values="alpha_A", aggfunc="first").sort_index()
    df_v = aggregate_alpha(records, alpha_wide)
    return df_v, records


def main():
    print("=== 3-way comparison: Universal vs Per-symbol vs Hybrid (all β-residual) ===\n",
          flush=True)
    t_total = time.time()

    panel = pd.read_parquet(PANEL_BTC)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel_syms = sorted(panel["symbol"].unique())
    print(f"Panel: {len(panel):,} rows × {len(panel_syms)} symbols", flush=True)
    folds_all = _multi_oos_splits(panel)
    listings = psl.get_listings()
    for s, t in panel.groupby("symbol")["open_time"].min().items():
        if s not in listings:
            t = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[s] = t

    print(f"\nFeature sets:")
    print(f"  WINNER_BTC (universal):       {len(WINNER_BTC)} features", flush=True)
    print(f"  WINNER_BTC_PERSYM (per-sym):  {len(WINNER_BTC_PERSYM)} features", flush=True)
    print(f"  Dropped for per-sym (constants within symbol): {sorted(PER_SYM_DROPS)}", flush=True)

    # === Variant 1: Universal — load existing predictions ===
    print("\n" + "="*80)
    print("  VARIANT 1 — Universal model (WINNER_BTC, single LGBM, all 51 pooled)")
    print("="*80)
    apd_uni = pd.read_parquet(APD_UNI)
    apd_uni["open_time"] = pd.to_datetime(apd_uni["open_time"], utc=True)
    apd_uni["exit_time"] = pd.to_datetime(apd_uni["exit_time"], utc=True)
    ic_uni, n_uni = per_cycle_ic(apd_uni)
    print(f"  per-cycle IC: {ic_uni:+.4f} (n_cycles={n_uni})", flush=True)

    # === Variant 2: Per-symbol model bank ===
    print("\n" + "="*80)
    print("  VARIANT 2 — Per-symbol model bank (51 LGBMs, WINNER_BTC_PERSYM)")
    print("="*80)
    apd_persym = train_per_symbol_bank(panel, folds_all, WINNER_BTC_PERSYM,
                                          "Per-symbol bank")
    apd_persym["exit_time"] = apd_persym["open_time"] + pd.Timedelta(minutes=HORIZON * 5)
    apd_persym = apd_persym.rename(columns={"alpha_beta": "alpha_A"})
    apd_persym.to_parquet(OUT_DIR / "persym_predictions.parquet", index=False)
    ic_persym, n_persym = per_cycle_ic(apd_persym)
    print(f"\n  per-cycle IC: {ic_persym:+.4f} (n_cycles={n_persym})", flush=True)

    # === Variant 3: Hybrid (per-symbol + universal_pred as feature) ===
    print("\n" + "="*80)
    print("  VARIANT 3 — Hybrid (51 per-symbol LGBMs + universal_pred as feature)")
    print("="*80)
    # Merge universal_pred into panel on (symbol, open_time)
    print("  Merging universal preds into panel as feature 'universal_pred'...", flush=True)
    apd_uni_for_merge = apd_uni[["symbol","open_time","pred","fold"]].rename(
        columns={"pred": "universal_pred"})
    panel_aug = panel.merge(apd_uni_for_merge, on=["symbol","open_time","fold"], how="left")
    print(f"  panel_aug: {len(panel_aug):,} rows; universal_pred valid: "
          f"{panel_aug['universal_pred'].notna().sum():,}/{len(panel_aug):,}", flush=True)

    apd_hybrid = train_per_symbol_bank(panel_aug, folds_all, WINNER_BTC_PERSYM,
                                          "Hybrid bank", extra_feat_col="universal_pred")
    apd_hybrid["exit_time"] = apd_hybrid["open_time"] + pd.Timedelta(minutes=HORIZON * 5)
    apd_hybrid = apd_hybrid.rename(columns={"alpha_beta": "alpha_A"})
    apd_hybrid.to_parquet(OUT_DIR / "hybrid_predictions.parquet", index=False)
    ic_hybrid, n_hybrid = per_cycle_ic(apd_hybrid)
    print(f"\n  per-cycle IC: {ic_hybrid:+.4f} (n_cycles={n_hybrid})", flush=True)

    # === Run V3.1 β-hedged for each ===
    print("\n" + "="*80)
    print("  V3.1 β-hedged execution for each variant")
    print("="*80)
    panel_syms_set = set(panel_syms)
    results = {}
    for label, apd in [("Universal", apd_uni), ("Per-symbol", apd_persym), ("Hybrid", apd_hybrid)]:
        print(f"\n  Running V3.1 for {label}...", flush=True)
        df_v, records = run_v31_beta_hedged(apd, panel_syms_set, listings)
        net = df_v["net_pnl_bps"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=1000)
        total_d = net.sum() / 1e4 * CAPITAL
        end_eq = CAPITAL + total_d
        results[label] = {
            "sharpe": sh, "totPnL_d": total_d, "end_eq": end_eq,
            "maxDD_bps": _max_dd(net),
            "gross_avg": df_v["gross_pnl_bps"].mean(),
            "cost_avg": df_v["cost_bps"].mean(),
            "turnover_avg": df_v["turnover"].mean(),
            "folds_pos": folds_positive(df_v),
            "n_traded": int(records["traded"].sum()),
            "n_cycles": len(records),
        }
        df_v.to_csv(OUT_DIR / f"{label}_v31.csv", index=False)

    # === Summary ===
    print("\n" + "="*100)
    print("  3-WAY COMPARISON SUMMARY  ($100 capital, β-hedged, β-residual target, 51-panel)")
    print("="*100)
    print(f"  {'variant':<14} {'Sharpe':>10} {'end-eq':>12} {'pnl%':>8} "
          f"{'per_cyc_IC':>14} {'gross':>10} {'cost':>8} {'turnover':>10} {'folds+':>7}",
          flush=True)
    ics = {"Universal": ic_uni, "Per-symbol": ic_persym, "Hybrid": ic_hybrid}
    for label in ["Universal", "Per-symbol", "Hybrid"]:
        r = results[label]
        pct = r['totPnL_d'] / CAPITAL * 100
        print(f"  {label:<14} {r['sharpe']:+10.2f} ${r['end_eq']:>10.2f} {pct:+7.1f}% "
              f"{ics[label]:>+13.4f} {r['gross_avg']:>+9.2f} {r['cost_avg']:>+7.2f} "
              f"{r['turnover_avg']:>10.3f} {r['folds_pos']:>4}/9", flush=True)

    print(f"\nTotal runtime: {time.time()-t_total:.0f}s", flush=True)


if __name__ == "__main__":
    main()
