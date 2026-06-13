"""Phase S prep: correlation diagnostic of 8 candidate selector scores on 51-panel.

For each rolling-IC refresh boundary, compute per-symbol:
  S1: raw_IC  (Spearman rank correlation of past pred vs alpha_A)
  S2: shrunk_IC = IC - λ * SE  (using λ=20 per L.2 calibration)
  S3: IC_tstat = IC / SE
  S4: past_directional_alpha_sharpe (mean(sign(pred)*alpha) / std)
  S5: past_hit_rate_after_cost (fraction of past obs where sign(pred)*alpha > 9bps)
  S6: hybrid = z(shrunk_IC) + z(past_directional_sharpe)
  S7: stable_hybrid = hybrid - z(rank_churn) - z(pred_scale_instability)
  S8: age_penalized = stable_hybrid - penalty(symbol_age_days < 180)

Then compute pairwise score correlations across all boundaries × symbols.
Output: score correlation matrix + per-boundary rank-overlap-with-raw-IC.
"""
from __future__ import annotations
import sys, warnings, time
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

OUT = REPO / "outputs/vBTC_selector_diag"
OUT.mkdir(parents=True, exist_ok=True)
APD_PATH = REPO / "outputs/vBTC_audit_panel/all_predictions.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"

HORIZON = 48
OOS_FOLDS = list(range(1, 10))
MIN_HISTORY_DAYS = 60
TOP_N = 15
IC_WINDOW_DAYS = 180
IC_UPDATE_DAYS = 90
MIN_OBS_PER_SYM = 100
SHRINK_LAMBDA = 20.0
COST_PER_LEG_BPS = 4.5
COST_HIT_THRESHOLD_BPS = 9.0  # round-trip cost
NEW_SYMBOL_AGE_DAYS = 180


def to_ms_int(s):
    ts = pd.to_datetime(s)
    if hasattr(ts.dtype, "tz") and ts.dtype.tz is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return ts.astype("datetime64[ms]").astype("int64").to_numpy()


def get_listings():
    L = {}
    for d in KLINES_DIR.iterdir():
        if not d.is_dir(): continue
        m5 = d / "5m"
        if not m5.exists(): continue
        f = sorted(m5.glob("*.parquet"))
        if not f: continue
        try: L[d.name] = pd.Timestamp(f[0].stem, tz="UTC")
        except Exception: pass
    return L


def compute_scores_at_boundary(past_df, boundary_ts, listings):
    """For each symbol with sufficient past obs, compute 8 scores."""
    rows = []
    for sym, gg in past_df.groupby("symbol"):
        if len(gg) < MIN_OBS_PER_SYM: continue
        gg = gg.dropna(subset=["pred", "alpha_A", "return_pct"])
        if len(gg) < MIN_OBS_PER_SYM: continue
        # IC and SE
        ic = gg["pred"].rank().corr(gg["alpha_A"].rank())
        if pd.isna(ic): continue
        n = len(gg)
        se = np.sqrt((1 - ic ** 2) / max(n - 2, 1))
        shrunk_ic = ic - SHRINK_LAMBDA * se
        ic_tstat = ic / max(se, 1e-6)

        # Directional signed alpha (proxy for past trade quality)
        pred_z = gg["pred"] - gg["pred"].mean()
        signed_alpha = np.sign(pred_z) * gg["alpha_A"]
        past_dir_sharpe = float(signed_alpha.mean() / max(signed_alpha.std(), 1e-6))

        # Hit rate after cost: fraction of past obs where directional return > 9 bps
        signed_ret_bps = np.sign(pred_z) * gg["return_pct"] * 1e4
        hit_rate_after_cost = float((signed_ret_bps > COST_HIT_THRESHOLD_BPS).mean())

        # Rank churn: std of within-cycle pred rank
        # Approximate via std of pred / mean abs pred
        pred_scale = float(gg["pred"].abs().mean())
        pred_vol = float(gg["pred"].std())

        # Rank churn via Spearman rank stability across consecutive cycles
        # Simplest proxy: std of pred over time
        rank_churn = pred_vol / max(pred_scale, 1e-6)

        # Symbol age in days at boundary
        listing_date = listings.get(sym)
        if listing_date:
            age_days = (boundary_ts - listing_date).days
        else:
            age_days = (boundary_ts - gg["open_time"].min()).days
        is_new = age_days < NEW_SYMBOL_AGE_DAYS

        rows.append({
            "boundary": boundary_ts, "symbol": sym, "n_obs": n,
            "s1_raw_IC": ic, "ic_se": se,
            "s2_shrunk_IC": shrunk_ic, "s3_IC_tstat": ic_tstat,
            "s4_past_dir_sharpe": past_dir_sharpe,
            "s5_hit_rate_after_cost": hit_rate_after_cost,
            "pred_scale": pred_scale, "rank_churn": rank_churn,
            "symbol_age_days": age_days, "is_new_sym": int(is_new),
        })
    df = pd.DataFrame(rows)
    if len(df) == 0: return df
    # z-score within boundary
    for col in ["s2_shrunk_IC", "s4_past_dir_sharpe", "rank_churn", "pred_scale"]:
        mu, sigma = df[col].mean(), df[col].std()
        df[f"z_{col}"] = (df[col] - mu) / max(sigma, 1e-6)
    df["s6_hybrid"] = df["z_s2_shrunk_IC"] + df["z_s4_past_dir_sharpe"]
    df["s7_stable_hybrid"] = df["s6_hybrid"] - df["z_rank_churn"]
    df["s8_age_penalized"] = df["s7_stable_hybrid"] - 1.0 * df["is_new_sym"]
    return df


def main():
    print("=== Selector score correlation diagnostic ===\n", flush=True)
    apd = pd.read_parquet(APD_PATH)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    apd["t_int"] = to_ms_int(apd["open_time"])
    apd["exit_t_int"] = to_ms_int(apd["exit_time"])
    apd_clean = apd.dropna(subset=["alpha_A"])
    print(f"  apd: {len(apd):,} rows, {apd.symbol.nunique()} syms", flush=True)

    listings = get_listings()
    panel_syms = set(apd["symbol"].unique())
    target_t = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::HORIZON]

    bar_ms = 5 * 60 * 1000
    window_ms = IC_WINDOW_DAYS * 288 * bar_ms
    update_ms = IC_UPDATE_DAYS * 288 * bar_ms
    t0_ms = int(pd.Timestamp(sampled_t[0]).timestamp() * 1000)
    boundaries = []
    seen_b = set()
    for t in sampled_t:
        t_ms = int(pd.Timestamp(t).timestamp() * 1000)
        n = (t_ms - t0_ms) // update_ms
        b_ms = t0_ms + n * update_ms
        if b_ms not in seen_b:
            seen_b.add(b_ms)
            boundaries.append(b_ms)
    print(f"  {len(boundaries)} unique refresh boundaries", flush=True)

    all_scores = []
    for b_ms in boundaries:
        ts = pd.Timestamp(b_ms, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        eligible = {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}
        past = apd_clean[(apd_clean["t_int"] >= b_ms - window_ms) &
                          (apd_clean["t_int"] < b_ms) &
                          (apd_clean["exit_t_int"] <= b_ms) &
                          (apd_clean["symbol"].isin(eligible))]
        if len(past) < 1000: continue
        df_s = compute_scores_at_boundary(past, ts, listings)
        if len(df_s) > 0:
            all_scores.append(df_s)
    full = pd.concat(all_scores, ignore_index=True)
    full.to_csv(OUT / "scores_per_boundary.csv", index=False)
    print(f"  total (boundary, symbol) rows: {len(full):,}", flush=True)
    print(f"  unique boundaries: {full['boundary'].nunique()}", flush=True)
    print(f"  symbols seen at any boundary: {full['symbol'].nunique()}", flush=True)

    score_cols = ["s1_raw_IC", "s2_shrunk_IC", "s3_IC_tstat",
                  "s4_past_dir_sharpe", "s5_hit_rate_after_cost",
                  "s6_hybrid", "s7_stable_hybrid", "s8_age_penalized"]

    print(f"\n=== Cross-score Spearman rank correlations ===\n", flush=True)
    corr = full[score_cols].corr(method="spearman")
    print(corr.round(2).to_string(), flush=True)
    corr.to_csv(OUT / "score_correlations.csv")

    print(f"\n=== Per-boundary top-15 overlap with raw_IC ===\n", flush=True)
    print(f"  {'score':<28}  {'mean Jaccard':>14}  {'min':>6}  {'max':>6}", flush=True)
    overlap_rows = []
    for sc in score_cols:
        if sc == "s1_raw_IC": continue
        jaccards = []
        for b, g in full.groupby("boundary"):
            top_raw = set(g.nlargest(TOP_N, "s1_raw_IC")["symbol"])
            top_sc = set(g.nlargest(TOP_N, sc)["symbol"])
            if len(top_raw | top_sc) > 0:
                j = len(top_raw & top_sc) / len(top_raw | top_sc)
                jaccards.append(j)
        ja = np.array(jaccards)
        overlap_rows.append({"score": sc, "mean_jaccard": ja.mean(),
                                "min": ja.min(), "max": ja.max()})
        print(f"  {sc:<28}  {ja.mean():>14.3f}  {ja.min():>6.3f}  {ja.max():>6.3f}",
              flush=True)
    pd.DataFrame(overlap_rows).to_csv(OUT / "topN_overlap.csv", index=False)

    # Print distribution of new-symbol exposure per top-15 by each score
    print(f"\n=== Per-boundary new-symbol count in top-15 ===\n", flush=True)
    print(f"  {'score':<28}  {'mean new in top15':>18}  {'max':>6}", flush=True)
    for sc in score_cols:
        new_counts = []
        for b, g in full.groupby("boundary"):
            top = g.nlargest(TOP_N, sc)
            new_counts.append(top["is_new_sym"].sum())
        nc = np.array(new_counts)
        print(f"  {sc:<28}  {nc.mean():>18.2f}  {nc.max():>6.0f}", flush=True)

    print(f"\n  saved → {OUT}", flush=True)


if __name__ == "__main__":
    main()
