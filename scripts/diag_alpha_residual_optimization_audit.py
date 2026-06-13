"""Diagnostic audit of β-hedged α-residual strategy: where can we optimize?

Runs 6 diagnostics on Phase 1D predictions:
  1. Per-cycle IC overall vs at tails (top-3 / bot-3) — does model skill exist at extremes?
  2. Pred dispersion vs realized dispersion calibration — is conv_gate signal real or noise?
  3. 4h vs 24h α horizon — model predicts 4h but strategy holds 24h. How much decay?
  4. Per-symbol contribution — which names carry the alpha, which dilute?
  5. Per-fold realized spread — is one fold driving everything?
  6. Pick persistence — natural pick autocorrelation, informs whether PM_M2 is needed

Output: structured findings + concrete optimization candidates.
"""
from __future__ import annotations
import sys, time, importlib.util, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location("psl", REPO / "scripts/phase_ah_sleeve.py")
psl = importlib.util.module_from_spec(spec); spec.loader.exec_module(psl)

APD_PATH = REPO / "outputs/vBTC_phase1d_rolling_beta/all_predictions.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
OUT = REPO / "outputs/vBTC_alpha_residual_audit"
OUT.mkdir(parents=True, exist_ok=True)
OOS_FOLDS = list(range(1, 10))
K = 3
HOLD_BARS = 288  # 24h


def main():
    print("=== α-residual optimization audit ===\n", flush=True)
    apd = pd.read_parquet(APD_PATH)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    panel_syms = sorted(apd["symbol"].unique())
    apd_oos = apd[apd["fold"].isin(OOS_FOLDS)].copy()
    print(f"OOS rows: {len(apd_oos):,}, symbols: {len(panel_syms)}\n", flush=True)

    # Sample at entry cadence
    times = sorted(apd_oos["open_time"].unique())
    sampled_t = set(times[::psl.HORIZON_ENTRY])
    apd_entries = apd_oos[apd_oos["open_time"].isin(sampled_t)].copy()
    print(f"Entry cycles: {apd_entries['open_time'].nunique()}", flush=True)

    # === DIAG 1: per-cycle IC overall vs at tails ===
    print("\n" + "="*80)
    print("DIAG 1 — Per-cycle IC: overall vs at K=3 tails")
    print("="*80)
    overall_ic = []
    top3_hit_rate = []  # fraction of pred-top-3 that are also in realized-top-3
    bot3_hit_rate = []
    top3_in_top15 = []  # softer: fraction in realized top-15 (i.e. above-median direction)
    bot3_in_bot15 = []
    rho_pred_alpha_at_top3 = []  # rank corr of pred and alpha JUST among picked long-side names
    cyc_top3_realized = []  # realized α of the 3 model-picked longs
    cyc_top3_oracle = []   # realized α of the 3 oracle longs
    cyc_bot3_realized = []
    cyc_bot3_oracle = []
    for t, g in apd_entries.groupby("open_time"):
        g = g.dropna(subset=["pred", "alpha_A"])
        if len(g) < 2*K + 1: continue
        pred_rank = g["pred"].rank()
        alpha_rank = g["alpha_A"].rank()
        overall_ic.append(pred_rank.corr(alpha_rank))
        preds = g["pred"].to_numpy(); alphas = g["alpha_A"].to_numpy()
        n = len(g)
        idx_pred_top = np.argpartition(-preds, K-1)[:K]
        idx_pred_bot = np.argpartition(preds, K-1)[:K]
        idx_alpha_top = np.argpartition(-alphas, K-1)[:K]
        idx_alpha_bot = np.argpartition(alphas, K-1)[:K]
        idx_alpha_top15 = np.argpartition(-alphas, min(14, n-1))[:15]
        idx_alpha_bot15 = np.argpartition(alphas, min(14, n-1))[:15]
        top3_hit_rate.append(len(set(idx_pred_top) & set(idx_alpha_top)) / K)
        bot3_hit_rate.append(len(set(idx_pred_bot) & set(idx_alpha_bot)) / K)
        top3_in_top15.append(len(set(idx_pred_top) & set(idx_alpha_top15)) / K)
        bot3_in_bot15.append(len(set(idx_pred_bot) & set(idx_alpha_bot15)) / K)
        cyc_top3_realized.append(alphas[idx_pred_top].mean())
        cyc_top3_oracle.append(alphas[idx_alpha_top].mean())
        cyc_bot3_realized.append(alphas[idx_pred_bot].mean())
        cyc_bot3_oracle.append(alphas[idx_alpha_bot].mean())

    overall_ic = np.array(overall_ic)
    print(f"\nOverall per-cycle IC (all 51 symbols rank corr):")
    print(f"  mean={overall_ic.mean():+.4f}  median={np.median(overall_ic):+.4f}  "
          f"std={overall_ic.std():.4f}  p25={np.percentile(overall_ic,25):+.4f}  "
          f"p75={np.percentile(overall_ic,75):+.4f}")

    th = np.array(top3_hit_rate); bh = np.array(bot3_hit_rate)
    th15 = np.array(top3_in_top15); bh15 = np.array(bot3_in_bot15)
    print(f"\nK=3 pick hit rates (out of {K} picks):")
    print(f"  Top-3 pick exact match with realized top-3:  mean={th.mean():.1%}  (random = {K/51:.1%})")
    print(f"  Bot-3 pick exact match with realized bot-3:  mean={bh.mean():.1%}  (random = {K/51:.1%})")
    print(f"  Top-3 pick within realized top-15:           mean={th15.mean():.1%}  (random = {15/51:.1%})")
    print(f"  Bot-3 pick within realized bot-15:           mean={bh15.mean():.1%}  (random = {15/51:.1%})")

    top3r = np.array(cyc_top3_realized); top3o = np.array(cyc_top3_oracle)
    bot3r = np.array(cyc_bot3_realized); bot3o = np.array(cyc_bot3_oracle)
    print(f"\nMean realized α_β captured by picks (per cycle, bps):")
    print(f"  Top-3 model picks:    {top3r.mean()*1e4:+.1f} bps    (oracle: {top3o.mean()*1e4:+.1f} bps, capture = {top3r.mean()/top3o.mean()*100:.1f}%)")
    print(f"  Bot-3 model picks:    {bot3r.mean()*1e4:+.1f} bps    (oracle: {bot3o.mean()*1e4:+.1f} bps, capture = {bot3r.mean()/bot3o.mean()*100:.1f}%)")
    spread_real = (top3r - bot3r) * 1e4
    spread_oracle = (top3o - bot3o) * 1e4
    print(f"  Spread top3 − bot3:    realized {spread_real.mean():+.1f} bps,  oracle {spread_oracle.mean():+.1f} bps,  capture = {spread_real.mean()/spread_oracle.mean()*100:.1f}%")

    # === DIAG 2: pred dispersion vs realized dispersion calibration ===
    print("\n" + "="*80)
    print("DIAG 2 — Pred dispersion vs realized α-spread (is conv_gate signal real?)")
    print("="*80)
    pred_disp = []
    realized_spread = []
    for t, g in apd_entries.groupby("open_time"):
        g = g.dropna(subset=["pred", "alpha_A"])
        if len(g) < 2*K + 1: continue
        preds = g["pred"].to_numpy(); alphas = g["alpha_A"].to_numpy()
        idx_pt = np.argpartition(-preds, K-1)[:K]
        idx_pb = np.argpartition(preds, K-1)[:K]
        pd_val = preds[idx_pt].mean() - preds[idx_pb].mean()
        rs_val = (alphas[idx_pt].mean() - alphas[idx_pb].mean()) * 1e4
        pred_disp.append(pd_val); realized_spread.append(rs_val)
    pred_disp = np.array(pred_disp); realized_spread = np.array(realized_spread)
    rho_disp = np.corrcoef(pred_disp, realized_spread)[0,1]
    print(f"\nCorrelation: pred_dispersion vs realized α-spread of picks:")
    print(f"  Pearson  rho = {rho_disp:+.4f}")
    print(f"  Spearman rho = {stats.spearmanr(pred_disp, realized_spread).correlation:+.4f}")
    # decile analysis
    deciles = pd.qcut(pred_disp, 10, labels=False, duplicates="drop")
    print(f"\nDecile of pred dispersion → mean realized α-spread of picks:")
    print(f"  {'decile':>7} {'n':>5} {'pred_disp':>10} {'realized_α_spread':>20}")
    for d in sorted(set(deciles)):
        mask = deciles == d
        print(f"  {d:>7} {mask.sum():>5} {pred_disp[mask].mean():>+10.3f} {realized_spread[mask].mean():>+15.1f} bps")

    # === DIAG 3: 4h vs 24h α horizon ===
    print("\n" + "="*80)
    print("DIAG 3 — 4h vs 24h α horizon (model predicts 4h, strategy holds 24h)")
    print("="*80)
    print("  Computing 24h cumulative α_β from kline data...", flush=True)
    t0 = time.time()
    frames = []
    for sym in panel_syms:
        sd = KLINES_DIR / sym / "5m"
        if not sd.exists(): continue
        files = sorted(sd.glob("*.parquet"))
        dfs = []
        for f in files:
            try: dfs.append(pd.read_parquet(f, columns=["open_time", "close"]))
            except Exception: pass
        if not dfs: continue
        df = pd.concat(dfs, ignore_index=True)
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
        df = df.dropna(subset=["open_time"]).drop_duplicates("open_time").set_index("open_time")
        df = df.rename(columns={"close": sym})
        frames.append(df)
    close_wide = pd.concat(frames, axis=1).sort_index()
    fwd_24h_ret = (close_wide.shift(-HOLD_BARS) - close_wide) / close_wide
    print(f"  fwd_24h_ret done ({time.time()-t0:.0f}s)", flush=True)

    # 24h cumulative α_β ≈ 24h cumulative return - β_PIT × 24h cumulative BTC return
    # For diagnostic, compare realized 4h α_β to 24h forward return (skip β-hedge calc, just rough)
    # Use sampled entries
    rho_4h_24h_list = []
    spread_24h_picks = []
    spread_24h_oracle_24h = []
    for t in sorted(sampled_t):
        if t not in fwd_24h_ret.index: continue
        g = apd_entries[apd_entries["open_time"] == t].dropna(subset=["alpha_A", "pred"])
        if len(g) < 2*K + 1: continue
        r24 = fwd_24h_ret.loc[t]
        merged = g.merge(r24.rename("r24").reset_index().rename(columns={"index":"symbol"}), on="symbol", how="inner")
        merged = merged.dropna(subset=["alpha_A", "r24"])
        if len(merged) < 2*K + 1: continue
        ar_4h = merged["alpha_A"]
        r_24h = merged["r24"]
        rho_4h_24h_list.append(ar_4h.rank().corr(r_24h.rank()))
        # picks by pred → realized 24h
        preds = merged["pred"].to_numpy()
        r24_arr = merged["r24"].to_numpy()
        idx_pt = np.argpartition(-preds, K-1)[:K]
        idx_pb = np.argpartition(preds, K-1)[:K]
        spread_24h_picks.append((r24_arr[idx_pt].mean() - r24_arr[idx_pb].mean()) * 1e4)
        # picks by realized 24h (oracle 24h)
        idx_o_t = np.argpartition(-r24_arr, K-1)[:K]
        idx_o_b = np.argpartition(r24_arr, K-1)[:K]
        spread_24h_oracle_24h.append((r24_arr[idx_o_t].mean() - r24_arr[idx_o_b].mean()) * 1e4)
    rho_arr = np.array(rho_4h_24h_list)
    s24p = np.array(spread_24h_picks); s24o = np.array(spread_24h_oracle_24h)
    print(f"\n  Cross-sectional rank corr(realized_4h_α_β, realized_24h_return) per cycle:")
    print(f"    mean={rho_arr.mean():+.4f}, median={np.median(rho_arr):+.4f}, std={rho_arr.std():.4f}")
    print(f"    → if rho is low, 4h α-β predictions don't carry into 24h hold horizon")
    print(f"\n  Model pick performance on 24h horizon (raw return, no β-hedge for diagnostic):")
    print(f"    Model 24h spread of picks: {s24p.mean():+.1f} bps")
    print(f"    Oracle 24h spread (knows realized 24h): {s24o.mean():+.1f} bps")
    print(f"    Capture rate: {s24p.mean()/s24o.mean()*100:.1f}%")

    # === DIAG 4: per-symbol contribution ===
    print("\n" + "="*80)
    print("DIAG 4 — Per-symbol contribution (which names carry the alpha)")
    print("="*80)
    sym_rows = []
    for sym in panel_syms:
        # times when sym was in model top-3 (pred-rank in top-3)
        times_picked_long = []
        times_picked_short = []
        sym_alpha_when_long = []
        sym_alpha_when_short = []
        sym_ic_full = []
        for t, g in apd_entries.groupby("open_time"):
            g = g.dropna(subset=["pred", "alpha_A"])
            if len(g) < 2*K + 1: continue
            preds = g["pred"].to_numpy()
            syms = g["symbol"].to_numpy()
            alphas = g["alpha_A"].to_numpy()
            top_set = set(syms[np.argpartition(-preds, K-1)[:K]])
            bot_set = set(syms[np.argpartition(preds, K-1)[:K]])
            if sym in top_set:
                idx_sym = np.where(syms == sym)[0][0]
                times_picked_long.append(t)
                sym_alpha_when_long.append(alphas[idx_sym])
            if sym in bot_set:
                idx_sym = np.where(syms == sym)[0][0]
                times_picked_short.append(t)
                sym_alpha_when_short.append(alphas[idx_sym])
        n_long = len(sym_alpha_when_long); n_short = len(sym_alpha_when_short)
        mean_when_long = np.mean(sym_alpha_when_long) * 1e4 if n_long else np.nan
        mean_when_short = np.mean(sym_alpha_when_short) * 1e4 if n_short else np.nan
        # contribution to PnL: longs benefit when alpha is positive; shorts benefit when alpha is negative
        # per-symbol per-pick PnL (in bps of one leg's notional)
        long_contrib_bps = mean_when_long if n_long else 0
        short_contrib_bps = -mean_when_short if n_short else 0  # short PnL = -alpha
        total_picks = n_long + n_short
        # net dollar contribution per pick: sum of long alphas + sum of (-short alphas)
        # weighted by participation: per-symbol pick frequency
        sym_rows.append({
            "symbol": sym,
            "n_long_picks": n_long,
            "n_short_picks": n_short,
            "frac_picked": total_picks / apd_entries["open_time"].nunique(),
            "mean_alpha_when_long_bps": mean_when_long,
            "mean_alpha_when_short_bps": mean_when_short,
            "long_contrib_bps": long_contrib_bps,
            "short_contrib_bps": short_contrib_bps,
        })
    sym_df = pd.DataFrame(sym_rows)
    print("\nTop 10 alpha contributors (long when picked + short when picked):")
    sym_df["net_contrib_bps_per_pick"] = (sym_df["long_contrib_bps"] * sym_df["n_long_picks"]
                                            + sym_df["short_contrib_bps"] * sym_df["n_short_picks"]) \
                                            / (sym_df["n_long_picks"] + sym_df["n_short_picks"]).replace(0, np.nan)
    sym_df["total_contrib_bps"] = (sym_df["long_contrib_bps"] * sym_df["n_long_picks"]
                                     + sym_df["short_contrib_bps"] * sym_df["n_short_picks"])
    sym_df = sym_df.sort_values("total_contrib_bps", ascending=False)
    pd.set_option("display.width", 200)
    print(sym_df.head(10)[["symbol","n_long_picks","n_short_picks",
                            "mean_alpha_when_long_bps","mean_alpha_when_short_bps",
                            "total_contrib_bps"]].to_string(index=False))
    print("\nBottom 10 (negative contributors — dilute the strategy):")
    print(sym_df.tail(10)[["symbol","n_long_picks","n_short_picks",
                            "mean_alpha_when_long_bps","mean_alpha_when_short_bps",
                            "total_contrib_bps"]].to_string(index=False))
    sym_df.to_csv(OUT / "per_symbol_contribution.csv", index=False)

    # === DIAG 5: per-fold breakdown ===
    print("\n" + "="*80)
    print("DIAG 5 — Per-fold realized spread (is one fold carrying everything?)")
    print("="*80)
    for fid in OOS_FOLDS:
        fold_t = apd_entries[apd_entries["fold"] == fid]["open_time"].unique()
        spreads = []
        for t in fold_t:
            g = apd_entries[apd_entries["open_time"] == t].dropna(subset=["pred","alpha_A"])
            if len(g) < 2*K + 1: continue
            preds = g["pred"].to_numpy(); alphas = g["alpha_A"].to_numpy()
            idx_pt = np.argpartition(-preds, K-1)[:K]
            idx_pb = np.argpartition(preds, K-1)[:K]
            sp = (alphas[idx_pt].mean() - alphas[idx_pb].mean()) * 1e4
            spreads.append(sp)
        spreads = np.array(spreads)
        print(f"  fold {fid}: n_cycles={len(spreads):>4}  mean_spread={spreads.mean():+7.1f} bps  "
              f"median={np.median(spreads):+7.1f} bps  hit_rate={(spreads > 0).mean():.1%}")

    # === DIAG 6: pick persistence ===
    print("\n" + "="*80)
    print("DIAG 6 — Pick persistence (natural autocorrelation, informs PM_M2 value)")
    print("="*80)
    long_picks_over_time = []
    short_picks_over_time = []
    for t in sorted(apd_entries["open_time"].unique()):
        g = apd_entries[apd_entries["open_time"] == t].dropna(subset=["pred"])
        if len(g) < 2*K + 1:
            long_picks_over_time.append(None); short_picks_over_time.append(None); continue
        preds = g["pred"].to_numpy(); syms = g["symbol"].to_numpy()
        long_picks_over_time.append(set(syms[np.argpartition(-preds, K-1)[:K]]))
        short_picks_over_time.append(set(syms[np.argpartition(preds, K-1)[:K]]))
    long_overlap_1, short_overlap_1 = [], []
    long_overlap_2, short_overlap_2 = [], []
    for i in range(1, len(long_picks_over_time)):
        if long_picks_over_time[i] is not None and long_picks_over_time[i-1] is not None:
            long_overlap_1.append(len(long_picks_over_time[i] & long_picks_over_time[i-1]))
            short_overlap_1.append(len(short_picks_over_time[i] & short_picks_over_time[i-1]))
        if i >= 2 and long_picks_over_time[i] is not None and long_picks_over_time[i-2] is not None:
            long_overlap_2.append(len(long_picks_over_time[i] & long_picks_over_time[i-2]))
            short_overlap_2.append(len(short_picks_over_time[i] & short_picks_over_time[i-2]))
    print(f"\n  Cycle-over-cycle pick overlap (out of K={K}):")
    print(f"    Long  cycle vs cycle-1:  mean = {np.mean(long_overlap_1):.2f} / {K}  "
          f"({np.mean(long_overlap_1)/K*100:.0f}% retention)")
    print(f"    Short cycle vs cycle-1:  mean = {np.mean(short_overlap_1):.2f} / {K}  "
          f"({np.mean(short_overlap_1)/K*100:.0f}% retention)")
    print(f"    Long  cycle vs cycle-2:  mean = {np.mean(long_overlap_2):.2f} / {K}")
    print(f"    Short cycle vs cycle-2:  mean = {np.mean(short_overlap_2):.2f} / {K}")

    # ========================================================================
    print("\n" + "="*80)
    print("SYNTHESIS — where the optimization headroom lives")
    print("="*80)
    print("(See findings above and per_symbol_contribution.csv)", flush=True)


if __name__ == "__main__":
    main()
