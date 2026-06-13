"""Step 36: Raw 4h-cycle baseline (NO sleeve overlay).

Tests whether V1 fixed and V2 fixed have lift at the clean 4h-cycle level,
without V3.1's 6-sleeve 24h overlay.

For each 4h cycle:
  - Pick K=3 long + K=3 short from rolling-IC top-15 universe via pred_B
  - Apply conv_gate + PM + filter_refill (same gates as V3.1)
  - Hold 4h (exit at t+48 bars), then re-enter on next cycle
  - PnL = K_avg(long_ret) - K_avg(short_ret) in bps
  - Cost = 2 × COST_PER_LEG = 9 bps per full-turnover cycle
    (each leg at weight 1/K bears 4.5/K bps round-trip; K longs + K shorts
    round-trip = 2 × K × 4.5/K = 9 bps total per cycle.)
    NOTE: pre-2026-05-14 versions of this script incorrectly used
    `2 × K × COST_PER_LEG = 27 bps` (triple-count). Sharpes scaled by 1/3
    after the fix (V2 raw 4h went from −7.45 to −2.55).

Compare to:
  - V2 with sleeve (Step 35): Sharpe +2.19
  - V1 with sleeve (Step 35): +1.21
  - K=3 production LGBM raw 4h-cycle: memory says +1.98
"""
from __future__ import annotations
import sys, time, importlib.util, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location("psl", REPO / "scripts/phase_ah_sleeve.py")
psl = importlib.util.module_from_spec(spec); spec.loader.exec_module(psl)
from ml.research.alpha_v4_xs import block_bootstrap_ci

PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
PREDS_DIR = REPO / "linear_model/results/step34_v1_fixed"
OUT = REPO / "linear_model/results"

OOS_FOLDS = list(range(1, 10))
MIN_HISTORY_DAYS = 60
TRAILING_IC_DAYS = 90
K = psl.K  # 3
# Correct cost: each leg at weight 1/K bears COST_PER_LEG / K bps round-trip.
# K longs + K shorts round-trip = 2 × (K × COST_PER_LEG / K) = 2 × COST_PER_LEG = 9 bps.
COST_PER_CYCLE = 2 * psl.COST_PER_LEG  # 2 sides × COST_PER_LEG = 9 bps full turnover


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(psl.CYCLES_PER_YEAR))


def folds_positive(df_v):
    return sum(1 for _, g in df_v.groupby("fold") if _sharpe(g["net_pnl_bps"]) > 0)


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


def compute_trailing_ic(apd, sampled_t, win_days=90):
    apd_s = apd[apd["open_time"].isin(set(sampled_t))].sort_values(
        ["symbol","open_time"]).reset_index(drop=True)
    win_cycles = win_days * 6
    rows = []
    for sym, g in apd_s.groupby("symbol"):
        g = g.sort_values("open_time").reset_index(drop=True)
        pred = g["pred_z"].to_numpy(); alpha = g["alpha_beta"].to_numpy()
        n = len(g)
        ics = np.full(n, np.nan)
        for i in range(50, n):
            lo = max(0, i - win_cycles)
            p, a = pred[lo:i], alpha[lo:i]
            mask = ~np.isnan(p) & ~np.isnan(a)
            if mask.sum() < 50: continue
            pr = pd.Series(p[mask]).rank().to_numpy()
            ar = pd.Series(a[mask]).rank().to_numpy()
            if pr.std() < 1e-6 or ar.std() < 1e-6: continue
            ics[i] = np.corrcoef(pr, ar)[0,1]
        for j, t in enumerate(g["open_time"]):
            rows.append({"symbol":sym, "open_time":t, "trail_ic":ics[j]})
    return pd.DataFrame(rows).fillna(0)


def aggregate_raw_4h(records, alpha_wide):
    """No sleeve — each cycle's basket earns its 4h return then exits.
    PnL per cycle = mean(long_alpha) - mean(short_alpha) in bps.
    Cost per cycle = 2K * COST_PER_LEG (full turnover, no overlap).
    """
    rows = []
    for _, rec in records.iterrows():
        t = rec["time"]; fold = rec["fold"]
        if not rec["traded"]:
            rows.append({"time":t, "fold":fold, "gross_pnl_bps":0.0,
                         "cost_bps":0.0, "net_pnl_bps":0.0})
            continue
        longs = list(rec["long_basket"])
        shorts = list(rec["short_basket"])
        if t not in alpha_wide.index or not longs or not shorts:
            rows.append({"time":t, "fold":fold, "gross_pnl_bps":0.0,
                         "cost_bps":0.0, "net_pnl_bps":0.0})
            continue
        a = alpha_wide.loc[t]
        long_alphas = [a[s] for s in longs if s in a.index and not pd.isna(a[s])]
        short_alphas = [a[s] for s in shorts if s in a.index and not pd.isna(a[s])]
        if not long_alphas or not short_alphas:
            rows.append({"time":t, "fold":fold, "gross_pnl_bps":0.0,
                         "cost_bps":0.0, "net_pnl_bps":0.0})
            continue
        gross = (np.mean(long_alphas) - np.mean(short_alphas)) * 1e4
        cost = COST_PER_CYCLE
        rows.append({"time":t, "fold":fold, "gross_pnl_bps":gross,
                     "cost_bps":cost, "net_pnl_bps":gross - cost})
    return pd.DataFrame(rows)


def run_variant(variant_slug, listings):
    print(f"\n{'='*100}", flush=True)
    print(f"  {variant_slug} (raw 4h cycle, no sleeve)", flush=True)
    print(f"{'='*100}", flush=True)
    apd_full = pd.read_parquet(PREDS_DIR / f"{variant_slug}_predictions.parquet")
    apd_full["open_time"] = pd.to_datetime(apd_full["open_time"], utc=True)
    apd_full["alpha_A"] = apd_full["alpha_beta"]
    if "exit_time" not in apd_full.columns or "return_pct" not in apd_full.columns:
        extra = pd.read_parquet(PANEL,
                                  columns=["symbol","open_time","exit_time","return_pct"])
        extra["open_time"] = pd.to_datetime(extra["open_time"], utc=True)
        extra["exit_time"] = pd.to_datetime(extra["exit_time"], utc=True)
        apd_full = apd_full.merge(extra, on=["symbol","open_time"], how="left")

    panel_syms = set(apd_full["symbol"].unique())
    for s, t in apd_full.groupby("symbol")["open_time"].min().items():
        if s not in listings:
            t = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[s] = t
    def elig_pit(b):
        ts = b if isinstance(b, pd.Timestamp) else pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}
    target_t = sorted(apd_full[apd_full["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::psl.HORIZON_ENTRY]

    apd_full["pred_B"] = apd_full["pred_z"] * apd_full["trail_ic"]
    apd_full["pred"] = apd_full["pred_z"]
    universe = psl.build_rolling_ic_universe(apd_full, sampled_t, psl.TOP_N, elig_pit)
    alpha_wide = apd_full.pivot_table(index="open_time", columns="symbol",
                                        values="alpha_A", aggfunc="first").sort_index()

    out_results = {}
    for sub, col in [("A", "pred_z"), ("B", "pred_B")]:
        apd_v = apd_full.copy(); apd_v["pred"] = apd_v[col]
        records = psl.run_production_protocol_save_sleeves(apd_v, universe)
        df_v = aggregate_raw_4h(records, alpha_wide)
        net = df_v["net_pnl_bps"].to_numpy()
        sh = _sharpe(net)
        n_traded = (df_v["gross_pnl_bps"] != 0).sum()
        sh_lo, sh_hi = block_bootstrap_ci(net, statistic=_sharpe,
                                            block_size=7, n_boot=1000)[1:]
        out_results[sub] = {"sharpe":sh, "sh_lo":sh_lo, "sh_hi":sh_hi,
                            "gross":df_v["gross_pnl_bps"].mean(),
                            "folds_pos":folds_positive(df_v),
                            "df_v":df_v, "n_traded":n_traded}
        sub_label = "baseline (pred_z)" if sub == "A" else "IC_signed (pred_B)"
        print(f"  {sub} {sub_label}: Sharpe={sh:+.2f} [{sh_lo:+.2f},{sh_hi:+.2f}]  "
              f"gross={df_v['gross_pnl_bps'].mean():+.2f}  "
              f"folds+={folds_positive(df_v)}/9  traded={n_traded}/{len(df_v)}", flush=True)

    # LOFO on B
    sh_B = out_results["B"]["sharpe"]; df_v_B = out_results["B"]["df_v"]
    print(f"  LOFO on B (Sharpe = {sh_B:+.2f}):", flush=True)
    lofo_rows = []
    for excl in range(1, 10):
        rem = df_v_B[df_v_B["fold"] != excl]["net_pnl_bps"].to_numpy()
        sh_rem = _sharpe(rem)
        d = sh_rem - sh_B
        flag = "  ← drives" if d < -0.4 else ""
        print(f"    excl {excl}: {sh_rem:+.2f} (Δ {d:+.2f}){flag}", flush=True)
        lofo_rows.append({"excl":excl, "sharpe":sh_rem, "delta":d})
    pd.DataFrame(lofo_rows).to_csv(OUT / f"step36_{variant_slug}_lofo.csv", index=False)
    out_results["B"]["df_v"].to_csv(OUT / f"step36_{variant_slug}_per_cycle.csv", index=False)
    return out_results


def main():
    print("=== Step 36: Raw 4h cycle baseline (no sleeve) ===\n", flush=True)
    print(f"K={K}, cost per cycle = {COST_PER_CYCLE:.1f} bps "
          f"(2K legs × {psl.COST_PER_LEG} bps)\n", flush=True)
    t0 = time.time()
    listings = get_listings()

    results = []
    for slug in ["v0_standard", "v1_fixed", "v2_fixed"]:
        r = run_variant(slug, listings)
        results.append({
            "variant":slug,
            "A_sharpe":r["A"]["sharpe"], "A_sh_lo":r["A"]["sh_lo"], "A_sh_hi":r["A"]["sh_hi"],
            "A_gross":r["A"]["gross"], "A_folds_pos":r["A"]["folds_pos"],
            "A_traded":r["A"]["n_traded"],
            "B_sharpe":r["B"]["sharpe"], "B_sh_lo":r["B"]["sh_lo"], "B_sh_hi":r["B"]["sh_hi"],
            "B_gross":r["B"]["gross"], "B_folds_pos":r["B"]["folds_pos"],
            "B_traded":r["B"]["n_traded"],
        })

    # Summary
    print(f"\n{'='*100}", flush=True)
    print(f"  RAW 4H CYCLE SUMMARY (NO SLEEVE OVERLAY)", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"  {'variant':<14}  {'A_baseline':>12}  {'B_IC_signed':>14}  "
          f"{'folds+ (B)':>12}  {'traded (B)':>12}", flush=True)
    for r in results:
        print(f"  {r['variant']:<14}  {r['A_sharpe']:+12.2f}  {r['B_sharpe']:+14.2f}  "
              f"{r['B_folds_pos']:>10}/9  {r['B_traded']:>12}", flush=True)

    print(f"\n  Reference comparisons:", flush=True)
    print(f"    V0/V1/V2 with V3.1 sleeve: +0.67 / +1.21 / +2.19", flush=True)
    print(f"    LGBM K=3 production (raw 4h, no sleeve): +1.98 (per memory)", flush=True)
    print(f"    LGBM K=3 + V3.1 sleeve (production): +2.23 (per memory)", flush=True)

    pd.DataFrame(results).to_csv(OUT / "step36_summary.csv", index=False)
    print(f"\n  Saved: results/step36_*.csv", flush=True)
    print(f"  Total: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
