"""Step 60: HL-native (70-sym) retrain re-measured under CAUSAL aggregator + funding.

Step 56 retrained Ridge from scratch on the 70 HL-executable symbols and got
Sharpe -1.29 — but under the LAGGED aggregator. clean-108's +2.34 is under the
CAUSAL aggregator (58b's correction). For an apples-to-apples comparison, and to
test the user's thesis ("if the system reliably identifies symbols it's fine,
and shorting crowded/pumping memes EARNS funding"), rerun Step 56's saved
predictions through the SAME causal aggregator + realized funding + P1/P2.

No retrain — reuses results/step56_hl_native/predictions.parquet (70 syms;
PIPPIN/BROCCOLI714 are NOT in this universe — the clean-108 drivers don't exist
in the executable set, which is exactly the question).
"""
from __future__ import annotations
import sys, time, importlib.util, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location("psl", REPO / "scripts/phase_ah_sleeve.py")
psl = importlib.util.module_from_spec(spec); spec.loader.exec_module(psl)
s59spec = importlib.util.spec_from_file_location(
    "s59", REPO / "linear_model/scripts/59_clean108_funding.py")
s59 = importlib.util.module_from_spec(s59spec); s59spec.loader.exec_module(s59)
from ml.research.alpha_v4_xs import block_bootstrap_ci

PREDS_DIR = REPO / "linear_model/results/step56_hl_native"
OUT = REPO / "linear_model/results/step60_hl_native_funding"
OUT.mkdir(parents=True, exist_ok=True)
OOS_FOLDS = list(range(1, 10))
MIN_HISTORY_DAYS = 60
N_PLACEBO = 100


def main():
    print("=" * 100, flush=True)
    print("  STEP 60: HL-native (70-sym) — CAUSAL aggregator + realized funding", flush=True)
    print("=" * 100, flush=True)
    print("  Step 56 reference (LAGGED aggregator): B = -1.29, A = -1.52", flush=True)
    t0 = time.time()
    listings = s59.get_listings()

    apd_full = pd.read_parquet(PREDS_DIR / "predictions.parquet")
    apd_full["open_time"] = pd.to_datetime(apd_full["open_time"], utc=True)
    apd_full["alpha_A"] = apd_full["alpha_beta"]
    panel_syms = sorted(apd_full["symbol"].unique())
    print(f"\nHL-native universe: {len(panel_syms)} symbols "
          f"(PIPPIN in? {'PIPPINUSDT' in panel_syms}; "
          f"BROCCOLI714 in? {'BROCCOLI714USDT' in panel_syms})", flush=True)

    for s, t in apd_full.groupby("symbol")["open_time"].min().items():
        if s not in listings:
            t = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[s] = t
    panel_syms_set = set(panel_syms)
    def elig_pit(b):
        ts = b if isinstance(b, pd.Timestamp) else pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms_set if listings.get(s) and listings[s] <= cutoff}
    target_t = sorted(apd_full[apd_full["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::psl.HORIZON_ENTRY]

    apd_full["pred_B"] = apd_full["pred_z"] * apd_full["trail_ic"]
    apd_full["pred"] = apd_full["pred_z"]
    universe_V2 = psl.build_rolling_ic_universe(apd_full, sampled_t, psl.TOP_N, elig_pit)
    alpha_wide = apd_full.pivot_table(index="open_time", columns="symbol",
                                        values="alpha_A", aggfunc="first").sort_index()

    print("\nInferring per-symbol funding cadence from data...", flush=True)
    fund_block, intervals = s59.infer_funding(panel_syms, sampled_t)
    from collections import Counter
    print(f"  funding interval dist (h): {dict(Counter(intervals.values()))}, "
          f"mean |block funding| = {fund_block.abs().mean().mean()*1e4:.2f} bps",
          flush=True)

    apd_v = apd_full.copy(); apd_v["pred"] = apd_v["pred_B"]
    records_real = psl.run_production_protocol_save_sleeves(apd_v, universe_V2)
    df = s59.aggregate_causal_funding(records_real, alpha_wide, fund_block)
    df.to_csv(OUT / "per_cycle_real_funding.csv", index=False)

    sh_f = s59._sharpe(df["net_pnl_bps"].to_numpy())
    sh_nf = s59._sharpe(df["net_nofund_bps"].to_numpy())
    lo, hi = block_bootstrap_ci(df["net_pnl_bps"].to_numpy(), statistic=s59._sharpe,
                                  block_size=7, n_boot=1000)[1:]
    print(f"\n{'='*100}", flush=True)
    print("  HL-NATIVE B_IC_signed — causal aggregator", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"  Sharpe causal NO funding : {sh_nf:+.2f}   "
          f"(vs Step 56 LAGGED -1.29)", flush=True)
    print(f"  Sharpe causal WITH fund  : {sh_f:+.2f}  [{lo:+.2f}, {hi:+.2f}]", flush=True)
    print(f"  mean gross   = {df['gross_pnl_bps'].mean():+.2f} bps/cyc", flush=True)
    print(f"  mean funding = {df['funding_pnl_bps'].mean():+.2f} bps/cyc "
          f"(>0 = strategy EARNS funding)", flush=True)
    print(f"  mean cost    = {df['cost_bps'].mean():+.2f} bps/cyc", flush=True)
    print(f"  mean net     = {df['net_pnl_bps'].mean():+.2f} bps/cyc", flush=True)
    print(f"  folds+ = {s59.folds_positive(df)}/9", flush=True)

    print(f"\n  Per-fold (gross / funding / cost / net / Sharpe):", flush=True)
    for fid, g in df.groupby("fold"):
        print(f"    fold {fid}: g={g['gross_pnl_bps'].mean():+7.2f}  "
              f"f={g['funding_pnl_bps'].mean():+6.2f}  c={g['cost_bps'].mean():5.2f}  "
              f"net={g['net_pnl_bps'].mean():+7.2f}  Sh={s59._sharpe(g['net_pnl_bps']):+.2f}",
              flush=True)

    universe_liq = s59.build_liquidity_universe(sampled_t, panel_syms, n_top=30)
    for name, univ in [("P1 (liq-univ random)", universe_liq),
                        ("P2 (V2-univ random)", universe_V2)]:
        ps = []
        for seed in range(N_PLACEBO):
            rp = psl.run_production_protocol_save_sleeves(apd_v, univ, placebo_seed=seed)
            dp = s59.aggregate_causal_funding(rp, alpha_wide, fund_block)
            ps.append(s59._sharpe(dp["net_pnl_bps"].to_numpy()))
        ps = np.array(ps); p95 = float(np.percentile(ps, 95))
        rank = (ps < sh_f).mean() * 100
        print(f"\n  {name} ×{N_PLACEBO} WITH funding: p95={p95:+.2f}  "
              f"real rank p{rank:.0f}  edge {sh_f - p95:+.2f}  "
              f"{'PASS' if sh_f > p95 else 'FAIL'}", flush=True)
        pd.DataFrame({name: ps}).to_csv(
            OUT / f"placebo_{name.split()[0]}_funding.csv", index=False)

    print(f"\n  VERDICT: clean-108 causal+fund = +2.34 (PASS p98); "
          f"HL-native causal+fund = {sh_f:+.2f} "
          f"— tests whether a tradeable version exists.", flush=True)
    print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
