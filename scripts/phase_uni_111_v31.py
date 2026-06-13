"""Phase UNI-111: run V3.1 on the full 111-symbol expanded panel.

The 111-sym set was already filtered by Phase E5b's $10M PIT daily-volume rule.
This script:
  1. Loads existing 111-sym audit panel (retrained model predictions)
  2. Builds rolling-IC top-15 universe from 111 candidates
  3. Runs V3.1 production protocol (K=3 + filter_refill + conv_gate + flat_real)
  4. Runs V3.1 6-sleeve aggregation
  5. Reports:
     - Final Sharpe vs 51-panel baseline +2.23
     - Per-symbol pick frequency: which symbols does rolling-IC pick?
     - Comparison of high-IC symbols (LTC, ASTER, NEAR from diagnostic): still picked?
     - Per-fold breakdown
"""
from __future__ import annotations
import sys, time, importlib.util
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location(
    "psl", REPO / "scripts/phase_ah_sleeve.py")
psl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(psl)
spec2 = importlib.util.spec_from_file_location(
    "svar", REPO / "scripts/phase_ah_sleeve_variants.py")
svar = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(svar)

OUT = REPO / "outputs/vBTC_uni_111"
OUT.mkdir(parents=True, exist_ok=True)

HORIZON = 48
HOLD_BARS = 288
N_SLEEVES = 6
OOS_FOLDS = list(range(1, 10))
CYCLES_PER_YEAR = (288 * 365) / 48


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR))


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def main():
    print("=== Phase UNI-111: V3.1 on 111-symbol expanded panel ===\n", flush=True)

    # Load 111-sym audit
    apd = pd.read_parquet(REPO / "outputs/vBTC_audit_panel_expanded/all_predictions.parquet")
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    all_syms = sorted(apd["symbol"].unique())
    print(f"  loaded {len(apd):,} predictions, {len(all_syms)} symbols", flush=True)
    print(f"  time range: {apd['open_time'].min()} → {apd['open_time'].max()}", flush=True)

    # Build rolling-IC universe (same machinery, but on 111-pool)
    listings = psl.get_listings()
    panel_first_obs = apd.groupby("symbol")["open_time"].min()
    for sym, t in panel_first_obs.items():
        if sym not in listings:
            listings[sym] = t
    def elig_at(b):
        ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=psl.MIN_HISTORY_DAYS)
        return {s for s in apd["symbol"].unique()
                  if listings.get(s) and listings[s] <= cutoff}
    tgt = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled = tgt[::HORIZON]
    print(f"\n  building rolling-IC top-15 universe from 111 candidates...", flush=True)
    t0 = time.time()
    universe = psl.build_rolling_ic_universe(apd, sampled, psl.TOP_N, elig_at)
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)

    # Pick frequency analysis: which symbols get into the rolling-IC top-15?
    pick_counts = Counter()
    for t, u in universe.items():
        for s in u:
            pick_counts[s] += 1
    total_cycles = len(universe)
    print(f"\n  Total cycle samplings: {total_cycles}", flush=True)
    print(f"\n  Rolling-IC top-15 membership frequency (% of cycles each symbol is in universe):",
          flush=True)
    print(f"  Symbols sorted by pick frequency:", flush=True)
    print(f"  {'symbol':<18}  {'pick_pct':>8}  {'n_cycles':>8}", flush=True)
    for sym, cnt in sorted(pick_counts.items(), key=lambda x: -x[1]):
        pct = cnt / total_cycles * 100
        if pct > 5:  # show only symbols picked > 5% of cycles
            print(f"  {sym:<18}  {pct:>7.1f}%  {cnt:>8d}", flush=True)
    print(f"  ... {sum(1 for v in pick_counts.values() if v/total_cycles*100 <= 5)} "
          f"more symbols picked ≤5% of cycles", flush=True)

    # Check key high-IC symbols from diagnostic
    diagnostic_top_ic = ["LTCUSDT", "ASTERUSDT", "NEARUSDT", "AAVEUSDT", "SUIUSDT",
                          "ORDIUSDT", "FILUSDT", "ETCUSDT", "TIAUSDT", "GMXUSDT"]
    diagnostic_bot_ic = ["ETHUSDT", "BIOUSDT", "JTOUSDT", "BNBUSDT", "PENGUUSDT",
                           "ZECUSDT", "ENAUSDT", "BCHUSDT", "PUMPUSDT", "JUPUSDT"]
    print(f"\n  Check: high-IC symbols (from diagnostic) frequency in 111-panel rolling-IC:",
          flush=True)
    for sym in diagnostic_top_ic:
        cnt = pick_counts.get(sym, 0)
        pct = cnt / total_cycles * 100
        print(f"    {sym:<14}  {pct:>5.1f}%", flush=True)
    print(f"\n  Check: low-IC symbols (from diagnostic) frequency in 111-panel rolling-IC:",
          flush=True)
    for sym in diagnostic_bot_ic:
        cnt = pick_counts.get(sym, 0)
        pct = cnt / total_cycles * 100
        print(f"    {sym:<14}  {pct:>5.1f}%", flush=True)

    # Run V3.1 protocol on 111-panel
    print(f"\n  Building sleeves with V3.1 protocol on 111-panel...", flush=True)
    t0 = time.time()
    sleeves = psl.run_production_protocol_save_sleeves(apd, universe)
    sleeves["time"] = pd.to_datetime(sleeves["time"], utc=True)
    n_tr = sleeves["traded"].sum()
    print(f"  done: traded {n_tr}/{len(sleeves)} cycles ({time.time()-t0:.0f}s)", flush=True)
    sleeves.to_parquet(OUT / "sleeves_v31_111.parquet", index=False)

    # Load close prices for all 111 syms
    print(f"  loading close prices for 111 syms...", flush=True)
    t0 = time.time()
    close_wide = svar.load_close_wide(all_syms)
    fwd_rets_4h = (close_wide.shift(-HORIZON) - close_wide) / close_wide
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)

    # V3.1 aggregation
    print(f"  V3.1 aggregation...", flush=True)
    df_v = svar.aggregate_sleeves_variant(sleeves, fwd_rets_4h, N_SLEEVES, HOLD_BARS,
                                                sleeve_weights=[1/6]*6)
    df_v.to_csv(OUT / "per_cycle_v31_111.csv", index=False)

    # Stats
    sh = _sharpe(df_v["net_pnl_bps"])
    dd = _max_dd(df_v["net_pnl_bps"])
    pnl = df_v["net_pnl_bps"].sum()
    npos = sum(1 for _, g in df_v.groupby("fold") if _sharpe(g["net_pnl_bps"]) > 0)
    print(f"\n  V3.1 on 111-panel:  Sharpe = {sh:+.3f}  maxDD = {dd:+.0f}  "
          f"PnL = {pnl:+.0f}  folds+ = {npos}/9", flush=True)
    print(f"  V3.1 on 51-panel (baseline): Sharpe = +2.229", flush=True)
    print(f"  Δ Sharpe (111 - 51): {sh - 2.229:+.3f}", flush=True)

    # Per-fold breakdown
    print(f"\n  Per-fold breakdown:", flush=True)
    print(f"  {'fold':>4}  {'PnL':>8}  {'Sharpe':>7}", flush=True)
    for f in OOS_FOLDS:
        g = df_v[df_v["fold"] == f]["net_pnl_bps"]
        if len(g) > 0:
            print(f"  {f:>4}  {g.sum():>+8.0f}  {_sharpe(g):>+7.2f}", flush=True)


if __name__ == "__main__":
    main()
