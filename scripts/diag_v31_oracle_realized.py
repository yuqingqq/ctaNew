"""Oracle ceiling on V3.1 trading logic — substitute realized 4h forward return for pred.

What this measures: "if our model had perfect knowledge of next-4h return, what would
V3.1 actually earn?" — using the full production stack (conv_gate, PM_M2, filter_refill,
6-sleeve overlay, production cost model).

Implementation:
  1. Reuse Phase 1D's predictions parquet (same universe, same model alpha_A)
  2. Substitute `pred` with `return_pct` (realized 4h forward return)
  3. Build rolling-IC universe using these new "preds" (= realized returns)
  4. Run production V3.1 protocol unchanged — all gates fire on oracle scores
  5. Aggregate sleeves and MTM on actual return_pct
  6. Report $-PnL on $100 capital base, Sharpe, max DD, what V3.1 would realize

Compare to Phase 1D's actual +0.65 Sharpe with model preds.
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
from ml.research.alpha_v4_xs import block_bootstrap_ci

APD_PATH = REPO / "outputs/vBTC_phase1d_rolling_beta/all_predictions.parquet"
OUT = REPO / "outputs/vBTC_v31_oracle_realized"
OUT.mkdir(parents=True, exist_ok=True)

OOS_FOLDS = list(range(1, 10))
CAPITAL = 100.0  # $100 capital base for clear $-PnL reporting


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(psl.CYCLES_PER_YEAR))


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def folds_positive(df_v):
    return sum(1 for _, g in df_v.groupby("fold") if _sharpe(g["net_pnl_bps"]) > 0)


def load_fwd_rets(panel_syms):
    print("  Loading 5m close prices for fwd_rets MTM...", flush=True)
    t0 = time.time()
    frames = []
    for sym in panel_syms:
        sd = psl.KLINES_DIR / sym / "5m"
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
    fwd_rets_4h = (close_wide.shift(-psl.HORIZON_ENTRY) - close_wide) / close_wide
    print(f"  ready ({time.time()-t0:.0f}s)", flush=True)
    return fwd_rets_4h


def main():
    print("=== Oracle ceiling on V3.1 trading logic (perfect 4h forward knowledge) ===\n",
          flush=True)
    apd_orig = pd.read_parquet(APD_PATH)
    apd_orig["open_time"] = pd.to_datetime(apd_orig["open_time"], utc=True)
    apd_orig["exit_time"] = pd.to_datetime(apd_orig["exit_time"], utc=True)
    panel_syms = sorted(apd_orig["symbol"].unique())
    listings = psl.get_listings()
    print(f"Loaded {len(apd_orig):,} prediction rows, {len(panel_syms)} symbols", flush=True)
    print(f"Capital base: ${CAPITAL:.0f}\n", flush=True)

    def elig_pit(b):
        ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=psl.MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}

    fwd_rets_4h = load_fwd_rets(panel_syms)
    target_t = sorted(apd_orig[apd_orig["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::psl.HORIZON_ENTRY]

    # ========================================================================
    # VARIANT 1: Phase 1D actual (model preds → V3.1 stack) — baseline
    # ========================================================================
    print("\n" + "=" * 80, flush=True)
    print("VARIANT 1 — Phase 1D actual: model preds → V3.1 full stack")
    print("=" * 80, flush=True)
    universe_pred = psl.build_rolling_ic_universe(apd_orig, sampled_t, psl.TOP_N, elig_pit)
    records_v1 = psl.run_production_protocol_save_sleeves(apd_orig, universe_pred)
    df_v1 = psl.aggregate_sleeves(records_v1, fwd_rets_4h)
    net_v1 = df_v1["net_pnl_bps"].to_numpy()

    # ========================================================================
    # VARIANT 2: Oracle preds → V3.1 stack
    # ========================================================================
    print("\n" + "=" * 80, flush=True)
    print("VARIANT 2 — Oracle: replace pred with realized 4h return, V3.1 stack unchanged")
    print("=" * 80, flush=True)
    apd_oracle = apd_orig.copy()
    apd_oracle["pred"] = apd_oracle["return_pct"]  # perfect 4h forward knowledge
    # alpha_A is also unchanged (used by build_rolling_ic_universe to compute past IC)
    # IC computed against itself → perfect IC = 1.0 for all symbols with valid pairs
    # Universe filter then picks ~top-15 randomly among symbols with full data
    universe_oracle = psl.build_rolling_ic_universe(apd_oracle, sampled_t, psl.TOP_N, elig_pit)
    print(f"  Universe size at first boundary: {len(next(iter(universe_oracle.values())))}",
          flush=True)
    records_v2 = psl.run_production_protocol_save_sleeves(apd_oracle, universe_oracle)
    df_v2 = psl.aggregate_sleeves(records_v2, fwd_rets_4h)
    net_v2 = df_v2["net_pnl_bps"].to_numpy()

    # ========================================================================
    # VARIANT 3: Oracle preds → V3.1 stack BUT universe = full 51 (no IC filter)
    # ========================================================================
    print("\n" + "=" * 80, flush=True)
    print("VARIANT 3 — Oracle picks from full 51 (no rolling-IC universe filter)")
    print("=" * 80, flush=True)
    universe_full = {t: elig_pit(int(pd.Timestamp(t).timestamp()*1000)) for t in sampled_t}
    records_v3 = psl.run_production_protocol_save_sleeves(apd_oracle, universe_full)
    df_v3 = psl.aggregate_sleeves(records_v3, fwd_rets_4h)
    net_v3 = df_v3["net_pnl_bps"].to_numpy()

    # ========================================================================
    # Reports
    # ========================================================================
    def report(label, net, df_v, records):
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=1000)
        total_pnl_bps = net.sum()
        total_pnl_dollars = total_pnl_bps / 1e4 * CAPITAL  # bps × capital
        maxdd_dollars = _max_dd(net) / 1e4 * CAPITAL
        avg_per_cycle_dollars = net.mean() / 1e4 * CAPITAL
        print(f"\n  {label}", flush=True)
        print(f"    cycles traded         : {records['traded'].sum()}/{len(records)} "
              f"({records['traded'].sum()/len(records)*100:.0f}%)", flush=True)
        print(f"    Sharpe                : {sh:+.2f} [{lo:+.2f}, {hi:+.2f}]", flush=True)
        print(f"    avg net per cycle     : {net.mean():+.2f} bps  = ${avg_per_cycle_dollars:+.3f} on $100", flush=True)
        print(f"    total PnL             : {total_pnl_bps:+.0f} bps  = ${total_pnl_dollars:+.2f} on $100", flush=True)
        print(f"    end-of-OOS equity     : ${CAPITAL + total_pnl_dollars:.2f} (from ${CAPITAL:.0f})", flush=True)
        print(f"    max DD                : {_max_dd(net):+.0f} bps  = ${maxdd_dollars:+.2f}", flush=True)
        print(f"    gross/cycle           : {df_v['gross_pnl_bps'].mean():+.2f} bps", flush=True)
        print(f"    cost/cycle            : {df_v['cost_bps'].mean():+.2f} bps", flush=True)
        print(f"    turnover/cycle        : {df_v['turnover'].mean():.3f}", flush=True)
        print(f"    folds positive        : {folds_positive(df_v)}/9", flush=True)

    report("V1: model preds + V3.1 stack (Phase 1D actual)", net_v1, df_v1, records_v1)
    report("V2: oracle picks + V3.1 stack + rolling-IC universe", net_v2, df_v2, records_v2)
    report("V3: oracle picks + V3.1 stack + full 51 universe", net_v3, df_v3, records_v3)

    print("\n" + "=" * 80, flush=True)
    print("  SUMMARY — what V3.1 logic actually realizes")
    print("=" * 80, flush=True)
    print(f"  Starting capital: ${CAPITAL:.0f}\n", flush=True)
    for label, net in [("V1 model preds (Phase 1D actual)", net_v1),
                        ("V2 oracle on IC-filtered universe", net_v2),
                        ("V3 oracle on full 51", net_v3)]:
        sh = _sharpe(net)
        tot_dollars = net.sum() / 1e4 * CAPITAL
        end_equity = CAPITAL + tot_dollars
        pct_return = tot_dollars / CAPITAL * 100
        print(f"  {label:<45}: Sharpe {sh:+6.2f}, end-equity ${end_equity:7.2f} "
              f"(+{pct_return:+.1f}%)", flush=True)

    df_v1.to_csv(OUT / "v1_phase1d_actual.csv", index=False)
    df_v2.to_csv(OUT / "v2_oracle_ic_filtered.csv", index=False)
    df_v3.to_csv(OUT / "v3_oracle_full_51.csv", index=False)
    print(f"\nSaved per-cycle CSVs to {OUT}/", flush=True)


if __name__ == "__main__":
    main()
