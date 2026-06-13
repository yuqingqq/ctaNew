"""TRUE oracle ceiling for V3.1: pick by realized 24h cumulative return.

Prior oracle test picked by 4h forward return — wrong for V3.1's 24h hold period.
A name with high 4h return at entry might mean-revert over the remaining 20h.

For V3.1's structure (each sleeve enters at t and holds 24h), the optimal pick is
by realized 24h cumulative return at sleeve entry. Each sleeve's PnL ≈ its 24h
spread. V3.1 aggregates 6 overlapping sleeves at 1/6 weight each.

This gives a TRUE upper bound on V3.1: any noisy picker is bounded above by this.

Comparison points:
  - Phase 1D actual (model preds, gates ON): Sharpe +0.65, $124.40 from $100
  - Oracle by 24h cumulative + gates ON (apples-to-apples upper bound)
  - Oracle by 24h cumulative + gates OFF (best V3.1 structure can do)
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
OUT = REPO / "outputs/vBTC_v31_oracle_24h"
OUT.mkdir(parents=True, exist_ok=True)
OOS_FOLDS = list(range(1, 10))
CAPITAL = 100.0
HOLD_BARS = 288  # 24h hold in 5-minute bars (matches V3.1 HOLD_BARS)


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(psl.CYCLES_PER_YEAR))


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def folds_positive(df_v):
    return sum(1 for _, g in df_v.groupby("fold") if _sharpe(g["net_pnl_bps"]) > 0)


def compute_fwd_24h_return(panel_syms, kline_dir):
    """Realized 24h cumulative return per (symbol, time)."""
    print(f"  Computing realized 24h cumulative return per (sym, t)...", flush=True)
    t0 = time.time()
    frames = []
    for sym in panel_syms:
        sd = kline_dir / sym / "5m"
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
    fwd_4h = (close_wide.shift(-psl.HORIZON_ENTRY) - close_wide) / close_wide
    fwd_24h = (close_wide.shift(-HOLD_BARS) - close_wide) / close_wide
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)
    return fwd_4h, fwd_24h


def run_oracle_24h_no_gates(apd, universe_fn, fwd_24h_map, K):
    """Pick top-K/bot-K by realized 24h cumulative return; no gates."""
    df = apd.sort_values(["open_time", "symbol"]).copy()
    df = df[df["fold"].isin(OOS_FOLDS)]
    times = sorted(df["open_time"].unique())
    keep_t = set(times[::psl.HORIZON_ENTRY])
    df = df[df["open_time"].isin(keep_t)]
    fold_lookup = df.groupby("open_time")["fold"].first().to_dict()
    by_t = {t: g for t, g in df.groupby("open_time")}
    records = []
    for t in sorted(by_t.keys()):
        g = by_t[t]
        u = universe_fn(t)
        g_u = g[g["symbol"].isin(u)] if u else g.iloc[0:0]
        # Pick by realized 24h cumulative
        if t not in fwd_24h_map.index: continue
        rets_at_t = fwd_24h_map.loc[t]
        # Build symbol → 24h return map
        score_rows = []
        for _, row in g_u.iterrows():
            sym = row["symbol"]
            if sym in rets_at_t.index and not pd.isna(rets_at_t[sym]):
                score_rows.append((sym, float(rets_at_t[sym])))
        if len(score_rows) < 2 * K + 1:
            records.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "long_basket": [], "short_basket": [], "traded": False})
            continue
        score_rows.sort(key=lambda x: x[1], reverse=True)
        long_picks = sorted([s for s, _ in score_rows[:K]])
        short_picks = sorted([s for s, _ in score_rows[-K:]])
        records.append({"time": t, "fold": fold_lookup.get(t, 0),
                          "long_basket": long_picks, "short_basket": short_picks,
                          "traded": True})
    return pd.DataFrame(records)


def main():
    print("=== TRUE oracle ceiling for V3.1 (picks by realized 24h cumulative) ===\n",
          flush=True)
    apd = pd.read_parquet(APD_PATH)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    panel_syms = sorted(apd["symbol"].unique())
    listings = psl.get_listings()
    print(f"Panel: {len(apd):,} rows, {len(panel_syms)} symbols, capital ${CAPITAL:.0f}\n",
          flush=True)

    def elig_pit(b):
        if isinstance(b, pd.Timestamp): ts = b
        else: ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=psl.MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}

    fwd_4h, fwd_24h = compute_fwd_24h_return(panel_syms, psl.KLINES_DIR)

    target_t = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::psl.HORIZON_ENTRY]

    # Diagnostic: 24h cumulative spread between best-3 and worst-3 each cycle
    print("\nDiagnostic: per-cycle 24h cumulative spread between top-3 / bot-3 (full 51)",
          flush=True)
    diag_rows = []
    for t in sampled_t:
        if t not in fwd_24h.index: continue
        elig = elig_pit(t)
        r = fwd_24h.loc[t]
        r = r.dropna()
        r = r[r.index.isin(elig)]
        if len(r) < 7: continue
        sorted_r = r.sort_values(ascending=False)
        spread = (sorted_r.iloc[:3].mean() - sorted_r.iloc[-3:].mean()) * 1e4
        diag_rows.append({"time": t, "spread_bps": spread, "n_elig": len(r)})
    diag = pd.DataFrame(diag_rows)
    s = diag["spread_bps"]
    print(f"  median = {s.median():+.0f} bps  mean = {s.mean():+.0f} bps  "
          f"p25 = {s.quantile(0.25):+.0f}  p75 = {s.quantile(0.75):+.0f}", flush=True)

    # === Reference: Phase 1D actual ===
    print("\n" + "=" * 80)
    print("Reference: Phase 1D actual (model preds, gates ON)")
    print("=" * 80)
    universe_real = psl.build_rolling_ic_universe(apd, sampled_t, psl.TOP_N, elig_pit)
    records_ref = psl.run_production_protocol_save_sleeves(apd, universe_real)
    df_ref = psl.aggregate_sleeves(records_ref, fwd_4h)
    net_ref = df_ref["net_pnl_bps"].to_numpy()

    # === Oracle 24h, gates OFF, full 51 ===
    print("\n" + "=" * 80)
    print("Oracle by realized 24h cumulative, NO gates, full 51 universe")
    print("=" * 80)
    records_A = run_oracle_24h_no_gates(apd, lambda t: elig_pit(t), fwd_24h, K=psl.K)
    df_A = psl.aggregate_sleeves(records_A, fwd_4h)
    net_A = df_A["net_pnl_bps"].to_numpy()

    # === Oracle 24h, gates OFF, rolling-IC top-15 universe ===
    print("\n" + "=" * 80)
    print("Oracle by realized 24h cumulative, NO gates, rolling-IC top-15 universe")
    print("=" * 80)
    records_B = run_oracle_24h_no_gates(apd, lambda t: universe_real.get(t, set()),
                                            fwd_24h, K=psl.K)
    df_B = psl.aggregate_sleeves(records_B, fwd_4h)
    net_B = df_B["net_pnl_bps"].to_numpy()

    def report(label, net, df_v, records):
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=1000)
        total_pnl_bps = net.sum()
        total_pnl_dollars = total_pnl_bps / 1e4 * CAPITAL
        end_equity = CAPITAL + total_pnl_dollars
        max_dd_d = _max_dd(net) / 1e4 * CAPITAL
        avg_per_cycle_d = net.mean() / 1e4 * CAPITAL
        print(f"\n  {label}", flush=True)
        print(f"    cycles traded      : {records['traded'].sum()}/{len(records)}", flush=True)
        print(f"    Sharpe             : {sh:+.2f} [{lo:+.2f}, {hi:+.2f}]", flush=True)
        print(f"    avg net per cycle  : {net.mean():+.2f} bps = ${avg_per_cycle_d:+.4f}", flush=True)
        print(f"    total PnL          : {total_pnl_bps:+.0f} bps = ${total_pnl_dollars:+.2f}",
              flush=True)
        print(f"    END-EQUITY         : ${end_equity:.2f}  (from ${CAPITAL:.0f}, {total_pnl_dollars/CAPITAL*100:+.1f}%)",
              flush=True)
        print(f"    max DD             : {_max_dd(net):+.0f} bps = ${max_dd_d:+.2f}", flush=True)
        print(f"    gross/cycle        : {df_v['gross_pnl_bps'].mean():+.2f} bps", flush=True)
        print(f"    cost/cycle         : {df_v['cost_bps'].mean():+.2f} bps", flush=True)
        print(f"    turnover/cycle     : {df_v['turnover'].mean():.3f}", flush=True)
        print(f"    folds positive     : {folds_positive(df_v)}/9", flush=True)

    report("REFERENCE: Phase 1D actual", net_ref, df_ref, records_ref)
    report("A: Oracle 24h, full 51, no gates", net_A, df_A, records_A)
    report("B: Oracle 24h, rolling-IC top-15, no gates", net_B, df_B, records_B)

    print("\n" + "=" * 80)
    print(f"  CEILINGS — capital base ${CAPITAL:.0f}")
    print("=" * 80)
    for label, net in [("Phase 1D actual (model + gates)", net_ref),
                        ("Oracle 24h on full 51 (no gates)", net_A),
                        ("Oracle 24h on top-15 IC universe (no gates)", net_B)]:
        sh = _sharpe(net); tot_d = net.sum() / 1e4 * CAPITAL
        print(f"  {label:<50}: Sharpe {sh:+6.2f}, end-equity ${CAPITAL+tot_d:8.2f} ({tot_d/CAPITAL*100:+.1f}%)",
              flush=True)


if __name__ == "__main__":
    main()
