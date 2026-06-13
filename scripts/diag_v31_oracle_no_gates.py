"""Oracle ceiling with V3.1 sleeve + cost but NO production gates.

Removes conv_gate / PM_M2 / filter_refill — those gates were calibrated for
noisy model preds and strangle oracle picks. Just: K=3 long/short by realized
4h return, V3.1 6-sleeve overlay, production cost model.

Tells us: what does the V3.1 trading structure deliver if the picker was perfect?
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

spec = importlib.util.spec_from_file_location("psl", REPO / "scripts/phase_ah_sleeve.py")
psl = importlib.util.module_from_spec(spec); spec.loader.exec_module(psl)
from ml.research.alpha_v4_xs import block_bootstrap_ci

APD_PATH = REPO / "outputs/vBTC_phase1d_rolling_beta/all_predictions.parquet"
OUT = REPO / "outputs/vBTC_v31_oracle_no_gates"
OUT.mkdir(parents=True, exist_ok=True)
OOS_FOLDS = list(range(1, 10))
CAPITAL = 100.0


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(psl.CYCLES_PER_YEAR))


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def folds_positive(df_v):
    return sum(1 for _, g in df_v.groupby("fold") if _sharpe(g["net_pnl_bps"]) > 0)


def run_oracle_no_gates(apd, universe_fn, K):
    """K=3 long/short picks every 4h, NO gates. Universe = universe_fn(t)."""
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
        # use realized return as score
        g_u = g_u.dropna(subset=["return_pct"])
        if len(g_u) < 2 * K + 1:
            records.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "long_basket": [], "short_basket": [], "traded": False})
            continue
        scores = g_u["return_pct"].to_numpy()
        syms = g_u["symbol"].to_numpy()
        idx_t = np.argpartition(-scores, K - 1)[:K]
        idx_b = np.argpartition(scores, K - 1)[:K]
        nl = sorted(syms[idx_t].tolist())
        ns = sorted(syms[idx_b].tolist())
        records.append({"time": t, "fold": fold_lookup.get(t, 0),
                          "long_basket": nl, "short_basket": ns, "traded": True})
    return pd.DataFrame(records)


def main():
    print("=== Oracle on V3.1 sleeve + cost, NO production gates ===\n", flush=True)
    apd = pd.read_parquet(APD_PATH)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    panel_syms = sorted(apd["symbol"].unique())
    listings = psl.get_listings()
    print(f"Loaded {len(apd):,} rows, {len(panel_syms)} symbols, capital ${CAPITAL:.0f}\n",
          flush=True)

    def elig_pit(b):
        if isinstance(b, pd.Timestamp): ts = b
        else: ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=psl.MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}

    # fwd_rets for MTM
    print("Loading close prices...", flush=True)
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
    print(f"  ready ({time.time()-t0:.0f}s)\n", flush=True)

    target_t = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::psl.HORIZON_ENTRY]

    # Variant A: oracle no-gates, full 51
    records_A = run_oracle_no_gates(apd, lambda t: elig_pit(t), K=psl.K)
    df_A = psl.aggregate_sleeves(records_A, fwd_rets_4h)
    net_A = df_A["net_pnl_bps"].to_numpy()

    # Variant B: oracle no-gates, rolling-IC universe (PIT, using actual pred for IC)
    universe_real = psl.build_rolling_ic_universe(apd, sampled_t, psl.TOP_N, lambda b: elig_pit(b))
    records_B = run_oracle_no_gates(apd, lambda t: universe_real.get(t, set()), K=psl.K)
    df_B = psl.aggregate_sleeves(records_B, fwd_rets_4h)
    net_B = df_B["net_pnl_bps"].to_numpy()

    def report(label, net, df_v, records):
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=1000)
        total_pnl_bps = net.sum()
        total_pnl_dollars = total_pnl_bps / 1e4 * CAPITAL
        end_equity = CAPITAL + total_pnl_dollars
        max_dd_d = _max_dd(net) / 1e4 * CAPITAL
        avg_per_cycle_d = net.mean() / 1e4 * CAPITAL
        print(f"\n  {label}", flush=True)
        print(f"    cycles traded         : {records['traded'].sum()}/{len(records)}", flush=True)
        print(f"    Sharpe                : {sh:+.2f} [{lo:+.2f}, {hi:+.2f}]", flush=True)
        print(f"    avg net per cycle     : {net.mean():+.2f} bps = ${avg_per_cycle_d:+.4f}", flush=True)
        print(f"    total PnL             : {total_pnl_bps:+.0f} bps = ${total_pnl_dollars:+.2f}", flush=True)
        print(f"    end-of-OOS equity     : ${end_equity:.2f} (from ${CAPITAL:.0f})", flush=True)
        print(f"    max DD                : {_max_dd(net):+.0f} bps = ${max_dd_d:+.2f}", flush=True)
        print(f"    gross/cycle           : {df_v['gross_pnl_bps'].mean():+.2f} bps", flush=True)
        print(f"    cost/cycle            : {df_v['cost_bps'].mean():+.2f} bps", flush=True)
        print(f"    turnover/cycle        : {df_v['turnover'].mean():.3f}", flush=True)
        print(f"    folds positive        : {folds_positive(df_v)}/9", flush=True)

    report("A: oracle no-gates on full 51", net_A, df_A, records_A)
    report("B: oracle no-gates on rolling-IC top-15 universe", net_B, df_B, records_B)

    print("\n" + "=" * 80)
    print("  CEILING — what V3.1 sleeve+cost structure achieves with perfect picks")
    print("=" * 80)
    print(f"  Starting capital: ${CAPITAL:.0f}\n")
    for label, net in [("Phase 1D actual (model preds + gates)", None),  # 0.65 known
                        ("A: oracle full 51 (no gates)", net_A),
                        ("B: oracle top-15 universe (no gates)", net_B)]:
        if net is None:
            sh = 0.65; tot_d = 24.40
        else:
            sh = _sharpe(net); tot_d = net.sum() / 1e4 * CAPITAL
        end_eq = CAPITAL + tot_d
        print(f"  {label:<48}: Sharpe {sh:+6.2f}, end-equity ${end_eq:8.2f} ({tot_d/CAPITAL*100:+.1f}%)",
              flush=True)

    df_A.to_csv(OUT / "oracle_full_51_no_gates.csv", index=False)
    df_B.to_csv(OUT / "oracle_universe_no_gates.csv", index=False)


if __name__ == "__main__":
    main()
