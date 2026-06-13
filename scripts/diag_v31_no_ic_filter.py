"""V3.1 with vs without IC universe filter — head-to-head, no retrain.

Reuses existing audit-panel predictions (51-sym WINNER_21 model) and the V3.1
sleeve overlay machinery. Only knob that changes: the universe filter step.

Variant A — production:  build_rolling_ic_universe(..., top_n=15)
Variant B — all_eligible: build_rolling_ic_universe(..., top_n=None)
  (returns the FULL eligible set per boundary, no IC ranking)

Both run identical K=3 picks + conv_gate + PM_M2 + flat_real + V3.1 6-sleeve overlay.
Reports Sharpe + turnover + cost/gross + folds positive.

If all_eligible beats or matches top_n=15 here, the IC filter is value-negative
under fair conditions. If all_eligible loses, the filter is doing real work that
my naked top-K/bot-K placebo missed.
"""
from __future__ import annotations
import sys, time, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location(
    "psl", REPO / "scripts/phase_ah_sleeve.py")
psl = importlib.util.module_from_spec(spec); spec.loader.exec_module(psl)
from ml.research.alpha_v4_xs import block_bootstrap_ci

OUT = REPO / "outputs/vBTC_no_ic_filter"
OUT.mkdir(parents=True, exist_ok=True)


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(psl.CYCLES_PER_YEAR))


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def folds_positive(df_v):
    return sum(1 for _, g in df_v.groupby("fold") if _sharpe(g["net_pnl_bps"]) > 0)


def run_variant(apd, sampled_t, eligibility_at, fwd_rets_4h, top_n, label):
    print(f"\n=== Variant: {label} (top_n={top_n}) ===", flush=True)
    t0 = time.time()
    universe = psl.build_rolling_ic_universe(apd, sampled_t, top_n, eligibility_at)
    sizes = [len(u) for u in universe.values()]
    print(f"  universe sizes per cycle — min/median/max: "
          f"{min(sizes)}/{int(np.median(sizes))}/{max(sizes)}", flush=True)
    records = psl.run_production_protocol_save_sleeves(apd, universe)
    n_trade = int(records["traded"].sum())
    print(f"  traded {n_trade}/{len(records)} cycles ({time.time()-t0:.0f}s)", flush=True)
    df_v = psl.aggregate_sleeves(records, fwd_rets_4h)
    net = df_v["net_pnl_bps"].to_numpy()
    sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=1000)
    return {
        "label": label,
        "top_n": top_n,
        "sharpe": sh, "sh_lo": lo, "sh_hi": hi,
        "totPnL": net.sum(),
        "maxDD": _max_dd(net),
        "gross_avg": df_v["gross_pnl_bps"].mean(),
        "cost_avg": df_v["cost_bps"].mean(),
        "net_avg": df_v["net_pnl_bps"].mean(),
        "cost_over_gross": df_v["cost_bps"].mean() / max(abs(df_v["gross_pnl_bps"].mean()), 1e-6),
        "turnover_avg": df_v["turnover"].mean(),
        "gross_exp_avg": df_v["gross_exposure"].mean(),
        "folds_pos": folds_positive(df_v),
        "n_traded": n_trade,
        "df_v": df_v,
        "records": records,
    }, net


def main():
    print("=== Phase: V3.1 with vs without IC universe filter ===\n", flush=True)
    apd = pd.read_parquet(psl.APD_PATH)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    panel_syms = sorted(apd["symbol"].unique())

    listings = psl.get_listings()
    def eligibility_at(b):
        ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=psl.MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}

    target_t = sorted(apd[apd["fold"].isin(psl.OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::psl.HORIZON_ENTRY]

    print("Loading close prices for 4h MtM...", flush=True)
    t0 = time.time()
    frames = []
    for sym in panel_syms:
        sd = psl.KLINES_DIR / sym / "5m"
        if not sd.exists(): continue
        files = sorted(sd.glob("*.parquet"))
        if not files: continue
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
    print(f"  close_wide {close_wide.shape}, fwd_rets ready ({time.time()-t0:.0f}s)", flush=True)

    rA, netA = run_variant(apd, sampled_t, eligibility_at, fwd_rets_4h,
                              top_n=15, label="production_top15_IC")
    rB, netB = run_variant(apd, sampled_t, eligibility_at, fwd_rets_4h,
                              top_n=None, label="all_eligible_no_IC_filter")

    print("\n" + "=" * 90)
    print("  HEAD-TO-HEAD")
    print("=" * 90)
    cols = ["sharpe", "sh_lo", "sh_hi", "totPnL", "maxDD", "gross_avg", "cost_avg",
            "net_avg", "cost_over_gross", "turnover_avg", "gross_exp_avg",
            "folds_pos", "n_traded"]
    rows = []
    for r in (rA, rB):
        d = {"variant": r["label"]}
        for c in cols: d[c] = r[c]
        rows.append(d)
    summary = pd.DataFrame(rows).set_index("variant")
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:+.3f}")
    print(summary.T.to_string(), flush=True)

    print("\n=== DELTAS (all_eligible − production) ===", flush=True)
    print(f"  Δ Sharpe         : {rB['sharpe']-rA['sharpe']:+.3f}", flush=True)
    print(f"  Δ totPnL         : {rB['totPnL']-rA['totPnL']:+.0f} bps", flush=True)
    print(f"  Δ maxDD          : {rB['maxDD']-rA['maxDD']:+.0f} bps", flush=True)
    print(f"  Δ turnover/cycle : {rB['turnover_avg']-rA['turnover_avg']:+.3f}", flush=True)
    print(f"  Δ cost/cycle     : {rB['cost_avg']-rA['cost_avg']:+.2f} bps", flush=True)
    print(f"  Δ gross/cycle    : {rB['gross_avg']-rA['gross_avg']:+.2f} bps", flush=True)
    print(f"  Δ cost/gross     : {rB['cost_over_gross']-rA['cost_over_gross']:+.1%}", flush=True)

    # Paired diff CI
    n_min = min(len(netA), len(netB))
    diff = netB[:n_min] - netA[:n_min]
    mean_d = diff.mean()
    lo_d, hi_d = np.percentile([np.random.choice(diff, n_min).mean()
                                   for _ in range(2000)], [2.5, 97.5])
    print(f"  Paired Δ net/cycle: {mean_d:+.2f} bps, 95% CI [{lo_d:+.2f}, {hi_d:+.2f}]",
          flush=True)

    rA["df_v"].to_csv(OUT / "production_top15_IC.csv", index=False)
    rB["df_v"].to_csv(OUT / "all_eligible_no_IC_filter.csv", index=False)
    print(f"\nSaved to {OUT}/", flush=True)


if __name__ == "__main__":
    main()
