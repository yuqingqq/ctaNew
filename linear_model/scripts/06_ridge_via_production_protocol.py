"""Step 6: Run Ridge predictions through the EXACT production protocol
(conv_gate + PM_M + filter_refill + V3.1 6-sleeve).

Compares apples-to-apples vs LGBM clean-PIT (+0.88 from step 5) on the
same pipeline. If Ridge is close to +0.88, model class isn't the bottleneck.
If Ridge is much lower, LGBM's tree-based non-linearity is doing real work
here.
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

PREDS = REPO / "linear_model/results/predictions.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
OUT = REPO / "linear_model/results"

OOS_FOLDS = list(range(1, 10))
MIN_HISTORY_DAYS = 60
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
        sw = 1.0 / psl.N_SLEEVES
        for sl in sleeve_queue:
            nL, nS = len(sl["longs"]), len(sl["shorts"])
            if nL == 0 or nS == 0: continue
            for s in sl["longs"]: target_weights[s] += sw * (1.0/nL)
            for s in sl["shorts"]: target_weights[s] -= sw * (1.0/nS)
        gross = 0.0
        if t in alpha_wide.index:
            a = alpha_wide.loc[t]
            for sym, w in prev_weights.items():
                if sym in a.index and not pd.isna(a[sym]):
                    gross += w * a[sym] * 1e4
        syms = set(target_weights.keys()) | set(prev_weights.keys())
        abs_d = sum(abs(target_weights.get(s,0)-prev_weights.get(s,0)) for s in syms)
        cost = abs_d * psl.COST_PER_UNIT_ABS_DELTA
        rows.append({"time":t, "fold":fold, "gross_pnl_bps":gross, "cost_bps":cost,
                     "net_pnl_bps":gross-cost, "turnover":abs_d})
        prev_weights = dict(target_weights)
    return pd.DataFrame(rows)


def main():
    print("=== Step 6: Ridge predictions through production protocol ===\n", flush=True)
    t0 = time.time()

    apd = pd.read_parquet(PREDS)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    apd = apd.rename(columns={"pred_z": "pred"})
    apd["alpha_A"] = apd["alpha_beta"]
    # Production protocol needs return_pct (4h fwd return) for picks_hist accounting.
    # Merge from base panel.
    base = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                           columns=["symbol", "open_time", "return_pct"])
    base["open_time"] = pd.to_datetime(base["open_time"], utc=True)
    apd = apd.merge(base, on=["symbol", "open_time"], how="left")
    print(f"  Ridge predictions: {len(apd):,} rows "
          f"(return_pct merged, NaN: {apd['return_pct'].isna().sum():,})", flush=True)

    listings = get_listings()
    panel_syms = set(apd["symbol"].unique())
    for s, t in apd.groupby("symbol")["open_time"].min().items():
        if s not in listings:
            t = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[s] = t

    def elig_pit(b):
        if isinstance(b, pd.Timestamp): ts = b
        else: ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=psl.MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}

    target_t = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::psl.HORIZON_ENTRY]
    universe = psl.build_rolling_ic_universe(apd, sampled_t, psl.TOP_N, elig_pit)
    print(f"  Universe: {len(universe)} time points "
          f"({time.time()-t0:.0f}s elapsed)", flush=True)

    # Full production protocol: conv_gate + PM_M + filter_refill
    records = psl.run_production_protocol_save_sleeves(apd, universe)
    alpha_wide = apd.pivot_table(index="open_time", columns="symbol",
                                   values="alpha_A", aggfunc="first").sort_index()
    df_v = aggregate_alpha(records, alpha_wide)
    net = df_v["net_pnl_bps"].to_numpy()
    sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=1000)
    total_d = net.sum() / 1e4 * CAPITAL
    end_eq = CAPITAL + total_d

    print(f"\n{'='*70}", flush=True)
    print(f"  RIDGE through PRODUCTION protocol (conv_gate + PM_M + filter_refill)",
          flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Sharpe          : {sh:+.2f} [{lo:+.2f},{hi:+.2f}]", flush=True)
    print(f"  end-equity $100 : ${end_eq:.2f}", flush=True)
    print(f"  totPnL          : {net.sum():+.0f} bps", flush=True)
    print(f"  maxDD           : {_max_dd(net):+.0f} bps", flush=True)
    print(f"  gross/cycle     : {df_v['gross_pnl_bps'].mean():+.2f} bps", flush=True)
    print(f"  cost/cycle      : {df_v['cost_bps'].mean():+.2f} bps", flush=True)
    print(f"  folds positive  : {folds_positive(df_v)}/9", flush=True)
    print(f"  traded cycles   : {records['traded'].sum()}/{len(records)} "
          f"({records['traded'].mean()*100:.0f}%)", flush=True)
    df_v.to_csv(OUT / "ridge_production_protocol_v31.csv", index=False)

    # Per-fold
    print(f"\n  Per-fold Sharpe:", flush=True)
    for fold in range(1, 10):
        g = df_v[df_v["fold"]==fold]
        print(f"    fold {fold}: Sharpe={_sharpe(g['net_pnl_bps']):+.2f} "
              f"(totPnL {g['net_pnl_bps'].sum():+.0f} bps)", flush=True)

    print(f"\n  References:", flush=True)
    print(f"    LGBM shift(1) leaky [original production +0.74]: Sharpe +0.74", flush=True)
    print(f"    LGBM shift(1) leaky [my rerun]:                 Sharpe -1.30", flush=True)
    print(f"    LGBM shift(49) clean-PIT [my rerun]:            Sharpe +0.88", flush=True)
    print(f"    Ridge shift(49) threshold-bps best:             Sharpe -0.46", flush=True)
    print(f"  Ridge production protocol (this run):              Sharpe {sh:+.2f}", flush=True)
    print(f"\n  Total: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
