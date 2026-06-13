"""V3.1 three-way: production (sym_id numeric) vs no sym_id vs sym_id categorical.

Each variant uses identical V3.1 stack on its own predictions parquet.
"""
from __future__ import annotations
import sys, time, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location("psl", REPO / "scripts/phase_ah_sleeve.py")
psl = importlib.util.module_from_spec(spec); spec.loader.exec_module(psl)
from ml.research.alpha_v4_xs import block_bootstrap_ci

OUT = REPO / "outputs/vBTC_sym_id_three_way"
OUT.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    ("with_sym_id_numeric_PROD", REPO / "outputs/vBTC_audit_panel/all_predictions.parquet"),
    ("no_sym_id",                REPO / "outputs/vBTC_audit_panel_no_sym_id/all_predictions.parquet"),
    ("sym_id_categorical",       REPO / "outputs/vBTC_audit_panel_sym_id_cat/all_predictions.parquet"),
    ("sym_id_permuted",          REPO / "outputs/vBTC_audit_panel_sym_id_permuted/all_predictions.parquet"),
]


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(psl.CYCLES_PER_YEAR))


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def folds_positive(df_v):
    return sum(1 for _, g in df_v.groupby("fold") if _sharpe(g["net_pnl_bps"]) > 0)


def run_variant(apd_path, fwd_rets_4h, listings, panel_syms, label):
    print(f"\n--- {label} ---", flush=True)
    apd = pd.read_parquet(apd_path)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)

    def elig(b):
        ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=psl.MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}

    target_t = sorted(apd[apd["fold"].isin(psl.OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::psl.HORIZON_ENTRY]
    universe = psl.build_rolling_ic_universe(apd, sampled_t, psl.TOP_N, elig)
    records = psl.run_production_protocol_save_sleeves(apd, universe)
    df_v = psl.aggregate_sleeves(records, fwd_rets_4h)
    net = df_v["net_pnl_bps"].to_numpy()
    sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=1000)

    cyc_ic = apd.dropna(subset=["alpha_A"]).groupby("open_time").apply(
        lambda g: g["pred"].rank().corr(g["alpha_A"].rank()) if len(g) >= 5 else np.nan
    ).dropna()
    return {
        "label": label,
        "sharpe": sh, "sh_lo": lo, "sh_hi": hi,
        "totPnL": net.sum(), "maxDD": _max_dd(net),
        "gross_avg": df_v["gross_pnl_bps"].mean(),
        "cost_avg": df_v["cost_bps"].mean(),
        "net_avg": df_v["net_pnl_bps"].mean(),
        "turnover_avg": df_v["turnover"].mean(),
        "folds_pos": folds_positive(df_v),
        "n_traded": int(records["traded"].sum()),
        "per_cycle_ic_mean": cyc_ic.mean(),
        "per_cycle_ic_median": cyc_ic.median(),
        "df_v": df_v,
        "universe": universe,
    }


def main():
    print("=== V3.1 three-way: sym_id encoding test ===\n", flush=True)
    apd = pd.read_parquet(VARIANTS[0][1])
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    panel_syms = sorted(apd["symbol"].unique())
    listings = psl.get_listings()
    print(f"Loading close prices...", flush=True)
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

    results = []
    for label, path in VARIANTS:
        r = run_variant(path, fwd_rets_4h, listings, panel_syms, label)
        results.append(r)

    print("\n" + "=" * 100)
    print("  HEAD-TO-HEAD")
    print("=" * 100)
    cols = ["per_cycle_ic_mean", "per_cycle_ic_median",
            "sharpe", "sh_lo", "sh_hi", "totPnL", "maxDD",
            "gross_avg", "cost_avg", "net_avg", "turnover_avg",
            "folds_pos", "n_traded"]
    rows = []
    for r in results:
        d = {"variant": r["label"]}
        for c in cols: d[c] = r[c]
        rows.append(d)
    summary = pd.DataFrame(rows).set_index("variant")
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: f"{x:+.3f}")
    print(summary.T.to_string(), flush=True)

    # Per-fold
    print("\n=== PER-FOLD Sharpe ===", flush=True)
    print(f"{'fold':>5}  " + "  ".join(f"{r['label']:>26}" for r in results), flush=True)
    for fid in psl.OOS_FOLDS:
        cells = []
        for r in results:
            g = r["df_v"][r["df_v"]["fold"] == fid]["net_pnl_bps"].to_numpy()
            cells.append(f"{_sharpe(g):+26.2f}")
        print(f"{fid:>5}  " + "  ".join(cells), flush=True)


if __name__ == "__main__":
    main()
