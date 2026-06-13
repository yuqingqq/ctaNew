"""V3.1 head-to-head: with sym_id (production) vs without sym_id (retrained).

Loads two prediction parquets (same training pipeline; only sym_id removed):
  A: outputs/vBTC_audit_panel/all_predictions.parquet            (production)
  B: outputs/vBTC_audit_panel_no_sym_id/all_predictions.parquet  (no sym_id)

Runs identical V3.1 stack on each: rolling-IC universe top-15, K=3, conv_gate,
PM_M2, filter_refill, flat_real, 6-sleeve overlay. Reports Sharpe / turnover /
cost / per-fold profile / boundary universe overlap.
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

OUT = REPO / "outputs/vBTC_no_sym_id_test"
OUT.mkdir(parents=True, exist_ok=True)

APD_PROD = REPO / "outputs/vBTC_audit_panel/all_predictions.parquet"
APD_NOSI = REPO / "outputs/vBTC_audit_panel_no_sym_id/all_predictions.parquet"


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
    print(f"\n=== Variant: {label} ===", flush=True)
    apd = pd.read_parquet(apd_path)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)

    def elig(b):
        ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=psl.MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}

    target_t = sorted(apd[apd["fold"].isin(psl.OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::psl.HORIZON_ENTRY]
    t0 = time.time()
    universe = psl.build_rolling_ic_universe(apd, sampled_t, psl.TOP_N, elig)
    records = psl.run_production_protocol_save_sleeves(apd, universe)
    df_v = psl.aggregate_sleeves(records, fwd_rets_4h)
    print(f"  built universe + protocol + sleeves ({time.time()-t0:.0f}s, "
          f"{records['traded'].sum()}/{len(records)} traded)", flush=True)
    net = df_v["net_pnl_bps"].to_numpy()
    sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=1000)

    # Boundary universe stability
    boundaries = sorted(set(
        tuple(sorted(u)) for u in universe.values()))
    bsets = [set(t) for t in sorted({tuple(sorted(u)) for u in universe.values()})]
    # Cleaner: walk unique boundary sets in time order
    seen_b = []
    for t in sampled_t:
        u = universe.get(t, set())
        if not seen_b or u != seen_b[-1]: seen_b.append(u)
    overlap_pairs = []
    for i in range(1, len(seen_b)):
        a, b = seen_b[i-1], seen_b[i]
        if not a or not b: continue
        overlap_pairs.append(len(a & b) / max(len(a | b), 1))

    return {
        "label": label,
        "sharpe": sh, "sh_lo": lo, "sh_hi": hi,
        "totPnL": net.sum(), "maxDD": _max_dd(net),
        "gross_avg": df_v["gross_pnl_bps"].mean(),
        "cost_avg": df_v["cost_bps"].mean(),
        "net_avg": df_v["net_pnl_bps"].mean(),
        "cost_over_gross": df_v["cost_bps"].mean() / max(abs(df_v["gross_pnl_bps"].mean()), 1e-6),
        "turnover_avg": df_v["turnover"].mean(),
        "folds_pos": folds_positive(df_v),
        "n_traded": int(records["traded"].sum()),
        "boundary_universe_overlap_mean": float(np.mean(overlap_pairs)) if overlap_pairs else np.nan,
        "n_boundary_changes": len(seen_b) - 1,
        "df_v": df_v,
        "universe": universe,
    }, net


def main():
    print("=== V3.1: with sym_id vs without sym_id ===\n", flush=True)
    # build fwd_rets once
    apd = pd.read_parquet(APD_PROD)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    panel_syms = sorted(apd["symbol"].unique())
    listings = psl.get_listings()
    print(f"Loading close prices for 4h MtM...", flush=True)
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
    print(f"  close_wide {close_wide.shape} ({time.time()-t0:.0f}s)", flush=True)

    rA, netA = run_variant(APD_PROD, fwd_rets_4h, listings, panel_syms, "with_sym_id_production")
    rB, netB = run_variant(APD_NOSI, fwd_rets_4h, listings, panel_syms, "no_sym_id_retrained")

    print("\n" + "=" * 90)
    print("  HEAD-TO-HEAD")
    print("=" * 90)
    cols = ["sharpe", "sh_lo", "sh_hi", "totPnL", "maxDD", "gross_avg", "cost_avg",
            "net_avg", "cost_over_gross", "turnover_avg", "folds_pos", "n_traded",
            "boundary_universe_overlap_mean", "n_boundary_changes"]
    rows = []
    for r in (rA, rB):
        d = {"variant": r["label"]}
        for c in cols: d[c] = r[c]
        rows.append(d)
    summary = pd.DataFrame(rows).set_index("variant")
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:+.3f}")
    print(summary.T.to_string(), flush=True)

    print("\n=== DELTAS (no_sym_id − production) ===", flush=True)
    print(f"  Δ Sharpe         : {rB['sharpe']-rA['sharpe']:+.3f}", flush=True)
    print(f"  Δ totPnL         : {rB['totPnL']-rA['totPnL']:+.0f} bps", flush=True)
    print(f"  Δ maxDD          : {rB['maxDD']-rA['maxDD']:+.0f} bps", flush=True)
    print(f"  Δ gross/cycle    : {rB['gross_avg']-rA['gross_avg']:+.2f} bps", flush=True)
    print(f"  Δ cost/cycle     : {rB['cost_avg']-rA['cost_avg']:+.2f} bps", flush=True)
    print(f"  Δ turnover/cycle : {rB['turnover_avg']-rA['turnover_avg']:+.3f}", flush=True)
    print(f"  Δ folds positive : {rB['folds_pos']-rA['folds_pos']:+d}", flush=True)
    print(f"  Δ universe overlap (boundary persistence): "
          f"{rB['boundary_universe_overlap_mean']-rA['boundary_universe_overlap_mean']:+.3f}", flush=True)

    # Per-fold breakdown
    print("\n=== PER-FOLD Sharpe ===", flush=True)
    for fid in psl.OOS_FOLDS:
        ga = rA["df_v"][rA["df_v"]["fold"] == fid]["net_pnl_bps"].to_numpy()
        gb = rB["df_v"][rB["df_v"]["fold"] == fid]["net_pnl_bps"].to_numpy()
        sha = _sharpe(ga); shb = _sharpe(gb)
        print(f"  fold {fid}: prod {sha:+.2f}  vs  no_sym_id {shb:+.2f}  (Δ {shb-sha:+.2f})",
              flush=True)

    # Universe set overlap between the two variants per boundary
    print("\n=== Universe set agreement (sym_id vs no_sym_id) at each rebalance boundary ===",
          flush=True)
    apd_p = pd.read_parquet(APD_PROD)
    apd_p["open_time"] = pd.to_datetime(apd_p["open_time"], utc=True)
    target_t = sorted(apd_p[apd_p["fold"].isin(psl.OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::psl.HORIZON_ENTRY]
    boundaries_seen = []
    for t in sampled_t:
        ua = rA["universe"].get(t, set())
        ub = rB["universe"].get(t, set())
        if not ua or not ub: continue
        if boundaries_seen and ua == boundaries_seen[-1]["ua"] and ub == boundaries_seen[-1]["ub"]:
            continue
        boundaries_seen.append({"t": t, "ua": ua, "ub": ub})
    for b in boundaries_seen:
        ua, ub = b["ua"], b["ub"]
        jacc = len(ua & ub) / max(len(ua | ub), 1)
        print(f"  {b['t']}: |intersect|={len(ua&ub)}/15  Jaccard={jacc:.2f}", flush=True)

    rA["df_v"].to_csv(OUT / "with_sym_id.csv", index=False)
    rB["df_v"].to_csv(OUT / "no_sym_id.csv", index=False)
    print(f"\nSaved to {OUT}/", flush=True)


if __name__ == "__main__":
    main()
