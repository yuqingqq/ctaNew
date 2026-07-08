#!/usr/bin/env python3
"""REGIME-DISCOVERY FRAMEWORK — find PIT buckets where E[net model_edge | bucket] > MARGIN, STABLY & robustly.

Buckets can be defined by a MACRO axis (BTC-30d, hardcoded schema) OR a MODEL-OUTPUT axis (pred_gap, pred_base_std,
...), the latter being the original goal: a data-driven, model-native regime. Model-output buckets are made PIT in
the walk-forward by learning the bin EDGES on TRAIN periods only (retrospective map uses full-sample edges = diagnostic).

Outputs:
  A. RETROSPECTIVE map (full-sample) — DIAGNOSTIC ONLY, leaks, NOT actionable.
  B. WALK-FORWARD router — ACTIONABLE: buckets classified (and, for model features, binned) on PRIOR periods only,
     applied to the next, with random-matched significance and a configurable fallback.

Bucket acceptance:
  FARM        = mean net>MARGIN & stability>=STAB_THR & WORST-period>WORST_FLOOR & PERIOD-LEVEL t>MIN_T &
                >= MIN_PERIODS_STRONG (3) qualifying periods.  (actionable)
  FARM_THIN   = same but only MIN_PERIODS (2) qualifying periods — real but thin evidence (NOT auto-routed).
  AVOID       = mirror on the negative side (mean<-MARGIN & stably-negative & BEST-period<-WORST_FLOOR & period-t<-MIN_T).
  FRAGILE     = else -> sit out.
Period-level t uses ~independent per-period means (the 24h label on a 4h grid overlaps 6x, so row-level t is inflated).

Usage: python3 live/regime_discovery.py --dataset <parquet> --outdir <dir> --cost 15 [--costs 10,15,20]
       [--bucket btc30|pred_gap|pred_base_std] [--fallback flat|bear_only] [--periods p1,p2,...]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")

BUCKET_ORDER = ["bear_deep","bear_mid","bear_mild","side_down","side_flat","side_up","bull_mild","bull_hot","bull_deep"]

def btc30_bucket(x: float) -> str:
    edges = [(-np.inf,-0.20,"bear_deep"),(-0.20,-0.15,"bear_mid"),(-0.15,-0.10,"bear_mild"),(-0.10,-0.05,"side_down"),
             (-0.05,0.05,"side_flat"),(0.05,0.10,"side_up"),(0.10,0.15,"bull_mild"),(0.15,0.20,"bull_hot"),(0.20,np.inf,"bull_deep")]
    for lo,hi,nm in edges:
        if lo <= x < hi: return nm
    return "side_flat"

def assign_bucket(df: pd.DataFrame, mode: str, edges: list[float] | None = None) -> pd.Series:
    """Bucket label per cycle. mode=btc30 -> fixed 9-bin schema; mode=<model feature> -> quantile bins by `edges`
    (edges learned on TRAIN data for the PIT walk-forward). Hardened: dedupes duplicate quantile edges, degenerate
    feature -> single 'q_all' bucket, NaN feature rows -> explicit 'q_nan' bucket (never a hidden bin)."""
    if mode == "btc30":
        return df["btc_ret_30d"].apply(btc30_bucket)
    if mode not in df.columns:
        raise SystemExit(f"--bucket {mode}: column not in dataset")
    col = df[mode].astype(float)
    e = list(col.quantile([1/3, 2/3])) if edges is None else list(edges)
    e = sorted({float(x) for x in e if pd.notna(x)})            # dedupe interior edges + drop NaN edges
    bins = [-np.inf] + e + [np.inf]
    out = pd.Series("q_nan", index=df.index, dtype=object)      # NaN-feature rows -> explicit bucket
    m = col.notna()
    if m.any():
        if col[m].nunique() <= 1 or len(e) == 0:                # degenerate (constant / no usable edge) -> one bucket
            out.loc[m] = "q_all"
        else:
            labels = [f"q{i}" for i in range(len(bins) - 1)]
            out.loc[m] = pd.cut(col[m], bins=bins, labels=labels).astype(object)
    return out.astype(str)

def classify(df: pd.DataFrame, bucket_col: str, periods: list[str],
             margin=5.0, min_n=30, min_periods=2, min_periods_strong=3, min_cyc_per_period=20,
             stab_thr=0.6, worst_floor=-40.0, min_t=1.0) -> dict:
    """Score+classify each bucket on `df` (TRAIN data; `net` precomputed)."""
    out = {}
    for b, g in df.groupby(bucket_col):
        net = float(g["net"].mean()); n = len(g)
        pv = [float(g.loc[g.period == p, "net"].mean()) for p in periods if (g.period == p).sum() >= min_cyc_per_period]
        n_periods = len(pv)
        pvs = float(np.std(pv, ddof=1)) if n_periods >= 2 else np.nan
        tstat = float(np.mean(pv) / (pvs / np.sqrt(n_periods))) if (n_periods >= 2 and pvs > 0) else np.nan
        worst = float(min(pv)) if pv else np.nan; best = float(max(pv)) if pv else np.nan
        base = {"net": net, "n": int(n), "n_periods": n_periods, "worst_period": worst, "best_period": best, "tstat": tstat}
        if n < min_n or n_periods < min_periods:
            out[b] = {**base, "verdict": "FRAGILE", "reason": "insufficient-evidence", "stab": np.nan}; continue
        stab = float(np.mean([v > 0 for v in pv])); stabneg = float(np.mean([v < 0 for v in pv]))
        t_ok = (not np.isnan(tstat))
        farm_core = (net > margin and stab >= stab_thr and worst > worst_floor and t_ok and tstat > min_t)
        avoid = (net < -margin and stabneg >= stab_thr and best < -worst_floor and t_ok and tstat < -min_t)  # mirror of FARM
        if farm_core:
            v = "FARM" if n_periods >= min_periods_strong else "FARM_THIN"
        elif avoid:
            v = "AVOID"
        else:
            v = "FRAGILE"
        out[b] = {**base, "verdict": v, "stab": stab}
    return out

def random_matched(vals: np.ndarray, n_trade: int, seeds=1000):
    if n_trade <= 0 or n_trade > len(vals): return np.nan, np.nan
    tot = [vals[np.random.default_rng(s).choice(len(vals), n_trade, replace=False)].sum() for s in range(seeds)]
    return float(np.mean(tot)), float(np.quantile(tot, 0.90))

def _fmt(v, w=5, nd=2):
    return (f"%{w}.{nd}f" % v) if (v is not None and not (isinstance(v, float) and np.isnan(v))) else " -".rjust(w)

def run(dataset: Path, outdir: Path, cost: float, costs: list[float], bucket: str, fallback: str, periods_arg,
        edge_col: str = "edge_bps"):
    d = pd.read_parquet(dataset)
    # Requirements are gated BY MODE so the tool works on arbitrary model-output datasets:
    if bucket == "btc30" and "btc_ret_30d" not in d.columns:
        raise SystemExit("--bucket btc30 needs a btc_ret_30d column")
    if edge_col not in d.columns:
        raise SystemExit(f"--edge-col {edge_col}: column not in dataset")
    has_macro = "macro_regime" in d.columns                     # only needed for macro baselines / bear_only fallback
    if fallback == "bear_only" and not has_macro:
        raise SystemExit("--fallback bear_only needs a macro_regime column (or use --fallback flat)")
    all_periods_avail = set(d["period"])
    if periods_arg:
        missing = [p for p in periods_arg if p not in all_periods_avail]
        if missing: raise SystemExit(f"--periods: labels not in dataset: {missing} (available: {sorted(all_periods_avail)})")
        periods = list(periods_arg)
    elif "open_time" in d.columns:
        periods = d.groupby("period")["open_time"].min().sort_values().index.tolist()
    else:
        periods = sorted(all_periods_avail)
    print(f"[periods: {periods}] [cost={cost}, bucket={bucket}, edge={edge_col}, fallback={fallback}]")
    outdir.mkdir(parents=True, exist_ok=True)
    d["net"] = d[edge_col] - cost
    d["bucket"] = assign_bucket(d, bucket)              # full-sample edges for the RETROSPECTIVE map (diagnostic)
    ORDER = BUCKET_ORDER if bucket == "btc30" else sorted(d["bucket"].unique())

    # ---- A. RETROSPECTIVE map (full-sample; DIAGNOSTIC ONLY) ----
    retro = classify(d, "bucket", periods)
    pd.DataFrame([{"bucket": b, **retro[b]} for b in ORDER if b in retro]).to_csv(outdir / "regime_retrospective_map.csv", index=False)
    print(f"\n=== A. RETROSPECTIVE map (bucket={bucket}, COST={cost}) — DIAGNOSTIC ONLY, full-sample, NOT for routing ===")
    print(f"  {'bucket':<11s} {'n':>5s} {'net':>7s} {'worst':>7s} {'best':>7s} {'t':>5s} {'stab':>5s} {'nP':>3s} {'verdict':>9s}")
    for b in ORDER:
        if b in retro:
            c = retro[b]
            print(f"  {b:<11s} {c['n']:>5d} {c['net']:>7.1f} {c['worst_period']:>7.1f} {c['best_period']:>7.1f} {_fmt(c['tstat'],5,1)} {_fmt(c['stab'],5,2)} {c['n_periods']:>3d} {c['verdict']:>9s}")

    # ---- cost SENSITIVITY (classify once per cost) ----
    per_cost = {}
    for cc in costs:
        d["net"] = d[edge_col] - cc; per_cost[cc] = classify(d, "bucket", periods)
    d["net"] = d[edge_col] - cost
    print(f"\n  cost-sensitivity (verdict per bucket at each cost):")
    print(f"  {'bucket':<11s} " + " ".join(f"c={x:g}".rjust(9) for x in costs))
    for b in ORDER:
        cells = [per_cost[x].get(b, {}).get("verdict", "-").rjust(9) for x in costs]
        if any(cell.strip() != "-" for cell in cells): print(f"  {b:<11s} " + " ".join(cells))

    # ---- B. WALK-FORWARD router (train-only; model-feature edges learned on train => PIT) ----
    print(f"\n=== B. WALK-FORWARD router (train-only; CALENDAR nets; routes FARM [strong] only; fallback={fallback}) ===")
    print(f"  {'test':>7s} | {'FARM-net':>8s} {'randp90':>8s} {'sig?':>4s} | {'skip-bull':>9s} {'bear-only':>9s} {'always':>7s} | FARM buckets")
    wf_rows = []
    for i in range(1, len(periods)):
        tr, te = periods[:i], periods[i]
        trf, tef = d[d.period.isin(tr)].copy(), d[d.period == te].copy()
        if len(tef) < 10: continue
        edges = None if bucket == "btc30" else list(trf[bucket].quantile([1/3, 2/3]))   # PIT: edges from TRAIN
        trf["bucket"] = assign_bucket(trf, bucket, edges); tef["bucket"] = assign_bucket(tef, bucket, edges)
        fset = {b for b, c in classify(trf, "bucket", tr).items() if c["verdict"] == "FARM"}   # actionable = strong FARM
        used_fallback = not fset
        if fset:
            fset_used, fb_label = sorted(fset), sorted(fset); farm_mask = tef["bucket"].isin(fset).to_numpy()
        elif fallback == "bear_only":
            fset_used, fb_label = None, "(fallback: bear-only [V4 prior])"; farm_mask = (tef.macro_regime == "bear").to_numpy()
        else:
            fset_used, fb_label = None, "(fallback: flat)"; farm_mask = np.zeros(len(tef), dtype=bool)
        vals = tef["net"].to_numpy(); n_all = len(tef)
        farm_net = float(np.where(farm_mask, vals, 0.0).mean())
        farm_total = float(vals[farm_mask].sum()); n_trade = int(farm_mask.sum())
        _, rp90 = random_matched(vals, n_trade)
        sig = "yes" if (not np.isnan(rp90) and farm_total > rp90) else "no"
        rp90_cal = (rp90 / n_all) if not np.isnan(rp90) else np.nan
        nb = float(np.where((tef.macro_regime != "bull").to_numpy(), vals, 0.0).mean()) if has_macro else np.nan
        bo = float(np.where((tef.macro_regime == "bear").to_numpy(), vals, 0.0).mean()) if has_macro else np.nan
        al = float(vals.mean())
        print(f"  {te:>7s} | {farm_net:>8.1f} {rp90_cal:>8.1f} {sig:>4s} | {nb:>9.1f} {bo:>9.1f} {al:>7.1f} | {fb_label}")
        wf_rows.append({"test_period": te, "farm_buckets": fset_used, "fallback_used": (fallback if used_fallback else None),
                        "n_trade": n_trade, "n_test": n_all, "farm_net_cal": farm_net, "farm_total": farm_total,
                        "rand_p90_total": rp90, "rand_p90_cal": rp90_cal, "beats_random": sig == "yes",
                        "skip_bull_net": nb, "bear_only_net": bo, "always_net": al})
    def _clean(o):     # NaN -> None so the JSON is strict-parser-safe; allow_nan=False guards regressions
        if isinstance(o, float): return None if np.isnan(o) else o
        if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list): return [_clean(v) for v in o]
        return o
    with open(outdir / "regime_walkforward_route.json", "w") as f:
        json.dump(_clean(wf_rows), f, indent=1, default=str, allow_nan=False)
    print(f"\n  routes FARM[strong,>=3 periods] only; FARM_THIN reported but not auto-routed. *-net are CALENDAR.")
    print(f"  wrote {outdir/'regime_retrospective_map.csv'} + {outdir/'regime_walkforward_route.json'}")
    print("REGIMEDISCDONE")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="live/V4_GATE_MODEL_DATASET.parquet")
    ap.add_argument("--outdir", default="live/state/longtail/regime_disc")
    ap.add_argument("--cost", type=float, default=15.0)
    ap.add_argument("--costs", default="10,15,20")
    ap.add_argument("--bucket", default="btc30", help="btc30 (macro) | pred_gap | pred_base_std | <any model-output col>")
    ap.add_argument("--fallback", choices=["flat","bear_only"], default="flat")
    ap.add_argument("--periods", default="")
    ap.add_argument("--edge-col", default="edge_bps", help="edge_bps (L/S) | long_edge_bps | short_edge_bps (per-leg; use ~half cost)")
    a = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    ds = Path(a.dataset); ds = ds if ds.is_absolute() else root / ds
    od = Path(a.outdir); od = od if od.is_absolute() else root / od
    costs = [float(x) for x in a.costs.split(",") if x]
    periods = [p for p in a.periods.split(",") if p] or None
    run(ds, od, a.cost, costs, a.bucket, a.fallback, periods, edge_col=a.edge_col)

if __name__ == "__main__":
    main()
