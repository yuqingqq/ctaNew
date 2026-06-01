"""X119 evaluation harness — contract gates G2/G4/G5/G6 from per-cycle parquets.

Reads X119_percycle_{HL70,S44}.parquet (cols: open_time, fold, pnl_base,
pnl_throttle, scale) emitted at 4.5 bps by the implementation. G8 (cost levels)
comes from re-running the backtest stdout; here we focus on the gates that need
the per-cycle series.

Metric conventions match X119 (ann = sqrt(6*365), maxDD on cumulative bps equity,
Calmar = annualized return bps / |maxDD|).
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
OUT = REPO/"research/convexity_portable_2026-05-20/results"
SEEDS = 500
RNG = np.random.default_rng(20260525)


def ann(x):
    x = pd.Series(x).dropna()
    return x.mean()/x.std()*np.sqrt(6*365) if len(x) > 2 and x.std() > 0 else np.nan


def stats(pnl):
    p = pd.Series(pnl).dropna()
    pb = p*1e4
    eq = pb.cumsum()
    dd = eq - eq.cummax()
    mdd = dd.min()
    annr = pb.mean()*6*365
    cal = (annr/abs(mdd)) if (mdd < 0 and np.isfinite(mdd)) else np.nan
    return {"sharpe": ann(p), "maxDD": mdd, "calmar": cal,
            "totPnL": eq.iloc[-1] if len(eq) else np.nan, "pct_pos": (pb > 0).mean()*100}


def g4_placebo(df, n_seeds=SEEDS):
    """Matched-magnitude random-timing placebo.
    Real throttle: pnl_throttle = pnl_base * scale (per cycle, since scale<=1 applied to net).
    Placebo: take the SAME multiset of `scale` values, randomly permute timing across cycles,
    apply to pnl_base, recompute Calmar/maxDD. Real must rank >= p95 to show TIMING matters
    (not just average de-lever level).
    """
    base = df["pnl_base"].to_numpy()
    scale = df["scale"].to_numpy()
    real_thr = df["pnl_throttle"].to_numpy()
    # sanity: throttle should equal base*scale (the throttle scales net linearly -> pnl gross part,
    # but cost also scales. So pnl_throttle != exactly base*scale. Use the actual throttle series for real.)
    real = stats(real_thr)
    cals = np.empty(n_seeds); mdds = np.empty(n_seeds); shs = np.empty(n_seeds); pnls = np.empty(n_seeds)
    for i in range(n_seeds):
        perm = RNG.permutation(len(scale))
        sh_scale = scale[perm]
        placebo_pnl = base * sh_scale   # apply shuffled scale to the (gross+cost) base per-cycle pnl
        s = stats(placebo_pnl)
        cals[i] = s["calmar"]; mdds[i] = s["maxDD"]; shs[i] = s["sharpe"]; pnls[i] = s["totPnL"]
    def pct(real_val, dist, higher_better=True):
        d = dist[np.isfinite(dist)]
        if higher_better:
            return (d < real_val).mean()*100
        return (d > real_val).mean()*100  # for maxDD, less negative (higher) is better -> same as higher
    return {
        "real": real,
        "calmar_pctile": (cals[np.isfinite(cals)] < real["calmar"]).mean()*100,
        "maxDD_pctile": (mdds < real["maxDD"]).mean()*100,  # real maxDD higher (less neg) = better
        "calmar_dist": (np.nanmedian(cals), np.nanpercentile(cals, 95), np.nanmax(cals)),
        "maxDD_dist": (np.median(mdds), np.percentile(mdds, 95), np.max(mdds)),
        "placebo_mean_scale": scale.mean(),
    }


def g5_lofo(df):
    """Per-fold DD improvement + LOFO on total-PnL diff and on aggregate maxDD."""
    folds = sorted(df["fold"].unique())
    rows = []
    n_better = 0; n_eval = 0
    for f in folds:
        m = df["fold"] == f
        if m.sum() < 3: continue
        b = df.loc[m, "pnl_base"].to_numpy(); t = df.loc[m, "pnl_throttle"].to_numpy()
        sb = stats(b); st = stats(t)
        impr = (1-abs(st["maxDD"])/abs(sb["maxDD"]))*100 if sb["maxDD"] < 0 else np.nan
        better = np.isfinite(impr) and impr > 0
        if np.isfinite(impr):
            n_eval += 1
            if better: n_better += 1
        rows.append((f, sb["maxDD"], st["maxDD"], impr, sb["sharpe"], st["sharpe"]))
    # LOFO: aggregate maxDD-improvement and totPnL-diff dropping one fold at a time
    full_b = stats(df["pnl_base"]); full_t = stats(df["pnl_throttle"])
    full_dd_impr = (1-abs(full_t["maxDD"])/abs(full_b["maxDD"]))*100
    lofo = []
    for f in folds:
        keep = df[df["fold"] != f]
        if len(keep) < 3: continue
        sb = stats(keep["pnl_base"]); st = stats(keep["pnl_throttle"])
        dd_impr = (1-abs(st["maxDD"])/abs(sb["maxDD"]))*100 if sb["maxDD"] < 0 else np.nan
        cal_diff = st["calmar"] - sb["calmar"]
        lofo.append((f, dd_impr, cal_diff, st["sharpe"]-sb["sharpe"]))
    return rows, n_better, n_eval, full_dd_impr, lofo


def g6_paired_ci(df, n_boot=2000):
    """Block-bootstrap by fold of paired per-cycle diff (throttle - base).
    Report Sharpe-diff CI and totPnL-diff CI."""
    folds = sorted(df["fold"].unique())
    fold_groups = {f: df[df["fold"] == f] for f in folds}
    sh_diffs = np.empty(n_boot); pnl_diffs = np.empty(n_boot)
    for i in range(n_boot):
        samp = pd.concat([fold_groups[RNG.choice(folds)] for _ in folds], ignore_index=True)
        sh_diffs[i] = stats(samp["pnl_throttle"])["sharpe"] - stats(samp["pnl_base"])["sharpe"]
        pnl_diffs[i] = (samp["pnl_throttle"].sum() - samp["pnl_base"].sum())*1e4
    return {
        "sharpe_diff_ci": (np.nanpercentile(sh_diffs, 2.5), np.nanpercentile(sh_diffs, 97.5), np.nanmean(sh_diffs)),
        "pnl_diff_ci": (np.percentile(pnl_diffs, 2.5), np.percentile(pnl_diffs, 97.5), np.mean(pnl_diffs)),
    }


def run(label):
    df = pd.read_parquet(OUT/f"X119_percycle_{label}.parquet")
    print(f"\n{'='*70}\n{label}  (n={len(df)} cycles, folds {sorted(df.fold.unique())})\n{'='*70}")
    sb = stats(df["pnl_base"]); st = stats(df["pnl_throttle"])
    print("--- G2 in-sample @4.5bps ---")
    print(f"  {'arm':>10}{'Sharpe':>9}{'maxDD':>9}{'Calmar':>9}{'totPnL':>10}{'%pos':>7}")
    for arm, s in [("base", sb), ("throttle", st)]:
        print(f"  {arm:>10}{s['sharpe']:>+9.2f}{s['maxDD']:>+9.0f}{s['calmar']:>+9.2f}{s['totPnL']:>+10.0f}{s['pct_pos']:>7.1f}")
    dd_red = (1-abs(st['maxDD'])/abs(sb['maxDD']))*100
    print(f"  Δ Sharpe {st['sharpe']-sb['sharpe']:+.2f} | Δ Calmar {st['calmar']-sb['calmar']:+.2f} | DD reduction {dd_red:+.1f}%")

    print("--- G4 matched random-timing placebo (500 seeds) ---")
    g4 = g4_placebo(df)
    cm, c95, cmax = g4["calmar_dist"]; dm, d95, dmax = g4["maxDD_dist"]
    print(f"  real Calmar {g4['real']['calmar']:+.2f} -> placebo median {cm:+.2f} p95 {c95:+.2f} max {cmax:+.2f}  => REAL PCTILE {g4['calmar_pctile']:.0f}")
    print(f"  real maxDD {g4['real']['maxDD']:+.0f} -> placebo median {dm:+.0f} p95(best) {d95:+.0f} max(best) {dmax:+.0f}  => REAL PCTILE {g4['maxDD_pctile']:.0f}")
    print(f"  (placebo uses same scale multiset, mean scale {g4['placebo_mean_scale']:.3f}, shuffled timing)")

    print("--- G5 per-fold DD + LOFO ---")
    rows, nb, ne, full_impr, lofo = g5_lofo(df)
    print(f"  {'fold':>5}{'baseDD':>10}{'thrDD':>10}{'DDimpr%':>9}{'baseSh':>8}{'thrSh':>8}")
    for f, bdd, tdd, impr, bsh, tsh in rows:
        print(f"  {f:>5}{bdd:>+10.0f}{tdd:>+10.0f}{impr:>+9.1f}{bsh:>+8.2f}{tsh:>+8.2f}")
    print(f"  DD improved in {nb}/{ne} folds; full-sample DD impr {full_impr:+.1f}%")
    print("  LOFO (drop one fold -> aggregate DD impr%, Calmar diff, Sharpe diff):")
    for f, dd_impr, cal_d, sh_d in lofo:
        print(f"    drop f{f}: DDimpr {dd_impr:+.1f}%  ΔCalmar {cal_d:+.2f}  ΔSharpe {sh_d:+.2f}")

    print("--- G6 block-bootstrap paired CI (2000 boots by fold) ---")
    g6 = g6_paired_ci(df)
    sl, su, sm = g6["sharpe_diff_ci"]; pl, pu, pm = g6["pnl_diff_ci"]
    print(f"  Sharpe-diff (thr-base) mean {sm:+.2f} CI95 [{sl:+.2f}, {su:+.2f}]  {'crosses 0' if sl<0<su else 'clears 0'}")
    print(f"  totPnL-diff (thr-base) mean {pm:+.0f} CI95 [{pl:+.0f}, {pu:+.0f}]  {'crosses 0' if pl<0<pu else 'clears 0'}")
    return {"label": label, "base": sb, "throttle": st, "dd_red": dd_red, "g4": g4,
            "folds_better": nb, "folds_eval": ne, "lofo": lofo, "g6": g6}


if __name__ == "__main__":
    import json
    res = {}
    for u in ["HL70", "S44"]:
        res[u] = run(u)
    # serialize a compact summary
    def clean(o):
        if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [clean(x) for x in o]
        if isinstance(o, (np.floating, np.integer)): return float(o)
        return o
    Path(OUT/"X119_eval_gates_summary.json").write_text(json.dumps(clean(res), indent=1, default=str))
    print("\nwrote X119_eval_gates_summary.json")
