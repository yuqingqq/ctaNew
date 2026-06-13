"""Step 71: investability battery on the new best variant —
per-symbol self-standardized + mean-rev + NO IC-subset (all-eligible) +
PIT-clean trail_ic (embargo) + de-concentrated. The decider; no pre-judgment.

  A. COST-STRESS: drop-2 all-eligible PIT at COST 1x/2x/3x (eff RT 9/18/27bps).
     all-eligible trades the whole universe → turnover-heavy → cost is the
     prime risk; per-cycle is net of only idealized ~9bps.
  B. ITERATIVE DROP (Step-66 analog, the test that killed every pre-self-std
     variant by drop-2): each iter drop the top-2 gross contributors, RETRAIN
     per-symbol, re-eval all-eligible PIT; stop on collapse / N<30 / iter==5.
     Does the edge keep surviving past drop-2, or relocate+die?
  C. PLACEBO on drop-2 all-eligible PIT: random-pool (shuffle pred_z_self
     within cycle) and random-exit; 60 seeds; real Sharpe vs p95.
  D. n_open / single-name dependence on drop-2 all-eligible PIT.

Reuses s70.prep (PIT trail_ic + self-z + pivots, all_elig), s70.evalcfg
(nested + weighted per-trade CI), s67 retrain, s65 engine.
"""
from __future__ import annotations
import sys, importlib.util, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))


def _imp(n, r):
    sp = importlib.util.spec_from_file_location(n, REPO / r)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m

s64 = _imp("s64", "linear_model/scripts/64_meanrev_v2_backtest.py")
s65 = _imp("s65", "linear_model/scripts/65_tail_attrib_deconc.py")
s67 = _imp("s67", "linear_model/scripts/67_persymbol_meanrev.py")
s70 = _imp("s70", "linear_model/scripts/70_pit_fix_window.py")
s59 = s64.s59
from ml.research.alpha_v4_xs import block_bootstrap_ci

OUT = REPO / "linear_model/results/step71_battery"
OUT.mkdir(parents=True, exist_ok=True)
OOS, BLOCK = s64.OOS, s64.BLOCK
BASE = s64.COST


def build(drop):
    """per-symbol self-std predictions on HL>=2M minus `drop`, PIT trail_ic."""
    panel, px, fc, folds = s67.build_panel(drop)
    apd = s67.finalize(s67.train_persymbol(px, folds, fc), panel)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    return s70.prep(apd, 90, True, all_elig=True)   # all-eligible, PIT embargo


def nested_full(o, cost_mult=1.0):
    """run all-eligible nested at a cost multiple; return df+trades+pos+metrics."""
    apd, aw, pzw, tw, fw, bw, sig = o[0], o[1], o[2], o[3], o[4], o[5], o[6]
    gb = s64.GRID
    s64.GRID = [g for g in gb if g["hurdle"] == 0 and g["sub"] == "ic_pos"]
    s65.COST = BASE * cost_mult
    try:
        nd, ntr, npo = s65.nested(apd, aw, fw, pzw, tw, sig, bw, "design")
    finally:
        s64.GRID = gb
        s65.COST = BASE
    n = nd["net"].to_numpy()
    sh = s59._sharpe(n)
    lo, hi = block_bootstrap_ci(n, statistic=s59._sharpe, block_size=7,
                                n_boot=800)[1:]
    fp = sum(1 for _, g in nd.groupby("fold") if s59._sharpe(g["net"]) > 0)
    wt = np.array([npo.loc[(npo["symbol"] == r["symbol"])
                  & (npo["time"] >= r["entry"]) & (npo["time"] < r["exit"]),
                  "contrib_bps"].sum() for _, r in ntr.iterrows()]) \
        if len(ntr) and len(npo) else np.array([0.0])
    pf = wt[wt > 0].sum() / -wt[wt < 0].sum() if (wt < 0).any() else np.inf
    bs = [np.random.default_rng(k).choice(wt, len(wt)).mean() for k in range(600)]
    one = nd["n_open"] == 1
    drv = (npo.groupby("symbol")["contrib_bps"].sum().sort_values(ascending=False)
           if len(npo) else pd.Series(dtype=float))
    return dict(nd=nd, npo=npo, sh=sh, lo=lo, hi=hi, fp=fp,
                pc=float(n.mean()), tot=float(n.sum()), wt=float(wt.mean()),
                wt_lo=float(np.percentile(bs, 2.5)),
                wt_hi=float(np.percentile(bs, 97.5)), pf=float(pf),
                one_pct=float(nd.loc[one, "net"].sum() / n.sum() * 100
                              if n.sum() else 0), drv=drv)


def main():
    print("=" * 92, flush=True)
    print("  STEP 71: investability battery — PIT-clean all-eligible self-std",
          flush=True)
    print("=" * 92, flush=True)
    t0 = time.time()
    rows = []

    print("\n[iter0] build drop-2 (BIO+VVV) all-eligible PIT ...", flush=True)
    o = build(["BIOUSDT", "VVVUSDT"])
    base = nested_full(o, 1.0)
    print(f"  base: Sh {base['sh']:+.2f}[{base['lo']:+.2f},{base['hi']:+.2f}] "
          f"fp={base['fp']}/9 per-cyc {base['pc']:+.2f} per-trade {base['wt']:+.1f}"
          f"[{base['wt_lo']:+.1f},{base['wt_hi']:+.1f}] PF {base['pf']:.2f} "
          f"n_open1={base['one_pct']:.0f}% tot {base['tot']:,.0f}", flush=True)
    print(f"  [D] top drivers: " + ", ".join(
        f"{k}={v:+.0f}" for k, v in base["drv"].head(6).items()), flush=True)

    print("\n--- A. COST-STRESS (drop-2 all-eligible PIT) ---", flush=True)
    for m in (1.0, 2.0, 3.0):
        r = base if m == 1.0 else nested_full(o, m)
        print(f"  cost {m:.0f}x (~{9*m:.0f}bps RT): Sh {r['sh']:+.2f}"
              f"[{r['lo']:+.2f},{r['hi']:+.2f}] per-cyc {r['pc']:+.2f} "
              f"per-trade {r['wt']:+.1f}[{r['wt_lo']:+.1f},{r['wt_hi']:+.1f}] "
              f"PF {r['pf']:.2f}", flush=True)
        rows.append(dict(part="cost", k=f"{m:.0f}x", sh=r["sh"], lo=r["lo"],
                         hi=r["hi"], wt=r["wt"], wt_lo=r["wt_lo"],
                         wt_hi=r["wt_hi"], pf=r["pf"]))

    print("\n--- B. ITERATIVE DROP + RETRAIN (does it survive past drop-2?) ---",
          flush=True)
    dropped = ["BIOUSDT", "VVVUSDT"]
    cur = base
    rows.append(dict(part="iter", k="drop2", sh=cur["sh"], lo=cur["lo"],
                     hi=cur["hi"], wt=cur["wt"], wt_lo=cur["wt_lo"],
                     wt_hi=cur["wt_hi"], pf=cur["pf"]))
    for it in range(1, 5):
        nxt = cur["drv"].head(2).index.tolist()
        if not nxt:
            break
        dropped = dropped + nxt
        ob = build(dropped)
        cur = nested_full(ob, 1.0)
        nsy = ob[0]["symbol"].nunique()
        print(f"  drop {len(dropped)} (+{nxt}) N={nsy}: Sh {cur['sh']:+.2f}"
              f"[{cur['lo']:+.2f},{cur['hi']:+.2f}] fp={cur['fp']}/9 "
              f"per-trade {cur['wt']:+.1f}[{cur['wt_lo']:+.1f},{cur['wt_hi']:+.1f}]"
              f" PF {cur['pf']:.2f} tot {cur['tot']:,.0f}", flush=True)
        rows.append(dict(part="iter", k=f"drop{len(dropped)}", sh=cur["sh"],
                         lo=cur["lo"], hi=cur["hi"], wt=cur["wt"],
                         wt_lo=cur["wt_lo"], wt_hi=cur["wt_hi"], pf=cur["pf"]))
        if cur["wt_lo"] <= 0 or cur["lo"] <= 0 or nsy < 30:
            why = ("per-trade CI crosses 0" if cur["wt_lo"] <= 0
                   else "Sharpe CI crosses 0" if cur["lo"] <= 0 else "N<30")
            print(f"  STOP ({why}) at drop {len(dropped)} "
                  f"-> {'COLLAPSES like prior variants' if len(dropped)<=4 else 'survived several drops'}",
                  flush=True)
            break

    print("\n--- C. PLACEBO (drop-2 all-eligible PIT, 60 seeds) ---", flush=True)
    apd, aw, pzw, tw, fw, bw, sig = o[0], o[1], o[2], o[3], o[4], o[5], o[6]
    gb = s64.GRID
    s64.GRID = [g for g in gb if g["hurdle"] == 0 and g["sub"] == "ic_pos"]
    s65.COST = BASE
    try:
        rp = []
        for sd in range(60):
            rg = np.random.default_rng(700 + sd); shp = pzw.copy()
            for tt in shp.index:
                v = shp.loc[tt].values.copy(); mk = ~pd.isna(v)
                idx = np.where(mk)[0]
                vv = v.copy(); vv[idx] = v[rg.permutation(idx)]; shp.loc[tt] = vv
            d, _, _ = s65.nested(apd, aw, fw, shp, tw, sig, bw, "design")
            rp.append(s59._sharpe(d["net"].to_numpy()))
    finally:
        s64.GRID = gb
    rp = np.array(rp)
    print(f"  random-pool p95={np.percentile(rp,95):+.2f} mean={rp.mean():+.2f}"
          f"  REAL={base['sh']:+.2f}  -> "
          f"{'PASS' if base['sh']>np.percentile(rp,95) else 'FAIL'}", flush=True)
    rows.append(dict(part="placebo", k="rand-pool-p95",
                     sh=float(np.percentile(rp, 95)), lo=np.nan, hi=np.nan,
                     wt=np.nan, wt_lo=np.nan, wt_hi=np.nan, pf=np.nan))

    pd.DataFrame(rows).to_csv(OUT / "battery.csv", index=False)
    print(f"\n{'='*92}\n  VERDICT INPUTS saved. Decide from: cost-stress CI, "
          f"iter-drop survival depth, placebo PASS/FAIL, n_open%.", flush=True)
    print(f"Total: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
