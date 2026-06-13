"""Step 69: VERIFICATION / AUDIT of the persym-selfstd result before any
validation battery. Triggered because the prior "per-trade negative" verdict
was an analytical error (unweighted cum_bps != weighted portfolio P&L).

Re-derives persym-selfstd (44-sym reuse Step67 preds; drop-BIO+VVV retrain) with
full logging via the validated s65.nested engine, then runs six checks:

  1. ACCOUNTING RECONCILIATION  Σ per-position weighted contrib == Σ portfolio
     gross (proves the engine accounting is internally consistent).
  2. CORRECT per-TRADE economics: weighted gross per trade (join trades<->pos),
     mean / PF / bootstrap CI — the honest version of the botched metric.
     Also show the (misleading) unweighted cum_bps for contrast.
  3. PER-FOLD Sharpe + LOFO  — is +2.79 spread or 1-2-fold driven?
  4. n_open split — fraction of PnL from single-name (n_open==1) cycles.
  5. PER-SYMBOL attribution of the 42-sym (BIO+VVV-removed) variant — did the
     edge relocate to the next volatile names (Step-66 structural pattern)?
  6. trail_ic LOOK-AHEAD robustness — rerun with trail_ic lagged one cycle
     (strictly-past subset); does the edge survive?
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
s68 = _imp("s68", "linear_model/scripts/68_persymbol_selfstd.py")
s59 = s64.s59
from ml.research.alpha_v4_xs import block_bootstrap_ci

STEP67 = REPO / "linear_model/results/step67_persymbol/persym_predictions.parquet"
OUT = REPO / "linear_model/results/step69_verify"
OUT.mkdir(parents=True, exist_ok=True)
SEMI = s65.SEMI_MEME


def run_nested(apd, lag_tic=False):
    """s68-style: pz=self-z, grid hurdle=0; optionally lag trail_ic by 1 cycle
    (strictly-past subset) to test look-ahead in the rolling-IC selector."""
    apd = apd.copy()
    apd["pred_z"] = apd["pred_z_self"]
    apd["pred_B"] = apd["pred_z"] * apd["trail_ic"]
    aw, pzw, tw, fw, bw, sig, nsy = s67._piv(apd)
    if lag_tic:
        tw = tw.shift(1)                      # subset uses only past trail_ic
    gb = s64.GRID
    s64.GRID = [g for g in gb if g["hurdle"] == 0]
    s65.COST = s64.COST
    try:
        nd, ntr, npo = s65.nested(apd, aw, fw, pzw, tw, sig, bw, "design")
    finally:
        s64.GRID = gb
    return nd, ntr, npo, nsy


def audit(tag, apd):
    nd, ntr, npo, nsy = run_nested(apd)
    n = nd["net"].to_numpy()
    sh = s59._sharpe(n)
    lo, hi = block_bootstrap_ci(n, statistic=s59._sharpe, block_size=7,
                                n_boot=1000)[1:]
    print(f"\n{'='*92}\n  {tag}  (N={nsy})\n{'='*92}", flush=True)
    print(f"  nested Sharpe {sh:+.2f} [{lo:+.2f},{hi:+.2f}] | per-cycle "
          f"{n.mean():+.2f} bps | total {n.sum():,.0f}", flush=True)

    # 1. reconciliation
    g_port = nd["gross"].sum()
    g_pos = npo["contrib_bps"].sum() if len(npo) else np.nan
    print(f"  [1] reconcile: Σ portfolio gross {g_port:,.1f} vs "
          f"Σ per-position weighted contrib {g_pos:,.1f}  -> "
          f"{'OK (consistent)' if abs(g_port-g_pos) < 1.0 else 'MISMATCH!'}",
          flush=True)

    # 2. correct weighted per-trade economics (join trades <-> pos)
    if len(ntr) and len(npo):
        po = npo.copy()
        rec = []
        for _, t in ntr.iterrows():
            m = ((po["symbol"] == t["symbol"]) & (po["time"] >= t["entry"])
                 & (po["time"] < t["exit"]))
            rec.append(po.loc[m, "contrib_bps"].sum())
        wt = np.array(rec, dtype=float)
        uw = ntr["cum_bps"].to_numpy(dtype=float)
        pf_w = wt[wt > 0].sum() / -wt[wt < 0].sum() if (wt < 0).any() else np.inf
        pf_u = uw[uw > 0].sum() / -uw[uw < 0].sum() if (uw < 0).any() else np.inf
        bs = [np.random.default_rng(k).choice(wt, len(wt)).mean()
              for k in range(1000)]
        print(f"  [2] per-trade WEIGHTED-gross (correct): mean {wt.mean():+.2f} "
              f"bps CI[{np.percentile(bs,2.5):+.2f},{np.percentile(bs,97.5):+.2f}]"
              f" PF {pf_w:.2f}  win {(wt>0).mean()*100:.0f}%  n={len(wt)}",
              flush=True)
        print(f"      (misleading UNWEIGHTED cum_bps for contrast: "
              f"mean {uw.mean():+.1f} PF {pf_u:.2f})", flush=True)

    # 3. per-fold + LOFO
    pf_rows = []
    for f, g in nd.groupby("fold"):
        pf_rows.append((int(f), s59._sharpe(g["net"]), g["net"].sum()))
    print("  [3] per-fold (Sharpe / total bps):", flush=True)
    print("      " + "  ".join(f"f{f}:{s:+.1f}/{tot:+.0f}"
                                for f, s, tot in pf_rows), flush=True)
    lofo = []
    for f in range(1, 10):
        rem = nd[nd["fold"] != f]["net"].to_numpy()
        lofo.append((f, s59._sharpe(rem) - sh))
    worst = min(lofo, key=lambda x: x[1])
    print(f"      LOFO worst: drop f{worst[0]} -> ΔSharpe {worst[1]:+.2f}  "
          f"({'concentrated' if worst[1] < -1.0 else 'spread'})", flush=True)

    # 4. n_open split
    one = nd["n_open"] == 1
    print(f"  [4] n_open==1: {one.sum()} cyc carry {nd.loc[one,'net'].sum():,.0f}"
          f" bps ({nd.loc[one,'net'].sum()/n.sum()*100 if n.sum() else 0:.0f}%)"
          f" ; Sharpe excl = {s59._sharpe(nd.loc[~one,'net'].to_numpy()):+.2f}",
          flush=True)

    # 5. per-symbol attribution (tail + overall)
    if len(npo):
        ov = npo.groupby("symbol")["contrib_bps"].sum().sort_values(
            ascending=False)
        sm = sum(v for k, v in ov.items() if k in SEMI)
        print(f"  [5] top contributors: " + ", ".join(
            f"{k}{'*' if k in SEMI else ''}={v:+.0f}"
            for k, v in ov.head(6).items()), flush=True)
        print(f"      semi-meme(*) share of gross: "
              f"{sm/ov[ov>0].sum()*100 if (ov>0).any() else 0:.0f}%", flush=True)
    return dict(tag=tag, N=nsy, sharpe=sh, lo=lo, hi=hi,
                pc=float(n.mean()), total=float(n.sum()))


def main():
    print("=" * 92, flush=True)
    print("  STEP 69: VERIFY/AUDIT persym-selfstd before validation battery",
          flush=True)
    print("=" * 92, flush=True)
    t0 = time.time()

    a44 = pd.read_parquet(STEP67)
    a44["open_time"] = pd.to_datetime(a44["open_time"], utc=True)
    a44 = s68.add_self_z(a44)
    r44 = audit("persym-selfstd-44", a44)

    panel2, px2, fc2, folds2 = s67.build_panel(["BIOUSDT", "VVVUSDT"])
    a42 = s68.add_self_z(s67.finalize(
        s67.train_persymbol(px2, folds2, fc2), panel2))
    r42 = audit("persym-selfstd-drop2-42", a42)

    # 6. trail_ic look-ahead robustness on the decisive (drop-2) variant
    nd_l, _, _, _ = run_nested(a42, lag_tic=True)
    n_l = nd_l["net"].to_numpy()
    sh_l = s59._sharpe(n_l)
    lo_l, hi_l = block_bootstrap_ci(n_l, statistic=s59._sharpe, block_size=7,
                                    n_boot=1000)[1:]
    print(f"\n{'='*92}", flush=True)
    print(f"  [6] trail_ic LOOK-AHEAD test (drop2, subset on LAGGED trail_ic): "
          f"Sharpe {sh_l:+.2f} [{lo_l:+.2f},{hi_l:+.2f}]  vs unlagged "
          f"{r42['sharpe']:+.2f} [{r42['lo']:+.2f},{r42['hi']:+.2f}]  -> "
          f"{'survives (no material leak)' if lo_l > 0 else 'DEGRADES — subset leak'}",
          flush=True)

    pd.DataFrame([r44, r42, dict(tag='drop2-lagged-tic', N=r42['N'],
                 sharpe=sh_l, lo=lo_l, hi=hi_l, pc=float(n_l.mean()),
                 total=float(n_l.sum()))]).to_csv(OUT / "audit.csv", index=False)
    print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
