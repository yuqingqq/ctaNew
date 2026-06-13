"""Step 68: per-symbol model + PER-SYMBOL SELF-STANDARDIZATION before the
cross-sectional step (user's fair-shot test of hypothesis #1).

Rationale: independent per-symbol Ridge models are not on a common scale
(pred_z std 3.8, varies by name) → feeding them straight into a cross-sectional
ranking/subset architecture compares incomparable outputs. Fix: re-express each
symbol's pred_z as a z-score vs its OWN strictly-trailing distribution
(90D rolling, closed='left' → no look-ahead), so all symbols compete on a
common "how extreme vs my own recent history" footing. Then the identical
mean-rev-v2 engine (trail_ic subset, cross-sectional z within subset, top-K=3,
|z|<1 reversion exit, BTC hedge) is applied unchanged.

Grid restricted to hurdle=0: the implied-bps gate (|pred_z|·σ_idio·1e4) is
ill-defined once pred_z is a self-relative z rather than a target_z-unit
prediction; cost is a separate, already-studied lever (Step 66).

Reuses Step 67 saved per-symbol predictions (no retrain for the 44-sym leg).
Honest prior: raw per-sym IC −0.0002 → expect self-std to confirm "no signal"
rather than rescue; this closes the question definitively either way.
"""
from __future__ import annotations
import sys, time, importlib.util, warnings
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
s59 = s64.s59
from ml.research.alpha_v4_xs import block_bootstrap_ci

STEP67 = REPO / "linear_model/results/step67_persymbol/persym_predictions.parquet"
OUT = REPO / "linear_model/results/step68_selfstd"
OUT.mkdir(parents=True, exist_ok=True)
OOS, BLOCK = s64.OOS, s64.BLOCK
SELF_WIN = "90D"
MIN_OBS = 2000          # ~7d of 5m bars before a symbol's self-z is trusted


def add_self_z(apd):
    """pred_z_self = (pred_z − trailing_mean) / trailing_std per symbol,
    strictly trailing (closed='left' excludes the current bar). PIT-safe."""
    apd = apd.sort_values(["symbol", "open_time"]).copy()
    out = np.zeros(len(apd), dtype=np.float64)
    pos = 0
    for sym, g in apd.groupby("symbol", sort=False):
        s = g.set_index("open_time")["pred_z"]
        r = s.rolling(SELF_WIN, closed="left")
        m, sd, c = r.mean(), r.std(), r.count()
        z = (s - m) / sd
        z = z.where((c >= MIN_OBS) & (sd > 1e-12), 0.0)
        z = z.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        out[pos:pos + len(g)] = z.values
        pos += len(g)
    apd["pred_z_self"] = out
    return apd


def cyc_ic(apd, col):
    o = apd.dropna(subset=[col, "alpha_beta"])
    ic = o.groupby("open_time").apply(
        lambda x: x[col].rank().corr(x["alpha_beta"].rank())
        if len(x) >= 5 else np.nan).dropna()
    return float(ic.mean())


def evaluate(apd, label):
    """Run mean-rev-v2 nested with pz = pred_z_self, grid restricted hurdle=0."""
    apd = apd.copy()
    apd["pred_z_orig"] = apd["pred_z"]
    apd["pred_z"] = apd["pred_z_self"]          # engine ranks/z/exits on self-z
    apd["pred_B"] = apd["pred_z"] * apd["trail_ic"]
    aw, pzw, tw, fw, bw, sig, nsy = s67._piv(apd)
    grid_bak = s64.GRID
    s64.GRID = [g for g in grid_bak if g["hurdle"] == 0]   # 4 cfgs (sub×zin)
    s65.COST = s64.COST
    try:
        nd, ntr, npo = s65.nested(apd, aw, fw, pzw, tw, sig, bw, "design")
    finally:
        s64.GRID = grid_bak
    n = nd["net"].to_numpy()
    sh = s59._sharpe(n)
    lo, hi = block_bootstrap_ci(n, statistic=s59._sharpe, block_size=7,
                                n_boot=1000)[1:]
    fp = sum(1 for _, g in nd.groupby("fold") if s59._sharpe(g["net"]) > 0)
    tr_pf = tr_m = tr_lo = tr_hi = np.nan
    if len(ntr):
        c = ntr["cum_bps"].to_numpy()
        tr_pf = (c[c > 0].sum() / -c[c < 0].sum()) if (c < 0).any() else np.inf
        tr_m = float(c.mean())
        bs = [np.random.default_rng(k).choice(c, len(c)).mean() for k in range(1000)]
        tr_lo, tr_hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    cb = [np.random.default_rng(k).choice(n, len(n)).mean() for k in range(1000)]
    pc_lo, pc_hi = np.percentile(cb, 2.5), np.percentile(cb, 97.5)
    ic_self = cyc_ic(apd, "pred_z")
    ic_raw = cyc_ic(apd.assign(pred_z=apd["pred_z_orig"]), "pred_z")
    print(f"  [{label}] N={nsy} | IC raw {ic_raw:+.4f} -> self {ic_self:+.4f} | "
          f"nested Sh {sh:+.2f} [{lo:+.2f},{hi:+.2f}] fp={fp}/9 | "
          f"per-CYCLE {n.mean():+.2f}bps CI[{pc_lo:+.2f},{pc_hi:+.2f}] | "
          f"per-TRADE {tr_m:+.1f}bps CI[{tr_lo:+.1f},{tr_hi:+.1f}] PF {tr_pf:.2f} "
          f"| tot {n.sum():,.0f}", flush=True)
    return dict(label=label, N=nsy, ic_raw=ic_raw, ic_self=ic_self, sharpe=sh,
                lo=lo, hi=hi, fp=fp, pc=float(n.mean()), pc_lo=float(pc_lo),
                pc_hi=float(pc_hi), tr=tr_m, tr_lo=float(tr_lo),
                tr_hi=float(tr_hi), pf=tr_pf, total=float(n.sum()))


def main():
    print("=" * 100, flush=True)
    print("  STEP 68: per-symbol SELF-STANDARDIZED predictions on mean-rev-v2",
          flush=True)
    print("=" * 100, flush=True)
    t0 = time.time()
    res = []

    print("\n--- 44-sym: reuse Step 67 per-symbol preds + self-standardize ---",
          flush=True)
    a = pd.read_parquet(STEP67)
    a["open_time"] = pd.to_datetime(a["open_time"], utc=True)
    a = add_self_z(a)
    print(f"  pred_z raw std {a['pred_z'].std():.3f} -> self-z std "
          f"{a['pred_z_self'].std():.3f} (target ~1; nonzero "
          f"{(a['pred_z_self']!=0).mean()*100:.0f}%)", flush=True)
    res.append(evaluate(a, "persym-selfstd-44"))

    print("\n--- drop BIO+VVV: retrain per-symbol + self-standardize ---",
          flush=True)
    panel2, px2, fc2, folds2 = s67.build_panel(["BIOUSDT", "VVVUSDT"])
    a2 = s67.finalize(s67.train_persymbol(px2, folds2, fc2), panel2)
    a2 = add_self_z(a2)
    res.append(evaluate(a2, "persym-selfstd-drop2-42"))

    pd.DataFrame(res).to_csv(OUT / "summary.csv", index=False)
    print(f"\n{'='*100}\n  VERDICT — did per-symbol self-standardization rescue it?",
          flush=True)
    print(f"{'='*100}", flush=True)
    print("  reference: pooled-44 nested +3.51 (artifact); "
          "persym-44 (no self-std) +1.90 CI-crosses-0, per-trade NEG, IC -0.0002",
          flush=True)
    for r in res:
        rescued = (r["lo"] > 0 and r["tr_lo"] > 0 and r["ic_self"] > 0.01)
        print(f"  {r['label']:24s}: IC {r['ic_raw']:+.4f}->{r['ic_self']:+.4f} | "
              f"nested {r['sharpe']:+.2f}[{r['lo']:+.2f},{r['hi']:+.2f}] | "
              f"per-trade {r['tr']:+.1f}bps CI[{r['tr_lo']:+.1f},{r['tr_hi']:+.1f}] "
              f"=> {'RESCUED (real+robust)' if rescued else 'NOT rescued — confirms no signal'}",
              flush=True)
    print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
