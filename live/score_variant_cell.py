"""Committed scorer for feature-variant cells (RESEARCH_LOOP_20260707 addendum 6b, BINDING).

PRIMARY book-level endpoints per the estimator law (no live-overlay replays):
  1. paired per-cycle cross-sectional rank-IC delta (variant − incumbent), day-block bootstrap CI
  2. top/bot-K selection alpha spread delta at production K (1 long by long-book, 2 shorts by base)
  3. big-|BTC-4h-move| quintile split of (1) — NOT IMPLEMENTED (results-review F5): it was never
     computed for the 2026-07-08 program (absence declared in FEATURE_TUNING_RESULTS.md);
     implement it before any reuse of this scorer.
  4. per-fold and per-year delta tables; population accounting (test-row & symbol coverage diffs)
Verdict text auto-generated from the pre-registered bars. Usage (tags as built, one per cell):
  python3 live/score_variant_cell.py <tag>   # C1=ret36h C2=retc1 C3=resid3 C4=ddc2 C5=corr12h T1=takerls
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/home/yuqing/ctaNew"); D = REPO / "live/state/convexity"
INC = {"rec": ("hl_tgt_res_base_clean", "hl_tgt_res_long_clean"),
       "oos": ("hl_v4base_oos_clean", "hl_v4long_oos_clean")}
rng = np.random.default_rng(7)

def load(book):
    d = pd.read_parquet(D / book / "v0full_hl60.parquet",
                        columns=["symbol", "open_time", "pred", "alpha_A", "fold"])
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    return d

def fwd24(panel_alpha):
    return panel_alpha  # alpha_A column is the per-row 4h residual; 24h tip needs the rolled sum

def per_cycle_stats(base_b, base_l, var_b, var_l, fwd):
    """per cycle: rank-IC (base-book pred vs fwd24) and K-selection spread, for each arm.

    RESULTS-REVIEW FIX (2026-07-08, F1): both arms MUST be scored on the SAME symbol population
    per cycle (incumbent ∩ variant ∩ fwd). The first version scored icB on `common` and icV on
    `commonV` separately; with per-cycle book membership differing (variant train-row minimums
    drop symbol-folds), that mixed population effects into the paired Δ — C2's "entirely
    negative" OOS CI and T1's positive Δ were both population artifacts. Never revert to
    per-arm populations."""
    out = []
    vb = var_b.set_index(["open_time", "symbol"])["pred"]
    vl = var_l.set_index(["open_time", "symbol"])["pred"]
    for t, g in base_b.groupby("open_time"):
        gl = base_l[base_l.open_time == t].set_index("symbol")["pred"]
        f = fwd.get(t)
        if f is None or len(g) < 5: continue
        g = g.set_index("symbol")
        try:
            vbt = vb.loc[t]; vlt = vl.loc[t]
        except KeyError:
            continue
        common = g.index.intersection(f.index).intersection(vbt.index)
        if len(common) < 5: continue
        fB = f.loc[common]
        icB = g.loc[common, "pred"].rank().corr(fB.rank())
        icV = vbt.loc[common].rank().corr(fB.rank())
        # selection spread: 1 long by long-book, 2 shorts by base-book — matched pops both arms
        li = gl.index.intersection(f.index).intersection(vlt.index)
        def spread(bpred, lpred):
            if len(common) < 3 or len(li) < 1: return np.nan
            L = lpred.loc[li].nlargest(1).index; S = bpred.loc[common].nsmallest(2).index
            return float(f.loc[L].mean() - f.loc[S].mean())
        spB = spread(g["pred"], gl)
        spV = spread(vbt, vlt)
        out.append((t, icB, icV, spB, spV))
    return pd.DataFrame(out, columns=["t", "icB", "icV", "spB", "spV"]).dropna()

def blockci(x, days, n=2000):
    per = [g.values for _, g in pd.Series(x.values, index=days).groupby(level=0)]
    ms = [np.concatenate([per[i] for i in rng.integers(0, len(per), len(per))]).mean() for _ in range(n)]
    return np.percentile(ms, [2.5, 97.5])

def main():
    tag = sys.argv[1]
    pan = pd.read_parquet(REPO / "outputs/vBTC_features/panel_expanded_v0.parquet",
                          columns=["symbol", "open_time", "alpha_vs_btc_realized"])
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    pan = pan[(pan.open_time.dt.hour % 4 == 0) & (pan.open_time.dt.minute == 0)].sort_values(["symbol", "open_time"])
    pan["fwd24"] = pan.groupby("symbol")["alpha_vs_btc_realized"].transform(
        lambda s: s.rolling(6).sum().shift(-5)) * 1e4
    fwd = {t: g.set_index("symbol")["fwd24"].dropna() for t, g in pan.groupby("open_time")}
    # NB: pre-registered endpoint 3 (|BTC-move| quintile split of the rank-IC delta) was NEVER
    # computed anywhere for the 2026-07-08 program (F5) — it is not "computed elsewhere".
    # Implement it here before this scorer is reused.
    for win, (ib, il) in INC.items():
        vb = load(f"hl_{tag}_base" + ("_oos" if win == "oos" else ""))
        vl = load(f"hl_{tag}_long" + ("_oos" if win == "oos" else ""))
        bb = load(ib); bl = load(il)
        st = per_cycle_stats(bb, bl, vb, vl, fwd)
        if st.empty:
            print(f"{win}: no overlap"); continue
        dic = st.icV - st.icB; dsp = st.spV - st.spB
        days = st.t.dt.date
        lo, hi = blockci(dic, days); slo, shi = blockci(dsp, days)
        # coverage / population accounting at test level
        cov_rows = len(vb) / max(len(bb), 1)
        print(f"\n===== {win.upper()} ({len(st)} scored cycles; variant/incumbent test-row ratio {cov_rows:.4f}) =====")
        print(f"rank-IC:  incumbent {st.icB.mean():+.4f}  variant {st.icV.mean():+.4f}  Δ {dic.mean():+.4f}  CI [{lo:+.4f},{hi:+.4f}] {'EXCLUDES 0' if lo>0 or hi<0 else 'crosses 0'}")
        print(f"K-spread: incumbent {st.spB.mean():+.1f}  variant {st.spV.mean():+.1f}  Δ {dsp.mean():+.2f} bps/cyc  CI [{slo:+.2f},{shi:+.2f}]")
        yr = dic.groupby(st.t.dt.year).mean().round(4).to_dict()
        print(f"Δrank-IC by year: {yr}")
        hit = dic.groupby(st.t.dt.to_period('M')).mean()
        print(f"monthly hit rate Δic>0: {(hit>0).sum()}/{len(hit)}")

if __name__ == "__main__":
    main()
