"""SK1 discrete crowding-skip short lever (RESEARCH_LOOP_20260707 addenda 15 + 15b).

Monetization test for the SQ1 crowding squeeze signal. FROZEN before pred generation.
Lever: the strategy shorts ranks 1-2 (base pred). For each, if its PIT P(squeeze) > the
prior-fold expanding p90, SKIP it (de-gross, no backfill; both flagged -> short gross 0).
Verdict-bearing comparison = treatment vs matched-per-cycle-skip-COUNT placebo (same cycles skip
same count; randomize which short when count=1). Co-primary endpoint = discrete squeeze-event
count avoided vs placebo p5; secondary = net short-side Sharpe/maxDD (underpowered). Baseline
(always-2-shorts) descriptive only. No sweep.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.sq1_crowding_predictive as sq
rng = np.random.default_rng(15)
NSEED = 300

def sharpe(x): return x.mean() / x.std(ddof=1) * np.sqrt(365) if x.std(ddof=1) > 0 else np.nan
def maxdd(x): eq = np.cumsum(x); return float((eq - np.maximum.accumulate(eq)).min())

def main():
    sl = sq.build_rows().reset_index(drop=True)
    sl["fx"] = sl["funding_rate_z_7d"] * sl["funding_dispersion"]
    sl["P"] = sq.nested_preds(sl, ["pred"] + sq.FEATS + ["fx"])   # pred+crowding arm (frozen)
    # taken shorts = ranks 1-2; keep P + outcome
    sh = sl[sl.r <= 2].dropna(subset=["fwd24"]).copy().sort_values(["fold", "open_time", "r"])
    X = sl[sl.r <= 2]["fwd24"].quantile(0.90)   # squeeze threshold (scoring only, cancels)
    sh["sqz"] = (sh["fwd24"] > X).astype(int)
    sh["spnl"] = -sh["fwd24"]                    # short-leg PnL = -alpha (bps)
    # PIT prior-fold expanding p90 threshold of P
    folds = sorted(sh.fold.unique()); thr = {}
    for f in folds:
        prior = sh[(sh.fold < f) & sh.P.notna()]["P"]
        thr[f] = prior.quantile(0.90) if len(prior) >= 80 else np.inf   # warmup -> no skip
    sh["skip"] = (sh["P"] > sh["fold"].map(thr)).fillna(False)
    # per-cycle: the 2 shorts, their skip flags, pnl, sqz
    COST = 9.0
    rows = []
    for t, g in sh.groupby("open_time"):
        if len(g) < 2: continue
        g = g.head(2)
        pnl = g["spnl"].to_numpy(); sq_ = g["sqz"].to_numpy(); sk = g["skip"].to_numpy()
        cnt = int(sk.sum())
        # treatment: skip flagged; each taken short weight 0.5 (of the 2-short side), de-gross
        taken = ~sk
        t_pnl = (pnl[taken] * 0.5).sum() - COST * 0.5 * taken.sum()
        t_ev = int(sq_[taken].sum())
        rows.append((t, g["fold"].iloc[0], cnt, pnl, sq_, t_pnl, t_ev, sk))
    E = pd.DataFrame(rows, columns=["t", "fold", "cnt", "pnl", "sq", "t_pnl", "t_ev", "sk"])
    days = E.t.dt.date
    # baseline (always 2 shorts) — descriptive only
    b_pnl = E.apply(lambda x: (x.pnl * 0.5).sum() - COST, axis=1)
    b_ev = E.apply(lambda x: int(x.sq.sum()), axis=1)
    def daily(series): return pd.Series(series.values, index=days).groupby(level=0).sum()
    print(f"OOS: {len(E)} cycles; skip-count dist {np.bincount(E.cnt, minlength=3).tolist()} "
          f"(0/1/2); total shorts skipped {int(E.cnt.sum())}")
    print(f"squeeze events taken: baseline {int(b_ev.sum())}  treatment {int(E.t_ev.sum())} "
          f"(avoided {int(b_ev.sum()-E.t_ev.sum())})")
    print(f"net Sharpe: baseline {sharpe(daily(b_pnl)):+.2f}  treatment {sharpe(daily(E.t_pnl)):+.2f}"
          f" | maxDD base {maxdd(daily(b_pnl)):+.0f} treat {maxdd(daily(E.t_pnl)):+.0f}")
    # matched-count placebo: same per-cycle skip COUNT, randomize which short
    pl_ev = []; pl_sh = []; pl_dd = []
    for _ in range(NSEED):
        ev = 0; prow = []
        for x in E.itertuples():
            k = x.cnt
            if k == 0: sk = np.array([False, False])
            elif k == 2: sk = np.array([True, True])
            else:
                sk = np.array([False, False]); sk[rng.integers(0, 2)] = True
            taken = ~sk
            ev += int(x.sq[taken].sum())
            prow.append((x.pnl[taken] * 0.5).sum() - 9.0 * 0.5 * taken.sum())
        pl_ev.append(ev); dser = pd.Series(prow, index=days).groupby(level=0).sum()
        pl_sh.append(sharpe(dser)); pl_dd.append(maxdd(dser))
    pl_ev = np.array(pl_ev); pl_sh = np.array(pl_sh); pl_dd = np.array(pl_dd)
    te = int(E.t_ev.sum())
    print(f"\n=== MATCHED-COUNT PLACEBO ({NSEED} seeds) — verdict-bearing ===")
    print(f"squeeze events taken: treatment {te}  placebo mean {pl_ev.mean():.1f} "
          f"[p5 {np.percentile(pl_ev,5):.0f}, p95 {np.percentile(pl_ev,95):.0f}]  "
          f"-> {'BEATS p5 (fewer)' if te < np.percentile(pl_ev,5) else 'within band'}")
    print(f"net Sharpe: treatment {sharpe(daily(E.t_pnl)):+.3f}  placebo mean {pl_sh.mean():+.3f} "
          f"[p5 {np.percentile(pl_sh,5):+.2f}, p95 {np.percentile(pl_sh,95):+.2f}]  "
          f"-> {'BEATS p95' if sharpe(daily(E.t_pnl))>np.percentile(pl_sh,95) else 'within band'}")
    print(f"maxDD: treatment {maxdd(daily(E.t_pnl)):+.0f}  placebo mean {pl_dd.mean():+.0f} "
          f"[p5 {np.percentile(pl_dd,5):+.0f}, p95 {np.percentile(pl_dd,95):+.0f}]  "
          f"-> {'BEATS p95 (less DD)' if maxdd(daily(E.t_pnl))>np.percentile(pl_dd,95) else 'within band'}")
    print("SK1DONE")

if __name__ == "__main__":
    main()
