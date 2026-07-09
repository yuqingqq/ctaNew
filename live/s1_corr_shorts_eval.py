"""S1 correlation-aware short selection — official run (addenda 10 + 10b, PRE-REGISTERED).

Rule: short-1 = bottom-1 by base pred (unchanged). Short-2 = of ranks 2-3, the one with the
LOWER trailing-180-cycle corr of alpha_vs_btc_realized to short-1 (shift(1); min 120 valid
paired obs else NO SWAP = keep rank-2). Longs unchanged. Evaluated on the existing incumbent
books; no retraining.

Endpoints per 10b: OOS-only CI for the joint-tail endpoint (2-day blocks; event-days + McNemar);
recent = non-contradiction vs placebo p95 only; 200-seed matched-swap-rate random placebo,
promotion needs OOS events < placebo p5; K-spread guardrail = gross-error catch; swap-cycle
short-leg Δ cost readout; worst-decile mean; dose + fallback diagnostics.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/home/yuqing/ctaNew"); D = REPO / "live/state/convexity"
INC = {"rec": ("hl_tgt_res_base_clean", "hl_tgt_res_long_clean"),
       "oos": ("hl_v4base_oos_clean", "hl_v4long_oos_clean")}
rng = np.random.default_rng(11)
CORR_W, CORR_MIN = 180, 120

def load(book):
    d = pd.read_parquet(D / book / "v0full_hl60.parquet",
                        columns=["symbol", "open_time", "pred"])
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    return d

def blockci(x, days, bd=2, n=2000):
    bid = np.array([pd.Timestamp(d).toordinal() // bd for d in days])
    per = [g.values for _, g in pd.Series(x.values if hasattr(x, "values") else x, index=bid).groupby(level=0)]
    ms = [np.concatenate([per[i] for i in rng.integers(0, len(per), len(per))]).mean() for _ in range(n)]
    return np.percentile(ms, [2.5, 97.5])

def main():
    pan = pd.read_parquet(REPO / "outputs/vBTC_features/panel_expanded_v0.parquet",
                          columns=["symbol", "open_time", "alpha_vs_btc_realized"])
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    pan = pan[(pan.open_time.dt.hour % 4 == 0) & (pan.open_time.dt.minute == 0)].sort_values(["symbol", "open_time"])
    pan["fwd24"] = pan.groupby("symbol")["alpha_vs_btc_realized"].transform(
        lambda s: s.rolling(6).sum().shift(-5)) * 1e4
    # alpha matrix for trailing pairwise corr (shift(1) = strictly prior cycles)
    A = pan.pivot_table(index="open_time", columns="symbol", values="alpha_vs_btc_realized").sort_index()
    A = A.shift(1)
    F = pan.pivot_table(index="open_time", columns="symbol", values="fwd24").sort_index()
    times = A.index

    for win, (ib, il) in INC.items():
        bb = load(ib); bl = load(il)
        rows = []
        tix = {t: i for i, t in enumerate(times)}
        Av = A.to_numpy(); cols = {s: j for j, s in enumerate(A.columns)}
        for t, g in bb.groupby("open_time"):
            if t not in tix: continue
            gl = bl[bl.open_time == t]
            f = F.loc[t].dropna() if t in F.index else None
            if f is None or len(g) < 5 or not len(gl): continue
            g3 = g.nsmallest(3, "pred")
            if len(g3) < 3: continue
            s1, s2, s3 = g3["symbol"].tolist()
            L = gl.nlargest(1, "pred")["symbol"].tolist()
            i1 = tix[t]; lo = max(0, i1 + 1 - CORR_W)
            def tc(sa, sb):
                if sa not in cols or sb not in cols: return np.nan, 0
                x = Av[lo:i1 + 1, cols[sa]]; y = Av[lo:i1 + 1, cols[sb]]
                ok = ~(np.isnan(x) | np.isnan(y))
                if ok.sum() < CORR_MIN: return np.nan, int(ok.sum())
                xs, ys = x[ok], y[ok]
                return float(np.corrcoef(xs, ys)[0, 1]), int(ok.sum())
            c12, n12 = tc(s1, s2); c13, n13 = tc(s1, s3)
            if np.isnan(c12) or np.isnan(c13):
                pick, swapped, fallback = s2, False, True
            else:
                swapped = c13 < c12; pick = s3 if swapped else s2; fallback = False
            def out(sym): return float(f[sym]) if sym in f.index else np.nan
            fl = np.nanmean([out(x) for x in L]) if L else np.nan
            rows.append((t, s1, s2, s3, pick, swapped, fallback,
                         out(s1), out(s2), out(s3), out(pick), fl, c12, c13))
        E = pd.DataFrame(rows, columns=["t", "s1", "s2", "s3", "pick", "swapped", "fallback",
                                        "f1", "f2", "f3", "fpick", "flong", "c12", "c13"]).dropna(
            subset=["f1", "f2", "fpick"])
        days = E.t.dt.date
        X = pd.concat([E.f1, E.f2]).quantile(0.90)   # incumbent short-leg p90 (same-window, 10b-6)
        jt_inc = ((E.f1 > X) & (E.f2 > X))
        jt_s1 = ((E.f1 > X) & (E.fpick > X))
        swap_rate = E.swapped.mean()
        print(f"\n===== {win.upper()} ({len(E)} cycles; X=p90={X:+.0f} bps; swap rate {swap_rate:.3f}; "
              f"corr fallback {E.fallback.mean():.3f}) =====")
        print(f"joint-tail events: incumbent {jt_inc.sum()} ({jt_inc.mean()*100:.2f}%) in "
              f"{days[jt_inc].nunique()} days | S1 {jt_s1.sum()} ({jt_s1.mean()*100:.2f}%) in "
              f"{days[jt_s1].nunique()} days")
        d = jt_s1.astype(int) - jt_inc.astype(int)
        lo_, hi_ = blockci(d, days, bd=2)
        print(f"paired Δ events/cycle {d.mean():+.5f}  2d-block CI [{lo_:+.5f},{hi_:+.5f}]"
              f"  {'EXCLUDES 0' if lo_ > 0 or hi_ < 0 else 'crosses 0'}")
        # event-day McNemar: days where arms disagree
        dd = pd.DataFrame({"inc": jt_inc.values, "s1": jt_s1.values}, index=days)
        dg = dd.groupby(level=0).any()
        b01 = int((dg.inc & ~dg.s1).sum()); b10 = int((~dg.inc & dg.s1).sum())
        print(f"event-day McNemar: incumbent-only days {b01}, S1-only days {b10}")
        # 200-seed matched random-swap placebo
        pl = []
        sw = E.swapped.to_numpy(); f2v = E.f2.to_numpy(); f3v = E.f3.to_numpy(); f1v = E.f1.to_numpy()
        for _ in range(200):
            r = rng.random(len(E)) < swap_rate
            fp = np.where(r, f3v, f2v)
            pl.append(int(((f1v > X) & (fp > X)).sum()))
        pl = np.array(pl)
        p5, p95 = np.percentile(pl, [5, 95])
        print(f"placebo (200 seeds, matched rate): mean {pl.mean():.1f} [p5 {p5:.0f}, p95 {p95:.0f}]"
              f"  -> S1 {jt_s1.sum()} {'< p5 PASS' if jt_s1.sum() < p5 else ('> p95 CONTRADICTION' if jt_s1.sum() > p95 else 'within band')}")
        # worst-decile mean (pooled short-leg fwd24, highest-alpha decile)
        inc_legs = pd.concat([E.f1, E.f2]); s1_legs = pd.concat([E.f1, E.fpick])
        wd_i = inc_legs[inc_legs >= inc_legs.quantile(0.9)].mean()
        wd_s = s1_legs[s1_legs >= s1_legs.quantile(0.9)].mean()
        print(f"worst-decile short-leg mean: incumbent {wd_i:+.0f} -> S1 {wd_s:+.0f} bps")
        # K-spread guardrail + swap-cycle cost
        sp_inc = E.flong - (E.f1 + E.f2) / 2; sp_s1 = E.flong - (E.f1 + E.fpick) / 2
        ds = (sp_s1 - sp_inc).dropna()
        slo, shi = blockci(ds, days.loc[ds.index], bd=1)
        print(f"K-spread guardrail: Δ {ds.mean():+.2f} bps/cyc CI [{slo:+.2f},{shi:+.2f}] (gross-error catch only)")
        sc = E[E.swapped]
        dcost = (sc.f3 - sc.f2)
        clo, chi = blockci(dcost, sc.t.dt.date, bd=1)
        print(f"swap-cycle cost readout: short-2 leg Δ (rank3−rank2 fwd) {dcost.mean():+.1f} bps "
              f"CI [{clo:+.1f},{chi:+.1f}] over {len(sc)} swapped cycles")
        # dose diagnostic (mechanical)
        print(f"dose (diagnostic): mean c(s1,pick) {np.where(E.swapped, E.c13, E.c12).mean():+.3f} "
              f"vs incumbent c(s1,s2) {E.c12.mean():+.3f}")
        # robustness re-reads (diagnostic only)
        for xn, xv in (("other-window X", None), ("fixed +700", 700.0)):
            if xv is None: continue
            ji = ((E.f1 > xv) & (E.f2 > xv)).sum(); js = ((E.f1 > xv) & (E.fpick > xv)).sum()
            print(f"robustness ({xn}): incumbent {ji} -> S1 {js} (diagnostic)")
    print("S1DONE")

if __name__ == "__main__":
    main()
