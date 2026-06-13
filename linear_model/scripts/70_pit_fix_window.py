"""Step 70: (1) proper PIT fix for the trailing-IC selector, (2) real PIT-clean
performance, (3) window sweep, (4) window-vs-signal diagnosis.

The leak: s58.compute_trailing_ic builds its IC window by POSITION (alpha[lo:i]);
alpha=alpha_beta is the 48-bar-FORWARD residual, so the most-recent obs's return
window closes only at the decision cycle. build_rolling_ic_universe avoids this
with an explicit `exit_time <= cutoff` embargo. FIX = add that embargo here:
for trail_ic at decision time t, use obs with open_time in [t-win, t) AND
exit_time <= t (forward window fully realized before the decision).

Then:
  A. real PIT-clean performance: persym-selfstd 44 & drop-BIO+VVV, weighted
     per-trade + CI (the correct metric), nested Sharpe + CI.
  B. window sweep {30,60,90,180}d (PIT-clean) on drop-2.
  C. all-eligible (no IC subset) vs IC-subset — does the selector ADD value?
  D. rho(trail_ic[t] , realized next-period per-symbol IC) — is the IC itself
     self-predictable at all? (answers window-optimizable vs no-signal)
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
OUT = REPO / "linear_model/results/step70_pitfix"
OUT.mkdir(parents=True, exist_ok=True)
OOS, BLOCK = s64.OOS, s64.BLOCK


def trailing_ic_pit(apd, sampled_t, win_days=90, embargo=True):
    """PIT-clean trailing IC. For decision cycle t: corr(pred_z, alpha_beta)
    over obs with open_time in [t-win, t) AND (embargo) exit_time <= t."""
    s = set(sampled_t)
    a = apd[apd["open_time"].isin(s)].sort_values(
        ["symbol", "open_time"]).reset_index(drop=True)
    win = pd.Timedelta(days=win_days)
    out = []
    for sym, g in a.groupby("symbol", sort=False):
        g = g.sort_values("open_time").reset_index(drop=True)
        ot = g["open_time"].values
        et = (pd.to_datetime(g["exit_time"]).values if "exit_time" in g
              else ot)
        pz = g["pred_z"].to_numpy(float)
        al = g["alpha_beta"].to_numpy(float)
        n = len(g)
        ics = np.zeros(n)
        for i in range(n):
            t = ot[i]
            lo = t - win
            m = (ot >= lo) & (ot < t)
            if embargo:
                m &= (et < t)             # forward label closed STRICTLY before t
                                          # (excludes the borderline exit==t obs)
            m &= ~np.isnan(pz) & ~np.isnan(al)
            if m.sum() < 50:
                continue
            p = pd.Series(pz[m]).rank().to_numpy()
            q = pd.Series(al[m]).rank().to_numpy()
            if p.std() < 1e-9 or q.std() < 1e-9:
                continue
            ics[i] = np.corrcoef(p, q)[0, 1]
        out.append(pd.DataFrame({"symbol": sym, "open_time": g["open_time"],
                                 "trail_ic": ics}))
    return pd.concat(out, ignore_index=True)


def prep(apd, win_days, embargo, all_elig=False):
    """rebuild trail_ic (PIT) + self-z, pivots for the engine."""
    apd = apd.copy()
    samp = sorted(apd[apd["fold"].isin(OOS)]["open_time"].unique())[::BLOCK]
    tic = trailing_ic_pit(apd, samp, win_days, embargo)
    apd = apd.drop(columns=["trail_ic"], errors="ignore").merge(
        tic, on=["symbol", "open_time"], how="left")
    apd["trail_ic"] = apd["trail_ic"].fillna(0)
    apd = s68.add_self_z(apd)
    apd["pred_z"] = apd["pred_z_self"]
    apd["pred_B"] = apd["pred_z"] * apd["trail_ic"]
    aw, pzw, tw, fw, bw, sig, nsy = s67._piv(apd)
    if all_elig:
        tw = tw.where(tw.isna(), 1.0)        # every symbol passes tic>0
    return apd, aw, pzw, tw, fw, bw, sig, nsy, tic


def evalcfg(apd, aw, pzw, tw, fw, bw, sig, label, force_icpos=False):
    gb = s64.GRID
    s64.GRID = [g for g in gb if g["hurdle"] == 0
                and (g["sub"] == "ic_pos" if force_icpos else True)]
    s65.COST = s64.COST
    try:
        nd, ntr, npo = s65.nested(apd, aw, fw, pzw, tw, sig, bw, "design")
    finally:
        s64.GRID = gb
    n = nd["net"].to_numpy()
    sh = s59._sharpe(n)
    lo, hi = block_bootstrap_ci(n, statistic=s59._sharpe, block_size=7,
                                n_boot=1000)[1:]
    fp = sum(1 for _, g in nd.groupby("fold") if s59._sharpe(g["net"]) > 0)
    wt_m = wt_lo = wt_hi = pf = np.nan
    if len(ntr) and len(npo):
        po = npo
        rec = [po.loc[(po["symbol"] == r["symbol"]) & (po["time"] >= r["entry"])
               & (po["time"] < r["exit"]), "contrib_bps"].sum()
               for _, r in ntr.iterrows()]
        wt = np.array(rec, float)
        pf = wt[wt > 0].sum() / -wt[wt < 0].sum() if (wt < 0).any() else np.inf
        bs = [np.random.default_rng(k).choice(wt, len(wt)).mean()
              for k in range(800)]
        wt_m, wt_lo, wt_hi = wt.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    print(f"  [{label}] Sh {sh:+.2f}[{lo:+.2f},{hi:+.2f}] fp={fp}/9 | "
          f"per-cyc {n.mean():+.2f} | per-trade(wtd) {wt_m:+.1f} "
          f"CI[{wt_lo:+.1f},{wt_hi:+.1f}] PF {pf:.2f} | tot {n.sum():,.0f}",
          flush=True)
    return dict(label=label, sharpe=sh, lo=lo, hi=hi, fp=fp,
                pc=float(n.mean()), wt=float(wt_m), wt_lo=float(wt_lo),
                wt_hi=float(wt_hi), pf=float(pf), total=float(n.sum()))


def ic_predictability(apd, samp, win_days):
    """Honest test: does PAST-window IC predict the DISJOINT FUTURE-window IC?
    future_ic[t] = corr(pred_z, alpha_beta) over open_time in (t, t+win];
    compared to trail_ic[t] (past [t-win, t)). Sample every `win` cycles so
    past/future windows of consecutive samples don't overlap (no spurious ρ)."""
    past = trailing_ic_pit(apd, samp, win_days, embargo=True).rename(
        columns={"trail_ic": "ic_past"})
    a = apd[apd["open_time"].isin(set(samp))].sort_values(
        ["symbol", "open_time"]).reset_index(drop=True)
    win = pd.Timedelta(days=win_days)
    fut = []
    for sym, g in a.groupby("symbol", sort=False):
        g = g.sort_values("open_time").reset_index(drop=True)
        ot = g["open_time"].values
        pz = g["pred_z"].to_numpy(float); al = g["alpha_beta"].to_numpy(float)
        fic = np.full(len(g), np.nan)
        for i in range(len(g)):
            t = ot[i]; m = (ot > t) & (ot <= t + win)
            m &= ~np.isnan(pz) & ~np.isnan(al)
            if m.sum() < 50:
                continue
            p = pd.Series(pz[m]).rank().to_numpy()
            q = pd.Series(al[m]).rank().to_numpy()
            if p.std() < 1e-9 or q.std() < 1e-9:
                continue
            fic[i] = np.corrcoef(p, q)[0, 1]
        fut.append(pd.DataFrame({"symbol": sym, "open_time": g["open_time"],
                                 "ic_fut": fic}))
    d = past.merge(pd.concat(fut, ignore_index=True),
                   on=["symbol", "open_time"], how="inner").dropna()
    d = d[(d.ic_past != 0)]
    # thin to non-overlapping samples per symbol (every win_cycles)
    step = max(1, win_days * 6)
    d = d.sort_values(["symbol", "open_time"]).groupby("symbol").apply(
        lambda g: g.iloc[::step]).reset_index(drop=True)
    if len(d) < 30:
        return np.nan, np.nan, len(d)
    rho = d["ic_past"].corr(d["ic_fut"], method="spearman")
    cs = d.groupby("open_time").apply(
        lambda g: g["ic_past"].rank().corr(g["ic_fut"].rank())
        if len(g) >= 5 else np.nan).dropna()
    return float(rho), float(cs.mean() if len(cs) else np.nan), len(d)


def main():
    print("=" * 92, flush=True)
    print("  STEP 70: PIT-fix trailing-IC + real perf + window-vs-signal",
          flush=True)
    print("=" * 92, flush=True)
    t0 = time.time()
    res = []

    a44 = pd.read_parquet(STEP67)
    a44["open_time"] = pd.to_datetime(a44["open_time"], utc=True)
    if "exit_time" in a44:
        a44["exit_time"] = pd.to_datetime(a44["exit_time"], utc=True)

    print("\n--- A. REAL PIT-clean performance (embargo ON, win=90) ---",
          flush=True)
    o = prep(a44, 90, True)
    res.append(evalcfg(*o[:7], "persym-selfstd-44 PIT"))
    s44 = sorted(o[0][o[0]["fold"].isin(OOS)]["open_time"].unique())[::BLOCK]
    rho44, cs44, n44 = ic_predictability(o[0], s44, 90)
    print(f"      [D] does PAST-90d IC predict DISJOINT FUTURE-90d IC? "
          f"ρ={rho44:+.3f}  cross-sec={cs44:+.3f}  (n={n44} non-overlap)",
          flush=True)

    panel2, px2, fc2, folds2 = s67.build_panel(["BIOUSDT", "VVVUSDT"])
    a42 = s67.finalize(s67.train_persymbol(px2, folds2, fc2), panel2)
    a42["exit_time"] = pd.to_datetime(a42["exit_time"], utc=True)
    o2 = prep(a42, 90, True)
    res.append(evalcfg(*o2[:7], "persym-selfstd-drop2 PIT"))

    print("\n--- C. all-eligible (NO IC subset) vs IC-subset, drop-2 PIT ---",
          flush=True)
    oae = prep(a42, 90, True, all_elig=True)
    res.append(evalcfg(*oae[:7], "drop2 PIT all-eligible", force_icpos=True))

    print("\n--- B. WINDOW SWEEP (PIT, drop-2): does any window help? ---",
          flush=True)
    for wd in (30, 60, 180):
        ow = prep(a42, wd, True)
        res.append(evalcfg(*ow[:7], f"drop2 PIT win={wd}d"))
        sw = sorted(ow[0][ow[0]["fold"].isin(OOS)]["open_time"].unique())[::BLOCK]
        rr, cc, nn = ic_predictability(ow[0], sw, wd)
        print(f"      win={wd}d: PAST→FUTURE IC ρ={rr:+.3f} cs={cc:+.3f} "
              f"(n={nn})", flush=True)

    pd.DataFrame(res).to_csv(OUT / "summary.csv", index=False)
    print(f"\n{'='*92}\n  DIAGNOSIS", flush=True)
    print(f"{'='*92}", flush=True)
    print("  - REAL PIT-clean perf = the embargo-ON numbers above (vs leaky "
          "+2.79/+3.48).", flush=True)
    print("  - If no window gives CI-excludes-0 AND IC-subset does not beat "
          "all-eligible AND ρ(IC_t,IC_t+1)≈0 -> it is the SIGNAL "
          "(features→IC≈0→trailing-IC unreliable), NOT the window.", flush=True)
    print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
