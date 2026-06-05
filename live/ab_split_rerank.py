"""A/B: STATIC (frozen-at-retrain) vs RE-RANK volatility split — full OOS replay through the real bot.

The flow book = top-80 symbols by trailing-30d rvol_7d. The question: do we re-rank that set over time,
or freeze it? Champion research said "static-at-retrain beats rolling re-rank" — this verifies it on the
full OOS window (2025-10-04 -> latest) with the production stack (K=3, 6-sleeve, conv_gate, flat_real).

Policies (each builds time-varying bookA/bookB preds files, then runs the identical replay+combine):
  never      rank once as-of OOS start, hold all months
  monthly    re-rank at each calendar-month start, hold within month  (= production static-at-retrain)
  daily      re-rank every cycle on trailing-30d rvol                 (= the daily-script bug)
  frozen_end frozen as-of fit_cut (current shipped split; mild look-ahead for early OOS — reference)

Usage: python3 live/ab_split_rerank.py [--n 80] [--policies never,monthly,daily,frozen_end]
"""
import sys, os, json, argparse, subprocess
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))

PANEL  = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
HLDIR  = REPO/os.environ.get("AB_HLDIR", "live/state/convexity/hl")
OUTBASE= REPO/os.environ.get("AB_OUTBASE", "live/state/ab_split")
OOS_START = pd.Timestamp("2025-10-04", tz="UTC")
FIT_CUT   = pd.Timestamp("2026-05-29", tz="UTC")


SPLIT_FEAT = os.environ.get("AB_SPLIT_FEAT", "rvol_7d")   # axis to partition flow vs price book (high -> flow book)

RVOL_WIN = int(os.environ.get("AB_RVOL_WIN", "30"))   # #180: ranking lookback (longer = smoother, less churn)
def trailing_rvol_asof(p, asof, win=RVOL_WIN):
    lo = asof - pd.Timedelta(days=win)
    q = p[(p.open_time >= lo) & (p.open_time < asof)]
    return q.groupby("symbol")[SPLIT_FEAT].mean()


def top_n_asof(p, oos, asof, n):
    rv = trailing_rvol_asof(p, asof).to_dict()
    return frozenset(sorted([s for s in oos if np.isfinite(rv.get(s, np.nan))], key=lambda s: -rv[s])[:n])


def build_membership(policy, times, p, oos, n):
    """Return dict open_time -> frozenset(flow-book symbols) for the policy."""
    days = sorted({t.normalize() for t in times})
    if policy == "never":
        s = top_n_asof(p, oos, OOS_START, n); return {t: s for t in times}
    if policy == "frozen_end":
        s = top_n_asof(p, oos, FIT_CUT, n); return {t: s for t in times}
    if policy == "monthly":
        anchors = sorted({OOS_START} | {d for d in days if d.day == 1})
        cache = {a: top_n_asof(p, oos, a, n) for a in anchors}
        def pick(t):
            a = max([a for a in anchors if a <= t], default=OOS_START); return cache[a]
        return {t: pick(t) for t in times}
    if policy == "daily":
        cache = {d: top_n_asof(p, oos, d + pd.Timedelta(days=1), n) for d in days}  # asof=day+1 (matches old bug)
        return {t: cache[t.normalize()] for t in times}
    if policy == "monthly_hyst":   # #180: band-hysteresis exclude set — enter high-vol at rank<=N_HI, leave below N_LO
        N_HI = int(os.environ.get("AB_HYST_HI", str(n)))          # enter-exclude threshold (clearly high-vol)
        N_LO = int(os.environ.get("AB_HYST_LO", str(n + 20)))     # leave-exclude threshold (clearly low-vol); band = [HI,LO]
        anchors = sorted({OOS_START} | {d for d in days if d.day == 1})
        rank = {a: {s: i for i, s in enumerate(  # rvol-desc rank (0 = highest vol) as-of anchor
                    sorted([s for s in oos if np.isfinite(trailing_rvol_asof(p, a).get(s, np.nan))],
                           key=lambda s: -trailing_rvol_asof(p, a).get(s, -np.inf)))} for a in anchors}
        cache = {}; prev = frozenset()
        for a in anchors:
            r = rank[a]
            excl = set()
            for s, rk in r.items():
                was = s in prev
                if (was and rk < N_LO) or ((not was) and rk < N_HI): excl.add(s)
            prev = frozenset(excl); cache[a] = prev
        def pick(t):
            a = max([x for x in anchors if x <= t], default=OOS_START); return cache[a]
        return {t: pick(t) for t in times}
    raise ValueError(policy)


def write_books(policy, memb, ff, v0, outdir):
    """bookA = flow preds for in-set syms; bookB = price preds for out-of-set syms (per cycle)."""
    from collections import defaultdict
    times_by_set = defaultdict(list); set_objs = {}
    for t, s in memb.items():
        times_by_set[id(s)].append(t); set_objs[id(s)] = s
    ffA_parts, v0B_parts = [], []
    for sid_, tlist in times_by_set.items():
        s = set_objs[sid_]; tset = pd.DatetimeIndex(tlist)
        sub = ff[ff.open_time.isin(tset)]; ffA_parts.append(sub[sub.symbol.isin(s)])
        sub = v0[v0.open_time.isin(tset)]; v0B_parts.append(sub[~sub.symbol.isin(s)])
    ffA = pd.concat(ffA_parts, ignore_index=True); v0B = pd.concat(v0B_parts, ignore_index=True)
    outdir.mkdir(parents=True, exist_ok=True)
    a, b = outdir/f"bookA_{policy}.parquet", outdir/f"bookB_{policy}.parquet"
    ffA.to_parquet(a); v0B.to_parquet(b)
    return a, b, ffA.symbol.nunique(), v0B.symbol.nunique()


def run_replay(preds_path, state_dir, hold=None, sidemode=None, k=None, skip=None, rrgate=None, preds_long=None):
    env = dict(os.environ, PYTHONPATH=str(REPO), CONVEXITY_PREDS_PATH=str(preds_path),
               CONVEXITY_STATE=str(state_dir), STRAT_K=str(k) if k is not None else os.environ.get("STRAT_K", "3"),
               SIDE_MODE=(sidemode or os.environ.get("SIDE_MODE", "default")))
    if preds_long: env["CONVEXITY_PREDS_LONG"] = str(preds_long)   # dual-pred long-leg ranker
    if hold is not None: env["STRAT_HOLD"] = str(hold)   # per-book hold override (bucket-specific sleeve count)
    if skip is not None: env["LONG_IDIO_SKIP_PCT"] = str(skip)   # per-book idio-vol long-skip pctile
    if rrgate is not None: env["LONG_RESIDREV_GATE"] = str(rrgate)   # per-book resid-rev long gate (0/1)
    Path(state_dir).mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, "-m", "live.convexity_paper_bot", "--replay-all"],
                       env=env, cwd=str(REPO), capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(f"\n[replay FAIL {state_dir}]\n{r.stderr[-2500:]}\n"); r.check_returncode()
    return Path(state_dir)/"cycles.csv"


def combine(book_a_cycles, book_b_cycles, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(REPO/"live/convexity_twobook_combine.py"),
                    "--book-a", str(book_a_cycles), "--book-b", str(book_b_cycles), "--out", str(outdir)],
                   cwd=str(REPO), check=True, capture_output=True)
    return json.load(open(outdir/"twobook_summary.json"))


def churn(memb, times):
    """avg # of symbols entering/leaving the flow book between consecutive distinct sets."""
    seen, prev, moves, steps = [], None, 0, 0
    for t in sorted(times):
        s = memb[t]
        if prev is not None and s is not prev and s != prev:
            moves += len(s ^ prev); steps += 1
        prev = s
    return (moves / steps) if steps else 0.0, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--policies", default="never,monthly,daily,frozen_end")
    a = ap.parse_args()
    policies = a.policies.split(",")

    p = pd.read_parquet(PANEL, columns=["symbol", "open_time", SPLIT_FEAT]); p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
    ff = pd.read_parquet(HLDIR/"fullflow_hl60.parquet"); v0 = pd.read_parquet(HLDIR/"v0full_hl60.parquet")
    for d in (ff, v0):
        d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    ff = ff[ff.open_time >= OOS_START]; v0 = v0[v0.open_time >= OOS_START]
    oos = sorted(set(ff.symbol.unique()))
    times = sorted(ff.open_time.unique())
    times = [pd.Timestamp(t) for t in times]
    print(f"OOS {times[0].date()} -> {times[-1].date()}  |  {len(times)} cycles  |  {len(oos)} flow-universe syms  |  N={a.n}\n")

    rows = []
    for pol in policies:
        memb = build_membership(pol, times, p, oos, a.n)
        ch, steps = churn(memb, times)
        od = OUTBASE/pol
        ba, bb, nA, nB = write_books(pol, memb, ff, v0, od)
        print(f"[{pol}] flow≈{nA} price≈{nB} syms-ever | re-rank steps={steps} avg-churn={ch:.1f} syms/step ... running replays")
        _hl_long = os.environ.get("AB_HLDIR_LONG")   # dual-pred: long-leg ranker preds dir (per-book file)
        plA = (REPO/_hl_long/"fullflow_hl60.parquet") if _hl_long else None
        plB = (REPO/_hl_long/"v0full_hl60.parquet") if _hl_long else None
        ca = run_replay(ba, od/"stateA", os.environ.get("AB_HOLD_A"), os.environ.get("AB_SIDEMODE_A"), os.environ.get("AB_K_A"), os.environ.get("AB_SKIP_A"), os.environ.get("AB_RRGATE_A"), plA)   # A = high-vol/flow book
        cb = run_replay(bb, od/"stateB", os.environ.get("AB_HOLD_B"), os.environ.get("AB_SIDEMODE_B"), os.environ.get("AB_K_B"), os.environ.get("AB_SKIP_B"), os.environ.get("AB_RRGATE_B"), plB)   # B = low-vol/price book
        summ = combine(ca, cb, od/"combine")
        rows.append({"policy": pol, "sharpe": summ["sharpe_both_active"], "totPnL": summ["totPnL_both_active"],
                     "maxDD": summ["maxDD_both_active"], "sharpe_A": summ["sharpe_bookA"], "sharpe_B": summ["sharpe_bookB"],
                     "book_corr": summ["book_pnl_corr"], "churn": ch, "rerank_steps": steps})

    R = pd.DataFrame(rows).set_index("policy")
    print("\n================  STATIC vs RE-RANK  (full OOS, K=3 two-book)  ================")
    print(R[["sharpe", "totPnL", "maxDD", "sharpe_A", "sharpe_B", "book_corr", "churn", "rerank_steps"]].round(3).to_string())
    if "monthly" in R.index and "daily" in R.index:
        d = R.loc["monthly", "sharpe"] - R.loc["daily", "sharpe"]
        print(f"\nmonthly(static-at-retrain) − daily(rerank) Sharpe = {d:+.3f}  "
              f"({'STATIC wins → do NOT rerank' if d > 0 else 'RERANK wins'})")
    R.to_csv(OUTBASE/"summary.csv")
    print(f"\nsaved -> {OUTBASE/'summary.csv'}")


if __name__ == "__main__":
    main()
