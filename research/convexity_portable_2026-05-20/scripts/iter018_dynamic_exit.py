"""iter-018 — DYNAMIC, THESIS-BASED EXIT on REALIZED residual convergence (HUMAN idea).

Human idea: the held book holds 24h via 6 overlapping 4h sleeves. Instead of the FIXED hold,
exit a position EARLY when the residual it bet on has CONVERGED (the predicted mean-reversion
has been realized / the edge is gone), and keep holding while the residual RETAINS (reversion
still pending). This is DIFFERENT from iter-016: iter-016 set the hold from ENTRY-time signal
strength (PREDICT the horizon -> failed nested-OOS). THIS is a DYNAMIC exit on REALIZED
convergence -- observe the position's residual evolve and exit when the bet has played out.
That is PIT-valid (convergence is OBSERVED, not predicted).

CRUX (test both): "residual retains" splits into
  (a) not-yet-reverted-but-will (hold = correct), vs
  (b) moving AGAINST you / WIDENING (oversold getting more oversold = the iter-006 falling knife
      = the -57% DD). The human's rule "hold if retains" holds BOTH -- and holding (b) is the
      disaster. So test the human's rule AS STATED (P2) and a variant with a DIVERGENCE CUT (P3).

POLICIES (built on the SAME entries / same K=5 long-top-pred, short-bottom-pred sleeves):
  P1 = fixed 24h baseline (X117 production held-book = 6-sleeve average).
  P2 = human rule: exit-on-converge, HOLD otherwise (incl. divergence).
  P3 = exit-on-converge + CUT-on-divergence (protect the falling knife), hold only favorable-retain.

PIT mechanics. A sleeve entered at cycle t0 has a long basket L (top-K pred) and short basket S
(bottom-K pred). Each subsequent cycle u in (t0, t0+HOLD-1] we OBSERVE (realized, <= u, lagged):
  - realized cumulative residual the sleeve has CAPTURED since entry:
      cap_u = sum over its current legs of  sign * cum_alpha_resid(leg, t0->u)
    (alpha_resid per leg per 4h cycle = alpha_A; we cumulate REALIZED per-bar residuals; PIT).
  - the sleeve's EXPECTED move proxy at entry (entry pred magnitude) to define "most captured".
  - fresh re-score: each leg's FRESH pred at u (already in the preds file, PIT at u).

  CONVERGED  (exit) := the edge is gone -> the fresh basket signal has decayed toward 0 / flipped:
       mean(sign0 * fresh_pred) <= conv_band  (the bet's directional pred no longer present)
  DIVERGES   (cut, P3 only) := the realized capture has gone AGAINST the bet beyond a band:
       cap_u <= -div_band  (in residual-return units; the falling knife)
  RETAINS    := neither -> fresh signal still same-sign & material, not yet diverged -> HOLD.

We re-evaluate each LIVE sleeve every cycle; once exited/cut it stays out (no re-entry of that
sleeve). The held book at cycle u = average of all sleeves still LIVE at u (weights /HOLD as in
production so gross is comparable; we DO NOT renormalize up -- exiting reduces gross, which is the
honest behavior and lets G4 test whether the EXIT TIMING beats random exits of matched avg-hold).

DECISIVE honesty:
  G4 matched-random-exit placebo: does the convergence/divergence EXIT timing beat a RANDOM exit
     of matched average hold? (>= p95). If random exit of same avg-hold does as well, the effect
     is "just shorter holding" -> already covered by iter-014.
  episode-LOFO (EXT) + per-fold + nested-OOS of the bands (conv_band, div_band).

Reuses X122/X123 build_universe machinery verbatim for L/S selection, betas, regime, preds.
Outputs: results/iter018_dynexit_{HL70,EXT}.parquet (per-cycle pnl per policy). Console: all tables.
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
RC = REPO/"research/convexity_portable_2026-05-20/results/_cache"
OUT = REPO/"research/convexity_portable_2026-05-20/results"
HL70_PREDS = RC/"x64_HL70_V5mv3_7cx_filter_t0.3_preds.parquet"
EXT_PREDS = RC/"x113_ext_v0_preds.parquet"
KLINES = REPO/"data/ml/test/parquet/klines"

K = 5
HOLD = 6
WIN = 180
COSTS_BPS = [1.0, 3.0, 4.5]
SEED = 12345
N_PLACEBO = 200

EXT_EPISODES = [
    ("2022_luna",   "2022-05-01", "2022-07-31"),
    ("2022_ftx",    "2022-11-01", "2023-01-31"),
    ("2024_summer", "2024-06-01", "2024-09-30"),
    ("2025_q4",     "2025-09-01", "2025-12-31"),
]


def load_close(sym):
    sd = KLINES/sym/"5m"
    if not sd.exists(): return None
    dfs = [pd.read_parquet(f, columns=["open_time", "close"]) for f in sorted(sd.glob("*.parquet"))]
    df = pd.concat(dfs, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df.set_index("open_time")["close"].astype(np.float64)


def ann(x):
    x = pd.Series(x).dropna()
    return x.mean()/x.std()*np.sqrt(6*365) if len(x) > 2 and x.std() > 0 else np.nan


def stats(pnl_series):
    p = pd.Series(pnl_series).dropna(); pb = p*1e4
    eq = pb.cumsum(); dd = eq - eq.cummax(); mdd = dd.min()
    annr = pb.mean()*6*365
    cal = (annr/abs(mdd)) if (mdd < 0 and np.isfinite(mdd)) else np.nan
    return {"sharpe": ann(p), "maxDD": mdd, "calmar": cal,
            "totPnL": eq.iloc[-1] if len(eq) else np.nan, "pct_pos": (pb > 0).mean()*100}


def calmar_of(pnl_arr):
    pb = pd.Series(pnl_arr).dropna()*1e4
    if len(pb) < 3: return np.nan
    eq = pb.cumsum(); mdd = (eq - eq.cummax()).min()
    return (pb.mean()*6*365/abs(mdd)) if (mdd < 0 and np.isfinite(mdd)) else np.nan


def maxdd_of(pnl_arr):
    pb = pd.Series(pnl_arr).dropna()*1e4
    if len(pb) < 1: return np.nan
    eq = pb.cumsum(); return (eq - eq.cummax()).min()


# --------------------------------------------------------------------------- build
def build_universe(preds_path, label):
    """Returns per-cycle SLEEVE specs: for each cycle t, the base L/S basket and leg weights
    (a/b beta-neutral in side), plus per-symbol per-cycle alpha_resid (alpha_A, realized residual)
    and fresh pred (for re-scoring), regime, fold."""
    print(f"\n--- building {label} ({preds_path.name}) ---", flush=True)
    cols = ["symbol", "open_time", "pred", "return_pct", "alpha_A", "fold"]
    d = pd.read_parquet(preds_path, columns=cols)
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    d = d[(d["open_time"].dt.hour % 4 == 0) & (d["open_time"].dt.minute == 0)].copy()

    btc = load_close("BTCUSDT"); b4 = btc[(btc.index.hour % 4 == 0) & (btc.index.minute == 0)]
    br = np.log(b4/b4.shift(1)); bvar = br.rolling(WIN, min_periods=42).var()
    syms = sorted(d["symbol"].unique())
    beta_map = {}
    for sym in syms:
        c = load_close(sym)
        if c is None: continue
        c4 = c[(c.index.hour % 4 == 0) & (c.index.minute == 0)]
        r = np.log(c4/c4.shift(1)); ri, bi = r.align(br, join="inner")
        beta_map[sym] = (ri.rolling(WIN, min_periods=42).cov(bi)/bvar.reindex(ri.index).replace(0, np.nan)).shift(1)
    betas = pd.concat([s.rename(k) for k, s in beta_map.items()], axis=1)

    btc30 = (b4/b4.shift(WIN)-1).to_frame("b30").reset_index(); btc30["open_time"] = pd.to_datetime(btc30["open_time"], utc=True)
    d = d.merge(btc30, on="open_time", how="left").dropna(subset=["b30"])
    d["regime"] = np.where(d["b30"] > 0.10, "bull", np.where(d["b30"] < -0.10, "bear", "side"))

    # need mom30 for the bull leg (rank by momentum). build from betas index closes.
    mom_rows = []
    for sym in syms:
        c = load_close(sym)
        if c is None: continue
        c4 = c[(c.index.hour % 4 == 0) & (c.index.minute == 0)]
        mom_rows.append(pd.DataFrame({"symbol": sym, "open_time": c4.index,
                                      "mom30": (c4/c4.shift(WIN)-1).shift(1).values}))
    mom = pd.concat(mom_rows, ignore_index=True); mom["open_time"] = pd.to_datetime(mom["open_time"], utc=True)
    d = d.merge(mom, on=["symbol", "open_time"], how="left")

    times = sorted(d["open_time"].unique()); by_t = {ot: g for ot, g in d.groupby("open_time")}
    fold_by_time = {ot: int(g["fold"].iloc[0]) for ot, g in by_t.items() if "fold" in g.columns}
    t_idx = {t: i for i, t in enumerate(times)}

    # per-cycle per-symbol fresh pred and realized residual (alpha_A) and return_pct lookups
    pred_by_ts = {}; aA_by_ts = {}; ret_by_ts = {}
    for ot in times:
        g = by_t[ot]
        pred_by_ts[ot] = dict(zip(g["symbol"], g["pred"]))
        aA_by_ts[ot]   = dict(zip(g["symbol"], g["alpha_A"]))
        ret_by_ts[ot]  = dict(zip(g["symbol"], g["return_pct"]))

    # Per-cycle ENTRY sleeve spec: long basket L, short basket S, leg weights (a per long, b per short)
    # and per-leg ENTRY-pred sign and magnitude. (matches X123 base construction)
    sleeves = []  # list aligned to times: each is dict or None
    regimes = []
    for ot in times:
        g = by_t[ot]; rg = g["regime"].iloc[0]; regimes.append(rg)
        if rg == "bear":
            sleeves.append(None); continue
        key = "mom30" if rg == "bull" else "pred"
        gg = g.dropna(subset=[key])
        if len(gg) < 2*K:
            sleeves.append(None); continue
        gg = gg.sort_values(key); L = gg.tail(K)["symbol"].tolist(); S = gg.head(K)["symbol"].tolist()
        a = b = 1.0
        if rg == "side":
            brow = betas.loc[ot] if ot in betas.index else None
            if brow is not None:
                mbL = np.nanmean([brow.get(s, np.nan) for s in L]); mbS = np.nanmean([brow.get(s, np.nan) for s in S])
                if np.isfinite(mbL) and np.isfinite(mbS) and mbL > 0 and mbS > 0:
                    a = 2*mbS/(mbL+mbS); b = 2*mbL/(mbL+mbS)
        # entry leg weights (per name): long +a/K, short -b/K  (book gross ~ a+b ~ 2)
        legw = {}
        for s in L: legw[s] = legw.get(s, 0)+a/K
        for s in S: legw[s] = legw.get(s, 0)-b/K
        sleeves.append({"t0": ot, "regime": rg, "legw": legw, "L": L, "S": S})

    n_side = sum(r == "side" for r in regimes)
    print(f"  {len(syms)} syms, {len(times)} cycles, {pd.Timestamp(times[0]).date()}->{pd.Timestamp(times[-1]).date()};"
          f" side {n_side} ({n_side/len(times)*100:.0f}%)", flush=True)
    return dict(times=times, t_idx=t_idx, sleeves=sleeves, regimes=regimes,
                pred_by_ts=pred_by_ts, aA_by_ts=aA_by_ts, ret_by_ts=ret_by_ts,
                fold_by_time=fold_by_time, syms=syms)


# --------------------------------------------------------------------------- dynamic-exit engine
def run_policy(U, cost, policy="P1", conv_band=0.0, div_band=0.05, only_side=True,
               exit_override=None):
    """Build held-book PnL where each sleeve is held HOLD cycles UNLESS it is exited early per policy.
    PIT: the exit decision for sleeve entered at t0, evaluated AT cycle u, uses only info realized
    through the PRIOR cycle (u-1) -> the position taken AT u (and earning return from u to u+1) is
    decided with realized residual cap[t0..u-1] and fresh pred at u (known at u, PIT). The leg earns
    ret_by_ts[u] from u to u+1.

    policy:
      P1   = fixed: never early-exit (hold full HOLD).  (== production held-book)
      P2   = exit on CONVERGE (fresh-pred decayed); HOLD on divergence (human rule).
      P3   = exit on CONVERGE OR CUT on DIVERGE (cap < -div_band).
      "rand_exit" via exit_override: a dict {sleeve_entry_index: exit_step} forcing exit step.

    only_side: dynamic exit applies ONLY to side-regime sleeves (the mean-rev book = the DD source).
               Bull momentum sleeves always held full HOLD (out of scope of the convergence thesis).
    Returns (pnl_array, avg_hold, n_exits_early, total_active_sleeve_steps)."""
    times = U["times"]; n = len(times)
    sleeves = U["sleeves"]; pred_by_ts = U["pred_by_ts"]; aA_by_ts = U["aA_by_ts"]; ret_by_ts = U["ret_by_ts"]

    # For each entered sleeve, precompute its alive-steps under the policy. step s in [0, HOLD-1]
    # means the sleeve is live during cycle t0_idx + s (earning ret from that cycle to next).
    # We decide alive[s] sequentially; once dead, stays dead.
    # cap before deciding step s = sum over s'<s of realized residual earned by the (still-live) legs
    #   residual earned in step s' = sum_legs legw_sign * aA_by_ts[t0_idx+s'][leg]
    # sign of the bet on each leg = sign(legw) (long bet on +resid, short bet on -resid; for short,
    #   sign*resid = -resid, so a short leg "captures" when resid<0). cap>0 = bet paying off.
    sleeve_alive = [None]*n  # per entry index: list of bools len HOLD (or None if no sleeve)
    holds = []
    n_exit_early = 0
    for i in range(n):
        sp = sleeves[i]
        if sp is None:
            sleeve_alive[i] = None; continue
        t0 = i; rg = sp["regime"]; legw = sp["legw"]
        dyn = (not only_side) or (rg == "side")
        alive = [False]*HOLD
        cap = 0.0  # realized captured residual so far (PIT, only steps < s)
        for s in range(HOLD):
            u = t0 + s
            if u >= n: break
            # decide if live AT step s using info through step s-1 (cap) and fresh pred at u
            if s == 0:
                live = True
            elif not dyn:
                live = True
            elif exit_override is not None:
                # forced random exit: alive until exit_step, then dead
                live = (s < exit_override.get(i, HOLD))
            else:
                # fresh re-score: mean over legs of sign(entry leg)*fresh_pred at u
                fp = pred_by_ts[times[u]]
                vals = []
                for leg, w in legw.items():
                    pv = fp.get(leg, np.nan)
                    if np.isfinite(pv): vals.append(np.sign(w)*pv)
                fresh = np.mean(vals) if vals else 0.0
                converged = (fresh <= conv_band)          # directional pred decayed/flipped -> edge gone
                diverged = (cap <= -div_band)              # realized capture went against the bet
                if policy == "P2":
                    live = not converged                   # hold even if diverging (human rule)
                elif policy == "P3":
                    live = not (converged or diverged)     # cut the falling knife too
                else:  # P1
                    live = True
            if not live:
                break  # sleeve dead from here on
            alive[s] = True
            # accrue realized residual captured during step s (for NEXT step's decision)
            aA = aA_by_ts[times[u]]
            step_cap = 0.0
            for leg, w in legw.items():
                rv = aA.get(leg, np.nan)
                if np.isfinite(rv): step_cap += np.sign(w)*rv
            cap += step_cap
        sleeve_alive[i] = alive
        nlive = sum(alive)
        holds.append(nlive)
        if nlive < HOLD and (sp["regime"] == "side" or not only_side):
            n_exit_early += 1

    # Now assemble the held book per cycle: at cycle u, net weight = sum over sleeves entered at
    #   t0 = u-s (s in 0..HOLD-1) that are ALIVE at step s, of legw/HOLD.
    pnl = np.zeros(n)
    prev = {}
    for u in range(n):
        net = {}
        for s in range(HOLD):
            t0 = u - s
            if t0 < 0: break
            sp = sleeves[t0]; al = sleeve_alive[t0]
            if sp is None or al is None: continue
            if not al[s]: continue  # this sleeve already exited before step s
            for leg, w in sp["legw"].items():
                net[leg] = net.get(leg, 0.0) + w/HOLD
        alls = set(net) | set(prev)
        turn = sum(abs(net.get(s, 0)-prev.get(s, 0)) for s in alls)
        rl = ret_by_ts[times[u]]
        c = sum(net.get(s, 0)*rl.get(s, 0.0) for s in net if np.isfinite(rl.get(s, 0.0)))
        if not np.isfinite(c): c = 0.0
        pnl[u] = c - turn*0.5*cost
        prev = net
    avg_hold = np.mean(holds) if holds else np.nan
    return pnl, avg_hold, n_exit_early, holds


# --------------------------------------------------------------------------- G4 matched-random-exit
def g4_random_exit(U, cost, rng, real_holds_side, policy_label):
    """Matched-random-exit placebo: for each side sleeve, force a RANDOM exit step drawn so the
    AVERAGE hold matches the real policy's avg side hold. Does the real EXIT TIMING beat random
    exits of matched avg-hold? We sample exit steps from the SAME empirical distribution of holds
    the real policy produced (so avg & dist match), but assigned to RANDOM sleeves.
    Returns (real not needed here) placebo cal/dd arrays — caller compares to the real policy."""
    times = U["times"]; n = len(times); sleeves = U["sleeves"]
    side_entry_idx = [i for i in range(n) if sleeves[i] is not None and sleeves[i]["regime"] == "side"]
    holds_pool = np.array(real_holds_side, dtype=int)
    cals = np.empty(N_PLACEBO); dds = np.empty(N_PLACEBO); tots = np.empty(N_PLACEBO)
    for k in range(N_PLACEBO):
        # shuffle the hold lengths across the side sleeves
        perm = rng.permutation(holds_pool)
        override = {}
        for j, i in enumerate(side_entry_idx):
            override[i] = int(perm[j % len(perm)])
        pnl, _, _, _ = run_policy(U, cost, policy="rand", exit_override=override, only_side=True)
        cals[k] = calmar_of(pnl); dds[k] = maxdd_of(pnl); tots[k] = (pd.Series(pnl)*1e4).sum()
    return cals, dds, tots


# --------------------------------------------------------------------------- nested-OOS of bands
def nested_oos_bands(U, cost):
    """Choose (conv_band, div_band) for P3 on PAST folds, apply forward; report nested-OOS Calmar
    vs P1. Grid is small + structural-ish. Folds are the preds' fold ids (expanding)."""
    times = U["times"]; fbt = U["fold_by_time"]
    folds = sorted(set(fbt.values()))
    fa = np.array([fbt.get(t, -1) for t in times])
    conv_grid = [-0.10, 0.0, 0.10, 0.25]
    div_grid = [0.03, 0.05, 0.10]
    # precompute P3 pnl for each (cb,db) and P1
    p1_pnl, _, _, _ = run_policy(U, cost, "P1")
    cache = {}
    for cb in conv_grid:
        for db in div_grid:
            cache[(cb, db)] = run_policy(U, cost, "P3", conv_band=cb, div_band=db)[0]
    # nested: for each fold f (from 2nd onward), pick best (cb,db) by Calmar on folds < f, apply to f
    oos_p3 = np.full(len(times), np.nan); oos_p1 = np.full(len(times), np.nan)
    chosen = []
    for fi, f in enumerate(folds):
        if fi == 0: continue
        past = fa < f; cur = fa == f
        if past.sum() < 10 or cur.sum() < 3: continue
        best = None; bestcal = -1e9
        for key, pnl in cache.items():
            cal = calmar_of(pnl[past])
            if np.isfinite(cal) and cal > bestcal: bestcal = cal; best = key
        if best is None: continue
        oos_p3[cur] = cache[best][cur]
        oos_p1[cur] = p1_pnl[cur]
        chosen.append((f, best))
    m = np.isfinite(oos_p3)
    cal_p3 = calmar_of(oos_p3[m]); cal_p1 = calmar_of(oos_p1[m])
    return cal_p1, cal_p3, chosen


# --------------------------------------------------------------------------- reporting
def per_fold(label, times, p1, parm, fold_by_time, armname):
    folds = sorted(set(fold_by_time.values())) if fold_by_time else []
    if not folds: return
    fa = np.array([fold_by_time.get(t, -1) for t in times])
    nb = 0; ne = 0
    print(f"\n  [G5 per-fold {armname} vs P1 — {label}] @4.5bps", flush=True)
    print(f"  {'fold':>5}{'n':>6}{'P1cal':>9}{'armCal':>9}{'P1DD':>9}{'armDD':>9}{'better?':>9}", flush=True)
    for f in folds:
        m = fa == f
        if m.sum() < 3: continue
        ne += 1; sb = stats(p1[m]); sa = stats(parm[m])
        better = (np.isfinite(sa["calmar"]) and np.isfinite(sb["calmar"]) and sa["calmar"] >= sb["calmar"])
        if better: nb += 1
        print(f"  {f:>5}{int(m.sum()):>6}{sb['calmar']:>+9.2f}{sa['calmar']:>+9.2f}"
              f"{sb['maxDD']:>+9.0f}{sa['maxDD']:>+9.0f}{('yes' if better else 'NO'):>9}", flush=True)
    print(f"  {armname} Calmar >= P1 in {nb}/{ne} folds (G5 spirit >=6/9)", flush=True)


def episode_lofo(label, times, p1, parm, episodes, armname):
    ti = pd.DatetimeIndex(times)
    cb = calmar_of(p1); ca = calmar_of(parm); full = ca - cb
    print(f"\n  [G5 episode-LOFO {armname} — {label}] full: P1 {cb:+.2f} arm {ca:+.2f} lift {full:+.2f}", flush=True)
    allpos = True
    for ename, a, bnd in episodes:
        m = (ti >= pd.Timestamp(a, tz="UTC")) & (ti <= pd.Timestamp(bnd, tz="UTC"))
        keep = ~np.asarray(m)
        cbb = calmar_of(p1[keep]); caa = calmar_of(parm[keep])
        lift = caa - cbb if (np.isfinite(caa) and np.isfinite(cbb)) else np.nan
        pos = (np.isfinite(lift) and lift > 0)
        if not pos: allpos = False
        print(f"    drop {ename:<14} P1 {cbb:+.2f} arm {caa:+.2f} lift {lift:+.2f} {'>0' if pos else 'NEG'}", flush=True)
    print(f"    episode-LOFO stays >0 dropping each: {'PASS' if allpos else 'FAIL'}", flush=True)


def episode_pnl(label, times, pol_pnl, episodes):
    ti = pd.DatetimeIndex(times)
    print(f"\n  [per-episode totPnL & maxDD by policy — {label}] @4.5bps", flush=True)
    print(f"  {'episode':<14}{'n':>5}{'P1pnl':>9}{'P2pnl':>9}{'P3pnl':>9}{'P1DD':>9}{'P2DD':>9}{'P3DD':>9}", flush=True)
    for ename, a, bnd in episodes:
        m = (ti >= pd.Timestamp(a, tz="UTC")) & (ti <= pd.Timestamp(bnd, tz="UTC"))
        if m.sum() < 5:
            print(f"  {ename:<14}{int(m.sum()):>5}  (too few)", flush=True); continue
        mm = np.asarray(m)
        row = f"  {ename:<14}{int(m.sum()):>5}"
        for p in ("P1", "P2", "P3"):
            row += f"{(np.asarray(pol_pnl[p])[mm]*1e4).sum():>+9.0f}"
        for p in ("P1", "P2", "P3"):
            row += f"{maxdd_of(np.asarray(pol_pnl[p])[mm]):>+9.0f}"
        print(row, flush=True)


# --------------------------------------------------------------------------- run
def run_universe(label, preds_path, rng, is_ext=False):
    U = build_universe(preds_path, label)
    times = U["times"]

    # default bands: conv_band=0 (exit when fresh basket pred no longer positive in bet direction),
    # div_band=0.05 (5% adverse residual capture = the falling-knife cut, ~2.6 sigma of 4h resid)
    CONV = 0.0; DIV = 0.05
    pol_pnl = {}; pol_hold = {}; pol_exits = {}; pol_holds_side = {}
    print(f"\n  [headline by policy — {label}] @4.5bps  (conv_band={CONV}, div_band={DIV})", flush=True)
    print(f"  {'policy':>6}{'Sharpe':>8}{'maxDD':>9}{'Calmar':>8}{'totPnL':>9}{'%pos':>7}{'avgHold':>9}{'earlyExit':>10}", flush=True)
    for p in ("P1", "P2", "P3"):
        pnl, avgh, nex, holds = run_policy(U, 4.5e-4, p, conv_band=CONV, div_band=DIV, only_side=True)
        pol_pnl[p] = pnl; pol_hold[p] = avgh; pol_exits[p] = nex
        s = stats(pnl)
        print(f"  {p:>6}{s['sharpe']:>+8.2f}{s['maxDD']:>+9.0f}{s['calmar']:>+8.2f}{s['totPnL']:>+9.0f}"
              f"{s['pct_pos']:>7.1f}{avgh:>9.2f}{nex:>10}", flush=True)

    # recompute side-only holds explicitly for G4 (need per-sleeve alive counts for P2/P3)
    def side_holds(policy):
        # re-run to extract holds for side sleeves only
        _, _, _, holds_all = run_policy(U, 4.5e-4, policy, conv_band=CONV, div_band=DIV, only_side=True)
        # holds_all is per-entered-sleeve (any regime); map back to side
        sidx = [i for i in range(len(times)) if U["sleeves"][i] is not None]
        side_h = [holds_all[j] for j, i in enumerate(sidx) if U["sleeves"][i]["regime"] == "side"]
        return side_h
    holds_P2 = side_holds("P2"); holds_P3 = side_holds("P3")
    print(f"\n  side-sleeve avg hold: P2 {np.mean(holds_P2):.2f}  P3 {np.mean(holds_P3):.2f}  (P1=6.00)", flush=True)

    # G8 cost sensitivity
    print(f"\n  [G8 cost sensitivity — {label}] Calmar by policy", flush=True)
    for cb in COSTS_BPS:
        row = f"    @{cb:>4.1f}bps "
        for p in ("P1", "P2", "P3"):
            pnl, _, _, _ = run_policy(U, cb*1e-4, p, conv_band=CONV, div_band=DIV, only_side=True)
            row += f" {p} Cal {calmar_of(pnl):+.2f} "
        print(row, flush=True)

    # G4 matched-random-exit placebo for P2 and P3
    for p, hsd in (("P2", holds_P2), ("P3", holds_P3)):
        real = pol_pnl[p]; real_cal = calmar_of(real); real_dd = maxdd_of(real); real_tot=(pd.Series(real)*1e4).sum()
        cals, dds, tots = g4_random_exit(U, 4.5e-4, rng, hsd, p)
        crank = (cals < real_cal).mean()*100; ddrank = (dds < real_dd).mean()*100
        print(f"\n  [G4 matched-random-exit placebo — {label} {p}] ({N_PLACEBO} seeds, side avg hold {np.mean(hsd):.2f})", flush=True)
        print(f"    real {p}: Calmar {real_cal:+.2f} maxDD {real_dd:+.0f} totPnL {real_tot:+.0f}", flush=True)
        print(f"    placebo Calmar p50 {np.nanpercentile(cals,50):+.2f} p95 {np.nanpercentile(cals,95):+.2f} "
              f"-> rank p{crank:.0f} {'PASS(>=p95)' if crank>=95 else 'FAIL'}", flush=True)
        print(f"    placebo maxDD  p50 {np.nanpercentile(dds,50):+.0f} p05 {np.nanpercentile(dds,5):+.0f} "
              f"(real DD rank p{ddrank:.0f}; higher=less-deep-than-random)", flush=True)

    # per-fold + nested-OOS
    per_fold(label, times, pol_pnl["P1"], pol_pnl["P2"], U["fold_by_time"], "P2")
    per_fold(label, times, pol_pnl["P1"], pol_pnl["P3"], U["fold_by_time"], "P3")
    cal_p1n, cal_p3n, chosen = nested_oos_bands(U, 4.5e-4)
    print(f"\n  [G3 nested-OOS of (conv_band,div_band) for P3 — {label}]", flush=True)
    print(f"    nested-OOS: P1 {cal_p1n:+.2f}  P3(nested) {cal_p3n:+.2f}  lift {cal_p3n-cal_p1n:+.2f}", flush=True)
    print(f"    chosen per fold: {chosen}", flush=True)

    if is_ext:
        episode_pnl(label, times, pol_pnl, EXT_EPISODES)
        episode_lofo(label, times, pol_pnl["P1"], pol_pnl["P2"], EXT_EPISODES, "P2")
        episode_lofo(label, times, pol_pnl["P1"], pol_pnl["P3"], EXT_EPISODES, "P3")

    # per-cycle parquet
    out = {"open_time": pd.to_datetime(times), "fold": [U["fold_by_time"].get(t, -1) for t in times],
           "regime": U["regimes"]}
    for p in ("P1", "P2", "P3"): out[f"pnl_{p}"] = pol_pnl[p]
    pd.DataFrame(out).to_parquet(OUT/f"iter018_dynexit_{label}.parquet", index=False)
    print(f"  per-cycle -> iter018_dynexit_{label}.parquet", flush=True)
    return pol_pnl


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    print("=== iter-018: DYNAMIC thesis-based EXIT on realized residual convergence ===", flush=True)
    print(f"P1 fixed-24h | P2 exit-on-converge(hold-divergence) | P3 exit-on-converge + cut-on-diverge. "
          f"K={K} HOLD={HOLD} win={WIN} seed={SEED}", flush=True)
    run_universe("HL70", HL70_PREDS, rng)
    run_universe("EXT", EXT_PREDS, rng, is_ext=True)
    print(f"\nDone [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
