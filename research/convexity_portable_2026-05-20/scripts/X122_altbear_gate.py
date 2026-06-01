"""X122 — parameter-free 2-axis alt-bear SIDE gate (iter-007 F1).

Spec (research/handoff.md, iter-007): a PARAMETER-FREE regime-map branch change on the
held book. Regime map becomes:

    bull (BTC-30d > +10%)              -> momentum (mom30), long top-K / short bot-K  UNCHANGED
    bear (BTC-30d < -10%)              -> FLAT (emit {})                              UNCHANGED
    side AND alt_index_30d <  btc_30d  -> FLAT (emit {})           <-- NEW (parameter-free)
    side AND alt_index_30d >= btc_30d  -> mean-rev (pred), beta-neutral leg sizing    UNCHANGED

The flagged side cycles emit an empty sleeve {} exactly like a bear cycle; the held book
decays prior sleeves normally (HOLD=6, K=5, equal weight). NO new model feature, NO retrain,
NO swept threshold (the boundary is a structural ±0 RELATIVE comparison `alt30 < btc30`, like
the ±10% BTC regime rule) -> G3 is WAIVED. This is a regime-DEFINITION change, not a sizing
overlay: a flagged cycle contributes nothing new and prior sleeves simply age out.

ALT-INDEX (PIT, look-ahead critical):
  alt_index_30d = trailing-30d (180x5m bars on the 4h grid) cumulative log-return of the
  EQUAL-WEIGHT, ex-BTC, ex-ETH subset of THE PANEL'S OWN traded universe, `.shift(1)` lagged.
  Compared at the SAME lag to the existing PIT btc_30d (also trailing-180-bar). Per-universe:
  HL70 alt-index from HL70 alts, EXT from the 23 alts, S44 from the 44 alts. No cross-universe
  carry, no forward window, no full-sample normalization.

Arms:
  base  = X117 production (bull mom / side mean-rev BN / bear FLAT). REPRODUCES X117 on HL70.
  f1    = base, EXCEPT side cycles with (alt30 < btc30) emit {} (FLAT).

Universes:
  HL70 (70-sym, x64 preds)          PRODUCTION — one DD episode.
  EXT  (23-sym 2021-26, x113 preds) MULTI-EPISODE validation (2022 LUNA/FTX, 2024 summer, 2025-Q4).
  S44  (44-sym 2023-26, x70 preds)  transport robustness.

Emits per-universe per-cycle parquet (for G4 matched-count side-pool placebo, G6 paired CI),
HL70 per-fold + fold-LOFO, and the DECISIVE EXT multi-episode + episode-LOFO table. Explicit
NaN guards (the totPnL +nan bug). Seeded RNG. Does NOT modify X116/X117/X121 or cached preds.
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
S44_PREDS = RC/"x70_v0_3yr_preds.parquet"
KLINES = REPO/"data/ml/test/parquet/klines"

K = 5
HOLD = 6
WIN = 180                       # 180 bars on the 4h grid = 30 days (matches X117/X121 betas/mom30)
COSTS_BPS = [1.0, 3.0, 4.5]
SEED = 12345
N_PLACEBO = 200

# Calendar DD episodes on EXT (per iter-007 insight). 2021_blowoff included as a 5th window
# for diagnostic completeness; the pre-registered G5 bar is the 4 alt-bear episodes below.
EXT_EPISODES = [
    ("2022_luna",   "2022-05-01", "2022-07-31"),
    ("2022_ftx",    "2022-11-01", "2023-01-31"),
    ("2024_summer", "2024-06-01", "2024-09-30"),
    ("2025_q4",     "2025-09-01", "2025-12-31"),
]


# --------------------------------------------------------------------------- helpers
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
    """pnl_series in raw return units. Contract metrics (bps where shown)."""
    p = pd.Series(pnl_series).dropna()
    pb = p*1e4
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
    """Build base-arm and F1-arm cycle weights + PIT alt30/btc30/flag per cycle.

    Returns dict with times, cyc_w_base, cyc_w_f1, rs, fold_by_time, regimes,
    alt30_by_time, btc30_by_time, flag_by_time, is_side.
    """
    print(f"\n--- building {label} ({preds_path.name}) ---", flush=True)
    cols = ["symbol", "open_time", "pred", "return_pct", "fold"]
    d = pd.read_parquet(preds_path, columns=cols)
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    d = d[(d["open_time"].dt.hour % 4 == 0) & (d["open_time"].dt.minute == 0)].copy()

    btc = load_close("BTCUSDT"); b4 = btc[(btc.index.hour % 4 == 0) & (btc.index.minute == 0)]
    br = np.log(b4/b4.shift(1)); bvar = br.rolling(WIN, min_periods=42).var()
    syms = sorted(d["symbol"].unique())
    mom_rows = []; beta_map = {}; ret_map = {}
    for sym in syms:
        c = load_close(sym)
        if c is None: continue
        c4 = c[(c.index.hour % 4 == 0) & (c.index.minute == 0)]
        mom_rows.append(pd.DataFrame({"symbol": sym, "open_time": c4.index,
                                      "mom30": (c4/c4.shift(WIN)-1).shift(1).values}))
        r = np.log(c4/c4.shift(1)); ri, bi = r.align(br, join="inner")
        beta_map[sym] = (ri.rolling(WIN, min_periods=42).cov(bi)/bvar.reindex(ri.index).replace(0, np.nan)).shift(1)
        ret_map[sym] = r
    mom = pd.concat(mom_rows, ignore_index=True); mom["open_time"] = pd.to_datetime(mom["open_time"], utc=True)
    betas = pd.concat([s.rename(k) for k, s in beta_map.items()], axis=1)
    ret4 = pd.concat([s.rename(k) for k, s in ret_map.items()], axis=1).sort_index()
    d = d.merge(mom, on=["symbol", "open_time"], how="left")
    btc30 = (b4/b4.shift(WIN)-1).to_frame("b30").reset_index(); btc30["open_time"] = pd.to_datetime(btc30["open_time"], utc=True)
    d = d.merge(btc30, on="open_time", how="left").dropna(subset=["b30"])
    d["regime"] = np.where(d["b30"] > 0.10, "bull", np.where(d["b30"] < -0.10, "bear", "side"))

    times = sorted(d["open_time"].unique()); by_t = {ot: g for ot, g in d.groupby("open_time")}
    fold_by_time = {ot: int(g["fold"].iloc[0]) for ot, g in by_t.items() if "fold" in g.columns}

    # ---- PIT alt-index of THIS universe's own traded alts (ex BTC/ETH), .shift(1) lagged ----
    altcols = [c for c in ret4.columns if c not in ("BTCUSDT", "ETHUSDT")]
    altidx = ret4[altcols].mean(axis=1)                       # eq-weight alt 4h log-return
    alt_cum = altidx.cumsum()
    alt30_full = (alt_cum - alt_cum.shift(WIN)).shift(1)       # trailing-30d cum log-ret, LAGGED (PIT)
    ti = pd.DatetimeIndex(times)
    alt30 = alt30_full.reindex(ti)
    # btc30 lagged the SAME way (.shift(1)) so the comparison is apples-to-apples at matched lag
    b30_lag = (b4/b4.shift(WIN)-1).shift(1).reindex(ti)
    alt30_by_time = dict(zip(times, alt30.values))
    btc30_by_time = dict(zip(times, b30_lag.values))

    cyc_w_base = []; cyc_w_f1 = []; rs = []; regimes = []; flag_by_time = {}
    for ot in times:
        g = by_t[ot]; rg = g["regime"].iloc[0]; regimes.append(rg)
        rs.append(dict(zip(g["symbol"], g["return_pct"])))

        a30 = alt30_by_time.get(ot, np.nan); b30 = btc30_by_time.get(ot, np.nan)
        # F1 flag: side regime AND alts underperform BTC over trailing 30d (both PIT-lagged).
        # NaN-guard: if alt30/btc30 unavailable (warm-up), do NOT flag (fall through to base).
        flag = (rg == "side") and np.isfinite(a30) and np.isfinite(b30) and (a30 < b30)
        flag_by_time[ot] = bool(flag)

        if rg == "bear":
            cyc_w_base.append({}); cyc_w_f1.append({}); continue
        if rg == "side" and flag:
            cyc_w_f1.append({})            # THE ONLY CHANGE vs base: flagged side cycle -> FLAT

        key = "mom30" if rg == "bull" else "pred"; gg = g.dropna(subset=[key])
        if len(gg) < 2*K:
            cyc_w_base.append({})
            if rg == "bull" or (rg == "side" and not flag): cyc_w_f1.append({})
            continue
        gg = gg.sort_values(key); L = gg.tail(K)["symbol"].tolist(); S = gg.head(K)["symbol"].tolist()
        a = b = 1.0
        if rg == "side":
            brow = betas.loc[ot] if ot in betas.index else None
            if brow is not None:
                mbL = np.nanmean([brow.get(s, np.nan) for s in L]); mbS = np.nanmean([brow.get(s, np.nan) for s in S])
                if np.isfinite(mbL) and np.isfinite(mbS) and mbL > 0 and mbS > 0:
                    a = 2*mbS/(mbL+mbS); b = 2*mbL/(mbL+mbS)
        w = {}
        for s in L: w[s] = w.get(s, 0)+a/K
        for s in S: w[s] = w.get(s, 0)-b/K
        cyc_w_base.append(w)
        if rg == "bull" or (rg == "side" and not flag): cyc_w_f1.append(w)

    assert len(cyc_w_base) == len(times) == len(cyc_w_f1), "arm length mismatch"
    is_side = [r == "side" for r in regimes]
    n_side = sum(is_side); n_flag = sum(1 for t in times if flag_by_time[t])
    print(f"  {len(syms)} syms, {len(times)} cycles, {pd.Timestamp(times[0]).date()}->{pd.Timestamp(times[-1]).date()}; "
          f"side {n_side} ({n_side/len(times)*100:.0f}%), F1-flagged {n_flag} of side "
          f"({(n_flag/n_side*100) if n_side else 0:.0f}%)", flush=True)
    return dict(times=list(times), cyc_w_base=cyc_w_base, cyc_w_f1=cyc_w_f1, rs=rs,
                fold_by_time=fold_by_time, regimes=regimes, is_side=is_side,
                alt30_by_time=alt30_by_time, btc30_by_time=btc30_by_time, flag_by_time=flag_by_time)


# --------------------------------------------------------------------------- engine
def heldbook(times, cyc_w, rs, cost):
    """Held-book loop identical to X117/X121. Arm fully determined by cyc_w list."""
    prev = {}; pnl = []
    for t in range(len(cyc_w)):
        active = cyc_w[max(0, t-HOLD+1):t+1]
        net = {}
        for w in active:
            for s, wt in w.items(): net[s] = net.get(s, 0)+wt/HOLD
        alls = set(net) | set(prev)
        turn = sum(abs(net.get(s, 0)-prev.get(s, 0)) for s in alls)
        rl = rs[t]
        cyc = sum(net.get(s, 0)*rl.get(s, 0.0) for s in net if np.isfinite(rl.get(s, 0.0)))
        if not np.isfinite(cyc): cyc = 0.0          # explicit NaN guard (totPnL +nan bug)
        pnl.append(cyc - turn*0.5*cost)
        prev = net
    return np.asarray(pnl, dtype=np.float64)


def heldbook_flatlist(times, cyc_w_base, rs, cost, flat_mask):
    """Build a held book that FLATs the cycles where flat_mask[t] is True (emit {}),
    otherwise uses the base-arm weight. Used for the matched-count placebo so the placebo
    uses the EXACT same construction/decay machinery as F1 (just a different flag set)."""
    cyc = [({} if flat_mask[t] else cyc_w_base[t]) for t in range(len(times))]
    return heldbook(times, cyc, rs, cost)


# --------------------------------------------------------------------------- reporting
def per_fold_report(label, times, pnl_base, pnl_f1, fold_by_time):
    folds = sorted(set(fold_by_time.values())) if fold_by_time else []
    if not folds:
        print("  (no fold column)", flush=True); return 0, 0
    fold_arr = np.array([fold_by_time.get(t, -1) for t in times])
    n_better = 0; n_eval = 0
    print(f"  {'fold':>5}{'n':>6}{'baseDD':>9}{'f1DD':>9}{'DDimp%':>8}"
          f"{'baseSh':>8}{'f1Sh':>8}{'baseCal':>9}{'f1Cal':>9}", flush=True)
    for f in folds:
        m = fold_arr == f
        if m.sum() < 3: continue
        n_eval += 1
        sb = stats(pnl_base[m]); sf = stats(pnl_f1[m])
        bdd, fdd = sb["maxDD"], sf["maxDD"]
        ddimp = (1-abs(fdd)/abs(bdd))*100 if (bdd < 0 and np.isfinite(bdd)) else np.nan
        cb = sb["calmar"]; cf = sf["calmar"]
        if np.isfinite(cb) and np.isfinite(cf) and cf >= cb: n_better += 1
        elif np.isfinite(ddimp) and ddimp > 0 and not np.isfinite(cb): n_better += 1
        print(f"  {f:>5}{int(m.sum()):>6}{bdd:>+9.0f}{fdd:>+9.0f}{ddimp:>+8.1f}"
              f"{sb['sharpe']:>+8.2f}{sf['sharpe']:>+8.2f}{cb:>+9.2f}{cf:>+9.2f}", flush=True)
    return n_better, n_eval


def fold_lofo_report(label, times, pnl_base, pnl_f1, fold_by_time):
    folds = sorted(set(fold_by_time.values())) if fold_by_time else []
    if not folds:
        print("  (no fold column for LOFO)", flush=True); return
    fold_arr = np.array([fold_by_time.get(t, -1) for t in times])
    cal_b_all = calmar_of(pnl_base); cal_f_all = calmar_of(pnl_f1); full_lift = cal_f_all - cal_b_all
    print(f"  FULL: base Calmar {cal_b_all:+.2f} | f1 Calmar {cal_f_all:+.2f} | lift {full_lift:+.2f}", flush=True)
    print(f"  {'drop':>6}{'baseCal':>9}{'f1Cal':>9}{'lift':>8}{'Δvs_full':>10}{'>0?':>5}", flush=True)
    all_pos = True
    for f in folds:
        keep = fold_arr != f
        cb = calmar_of(pnl_base[keep]); cf = calmar_of(pnl_f1[keep])
        lift = cf - cb if (np.isfinite(cf) and np.isfinite(cb)) else np.nan
        dvs = lift - full_lift if np.isfinite(lift) else np.nan
        pos = "yes" if (np.isfinite(lift) and lift > 0) else "NO"
        if not (np.isfinite(lift) and lift > 0): all_pos = False
        print(f"  {('-f'+str(f)):>6}{cb:>+9.2f}{cf:>+9.2f}{lift:>+8.2f}{dvs:>+10.2f}{pos:>5}", flush=True)
    print(f"  fold-LOFO lift stays >0 dropping EACH fold: {'PASS' if all_pos else 'FAIL'}", flush=True)


def episode_report(times, pnl_base, pnl_f1, episodes):
    """EXT multi-episode: per-episode maxDD/Calmar/PnL base vs F1; need improvement in >=3/4.
    Then episode-LOFO: drop each episode's cycles, recompute full-series Calmar lift; >0 each."""
    ti = pd.DatetimeIndex(times)
    pb = pd.Series(pnl_base, index=ti); pf = pd.Series(pnl_f1, index=ti)
    print(f"  {'episode':<14}{'n':>5}{'baseDD':>9}{'f1DD':>9}{'DDimp%':>8}"
          f"{'baseCal':>9}{'f1Cal':>9}{'basePnL':>9}{'f1PnL':>9}{'improved?':>11}", flush=True)
    n_improved = 0; n_eval = 0
    ep_masks = {}
    for ename, a, bnd in episodes:
        m = (ti >= pd.Timestamp(a, tz="UTC")) & (ti <= pd.Timestamp(bnd, tz="UTC"))
        ep_masks[ename] = m
        if m.sum() < 5:
            print(f"  {ename:<14}{int(m.sum()):>5}  (too few cycles)", flush=True); continue
        n_eval += 1
        sb = stats(pb[m].values); sf = stats(pf[m].values)
        bdd, fdd = sb["maxDD"], sf["maxDD"]
        ddimp = (1-abs(fdd)/abs(bdd))*100 if (bdd < 0 and np.isfinite(bdd)) else np.nan
        # "improved" = maxDD strictly less negative (the gate's stated purpose is DD reduction)
        improved = np.isfinite(ddimp) and ddimp > 0.5
        if improved: n_improved += 1
        print(f"  {ename:<14}{int(m.sum()):>5}{bdd:>+9.0f}{fdd:>+9.0f}{ddimp:>+8.1f}"
              f"{sb['calmar']:>+9.2f}{sf['calmar']:>+9.2f}{sb['totPnL']:>+9.0f}{sf['totPnL']:>+9.0f}"
              f"{('YES' if improved else 'no'):>11}", flush=True)
    print(f"  episodes with maxDD improvement: {n_improved}/{n_eval}  "
          f"(G5 bar: >=3/4) -> {'PASS' if n_improved >= 3 else 'FAIL'}", flush=True)

    # ---- episode-LOFO: drop each episode's cycles, recompute full-series F1-vs-base lift ----
    cal_b_all = calmar_of(pnl_base); cal_f_all = calmar_of(pnl_f1); full_lift = cal_f_all - cal_b_all
    print(f"\n  --- EXT episode-LOFO: drop each episode, recompute full-series Calmar lift ---", flush=True)
    print(f"  FULL: base Calmar {cal_b_all:+.2f} | f1 Calmar {cal_f_all:+.2f} | lift {full_lift:+.2f}", flush=True)
    print(f"  {'drop_ep':<14}{'baseCal':>9}{'f1Cal':>9}{'lift':>8}{'Δvs_full':>10}{'>0?':>5}", flush=True)
    all_pos = True
    for ename, m in ep_masks.items():
        keep = ~np.asarray(m)
        cb = calmar_of(pnl_base[keep]); cf = calmar_of(pnl_f1[keep])
        lift = cf - cb if (np.isfinite(cf) and np.isfinite(cb)) else np.nan
        dvs = lift - full_lift if np.isfinite(lift) else np.nan
        pos = "yes" if (np.isfinite(lift) and lift > 0) else "NO"
        if not (np.isfinite(lift) and lift > 0): all_pos = False
        print(f"  {('-'+ename):<14}{cb:>+9.2f}{cf:>+9.2f}{lift:>+8.2f}{dvs:>+10.2f}{pos:>5}", flush=True)
    print(f"  episode-LOFO lift stays >0 dropping EACH episode: {'PASS' if all_pos else 'FAIL'}", flush=True)


def matched_placebo(label, U, cost, rng):
    """G4: FLAT the SAME COUNT of RANDOM side cycles as F1 flags, drawn from the side-cycle
    pool, using the same construction machinery. >=N_PLACEBO seeds. Report F1's percentile
    rank of Calmar (and maxDD) vs the placebo distribution. MANDATORY >= p95."""
    times = U["times"]; rs = U["rs"]; cyc_w_base = U["cyc_w_base"]
    side_idx = np.array([i for i, s in enumerate(U["is_side"]) if s], dtype=int)
    flag_idx = np.array([i for i, t in enumerate(times) if U["flag_by_time"][t]], dtype=int)
    n_flag = len(flag_idx)
    if n_flag == 0 or len(side_idx) < n_flag:
        print(f"  (G4 {label}: n_flag={n_flag}, side_pool={len(side_idx)} — cannot run placebo)", flush=True)
        return
    # real F1
    f1_mask = np.zeros(len(times), dtype=bool); f1_mask[flag_idx] = True
    pnl_f1 = heldbook_flatlist(times, cyc_w_base, rs, cost, f1_mask)
    real_cal = calmar_of(pnl_f1); real_dd = maxdd_of(pnl_f1)
    cals = np.empty(N_PLACEBO); dds = np.empty(N_PLACEBO)
    for i in range(N_PLACEBO):
        pick = rng.choice(side_idx, size=n_flag, replace=False)
        mask = np.zeros(len(times), dtype=bool); mask[pick] = True
        pp = heldbook_flatlist(times, cyc_w_base, rs, cost, mask)
        cals[i] = calmar_of(pp); dds[i] = maxdd_of(pp)
    cal_rank = (cals < real_cal).mean()*100
    dd_rank = (dds < real_dd).mean()*100   # higher (less negative) maxDD is better -> rank of being above
    print(f"  G4 matched-count side-pool placebo ({N_PLACEBO} seeds, FLAT {n_flag} of {len(side_idx)} side cycles):", flush=True)
    print(f"     real F1 Calmar {real_cal:+.2f} | placebo Calmar p50 {np.nanpercentile(cals,50):+.2f} "
          f"p95 {np.nanpercentile(cals,95):+.2f} max {np.nanmax(cals):+.2f} -> rank p{cal_rank:.0f} "
          f"{'PASS(>=p95)' if cal_rank >= 95 else 'FAIL'}", flush=True)
    print(f"     real F1 maxDD  {real_dd:+.0f} | placebo maxDD p50 {np.nanpercentile(dds,50):+.0f} "
          f"p95 {np.nanpercentile(dds,95):+.0f} -> rank p{dd_rank:.0f}", flush=True)


def paired_ci(label, times, pnl_base, pnl_f1, fold_by_time, rng):
    """G6: block-bootstrap (blocks by fold) the paired per-cycle PnL diff (F1 - base). CI must
    not cross 0. Resample folds with replacement; report mean diff and 95% CI in bps."""
    diff = (pnl_f1 - pnl_base)*1e4
    fold_arr = np.array([fold_by_time.get(t, -1) for t in times])
    folds = sorted(set(fold_arr.tolist()))
    blocks = {f: diff[fold_arr == f] for f in folds}
    means = np.empty(2000)
    for i in range(2000):
        pick = rng.choice(folds, size=len(folds), replace=True)
        cat = np.concatenate([blocks[f] for f in pick])
        means[i] = cat.mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    obs = diff.mean()
    crosses = (lo < 0 < hi)
    print(f"  G6 paired CI {label}: obs mean diff (F1-base) {obs:+.3f} bps/cyc | "
          f"95% block-bootstrap CI [{lo:+.3f}, {hi:+.3f}] -> {'CROSSES 0' if crosses else 'clears 0'}", flush=True)


def emit_percycle(label, U, pnl_b_by_cost, pnl_f_by_cost):
    times = U["times"]
    out = {
        "open_time": pd.to_datetime(times),
        "fold": [U["fold_by_time"].get(t, -1) for t in times],
        "regime": U["regimes"],
        "is_side": U["is_side"],
        "is_active_base": [bool(w) for w in U["cyc_w_base"]],
        "alt30": [U["alt30_by_time"].get(t, np.nan) for t in times],
        "btc30": [U["btc30_by_time"].get(t, np.nan) for t in times],
        "flag": [U["flag_by_time"][t] for t in times],
        "side_flat": [U["flag_by_time"][t] for t in times],   # mask of cycles F1 FLATs vs base
    }
    for cost_bps in COSTS_BPS:
        tag = f"{int(round(cost_bps*10)):03d}"
        out[f"pnl_base_c{tag}"] = pnl_b_by_cost[cost_bps]
        out[f"pnl_f1_c{tag}"] = pnl_f_by_cost[cost_bps]
    out["pnl_base"] = pnl_b_by_cost[4.5]
    out["pnl_f1"] = pnl_f_by_cost[4.5]
    df = pd.DataFrame(out)
    pq = OUT/f"X122_percycle_{label}.parquet"
    df.to_parquet(pq, index=False)
    print(f"  per-cycle series -> {pq}", flush=True)


def run_universe(label, preds_path, rng, is_ext=False):
    U = build_universe(preds_path, label)
    times = U["times"]

    print(f"\n=== {label}: base vs F1 by cost (G8) ===", flush=True)
    print(f"  {'cost':>5}{'arm':>7}{'Sharpe':>8}{'maxDD':>9}{'Calmar':>8}{'totPnL':>9}{'%pos':>7}", flush=True)
    pnl_b_by_cost = {}; pnl_f_by_cost = {}
    for cost_bps in COSTS_BPS:
        cost = cost_bps*1e-4
        pnl_b = heldbook(times, U["cyc_w_base"], U["rs"], cost)
        pnl_f = heldbook(times, U["cyc_w_f1"], U["rs"], cost)
        pnl_b_by_cost[cost_bps] = pnl_b; pnl_f_by_cost[cost_bps] = pnl_f
        sb = stats(pnl_b); sf = stats(pnl_f)
        print(f"  {cost_bps:>5.1f}{'base':>7}{sb['sharpe']:>+8.2f}{sb['maxDD']:>+9.0f}{sb['calmar']:>+8.2f}{sb['totPnL']:>+9.0f}{sb['pct_pos']:>7.1f}", flush=True)
        print(f"  {cost_bps:>5.1f}{'F1':>7}{sf['sharpe']:>+8.2f}{sf['maxDD']:>+9.0f}{sf['calmar']:>+8.2f}{sf['totPnL']:>+9.0f}{sf['pct_pos']:>7.1f}", flush=True)

    pnl_b = pnl_b_by_cost[4.5]; pnl_f = pnl_f_by_cost[4.5]
    sb = stats(pnl_b); sf = stats(pnl_f)
    dd_red = (1-abs(sf['maxDD'])/abs(sb['maxDD']))*100 if sb['maxDD'] < 0 else np.nan
    print(f"\n  [G7 @4.5bps] base: Sharpe {sb['sharpe']:+.2f} maxDD {sb['maxDD']:+.0f} Calmar {sb['calmar']:+.2f} totPnL {sb['totPnL']:+.0f}", flush=True)
    print(f"  [G7 @4.5bps] F1:   Sharpe {sf['sharpe']:+.2f} (Δ{sf['sharpe']-sb['sharpe']:+.2f}) "
          f"maxDD {sf['maxDD']:+.0f} (DDred {dd_red:+.1f}%) Calmar {sf['calmar']:+.2f} (Δ{sf['calmar']-sb['calmar']:+.2f}) "
          f"totPnL {sf['totPnL']:+.0f}", flush=True)

    print(f"\n  --- {label} per-fold (G5) @4.5bps ---", flush=True)
    nb, ne = per_fold_report(label, times, pnl_b, pnl_f, U["fold_by_time"])
    print(f"  F1 Calmar >= base Calmar in {nb}/{ne} folds (G5 spirit: >=6/9)", flush=True)

    print(f"\n  --- {label} fold-LOFO (G5) @4.5bps ---", flush=True)
    fold_lofo_report(label, times, pnl_b, pnl_f, U["fold_by_time"])

    if is_ext:
        print(f"\n  --- {label} MULTI-EPISODE (G5, the decisive test) @4.5bps ---", flush=True)
        episode_report(times, pnl_b, pnl_f, EXT_EPISODES)

    # G4 placebo (HL70 and EXT mandatory)
    if label in ("HL70", "EXT"):
        print(f"\n  --- {label} G4 placebo @4.5bps ---", flush=True)
        matched_placebo(label, U, 4.5e-4, rng)

    # G6 paired CI (HL70 and EXT)
    if label in ("HL70", "EXT"):
        print(f"\n  --- {label} G6 paired CI @4.5bps ---", flush=True)
        paired_ci(label, times, pnl_b, pnl_f, U["fold_by_time"], rng)

    print(f"\n  --- {label} per-cycle parquet ---", flush=True)
    emit_percycle(label, U, pnl_b_by_cost, pnl_f_by_cost)


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    print("=== X122 parameter-free 2-axis alt-bear SIDE gate (iter-007 F1) ===", flush=True)
    print(f"regime map: bull->mom30, bear->FLAT, side&(alt30<btc30)->FLAT (NEW, parameter-free), "
          f"side&(alt30>=btc30)->mean-rev BN. K={K}, HOLD={HOLD}, win={WIN}, costs {COSTS_BPS} bps. seed={SEED}", flush=True)
    run_universe("HL70", HL70_PREDS, rng)
    run_universe("EXT", EXT_PREDS, rng, is_ext=True)
    run_universe("S44", S44_PREDS, rng)
    print(f"\nDone [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
