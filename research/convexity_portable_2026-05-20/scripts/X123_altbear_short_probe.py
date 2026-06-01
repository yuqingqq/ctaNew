"""X123 — iter-008: can we FLIP TO NET-SHORT in the classified alt-bear (instead of FLAT)?

Human idea: the DD is a correlated ALT-BEAR where the mean-rev book buys "oversold" alts that
keep falling. iter-006/007 tested FLAT-the-side-in-alt-bear (rejected: single-episode artifact,
sign-flips across universes). This iteration tests the OPPOSITE/extension: in the classified
alt-bear, FLIP to NET-SHORT-BETA / short-momentum (symmetric to bull->long-momentum), profiting
from the CONTINUED fall.

Reuses X122 machinery verbatim: PIT alt-index (.shift(1) eq-weight cum log-ret of the panel's own
alts), btc30, the held-book engine, the matched-placebo loop. Adds:

STEP A — FORWARD-CLASSIFIABILITY: conditional on the classifier (alt30<btc30, side regime), what is
  the FORWARD alt-index return over the next HOLD/h bars? Persistent downtrend (keeps falling) =>
  short pays. Bottom/squeeze (bounces) => short gets run over. Measured per universe + per episode.

STEP B — NET-SHORT SLEEVE PnL: in flagged side cycles, instead of FLAT, emit a NET-SHORT-MOMENTUM
  book (short top-K by mom30 = the high-momentum/high-beta alts, no offsetting long => net short).
  Also test net-short-beta (short top-K by trailing beta). Measure per-episode PnL + the WHIPSAW
  give-back (PnL in the bear-rally / bottom sub-windows).

STEP C — vs FLAT (iter-007) and vs long-momentum-everywhere baseline-ablation.

STEP D — G4 pre-check: matched RANDOM-timing short placebo. Replace the flagged cycles' short book
  at the SAME COUNT of RANDOM side cycles. Does the classification carry edge over random-timing
  short? Plus multi-episode LOFO.

Outputs: results/X123_altbear_short_{HL70,EXT,S44}.parquet (per-cycle). Console: all tables.
Does NOT modify X116/X117/X121/X122 or cached preds.
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

    altcols = [c for c in ret4.columns if c not in ("BTCUSDT", "ETHUSDT")]
    altidx = ret4[altcols].mean(axis=1)
    alt_cum = altidx.cumsum()
    alt30_full = (alt_cum - alt_cum.shift(WIN)).shift(1)          # PIT trailing-30d, lagged
    ti = pd.DatetimeIndex(times)
    alt30 = alt30_full.reindex(ti)
    b30_lag = (b4/b4.shift(WIN)-1).shift(1).reindex(ti)
    # FORWARD alt-index return over next HOLD bars (for forward-classifiability; NOT used in trading)
    alt_fwd_hold = (alt_cum.shift(-HOLD) - alt_cum).reindex(ti)   # next-HOLD-bar cum log-ret
    alt_fwd_win = (alt_cum.shift(-WIN) - alt_cum).reindex(ti)     # next-30d cum log-ret (persistence)
    alt30_by_time = dict(zip(times, alt30.values))
    btc30_by_time = dict(zip(times, b30_lag.values))
    altfwdh_by_time = dict(zip(times, alt_fwd_hold.values))
    altfwdw_by_time = dict(zip(times, alt_fwd_win.values))

    # Per-cycle weight builders for each ARM. Each arm differs ONLY on flagged side cycles.
    cyc = {a: [] for a in ("base", "flat", "shortmom", "shortbeta", "longmom_all")}
    rs = []; regimes = []; flag_by_time = {}
    for ot in times:
        g = by_t[ot]; rg = g["regime"].iloc[0]; regimes.append(rg)
        rl = dict(zip(g["symbol"], g["return_pct"])); rs.append(rl)
        a30 = alt30_by_time.get(ot, np.nan); b30 = btc30_by_time.get(ot, np.nan)
        flag = (rg == "side") and np.isfinite(a30) and np.isfinite(b30) and (a30 < b30)
        flag_by_time[ot] = bool(flag)

        # ---- base book for this cycle (X117 production) ----
        if rg == "bear":
            base_w = {}
        else:
            key = "mom30" if rg == "bull" else "pred"; gg = g.dropna(subset=[key])
            if len(gg) < 2*K:
                base_w = {}
            else:
                gg = gg.sort_values(key); L = gg.tail(K)["symbol"].tolist(); S = gg.head(K)["symbol"].tolist()
                a = b = 1.0
                if rg == "side":
                    brow = betas.loc[ot] if ot in betas.index else None
                    if brow is not None:
                        mbL = np.nanmean([brow.get(s, np.nan) for s in L]); mbS = np.nanmean([brow.get(s, np.nan) for s in S])
                        if np.isfinite(mbL) and np.isfinite(mbS) and mbL > 0 and mbS > 0:
                            a = 2*mbS/(mbL+mbS); b = 2*mbL/(mbL+mbS)
                base_w = {}
                for s in L: base_w[s] = base_w.get(s, 0)+a/K
                for s in S: base_w[s] = base_w.get(s, 0)-b/K

        # ---- net-short-momentum book (short the top-K high-momentum/high-beta alts) ----
        gm = g.dropna(subset=["mom30"])
        shortmom_w = {}
        if len(gm) >= K:
            gm = gm.sort_values("mom30"); topmom = gm.tail(K)["symbol"].tolist()
            for s in topmom: shortmom_w[s] = -1.0/K            # NET SHORT (no offsetting long)
        # ---- net-short-beta book (short the top-K highest-beta alts) ----
        shortbeta_w = {}
        brow = betas.loc[ot] if ot in betas.index else None
        if brow is not None:
            bb = brow.dropna()
            if len(bb) >= K:
                topbeta = bb.sort_values().tail(K).index.tolist()
                for s in topbeta:
                    if s in rl: shortbeta_w[s] = -1.0/K
        # ---- long-momentum-everywhere (ablation): use mom30 long/short even in side ----
        glm = g.dropna(subset=["mom30"]); longmom_w = {}
        if len(glm) >= 2*K:
            glm = glm.sort_values("mom30"); Ll = glm.tail(K)["symbol"].tolist(); Sl = glm.head(K)["symbol"].tolist()
            for s in Ll: longmom_w[s] = longmom_w.get(s, 0)+1.0/K
            for s in Sl: longmom_w[s] = longmom_w.get(s, 0)-1.0/K

        # assemble arms
        cyc["base"].append(base_w)
        # flat arm: flagged side -> {}
        cyc["flat"].append({} if flag else base_w)
        # shortmom arm: flagged side -> net-short-mom; else base
        cyc["shortmom"].append(shortmom_w if flag else base_w)
        # shortbeta arm: flagged side -> net-short-beta; else base
        cyc["shortbeta"].append(shortbeta_w if flag else base_w)
        # longmom_all arm: every active cycle uses long-momentum (ablation of "mom everywhere")
        cyc["longmom_all"].append({} if rg == "bear" else longmom_w)

    n = len(times)
    for a in cyc: assert len(cyc[a]) == n
    is_side = [r == "side" for r in regimes]
    n_side = sum(is_side); n_flag = sum(1 for t in times if flag_by_time[t])
    print(f"  {len(syms)} syms, {n} cycles, {pd.Timestamp(times[0]).date()}->{pd.Timestamp(times[-1]).date()}; "
          f"side {n_side} ({n_side/n*100:.0f}%), flagged {n_flag} of side "
          f"({(n_flag/n_side*100) if n_side else 0:.0f}%)", flush=True)
    return dict(times=list(times), cyc=cyc, rs=rs, fold_by_time=fold_by_time, regimes=regimes,
                is_side=is_side, alt30_by_time=alt30_by_time, btc30_by_time=btc30_by_time,
                altfwdh_by_time=altfwdh_by_time, altfwdw_by_time=altfwdw_by_time, flag_by_time=flag_by_time)


# --------------------------------------------------------------------------- engine
def heldbook(times, cyc_w, rs, cost):
    prev = {}; pnl = []
    for t in range(len(cyc_w)):
        active = cyc_w[max(0, t-HOLD+1):t+1]
        net = {}
        for w in active:
            for s, wt in w.items(): net[s] = net.get(s, 0)+wt/HOLD
        alls = set(net) | set(prev)
        turn = sum(abs(net.get(s, 0)-prev.get(s, 0)) for s in alls)
        rl = rs[t]
        c = sum(net.get(s, 0)*rl.get(s, 0.0) for s in net if np.isfinite(rl.get(s, 0.0)))
        if not np.isfinite(c): c = 0.0
        pnl.append(c - turn*0.5*cost)
        prev = net
    return np.asarray(pnl, dtype=np.float64)


def heldbook_subst(times, cyc_base, rs, cost, subst_mask, subst_w):
    """Held book = base, but at cycles where subst_mask[t] use subst_w[t] instead. For matched
    placebo: substitute the net-short book at a RANDOM set of side cycles."""
    cyc = [(subst_w[t] if subst_mask[t] else cyc_base[t]) for t in range(len(times))]
    return heldbook(times, cyc, rs, cost)


# --------------------------------------------------------------------------- STEP A forward-classify
def forward_classify(label, U, episodes=None):
    """Conditional on classifier (side & alt30<btc30), what is FORWARD alt-index return?
    Persistent downtrend => negative fwd (short pays). Squeeze/bottom => positive fwd."""
    times = U["times"]
    flagged = [t for t in times if U["flag_by_time"][t]]
    side_unflag = [t for i, t in enumerate(times) if U["is_side"][i] and not U["flag_by_time"][t]]

    def summ(tlist, key):
        v = np.array([U[key].get(t, np.nan) for t in tlist], dtype=float)
        v = v[np.isfinite(v)]
        if len(v) == 0: return (np.nan, np.nan, np.nan)
        return (v.mean(), np.median(v), (v < 0).mean()*100)

    print(f"\n  [STEP A: forward-classifiability — {label}]", flush=True)
    for nm, key in [("fwd_HOLD (next 6 bars)", "altfwdh_by_time"), ("fwd_30d (next 180 bars)", "altfwdw_by_time")]:
        fm, fmd, fneg = summ(flagged, key)
        um, umd, uneg = summ(side_unflag, key)
        print(f"    {nm:>24}: FLAGGED mean {fm:+.4f} med {fmd:+.4f} %neg {fneg:.0f}  |  "
              f"UNFLAGGED side mean {um:+.4f} med {umd:+.4f} %neg {uneg:.0f}", flush=True)
    print(f"    READ: flagged fwd more negative than unflagged => classifier catches PERSISTENT "
          f"downtrend (short pays). Near-0 or positive => bottom/squeeze (short gets run over).", flush=True)

    if episodes:
        ti = pd.DatetimeIndex(times)
        print(f"    per-episode forward alt-index (fwd_HOLD) on FLAGGED cycles:", flush=True)
        for ename, a, bnd in episodes:
            m = (ti >= pd.Timestamp(a, tz="UTC")) & (ti <= pd.Timestamp(bnd, tz="UTC"))
            ep_t = [times[i] for i in range(len(times)) if m[i] and U["flag_by_time"][times[i]]]
            mm, md, neg = summ(ep_t, "altfwdh_by_time")
            print(f"      {ename:<14} n={len(ep_t):>4}  fwd mean {mm:+.4f} med {md:+.4f} %neg {neg:.0f}", flush=True)


# --------------------------------------------------------------------------- reporting
def episode_pnl(label, times, arms_pnl, episodes):
    """Per-episode PnL/maxDD/Calmar for each arm. Whipsaw: identify the bear-RALLY sub-window
    (within episode, the trough->recovery) and report short-arm PnL there."""
    ti = pd.DatetimeIndex(times)
    print(f"\n  [STEP B/C: per-episode PnL by arm — {label}] @4.5bps", flush=True)
    hdr = f"  {'episode':<14}{'n':>5}"
    for a in arms_pnl: hdr += f"{a[:9]:>11}"
    print(hdr + "  (totPnL bps)", flush=True)
    for ename, a, bnd in episodes:
        m = (ti >= pd.Timestamp(a, tz="UTC")) & (ti <= pd.Timestamp(bnd, tz="UTC"))
        if m.sum() < 5:
            print(f"  {ename:<14}{int(m.sum()):>5}  (too few)", flush=True); continue
        row = f"  {ename:<14}{int(m.sum()):>5}"
        for arm, pnl in arms_pnl.items():
            tot = (pd.Series(pnl)[m]*1e4).sum()
            row += f"{tot:>+11.0f}"
        print(row, flush=True)
    # whipsaw: per-episode max drawdown of the shortmom arm within the episode window
    print(f"\n  [WHIPSAW: shortmom arm intra-episode maxDD (bear-rally give-back) — {label}]", flush=True)
    for ename, a, bnd in episodes:
        m = (ti >= pd.Timestamp(a, tz="UTC")) & (ti <= pd.Timestamp(bnd, tz="UTC"))
        if m.sum() < 5: continue
        for arm in ("shortmom", "shortbeta"):
            if arm not in arms_pnl: continue
            sub = pd.Series(arms_pnl[arm])[m]*1e4
            eq = sub.cumsum(); mdd = (eq-eq.cummax()).min()
            print(f"    {ename:<14} {arm:<10} intra-episode totPnL {eq.iloc[-1]:>+8.0f}  maxDD {mdd:>+8.0f}", flush=True)


def per_fold(label, times, pnl_base, pnl_arm, fold_by_time, armname):
    folds = sorted(set(fold_by_time.values())) if fold_by_time else []
    if not folds: return
    fa = np.array([fold_by_time.get(t, -1) for t in times])
    nb = 0; ne = 0
    print(f"\n  [G5 per-fold {armname} vs base — {label}] @4.5bps", flush=True)
    print(f"  {'fold':>5}{'n':>6}{'baseCal':>9}{'armCal':>9}{'baseDD':>9}{'armDD':>9}{'better?':>9}", flush=True)
    for f in folds:
        m = fa == f
        if m.sum() < 3: continue
        ne += 1; sb = stats(pnl_base[m]); sa = stats(pnl_arm[m])
        better = (np.isfinite(sa["calmar"]) and np.isfinite(sb["calmar"]) and sa["calmar"] >= sb["calmar"])
        if better: nb += 1
        print(f"  {f:>5}{int(m.sum()):>6}{sb['calmar']:>+9.2f}{sa['calmar']:>+9.2f}"
              f"{sb['maxDD']:>+9.0f}{sa['maxDD']:>+9.0f}{('yes' if better else 'NO'):>9}", flush=True)
    print(f"  {armname} Calmar >= base in {nb}/{ne} folds (G5 spirit >=6/9)", flush=True)


def episode_lofo(label, times, pnl_base, pnl_arm, episodes, armname):
    ti = pd.DatetimeIndex(times)
    cb = calmar_of(pnl_base); ca = calmar_of(pnl_arm); full = ca - cb
    print(f"\n  [G5 episode-LOFO {armname} — {label}] full: base {cb:+.2f} arm {ca:+.2f} lift {full:+.2f}", flush=True)
    allpos = True
    for ename, a, bnd in episodes:
        m = (ti >= pd.Timestamp(a, tz="UTC")) & (ti <= pd.Timestamp(bnd, tz="UTC"))
        keep = ~np.asarray(m)
        cbb = calmar_of(pnl_base[keep]); caa = calmar_of(pnl_arm[keep])
        lift = caa - cbb if (np.isfinite(caa) and np.isfinite(cbb)) else np.nan
        pos = (np.isfinite(lift) and lift > 0)
        if not pos: allpos = False
        print(f"    drop {ename:<14} base {cbb:+.2f} arm {caa:+.2f} lift {lift:+.2f} {'>0' if pos else 'NEG'}", flush=True)
    print(f"    episode-LOFO stays >0 dropping each: {'PASS' if allpos else 'FAIL'}", flush=True)


def g4_short_placebo(label, U, cost, rng, armname="shortmom"):
    """G4 pre-check: matched RANDOM-timing short. Substitute the SAME short book at the SAME COUNT
    of RANDOM side cycles (drawn from side pool). Does the CLASSIFICATION beat random-timing short?"""
    times = U["times"]; rs = U["rs"]; cyc_base = U["cyc"]["base"]; subst_w = U["cyc"][armname]
    side_idx = np.array([i for i, s in enumerate(U["is_side"]) if s], dtype=int)
    flag_idx = np.array([i for i, t in enumerate(times) if U["flag_by_time"][t]], dtype=int)
    n_flag = len(flag_idx)
    if n_flag == 0 or len(side_idx) < n_flag:
        print(f"  (G4 {label} {armname}: n_flag={n_flag}, side_pool={len(side_idx)} — skip)", flush=True); return
    real_mask = np.zeros(len(times), dtype=bool); real_mask[flag_idx] = True
    pnl_real = heldbook_subst(times, cyc_base, rs, cost, real_mask, subst_w)
    real_cal = calmar_of(pnl_real); real_tot = (pd.Series(pnl_real)*1e4).sum(); real_dd = maxdd_of(pnl_real)
    cals = np.empty(N_PLACEBO); tots = np.empty(N_PLACEBO)
    for i in range(N_PLACEBO):
        pick = rng.choice(side_idx, size=n_flag, replace=False)
        mask = np.zeros(len(times), dtype=bool); mask[pick] = True
        pp = heldbook_subst(times, cyc_base, rs, cost, mask, subst_w)
        cals[i] = calmar_of(pp); tots[i] = (pd.Series(pp)*1e4).sum()
    crank = (cals < real_cal).mean()*100; trank = (tots < real_tot).mean()*100
    print(f"\n  [G4 PRE-CHECK: classification vs RANDOM-timing {armname} — {label}] ({N_PLACEBO} seeds, "
          f"sub {n_flag} of {len(side_idx)} side)", flush=True)
    print(f"    real(classified) Calmar {real_cal:+.2f} totPnL {real_tot:+.0f} maxDD {real_dd:+.0f}", flush=True)
    print(f"    placebo Calmar p50 {np.nanpercentile(cals,50):+.2f} p95 {np.nanpercentile(cals,95):+.2f} "
          f"-> rank p{crank:.0f} {'PASS(>=p95)' if crank>=95 else 'FAIL'}", flush=True)
    print(f"    placebo totPnL p50 {np.nanpercentile(tots,50):+.0f} p95 {np.nanpercentile(tots,95):+.0f} "
          f"-> rank p{trank:.0f} {'PASS(>=p95)' if trank>=95 else 'FAIL'}", flush=True)


def run_universe(label, preds_path, rng, is_ext=False):
    U = build_universe(preds_path, label)
    times = U["times"]
    eps = EXT_EPISODES if is_ext else None
    forward_classify(label, U, eps)

    arms = ("base", "flat", "shortmom", "shortbeta", "longmom_all")
    pnl = {}
    print(f"\n  [headline by arm — {label}] @4.5bps", flush=True)
    print(f"  {'arm':>12}{'Sharpe':>8}{'maxDD':>9}{'Calmar':>8}{'totPnL':>9}{'%pos':>7}", flush=True)
    for a in arms:
        pnl[a] = heldbook(times, U["cyc"][a], U["rs"], 4.5e-4)
        s = stats(pnl[a])
        print(f"  {a:>12}{s['sharpe']:>+8.2f}{s['maxDD']:>+9.0f}{s['calmar']:>+8.2f}{s['totPnL']:>+9.0f}{s['pct_pos']:>7.1f}", flush=True)

    # cost sensitivity for shortmom (G8)
    print(f"\n  [G8 cost sensitivity shortmom vs base — {label}]", flush=True)
    for cb in COSTS_BPS:
        sb = stats(heldbook(times, U["cyc"]["base"], U["rs"], cb*1e-4))
        sm = stats(heldbook(times, U["cyc"]["shortmom"], U["rs"], cb*1e-4))
        print(f"    @{cb:>4.1f}bps base Cal {sb['calmar']:+.2f} | shortmom Cal {sm['calmar']:+.2f} "
              f"(Δ{sm['calmar']-sb['calmar']:+.2f})", flush=True)

    if is_ext:
        episode_pnl(label, times, {k: pnl[k] for k in ("base", "flat", "shortmom", "shortbeta")}, EXT_EPISODES)
        for arm in ("shortmom", "shortbeta"):
            episode_lofo(label, times, pnl["base"], pnl[arm], EXT_EPISODES, arm)

    per_fold(label, times, pnl["base"], pnl["shortmom"], U["fold_by_time"], "shortmom")

    if label in ("HL70", "EXT"):
        g4_short_placebo(label, U, 4.5e-4, rng, "shortmom")
        g4_short_placebo(label, U, 4.5e-4, rng, "shortbeta")

    # per-cycle parquet
    out = {"open_time": pd.to_datetime(times),
           "fold": [U["fold_by_time"].get(t, -1) for t in times],
           "regime": U["regimes"], "is_side": U["is_side"],
           "flag": [U["flag_by_time"][t] for t in times],
           "alt30": [U["alt30_by_time"].get(t, np.nan) for t in times],
           "btc30": [U["btc30_by_time"].get(t, np.nan) for t in times],
           "alt_fwd_hold": [U["altfwdh_by_time"].get(t, np.nan) for t in times]}
    for a in arms: out[f"pnl_{a}"] = pnl[a]
    pd.DataFrame(out).to_parquet(OUT/f"X123_altbear_short_{label}.parquet", index=False)
    print(f"  per-cycle -> X123_altbear_short_{label}.parquet", flush=True)


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    print("=== X123 iter-008: FLIP-TO-NET-SHORT in classified alt-bear (vs FLAT) ===", flush=True)
    print(f"arms: base(X117) / flat(iter-007) / shortmom(net-short top-K mom) / shortbeta(net-short top-K beta) / "
          f"longmom_all(ablation). K={K} HOLD={HOLD} win={WIN}. seed={SEED}", flush=True)
    run_universe("HL70", HL70_PREDS, rng)
    run_universe("EXT", EXT_PREDS, rng, is_ext=True)
    run_universe("S44", S44_PREDS, rng)
    print(f"\nDone [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
