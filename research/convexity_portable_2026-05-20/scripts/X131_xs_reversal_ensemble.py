"""X131 — sideways-regime cross-sectional REVERSAL ensemble re-rank (iter-022).

Spec (research/handoff.md, iter-022): a STRUCTURAL / UNTUNED rank-key change on the held
book, in the SIDEWAYS regime only. Regime map becomes:

    bull (BTC-30d > +10%)   -> momentum (mom30), long top-K / short bot-K   UNCHANGED
    bear (BTC-30d < -10%)   -> FLAT (emit {})                               UNCHANGED
    side (|BTC-30d| <= 10%) -> rank by ENSEMBLE score (NEW), long top-K / short bot-K
                               score = z_xs(pred) - z_xs(rel_ret_1d)
                               (per-cycle cross-sectional z; minus sign = seesaw reversal;
                                EQUAL WEIGHT 0.5/0.5, UNTUNED -> G3 WAIVED)
    base side regime ranks by `pred` (mean-reversion). The ONLY change is the side rank key.

Everything else is fixed and unchanged from X117/X122: K=5, HOLD=6 (6 sleeves), beta-neutral
leg sizing in the side regime, 4.5bps conservative cost (reported also @1/3bps), held-book
decay machinery.

------------------------------------------------------------------------------------------
rel_ret_1d (the reversal signal) — STRICTLY PIT, built VERBATIM as in the pre-check
(iter022_leadlag_transport_precheck.py / iter022_orth_fast.py):

    rel_ret_1d[sym,t] = return_1d[sym,t]  -  cross-sectional mean of return_1d over the
                        ACTIVE universe at the SAME entry cycle t.

where the panel column `return_1d[sym,t] = close[t]/close[t-288] - 1` is a TRAILING 1-day
(288 x 5m) return whose window ENDS at the decision-bar close close[t] (verified: it is NOT
shifted, it equals close.pct_change(288) at open_time t). The forward 4h target the book
trades is return_pct[sym,t] = close[t+48]/close[t] - 1, whose window STARTS at close[t].

=> the trailing-1d window [t-288, t] and the forward target window [t, t+48] share ONLY the
boundary price close[t], which is KNOWN at decision time t. There is NO overlap of the
trailing signal with the forward return. The cross-sectional demean is computed within each
entry cycle from values all available at-or-before t. Strictly point-in-time. (Pre-check XS-IC
= -0.036, far below the +0.10 look-ahead red flag.)
------------------------------------------------------------------------------------------

Universes:
  HL70 (70-sym, x64 preds)          PRODUCTION — base REPRODUCES X117 on HL70.
  EXT  (23-sym 2021-26, x113 preds) MULTI-EPISODE + per-episode PnL transport (the decisive
                                    PnL-layer test — iter-018 had good HL70 IC but died here).
  S44  (44-sym 2023-26, x70 preds)  transport robustness.

Emits per-universe per-cycle parquet (G4 within-cycle rel-shuffle placebo, G6 paired CI),
HL70/EXT/S44 cost {1,3,4.5}bps with Sharpe/maxDD/Calmar/totPnL/folds_positive AND GROSS PnL
(pre-cost) AND turnover per universe (the decisive cost check — reversal ranks recent movers),
HL70+EXT+S44 per-fold (G5), EXT per-episode PnL. Explicit NaN guards. Seeded RNG. Does NOT
modify X116/X117/X122 or cached preds.
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
RC = REPO/"research/convexity_portable_2026-05-20/results/_cache"
OUT = REPO/"research/convexity_portable_2026-05-20/results"
KLINES = REPO/"data/ml/test/parquet/klines"

# (preds, panel) per universe; panel supplies the TRAILING return_1d for rel_ret_1d.
UNIVERSES = {
    "HL70": (RC/"x64_HL70_V5mv3_7cx_filter_t0.3_preds.parquet",
             REPO/"outputs/vBTC_features/panel_hl70.parquet"),
    "EXT":  (RC/"x113_ext_v0_preds.parquet",
             REPO/"outputs/vBTC_features/panel_ext2021_v0.parquet"),
    "S44":  (RC/"x70_v0_3yr_preds.parquet",
             REPO/"outputs/vBTC_features/panel_3yr_v0.parquet"),
}

K = 5
HOLD = 6
WIN = 180                        # 180 bars on the 4h grid = 30 days (matches X117/X122 betas/mom30)
COSTS_BPS = [1.0, 3.0, 4.5]
SEED = 12345
N_PLACEBO = 200

EXT_EPISODES = [
    ("2021_blowoff", "2021-04-01", "2021-07-31"),
    ("2022_luna",    "2022-05-01", "2022-07-31"),
    ("2022_ftx",     "2022-11-01", "2023-01-31"),
    ("2024_summer",  "2024-06-01", "2024-09-30"),
    ("2025_q4",      "2025-09-01", "2025-12-31"),
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


def zscore(v):
    """Cross-sectional z-score of a 1-D array (per cycle). Guards std==0 / NaN."""
    v = np.asarray(v, dtype=np.float64)
    m = np.isfinite(v)
    out = np.full(v.shape, np.nan)
    if m.sum() < 2:
        return out
    mu = v[m].mean(); sd = v[m].std()
    if not np.isfinite(sd) or sd == 0:
        out[m] = 0.0
        return out
    out[m] = (v[m] - mu)/sd
    return out


# --------------------------------------------------------------------------- build
def build_universe(label, preds_path, panel_path):
    """Build base-arm (rank by pred) and ENS-arm (side rank by ensemble) cycle weights.

    Side regime: ENS ranks by score = z_xs(pred) - z_xs(rel_ret_1d). Bull/bear identical to
    base. Returns dict with times, cyc_w_base, cyc_w_ens, rs, fold_by_time, regimes, is_side,
    and per-cycle (symbol-aligned) pred / rel_ret_1d arrays for the within-cycle placebo.
    """
    print(f"\n--- building {label} (preds={preds_path.name}, panel={panel_path.name}) ---", flush=True)
    d = pd.read_parquet(preds_path, columns=["symbol", "open_time", "pred", "return_pct", "fold"])
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    d = d[(d["open_time"].dt.hour % 4 == 0) & (d["open_time"].dt.minute == 0)].copy()

    # ---- merge the TRAILING return_1d from the panel (the rel_ret_1d source), then demean
    #      within each entry cycle EXACTLY as the pre-check did (transport_precheck/orth_fast).
    pan = pd.read_parquet(panel_path, columns=["symbol", "open_time", "return_1d"])
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    pan = pan[(pan["open_time"].dt.hour % 4 == 0) & (pan["open_time"].dt.minute == 0)]
    d = d.merge(pan, on=["symbol", "open_time"], how="left")
    # rel_ret_1d = own trailing-1d return minus the cross-sectional mean over the active universe
    # at the SAME cycle (PIT: all inputs known at-or-before the decision bar t). VERBATIM pre-check.
    g_ot = d.groupby("open_time")
    d["ret1d_xsmean"] = g_ot["return_1d"].transform("mean")
    d["rel_ret_1d"] = d["return_1d"] - d["ret1d_xsmean"]
    n_relna = int(d["rel_ret_1d"].isna().sum())
    print(f"  merged return_1d; rel_ret_1d NaN rows {n_relna} of {len(d)} "
          f"({n_relna/len(d)*100:.2f}%)", flush=True)

    btc = load_close("BTCUSDT"); b4 = btc[(btc.index.hour % 4 == 0) & (btc.index.minute == 0)]
    br = np.log(b4/b4.shift(1)); bvar = br.rolling(WIN, min_periods=42).var()
    syms = sorted(d["symbol"].unique())
    mom_rows = []; beta_map = {}
    for sym in syms:
        c = load_close(sym)
        if c is None: continue
        c4 = c[(c.index.hour % 4 == 0) & (c.index.minute == 0)]
        mom_rows.append(pd.DataFrame({"symbol": sym, "open_time": c4.index,
                                      "mom30": (c4/c4.shift(WIN)-1).shift(1).values}))
        r = np.log(c4/c4.shift(1)); ri, bi = r.align(br, join="inner")
        beta_map[sym] = (ri.rolling(WIN, min_periods=42).cov(bi)/bvar.reindex(ri.index).replace(0, np.nan)).shift(1)
    mom = pd.concat(mom_rows, ignore_index=True); mom["open_time"] = pd.to_datetime(mom["open_time"], utc=True)
    betas = pd.concat([s.rename(k) for k, s in beta_map.items()], axis=1)
    d = d.merge(mom, on=["symbol", "open_time"], how="left")
    btc30 = (b4/b4.shift(WIN)-1).to_frame("b30").reset_index(); btc30["open_time"] = pd.to_datetime(btc30["open_time"], utc=True)
    d = d.merge(btc30, on="open_time", how="left").dropna(subset=["b30"])
    d["regime"] = np.where(d["b30"] > 0.10, "bull", np.where(d["b30"] < -0.10, "bear", "side"))

    times = sorted(d["open_time"].unique()); by_t = {ot: g for ot, g in d.groupby("open_time")}
    fold_by_time = {ot: int(g["fold"].iloc[0]) for ot, g in by_t.items() if "fold" in g.columns}

    cyc_w_base = []; cyc_w_ens = []; rs = []; regimes = []
    # store per-cycle symbol/pred/rel arrays for the within-cycle placebo (G4)
    cyc_syms = []; cyc_pred = []; cyc_rel = []; cyc_beta = []
    for ot in times:
        g = by_t[ot]; rg = g["regime"].iloc[0]; regimes.append(rg)
        rs.append(dict(zip(g["symbol"], g["return_pct"])))

        if rg == "bear":
            cyc_w_base.append({}); cyc_w_ens.append({})
            cyc_syms.append(np.array([])); cyc_pred.append(np.array([]))
            cyc_rel.append(np.array([])); cyc_beta.append(np.array([])); continue

        if rg == "bull":
            # momentum: rank by mom30 (UNCHANGED, identical for base and ens)
            gg = g.dropna(subset=["mom30"])
            if len(gg) < 2*K:
                cyc_w_base.append({}); cyc_w_ens.append({})
                cyc_syms.append(np.array([])); cyc_pred.append(np.array([]))
                cyc_rel.append(np.array([])); cyc_beta.append(np.array([])); continue
            gg = gg.sort_values("mom30"); L = gg.tail(K)["symbol"].tolist(); S = gg.head(K)["symbol"].tolist()
            w = {}
            for s in L: w[s] = w.get(s, 0)+1.0/K
            for s in S: w[s] = w.get(s, 0)-1.0/K
            cyc_w_base.append(w); cyc_w_ens.append(w)        # identical
            cyc_syms.append(np.array([])); cyc_pred.append(np.array([]))
            cyc_rel.append(np.array([])); cyc_beta.append(np.array([])); continue

        # ---- SIDE regime ----
        # base ranks by pred; ENS ranks by score = z_xs(pred) - z_xs(rel_ret_1d).
        brow = betas.loc[ot] if ot in betas.index else None

        def make_w(rank_syms_sorted_asc):
            L = rank_syms_sorted_asc[-K:]; S = rank_syms_sorted_asc[:K]
            a = b = 1.0
            if brow is not None:
                mbL = np.nanmean([brow.get(s, np.nan) for s in L]); mbS = np.nanmean([brow.get(s, np.nan) for s in S])
                if np.isfinite(mbL) and np.isfinite(mbS) and mbL > 0 and mbS > 0:
                    a = 2*mbS/(mbL+mbS); b = 2*mbL/(mbL+mbS)
            w = {}
            for s in L: w[s] = w.get(s, 0)+a/K
            for s in S: w[s] = w.get(s, 0)-b/K
            return w

        # base side: rank by pred
        gb = g.dropna(subset=["pred"])
        if len(gb) >= 2*K:
            gb = gb.sort_values("pred"); cyc_w_base.append(make_w(gb["symbol"].tolist()))
        else:
            cyc_w_base.append({})

        # ENS side: need both pred AND rel_ret_1d present; z each cross-sectionally; combine.
        ge = g.dropna(subset=["pred", "rel_ret_1d"])
        if len(ge) >= 2*K:
            zp = zscore(ge["pred"].values)
            zr = zscore(ge["rel_ret_1d"].values)
            score = zp - zr                               # equal weight, minus sign = seesaw
            order = np.argsort(score, kind="mergesort")   # ascending; stable
            syms_sorted = ge["symbol"].values[order]
            cyc_w_ens.append(make_w(syms_sorted.tolist()))
            # placebo materials: the side-cycle pred/rel/beta arrays (symbol-aligned to ge)
            cyc_syms.append(ge["symbol"].values.copy())
            cyc_pred.append(ge["pred"].values.copy())
            cyc_rel.append(ge["rel_ret_1d"].values.copy())
            cyc_beta.append(np.array([brow.get(s, np.nan) if brow is not None else np.nan
                                      for s in ge["symbol"].values]))
        else:
            cyc_w_ens.append({})
            cyc_syms.append(np.array([])); cyc_pred.append(np.array([]))
            cyc_rel.append(np.array([])); cyc_beta.append(np.array([]))

    assert len(cyc_w_base) == len(times) == len(cyc_w_ens), "arm length mismatch"
    is_side = [r == "side" for r in regimes]
    n_side = sum(is_side)
    n_ens_active = sum(1 for i in range(len(times)) if is_side[i] and cyc_w_ens[i])
    print(f"  {len(syms)} syms, {len(times)} cycles, {pd.Timestamp(times[0]).date()}->{pd.Timestamp(times[-1]).date()}; "
          f"side {n_side} ({n_side/len(times)*100:.0f}%), ENS re-ranked side {n_ens_active}", flush=True)
    return dict(times=list(times), cyc_w_base=cyc_w_base, cyc_w_ens=cyc_w_ens, rs=rs,
                fold_by_time=fold_by_time, regimes=regimes, is_side=is_side,
                cyc_syms=cyc_syms, cyc_pred=cyc_pred, cyc_rel=cyc_rel, cyc_beta=cyc_beta)


# --------------------------------------------------------------------------- engine
def heldbook(times, cyc_w, rs, cost, return_gross_turn=False):
    """Held-book loop identical to X117/X122. Net PnL per cycle = Σ wᵢ·retᵢ − turn·0.5·cost.
    If return_gross_turn: also return (gross PnL per cycle, turnover per cycle)."""
    prev = {}; pnl = []; gross = []; turns = []
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
        gross.append(cyc); turns.append(turn)
        prev = net
    if return_gross_turn:
        return (np.asarray(pnl, np.float64), np.asarray(gross, np.float64), np.asarray(turns, np.float64))
    return np.asarray(pnl, dtype=np.float64)


def heldbook_from_weightlist(times, cyc_w, rs, cost):
    return heldbook(times, cyc_w, rs, cost)


# --------------------------------------------------------------------------- reporting
def per_fold_report(label, times, pnl_base, pnl_ens, fold_by_time):
    folds = sorted(set(fold_by_time.values())) if fold_by_time else []
    if not folds:
        print("  (no fold column)", flush=True); return 0, 0
    fold_arr = np.array([fold_by_time.get(t, -1) for t in times])
    n_better = 0; n_eval = 0
    print(f"  {'fold':>5}{'n':>6}{'baseSh':>8}{'ensSh':>8}{'baseCal':>9}{'ensCal':>9}"
          f"{'basePnL':>9}{'ensPnL':>9}{'better?':>8}", flush=True)
    for f in folds:
        m = fold_arr == f
        if m.sum() < 3: continue
        n_eval += 1
        sb = stats(pnl_base[m]); sf = stats(pnl_ens[m])
        cb, cf = sb["calmar"], sf["calmar"]
        better = (np.isfinite(cf) and np.isfinite(cb) and cf >= cb) or \
                 (sf["totPnL"] > sb["totPnL"] and not (np.isfinite(cb) and np.isfinite(cf)))
        if better: n_better += 1
        print(f"  {f:>5}{int(m.sum()):>6}{sb['sharpe']:>+8.2f}{sf['sharpe']:>+8.2f}"
              f"{cb:>+9.2f}{cf:>+9.2f}{sb['totPnL']:>+9.0f}{sf['totPnL']:>+9.0f}"
              f"{('YES' if better else 'no'):>8}", flush=True)
    return n_better, n_eval


def fold_lofo_report(label, times, pnl_base, pnl_ens, fold_by_time):
    folds = sorted(set(fold_by_time.values())) if fold_by_time else []
    if not folds:
        print("  (no fold column for LOFO)", flush=True); return
    fold_arr = np.array([fold_by_time.get(t, -1) for t in times])
    cal_b_all = calmar_of(pnl_base); cal_f_all = calmar_of(pnl_ens); full_lift = cal_f_all - cal_b_all
    print(f"  FULL: base Calmar {cal_b_all:+.2f} | ens Calmar {cal_f_all:+.2f} | lift {full_lift:+.2f}", flush=True)
    print(f"  {'drop':>6}{'baseCal':>9}{'ensCal':>9}{'lift':>8}{'Δvs_full':>10}{'>0?':>5}", flush=True)
    all_pos = True
    for f in folds:
        keep = fold_arr != f
        cb = calmar_of(pnl_base[keep]); cf = calmar_of(pnl_ens[keep])
        lift = cf - cb if (np.isfinite(cf) and np.isfinite(cb)) else np.nan
        dvs = lift - full_lift if np.isfinite(lift) else np.nan
        pos = "yes" if (np.isfinite(lift) and lift > 0) else "NO"
        if not (np.isfinite(lift) and lift > 0): all_pos = False
        print(f"  {('-f'+str(f)):>6}{cb:>+9.2f}{cf:>+9.2f}{lift:>+8.2f}{dvs:>+10.2f}{pos:>5}", flush=True)
    print(f"  fold-LOFO lift stays >0 dropping EACH fold: {'PASS' if all_pos else 'FAIL'}", flush=True)


def episode_report(times, pnl_base, pnl_ens, episodes):
    """EXT multi-episode PnL transport (the decisive PnL-layer test): per-episode PnL/Calmar/DD
    base vs ENS. iter-018 had good HL70 IC but died here."""
    ti = pd.DatetimeIndex(times)
    pb = pd.Series(pnl_base, index=ti); pf = pd.Series(pnl_ens, index=ti)
    print(f"  {'episode':<14}{'n':>5}{'basePnL':>9}{'ensPnL':>9}{'PnLΔ':>8}"
          f"{'baseSh':>8}{'ensSh':>8}{'baseDD':>9}{'ensDD':>9}{'baseCal':>8}{'ensCal':>8}{'better?':>8}", flush=True)
    n_improved = 0; n_eval = 0; ep_masks = {}
    for ename, a, bnd in episodes:
        m = (ti >= pd.Timestamp(a, tz="UTC")) & (ti <= pd.Timestamp(bnd, tz="UTC"))
        ep_masks[ename] = m
        if m.sum() < 5:
            print(f"  {ename:<14}{int(m.sum()):>5}  (too few cycles)", flush=True); continue
        n_eval += 1
        sb = stats(pb[m].values); sf = stats(pf[m].values)
        pnld = sf["totPnL"] - sb["totPnL"]
        better = pnld > 0
        if better: n_improved += 1
        print(f"  {ename:<14}{int(m.sum()):>5}{sb['totPnL']:>+9.0f}{sf['totPnL']:>+9.0f}{pnld:>+8.0f}"
              f"{sb['sharpe']:>+8.2f}{sf['sharpe']:>+8.2f}{sb['maxDD']:>+9.0f}{sf['maxDD']:>+9.0f}"
              f"{sb['calmar']:>+8.2f}{sf['calmar']:>+8.2f}{('YES' if better else 'no'):>8}", flush=True)
    print(f"  episodes with ENS PnL > base PnL: {n_improved}/{n_eval}", flush=True)


def gross_turnover_report(label, U, cost_bps):
    """G8 decisive cost check: GROSS (pre-cost) PnL + total turnover, base vs ENS.
    Reversal ranks recent movers -> turnover may spike; gross-up-but-net-flat = eaten by cost."""
    times = U["times"]; rs = U["rs"]; cost = cost_bps*1e-4
    nb, gb, tb = heldbook(times, U["cyc_w_base"], rs, cost, return_gross_turn=True)
    ne, ge, te = heldbook(times, U["cyc_w_ens"], rs, cost, return_gross_turn=True)
    gross_b = gb.sum()*1e4; gross_e = ge.sum()*1e4
    net_b = nb.sum()*1e4; net_e = ne.sum()*1e4
    turn_b = tb.sum(); turn_e = te.sum()
    cost_b = (gross_b - net_b); cost_e = (gross_e - net_e)
    print(f"  [{label} @ {cost_bps}bps GROSS/turnover]", flush=True)
    print(f"    {'arm':>6}{'grossPnL':>10}{'netPnL':>10}{'costPnL':>10}{'turnover':>11}{'cost/gross%':>12}", flush=True)
    cgb = (cost_b/abs(gross_b)*100) if gross_b != 0 else np.nan
    cge = (cost_e/abs(gross_e)*100) if gross_e != 0 else np.nan
    print(f"    {'base':>6}{gross_b:>+10.0f}{net_b:>+10.0f}{cost_b:>+10.0f}{turn_b:>11.1f}{cgb:>+12.1f}", flush=True)
    print(f"    {'ens':>6}{gross_e:>+10.0f}{net_e:>+10.0f}{cost_e:>+10.0f}{turn_e:>11.1f}{cge:>+12.1f}", flush=True)
    print(f"    Δ gross {gross_e-gross_b:+.0f}bps | Δ net {net_e-net_b:+.0f}bps | "
          f"Δ turnover {turn_e-turn_b:+.1f} ({(turn_e/turn_b-1)*100 if turn_b else float('nan'):+.1f}%) | "
          f"-> edge is {'REAL (net Δ>0)' if (net_e-net_b)>0 else 'EATEN BY COST / NEGATIVE (net Δ<=0)'}", flush=True)


def matched_placebo(label, U, cost, rng):
    """G4: within-cycle SCORE-SHUFFLE placebo. In each side cycle, SHUFFLE rel_ret_1d across
    symbols (break the rel-to-symbol link) then rebuild the same ensemble score & beta-neutral
    weights via the same machinery; bull/bear unchanged. >=N_PLACEBO seeds. Reports ENS's
    percentile rank of Calmar vs the shuffle distribution. Tests skill (real rel) vs rank-churn."""
    times = U["times"]; rs = U["rs"]; is_side = U["is_side"]
    cyc_w_base = U["cyc_w_base"]
    cyc_syms = U["cyc_syms"]; cyc_pred = U["cyc_pred"]; cyc_rel = U["cyc_rel"]; cyc_beta = U["cyc_beta"]

    def build_w_from_arrays(syms, pred, rel, beta):
        if len(syms) < 2*K: return None
        zp = zscore(pred); zr = zscore(rel)
        score = zp - zr
        order = np.argsort(score, kind="mergesort")
        ss = syms[order]
        L = ss[-K:].tolist(); S = ss[:K].tolist()
        bmap = dict(zip(syms, beta))
        a = b = 1.0
        mbL = np.nanmean([bmap.get(s, np.nan) for s in L]); mbS = np.nanmean([bmap.get(s, np.nan) for s in S])
        if np.isfinite(mbL) and np.isfinite(mbS) and mbL > 0 and mbS > 0:
            a = 2*mbS/(mbL+mbS); b = 2*mbL/(mbL+mbS)
        w = {}
        for s in L: w[s] = w.get(s, 0)+a/K
        for s in S: w[s] = w.get(s, 0)-b/K
        return w

    side_pos = [i for i in range(len(times)) if is_side[i] and len(cyc_syms[i]) >= 2*K]
    n_active = len(side_pos)
    if n_active == 0:
        print(f"  (G4 {label}: no re-ranked side cycles — cannot run placebo)", flush=True); return
    real_cal = calmar_of(heldbook(times, U["cyc_w_ens"], rs, cost))
    cals = np.empty(N_PLACEBO)
    for it in range(N_PLACEBO):
        cyc_w = list(cyc_w_base)        # start from base (bull/bear + non-active side = base)
        # overwrite the active side cycles with a within-cycle rel-SHUFFLED ensemble
        for i in side_pos:
            relsh = cyc_rel[i].copy(); rng.shuffle(relsh)
            w = build_w_from_arrays(cyc_syms[i], cyc_pred[i], relsh, cyc_beta[i])
            cyc_w[i] = w if w is not None else {}
        cals[it] = calmar_of(heldbook(times, cyc_w, rs, cost))
    rank = (cals < real_cal).mean()*100
    print(f"  G4 within-cycle rel-shuffle placebo ({N_PLACEBO} seeds, shuffle rel_ret_1d in "
          f"{n_active} re-ranked side cycles):", flush=True)
    print(f"     real ENS Calmar {real_cal:+.2f} | placebo Calmar p50 {np.nanpercentile(cals,50):+.2f} "
          f"p95 {np.nanpercentile(cals,95):+.2f} max {np.nanmax(cals):+.2f} -> rank p{rank:.0f} "
          f"{'PASS(>=p95)' if rank >= 95 else 'FAIL'}", flush=True)


def paired_ci(label, times, pnl_base, pnl_ens, fold_by_time, rng):
    """G6: block-bootstrap (blocks by fold) the paired per-cycle PnL diff (ENS - base)."""
    diff = (pnl_ens - pnl_base)*1e4
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
    print(f"  G6 paired CI {label}: obs mean diff (ENS-base) {obs:+.3f} bps/cyc | "
          f"95% block-bootstrap CI [{lo:+.3f}, {hi:+.3f}] -> {'CROSSES 0' if crosses else 'clears 0'}", flush=True)


def emit_percycle(label, U, pnl_b_by_cost, pnl_e_by_cost):
    times = U["times"]
    out = {
        "open_time": pd.to_datetime(times),
        "fold": [U["fold_by_time"].get(t, -1) for t in times],
        "regime": U["regimes"],
        "is_side": U["is_side"],
        "is_active_base": [bool(w) for w in U["cyc_w_base"]],
        "is_active_ens": [bool(w) for w in U["cyc_w_ens"]],
    }
    for cost_bps in COSTS_BPS:
        tag = f"{int(round(cost_bps*10)):03d}"
        out[f"pnl_base_c{tag}"] = pnl_b_by_cost[cost_bps]
        out[f"pnl_ens_c{tag}"] = pnl_e_by_cost[cost_bps]
    out["pnl_base"] = pnl_b_by_cost[4.5]
    out["pnl_ens"] = pnl_e_by_cost[4.5]
    df = pd.DataFrame(out)
    pq = OUT/f"X131_percycle_{label}.parquet"
    df.to_parquet(pq, index=False)
    print(f"  per-cycle series -> {pq}", flush=True)


def run_universe(label, preds_path, panel_path, rng, is_ext=False):
    U = build_universe(label, preds_path, panel_path)
    times = U["times"]

    print(f"\n=== {label}: base vs ENS by cost (G8) ===", flush=True)
    print(f"  {'cost':>5}{'arm':>7}{'Sharpe':>8}{'maxDD':>9}{'Calmar':>8}{'totPnL':>9}{'%pos':>7}", flush=True)
    pnl_b_by_cost = {}; pnl_e_by_cost = {}
    for cost_bps in COSTS_BPS:
        cost = cost_bps*1e-4
        pnl_b = heldbook(times, U["cyc_w_base"], U["rs"], cost)
        pnl_e = heldbook(times, U["cyc_w_ens"], U["rs"], cost)
        pnl_b_by_cost[cost_bps] = pnl_b; pnl_e_by_cost[cost_bps] = pnl_e
        sb = stats(pnl_b); sf = stats(pnl_e)
        print(f"  {cost_bps:>5.1f}{'base':>7}{sb['sharpe']:>+8.2f}{sb['maxDD']:>+9.0f}{sb['calmar']:>+8.2f}{sb['totPnL']:>+9.0f}{sb['pct_pos']:>7.1f}", flush=True)
        print(f"  {cost_bps:>5.1f}{'ens':>7}{sf['sharpe']:>+8.2f}{sf['maxDD']:>+9.0f}{sf['calmar']:>+8.2f}{sf['totPnL']:>+9.0f}{sf['pct_pos']:>7.1f}", flush=True)

    # G8 decisive: GROSS PnL + turnover at all 3 costs
    print(f"\n  --- {label} GROSS PnL + turnover (G8 decisive cost check) ---", flush=True)
    for cost_bps in COSTS_BPS:
        gross_turnover_report(label, U, cost_bps)

    pnl_b = pnl_b_by_cost[4.5]; pnl_e = pnl_e_by_cost[4.5]
    sb = stats(pnl_b); sf = stats(pnl_e)
    dd_red = (1-abs(sf['maxDD'])/abs(sb['maxDD']))*100 if sb['maxDD'] < 0 else np.nan
    print(f"\n  [G7 @4.5bps] base: Sharpe {sb['sharpe']:+.2f} maxDD {sb['maxDD']:+.0f} Calmar {sb['calmar']:+.2f} totPnL {sb['totPnL']:+.0f}", flush=True)
    print(f"  [G7 @4.5bps] ENS:  Sharpe {sf['sharpe']:+.2f} (Δ{sf['sharpe']-sb['sharpe']:+.2f}) "
          f"maxDD {sf['maxDD']:+.0f} (DDΔ {dd_red:+.1f}%) Calmar {sf['calmar']:+.2f} (Δ{sf['calmar']-sb['calmar']:+.2f}) "
          f"totPnL {sf['totPnL']:+.0f}", flush=True)

    print(f"\n  --- {label} per-fold (G5) @4.5bps ---", flush=True)
    nb, ne = per_fold_report(label, times, pnl_b, pnl_e, U["fold_by_time"])
    print(f"  ENS better than base in {nb}/{ne} folds (G5: >=6/9) -> "
          f"{'PASS' if (ne and nb >= 6) else 'see LOFO'}", flush=True)

    print(f"\n  --- {label} fold-LOFO (G5) @4.5bps ---", flush=True)
    fold_lofo_report(label, times, pnl_b, pnl_e, U["fold_by_time"])

    if is_ext:
        print(f"\n  --- {label} per-EPISODE PnL transport (decisive G7) @4.5bps ---", flush=True)
        episode_report(times, pnl_b, pnl_e, EXT_EPISODES)

    if label in ("HL70", "EXT"):
        print(f"\n  --- {label} G4 within-cycle rel-shuffle placebo @4.5bps ---", flush=True)
        matched_placebo(label, U, 4.5e-4, rng)
        print(f"\n  --- {label} G6 paired CI @4.5bps ---", flush=True)
        paired_ci(label, times, pnl_b, pnl_e, U["fold_by_time"], rng)

    print(f"\n  --- {label} per-cycle parquet ---", flush=True)
    emit_percycle(label, U, pnl_b_by_cost, pnl_e_by_cost)


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    print("=== X131 sideways-regime cross-sectional REVERSAL ensemble re-rank (iter-022) ===", flush=True)
    print(f"regime map: bull->mom30 (UNCHANGED), bear->FLAT (UNCHANGED), "
          f"side-> rank by z_xs(pred)-z_xs(rel_ret_1d) (NEW, equal-weight 0.5/0.5, UNTUNED -> G3 waived). "
          f"K={K}, HOLD={HOLD}, win={WIN}, costs {COSTS_BPS} bps. seed={SEED}", flush=True)
    run_universe("HL70", *UNIVERSES["HL70"], rng)
    run_universe("EXT", *UNIVERSES["EXT"], rng, is_ext=True)
    run_universe("S44", *UNIVERSES["S44"], rng)
    print(f"\nDone [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
