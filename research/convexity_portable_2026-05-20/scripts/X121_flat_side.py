"""X121 — Structural `side -> FLAT` regime-map change on the held book.

Spec (research/handoff.md, iter-003): a PARAMETER-FREE structural regime-branch
change. Extend the existing `bear -> FLAT` rule to the sideways regime:

    bull  (BTC-30d > +10%)  -> momentum (mom30), long top-K / short bot-K   UNCHANGED
    bear  (BTC-30d < -10%)  -> FLAT (emit {})                               UNCHANGED
    side  (else)            -> FLAT (emit {})   <-- THE ONLY CHANGE (was mean-rev BN)

There is NO new feature, NO model retrain, NO threshold, NO per-cycle signal selecting
*which* side cycles to skip. The held book then only ever holds bull-regime sleeves
(overlapping HOLD=6 sleeves, equal weight, K=5), exactly as today. This is a one-line
regime-branch change — the natural structural extension of bear->FLAT under the iter-002/003
finding that side is also a net-zero regime that supplies the entire drawdown.

Arms:
  base       = X117 production (bull mom / side mean-rev BN / bear FLAT). REPRODUCES X117.
  flat_side  = bull mom / side FLAT / bear FLAT.

Emits everything Evaluation needs to adjudicate structural-improvement vs hindsight artifact:
  - per-cycle parquet (HL70 + S44): pnl_base, pnl_flatside @ {1,3,4.5} bps, regime, fold,
    is_side mask. (For G4 matched-active-cycle placebo, G6 paired CI.)
  - per-fold breakdown: base vs flat_side Sharpe/maxDD/Calmar AND the side-regime-only PnL
    contribution per fold (the f2/f8 positive, f4 disaster anatomy claim).
  - LOFO: leave-one-fold-out flat_side-vs-base Calmar improvement, dropping each fold in turn
    (does the in-sample +4.27 Calmar lift survive dropping the single worst (deep-DD) fold?).
  - by-year breakdown.

Runs on HL70 (production, primary) and S44 (x70_v0_3yr_preds, robustness). Explicit NaN
guards (the totPnL +nan bug). Seeded RNG. Does NOT modify X116/X117/X120 or cached preds.
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = Path("/home/yuqing/ctaNew")
RC = REPO/"research/convexity_portable_2026-05-20/results/_cache"
OUT = REPO/"research/convexity_portable_2026-05-20/results"
HL70_PREDS = RC/"x64_HL70_V5mv3_7cx_filter_t0.3_preds.parquet"
S44_PREDS = RC/"x70_v0_3yr_preds.parquet"
KLINES = REPO/"data/ml/test/parquet/klines"

K = 5
HOLD = 6
COSTS_BPS = [1.0, 3.0, 4.5]
SEED = 12345


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
    """pnl_series in raw return units. Returns contract metrics (bps where shown)."""
    p = pd.Series(pnl_series).dropna()
    pb = p*1e4
    eq = pb.cumsum()
    dd = (eq - eq.cummax())
    mdd = dd.min()
    annr = pb.mean()*6*365
    cal = (annr/abs(mdd)) if (mdd < 0 and np.isfinite(mdd)) else np.nan
    return {"sharpe": ann(p), "maxDD": mdd, "calmar": cal,
            "totPnL": eq.iloc[-1] if len(eq) else np.nan, "pct_pos": (pb > 0).mean()*100}


def calmar_of(pnl_arr):
    """Calmar of a raw-return pnl array (helper for LOFO)."""
    pb = pd.Series(pnl_arr).dropna()*1e4
    if len(pb) < 3: return np.nan
    eq = pb.cumsum(); mdd = (eq - eq.cummax()).min()
    return (pb.mean()*6*365/abs(mdd)) if (mdd < 0 and np.isfinite(mdd)) else np.nan


# ---------------------------------------------------------------------------
# Build the regime-hybrid cycle weights for BOTH arms (identical construction to
# X117/X116 for base; side branch emits {} for flat_side). Returns:
# times, cyc_w_base, cyc_w_flat, rs, fold_by_time, regimes.
# ---------------------------------------------------------------------------
def build_universe(preds_path, label):
    print(f"\n--- building {label} ({preds_path.name}) ---", flush=True)
    cols = ["symbol", "open_time", "pred", "return_pct", "fold"]
    d = pd.read_parquet(preds_path, columns=cols)
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    d = d[(d["open_time"].dt.hour % 4 == 0) & (d["open_time"].dt.minute == 0)].copy()

    btc = load_close("BTCUSDT"); b4 = btc[(btc.index.hour % 4 == 0) & (btc.index.minute == 0)]
    br = np.log(b4/b4.shift(1)); bvar = br.rolling(180, min_periods=42).var()
    syms = sorted(d["symbol"].unique()); mom_rows = []; beta_map = {}
    for sym in syms:
        c = load_close(sym)
        if c is None: continue
        c4 = c[(c.index.hour % 4 == 0) & (c.index.minute == 0)]
        mom_rows.append(pd.DataFrame({"symbol": sym, "open_time": c4.index,
                                      "mom30": (c4/c4.shift(180)-1).shift(1).values}))
        r = np.log(c4/c4.shift(1)); ri, bi = r.align(br, join="inner")
        beta_map[sym] = (ri.rolling(180, min_periods=42).cov(bi)/bvar.reindex(ri.index).replace(0, np.nan)).shift(1)
    mom = pd.concat(mom_rows, ignore_index=True); mom["open_time"] = pd.to_datetime(mom["open_time"], utc=True)
    betas = pd.concat([s.rename(k) for k, s in beta_map.items()], axis=1)
    d = d.merge(mom, on=["symbol", "open_time"], how="left")
    btc30 = (b4/b4.shift(180)-1).to_frame("b30").reset_index(); btc30["open_time"] = pd.to_datetime(btc30["open_time"], utc=True)
    d = d.merge(btc30, on="open_time", how="left").dropna(subset=["b30"])
    d["regime"] = np.where(d["b30"] > 0.10, "bull", np.where(d["b30"] < -0.10, "bear", "side"))

    times = sorted(d["open_time"].unique()); by_t = {ot: g for ot, g in d.groupby("open_time")}
    fold_by_time = {ot: int(g["fold"].iloc[0]) for ot, g in by_t.items() if "fold" in g.columns}

    cyc_w_base = []; cyc_w_flat = []; rs = []; regimes = []
    for ot in times:
        g = by_t[ot]; rg = g["regime"].iloc[0]; regimes.append(rg)
        rs.append(dict(zip(g["symbol"], g["return_pct"])))

        # ---- flat_side arm: bull traded, bear+side FLAT ----
        # ---- base arm: bull traded, bear FLAT, side mean-rev BN (X117) ----
        if rg == "bear":
            cyc_w_base.append({}); cyc_w_flat.append({}); continue
        if rg == "side":
            cyc_w_flat.append({})        # THE ONLY CHANGE vs base: side -> FLAT (parameter-free)
        # base-arm weight build (bull for both arms; side for base only)
        key = "mom30" if rg == "bull" else "pred"; gg = g.dropna(subset=[key])
        if len(gg) < 2*K:
            cyc_w_base.append({})
            if rg == "bull": cyc_w_flat.append({})
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
        if rg == "bull": cyc_w_flat.append(w)   # bull sleeves identical across arms

    assert len(cyc_w_base) == len(times) == len(cyc_w_flat), "arm length mismatch"
    n_side = sum(1 for r in regimes if r == "side")
    print(f"  {len(syms)} syms, {len(times)} cycles, {pd.Timestamp(times[0]).date()}->{pd.Timestamp(times[-1]).date()}; "
          f"side {n_side} ({n_side/len(times)*100:.0f}%)", flush=True)
    return list(times), cyc_w_base, cyc_w_flat, rs, fold_by_time, regimes


# ---------------------------------------------------------------------------
# Held-book loop (identical engine to X117/X120). No gate, no signal: the arm is
# fully determined by the cyc_w list passed in. A FLATted side cycle simply
# contributed an empty sleeve {} at build time (exactly like a bear cycle), so the
# held book decays prior bull sleeves normally. Returns (pnl array).
# ---------------------------------------------------------------------------
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
        cyc = sum(net.get(s, 0)*rl.get(s, 0.0) for s in net if np.isfinite(rl.get(s, 0.0)))
        if not np.isfinite(cyc): cyc = 0.0          # explicit NaN guard (totPnL +nan bug)
        pnl.append(cyc - turn*0.5*cost)
        prev = net
    return np.asarray(pnl, dtype=np.float64)


# Side-regime-only PnL contribution: held-book PnL on cycles labelled "side" (base arm).
# This is the per-cycle base PnL summed over side cycles in a fold — the anatomy claim
# (f2/f8 positive, f4 disaster). It is what side->FLAT removes from the base book.
def side_contrib_by_fold(times, pnl_base, regimes, fold_by_time):
    folds = sorted(set(fold_by_time.values())) if fold_by_time else []
    out = {}
    fold_arr = np.array([fold_by_time.get(t, -1) for t in times])
    side_arr = np.array([r == "side" for r in regimes])
    for f in folds:
        m = (fold_arr == f) & side_arr
        out[f] = float((pnl_base[m]*1e4).sum()) if m.any() else 0.0
    return out


def per_fold_report(label, times, pnl_base, pnl_flat, regimes, fold_by_time):
    folds = sorted(set(fold_by_time.values())) if fold_by_time else []
    if not folds:
        print("  (no fold column)", flush=True); return 0, 0
    fold_arr = np.array([fold_by_time.get(t, -1) for t in times])
    side_pnl = side_contrib_by_fold(times, pnl_base, regimes, fold_by_time)
    n_better = 0; n_eval = 0
    print(f"  {'fold':>5}{'n':>6}{'baseDD':>9}{'flatDD':>9}{'DDimp%':>8}"
          f"{'baseSh':>8}{'flatSh':>8}{'baseCal':>8}{'flatCal':>8}{'sidePnL':>9}", flush=True)
    for f in folds:
        m = fold_arr == f
        if m.sum() < 3: continue
        n_eval += 1
        sb = stats(pnl_base[m]); sf = stats(pnl_flat[m])
        bdd, fdd = sb["maxDD"], sf["maxDD"]
        ddimp = (1-abs(fdd)/abs(bdd))*100 if (bdd < 0 and np.isfinite(bdd)) else np.nan
        cal_b = sb["calmar"] if np.isfinite(sb["calmar"]) else np.nan
        cal_f = sf["calmar"] if np.isfinite(sf["calmar"]) else np.nan
        # "better" = flat-side Calmar >= base Calmar (NaN base Calmar -> skip)
        if np.isfinite(cal_b) and np.isfinite(cal_f) and cal_f >= cal_b: n_better += 1
        elif np.isfinite(ddimp) and ddimp > 0 and not np.isfinite(cal_b): n_better += 1
        print(f"  {f:>5}{int(m.sum()):>6}{bdd:>+9.0f}{fdd:>+9.0f}{ddimp:>+8.1f}"
              f"{sb['sharpe']:>+8.2f}{sf['sharpe']:>+8.2f}{cal_b:>+8.2f}{cal_f:>+8.2f}{side_pnl.get(f,0.0):>+9.0f}", flush=True)
    return n_better, n_eval


# ---------------------------------------------------------------------------
# LOFO: leave-one-fold-out. Recompute the flat_side-vs-base Calmar IMPROVEMENT on
# the full series with one fold's cycles removed (drop in turn). Tests whether the
# in-sample Calmar lift survives without the single worst (deep-DD) fold.
# ---------------------------------------------------------------------------
def lofo_report(label, times, pnl_base, pnl_flat, regimes, fold_by_time):
    folds = sorted(set(fold_by_time.values())) if fold_by_time else []
    if not folds:
        print("  (no fold column for LOFO)", flush=True); return
    fold_arr = np.array([fold_by_time.get(t, -1) for t in times])
    side_pnl = side_contrib_by_fold(times, pnl_base, regimes, fold_by_time)

    cal_b_all = calmar_of(pnl_base); cal_f_all = calmar_of(pnl_flat)
    print(f"  FULL: base Calmar {cal_b_all:+.2f} | flat Calmar {cal_f_all:+.2f} | "
          f"lift {cal_f_all-cal_b_all:+.2f}", flush=True)
    print(f"  {'drop':>5}{'sidePnL':>9}{'baseCal':>9}{'flatCal':>9}{'lift':>8}{'Δlift_vs_full':>14}", flush=True)
    for f in folds:
        keep = fold_arr != f
        cb = calmar_of(pnl_base[keep]); cf = calmar_of(pnl_flat[keep])
        lift = cf - cb if (np.isfinite(cf) and np.isfinite(cb)) else np.nan
        dvs = lift - (cal_f_all - cal_b_all) if np.isfinite(lift) else np.nan
        print(f"  {('-f'+str(f)):>5}{side_pnl.get(f,0.0):>+9.0f}{cb:>+9.2f}{cf:>+9.2f}{lift:>+8.2f}{dvs:>+14.2f}", flush=True)


def run_universe(label, preds_path, want_chart):
    times, cyc_w_base, cyc_w_flat, rs, fold_by_time, regimes = build_universe(preds_path, label)
    n_side = sum(1 for r in regimes if r == "side")

    # ---- summary table: base vs flat_side, all cost levels ----
    print(f"\n=== {label}: base vs flat_side by cost ===", flush=True)
    print(f"  {'cost':>5}{'arm':>11}{'Sharpe':>8}{'maxDD':>9}{'Calmar':>8}{'totPnL':>9}{'%pos':>7}", flush=True)
    pnl_b_by_cost = {}; pnl_f_by_cost = {}
    for cost_bps in COSTS_BPS:
        cost = cost_bps*1e-4
        pnl_b = heldbook(times, cyc_w_base, rs, cost)
        pnl_f = heldbook(times, cyc_w_flat, rs, cost)
        pnl_b_by_cost[cost_bps] = pnl_b; pnl_f_by_cost[cost_bps] = pnl_f
        sb = stats(pnl_b); sf = stats(pnl_f)
        print(f"  {cost_bps:>5.1f}{'base':>11}{sb['sharpe']:>+8.2f}{sb['maxDD']:>+9.0f}{sb['calmar']:>+8.2f}{sb['totPnL']:>+9.0f}{sb['pct_pos']:>7.1f}", flush=True)
        print(f"  {cost_bps:>5.1f}{'flat_side':>11}{sf['sharpe']:>+8.2f}{sf['maxDD']:>+9.0f}{sf['calmar']:>+8.2f}{sf['totPnL']:>+9.0f}{sf['pct_pos']:>7.1f}", flush=True)

    # headline @4.5bps
    pnl_b = pnl_b_by_cost[4.5]; pnl_f = pnl_f_by_cost[4.5]
    sb = stats(pnl_b); sf = stats(pnl_f)
    dd_red = (1-abs(sf['maxDD'])/abs(sb['maxDD']))*100 if sb['maxDD'] < 0 else np.nan
    print(f"\n  [@4.5bps] base:      Sharpe {sb['sharpe']:+.2f}  maxDD {sb['maxDD']:+.0f}  Calmar {sb['calmar']:+.2f}  totPnL {sb['totPnL']:+.0f}", flush=True)
    print(f"  [@4.5bps] flat_side: Sharpe {sf['sharpe']:+.2f} (Δ {sf['sharpe']-sb['sharpe']:+.2f})  "
          f"maxDD {sf['maxDD']:+.0f} (DD red {dd_red:+.1f}%)  Calmar {sf['calmar']:+.2f} (Δ {sf['calmar']-sb['calmar']:+.2f})  "
          f"totPnL {sf['totPnL']:+.0f}  (side cycles {n_side}/{len(times)} now FLAT)", flush=True)

    # per-fold (G5) @4.5bps
    print(f"\n  --- {label} per-fold (G5) @4.5bps: base vs flat_side + side-regime PnL contribution ---", flush=True)
    nb, ne = per_fold_report(label, times, pnl_b, pnl_f, regimes, fold_by_time)
    print(f"  flat_side Calmar >= base Calmar in {nb}/{ne} folds", flush=True)

    # LOFO @4.5bps
    print(f"\n  --- {label} LOFO (G5): drop each fold, recompute flat_side-vs-base Calmar lift @4.5bps ---", flush=True)
    lofo_report(label, times, pnl_b, pnl_f, regimes, fold_by_time)

    # by-year @4.5bps
    pbb = pd.Series(pnl_b*1e4, index=pd.to_datetime(times)); pff = pd.Series(pnl_f*1e4, index=pd.to_datetime(times))
    print(f"\n  --- {label} by-year @4.5bps ---", flush=True)
    for yr in sorted(set(pbb.index.year)):
        gb = pbb[pbb.index.year == yr]; gf = pff[pff.index.year == yr]
        eb = gb.cumsum(); ef = gf.cumsum()
        print(f"    {yr}: base Sh {ann(gb/1e4):+.2f} PnL {gb.sum():+.0f} DD {(eb-eb.cummax()).min():+.0f} | "
              f"flat Sh {ann(gf/1e4):+.2f} PnL {gf.sum():+.0f} DD {(ef-ef.cummax()).min():+.0f}", flush=True)

    # ---- per-cycle parquet for Evaluation (G4 matched-active placebo, G6 paired CI) ----
    out = {
        "open_time": pd.to_datetime(times),
        "fold": [fold_by_time.get(t, -1) for t in times],
        "regime": regimes,
        "is_side": [r == "side" for r in regimes],
        "is_active_base": [bool(w) for w in cyc_w_base],   # cycles the BASE arm actually trades
    }
    for cost_bps in COSTS_BPS:
        tag = f"{int(round(cost_bps*10)):03d}"             # 010, 030, 045 (bps*10)
        out[f"pnl_base_c{tag}"] = pnl_b_by_cost[cost_bps]
        out[f"pnl_flatside_c{tag}"] = pnl_f_by_cost[cost_bps]
    # convenience aliases at the 4.5bps production cost
    out["pnl_base"] = pnl_b_by_cost[4.5]
    out["pnl_flatside"] = pnl_f_by_cost[4.5]
    df_out = pd.DataFrame(out)
    pq = OUT/f"X121_percycle_{label}.parquet"
    df_out.to_parquet(pq, index=False)
    print(f"\n  per-cycle series -> {pq}", flush=True)

    if want_chart:
        eqb = pbb.cumsum(); eqf = pff.cumsum()
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        a1.plot(eqb.index, eqb.values, color="grey", lw=1.2, label=f"base (Sh {sb['sharpe']:+.2f}, DD {sb['maxDD']:+.0f}, Cal {sb['calmar']:+.2f})")
        a1.plot(eqf.index, eqf.values, color="navy", lw=1.3, label=f"flat_side (Sh {sf['sharpe']:+.2f}, DD {sf['maxDD']:+.0f}, Cal {sf['calmar']:+.2f})")
        a1.axhline(0, color="grey", lw=0.6); a1.grid(alpha=0.3); a1.legend(loc="upper left", fontsize=8)
        a1.set_title(f"{label}: structural side->FLAT @4.5bps (side {n_side}/{len(times)} cycles FLATted)")
        a1.set_ylabel("cum PnL (bps)")
        ddb = eqb-eqb.cummax(); ddf = eqf-eqf.cummax()
        a2.fill_between(ddb.index, ddb.values, 0, color="grey", alpha=0.35, label="base DD")
        a2.fill_between(ddf.index, ddf.values, 0, color="navy", alpha=0.40, label="flat_side DD")
        a2.grid(alpha=0.3); a2.set_ylabel("drawdown (bps)"); a2.legend(loc="lower left", fontsize=8)
        fig.tight_layout(); png = OUT/f"X121_equity_dd_{label}.png"; fig.savefig(png, dpi=110)
        print(f"  chart -> {png}", flush=True)


def main():
    t0 = time.time()
    np.random.seed(SEED)
    print("=== X121 structural side->FLAT regime-map change ===", flush=True)
    print(f"regime map: bull->mom30 (unchanged), bear->FLAT (unchanged), side->FLAT (NEW, "
          f"parameter-free). K={K}, HOLD={HOLD}, costs {COSTS_BPS} bps. seed={SEED}", flush=True)
    run_universe("HL70", HL70_PREDS, want_chart=True)
    run_universe("S44", S44_PREDS, want_chart=True)
    print(f"\nDone [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
