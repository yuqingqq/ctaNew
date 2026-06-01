"""X120 — Correlation-aware sideways regime gate on the held book.

Spec (research/handoff.md, iter-002): the deep HL70 drawdown is a side-regime, long-beta
grind that occurs precisely when alts co-move (cross-sectional mean-reversion spread
collapses). Discriminator = trailing cross-asset correlation `corr7d`. Add a correlation
axis to the regime rule:

    bull  (BTC-30d > +10%)  -> momentum (mom30), UNCHANGED. corr does NOT gate bull.
    bear  (BTC-30d < -10%)  -> FLAT, UNCHANGED.
    side                    -> if pr_t >= THR -> FLAT (empty book, like bear);
                               else            -> mean-rev beta-neutral (UNCHANGED).

Signal construction (all PIT):
  corr7d_t : mean upper-triangle pairwise correlation of alt 4h log-returns over the
             STRICTLY TRAILING 42-cycle (~7d) window. The window EXCLUDES the current
             cycle (rt.iloc[lo:i], not :i+1). min_periods 10 (corr) / 20 (window). Finite
             upper-triangle mean. Identical to iter002_hl70_dd_anatomy.py::avg_pair_corr.
  pr_t     : EXPANDING percentile rank of corr7d within its own past history,
             pr_t = mean(hist[:t] <= corr7d_t). Strictly running/expanding — NO full-series
             pd.qcut / quantile. Warmup 100 cycles -> NaN (treated as "trade", never FLAT).
  pr_lag   : pr_t lagged 1 cycle (conservative vs sleeve overlap). The gate at cycle t uses
             pr_lag_t = pr_{t-1}, i.e. info available strictly before the decision cycle.

THR (the single TUNED parameter): grid {0.60, 0.70, 0.80}. The script emits per-cycle
pnl_base + pnl_gated_<THR> + side_flat_<THR> mask + fold at cost {1,3,4.5} bps so the
Evaluation agent can (a) run nested-OOS THR selection on past folds, (b) G4 matched
side-pool placebo (randomly FLAT the same number of side cycles), (c) G6 paired CI.
Structural fallback THR = 0.70.

Runs on HL70 (production) and S44 (x70_v0_3yr_preds, robustness). Base arm reproduces
X117 (Sharpe +1.93 / maxDD -5674 @4.5bps on HL70). Explicit NaN guards (totPnL nan bug).
Does NOT modify X116/X117/X119 or cached preds.
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
CORR_W = 42        # trailing-corr window (cycles, ~7d) — fixed structural value
WARMUP = 100       # expanding-percentile warmup (cycles) — fixed structural value
THRS = [0.60, 0.70, 0.80]   # the single TUNED parameter (grid)
THR_FALLBACK = 0.70         # structural fallback if nested-OOS is unstable
COSTS_BPS = [1.0, 3.0, 4.5]
SEED = 12345
Y26 = pd.Timestamp("2026-01-01", tz="UTC")


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


# ---------------------------------------------------------------------------
# corr7d: mean upper-triangle pairwise correlation of alt 4h returns, strictly
# trailing CORR_W cycles (window EXCLUDES current cycle). Identical to the anatomy
# script's avg_pair_corr. rt is the per-symbol 4h-return matrix aligned to the
# cycle grid `idx`.
# ---------------------------------------------------------------------------
def avg_pair_corr(rt, idx, window=CORR_W):
    out = pd.Series(index=idx, dtype=float)
    for i in range(len(idx)):
        lo = max(0, i-window)
        sub = rt.iloc[lo:i]                      # strictly trailing — excludes current cycle i
        if len(sub) < 20: continue
        cmat = sub.corr(min_periods=10).values
        iu = np.triu_indices_from(cmat, k=1)
        vals = cmat[iu]; vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            out.iloc[i] = np.nanmean(vals)
    return out


# ---------------------------------------------------------------------------
# Expanding (PIT) percentile rank of a series within its own past history.
# pr_t = mean(hist[:t] <= x_t). Strictly running — no full-series quantile.
# Warmup -> NaN (caller treats NaN as "trade", never FLAT).
# ---------------------------------------------------------------------------
def expanding_pct_rank(s, warmup=WARMUP):
    vals = s.to_numpy(dtype=np.float64)
    out = np.full(len(vals), np.nan)
    hist = []                                    # finite corr7d values seen so far (strictly past)
    for i in range(len(vals)):
        x = vals[i]
        if len(hist) >= warmup and np.isfinite(x):
            h = np.asarray(hist, dtype=np.float64)
            out[i] = float((h <= x).mean())      # expanding percentile vs strictly-prior history
        if np.isfinite(x):
            hist.append(x)                        # append AFTER ranking -> hist[:t] excludes x_t
    return pd.Series(out, index=s.index)


# ---------------------------------------------------------------------------
# Build the regime-hybrid cycle weights once (identical construction to X117/X116).
# Additionally compute the PIT corr-gate signal (pr_lag) aligned to the cycle grid.
# Returns: times, cyc_w, rs, fold_by_time, regimes, pr_lag (np array over times).
# ---------------------------------------------------------------------------
def build_universe(preds_path, label):
    print(f"\n--- building {label} ({preds_path.name}) ---", flush=True)
    cols = ["symbol", "open_time", "pred", "return_pct", "fold"]
    d = pd.read_parquet(preds_path, columns=cols)
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    d = d[(d["open_time"].dt.hour % 4 == 0) & (d["open_time"].dt.minute == 0)].copy()

    btc = load_close("BTCUSDT"); b4 = btc[(btc.index.hour % 4 == 0) & (btc.index.minute == 0)]
    br = np.log(b4/b4.shift(1)); bvar = br.rolling(180, min_periods=42).var()
    syms = sorted(d["symbol"].unique()); mom_rows = []; beta_map = {}; ret4_map = {}
    for sym in syms:
        c = load_close(sym)
        if c is None: continue
        c4 = c[(c.index.hour % 4 == 0) & (c.index.minute == 0)]
        mom_rows.append(pd.DataFrame({"symbol": sym, "open_time": c4.index,
                                      "mom30": (c4/c4.shift(180)-1).shift(1).values}))
        r = np.log(c4/c4.shift(1)); ri, bi = r.align(br, join="inner")
        beta_map[sym] = (ri.rolling(180, min_periods=42).cov(bi)/bvar.reindex(ri.index).replace(0, np.nan)).shift(1)
        ret4_map[sym] = r                          # 4h log return for cross-asset correlation
    mom = pd.concat(mom_rows, ignore_index=True); mom["open_time"] = pd.to_datetime(mom["open_time"], utc=True)
    betas = pd.concat([s.rename(k) for k, s in beta_map.items()], axis=1)
    ret4 = pd.concat([s.rename(k) for k, s in ret4_map.items()], axis=1).sort_index()
    d = d.merge(mom, on=["symbol", "open_time"], how="left")
    btc30 = (b4/b4.shift(180)-1).to_frame("b30").reset_index(); btc30["open_time"] = pd.to_datetime(btc30["open_time"], utc=True)
    d = d.merge(btc30, on="open_time", how="left").dropna(subset=["b30"])
    d["regime"] = np.where(d["b30"] > 0.10, "bull", np.where(d["b30"] < -0.10, "bear", "side"))

    times = sorted(d["open_time"].unique()); by_t = {ot: g for ot, g in d.groupby("open_time")}
    fold_by_time = {ot: int(g["fold"].iloc[0]) for ot, g in by_t.items() if "fold" in g.columns}

    cyc_w = []; rs = []; regimes = []
    for ot in times:
        g = by_t[ot]; rg = g["regime"].iloc[0]; regimes.append(rg)
        rs.append(dict(zip(g["symbol"], g["return_pct"])))
        if rg == "bear": cyc_w.append({}); continue
        key = "mom30" if rg == "bull" else "pred"; gg = g.dropna(subset=[key])
        if len(gg) < 2*K: cyc_w.append({}); continue
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
        cyc_w.append(w)

    # ---- PIT corr-gate signal aligned to the cycle grid ----
    idx = pd.DatetimeIndex(times)
    rt = ret4.reindex(idx)
    corr7d = avg_pair_corr(rt, idx, CORR_W)        # strictly trailing, excludes current cycle
    pr = expanding_pct_rank(corr7d, WARMUP)        # expanding percentile, warmup -> NaN
    pr_lag = pr.shift(1)                            # lag 1 cycle for the gate decision
    pr_lag_arr = pr_lag.to_numpy(dtype=np.float64)

    n_warm = int(np.isnan(pr_lag_arr).sum())
    print(f"  {len(syms)} syms, {len(times)} cycles, {pd.Timestamp(times[0]).date()}->{pd.Timestamp(times[-1]).date()}", flush=True)
    print(f"  corr7d: non-nan {corr7d.notna().sum()}/{len(corr7d)} mean {corr7d.mean():.3f}; "
          f"pr_lag warmup/NaN {n_warm} ({n_warm/len(pr_lag_arr)*100:.1f}%, treated as TRADE)", flush=True)
    return list(times), cyc_w, rs, fold_by_time, regimes, pr_lag_arr


# ---------------------------------------------------------------------------
# Held-book loop. thr=None -> base (reproduces X117). thr float -> corr gate:
# in a side cycle, if pr_lag[t] >= thr -> emit empty net book (FLAT) for that cycle's
# new sleeve (the cycle contributes no NEW weights; held book still decays prior sleeves
# exactly like X117's bear cycles). Returns (pnl array, side_flat mask array).
# ---------------------------------------------------------------------------
def heldbook(times, cyc_w, rs, regimes, pr_lag, cost, thr):
    prev = {}; pnl = []; flat_mask = []
    for t in range(len(cyc_w)):
        # corr-gate: only the side branch changes. NaN pr_lag (warmup) -> trade.
        is_flat = False
        if thr is not None and regimes[t] == "side":
            p = pr_lag[t]
            if np.isfinite(p) and p >= thr:
                is_flat = True
        flat_mask.append(is_flat)

        # Held book = overlap of the last HOLD sleeves. A FLATted side cycle emits an
        # empty sleeve (exactly like the bear branch). When thr is not None we honour the
        # FLAT decision made AT each historical sleeve's own cycle: flat_mask[k] is set for
        # all k<=t (forward iteration), so this is PIT-correct (no future decision used).
        if thr is not None:
            active = [({} if flat_mask[k] else cyc_w[k]) for k in range(max(0, t-HOLD+1), t+1)]
        else:
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
    return np.asarray(pnl, dtype=np.float64), np.asarray(flat_mask, dtype=bool)


def per_fold_report(times, pnl_base, pnl_gated, fold_by_time):
    folds = sorted(set(fold_by_time.values())) if fold_by_time else []
    if not folds: return 0, 0
    fold_arr = np.array([fold_by_time.get(t, -1) for t in times])
    n_better = 0; n_eval = 0
    print(f"  {'fold':>5}{'baseDD':>10}{'gateDD':>10}{'DDimprv%':>10}{'baseSh':>8}{'gateSh':>8}", flush=True)
    for f in folds:
        m = fold_arr == f
        if m.sum() < 3: continue
        n_eval += 1
        bdd = (np.cumsum(pnl_base[m]*1e4)-np.maximum.accumulate(np.cumsum(pnl_base[m]*1e4))).min()
        tdd = (np.cumsum(pnl_gated[m]*1e4)-np.maximum.accumulate(np.cumsum(pnl_gated[m]*1e4))).min()
        impr = (1-abs(tdd)/abs(bdd))*100 if bdd < 0 else np.nan
        if np.isfinite(impr) and impr > 0: n_better += 1
        print(f"  {f:>5}{bdd:>+10.0f}{tdd:>+10.0f}{impr:>+10.1f}{ann(pnl_base[m]):>+8.2f}{ann(pnl_gated[m]):>+8.2f}", flush=True)
    return n_better, n_eval


def run_universe(label, preds_path, want_chart):
    times, cyc_w, rs, fold_by_time, regimes, pr_lag = build_universe(preds_path, label)
    n_side = sum(1 for r in regimes if r == "side")

    # ---- summary table: base + each THR, all cost levels ----
    print(f"\n=== {label}: base vs corr-gate by THR x cost ===", flush=True)
    print(f"  {'cost':>5}{'arm':>14}{'Sharpe':>8}{'maxDD':>9}{'Calmar':>8}{'totPnL':>9}{'%pos':>7}{'nFLAT':>7}", flush=True)
    keep = None
    for cost_bps in COSTS_BPS:
        cost = cost_bps*1e-4
        pnl_b, _ = heldbook(times, cyc_w, rs, regimes, pr_lag, cost, thr=None)
        sb = stats(pnl_b)
        print(f"  {cost_bps:>5.1f}{'base':>14}{sb['sharpe']:>+8.2f}{sb['maxDD']:>+9.0f}{sb['calmar']:>+8.2f}{sb['totPnL']:>+9.0f}{sb['pct_pos']:>7.1f}{0:>7}", flush=True)
        gated = {}
        for thr in THRS:
            pnl_g, mask = heldbook(times, cyc_w, rs, regimes, pr_lag, cost, thr=thr)
            sg = stats(pnl_g); nflat = int(mask.sum())
            print(f"  {cost_bps:>5.1f}{('gate@'+format(thr,'.2f')):>14}{sg['sharpe']:>+8.2f}{sg['maxDD']:>+9.0f}{sg['calmar']:>+8.2f}{sg['totPnL']:>+9.0f}{sg['pct_pos']:>7.1f}{nflat:>7}", flush=True)
            gated[thr] = (pnl_g, mask)
        if abs(cost_bps-4.5) < 1e-9:
            keep = (pnl_b, gated)

    pnl_b, gated = keep
    sb = stats(pnl_b)
    print(f"\n  [@4.5bps] base: Sharpe {sb['sharpe']:+.2f} maxDD {sb['maxDD']:+.0f} Calmar {sb['calmar']:+.2f} "
          f"(side cycles {n_side}/{len(times)})", flush=True)
    for thr in THRS:
        pnl_g, mask = gated[thr]; sg = stats(pnl_g)
        dd_red = (1-abs(sg['maxDD'])/abs(sb['maxDD']))*100 if sb['maxDD'] < 0 else np.nan
        print(f"  [@4.5bps] gate@{thr:.2f}: Sharpe {sg['sharpe']:+.2f} (Δ {sg['sharpe']-sb['sharpe']:+.2f}) "
              f"maxDD {sg['maxDD']:+.0f} (DD red {dd_red:+.1f}%) Calmar {sg['calmar']:+.2f} "
              f"(Δ {sg['calmar']-sb['calmar']:+.2f}) | FLAT {int(mask.sum())} of {n_side} side cycles", flush=True)

    # per-fold (G5) for the fallback THR
    print(f"\n  --- {label} per-fold DD (G5) @4.5bps, gate@{THR_FALLBACK:.2f} (fallback) ---", flush=True)
    nb, ne = per_fold_report(times, pnl_b, gated[THR_FALLBACK][0], fold_by_time)
    print(f"  DD improved in {nb}/{ne} folds (gate@{THR_FALLBACK:.2f})", flush=True)

    # by-year context for the fallback THR
    pbb = pd.Series(pnl_b*1e4, index=pd.to_datetime(times)); ptb = pd.Series(gated[THR_FALLBACK][0]*1e4, index=pd.to_datetime(times))
    print(f"  --- {label} by-year @4.5bps (gate@{THR_FALLBACK:.2f}) ---", flush=True)
    for yr in sorted(set(pbb.index.year)):
        gb = pbb[pbb.index.year == yr]; gt = ptb[ptb.index.year == yr]
        eb = gb.cumsum(); et = gt.cumsum()
        print(f"    {yr}: base Sh {ann(gb/1e4):+.2f} DD {(eb-eb.cummax()).min():+.0f} | "
              f"gate Sh {ann(gt/1e4):+.2f} DD {(et-et.cummax()).min():+.0f}", flush=True)

    # ---- per-cycle parquet for Evaluation (G4 side-pool placebo, G6 paired CI, nested-OOS THR) ----
    out = {
        "open_time": pd.to_datetime(times),
        "fold": [fold_by_time.get(t, -1) for t in times],
        "regime": regimes,
        "pr_lag": pr_lag,                          # lagged expanding percentile rank (NaN in warmup)
        "is_side": [r == "side" for r in regimes], # the side-regime eligible pool for G4
        "pnl_base": pnl_b,
    }
    for thr in THRS:
        pnl_g, mask = gated[thr]
        tag = f"{int(round(thr*100)):03d}"
        out[f"pnl_gated_t{tag}"] = pnl_g
        out[f"side_flat_t{tag}"] = mask            # cycles the gate FLATted (all are side cycles)
    df_out = pd.DataFrame(out)
    pq = OUT/f"X120_percycle_{label}.parquet"
    df_out.to_parquet(pq, index=False)
    print(f"  per-cycle series -> {pq}", flush=True)

    if want_chart:
        eqb = pbb.cumsum()
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        a1.plot(eqb.index, eqb.values, color="grey", lw=1.1, label=f"base (Sh {sb['sharpe']:+.2f}, DD {sb['maxDD']:+.0f})")
        for thr, col in zip(THRS, ["seagreen", "navy", "darkorange"]):
            pnl_g, _ = gated[thr]; sg = stats(pnl_g)
            eqg = pd.Series(pnl_g*1e4, index=pd.to_datetime(times)).cumsum()
            a1.plot(eqg.index, eqg.values, color=col, lw=1.2, label=f"gate@{thr:.2f} (Sh {sg['sharpe']:+.2f}, DD {sg['maxDD']:+.0f})")
        a1.axhline(0, color="grey", lw=0.6); a1.grid(alpha=0.3); a1.legend(loc="upper left", fontsize=8)
        a1.set_title(f"{label}: corr-aware side gate (CORR_W={CORR_W}, warmup={WARMUP}) @4.5bps")
        a1.set_ylabel("cum PnL (bps)")
        ptg = pd.Series(gated[THR_FALLBACK][0]*1e4, index=pd.to_datetime(times)).cumsum()
        ddb = eqb-eqb.cummax(); ddg = ptg-ptg.cummax()
        a2.fill_between(ddb.index, ddb.values, 0, color="grey", alpha=0.35, label="base DD")
        a2.fill_between(ddg.index, ddg.values, 0, color="navy", alpha=0.40, label=f"gate@{THR_FALLBACK:.2f} DD")
        a2.grid(alpha=0.3); a2.set_ylabel("drawdown (bps)"); a2.legend(loc="lower left", fontsize=8)
        fig.tight_layout(); png = OUT/f"X120_equity_dd_{label}.png"; fig.savefig(png, dpi=110)
        print(f"  chart -> {png}", flush=True)


def main():
    t0 = time.time()
    np.random.seed(SEED)
    print("=== X120 correlation-aware sideways regime gate ===", flush=True)
    print(f"CORR_W={CORR_W} cycles (~7d, strictly trailing), expanding-pct warmup={WARMUP}, "
          f"pr lagged 1 cycle, THR grid {THRS} (fallback {THR_FALLBACK}), costs {COSTS_BPS} bps", flush=True)
    run_universe("HL70", HL70_PREDS, want_chart=True)
    run_universe("S44", S44_PREDS, want_chart=True)
    print(f"\nDone [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
