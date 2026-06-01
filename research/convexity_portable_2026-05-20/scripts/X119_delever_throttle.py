"""X119 — De-lever-ONLY parameter-free realized-vol throttle on the held book.

Spec (research/handoff.md, iter-001): scale ALL held-book weights by a single
book-level factor

    rv_t  = std(pnl[t-W : t])             # trailing realized vol of the book's OWN per-cycle net PnL
    tgt_t = expanding/PIT median of rv     # reference, no future data, warmup -> s_t = 1.0
    s_t   = clip(tgt_t / rv_t, FLOOR, 1.0) # HARD CAP 1.0 -- never lever up (no min(2.0,...))
    net   = {s: v * s_t}                   # applied BEFORE turnover/cost so cost scales with gross

Single structural constants (NOT swept for Sharpe):
    W     = 42 cycles  (~7d, one sleeve length, as X97's trailing-vol window)
    FLOOR = 0.3        (guard so a noisy spike can't fully exit)

PIT: rv_t uses only PnL realized strictly before cycle t. Because the held book
overlaps HOLD=6 sleeves, a cycle's net PnL is not fully realized until HOLD cycles
later, so we LAG the trailing-vol window by HOLD (rv_t = std of pnl[<= t-HOLD]).
This is the conservative reading of the spec ("if any sleeve overlap could leak,
lag the window by HOLD"). pnl[t] is appended AFTER sizing (X97/X117 ordering).

Runs on HL70 (production) and the 44-sym base (robustness), base vs throttled, at
cost {1,3,4.5} bps/leg. Emits per-cycle net PnL + applied scale + fold for both
arms to parquet so the Evaluation agent can run the matched placebo (G4) and the
block-bootstrap paired CI (G6). Also prints per-fold and by-year breakdowns (G5).

This script does NOT modify X116/X117/X97 or any cached preds.
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
W = 42          # trailing-vol window (cycles) — fixed structural value, NOT swept
FLOOR = 0.3     # de-lever floor — fixed structural guard, NOT swept
WARMUP = W      # need >= W pre-cycle (post-lag) realized PnL before throttling
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
    """pnl_series in raw return units. Returns dict of contract metrics (bps where shown)."""
    p = pd.Series(pnl_series).dropna()
    pb = p*1e4
    eq = pb.cumsum()
    dd = (eq - eq.cummax())
    mdd = dd.min()
    sh = ann(p)
    annr = pb.mean()*6*365
    cal = (annr/abs(mdd)) if (mdd < 0 and np.isfinite(mdd)) else np.nan
    return {"sharpe": sh, "maxDD": mdd, "calmar": cal, "totPnL": eq.iloc[-1] if len(eq) else np.nan,
            "pct_pos": (pb > 0).mean()*100}


# ---------------------------------------------------------------------------
# Build the regime-hybrid cycle weights once (identical construction to X117/X116).
# Returns: times (list of timestamps), cyc_w (list of {sym: weight}), rs (list of
# {sym: return_pct}), fold_by_time (dict ts->fold).
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

    cyc_w = []; rs = []
    for ot in times:
        g = by_t[ot]; rg = g["regime"].iloc[0]
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
    print(f"  {len(syms)} syms, {len(times)} cycles, {pd.Timestamp(times[0]).date()}->{pd.Timestamp(times[-1]).date()}", flush=True)
    return list(times), cyc_w, rs, fold_by_time


# ---------------------------------------------------------------------------
# Held-book loop. throttle=False reproduces the X117 base; throttle=True applies
# the de-lever-only realized-vol scale. Returns (pnl array, scale array).
# ---------------------------------------------------------------------------
def heldbook(times, cyc_w, rs, cost, throttle):
    prev = {}; pnl = []; scales = []
    rv_hist = []          # PIT history of trailing-vol values rv_i (each = std over a W-window)
    for t in range(len(cyc_w)):
        active = cyc_w[max(0, t-HOLD+1):t+1]; net = {}
        for w in active:
            for s, wt in w.items(): net[s] = net.get(s, 0)+wt/HOLD

        # ---- de-lever throttle (PIT: only pre-cycle realized PnL, HOLD-lagged) ----
        s_t = 1.0
        if throttle:
            # cycle t's own PnL is not fully realized until t+HOLD (sleeve overlap),
            # so the freshest PnL usable at decision time t is pnl[t-HOLD]. We use the
            # set of PnLs realized strictly before cycle (t-HOLD+1): avail = pnl[:t-HOLD+1].
            n_avail = max(0, t-HOLD+1)
            if n_avail >= W:
                win = np.asarray(pnl[n_avail-W:n_avail], dtype=np.float64)  # most recent W realized PnLs
                rv = win.std() if np.isfinite(win).all() else np.nan
                if np.isfinite(rv) and rv > 0:
                    rv_hist.append(rv)                       # extend PIT trailing-vol history
                    tgt = float(np.median(rv_hist))          # expanding/PIT median (no future data)
                    s_t = float(np.clip(tgt/rv, FLOOR, 1.0)) # HARD CAP 1.0 — never lever up
        net = {s: v*s_t for s, v in net.items()}
        scales.append(s_t)

        alls = set(net) | set(prev)
        turn = sum(abs(net.get(s, 0)-prev.get(s, 0)) for s in alls)
        rl = rs[t]
        cyc = sum(net.get(s, 0)*rl.get(s, 0.0) for s in net if np.isfinite(rl.get(s, 0.0)))
        if not np.isfinite(cyc): cyc = 0.0          # explicit NaN guard (totPnL +nan bug)
        pnl.append(cyc - turn*0.5*cost)
        prev = net
    return np.asarray(pnl, dtype=np.float64), np.asarray(scales, dtype=np.float64)


def per_fold_report(times, pnl_base, pnl_thr, fold_by_time):
    folds = sorted(set(fold_by_time.values())) if fold_by_time else []
    if not folds:
        return 0, 0
    ts = [pd.Timestamp(t) for t in times]
    fold_arr = np.array([fold_by_time.get(t, -1) for t in times])
    n_better = 0; n_eval = 0
    print(f"  {'fold':>5}{'baseDD':>10}{'thrDD':>10}{'DDimprv%':>10}{'baseSh':>8}{'thrSh':>8}", flush=True)
    for f in folds:
        m = fold_arr == f
        if m.sum() < 3: continue
        n_eval += 1
        bdd = (np.cumsum(pnl_base[m]*1e4)-np.maximum.accumulate(np.cumsum(pnl_base[m]*1e4))).min()
        tdd = (np.cumsum(pnl_thr[m]*1e4)-np.maximum.accumulate(np.cumsum(pnl_thr[m]*1e4))).min()
        impr = (1-abs(tdd)/abs(bdd))*100 if bdd < 0 else np.nan
        if np.isfinite(impr) and impr > 0: n_better += 1
        print(f"  {f:>5}{bdd:>+10.0f}{tdd:>+10.0f}{impr:>+10.1f}{ann(pnl_base[m]):>+8.2f}{ann(pnl_thr[m]):>+8.2f}", flush=True)
    return n_better, n_eval


def run_universe(label, preds_path, want_chart):
    times, cyc_w, rs, fold_by_time = build_universe(preds_path, label)
    rows = []
    # primary cost = 4.5 for the per-cycle series + per-fold + chart
    for cost_bps in COSTS_BPS:
        cost = cost_bps*1e-4
        pnl_b, sc_b = heldbook(times, cyc_w, rs, cost, throttle=False)
        pnl_t, sc_t = heldbook(times, cyc_w, rs, cost, throttle=True)
        sb = stats(pnl_b); st = stats(pnl_t)
        rows.append((cost_bps, "base", sb)); rows.append((cost_bps, "throttle", st))
        if abs(cost_bps-4.5) < 1e-9:
            keep = (times, pnl_b, sc_b, pnl_t, sc_t, fold_by_time)

    print(f"\n=== {label}: base vs throttle by cost ===", flush=True)
    print(f"  {'cost':>5}{'arm':>10}{'Sharpe':>8}{'maxDD':>9}{'Calmar':>8}{'totPnL':>9}{'%pos':>7}", flush=True)
    for cb, arm, s in rows:
        print(f"  {cb:>5.1f}{arm:>10}{s['sharpe']:>+8.2f}{s['maxDD']:>+9.0f}{s['calmar']:>+8.2f}{s['totPnL']:>+9.0f}{s['pct_pos']:>7.1f}", flush=True)

    times, pnl_b, sc_b, pnl_t, sc_t, fold_by_time = keep
    sb = stats(pnl_b); st = stats(pnl_t)
    print(f"\n  [@4.5bps] maxDD {sb['maxDD']:+.0f} -> {st['maxDD']:+.0f}  "
          f"(DD reduction {(1-abs(st['maxDD'])/abs(sb['maxDD']))*100:+.1f}%);  "
          f"Sharpe {sb['sharpe']:+.2f} -> {st['sharpe']:+.2f} (Δ {st['sharpe']-sb['sharpe']:+.2f});  "
          f"Calmar {sb['calmar']:+.2f} -> {st['calmar']:+.2f}", flush=True)
    print(f"  scale dist: min {sc_t.min():.2f}  p5 {np.percentile(sc_t,5):.2f}  median {np.median(sc_t):.2f}  "
          f"mean {sc_t.mean():.2f}  frac<1.0 {(sc_t<0.999).mean()*100:.1f}%  frac=FLOOR {(sc_t<=FLOOR+1e-9).mean()*100:.1f}%", flush=True)

    # per-fold (G5)
    print(f"\n  --- {label} per-fold DD (G5) @4.5bps ---", flush=True)
    nb, ne = per_fold_report(times, pnl_b, pnl_t, fold_by_time)
    print(f"  DD improved in {nb}/{ne} folds", flush=True)

    # by-year (extra context)
    pbb = pd.Series(pnl_b*1e4, index=pd.to_datetime(times)); ptb = pd.Series(pnl_t*1e4, index=pd.to_datetime(times))
    print(f"  --- {label} by-year @4.5bps ---", flush=True)
    for yr in sorted(set(pbb.index.year)):
        gb = pbb[pbb.index.year == yr]; gt = ptb[ptb.index.year == yr]
        eb = gb.cumsum(); et = gt.cumsum()
        print(f"    {yr}: base Sh {ann(gb/1e4):+.2f} DD {(eb-eb.cummax()).min():+.0f} | "
              f"thr Sh {ann(gt/1e4):+.2f} DD {(et-et.cummax()).min():+.0f}", flush=True)

    # save per-cycle series for placebo/bootstrap (G4/G6)
    df_out = pd.DataFrame({
        "open_time": pd.to_datetime(times),
        "fold": [fold_by_time.get(t, -1) for t in times],
        "pnl_base": pnl_b,
        "pnl_throttle": pnl_t,
        "scale": sc_t,
    })
    pq = OUT/f"X119_percycle_{label}.parquet"
    df_out.to_parquet(pq, index=False)
    print(f"  per-cycle series -> {pq}", flush=True)

    if want_chart:
        eqb = (pbb.cumsum()); eqt = (ptb.cumsum())
        ddb = eqb-eqb.cummax(); ddt = eqt-eqt.cummax()
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        a1.plot(eqb.index, eqb.values, color="grey", lw=1.1, label=f"base (Sh {sb['sharpe']:+.2f}, DD {sb['maxDD']:+.0f})")
        a1.plot(eqt.index, eqt.values, color="navy", lw=1.3, label=f"de-lever throttle (Sh {st['sharpe']:+.2f}, DD {st['maxDD']:+.0f})")
        a1.axhline(0, color="grey", lw=0.6); a1.grid(alpha=0.3); a1.legend(loc="upper left", fontsize=9)
        a1.set_title(f"{label}: de-lever-only vol throttle (W={W}, FLOOR={FLOOR}, cap=1.0) @4.5bps")
        a1.set_ylabel("cum PnL (bps)")
        a2.fill_between(ddb.index, ddb.values, 0, color="grey", alpha=0.35)
        a2.fill_between(ddt.index, ddt.values, 0, color="crimson", alpha=0.45)
        a2.grid(alpha=0.3); a2.set_ylabel("drawdown (bps)")
        fig.tight_layout(); png = OUT/f"X119_equity_dd_{label}.png"; fig.savefig(png, dpi=110)
        print(f"  chart -> {png}", flush=True)


def main():
    t0 = time.time()
    np.random.seed(SEED)
    print("=== X119 de-lever-ONLY realized-vol throttle (parameter-free) ===", flush=True)
    print(f"W={W} cycles, FLOOR={FLOOR}, cap=1.0 (NO lever-up), HOLD-lagged PIT, costs {COSTS_BPS} bps", flush=True)
    run_universe("HL70", HL70_PREDS, want_chart=True)
    run_universe("S44", S44_PREDS, want_chart=True)
    print(f"\nDone [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
