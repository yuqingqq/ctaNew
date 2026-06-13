"""iter-031 — DEPLOYMENT decision: pick the live SYMBOL SET (deploy universe) for HL70.

Decision-support (NOT adopt/reject). The full champion = regime-hybrid held-book (X117/X116) +
iter-012 vol-normalized reactive equity-DD stop (X125, k=2.0). We pick a ROBUST, RULE-BASED deploy
universe (NOT cherry-picked by past-IC — that overfits, proven across the project).

STEP 2 analyses:
  1. Liquidity ranking of HL70 by rolling dollar-volume (quote_volume from klines).
  2. Breadth-N sweep: champion on top-N-by-liquidity subsets (N=20,30,40,50,70). Sharpe/maxDD/Calmar.
  3. Composition stress: random-subset draws at recommended N + full — broad-based vs few-names.
  4. Liquidity-tier vs random: does top-N-by-LIQUIDITY do >= full/random-N (ex-ante tradable rule)?
  5. Transport: do breadth-N conclusions hold on EXT 2021-26?

The champion is universe-subset-aware: we rebuild the regime-hybrid cycle weights restricting the
long/short selection pool to the chosen symbol subset, then run the held book + vol-norm stop on it.
BTC regime label uses BTC (always present); alt-index for the stop's equity is the strategy's own PnL.
"""
from __future__ import annotations
import time
from pathlib import Path
import importlib.util as _ilu
import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
SCRIPTS = REPO/"research/convexity_portable_2026-05-20/scripts"
RC = REPO/"research/convexity_portable_2026-05-20/results/_cache"
KLINES = REPO/"data/ml/test/parquet/klines"
OUT = REPO/"outputs/iter031"; OUT.mkdir(parents=True, exist_ok=True)

HL70_PREDS = RC/"x64_HL70_V5mv3_7cx_filter_t0.3_preds.parquet"
EXT_PREDS = RC/"x113_ext_v0_preds.parquet"

K = 5; HOLD = 6; WIN = 180
PRIMARY_COST = 4.5e-4
ANN = 6*365
SEED = 12345

# iter-012 vol-norm reactive stop (fixed policy)
GFLOOR = 0.40; HEAL = 0.50; RDAYS = 90; VOL_WIN = 180; WARMUP = 60; REC_K = 2.0
SQRT_WIN = np.sqrt(VOL_WIN)

# stable/wrapped/duplicate hygiene exclusions (gold-pegged / USD stable proxies on HL)
HYGIENE_EXCLUDE = {"PAXGUSDT"}   # PAXG = tokenized gold (not a crypto-beta name); no stables in HL70

# ---- reuse X123 build_universe + X125 stop engine verbatim ----
_s123 = _ilu.spec_from_file_location("x123", SCRIPTS/"X123_altbear_short_probe.py")
x123 = _ilu.module_from_spec(_s123); _s123.loader.exec_module(x123)
load_close = x123.load_close


def ann(x):
    x = pd.Series(x).dropna()
    return x.mean()/x.std()*np.sqrt(ANN) if len(x) > 2 and x.std() > 0 else np.nan


def metrics(pnl_bps):
    p = pd.Series(pnl_bps).dropna()
    if len(p) < 3:
        return dict(Sharpe=np.nan, maxDD=np.nan, Calmar=np.nan, tot=np.nan, pct_pos=np.nan)
    eq = p.cumsum(); mdd = (eq - eq.cummax()).min()
    annr = p.mean()*ANN
    cal = annr/abs(mdd) if (mdd < 0 and np.isfinite(mdd)) else np.nan
    return dict(Sharpe=ann(p/1e4), maxDD=mdd, Calmar=cal, tot=eq.iloc[-1], pct_pos=(p > 0).mean()*100)


def build_panel(preds_path, label):
    """Build the per-cycle data needed to construct the regime-hybrid book for ANY symbol subset.
    Returns: times, by_t (per-time symbol->(pred,mom30,ret_pct)), regimes, betas df, fold_by_time,
    and the quote-volume liquidity ranking."""
    cols = ["symbol", "open_time", "pred", "return_pct", "fold"]
    d = pd.read_parquet(preds_path, columns=cols)
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    d = d[(d["open_time"].dt.hour % 4 == 0) & (d["open_time"].dt.minute == 0)].copy()

    btc = load_close("BTCUSDT"); b4 = btc[(btc.index.hour % 4 == 0) & (btc.index.minute == 0)]
    br = np.log(b4/b4.shift(1)); bvar = br.rolling(WIN, min_periods=42).var()
    syms = sorted(d["symbol"].unique())
    mom_rows = []; beta_map = {}; qv_med = {}
    for sym in syms:
        c = load_close(sym)
        if c is None:
            continue
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

    # liquidity ranking: median daily dollar volume (quote_volume) over the backtest window.
    # quote_volume is in quote (USDT) units -> a direct dollar-volume proxy. We aggregate 5m to daily
    # sums then take the median across the panel's date range (robust to spikes).
    t0d = pd.Timestamp(times[0]); t1d = pd.Timestamp(times[-1])
    for sym in syms:
        sd = KLINES/sym/"5m"
        if not sd.exists():
            qv_med[sym] = np.nan; continue
        dfs = [pd.read_parquet(f, columns=["open_time", "quote_volume"]) for f in sorted(sd.glob("*.parquet"))]
        q = pd.concat(dfs, ignore_index=True).drop_duplicates("open_time")
        q["open_time"] = pd.to_datetime(q["open_time"], utc=True)
        q = q[(q["open_time"] >= t0d) & (q["open_time"] <= t1d)]
        if len(q) == 0:
            qv_med[sym] = np.nan; continue
        daily = q.set_index("open_time")["quote_volume"].resample("1D").sum()
        daily = daily[daily > 0]
        qv_med[sym] = float(daily.median()) if len(daily) else np.nan
    liq = pd.Series(qv_med).sort_values(ascending=False)
    return dict(times=list(times), by_t=by_t, betas=betas, fold_by_time=fold_by_time, syms=syms, liq=liq)


def build_cyc_weights(P, subset):
    """Regime-hybrid cycle weights (X117 production) restricting the selection pool to `subset`.
    Returns (cyc_w list, rs list) aligned to P['times']."""
    subset = set(subset)
    times = P["times"]; by_t = P["by_t"]; betas = P["betas"]
    cyc_w = []; rs = []
    for ot in times:
        g0 = by_t[ot]
        g = g0[g0["symbol"].isin(subset)]
        rl = dict(zip(g["symbol"], g["return_pct"])); rs.append(rl)
        rg = g0["regime"].iloc[0]
        if rg == "bear" or len(g) < 2*K:
            cyc_w.append({}); continue
        key = "mom30" if rg == "bull" else "pred"
        gg = g.dropna(subset=[key])
        if len(gg) < 2*K:
            cyc_w.append({}); continue
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
    return cyc_w, rs


def heldbook(cyc_w, rs, cost):
    prev = {}; pnl = []
    for t in range(len(cyc_w)):
        active = cyc_w[max(0, t-HOLD+1):t+1]; net = {}
        for w in active:
            for s, wt in w.items(): net[s] = net.get(s, 0)+wt/HOLD
        alls = set(net) | set(prev)
        turn = sum(abs(net.get(s, 0)-prev.get(s, 0)) for s in alls)
        rl = rs[t]
        c = sum(net.get(s, 0)*rl.get(s, 0.0) for s in net if np.isfinite(rl.get(s, 0.0)))
        if not np.isfinite(c): c = 0.0
        pnl.append(c - turn*0.5*cost); prev = net
    return np.asarray(pnl, dtype=np.float64)


def volnorm_stop(cyc_w, rs, cost, k=REC_K):
    """Full vol-norm reactive stop (X125 mechanics): gross applied to positions BEFORE turnover/cost.
    Returns pnl_bps array."""
    n = len(cyc_w)
    pnl = np.empty(n); incr = np.empty(n); prev = {}
    eq = 0.0; peak = 0.0; stopped = False
    stop_peak = 0.0; trough = 0.0; stop_t = 0
    for t in range(n):
        dd = eq - peak
        if t >= 2:
            lo = max(0, t-VOL_WIN); seg = incr[lo:t]; seg = seg[np.isfinite(seg)]
            sigma = float(seg.std()) if len(seg) >= 2 else 0.0
        else:
            sigma = 0.0
        trig = k*sigma*SQRT_WIN
        can_fire = (t >= WARMUP) and (sigma > 0)
        if not stopped:
            if can_fire and (-dd >= trig):
                stopped = True; stop_peak = peak; trough = eq; stop_t = t
        else:
            trough = min(trough, eq); gap = stop_peak - trough
            healed = (gap > 0) and ((eq - trough) >= HEAL*gap)
            timed = (t - stop_t) >= RDAYS
            if (healed and eq > trough) or timed:
                stopped = False
        g = GFLOOR if stopped else 1.0
        active = cyc_w[max(0, t-HOLD+1):t+1]; net = {}
        for w in active:
            for s, wt in w.items(): net[s] = net.get(s, 0)+wt/HOLD
        scaled = {s: g*v for s, v in net.items()}
        alls = set(scaled) | set(prev)
        turn = sum(abs(scaled.get(s, 0.0)-prev.get(s, 0.0)) for s in alls)
        rl = rs[t]
        c = sum(scaled.get(s, 0.0)*rl.get(s, 0.0) for s in scaled if np.isfinite(rl.get(s, 0.0)))
        if not np.isfinite(c): c = 0.0
        pnl[t] = c - turn*0.5*cost
        step = pnl[t]*1e4; incr[t] = step if np.isfinite(step) else 0.0
        eq += incr[t]
        if eq > peak: peak = eq
        prev = scaled
    return pnl*1e4


def run_subset(P, subset, with_stop=True, cost=PRIMARY_COST):
    cyc_w, rs = build_cyc_weights(P, subset)
    if with_stop:
        pnl = volnorm_stop(cyc_w, rs, cost)
    else:
        pnl = heldbook(cyc_w, rs, cost)*1e4
    return pnl, metrics(pnl)


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    print("="*100)
    print("iter-031 — DEPLOY UNIVERSE: liquidity-ranked breadth-N for the full champion (hybrid + iter-012 stop)")
    print("="*100, flush=True)

    # ============================================================ build panels
    PH = build_panel(HL70_PREDS, "HL70")
    PE = build_panel(EXT_PREDS, "EXT")

    # ---- 1. liquidity ranking (HL70) ----
    print("\n" + "="*100)
    print("1. LIQUIDITY RANKING — HL70 by median daily dollar-volume (quote_volume, USDT)")
    print("="*100, flush=True)
    liq = PH["liq"].dropna()
    print(f"  {len(liq)} ranked syms; hygiene-exclude {sorted(HYGIENE_EXCLUDE)} (tokenized-gold/non-crypto-beta)")
    print(f"  {'rank':>4} {'symbol':<16}{'medDailyVol($M)':>16}")
    for i, (s, v) in enumerate(liq.items(), 1):
        tag = "  <-EXCLUDE(hygiene)" if s in HYGIENE_EXCLUDE else ""
        print(f"  {i:>4} {s:<16}{v/1e6:>16.1f}{tag}", flush=True)

    # tradable liquidity-ranked universe AFTER hygiene
    liq_clean = [s for s in liq.index if s not in HYGIENE_EXCLUDE]
    liq_df = pd.DataFrame({"symbol": liq.index, "med_daily_vol_usd": liq.values})
    liq_df["rank"] = np.arange(1, len(liq_df)+1)
    liq_df["excluded"] = liq_df["symbol"].isin(HYGIENE_EXCLUDE)
    liq_df.to_csv(OUT/"iter031_hl70_liquidity_rank.csv", index=False)

    # ---- 2. breadth-N sweep (HL70) ----
    print("\n" + "="*100)
    print("2. BREADTH-N SWEEP (HL70) — full champion on top-N-by-liquidity (hygiene-clean). @4.5bps")
    print("   base = regime-hybrid held-book; stop = + iter-012 vol-norm reactive stop (k=2.0)")
    print("="*100, flush=True)
    NGRID = [20, 30, 40, 50, 69]   # 69 = all HL70 minus PAXG hygiene; full=70 shown too
    print(f"  {'N':>4} {'kind':<6}{'Sharpe':>8}{'maxDD':>9}{'Calmar':>8}{'totPnL':>9}{'%pos':>7}", flush=True)
    bn_rows = []
    for N in NGRID + [70]:
        if N == 70:
            sub = list(liq.index)  # full incl PAXG
            tag = "ALL70"
        else:
            sub = liq_clean[:N]
            tag = f"top{N}"
        for kind, ws in (("base", False), ("stop", True)):
            pnl, m = run_subset(PH, sub, with_stop=ws)
            print(f"  {N:>4} {kind:<6}{m['Sharpe']:>+8.2f}{m['maxDD']:>+9.0f}{m['Calmar']:>+8.2f}"
                  f"{m['tot']:>+9.0f}{m['pct_pos']:>7.1f}", flush=True)
            bn_rows.append(dict(universe="HL70", N=N, tag=tag, kind=kind, **m))
    pd.DataFrame(bn_rows).to_csv(OUT/"iter031_breadthN_HL70.csv", index=False)

    # ---- 3. composition stress (HL70): random subsets at recommended N + full ----
    print("\n" + "="*100)
    print("3. COMPOSITION STRESS (HL70) — random subsets (no liquidity rule) at N=30,40 + reference top-N.")
    print("   30 random draws per N. Edge broad-based or hinging on a few names? @4.5bps, base book.")
    print("="*100, flush=True)
    N_DRAWS = 30
    cs_rows = []
    pool = liq_clean   # draw from the hygiene-clean pool
    for N in (30, 40):
        # reference: top-N by liquidity (the rule)
        _, mtop = run_subset(PH, liq_clean[:N], with_stop=False)
        sh = []; dd = []; cal = []
        for i in range(N_DRAWS):
            sub = list(rng.choice(pool, size=N, replace=False))
            _, m = run_subset(PH, sub, with_stop=False)
            sh.append(m["Sharpe"]); dd.append(m["maxDD"]); cal.append(m["Calmar"])
        sh = np.array(sh); dd = np.array(dd); cal = np.array(cal)
        liq_rank = (sh < mtop["Sharpe"]).mean()*100
        print(f"\n  N={N}: top-N(liquidity) Sharpe {mtop['Sharpe']:+.2f} maxDD {mtop['maxDD']:+.0f} "
              f"Calmar {mtop['Calmar']:+.2f}")
        print(f"        RANDOM-{N} ({N_DRAWS} draws): Sharpe mean {np.nanmean(sh):+.2f} std {np.nanstd(sh):.2f} "
              f"min {np.nanmin(sh):+.2f} max {np.nanmax(sh):+.2f}")
        print(f"        random maxDD mean {np.nanmean(dd):+.0f} worst {np.nanmin(dd):+.0f}; "
              f"random Calmar mean {np.nanmean(cal):+.2f} worst {np.nanmin(cal):+.2f}")
        print(f"        => top-N(liquidity) ranks p{liq_rank:.0f} of random-{N} draws "
              f"({'>= random (rule OK)' if liq_rank >= 50 else '< random median'})", flush=True)
        cs_rows.append(dict(N=N, top_sharpe=mtop["Sharpe"], rand_sh_mean=float(np.nanmean(sh)),
                            rand_sh_std=float(np.nanstd(sh)), rand_sh_min=float(np.nanmin(sh)),
                            rand_sh_max=float(np.nanmax(sh)), top_rank_pct=liq_rank))
    pd.DataFrame(cs_rows).to_csv(OUT/"iter031_composition_stress_HL70.csv", index=False)

    # ---- 4. liquidity-tier vs random at every N (does liquidity rule >= random median?) ----
    print("\n" + "="*100)
    print("4. LIQUIDITY-TIER vs RANDOM — is top-N-by-liquidity (ex-ante rule) >= random-N median? (Sharpe)")
    print("   20 random draws per N. @4.5bps base.")
    print("="*100, flush=True)
    lt_rows = []
    for N in (20, 30, 40, 50):
        _, mtop = run_subset(PH, liq_clean[:N], with_stop=False)
        # also bottom-N (least liquid) for contrast
        _, mbot = run_subset(PH, liq_clean[-N:], with_stop=False)
        sh = []
        for i in range(20):
            sub = list(rng.choice(pool, size=N, replace=False))
            _, m = run_subset(PH, sub, with_stop=False)
            sh.append(m["Sharpe"])
        sh = np.array(sh); rank = (sh < mtop["Sharpe"]).mean()*100
        print(f"  N={N:>2}: TOP-liq {mtop['Sharpe']:+.2f} | BOTTOM-liq {mbot['Sharpe']:+.2f} | "
              f"RANDOM median {np.nanmedian(sh):+.2f} (mean {np.nanmean(sh):+.2f}) -> top ranks p{rank:.0f}",
              flush=True)
        lt_rows.append(dict(N=N, top_sharpe=mtop["Sharpe"], bottom_sharpe=mbot["Sharpe"],
                            rand_median=float(np.nanmedian(sh)), top_rank=rank))
    pd.DataFrame(lt_rows).to_csv(OUT/"iter031_liqtier_vs_random_HL70.csv", index=False)

    # ---- 5. transport on EXT 2021-26 ----
    print("\n" + "="*100)
    print("5. TRANSPORT — breadth-N on EXT 2021-26 (23 syms). Do conclusions hold? @4.5bps")
    print("="*100, flush=True)
    liqE = PE["liq"].dropna()
    liqE_clean = [s for s in liqE.index if s not in HYGIENE_EXCLUDE]
    print(f"  EXT liquidity-ranked ({len(liqE_clean)} clean syms). N grid limited by 23-sym panel.")
    print(f"  {'N':>4} {'kind':<6}{'Sharpe':>8}{'maxDD':>9}{'Calmar':>8}{'totPnL':>9}", flush=True)
    et_rows = []
    for N in [10, 15, 20, 23]:
        sub = liqE_clean[:N] if N < 23 else list(liqE.index)
        for kind, ws in (("base", False), ("stop", True)):
            pnl, m = run_subset(PE, sub, with_stop=ws)
            print(f"  {N:>4} {kind:<6}{m['Sharpe']:>+8.2f}{m['maxDD']:>+9.0f}{m['Calmar']:>+8.2f}{m['tot']:>+9.0f}",
                  flush=True)
            et_rows.append(dict(universe="EXT", N=N, kind=kind, **m))
    pd.DataFrame(et_rows).to_csv(OUT/"iter031_breadthN_EXT.csv", index=False)

    print(f"\nDone [{time.time()-t0:.0f}s]  artifacts in {OUT}", flush=True)


if __name__ == "__main__":
    main()
