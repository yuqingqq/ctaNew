"""iter-010 — DO FASTER selloff-onset metrics LEAD the alt-bear AND show forward DOWN-CONTINUATION?

Human idea: the prior alt-bear detector used alt_index_30d (SLOW 30d trailing return) that only flags
the selloff LATE and (iter-008) was a COINCIDENT BOTTOM-detector — flagged cycles BOUNCED (median fwd
+0.0036). The genuinely NEW angle: a FAST-reacting onset metric (a) detects the rollover days EARLIER
and (b) — crucially — captures DOWN-MOMENTUM / crash-continuation: a sharp drop CONTINUES short-term
(liquidation cascades + vol clustering), whereas the slow 30d grind mean-reverts. If true a fast
crash-onset flag could power a de-risk gate the slow alt30 could not.

We reuse the X123 EXT held-book per-cycle panel (results/X123_altbear_short_EXT.parquet: pnl_base,
regime, fold, is_side, alt30, btc30, alt_fwd_hold) and rebuild the SAME PIT eq-weight alt-index from
klines (verbatim X122/X123 construction) so we can derive FASTER metrics on it. Every metric is PIT:
trailing window built from data through t, then .shift(1) so the decision at cycle t uses only t-1 data.

THE DECISIVE TESTS:
 (1) LEAD TIME: per episode, how many days earlier does each fast metric cross its threshold vs
     alt30<btc30?
 (2) FORWARD CONTINUATION (make-or-break): conditional on each fast flag firing, does the alt-index /
     book PnL CONTINUE DOWN over the next trade horizon (HOLD=6 bars/24h) and next few days, or BOUNCE?
     Forward distribution (mean/median/%neg). Contrast with alt30 flag (which BOUNCED, median +0.0036).
 (3) IC future vs IC past at the trade horizon (a real lead has |IC_fut| > |IC_past|).
 (4) G4 PRE-CHECK on the best candidate: fast-flag FLAT-side gate vs matched-random-timing (>=p95?) +
     multi-episode LOFO.

Output: results/iter010_fast_metrics_EXT.parquet. Console: all tables. Modifies nothing prior.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

REPO = Path("/home/yuqing/ctaNew")
RC = REPO/"research/convexity_portable_2026-05-20/results/_cache"
RES = REPO/"research/convexity_portable_2026-05-20/results"
KLINES = REPO/"data/ml/test/parquet/klines"
EXT_PANEL = RES/"X123_altbear_short_EXT.parquet"
EXT_PREDS = RC/"x113_ext_v0_preds.parquet"

WIN = 180          # 4h-grid bars ~ 30d trailing (the SLOW alt30 window — our reference)
HOLD = 6           # trade horizon in 4h bars (24h) — what a de-risk gate acts on
SEED = 12345
N_PLACEBO = 300

EPISODES = [
    ("2022_luna",   "2022-05-01", "2022-07-31"),
    ("2022_ftx",    "2022-11-01", "2023-01-31"),
    ("2024_summer", "2024-06-01", "2024-09-30"),
    ("2025_q4",     "2025-09-01", "2025-12-31"),
]


def load_close(sym):
    sd = KLINES/sym/"5m"
    if not sd.exists():
        return None
    dfs = [pd.read_parquet(f, columns=["open_time", "close"]) for f in sorted(sd.glob("*.parquet"))]
    df = pd.concat(dfs, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df.set_index("open_time")["close"].astype(np.float64)


def calmar_of(pnl_arr):
    pb = pd.Series(pnl_arr).dropna()*1e4
    if len(pb) < 3:
        return np.nan
    eq = pb.cumsum(); mdd = (eq - eq.cummax()).min()
    return (pb.mean()*6*365/abs(mdd)) if (mdd < 0 and np.isfinite(mdd)) else np.nan


def maxdd_of(pnl_arr):
    pb = pd.Series(pnl_arr).dropna()*1e4
    if len(pb) < 1:
        return np.nan
    eq = pb.cumsum(); return (eq - eq.cummax()).min()


def build_fast_metrics():
    """Rebuild PIT eq-weight alt-index from klines (X122/X123 verbatim), then derive FAST metrics.
    Returns a DataFrame indexed by 4h open_time, each column .shift(1)-lagged (PIT)."""
    print("=== rebuilding PIT eq-weight alt-index + FAST onset metrics ===", flush=True)
    d = pd.read_parquet(EXT_PREDS, columns=["symbol"])
    syms = sorted(d["symbol"].unique())
    ret_map = {}
    px_map = {}
    for sym in syms:
        c = load_close(sym)
        if c is None:
            continue
        c4 = c[(c.index.hour % 4 == 0) & (c.index.minute == 0)]
        ret_map[sym] = np.log(c4/c4.shift(1))
        px_map[sym] = c4
    ret4 = pd.concat([s.rename(k) for k, s in ret_map.items()], axis=1).sort_index()
    px4 = pd.concat([s.rename(k) for k, s in px_map.items()], axis=1).sort_index()
    altcols = [c for c in ret4.columns if c not in ("BTCUSDT", "ETHUSDT")]
    altidx = ret4[altcols].mean(axis=1)            # eq-weight per-bar log-return of alt complex
    alt_cum = altidx.cumsum()                       # cum log-ret level (the alt-index)
    print(f"  alt-index from {len(altcols)} alts, {len(alt_cum)} 4h bars "
          f"{alt_cum.index.min().date()}->{alt_cum.index.max().date()}", flush=True)

    f = pd.DataFrame(index=alt_cum.index)

    # (the SLOW alt30 reference already lives in the EXT panel; we don't duplicate it here)

    # ---------- (A) SHORT-WINDOW alt-index returns (fast momentum) ----------
    B1, B3, B7 = 6, 18, 42                                          # 1d, 3d, 7d in 4h bars
    f["alt_1d"] = (alt_cum - alt_cum.shift(B1))
    f["alt_3d"] = (alt_cum - alt_cum.shift(B3))
    f["alt_7d"] = (alt_cum - alt_cum.shift(B7))

    # ---------- (B) DRAWDOWN-FROM-RECENT-HIGH (rolls the instant a top rolls over) ----------
    # alt-index level minus its trailing-N-bar running max (<=0; more negative = deeper fresh DD)
    f["alt_dd10"] = alt_cum - alt_cum.rolling(60, min_periods=10).max()   # vs trailing 10d high
    f["alt_dd20"] = alt_cum - alt_cum.rolling(120, min_periods=20).max()  # vs trailing 20d high

    # ---------- (C) REALIZED-VOL SPIKE (short rvol vs trailing baseline) ----------
    rvol_s = altidx.rolling(B3, min_periods=6).std()               # 3d realized vol
    rvol_base = altidx.rolling(WIN, min_periods=42).std()          # 30d baseline
    f["alt_rvol_spike"] = rvol_s / rvol_base                       # >1 = vol spiking up

    # ---------- (D) BREADTH (fraction of alts in a fast downtrend) ----------
    px_ma = px4.rolling(B7, min_periods=6).mean()                  # 7d MA per alt
    below_ma = (px4[altcols] < px_ma[altcols])
    f["breadth_below_ma7"] = below_ma.mean(axis=1)                 # fraction below 7d MA
    neg7 = ((px4[altcols] / px4[altcols].shift(B7) - 1.0) < 0)
    f["breadth_neg_7d"] = neg7.mean(axis=1)                        # fraction with neg 7d return

    # ---------- (E) RETURN ACCELERATION (2nd difference of the fast move) ----------
    # alt_3d minus the prior (lagged) alt_3d => is the drop ACCELERATING? (more neg = accelerating down)
    f["alt_accel_3d"] = f["alt_3d"] - f["alt_3d"].shift(B3)

    # PIT lag: decision at t uses only data through t-1
    f = f.shift(1)
    return f, alt_cum


# fast metrics and the direction in which "selloff onset" is flagged.
# down_flag = the metric crossing into the bearish region (the gate would fire here).
FAST = {
    "alt_1d":          ("lt", 0.0),     # alt 1d return < 0
    "alt_3d":          ("lt", 0.0),
    "alt_7d":          ("lt", 0.0),
    "alt_dd10":        ("lt", -0.05),   # >5% below trailing-10d high
    "alt_dd20":        ("lt", -0.05),
    "alt_rvol_spike":  ("gt", 1.25),    # short rvol 25% above baseline
    "breadth_below_ma7": ("gt", 0.6),   # >60% of alts below 7d MA
    "breadth_neg_7d":  ("gt", 0.6),     # >60% of alts negative over 7d
    "alt_accel_3d":    ("lt", 0.0),     # drop accelerating
}


def flag_series(s, mode, thr):
    if mode == "lt":
        return s < thr
    return s > thr


def fwd_dist(arr):
    a = pd.Series(arr).dropna().values
    if len(a) == 0:
        return (np.nan, np.nan, np.nan, 0)
    return (np.mean(a), np.median(a), (a < 0).mean()*100, len(a))


def main():
    rng = np.random.default_rng(SEED)
    feats, alt_cum = build_fast_metrics()

    p = pd.read_parquet(EXT_PANEL)
    p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
    p = p.sort_values("open_time").reset_index(drop=True)
    print(f"\nEXT panel: {len(p)} cycles {p['open_time'].min().date()}->{p['open_time'].max().date()}",
          flush=True)

    # merge fast features onto the cycle grid (exact 4h grid match; features already .shift(1))
    fcols = list(feats.columns)
    feats = feats.copy()
    feats.index = pd.to_datetime(feats.index, utc=True).as_unit("us")
    feats = feats.reset_index()
    feats.columns = ["create_time"] + fcols
    p["open_time"] = p["open_time"].dt.as_unit("us")
    p = pd.merge_asof(p, feats, left_on="open_time", right_on="create_time",
                      direction="backward", tolerance=pd.Timedelta("4h"))

    # forward / past targets on the production held book
    p["pnl_fwd_hold"] = p["pnl_base"][::-1].rolling(HOLD, min_periods=1).sum()[::-1]
    p["pnl_fwd_3d"]   = p["pnl_base"][::-1].rolling(18, min_periods=6).sum()[::-1]
    p["pnl_past_hold"]= p["pnl_base"].rolling(HOLD, min_periods=1).sum()
    p["pnl_past_3d"]  = p["pnl_base"].rolling(18, min_periods=6).sum()
    # forward alt-index move at trade horizon (already present) + a 3d forward alt move
    ac = alt_cum.copy(); ac.index = pd.to_datetime(ac.index, utc=True).as_unit("us")
    fwd3 = (ac.shift(-18) - ac)
    p["alt_fwd_3d"] = p["open_time"].map(fwd3.to_dict())
    # alt30-flag forward target reference is alt_fwd_hold (next 24h alt move)

    valid = p.copy()

    # =====================================================================================
    # STEP 3.(1) — LEAD TIME: for each big episode, how many days EARLIER does each fast metric
    # cross into its bearish region vs the SLOW alt30<btc30 flag (the iter-007/008 reference)?
    # =====================================================================================
    print("\n" + "="*112)
    print("STEP (1) LEAD TIME — first bearish crossing of each fast metric vs first alt30<btc30, per episode")
    print("  positive lead_days = fast metric fires EARLIER than slow alt30<btc30 (faster onset detection)")
    print("="*112)
    # slow flag = alt30 < btc30 (the iter-007 form), using panel's alt30/btc30
    valid["slow_flag"] = (valid["alt30"] < valid["btc30"])
    hdr = f"{'episode':<13}{'slow_first':>13}" + "".join(f"{k[:10]:>12}" for k in FAST)
    print(hdr)
    lead_rows = {}
    for nm, s, e in EPISODES:
        w = valid[(valid["open_time"] >= s) & (valid["open_time"] <= e)].sort_values("open_time")
        if len(w) < 40:
            print(f"{nm:<13} (insufficient {len(w)})"); continue
        sf = w[w["slow_flag"]]
        slow_t = sf["open_time"].iloc[0] if len(sf) else None
        cells = []
        lead_rows[nm] = {}
        for k, (mode, thr) in FAST.items():
            fl = w[flag_series(w[k], mode, thr).fillna(False)]
            ft = fl["open_time"].iloc[0] if len(fl) else None
            if slow_t is not None and ft is not None:
                lead_d = (slow_t - ft).total_seconds()/86400.0   # +ve = fast fires before slow
            else:
                lead_d = np.nan
            cells.append(lead_d); lead_rows[nm][k] = lead_d
        st = str(slow_t.date()) if slow_t is not None else "never"
        print(f"{nm:<13}{st:>13}" + "".join(
            (f"{c:>+12.1f}" if np.isfinite(c) else f"{'NA':>12}") for c in cells))

    # =====================================================================================
    # STEP 3.(2) — FORWARD CONTINUATION (make-or-break): conditional on each fast flag firing,
    # does the alt-index / book PnL CONTINUE DOWN over the next trade horizon (24h) and next 3d,
    # or BOUNCE? Contrast with the SLOW alt30 flag (iter-008: median fwd alt +0.0036 = bounce).
    # =====================================================================================
    print("\n" + "="*112)
    print("STEP (2) FORWARD CONTINUATION — conditional on flag firing: forward ALT-index move @24h (HOLD)")
    print("  CONTINUES DOWN if median<0 AND %neg>>50.  BOUNCE if median>=0 / %neg<=50 (the slow-alt30 failure).")
    print("  ref: SLOW alt30<btc30 flag (iter-008) bounced: median +0.0036, %neg 47-51")
    print("="*112)
    print(f"{'flag':<20}{'n':>7}{'fwdAlt_mean':>13}{'fwdAlt_med':>12}{'fwdAlt%neg':>11}"
          f"{'fwdPnL_med':>12}{'fwdPnL%neg':>11}  read")
    # SLOW reference first
    for label, mask in [("SLOW alt30<btc30", valid["slow_flag"].fillna(False).values)]:
        m, md, pn, n = fwd_dist(valid.loc[mask, "alt_fwd_hold"].values)
        pm_m, pm_md, pm_pn, _ = fwd_dist(valid.loc[mask, "pnl_fwd_hold"].values)
        read = "BOUNCE" if (not np.isfinite(md) or md >= 0 or pn <= 52) else "down-cont?"
        print(f"{label:<20}{n:>7}{m:>+13.4f}{md:>+12.4f}{pn:>11.1f}"
              f"{pm_md*1e4:>+12.2f}{pm_pn:>11.1f}  {read}")
    for k, (mode, thr) in FAST.items():
        mask = flag_series(valid[k], mode, thr).fillna(False).values
        m, md, pn, n = fwd_dist(valid.loc[mask, "alt_fwd_hold"].values)
        pm_m, pm_md, pm_pn, _ = fwd_dist(valid.loc[mask, "pnl_fwd_hold"].values)
        cont = (np.isfinite(md) and md < 0 and pn > 52)
        read = "DOWN-CONT" if cont else "bounce/flat"
        print(f"{k:<20}{n:>7}{m:>+13.4f}{md:>+12.4f}{pn:>11.1f}"
              f"{pm_md*1e4:>+12.2f}{pm_pn:>11.1f}  {read}")

    # the 3-day forward continuation (does the down-move persist beyond the single trade horizon?)
    print("\n  --- forward continuation @3d (next 18 bars) — does it persist beyond 24h? ---")
    print(f"{'flag':<20}{'n':>7}{'fwdAlt3d_med':>14}{'fwdAlt3d%neg':>13}")
    md, pn, _ = fwd_dist(valid.loc[valid['slow_flag'].fillna(False).values, 'alt_fwd_3d'].values)[1:]
    print(f"{'SLOW alt30<btc30':<20}{int(valid['slow_flag'].fillna(False).sum()):>7}{md:>+14.4f}{pn:>13.1f}")
    for k, (mode, thr) in FAST.items():
        mask = flag_series(valid[k], mode, thr).fillna(False).values
        _, md, pn, n = fwd_dist(valid.loc[mask, "alt_fwd_3d"].values)
        print(f"{k:<20}{n:>7}{md:>+14.4f}{pn:>13.1f}")

    # PER-EPISODE forward continuation for the strongest fast candidates (the multi-episode test)
    print("\n" + "="*112)
    print("STEP (2b) PER-EPISODE forward continuation @24h — does any fast flag CONTINUE DOWN across >=2 episodes?")
    print("  cell = median fwd ALT move @24h on flagged cycles (neg=down-continuation). (%neg in parens)")
    print("="*112)
    cand = ["alt_1d", "alt_3d", "alt_dd10", "alt_dd20", "alt_rvol_spike",
            "breadth_below_ma7", "alt_accel_3d", "slow_flag"]
    print(f"{'episode':<13}" + "".join(f"{c[:13]:>17}" for c in cand))
    for nm, s, e in EPISODES:
        w = valid[(valid["open_time"] >= s) & (valid["open_time"] <= e)]
        if len(w) < 40:
            print(f"{nm:<13}(insufficient)"); continue
        cells = []
        for c in cand:
            if c == "slow_flag":
                mask = w["slow_flag"].fillna(False).values
            else:
                mode, thr = FAST[c]; mask = flag_series(w[c], mode, thr).fillna(False).values
            _, md, pn, n = fwd_dist(w.loc[mask, "alt_fwd_hold"].values)
            cells.append(f"{md:+.4f}({pn:.0f}%)" if n > 5 else f"n={n}")
        print(f"{nm:<13}" + "".join(f"{c:>17}" for c in cells))

    # =====================================================================================
    # STEP 3.(3) — IC FUTURE vs IC PAST at the trade horizon (a real lead has |IC_fut| > |IC_past|)
    # =====================================================================================
    print("\n" + "="*112)
    print("STEP (3) IC future vs past @ TRADE HORIZON (HOLD=24h). LEAD => |IC_fut|>|IC_past|.")
    print("  metric(t) vs fwd-24h book PnL and fwd-24h alt move; vs past-24h (the coincident/lag check)")
    print("="*112)
    print(f"{'feature':<20}{'IC_fwd_pnl':>12}{'IC_past_pnl':>13}{'IC_fwd_alt':>12}{'IC_past_alt':>13}"
          f"{'|fut|>|past|':>13}")
    for k in list(FAST.keys()):
        sub = valid.dropna(subset=[k])
        def ic(col):
            ss = sub.dropna(subset=[col])
            if len(ss) < 80:
                return np.nan
            r, _ = spearmanr(ss[k], ss[col]); return r
        icfp = ic("pnl_fwd_hold"); icpp = ic("pnl_past_hold")
        icfa = ic("alt_fwd_hold")
        # past alt move at trade horizon = alt over prior HOLD bars
        icpa = np.nan
        s2 = sub.dropna(subset=["alt_1d"])  # alt_1d IS the trailing-24h alt move (proxy for past alt)
        if len(s2) >= 80:
            r, _ = spearmanr(s2[k], s2["alt_1d"]); icpa = r
        lead = (abs(icfa) > abs(icpa)) if (np.isfinite(icfa) and np.isfinite(icpa)) else False
        print(f"{k:<20}{icfp:>+12.4f}{icpp:>+13.4f}{icfa:>+12.4f}{icpa:>+13.4f}"
              f"{('YES' if lead else 'no'):>13}")

    # =====================================================================================
    # STEP 3.(4) — G4 PRE-CHECK on the best fast candidate: FLAT-side gate vs matched-random-timing.
    # We pick the fast metric with the strongest forward DOWN-continuation at the trade horizon
    # (lowest median fwd-alt with %neg>52) and test whether FLATting those side cycles beats a
    # matched-random FLAT of the same count. Also report multi-episode LOFO.
    # =====================================================================================
    print("\n" + "="*112)
    print("STEP (4) G4 PRE-CHECK — fast-flag FLAT-side gate vs matched-random-timing (300 seeds) + LOFO")
    print("="*112)
    pf = valid.copy().reset_index(drop=True)
    pf["pnl_base"] = pf["pnl_base"].fillna(0.0)
    base_cal = calmar_of(pf["pnl_base"].values)
    base_mdd = maxdd_of(pf["pnl_base"].values)
    side_idx = np.where(pf["is_side"].values)[0]

    # rank fast candidates by side-cycle forward DOWN-continuation strength (lower fwd-alt median = better)
    rankcand = []
    for k, (mode, thr) in FAST.items():
        mask = pf["is_side"].values & flag_series(pf[k], mode, thr).fillna(False).values
        _, md, pn, n = fwd_dist(pf.loc[mask, "alt_fwd_hold"].values)
        if n >= 30:
            rankcand.append((k, md, pn, n))
    rankcand.sort(key=lambda x: x[1])   # most-negative forward alt move first
    print("  side-flag forward-continuation ranking (most negative fwd alt @24h first):")
    for k, md, pn, n in rankcand:
        print(f"    {k:<20} fwd_alt_med={md:+.4f}  %neg={pn:.0f}  n={n}")

    for k, md, pn, n in rankcand[:3]:
        mode, thr = FAST[k]
        real_flat = pf["is_side"].values & flag_series(pf[k], mode, thr).fillna(False).values
        n_flat = int(real_flat.sum())
        rp = pf["pnl_base"].values.copy(); rp[real_flat] = 0.0
        real_cal = calmar_of(rp); real_mdd = maxdd_of(rp)
        pl = []
        for _ in range(N_PLACEBO):
            pick = rng.choice(side_idx, size=min(n_flat, len(side_idx)), replace=False)
            a = pf["pnl_base"].values.copy(); a[pick] = 0.0
            pl.append(calmar_of(a))
        pl = np.array([x for x in pl if np.isfinite(x)])
        rank = (real_cal > pl).mean()*100 if len(pl) else np.nan
        print(f"\n  [{k}] base Calmar {base_cal:+.3f} (mdd {base_mdd:+.0f}) -> FLAT-side Calmar "
              f"{real_cal:+.3f} (mdd {real_mdd:+.0f}), n_flat={n_flat}")
        print(f"     matched-random placebo: p50={np.percentile(pl,50):+.3f} p95={np.percentile(pl,95):+.3f} "
              f"max={pl.max():+.3f}  REAL ranks p{rank:.0f}  "
              f"{'PASS >=p95' if rank>=95 else 'FAIL <p95 (run-smaller, not skill)'}")
        # multi-episode LOFO: does the gate lift survive dropping each episode?
        lifts = []
        for nm, s, e in EPISODES:
            keep = ~((pf["open_time"] >= s) & (pf["open_time"] <= e)).values
            bcal = calmar_of(pf["pnl_base"].values[keep])
            gcal = calmar_of(rp[keep])
            lifts.append((nm, gcal - bcal))
        print("     episode-LOFO lift (gate-base Calmar, dropping each episode):",
              "  ".join(f"{nm}:{lf:+.2f}" for nm, lf in lifts))

    out = RES/"iter010_fast_metrics_EXT.parquet"
    pf.to_parquet(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
