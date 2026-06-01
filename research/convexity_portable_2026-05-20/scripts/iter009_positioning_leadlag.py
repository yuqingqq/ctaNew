"""iter-009 — DOES POSITIONING/LEVERAGE FRAGILITY LEAD THE SELLOFF? (the make-or-break free-data test)

Human core ask: find the selloffs EARLIER so we stop catching the falling knife. iters 1-8 found
every FREE observable (price, funding-as-feature, DVOL implied-vol, alt-direction flag) is
COINCIDENT/LAGGING. The mechanistic candidate this iteration: crowded-long POSITIONING (open-interest
buildup + extreme long/short ratios) should accumulate BEFORE the deleverage cascade.

NEW FREE DATA: data/ml/cache/metrics_<SYM>.parquet (Binance METRICS, 5-min, 2021-12->2026-05) for
the 23 EXT alts + BTC. Index=create_time. Cols: sum_open_interest, sum_open_interest_value,
count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio, count_long_short_ratio,
sum_taker_long_short_vol_ratio.

THE DECISIVE TEST (STEP 2): build market-wide POSITIONING-FRAGILITY features (PIT, cross-sym
aggregated, aligned to the EXT 4h grid, .shift(1) lagged), then ask:
  does positioning(t) predict the FUTURE book drawdown / forward alt-index move, and does it RISE
  BEFORE the selloff ONSET?  Leading signal => |IC_future| > |IC_past| (the OPPOSITE of DVOL which
  had IC_past 0.259 > IC_future 0.228). Run per-episode (luna/ftx/2024summer/2025q4).

We reuse the X123 EXT per-cycle panel (results/X123_altbear_short_EXT.parquet) which already carries
the production held-book per-cycle PnL (pnl_base), regime, fold, alt30, btc30, and alt_fwd_hold
(next-HOLD-bar forward alt-index cum log-ret). We only ADD the positioning features and run the
lead-lag IC + per-episode build-timing + a G4 pre-check.

Output: results/iter009_positioning_features_EXT.parquet (per-cycle features+pnl). Console: all tables.
Does NOT modify any prior script/cache.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

REPO = Path("/home/yuqing/ctaNew")
MDIR = REPO/"data/ml/cache"
RES = REPO/"research/convexity_portable_2026-05-20/results"
EXT_PANEL = RES/"X123_altbear_short_EXT.parquet"

WIN = 180          # 4h-grid bars ~ 30d trailing (matches alt30/btc30/beta windows in the engine)
HOLD = 6           # trade horizon in 4h bars (24h) — matches the engine
SEED = 12345
N_PLACEBO = 200

EXT_ALTS = ['AAVEUSDT','ADAUSDT','ATOMUSDT','AVAXUSDT','AXSUSDT','BCHUSDT','BNBUSDT','DOGEUSDT',
            'DOTUSDT','ETCUSDT','ETHUSDT','FILUSDT','HBARUSDT','ICPUSDT','LINKUSDT','LTCUSDT',
            'NEARUSDT','RUNEUSDT','SOLUSDT','TRBUSDT','UNIUSDT','XRPUSDT','ZECUSDT']

EPISODES = [
    ("2022_luna",   "2022-05-01", "2022-07-31"),
    ("2022_ftx",    "2022-11-01", "2023-01-31"),
    ("2024_summer", "2024-06-01", "2024-09-30"),
    ("2025_q4",     "2025-09-01", "2025-12-31"),
]


def load_metrics_4h(sym):
    """Return 4h-grid metrics frame for one symbol (raw values at decision-grid times)."""
    f = MDIR/f"metrics_{sym}.parquet"
    if not f.exists():
        return None
    m = pd.read_parquet(f)
    m = m[(m.index.hour % 4 == 0) & (m.index.minute == 0)].copy()
    m = m[~m.index.duplicated(keep="last")].sort_index()
    return m


def pit_pctile(s, win):
    """Expanding/rolling trailing percentile rank of the LAST value within trailing `win` window.
    PIT: rank of s[t] among s[t-win+1..t]; we then .shift(1) outside. No full-sample info."""
    return s.rolling(win, min_periods=win//3).apply(
        lambda x: (x[-1] >= x[:-1]).mean() if len(x) > 1 else np.nan, raw=True)


def build_positioning_features():
    print("=== building market-wide POSITIONING-FRAGILITY features (cross-sym, PIT, 4h) ===", flush=True)
    # Per-symbol 4h series of the raw metrics, then cross-sym aggregate.
    oi_val = {}          # sum_open_interest_value (USD notional) per sym
    cl_lsr = {}          # count_long_short_ratio (retail crowd, # accounts)
    tt_lsr = {}          # sum_toptrader_long_short_ratio (top-trader positions)
    cnt_tt = {}          # count_toptrader_long_short_ratio (top-trader accounts)
    taker = {}           # sum_taker_long_short_vol_ratio (taker buy/sell aggression)
    for sym in EXT_ALTS:
        m = load_metrics_4h(sym)
        if m is None:
            continue
        oi_val[sym] = m["sum_open_interest_value"].astype(float)
        cl_lsr[sym] = m["count_long_short_ratio"].astype(float)
        tt_lsr[sym] = m["sum_toptrader_long_short_ratio"].astype(float)
        cnt_tt[sym] = m["count_toptrader_long_short_ratio"].astype(float)
        taker[sym]  = m["sum_taker_long_short_vol_ratio"].astype(float)
    OI = pd.DataFrame(oi_val).sort_index()
    CL = pd.DataFrame(cl_lsr).sort_index()
    TT = pd.DataFrame(tt_lsr).sort_index()
    CTT = pd.DataFrame(cnt_tt).sort_index()
    TK = pd.DataFrame(taker).sort_index()
    print(f"  loaded {OI.shape[1]} syms, {OI.shape[0]} 4h rows, "
          f"{OI.index.min().date()}->{OI.index.max().date()}", flush=True)

    feats = pd.DataFrame(index=OI.index)

    # --- (1) Aggregate OPEN-INTEREST BUILDUP (leverage building) ---
    # per-sym OI growth vs its own trailing-30d mean (scale-free), then cross-sym mean.
    oi_growth = OI / OI.rolling(WIN, min_periods=WIN//3).mean() - 1.0
    feats["oi_buildup"] = oi_growth.mean(axis=1)
    # total notional OI growth (aggregate leverage), trailing-30d
    tot_oi = OI.sum(axis=1, min_count=1)
    feats["oi_total_growth"] = tot_oi / tot_oi.rolling(WIN, min_periods=WIN//3).mean() - 1.0

    # --- (2) CROWDED-LONG EXTREMITY (everyone long) ---
    # cross-sym mean of retail crowd long/short ratio; >1 = net long. Higher = more crowded long.
    feats["crowd_long"] = CL.mean(axis=1)
    # breadth: fraction of alts where retail crowd is net-long (>1)
    feats["crowd_long_breadth"] = (CL > 1.0).mean(axis=1)

    # --- (3) TOP-TRADER vs CROWD DIVERGENCE (smart money de-risking while crowd still long) ---
    # divergence = crowd LSR - toptrader LSR. Positive & rising = crowd more long than smart money.
    feats["smart_dumb_div"] = (CL.mean(axis=1) - TT.mean(axis=1))

    # --- (4) TAKER buy/sell AGGRESSION (market buying pressure; >1 = buyers aggressive) ---
    feats["taker_aggr"] = TK.mean(axis=1)

    # --- (5) COMPOSITE FRAGILITY: high OI buildup AND crowded long (both in trailing-PIT top) ---
    # PIT percentile of each ingredient (rank of current within trailing WIN), then product.
    oi_pct = pit_pctile(feats["oi_buildup"], WIN)
    crowd_pct = pit_pctile(feats["crowd_long"], WIN)
    feats["fragility_composite"] = oi_pct * crowd_pct          # high only when BOTH high
    feats["oi_buildup_pct"] = oi_pct
    feats["crowd_long_pct"] = crowd_pct

    # PIT lag: every feature must use only data through t-1 at decision time t.
    feats = feats.shift(1)
    return feats


def main():
    rng = np.random.default_rng(SEED)
    feats = build_positioning_features()

    # ---- load the EXT per-cycle held-book panel (production pnl_base etc.) ----
    p = pd.read_parquet(EXT_PANEL)
    p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
    p = p.sort_values("open_time").reset_index(drop=True)
    print(f"\nEXT panel: {len(p)} cycles {p['open_time'].min().date()}->{p['open_time'].max().date()}", flush=True)

    # as-of merge features onto cycle grid (features already .shift(1) lagged; exact 4h grid match)
    fcols = list(feats.columns)
    feats = feats.copy()
    feats.index = pd.to_datetime(feats.index, utc=True).as_unit("us")
    feats = feats.reset_index()
    feats.columns = ["create_time"] + fcols
    p["open_time"] = p["open_time"].dt.as_unit("us")
    p = pd.merge_asof(p, feats, left_on="open_time",
                      right_on="create_time", direction="backward",
                      tolerance=pd.Timedelta("4h"))

    # ---- build FORWARD and PAST book-PnL targets on the production book ----
    # forward book drawdown proxy: sum of next-HOLD pnl_base (the trade we are about to put on)
    p["pnl_fwd_hold"] = p["pnl_base"][::-1].rolling(HOLD, min_periods=1).sum()[::-1]
    # also a longer forward window (30d) cumulative book pnl = the drawdown we want to anticipate
    p["pnl_fwd_30d"] = p["pnl_base"][::-1].rolling(WIN, min_periods=WIN//3).sum()[::-1]
    # past book pnl (trailing 30d) — for the IC_future vs IC_past comparison
    p["pnl_past_30d"] = p["pnl_base"].rolling(WIN, min_periods=WIN//3).sum()
    # forward alt-index move is already present as alt_fwd_hold; build a 30d-forward alt move
    # (use alt30 shifted: alt30 is trailing-30d, so alt30 at t+WIN approximates fwd-30d move)
    p["alt_fwd_30d"] = p["alt30"].shift(-WIN)

    valid = p.dropna(subset=fcols, how="all").copy()
    print(f"cycles with positioning features: {valid[fcols[0]].notna().sum()} / {len(p)} "
          f"(features start {feats['create_time'][feats[fcols[0]].notna()].min().date() if feats[fcols[0]].notna().any() else 'NA'})", flush=True)

    # =====================================================================================
    # STEP 2A — THE LEAD-LAG TEST: IC of each positioning feature vs FUTURE vs PAST book PnL
    # A LEADING signal: high fragility => LOW future pnl (negative IC_future) AND |IC_fut|>|IC_past|.
    # =====================================================================================
    print("\n" + "="*100)
    print("STEP 2A — LEAD-LAG: Spearman IC of positioning(t) vs FUTURE vs PAST book PnL (full panel)")
    print("  leading => |IC_fut| > |IC_past|  (DVOL was the opposite: |past|0.259 > |fut|0.228)")
    print("  sign: fragility should predict LOWER future pnl => NEGATIVE IC_future is 'works'")
    print("="*100)
    print(f"{'feature':<22} {'IC_fwdHOLD':>11} {'IC_fwd30d':>11} {'IC_past30d':>11} "
          f"{'IC_altfwdH':>11} {'|fut|>|past|?':>13}")
    rows = []
    for f in fcols:
        sub = valid.dropna(subset=[f])
        def ic(col):
            s = sub.dropna(subset=[col])
            if len(s) < 50: return np.nan
            r, _ = spearmanr(s[f], s[col]); return r
        ic_fwdh = ic("pnl_fwd_hold"); ic_fwd30 = ic("pnl_fwd_30d")
        ic_past = ic("pnl_past_30d"); ic_altfh = ic("alt_fwd_hold")
        lead = (abs(ic_fwd30) > abs(ic_past)) if (np.isfinite(ic_fwd30) and np.isfinite(ic_past)) else False
        rows.append((f, ic_fwdh, ic_fwd30, ic_past, ic_altfh, lead))
        print(f"{f:<22} {ic_fwdh:>11.4f} {ic_fwd30:>11.4f} {ic_past:>11.4f} {ic_altfh:>11.4f} "
              f"{'YES-leads' if lead else 'no':>13}")

    # =====================================================================================
    # STEP 2B — PER-EPISODE: does fragility BUILD before the alt rollover ONSET?
    # For each episode: find the alt-index peak (rollover onset) within the window using pnl_base
    # cumulative book equity; measure fragility level/rank in the 10d BEFORE onset vs the rest.
    # =====================================================================================
    print("\n" + "="*100)
    print("STEP 2B — PER-EPISODE build-timing: is fragility ELEVATED in the run-up BEFORE the rollover?")
    print("  onset = peak of book equity within episode; pre = 30 cycles (~5d) before onset")
    print("  LEADS => fragility_composite pctile in PRE-window markedly > episode median (rises early)")
    print("="*100)
    KEYF = ["oi_buildup", "crowd_long", "smart_dumb_div", "fragility_composite", "taker_aggr"]
    print(f"{'episode':<13} {'onset':<12} " + " ".join(f"{f[:10]:>11}" for f in ['pre/all '+k[:6] for k in KEYF]))
    ep_lead = {}
    for nm, s, e in EPISODES:
        w = valid[(valid["open_time"] >= s) & (valid["open_time"] <= e)].copy()
        if len(w) < 40:
            print(f"{nm:<13} (insufficient cycles: {len(w)})"); continue
        eq = w["pnl_base"].cumsum()
        onset_i = eq.values.argmax()                     # peak = rollover onset
        onset_t = w["open_time"].iloc[onset_i]
        pre = w.iloc[max(0, onset_i-30):onset_i]          # ~5d before onset
        post = w.iloc[onset_i:]                            # the selloff
        cells = []
        lead_flags = []
        for f in KEYF:
            allmed = w[f].median()
            premed = pre[f].median() if len(pre) else np.nan
            # ratio of pre-onset level to whole-episode level (>1 => elevated in run-up)
            if np.isfinite(allmed) and abs(allmed) > 1e-9:
                ratio = premed/allmed
            else:
                ratio = np.nan
            cells.append(ratio)
            # "leads" if the feature was ALREADY elevated (top-third) before onset
            lead_flags.append(ratio)
        ep_lead[nm] = dict(zip(KEYF, cells))
        print(f"{nm:<13} {str(onset_t.date()):<12} " + " ".join(f"{c:>11.3f}" if np.isfinite(c) else f"{'NA':>11}" for c in cells))

    # also: per-episode IC_future (fragility vs forward book pnl) within each episode
    print("\n  per-episode IC(fragility_composite, pnl_fwd_hold) [neg=fragility predicts loss]:")
    for nm, s, e in EPISODES:
        w = valid[(valid["open_time"] >= s) & (valid["open_time"] <= e)].dropna(subset=["fragility_composite", "pnl_fwd_hold"])
        if len(w) < 40:
            print(f"    {nm:<13} insufficient"); continue
        r_fut, _ = spearmanr(w["fragility_composite"], w["pnl_fwd_hold"])
        r_past = np.nan
        wp = w.dropna(subset=["pnl_past_30d"])
        if len(wp) >= 40:
            r_past, _ = spearmanr(wp["fragility_composite"], wp["pnl_past_30d"])
        print(f"    {nm:<13} IC_fut={r_fut:+.4f}  IC_past={r_past:+.4f}  "
              f"{'LEADS' if (np.isfinite(r_fut) and np.isfinite(r_past) and abs(r_fut)>abs(r_past)) else 'lags/coincident'}")

    # =====================================================================================
    # STEP 2C — SIDE-REGIME focus: the loss is in side. Does fragility separate the losing
    # side cycles FORWARD? (mean forward book pnl on high-fragility vs low-fragility side cycles)
    # =====================================================================================
    print("\n" + "="*100)
    print("STEP 2C — SIDE-regime forward separation (the DD lives in side):")
    print("  split side cycles by fragility_composite >= its trailing-PIT median; compare FWD book pnl")
    print("="*100)
    side = valid[valid["is_side"]].dropna(subset=["fragility_composite", "pnl_fwd_hold"]).copy()
    if len(side) > 100:
        thr = side["fragility_composite"].median()
        hi = side[side["fragility_composite"] >= thr]; lo = side[side["fragility_composite"] < thr]
        print(f"  side cycles {len(side)}: HI-fragility fwd pnl mean {hi['pnl_fwd_hold'].mean()*1e4:+.2f} bps "
              f"(n={len(hi)}) vs LO {lo['pnl_fwd_hold'].mean()*1e4:+.2f} bps (n={len(lo)})  "
              f"sep={ (lo['pnl_fwd_hold'].mean()-hi['pnl_fwd_hold'].mean())*1e4:+.2f} bps "
              f"(positive sep = HI-fragility => worse fwd = LEADS)")
        # decile monotonicity
        side["dec"] = pd.qcut(side["fragility_composite"], 10, labels=False, duplicates="drop")
        dec = side.groupby("dec")["pnl_fwd_hold"].mean()*1e4
        print("  fwd-pnl by fragility decile (bps):", " ".join(f"{v:+.1f}" for v in dec.values))

    # =====================================================================================
    # STEP 3 — G4 PRE-CHECK (mandated): a fragility de-risk gate vs matched-random-timing.
    # Candidate gate: FLAT side cycles whose fragility_composite is in the trailing-PIT top tercile.
    # Compare real (signal-aligned) FLAT to FLAT-ing the SAME COUNT of RANDOM side cycles.
    # =====================================================================================
    print("\n" + "="*100)
    print("STEP 3 — G4 PRE-CHECK: fragility-FLAT-side gate vs matched-random-timing (200 seeds)")
    print("="*100)
    pf = valid.copy().reset_index(drop=True)
    pf["pnl_base"] = pf["pnl_base"].fillna(0.0)

    def calmar(arr):
        pb = pd.Series(arr).dropna()*1e4
        if len(pb) < 3: return np.nan
        eq = pb.cumsum(); mdd = (eq-eq.cummax()).min()
        return (pb.mean()*6*365/abs(mdd)) if (mdd < 0 and np.isfinite(mdd)) else np.nan

    base_cal = calmar(pf["pnl_base"].values)
    # real gate: flat flagged side cycles where fragility in top tercile (PIT)
    fc = pf["fragility_composite"]
    side_mask = pf["is_side"].values & fc.notna().values
    thr_top = pf.loc[pf["is_side"], "fragility_composite"].quantile(2/3)
    real_flat = (pf["is_side"].values & (fc.values >= thr_top))
    n_flat = int(np.nansum(real_flat))
    real_pnl = pf["pnl_base"].values.copy(); real_pnl[real_flat] = 0.0
    real_cal = calmar(real_pnl)
    real_mdd = (pd.Series(real_pnl*1e4).cumsum().cummax() - pd.Series(real_pnl*1e4).cumsum()).max()

    # matched-random: flat n_flat random SIDE cycles
    side_idx = np.where(pf["is_side"].values)[0]
    pl_cal = []
    for _ in range(N_PLACEBO):
        pick = rng.choice(side_idx, size=min(n_flat, len(side_idx)), replace=False)
        a = pf["pnl_base"].values.copy(); a[pick] = 0.0
        pl_cal.append(calmar(a))
    pl_cal = np.array([x for x in pl_cal if np.isfinite(x)])
    rank = (real_cal > pl_cal).mean()*100 if len(pl_cal) else np.nan
    print(f"  base Calmar {base_cal:+.3f}; fragility-FLAT (top-tercile side, n_flat={n_flat}) "
          f"Calmar {real_cal:+.3f}")
    print(f"  matched-random-FLAT placebo: p50={np.percentile(pl_cal,50):+.3f} p95={np.percentile(pl_cal,95):+.3f} "
          f"max={pl_cal.max():+.3f}; REAL ranks p{rank:.0f}  "
          f"{'PASS >=p95' if rank>=95 else 'FAIL <p95 (effect is run-smaller, not skill)'}")

    pf.to_parquet(RES/"iter009_positioning_features_EXT.parquet")
    print(f"\nwrote {RES/'iter009_positioning_features_EXT.parquet'}")


if __name__ == "__main__":
    main()
