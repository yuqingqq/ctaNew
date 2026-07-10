"""DIV2: crypto time-series-momentum (CTA) sleeve × v4 — WITHIN-CRYPTO era-diversification test.

Motivation (RESEARCH_LOOP addendum 23i, reviewer 445fcfb): 23h wrongly claimed within-crypto
diversification is unavailable because crypto ASSET corr spikes toward 1 in a crisis. That conflated
ASSET corr with STRATEGY corr. v4 is BETA-NEUTRAL; its 2022 failure is alpha/cost, not beta. A
DIRECTIONAL crypto trend/CTA sleeve is SHORT in a sustained decline (2022) → earns when v4 FAILS →
crisis-alpha, NOT corr→1. This tests that with the SAME DIV1 machinery (swap the xyz-v7 equity PnL
series for a canonical crypto-TSMOM series): overall corr, trend-mean-in-v4-bad-weeks, 2022-crisis
corr, matched-vol combined maxDD.

PINNED SPEC (binding pre-registration, reviewer 2d2de26 — NO sweep, W1b refuses post-hoc lookback
tuning). Chosen BEFORE seeing results, by literature default not by 2022 fit:
  - universe: 20 majors with full 2021-26 coverage (fixed).
  - signal: canonical time-series momentum = sign(trailing 12-MONTH / 365d return). This is the
    Moskowitz-Ooi-Pedersen (2012) academic default lookback — pinned for being THE literature
    standard, not for its crypto/2022 behavior.
  - sizing: inverse trailing-30d realized vol, gross-normalized to 1, PIT-shifted (signal+vol
    through t-1 applied to day t; no look-ahead — the DIV1 inverse-vol lesson).
  - cost: 4.5 bps one-way taker × |Δw| turnover (§8; trend turnover is NOT free).
  - rebalance: daily.
The 2022-crisis corr is SPEC-SENSITIVE (12m-trend is short through 2022's decline → expected to
diversify; a short lookback would whipsaw) — this is stated, not swept. Feasibility pass ONLY: a NEW
standalone strategy, own forward-validity unproven; measures ONLY whether it diversifies v4's bad
eras from within crypto. Inherits the DIV1 candidate ceiling PLUS crypto-TSMOM's weak forward
validity + flash-crash whipsaw risk (only 2022's sustained trend is in-sample).
"""
import sys, warnings, glob
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); rng = np.random.default_rng(23)
REPO = Path("/home/yuqing/ctaNew"); KD = REPO/"data/ml/test/parquet/klines"
VC = REPO/"live/state/convexity/div1_v4cyc"
MAJORS = ["BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","ADAUSDT","SOLUSDT","DOGEUSDT","AVAXUSDT",
          "LINKUSDT","LTCUSDT","BCHUSDT","ATOMUSDT","ETCUSDT","DOTUSDT","NEARUSDT","FILUSDT",
          "AAVEUSDT","UNIUSDT","TRXUSDT","XLMUSDT"]
COST_OW = 4.5  # one-way taker bps, charged on |Δw| turnover (§8 convention: 9 RT = 4.5 one-way)

def load_daily_closes():
    cols = {}
    for s in MAJORS:
        fs = sorted(glob.glob(str(KD/s/"5m"/"*.parquet")))
        if not fs: continue
        d = pd.concat([pd.read_parquet(f, columns=["open_time","close"]) for f in fs], ignore_index=True)
        d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
        d = d.drop_duplicates("open_time").sort_values("open_time").set_index("open_time")["close"]
        cols[s] = d.resample("1D").last()
    px = pd.DataFrame(cols).sort_index()
    return px

def tsmom_weekly(px, lookback=90, volwin=30):
    """Diversified TSMOM daily net-bps → weekly bps. PIT: position_t from data through t-1."""
    ret = px.pct_change()
    mom = px.pct_change(lookback)                    # trailing lookback-day return (through t)
    vol = ret.rolling(volwin, min_periods=volwin//2).std()
    raw = np.sign(mom) / vol.clip(lower=1e-4)         # inverse-vol, signed
    raw = raw.where(mom.notna() & vol.notna())
    w = raw.div(raw.abs().sum(axis=1), axis=0).fillna(0.0)   # gross-normalize to 1
    w = w.shift(1).fillna(0.0)                        # PIT: hold position formed at t-1 over day t
    pnl = (w * ret).sum(axis=1) * 1e4                 # daily gross pnl in bps of gross-1
    turn = w.diff().abs().sum(axis=1).fillna(0.0)
    net = pnl - COST_OW * turn                        # daily net bps
    net = net.iloc[max(lookback, volwin):]            # drop warmup
    s = pd.Series(net.values, index=net.index)
    wk = s.groupby(s.index.to_period("W").astype(str)).sum()
    return wk.rename("trend")

def v4_weekly_fullstack():
    parts = []
    for f in ("y2022.csv","oos.csv","recent.csv"):
        c = pd.read_csv(VC/f); c["open_time"] = pd.to_datetime(c["open_time"], utc=True)
        parts.append(c[["open_time","pnl_bps"]])
    c = pd.concat(parts, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    c["week"] = c["open_time"].dt.to_period("W").astype(str)
    return c.groupby("week")["pnl_bps"].sum().rename("v4")

def block_corr_ci(a, b, L=6, n=2000):
    m = len(a); nb = int(np.ceil(m/L)); cs = []
    for _ in range(n):
        starts = rng.integers(0, max(1, m-L+1), nb)
        idx = np.concatenate([np.arange(s, s+L) for s in starts])[:m] % m
        if np.std(a[idx]) > 0 and np.std(b[idx]) > 0: cs.append(np.corrcoef(a[idx], b[idx])[0,1])
    return (np.percentile(cs,[2.5,97.5]) if cs else (np.nan,np.nan))

def sh(x): return x.mean()/x.std(ddof=1)*np.sqrt(52) if x.std(ddof=1)>0 else np.nan
def mdd(x): e=np.cumsum(x); return float((e-np.maximum.accumulate(e)).min())

def analyze(v4, tr, lbl):
    m = pd.concat([v4, tr], axis=1).dropna().sort_index(); m["yr"] = m.index.str[:4]
    m23 = m[m.yr != "2022"]
    print(f"\n===== DIV2 [{lbl}] : crypto-TSMOM × v4 (within-crypto) =====")
    print(f"  standalone weekly Sharpe: v4 {sh(m23.v4):+.2f} | trend {sh(m23.trend):+.2f}   (2023-26, n={len(m23)})")
    print(f"  OVERALL corr(v4, trend) 2023-26 {m23.v4.corr(m23.trend):+.3f}")
    bad = m23[m23.v4 < 0]
    print(f"  ** trend mean in v4-BAD weeks (n={len(bad)}): {bad.trend.mean():+.1f} bps ** (>0 = pays when v4 down)")
    # trailing inverse-vol combined (PIT)
    sv = m23.v4.expanding(12).std().shift(1); st = m23.trend.expanding(12).std().shift(1)
    wv = (1/sv)/((1/sv)+(1/st)); comb = (wv*m23.v4 + (1-wv)*m23.trend).dropna()
    v4v = m23.v4.reindex(comb.index); cm = comb*(v4v.std()/comb.std())
    dd_v4, dd_cm = mdd(v4v.values), mdd(cm.values)
    print(f"  inv-vol combined Sharpe {sh(comb):+.2f} (v4 {sh(v4v):+.2f})")
    print(f"  ** MATCHED-VOL maxDD: v4 {dd_v4:+.0f} -> combined {dd_cm:+.0f} = {(1-dd_cm/dd_v4)*100:+.0f}% DD cut ** (combined vol {comb.std()/v4v.std():.2f}x v4)")
    # 2022 crisis (the decisive test)
    m22 = m[m.yr == "2022"]
    print(f"  --- 2022 CRISIS (n={len(m22)} wk) ---")
    if len(m22) >= 10:
        l2,h2 = block_corr_ci(m22.v4.values, m22.trend.values)
        print(f"  corr(v4_2022, trend_2022) {m22.v4.corr(m22.trend):+.3f} block-CI[{l2:+.2f},{h2:+.2f}]")
        print(f"  means: v4 {m22.v4.mean():+.0f}/wk (FAIL era) | trend {m22.trend.mean():+.0f}/wk  ({'DIVERSIFIES: trend pays when v4 fails' if m22.trend.mean()>0 and m22.v4.mean()<0 else 'does NOT diversify crisis'})")
        # matched-vol combined maxDD within 2022
        c22 = (0.5*m22.v4 + 0.5*m22.trend); c22m = c22*(m22.v4.std()/c22.std())
        print(f"  2022 matched-vol maxDD: v4 {mdd(m22.v4.values):+.0f} -> 50/50 combined {mdd(c22m.values):+.0f} ({(1-mdd(c22m.values)/mdd(m22.v4.values))*100 if mdd(m22.v4.values)<0 else 0:+.0f}%)")
    else: print(f"  INSUFFICIENT (n={len(m22)})")

def main():
    print("loading daily closes for 20 majors (5m->1D)...", flush=True)
    px = load_daily_closes()
    print(f"  panel {px.shape[0]} days {px.index.min().date()}..{px.index.max().date()}, {px.shape[1]} syms", flush=True)
    v4 = v4_weekly_fullstack()
    LOOKBACK = 365  # PINNED (MOP-2012 12-month canonical); binding pre-registration — NO sweep
    tr = tsmom_weekly(px, lookback=LOOKBACK)
    analyze(v4, tr, f"PINNED lookback={LOOKBACK}d (12m canonical)")
    print("\nCEILING: NEW standalone sleeve, own forward-validity unproven → feasibility signal, not confirmation.")
    print("DIV2DONE")

if __name__ == "__main__":
    main()
