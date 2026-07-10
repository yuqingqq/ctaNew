"""SWITCH1 — dynamic regime switcher trend<->v4 (SWITCH1_PREREG.md + addenda 23t/23u).

PINNED: BTC 30d Kaufman efficiency ratio (PIT) -> soft tilt w_trend = PIT percentile-rank of ER over
trailing 252d; both weekly streams z-normed to unit trailing-26w vol; switched = w_trend*trend_norm +
(1-w_trend)*v4_norm. Gates: S-1 block-shuffled-regime PLACEBO (decisive), S-2 beats static blend, S-3
DD-cut>0 vs v4 drop-one-year jackknife (read JOINTLY with S-1 per 23u — S-1 pass not surviving
drop-2026H1 = mirage).

CRITICAL PIT (review d28424f-#2): w_trend for week t is set from ER through the PRIOR week (weekly
shift(1)) applied to week-t return. The look-ahead (unshifted) variant is computed too and reported as
a gap, to PROVE the headline uses the shifted/PIT weight and to size any bar-weights-itself skill.
"""
import sys, warnings
sys.path.insert(0, "live")
import numpy as np, pandas as pd
import div2_crypto_trend as d
from div2_validate import tsmom_daily, to_weekly, sh_w, mdd
warnings.filterwarnings("ignore"); rng = np.random.default_rng(1)

def efficiency_ratio_pct(btc_close, win=30, pct_win=252):
    """Daily Kaufman ER then PIT percentile-rank over trailing pct_win days. All trailing (PIT thru t)."""
    net = (btc_close - btc_close.shift(win)).abs()
    path = btc_close.diff().abs().rolling(win).sum()
    er = (net/path).replace([np.inf,-np.inf], np.nan)
    # percentile rank of the CURRENT value within trailing window (uses only days <= t)
    erp = er.rolling(pct_win).apply(lambda a: (a <= a[-1]).mean(), raw=True)
    return erp

def weekly_last(daily, wkindex_name="week"):
    s = daily.dropna()
    return s.groupby(s.index.to_period("W").astype(str)).last()

def matched_dd_cut(v4s, xs):
    if len(xs) < 4 or xs.std()==0: return np.nan
    cm = xs*(v4s.std()/xs.std()); ddv, ddc = mdd(v4s.values), mdd(cm.values)
    return (1-ddc/ddv)*100 if ddv<0 else 0.0

def main():
    px = d.load_daily_closes()
    net_d,_ = tsmom_daily(px); trw = to_weekly(net_d); v4 = d.v4_weekly_fullstack()
    # regime signal (daily -> weekly end-of-week state)
    erp_d = efficiency_ratio_pct(px["BTCUSDT"])
    erp_w = weekly_last(erp_d).rename("erp")

    m = pd.concat([v4.rename("v4"), trw.rename("tr"), erp_w], axis=1).dropna().sort_index()
    m = m[m.index.str[:4] != "2022"]  # non-crisis OOS per pre-reg
    # PIT unit-vol normalization (trailing 26w, shift(1))
    m["trn"] = m.tr / m.tr.rolling(26).std().shift(1)
    m["v4n"] = m.v4 / m.v4.rolling(26).std().shift(1)
    # ---- CRITICAL PIT SHIFT: week-t tilt uses ER through week t-1 ----
    m["w_pit"]  = m.erp.shift(1)          # PIT (headline)
    m["w_look"] = m.erp                    # look-ahead (diagnostic only)
    m = m.dropna(subset=["trn","v4n","w_pit"])
    m["yr"] = m.index.str[:4]

    def switched(w): return w*m.trn + (1-w)*m.v4n
    sw   = switched(m.w_pit)
    swla = switched(m.w_look)
    static = 0.5*m.trn + 0.5*m.v4n
    v4b = m.v4n

    print("===== PIT VERIFICATION (headline = shifted/PIT) =====")
    print(f"  PIT switched Sharpe {sh_w(sw):+.3f}  |  LOOK-AHEAD (unshifted) Sharpe {sh_w(swla):+.3f}  gap {sh_w(swla)-sh_w(sw):+.3f}")
    print(f"  (headline uses w_pit = erp.shift(1); look-ahead shown only to size bar-weights-itself skill)")

    print("\n===== base numbers (2023-26 non-crisis, unit-vol streams) =====")
    print(f"  v4 Sh {sh_w(v4b):+.2f} | trend_norm Sh {sh_w(m.trn):+.2f} | static-blend Sh {sh_w(static):+.2f} | SWITCHED Sh {sh_w(sw):+.2f}")
    ddc_sw = matched_dd_cut(v4b, sw); ddc_st = matched_dd_cut(v4b, static)
    print(f"  matched-vol DD-cut vs v4: switched {ddc_sw:+.0f}% | static {ddc_st:+.0f}%")

    # ---------- GATE S-1: block-shuffled-regime placebo ----------
    print("\n===== GATE S-1: block-shuffled-regime PLACEBO (decisive) =====")
    wv = m.w_pit.values; n=len(wv); BLK=10; N=200
    def block_shuffle(a):
        nb=int(np.ceil(n/BLK)); blocks=[a[i*BLK:(i+1)*BLK] for i in range(nb)]
        order=rng.permutation(len(blocks)); out=np.concatenate([blocks[i] for i in order])[:n]
        return out
    plc=[]
    for _ in range(N):
        wsh=block_shuffle(wv); plc.append(sh_w(wsh*m.trn.values + (1-wsh)*m.v4n.values))
    plc=np.array(plc); p95=np.percentile(plc,95); rank=(sw.pipe(sh_w)>plc).mean()*100
    print(f"  real switched Sharpe {sh_w(sw):+.3f} | placebo mean {plc.mean():+.3f} p95 {p95:+.3f} max {plc.max():+.3f} | real rank p{rank:.0f}")
    s1 = sh_w(sw) > p95
    print(f"  >> GATE S-1: {'PASS' if s1 else 'FAIL'} (real > placebo p95)")

    # ---------- GATE S-2: beats static ----------
    s2 = (ddc_sw > ddc_st) and (sh_w(sw) >= sh_w(static))
    print(f"\n===== GATE S-2: beats static blend =====\n  DD-cut sw {ddc_sw:+.0f}% vs static {ddc_st:+.0f}% ; Sharpe sw {sh_w(sw):+.2f} vs static {sh_w(static):+.2f}  >> {'PASS' if s2 else 'FAIL'}")

    # ---------- GATE S-3: vs v4, drop-one-year jackknife ----------
    print("\n===== GATE S-3: DD-cut>0 vs v4, drop-one-year jackknife =====")
    full = matched_dd_cut(v4b, sw); print(f"  full DD-cut vs v4 {full:+.0f}%")
    jk_ok=True
    for yr in sorted(m.yr.unique()):
        idx=m.yr!=yr; p=matched_dd_cut(v4b[idx], sw[idx])
        drop26h1 = ""
        print(f"     drop {yr}: DD-cut {p:+.0f}%")
        if p<=0: jk_ok=False
    # explicit drop-2026H1 half (the mirage test)
    half = np.array([f"{t[:4]}H{'1' if int(t[5:7])<=26 else '2'}" for t in m.index]) if False else None
    dt = pd.to_datetime(pd.Index(m.index).str[:10], utc=True, errors="coerce")
    is26h1 = np.array([(t.year==2026 and t.month<=6) for t in dt])
    p_ex = matched_dd_cut(v4b[~is26h1], sw[~is26h1])
    print(f"     drop 2026H1 (mirage test): DD-cut {p_ex:+.0f}%")
    s3 = (full>0) and jk_ok and (p_ex>0)
    print(f"  >> GATE S-3: {'PASS' if s3 else 'FAIL'} (full>0 AND every drop-year>0 AND drop-2026H1>0)")

    print("\n========== SWITCH1 SUMMARY (read S-1 & S-3 JOINTLY per 23u) ==========")
    print(f"  S-1 placebo (decisive): {'PASS' if s1 else 'FAIL'}")
    print(f"  S-2 beats static:       {'PASS' if s2 else 'FAIL'}")
    print(f"  S-3 vs v4 concentration:{'PASS' if s3 else 'FAIL'} (drop-2026H1 {p_ex:+.0f}%)")
    joint = s1 and s3
    verdict = ("REAL: timed value is non-random AND survives drop-2026H1" if joint
               else ("MIRAGE: S-1 pass does NOT survive S-3 drop-2026H1" if (s1 and not s3)
                     else "DEAD: fails placebo (timing carries no info)"))
    print(f"  >>> JOINT VERDICT: {verdict}")
    print("SWITCH1DONE")

if __name__ == "__main__":
    main()
