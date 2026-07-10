"""DIV2-HE hedge-economics of the crypto-trend OVERLAY (DIV2_BUILD_PREREG + addenda 23q/23r).

Tests whether crypto-trend is a deploy-worthy DIVERSIFYING OVERLAY on v4 (distinct from GATE 1a's
standalone question, which failed). Corrected gates per review 6d22ce9:
  HE-1 (PRIMARY = DD only): matched-vol DD-cut>0 EX-2026H1 (mirage = benefit is 2026H1-driven; test on
       DD not Sharpe — ex-2026H1=2025 is the trend's bleed window). Sharpe>=v4 SECONDARY. + full
       drop-one-half JACKKNIFE of DD-cut.
  HE-2 (breadth): per-year DD-cut>0 in >=2/3 of {2023,2024,2025} AND aggregate combined Sharpe>=v4.
       Carry NOT double-counted (trend already net of turnover). 2025 trend loss = carry paid.
Same pinned 365/30 TSMOM + PIT inverse-vol combination (binding, no change).
"""
import sys, warnings
sys.path.insert(0, "live")
import numpy as np, pandas as pd
import div2_crypto_trend as d
from div2_validate import tsmom_daily, to_weekly, sh_w, mdd
warnings.filterwarnings("ignore")

def half_of(idx):
    dt = pd.to_datetime(pd.Index(idx).str[:10], utc=True, errors="coerce")
    return np.array([f"{t.year}H{'1' if t.month<=6 else '2'}" for t in dt])

def matched_dd_cut(v4s, combs):
    """matched-vol DD cut: lever combined to v4 vol, compare maxDD. returns (ddv, ddc, pct)."""
    if len(combs) < 4 or combs.std() == 0: return (np.nan, np.nan, np.nan)
    cm = combs * (v4s.std()/combs.std())
    ddv, ddc = mdd(v4s.values), mdd(cm.values)
    return ddv, ddc, ((1-ddc/ddv)*100 if ddv < 0 else 0.0)

def main():
    px = d.load_daily_closes()
    net_d, _ = tsmom_daily(px)
    trw = to_weekly(net_d); v4 = d.v4_weekly_fullstack()
    m = pd.concat([v4.rename("v4"), trw.rename("tr")], axis=1).dropna().sort_index()
    m = m[m.index.str[:4] != "2022"]   # non-crisis OOS
    m["yr"] = m.index.str[:4]; m["half"] = half_of(m.index)
    # PIT inverse-vol combination over the FULL non-crisis series (weights don't reset on subsets)
    sv = m.v4.expanding(12).std().shift(1); st = m.tr.expanding(12).std().shift(1)
    wv = (1/sv)/((1/sv)+(1/st)); m["comb"] = (wv*m.v4 + (1-wv)*m.tr)
    m = m.dropna(subset=["comb"])

    # ---------- HE-1: concentration kill (2025-26 confirm; DD-primary) ----------
    print("===== HE-1: concentration kill (2025-26 OOS confirm; DD PRIMARY) =====")
    conf = m[m.yr >= "2025"]
    ddv,ddc,pct = matched_dd_cut(conf.v4, conf.comb)
    print(f"  FULL 2025-26 (n={len(conf)}): matched-vol DD v4 {ddv:+.0f}->comb {ddc:+.0f} = {pct:+.0f}% | comb Sh {sh_w(conf.comb):+.2f} v4 Sh {sh_w(conf.v4):+.2f}")
    ex = conf[conf.half != "2026H1"]   # drop the strong trend half
    ddv2,ddc2,pct2 = matched_dd_cut(ex.v4, ex.comb)
    print(f"  EX-2026H1 = 2025 (n={len(ex)}): matched-vol DD v4 {ddv2:+.0f}->comb {ddc2:+.0f} = {pct2:+.0f}% | comb Sh {sh_w(ex.comb):+.2f} v4 Sh {sh_w(ex.v4):+.2f}  (2025 = trend BLEED window)")
    he1_primary = pct2 > 0
    he1_secondary = sh_w(ex.comb) >= sh_w(ex.v4)
    print(f"  >> HE-1 PRIMARY (DD-cut>0 ex-2026H1): {'PASS' if he1_primary else 'FAIL'} | SECONDARY (Sharpe>=v4): {'pass' if he1_secondary else 'no (expected in bleed window)'}")
    # full drop-one-half jackknife of DD-cut over 2025-26
    print("  drop-one-half JACKKNIFE (2025-26 DD-cut, shows 2026H1 leverage):")
    for h in sorted(conf.half.unique()):
        sub = conf[conf.half != h]
        _,_,p = matched_dd_cut(sub.v4, sub.comb)
        print(f"     drop {h}: DD-cut {p:+.0f}% (n={len(sub)})")

    # ---------- HE-2: full-sample breadth ----------
    print("\n===== HE-2: full-sample breadth (per-year DD-cut; carry NOT double-counted) =====")
    yr_pass = 0; yrs = ["2023","2024","2025"]
    for yr in yrs:
        g = m[m.yr == yr]
        _,_,p = matched_dd_cut(g.v4, g.comb)
        cs, vs = sh_w(g.comb), sh_w(g.v4); tr_yr = g.tr.sum()
        yr_pass += (p > 0)
        print(f"  {yr} (n={len(g)}): DD-cut {p:+.0f}%  | comb Sh {cs:+.2f} v4 Sh {vs:+.2f}  | trend standalone {tr_yr:+.0f} bps ({'CARRY/BLEED PAID' if tr_yr<0 else 'trend +'} this yr)")
    agg_cs, agg_vs = sh_w(m.comb), sh_w(m.v4)
    print(f"  AGGREGATE 2023-26 (already net of carry): comb Sh {agg_cs:+.2f} vs v4 Sh {agg_vs:+.2f}")
    he2 = (yr_pass >= 2) and (agg_cs >= agg_vs)
    print(f"  >> HE-2: {'PASS' if he2 else 'FAIL'} (DD-cut>0 in {yr_pass}/3 years [need>=2] AND agg comb Sharpe>=v4 [{agg_cs>=agg_vs}])")

    # ---------- HE-3: carry tolerance (descriptive) ----------
    print("\n===== HE-3: carry tolerance (descriptive) =====")
    worst_yr = min(yrs, key=lambda y: m[m.yr==y].tr.sum())
    print(f"  worst bleed year {worst_yr}: trend standalone {m[m.yr==worst_yr].tr.sum():+.0f} bps (the insurance premium paid that year)")
    print(f"  full-2022 crisis DD cut = +48% (single-episode, forward-only; from 23k) — the payoff the premium buys")

    print("\n========== DIV2-HE SUMMARY ==========")
    print(f"  HE-1 (concentration, DD-primary): {'PASS' if he1_primary else 'FAIL'}")
    print(f"  HE-2 (breadth net of carry):      {'PASS' if he2 else 'FAIL'}")
    deploy = he1_primary and he2
    print(f"  >>> OVERLAY DEPLOY-WORTHY (HE-1 & HE-2): {'PASS -> forward ledger (NOT live)' if deploy else 'FAIL -> overlay not deploy-worthy on this data'}")
    print("DIV2HEDONE")

if __name__ == "__main__":
    main()
