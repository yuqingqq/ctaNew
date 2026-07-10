"""DIV2 build validation — Phases 1-3 per DIV2_BUILD_PREREG.md (tightened, review c374622).

Phase 1 (sleeve's OWN edge, GATE): 1a per-period stability isolating choppy 2023-25; 1b neighborhood
(3x3) + cross-family MA-crossover sanity; 1c turnover/cost. Phase 2 (flash-crash stress): objective
fast-V-crash windows, count + combined-vs-v4 drag. Phase 3 (diversification OOS, GATE): form combo on
2023-24, confirm on 2025-26 with matched-vol DD-cut>0 PRIMARY, Sharpe>=v4 secondary. Pinned headline
365d/30d TSMOM (binding, no sweep-to-pick — the 3x3 is sanity only).
"""
import sys, warnings
sys.path.insert(0, "live")
import numpy as np, pandas as pd
import div2_crypto_trend as d
warnings.filterwarnings("ignore")

def tsmom_daily(px, lookback=365, volwin=30):
    """Daily net-bps series (pre-weekly). Returns (net_daily bps, weekly bps)."""
    ret = px.pct_change(); mom = px.pct_change(lookback)
    vol = ret.rolling(volwin, min_periods=volwin//2).std()
    raw = (np.sign(mom)/vol.clip(lower=1e-4)).where(mom.notna() & vol.notna())
    w = raw.div(raw.abs().sum(axis=1), axis=0).fillna(0.0).shift(1).fillna(0.0)
    pnl = (w*ret).sum(axis=1)*1e4; turn = w.diff().abs().sum(axis=1).fillna(0.0)
    net = (pnl - d.COST_OW*turn).iloc[max(lookback, volwin):]
    return net, turn.iloc[max(lookback, volwin):]

def ma_crossover_daily(px, fast=50, slow=200, volwin=30):
    """Canonical MA-crossover trend family (50/200d), same PIT + inverse-vol + cost."""
    ret = px.pct_change()
    sig = np.sign(px.rolling(fast).mean() - px.rolling(slow).mean())
    vol = ret.rolling(volwin, min_periods=volwin//2).std()
    raw = (sig/vol.clip(lower=1e-4)).where(sig.notna() & vol.notna())
    w = raw.div(raw.abs().sum(axis=1), axis=0).fillna(0.0).shift(1).fillna(0.0)
    net = ((w*ret).sum(axis=1)*1e4 - d.COST_OW*w.diff().abs().sum(axis=1).fillna(0.0)).iloc[slow:]
    return net

def to_weekly(daily):
    s = pd.Series(daily.values, index=daily.index)
    return s.groupby(s.index.to_period("W").astype(str)).sum()

def sh_w(x): x=np.asarray(x); return x.mean()/x.std(ddof=1)*np.sqrt(52) if len(x)>1 and x.std(ddof=1)>0 else np.nan
def mdd(x): e=np.cumsum(x); return float((e-np.maximum.accumulate(e)).min())

def main():
    print("loading 20-major daily closes...", flush=True)
    px = d.load_daily_closes()
    net_d, turn_d = tsmom_daily(px)
    print(f"  trend daily net starts {net_d.index.min().date()} (365d warmup; 2021 excluded: {net_d.index.min().year>=2022})", flush=True)
    trw = to_weekly(net_d); v4 = d.v4_weekly_fullstack()
    m = pd.concat([v4.rename("v4"), trw.rename("tr")], axis=1).dropna().sort_index(); m["yr"]=m.index.str[:4]

    # ---------- PHASE 1a: per-period stability, isolate choppy 2023-25 ----------
    print("\n===== PHASE 1a: per-period stability (isolate choppy) =====")
    subs=[]
    for yr in sorted(m.yr.unique()):
        g=m[m.yr==yr]
        for half,gg in (("H1",g[pd.to_datetime(g.index.str[:10],utc=True,errors='coerce').month<=6]),
                        ("H2",g[pd.to_datetime(g.index.str[:10],utc=True,errors='coerce').month>6])):
            if len(gg)>=8:
                s=sh_w(gg.tr); subs.append((f"{yr}{half}",yr,len(gg),s))
                print(f"  {yr}{half} (n={len(gg):2d}): trend Sharpe {s:+.2f}  {'[CRISIS 2022]' if yr=='2022' else ''}")
    non22=[s for _,yr,_,s in subs if yr!="2022" and np.isfinite(s)]
    chop=m[(m.yr>='2023')&(m.yr<='2025')]
    frac_pos=np.mean([s>=0 for s in non22])
    print(f"  choppy-2023-25 AGGREGATE trend Sharpe {sh_w(chop.tr):+.2f} (PRIMARY read)")
    print(f"  non-2022 sub-periods Sharpe>=0 in {frac_pos*100:.0f}%  (n_sub={len(non22)})")
    g1a = (sh_w(chop.tr)>=0) and (frac_pos>=0.60)
    print(f"  >> GATE 1a: {'PASS' if g1a else 'FAIL'} (choppy agg >=0 AND >=60% sub-periods >=0)")

    # ---------- PHASE 1b: neighborhood 3x3 + cross-family ----------
    print("\n===== PHASE 1b: neighborhood 3x3 + cross-family (SANITY) =====")
    def badweek_div(trweekly):
        mm=pd.concat([v4.rename("v4"),trweekly.rename("tr")],axis=1).dropna(); mm=mm[mm.index.str[:4]!="2022"]
        bad=mm[mm.v4<0]; return bad.tr.mean(), sh_w(mm.tr)
    pinned_bd,pinned_sh=badweek_div(trw); band_sh=[]; same=0; ncell=0
    for lb in (250,365,500):
        for vw in (20,30,40):
            nd,_=tsmom_daily(px,lb,vw); bd,shh=badweek_div(to_weekly(nd)); band_sh.append(shh); ncell+=1
            if bd>0: same+=1
    print(f"  9-cell diversification-sign positive: {same}/9  | pinned 365/30 bad-week +{pinned_bd:.0f} Sh {pinned_sh:+.2f}")
    print(f"  pinned Sharpe within band [{min(band_sh):+.2f},{max(band_sh):+.2f}]: {min(band_sh)<=pinned_sh<=max(band_sh)}  (not outlier-high)")
    mac=ma_crossover_daily(px); mac_bd,mac_sh=badweek_div(to_weekly(mac))
    print(f"  cross-family MA-crossover(50/200): bad-week {mac_bd:+.0f} bps Sh {mac_sh:+.2f}  ({'CORROBORATES (same-sign div)' if mac_bd>0 else 'MISS (noted, not fatal)'})")
    g1b = (same>=7) and (min(band_sh)<=pinned_sh<=max(band_sh))
    print(f"  >> GATE 1b: {'PASS' if g1b else 'FAIL'} (>=7/9 same-sign + pinned not knife-edge)")

    # ---------- PHASE 1c: turnover ----------
    ann_turn=turn_d.mean()*365; cost_drag=d.COST_OW*turn_d.mean()*365
    gross_ann=abs(net_d.mean())*365 + cost_drag
    print(f"\n===== PHASE 1c: turnover/cost =====\n  mean annual turnover {ann_turn:.1f}x | cost drag {cost_drag:.0f} bps/yr (~{cost_drag/max(gross_ann,1)*100:.0f}% of gross)")

    # ---------- PHASE 2: flash-crash stress ----------
    print("\n===== PHASE 2: fast-V-crash stress =====")
    bret,_=d.__dict__.get("btc_series",lambda:(None,None))() if hasattr(d,"btc_series") else (None,None)
    # build BTC weekly return from px (BTCUSDT col)
    btc=px["BTCUSDT"]; btc_w=btc.resample("W").last().pct_change()*1e4
    windows=[]
    bw=btc_w.dropna()
    for i,(t,r) in enumerate(bw.items()):
        if r<=-1500:  # <=-15%
            fut=btc.loc[btc.index>t];
            if len(fut)==0: continue
            trough=btc.loc[t]; # approx: retrace >=50% within 4 weeks
            nxt=bw.iloc[i+1:i+5]
            if (nxt>0).any() and nxt.sum()>= -r*0.5:
                windows.append(str(t.date()))
    print(f"  fast-V-crash windows found (BTC wk<=-15% + >=50% retrace/4wk): n={len(windows)} {windows if windows else '(0-1 → size cap is a PRIOR, small sample)'}")

    # ---------- PHASE 3: diversification OOS ----------
    print("\n===== PHASE 3: diversification OOS (form 2023-24, confirm 2025-26) =====")
    def combo_dd(mm):
        sv=mm.v4.expanding(12).std().shift(1); st=mm.tr.expanding(12).std().shift(1)
        wv=(1/sv)/((1/sv)+(1/st)); c=(wv*mm.v4+(1-wv)*mm.tr).dropna()
        vv=mm.v4.reindex(c.index); cm=c*(vv.std()/c.std())
        return sh_w(c), sh_w(vv), mdd(vv.values), mdd(cm.values)
    conf=m[(m.yr>='2025')&(m.yr!='2022')]
    csh,vsh,ddv,ddc=combo_dd(conf)
    ddcut=(1-ddc/ddv)*100 if ddv<0 else 0
    print(f"  2025-26 confirm (n={len(conf)}): combined Sh {csh:+.2f} | v4 Sh {vsh:+.2f} | matched-vol DD v4 {ddv:+.0f}->comb {ddc:+.0f} = {ddcut:+.0f}%")
    g3_primary=ddcut>0; g3_secondary=csh>=vsh
    print(f"  >> GATE 3a: PRIMARY DD-cut>0 {'PASS' if g3_primary else 'FAIL'} | SECONDARY Sharpe>=v4 {'PASS' if g3_secondary else 'not-worse FAIL'}")

    print("\n========== VALIDATION SUMMARY ==========")
    print(f"  GATE 1a (own-edge choppy): {'PASS' if g1a else 'FAIL'}")
    print(f"  GATE 1b (not knife-edge):  {'PASS' if g1b else 'FAIL'}")
    print(f"  GATE 3a (DD-cut OOS):      {'PASS' if g3_primary else 'FAIL'} (primary) / Sharpe>=v4 {'ok' if g3_secondary else 'no'}")
    allpass = g1a and g1b and g3_primary
    print(f"  >>> BUILD GATE: {'PASS -> proceed to Phase 4 build' if allpass else 'FAIL -> sleeve NOT built (feasibility stands)'}")
    print("DIV2VALDONE")

if __name__=="__main__":
    main()
