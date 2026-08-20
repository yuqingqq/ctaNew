"""EXP-BLEND v2 — corrected: per-coin sigma, admissibility filter, drift control.

v1 flaws found in review:
  * ONE pooled sigma across seven coins -- worse than static-per-coin, since
    BTC and DOGE have very different volatility. A 2x sigma error moves p_hat
    by up to 23 cents, so v1's "model loses to book" may be an artefact.
  * no admissibility filter (E0 finds 19% of windows lack TWAP coverage).
  * calibration not controlled for the sample's +4.5pp up-drift.

v2 fits sigma PER COIN walk-forward, restricts to admissible windows, and
reports the calibration slope both raw and drift-adjusted. Drift enters
probability space through phi(d), so the adjustment is hump-shaped, not a
level shift -- a uniform subtraction would be wrong.
"""
import glob, gzip, json, math, statistics as st
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
import importlib.util
spec=importlib.util.spec_from_file_location("bm","live/pm_research/exp_blend_model.py")
bm=importlib.util.module_from_spec(spec); spec.loader.exec_module(bm)
N=bm.N; PM=bm.PM; W=bm.W; GRID=bm.GRID
phi=lambda x: math.exp(-x*x/2)/math.sqrt(2*math.pi)

tw=bm.load_twap60()
markets={}
for ln in open(PM/"markets.jsonl"):
    try: m=json.loads(ln); markets[m["slug"]]=m
    except: pass
res={}
for ln in open(PM/"resolutions.jsonl"):
    try:
        r=json.loads(ln)
        if r.get("closed") is True and r.get("winners"): res[r["slug"]]=bool(r["winners"].get("Up"))
    except: pass
idx={s:[a for a,_ in zip(*[v[0],v[1]])] for s,v in tw.items()}

rows=[]
for slug,up in sorted(res.items()):
    m=markets.get(slug)
    if not m: continue
    sym=bm.COINS.get(m["coin"]); s=tw.get(sym)
    if not s: continue
    t0,T=m["window_start"]*1000,m["window_end"]*1000
    # ADMISSIBILITY: >=90% of expected 1s ticks over [t0-5s, T+5s]
    tk=s[0]; lo,hi=bisect_right(tk,t0-5000),bisect_right(tk,T+5000)
    if (hi-lo) < 0.9*((T+5000-(t0-5000))/1000): continue
    x0=bm.at_known(s,t0)
    if not x0: continue
    toks=m.get("clobTokenIds") or []; outs=m.get("outcomes")
    if not (toks and outs): continue
    o=json.loads(outs) if isinstance(outs,str) else outs
    ua=toks[o.index("Up")] if "Up" in o else toks[0]
    bs=bm.book_mid_series(slug,ua)
    if not bs: continue
    day=m["window_start"]//86400
    for t in GRID:
        ms=t0+t*1000; ex=bm.at_known(s,ms); b=bm.at_known(bs,ms)
        if ex is None or b is None: continue
        r=300.0-t; vf=(r-2*W/3) if r>W else (r**3)/(3*W*W)
        if vf<=0: continue
        rows.append((day,m["coin"],r,(ex-x0)/x0/math.sqrt(vf),up,b))
json.dump(rows,open("/tmp/rows2.json","w"))
print(f"admissible samples: {len(rows)}  windows: {len(set((d,c) for d,c,_,_,_,_ in rows))}")

def nll(sig,rs):
    return -sum(math.log(min(max(N(z/sig),1e-6),1-1e-6) if u else 1-min(max(N(z/sig),1e-6),1-1e-6)) for _,_,_,z,u,_ in rs)
def fit(rs):
    lo,hi=1e-6,5e-2
    for _ in range(60):
        a=lo+(hi-lo)/3; b2=hi-(hi-lo)/3
        if nll(a,rs)<nll(b2,rs): hi=b2
        else: lo=a
    return (lo+hi)/2

print("\n=== per-coin sigma (MLE on winners, all data) ===")
print(f"{'coin':<8} {'n':>6} {'bps/sqrt(s)':>12} {'ann %':>8}")
sig_coin={}
for c in sorted(set(r[1] for r in rows)):
    v=[r for r in rows if r[1]==c]
    sg=fit(v); sig_coin[c]=sg
    print(f"{c:<8} {len(v):>6} {sg*1e4:>12.3f} {sg*math.sqrt(365*24*3600)*100:>7.0f}%")
sig_pool=fit(rows)
print(f"{'POOLED':<8} {len(rows):>6} {sig_pool*1e4:>12.3f} {sig_pool*math.sqrt(365*24*3600)*100:>7.0f}%")
print(f"  spread across coins: {min(sig_coin.values())*1e4:.2f} - {max(sig_coin.values())*1e4:.2f} bps/sqrt(s)"
      f"  ({max(sig_coin.values())/min(sig_coin.values()):.1f}x)")

days=sorted(set(r[0] for r in rows))
print(f"\n=== walk-forward: model (per-coin sigma) vs book ===")
print(f"{'day':<8} {'n':>6} {'Brier model':>12} {'Brier book':>11} {'delta':>8}")
for i,d in enumerate(days):
    if i==0: continue
    tr=[r for r in rows if r[0]<d]; te=[r for r in rows if r[0]==d]
    if len(tr)<200 or not te: continue
    sc={c:fit([r for r in tr if r[1]==c]) for c in set(r[1] for r in tr)}
    bmr=st.mean((min(max(N(z/sc.get(c,sig_pool)),1e-6),1-1e-6)-(1.0 if u else 0.0))**2 for _,c,_,z,u,_ in te)
    bbr=st.mean((b-(1.0 if u else 0.0))**2 for _,_,_,_,u,b in te)
    print(f"{d:<8} {len(te):>6} {bmr:>12.4f} {bbr:>11.4f} {bmr-bbr:>+8.4f}")

print("\n=== book calibration, admissible only, raw vs drift-adjusted ===")
mu=st.mean(1.0 if u else 0.0 for _,_,_,_,u,_ in rows)-0.5
print(f"sample up-drift in probability space: {mu:+.4f}")
buck=defaultdict(list)
for _,c,r,z,u,b in rows: buck[min(int(b*10),9)].append((b,u,z,sig_coin[c]))
print(f"{'bucket':>11} {'n':>6} {'book':>7} {'realised':>9} {'raw gap':>8} {'drift adj':>10} {'net':>8}")
for k in sorted(buck):
    v=buck[k]; mb=st.mean(b for b,_,_,_ in v); mr=st.mean(1.0 if u else 0.0 for _,u,_,_ in v)
    # drift enters via phi(d): largest ATM, ~0 at the extremes
    adj=st.mean(phi(z/sg)*(mu/max(phi(0),1e-9)) for _,_,z,sg in v)
    print(f"{k/10:>5.1f}-{(k+1)/10:<4.1f} {len(v):>6} {mb:>7.3f} {mr:>9.3f} {mr-mb:>+8.3f} {adj:>+10.3f} {mr-mb-adj:>+8.3f}")
