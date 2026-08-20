"""EXP-BLEND v3 — ROLLING per-window sigma. No static vol anywhere.

v1 used one pooled sigma; v2 used one static sigma per coin. Both are the thing
we ruled out: volatility clusters, so a constant cannot distinguish a quiet
window from a violent one, and p_hat is the most sigma-sensitive object we have
(a 2x error moves it up to 23 cents).

v3 estimates sigma FOR EACH WINDOW from the trailing tape, using the exact
relation between the observed TWAP increment and the underlying:

    for a w-second TWAP, Var[X_{t+h} - X_t] = sigma^2 (h^2/w - h^3/3w^2)
    at h = w = 60s this is exactly 40 * sigma^2
    => sigma_hat^2 = Var60_trailing / 40                       (horizon-matched,
       non-overlapping 60s increments, so no MA(60) bias)

Multi-scale: a fast (15 min) and a slow (60 min) trailing estimate blended by a
single weight fitted walk-forward. Sigma is then mapped to the settlement
horizon through the verified variance law, NOT annualised -- annualising a
5-minute contract's vol was itself a category error.
"""
import glob, json, math, statistics as st
from bisect import bisect_right, bisect_left
from collections import defaultdict
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

def trailing_sigma(series, t0_ms, lookback_s):
    """sigma_hat from non-overlapping 60s TWAP increments ending at t0.
       Var[X_{t+60}-X_t] = 40*sigma^2 for a 60s TWAP  =>  sigma = sqrt(var/40)."""
    tk, val = series
    pts=[]
    t = t0_ms - lookback_s*1000
    while t <= t0_ms:
        i = bisect_right(tk, t) - 1
        if i >= 0 and tk[i] > t - 3000:      # tick must be fresh, not stale-filled
            pts.append(val[i])
        t += 60_000
    if len(pts) < 6: return None
    rel=[(b-a)/a for a,b in zip(pts, pts[1:]) if a>0]
    if len(rel) < 5: return None
    v=st.pvariance(rel)
    return math.sqrt(v/40.0) if v>0 else None

rows=[]
for slug,up in sorted(res.items()):
    m=markets.get(slug)
    if not m: continue
    s=tw.get(bm.COINS.get(m["coin"]))
    if not s: continue
    t0,T=m["window_start"]*1000,m["window_end"]*1000
    tk=s[0]; lo,hi=bisect_right(tk,t0-5000),bisect_right(tk,T+5000)
    if (hi-lo) < 0.9*((T+5000-(t0-5000))/1000): continue
    x0=bm.at_known(s,t0)
    if not x0: continue
    sf=trailing_sigma(s,t0,15*60); ss=trailing_sigma(s,t0,60*60)
    if sf is None or ss is None: continue
    toks=m.get("clobTokenIds") or []; outs=m.get("outcomes")
    if not (toks and outs): continue
    o=json.loads(outs) if isinstance(outs,str) else outs
    ua=toks[o.index("Up")] if "Up" in o else toks[0]
    bs=bm.book_mid_series(slug,ua)
    if not bs: continue
    for t in GRID:
        ms=t0+t*1000; ex=bm.at_known(s,ms); b=bm.at_known(bs,ms)
        if ex is None or b is None: continue
        r=300.0-t; vf=(r-2*W/3) if r>W else (r**3)/(3*W*W)
        if vf<=0: continue
        rows.append((m["window_start"]//86400, m["coin"], r,
                     (ex-x0)/x0, vf, sf, ss, up, b))
json.dump(rows, open("/tmp/rows3.json","w"))
print(f"samples {len(rows)}  windows {len(rows)//len(GRID)}")

print("\n=== rolling sigma: how much does it MOVE within a coin? ===")
print(f"{'coin':<7} {'n win':>6} {'p10':>9} {'p50':>9} {'p90':>9} {'p90/p10':>8}")
for c in sorted(set(r[1] for r in rows)):
    v=sorted(r[5] for r in rows if r[1]==c)[::len(GRID)]
    if len(v)<10: continue
    q=lambda f: v[min(int(len(v)*f),len(v)-1)]
    print(f"{c:<7} {len(v):>6} {q(.1)*1e4:>8.2f} {q(.5)*1e4:>8.2f} {q(.9)*1e4:>8.2f} {q(.9)/max(q(.1),1e-12):>8.1f}x")
print("  (bps/sqrt(s); a static sigma per coin collapses this whole range to one number)")

def pm_of(rec, wt, k):
    _,_,_,d,vf,sf,ss,_,_ = rec
    sg = k*(wt*sf + (1-wt)*ss)
    return min(max(N(d/(sg*math.sqrt(vf))),1e-6),1-1e-6)
def nll(rs, wt, k):
    return -sum(math.log(p if r[7] else 1-p) for r in rs for p in [pm_of(r,wt,k)])
def fit(rs):
    best=(None,1e18)
    for wt in [i/10 for i in range(11)]:
        lo,hi=.3,3.0
        for _ in range(40):
            a=lo+(hi-lo)/3; b2=hi-(hi-lo)/3
            if nll(rs,wt,a)<nll(rs,wt,b2): hi=b2
            else: lo=a
        k=(lo+hi)/2; v=nll(rs,wt,k)
        if v<best[1]: best=((wt,k),v)
    return best[0]

days=sorted(set(r[0] for r in rows))
print("\n=== walk-forward: ROLLING sigma model vs book ===")
print(f"{'day':<8} {'n':>6} {'w_fast':>7} {'scale':>6} {'Brier model':>12} {'Brier book':>11} {'delta':>8}")
for i,d in enumerate(days):
    if i==0: continue
    tr=[r for r in rows if r[0]<d]; te=[r for r in rows if r[0]==d]
    if len(tr)<200 or not te: continue
    wt,k=fit(tr)
    bmr=st.mean((pm_of(r,wt,k)-(1.0 if r[7] else 0.0))**2 for r in te)
    bbr=st.mean((r[8]-(1.0 if r[7] else 0.0))**2 for r in te)
    print(f"{d:<8} {len(te):>6} {wt:>7.1f} {k:>6.2f} {bmr:>12.4f} {bbr:>11.4f} {bmr-bbr:>+8.4f}")

print("\n=== v1 static-pooled vs v2 static-per-coin vs v3 rolling (same test day) ===")
d=days[-1]; tr=[r for r in rows if r[0]<d]; te=[r for r in rows if r[0]==d]
wt,k=fit(tr)
def static_fit(rs, per_coin):
    out={}
    for c in (set(r[1] for r in rs) if per_coin else ["*"]):
        sub=[r for r in rs if per_coin==False or r[1]==c]
        lo,hi=1e-6,5e-2
        for _ in range(50):
            a=lo+(hi-lo)/3; b2=hi-(hi-lo)/3
            f=lambda sg: -sum(math.log(max(min(N(r[3]/(sg*math.sqrt(r[4]))),1-1e-6),1e-6) if r[7]
                              else 1-max(min(N(r[3]/(sg*math.sqrt(r[4]))),1-1e-6),1e-6)) for r in sub)
            if f(a)<f(b2): hi=b2
            else: lo=a
        out[c]=(lo+hi)/2
    return out
sp=static_fit(tr,False); sc=static_fit(tr,True)
b_pool=st.mean((max(min(N(r[3]/(sp["*"]*math.sqrt(r[4]))),1-1e-6),1e-6)-(1.0 if r[7] else 0.0))**2 for r in te)
b_coin=st.mean((max(min(N(r[3]/(sc.get(r[1],sp["*"])*math.sqrt(r[4]))),1-1e-6),1e-6)-(1.0 if r[7] else 0.0))**2 for r in te)
b_roll=st.mean((pm_of(r,wt,k)-(1.0 if r[7] else 0.0))**2 for r in te)
b_book=st.mean((r[8]-(1.0 if r[7] else 0.0))**2 for r in te)
for lbl,v in (("static pooled",b_pool),("static per-coin",b_coin),("ROLLING per-window",b_roll),("book",b_book)):
    print(f"  {lbl:<20} Brier {v:.4f}   delta vs book {v-b_book:+.4f}")
