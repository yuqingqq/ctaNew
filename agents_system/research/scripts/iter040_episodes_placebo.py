"""
iter-040 STEP 3 — the DECISIVE honest tests for the regime-gated new-listing short.
1. PER-BEAR-EPISODE decomposition: identify distinct alt-bear episodes (contiguous alt30<X spells),
   assign each gated event to its episode, report per-episode PnL + event count. If essentially ALL
   the gated PnL comes from ONE bear episode (the 2025 alt-bear), the gate is just re-selecting the
   2025 artifact -> cannot certify a forward regime edge.
2. REGIME-SHUFFLE PLACEBO: shuffle which entry-dates are 'bear' (preserve the bear FRACTION), pick a
   matched-count random subset of events, compare gated mean to the placebo distribution (>=p95?).
3. THIN-EVENT BOOTSTRAP CI on the gated short (P(mean>0)>=95%?).
"""
import pandas as pd, numpy as np
np.seterr(all='ignore')
SDIR='/home/yuqing/ctaNew/agents_system/research/scripts'
T = pd.read_parquet(f'{SDIR}/iter040_events_gated.parquet')
T = T.sort_values('entry_date').reset_index(drop=True)

# reload the daily alt-index to identify episodes
import glob, os
CACHE='/home/yuqing/ctaNew/data/ml/cache'
closes={}
for f in sorted(glob.glob(f'{CACHE}/xs_feats_*.parquet')):
    sym=os.path.basename(f).replace('xs_feats_','').replace('.parquet','')
    c=pd.read_parquet(f,columns=['close'])['close'].dropna()
    if len(c): closes[sym]=c.resample('1D').last()
px=pd.DataFrame(closes).sort_index()
ret30=px/px.shift(30)-1.0
alt=ret30.mean(axis=1,skipna=True).shift(1).dropna()

def episodes(X):
    """Contiguous spells where alt<X. Returns list of (start,end) date tuples."""
    flag=(alt<X)
    eps=[]; in_ep=False; st=None
    for d,v in flag.items():
        if v and not in_ep: in_ep=True; st=d
        elif not v and in_ep: in_ep=False; eps.append((st, prev))
        prev=d
    if in_ep: eps.append((st, prev))
    return eps

def assign_episode(entry_date, eps):
    for i,(s,e) in enumerate(eps):
        if s <= entry_date <= e: return i
    return -1

def boot_pgt(p, seed=0, nb=5000):
    p=np.asarray(p); rng=np.random.default_rng(seed)
    b=np.array([rng.choice(p,len(p),replace=True).mean() for _ in range(nb)])
    return b.mean(), np.percentile(b,[2.5,97.5]), (b>0).mean()

for X, pnlcol in [(-0.10,'pnl_stop30'), (-0.10,'pnl_naked'), (0.0,'pnl_stop30')]:
    print(f'\n{"="*78}\n PER-BEAR-EPISODE DECOMPOSITION  (gate alt30<{X:+.2f}, pnl={pnlcol})\n{"="*78}')
    eps = episodes(X)
    # merge episodes separated by < 14d gap (treat as same regime), and require min length 14d
    merged=[]
    for s,e in eps:
        if merged and (s - merged[-1][1]).days < 14:
            merged[-1]=(merged[-1][0], e)
        else: merged.append((s,e))
    eps=[(s,e) for s,e in merged if (e-s).days>=14]
    print(f' distinct alt-bear episodes (>=14d, gaps<14d merged): {len(eps)}')
    for i,(s,e) in enumerate(eps):
        print(f'   ep{i}: {s.date()} -> {e.date()}  ({(e-s).days}d)')

    bear = T[T['alt30']<X].copy()
    bear['ep'] = bear['entry_date'].apply(lambda d: assign_episode(pd.Timestamp(d), eps))
    print(f'\n per-episode gated-short PnL (events with valid episode):')
    tot_pnl=0; tot_n=0
    for i,(s,e) in enumerate(eps):
        g=bear[bear['ep']==i]
        if len(g)==0: continue
        sp=g[pnlcol].sum()
        print(f'   ep{i} {s.date()}..{e.date()}: n={len(g):>3} mean={g[pnlcol].mean():+.4f} '
              f'sumPnL={sp:+.3f}')
        tot_pnl+=sp; tot_n+=len(g)
    orphan=bear[bear['ep']==-1]
    print(f'   (orphan/edge events not in a >=14d episode: n={len(orphan)} '
          f'sumPnL={orphan[pnlcol].sum():+.3f})')
    print(f'   TOTAL in-episode: n={tot_n} sumPnL={tot_pnl:+.3f}')
    # concentration: largest single episode share of total positive PnL
    ep_sums=[bear[bear["ep"]==i][pnlcol].sum() for i in range(len(eps))]
    if ep_sums:
        srt=sorted(ep_sums,reverse=True)
        print(f'   largest episode sumPnL share of total: {srt[0]/sum(ep_sums)*100:.0f}% '
              f'| top-1 ep PnL={srt[0]:+.3f} of total {sum(ep_sums):+.3f}')

# focus the placebo on the best config: alt30<-0.10, stop+30%
print(f'\n{"="*78}\n REGIME-SHUFFLE PLACEBO  (gate alt30<-0.10, stop+30%)\n{"="*78}')
X=-0.10; pnlcol='pnl_stop30'
bear = T[T['alt30']<X]
real_mean = bear[pnlcol].mean(); k=len(bear); N=len(T)
allp = T[pnlcol].values
# placebo: pick a random matched-count k subset of ALL events -> distribution of means
rng=np.random.default_rng(7)
plac=np.array([rng.choice(allp,k,replace=False).mean() for _ in range(5000)])
rank=(plac<real_mean).mean()*100
print(f' real gated mean={real_mean:+.4f} (k={k} of N={N})')
print(f' random-{k}-subset placebo: mean={plac.mean():+.4f} p5={np.percentile(plac,5):+.4f} '
      f'p95={np.percentile(plac,95):+.4f} max={plac.max():+.4f}')
print(f' real ranks p{rank:.0f} of random-subset placebo  '
      f'({"PASS >=p95" if rank>=95 else "FAIL <p95"})')

# regime-shuffle placebo: keep the bear FRACTION but assign bear-flag to random entry-dates by
# circularly rotating the alt-index relative to events (preserves autocorrelation of regime).
print(f'\n CIRCULAR-ROTATION regime placebo (preserve regime autocorr, shift alt-index vs events):')
ent=[pd.Timestamp(x).tz_localize(None) if pd.Timestamp(x).tz is None else pd.Timestamp(x).tz_convert(None) for x in T['entry_date']]
alt_nv = alt.copy(); alt_nv.index = alt_nv.index.tz_localize(None) if alt_nv.index.tz is not None else alt_nv.index
def gated_mean_with_offset(off_days):
    altsh = alt_nv.copy(); altsh.index = altsh.index + pd.Timedelta(days=off_days)
    sel=[]
    for ed,p in zip(ent, T[pnlcol].values):
        s=altsh[altsh.index<=ed]
        if len(s) and s.iloc[-1]<X: sel.append(p)
    return np.mean(sel) if len(sel)>=10 else np.nan, len(sel)
offs=[o for o in range(-360,361,15) if abs(o)>=60]
rot=[];
for o in offs:
    m,n=gated_mean_with_offset(o)
    if np.isfinite(m): rot.append(m)
rot=np.array(rot)
rrank=(rot<real_mean).mean()*100
print(f'  rotation placebo means: n={len(rot)} mean={rot.mean():+.4f} p95={np.percentile(rot,95):+.4f} max={rot.max():+.4f}')
print(f'  real ranks p{rrank:.0f} of rotation placebo  ({"PASS" if rrank>=95 else "FAIL"})')

# thin-event bootstrap CI
m,ci,pgt=boot_pgt(bear[pnlcol].values, seed=11)
print(f'\n THIN-EVENT BOOTSTRAP CI (gated stop+30%, alt30<-0.10): mean={m:+.4f} '
      f'CI95[{ci[0]:+.4f},{ci[1]:+.4f}] P(>0)={pgt:.0%}  ({"PASS >=95%" if pgt>=0.95 else "FAIL <95%"})')
