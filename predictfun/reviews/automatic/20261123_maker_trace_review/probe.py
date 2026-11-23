"""Read-only, bounded review of target 1af85c5691d5ca4c2f9565d94a13cd9680688d3c.
Uses the detached committed source, not the development working tree.
No raw tape replay, fitting or output repair. Run only through PF_SLOT=codex run_light.sh.
"""
from pathlib import Path
import sys, json, importlib.util, time, hashlib, gc
import numpy as np
import pandas as pd

PIN = Path('/tmp/pf-mt-review-1af85c56')
ROOT = Path('/home/yuqing/ctaNew')
OUT = ROOT / 'data/predict_fun/research/maker_trace_v1_20261122'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIN / 'predictfun/research'))
import pf_mt_lifecycle as LC
spec = importlib.util.spec_from_file_location('checker', PIN / 'predictfun/reviews/maker_trace_20261122/mt_checks.py')
CK = importlib.util.module_from_spec(spec); spec.loader.exec_module(CK)
START = time.process_time()
R = {'target': '1af85c5691d5ca4c2f9565d94a13cd9680688d3c', 'markets': {}}

def frame(p, **kw):
    return pd.read_parquet(p, **kw)

for day, mid in [('20261024',3697902), ('20261025',3734864), ('20261108',4353706), ('20261108',4334454)]:
    stem = f'{day}_{mid}'; r = {}; R['markets'][stem] = r
    L = frame(OUT/f'legs_{day}.parquet', filters=[('mid','=',mid)])
    records = frame(OUT/f'records_{day}.parquet', filters=[('mid','=',mid)])
    F = frame(OUT/'life'/f'{stem}_fills.parquet')
    O = frame(OUT/'life'/f'{stem}_orders.parquet')
    FR = frame(OUT/'depth'/f'{stem}_frames.parquet')
    LV = frame(OUT/'depth'/f'{stem}_levels.parquet')
    for obj in (L,F,O): obj['h'] = obj.mk_hash.str.lower()
    clean = CK.only_flags(L.status, {'ok','boundary_censored'})
    keys = set(zip(L.sid[clean],L.mk_idx[clean]))
    Fb = F.assign(lo_ms=F.x_lo_hard_ms)
    U = CK.units_of(Fb[[k in keys for k in zip(F.sid,F.mk_idx)]])
    T,S = CK.book_arrays(FR, LV[CK.usable(LV.status)], {k[1:] for k in U})
    viol, members, caps, zeros = CK.h3_scan(U,T,S)
    capd = {k[0]:v for k,v in caps.items()}
    bad = O[(O.status=='ok') & O.h.isin(capd) & (O.post_lo_ms < O.h.map(capd))]
    cap_cases=[]
    for h in bad.h.unique():
        unit = next(k for k in U if k[0]==h); u=U[unit]; tt=capd[h]; j=np.flatnonzero(T==tt)
        cap_cases.append(dict(h=h, t=int(tt), posted=bad[bad.h==h].post_lo_ms.tolist(), cum=u['cum'], shown=S[unit[1:]][j].tolist(), deficit=(u['cum']-S[unit[1:]][j]).tolist()))
    r['capacity']={'n_rows':len(bad),'n_hashes':len(cap_cases),'cases':cap_cases[:4],
                   'max_deficit':max((max(x['deficit']) for x in cap_cases),default=0)}
    pe=CK.priority_evidence(L[clean],L)
    last=L.groupby('h').settle_ms.max(); hk=L.set_index(['sid','mk_idx'])
    ev={hk.h[k] for k in pe['ev'] if hk.settle_ms[k]>=last[hk.h[k]]}
    b=O[(O.status=='ok') & O.scenario.isin(CK.FIFO_FAMILY) & O.h.isin(ev) & ~np.isclose(O.size_exact,O.cum_shares,rtol=0,atol=1e-9)]
    bad_ex=[]
    for h in b.h.unique():
        lf=L[L.h==h]; row=lf.sort_values(['settle_ms','sid','mk_idx']).iloc[-1]; grp=L[L.sid==row.sid]
        ties=[str(hh) for hh in grp.h if ((L.h==hh)&(L.settle_ms==row.settle_ms)).sum()>1]
        bad_ex.append(dict(h=h,sid=row.sid, own_settle=lf.settle_ms.tolist(), group_tied_hashes=ties, status=b[b.h==h].status.tolist()))
    r['exhaustion']={'n_rows':len(b),'n_hashes':len(bad_ex),'all_have_group_ties':all(x['group_tied_hashes'] for x in bad_ex),'examples':bad_ex[:3]}
    L['hi']=np.minimum(L.settle_ms,L.first_recv_ns//10**6)
    first=L[L.hi==L.groupby('h').hi.transform('min')]
    vals={}
    for s,i,h in zip(first.sid,first.mk_idx,first.h): vals.setdefault(h,set()).add(pe['ahead'].get((s,int(i)),0.))
    b=O[(O.status=='ok') & O.scenario.isin(CK.FIFO_FAMILY) & np.isfinite(O.ahead_lo)]
    b=b[[not any(abs(float(x)-v)<=1e-9 for v in vals.get(h,{0.})) for h,x in zip(b.h,b.ahead_lo)]]
    ac=[]
    for h in b.h.unique():
        lf=first[first.h==h]; s=lf.sid.iloc[0]; grp=L[L.sid==s]
        tied=[str(hh) for hh in grp.h if ((L.h==hh)&(L.settle_ms==lf.settle_ms.iloc[0])).sum()>1]
        ac.append(dict(h=h,sid=s,expected=sorted(vals[h]),got=b[b.h==h].ahead_lo.tolist(),group_tied_hashes=tied))
    r['ahead']={'n_rows':len(b),'all_have_group_ties':all(x['group_tied_hashes'] for x in ac),'examples':ac[:3]}
    t2=[]
    for s,g in L[clean].groupby('sid',sort=False):
        buy=g.tk_y_side.iloc[0]=='bid'; p=g.y_ptick.to_numpy(); worst=p.max() if buy else p.min(); w=g[g.y_ptick==worst]
        later=w[w.h.map(last)>w.settle_ms]
        if len(later)>=2:
            t2.append(dict(sid=s, later_h= later.h.tolist(),tied_group_hashes=[str(hh) for hh in w.h if ((L.h==hh)&(L.settle_ms==w.settle_ms.iloc[0])).sum()>1]))
    r['T2']={'checker':pe['t2'],'cases':t2}
    agg=F.groupby('h').agg(n=('shares','size'), hmin=('hi_ms','min'))
    o=O.join(agg,on='h').join(Fb[Fb.lo_ms>=0].groupby('h').lo_ms.max().rename('Lmx'),on='h')
    em=(o.n<=1) | o.Lmx.isna() | (o.hmin>o.Lmx)
    b=o[(o.persist_lo_ms!=np.where(em,-1,o.hmin)) | (o.persist_hi_ms!=np.where(em,-1,o.Lmx))]
    r['H2']={'n_rows':len(b),'rows':b[['h','scenario','status','n','persist_lo_ms','persist_hi_ms','hmin','Lmx']].to_dict('records')}
    tb=frame(OUT/f'tob_{day}.parquet',filters=[('mid','=',mid)])
    tb=tb[(tb.upd_ms>=FR.upd_ms.min()) & (tb.upd_ms<=FR.upd_ms.max())]
    key=['upd_ms','recv_ns','conn']
    xx=FR.merge(tb,on=key,suffixes=('','_t'))
    # Same raw receipt can have multiple sequential frame messages: retain occurrence, not Cartesian join.
    fr2=FR.assign(occ=FR.groupby(key,sort=False).cumcount()); tb2=tb.assign(occ=tb.groupby(key,sort=False).cumcount())
    yy=fr2.merge(tb2,on=key+['occ'],suffixes=('','_t'),validate='one_to_one')
    def tob_diff(x):
        z={}; valid=LV[CK.usable(LV.status)]
        for side,how in [('bid','max'),('ask','min')]:
            vv=valid[valid.side==side]; best=getattr(vv.groupby('fid').ptick,how)(); ss=vv.set_index(['fid','ptick']).shares
            got=x[f'best_{side}_sh']; expect=np.array([ss.get((f,int(p)),np.nan) if np.isfinite(p) else np.nan for f,p in zip(x.fid,x.fid.map(best))])
            z[side]=int((~np.isclose(got,expect,rtol=0,atol=1e-9,equal_nan=True)).sum())
        return z
    r['tob']={'frames':len(FR),'tob':len(tb),'cartesian_rows':len(xx),'occurrence_rows':len(yy),'cartesian_differences':tob_diff(xx),'occurrence_differences':tob_diff(yy),
              'duplicate_receipt_keys':int(FR.duplicated(key,keep=False).sum())}
    print(stem, json.dumps(r,default=str)[:500],flush=True)
    del L,records,F,O,FR,LV,Fb,o,b,xx,yy,tb
    gc.collect()

P=PIN/'predictfun/reviews/maker_trace_20261122'
S=pd.read_csv(P/'outputs/econ_segments.csv',dtype={'day':str})
R['segments']={}
for grp in ['in','field']:
    sel=S[(S.id_set=='exante_sh') & (S.grp==grp) & (S.role=='maker') & (S.day=='pooled') & S.dim.isin(['depth','sweep'])]
    rows=[]
    for _,r in sel.iterrows():
        item={c:r[c] for c in ['dim','bucket','sh','net_res']}
        for hz in ['50ms','100ms','1s','5s','30s','end']:
            if 'v_'+hz in r.index: item['c_'+hz]=100*r['v_'+hz]/r['sh_'+hz] if r['sh_'+hz]>0 else None
        rows.append(item)
    R['segments'][grp]=rows
R['cpu_s']=time.process_time()-START
R['peak_mib']=LC.vmhwm_mib()
(HERE/'probe.json').write_text(json.dumps(R,indent=2,default=str,allow_nan=True)+'\n')
print(json.dumps({'cpu_s':R['cpu_s'],'peak_mib':R['peak_mib']},indent=2))
