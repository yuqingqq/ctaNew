"""Neutral replay and full-engine fork labels, with independent KEEP reconciliation."""
import copy
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from . import pf_continuation_spec as S
_INSTALL=None


def install():
    global _INSTALL
    if _INSTALL is not None:return _INSTALL
    from .pf_default_pull_known_empty import install as original_install
    prior, Window, _ = original_install()
    from . import pf_default_pull_hooks as H
    Base = H.Hook

    class Hook(Base):
        def __init__(self,*args,**kw):
            super().__init__(*args,**kw)
            self.in_fork=False;self.forks={}

        def __deepcopy__(self,memo):
            z=type(self).__new__(type(self));memo[id(self)]=z
            shared={'parent','w','trades','tt','events','guard','value','line'}
            for k,v in self.__dict__.items():
                setattr(z,k,{} if k=='forks' else v if k in shared else copy.deepcopy(v,memo))
            return z

        def selected(self,row):
            return S.selected(row['id']) and row['decision']+30000<=self.ws+240000

        def snapshot(self,*args):pass
        def label(self,*args):pass

        def finish(self,rawfills,*args):
            for gid,paths in self.forks.items():
                row=dict(self.records[gid]);t=row['decision'];side=int(row['side']);price=row['order_price']
                if set(paths)!={'keep','pull'}:raise ValueError('unpaired action')
                for h in S.HORIZONS:
                    actual=[f for f in rawfills if t<=f[0]<t+h]
                    expected=paths['keep']['horizons'][h]['fills']
                    equal_fills(actual,expected)
                    row['keep_reconciled_'+str(h)]=True
                    a=self.trades[np.searchsorted(self.tt,t):np.searchsorted(self.tt,t+h)]
                    row['flow_'+str(h)]=sum(q for _,s,p,q,_ in a if s==side and (p<=price if side==0 else p>=price))
                    for branch in ('keep','pull'):
                        b=paths[branch];z=b['horizons'][h];fs=z['fills'];prefix=f'{branch}_{h}_'
                        signed=sum((1 if f[1]==0 else -1)*f[3] for f in fs)
                        cash=sum(-(1 if f[1]==0 else -1)*f[3]*f[2]/100 for f in fs)
                        if not np.isclose(z['net'],b['initial_net']+signed,atol=1e-7,rtol=0):raise ValueError('fork inventory reconciliation')
                        if not np.isclose(z['cash'],b['initial_cash']+cash,atol=1e-7,rtol=0):raise ValueError('fork cash reconciliation')
                        q=sum(f[3] for f in fs);same=sum(f[3] for f in fs if f[1]==side)
                        terms=[(1 if f[1]==0 else -1)*f[3]*(.5*(self.w.mid(f[0]+1000)+self.w.mid(f[0]+5000))-f[2]) for f in fs]
                        mark=sum(terms) if all(np.isfinite(terms)) else np.nan
                        risk=.01*(z['net']**2-b['initial_net']**2)
                        row.update({prefix+'q':q,prefix+'q2':q*q,prefix+'same_q':same,
                                    prefix+'gain':max(mark,0),prefix+'loss':max(-mark,0),prefix+'risk':risk,
                                    prefix+'value':mark-risk,prefix+'net2':z['net']**2,
                                    prefix+'status':'eligible' if np.isfinite(mark) else 'missing_mark'})
                        if abs(z['net'])>10+1e-6:raise ValueError('fork inventory cap')
                for branch in ('keep','pull'):
                    fs=paths[branch]['horizons'][30000]['fills']
                    first=min((f[0]-t for f in fs if f[1]==side),default=np.inf)
                    for h in (250,1000,2000,10000,30000):row[f'{branch}_first_{h}']=float(first<h)
                    prefix=[f for f in fs if f[0]<t+self.c['cancel']]
                    row[branch+'_prefix']=prefix
                places=[p for p in paths['pull']['horizons'][30000]['placements'] if p[1]==side]
                if any(p[0]<t+1000 for p in places):raise ValueError('re-entry before cooldown')
                for at,ss,px,_ in places:
                    i=self.w.index(at)
                    if i<0 or px!=self.w.F[i]['b1_p' if ss==0 else 'a1_p']:raise ValueError('re-entry not at known touch')
                row['pull_first_place_delay']=min((p[0]-t for p in places),default=np.nan)
                row['pull_first_place_price']=places[0][2] if places else np.nan
                # Cancellation changes the zombie flag immediately, while the
                # unavoidable economic fills must retain time/side/price/size.
                equal_fills([f[:4] for f in row.pop('keep_prefix')],
                            [f[:4] for f in row.pop('pull_prefix')])
                row['status']='eligible' if all(row[f'{b}_{h}_status']=='eligible' for b in ('keep','pull') for h in S.HORIZONS) else 'missing_mark'
                self.parent.paired.append(row)
            for row in self.records.values():
                if np.isfinite(row['decision']):
                    status='not_sampled' if not S.selected(row['id']) else 'censored_30s' if row['decision']+30000>self.ws+240000 else 'forked'
                    self.parent.population.append(dict(id=row['id'],day=self.parent.day,status=status))
                else:self.parent.population.append(dict(id=row['id'],day=self.parent.day,status=row['status']))
            self.parent.reconciled+=len(self.forks)
            super().finish(rawfills,*args)

    class Manager(H.Manager):
        def __init__(self,*args,**kw):
            super().__init__(*args,**kw);self.paired=[];self.population=[];self.reconciled=0
        def window(self,mid,ws,rules,F,kf,trades,events,guard,value,line):
            return Hook(self,'default',mid,ws,F,kf,trades,events,guard,value,line)
        def close(self):
            super().close()
            pd.DataFrame(self.paired).to_parquet(self.folder/'paired.parquet',index=False)
            pd.DataFrame(self.population).to_parquet(self.folder/'population.parquet',index=False)
    _INSTALL=(prior,Manager)
    return _INSTALL


def equal_fills(actual,expected):
    # Full engine tuples, including timestamps, price, quantity, queue and order age.
    if len(actual)!=len(expected):raise ValueError('KEEP fork fill count')
    for a,b in zip(actual,expected):
        if len(a)!=len(b):raise ValueError('KEEP fork fill schema')
        for x,y in zip(a,b):
            if isinstance(x,str):
                if x!=y:raise ValueError('KEEP fork category')
            elif not np.isclose(x,y,atol=1e-9,rtol=0,equal_nan=True):raise ValueError(('KEEP fork mismatch',a,b))


def run(day,smoke=False):
    d=S.check()
    if day not in S.DAYS:raise ValueError('undeclared day')
    old,Manager=install()
    from .pf_continuation_engine import source,verify_transform
    from .pf_order_policy_inputs import SafeReader
    from .pf_order_policy_verify import exact,reconcile
    verify_transform()
    c=old.TIMINGS['X'];dest=S.OUT/('smoke' if smoke else 'runs')/day
    if dest.exists():raise ValueError('refuse overwrite')
    for k in list(os.environ):
        if k.startswith('PF_'):del os.environ[k]
    os.environ.update(PF_FORWARD='20261104',PF_RL_DDEC='2',PF_RL_LPLACE=str(c['place']),PF_RL_LCANCEL=str(c['cancel']),
        PF_RL_LPENDCANCEL='95',PF_RL_LFILL=str(c['notice']),PF_TIMING='e2',PF_TXTIME='ws',PF_TXKEY='tk',
        PF_TXSHIFT=str(c['shift']),PF_QACT='max',PF_PREDS='preds_t60k',PF_ARMS=old.arm('default'),PF_MAX_WINDOWS='4' if smoke else '0')
    manager=Manager(day,'X',dest,'build');code=source('M')
    sys.argv=[str(old.HERE/'pf_step4_ladder_v3.py'),day,'tokyoreal','custom']
    original=pd.read_parquet;pd.read_parquet=SafeReader(day,original)
    S.progress('build_'+day,'running',f'Full-state KEEP/PULL forks; {"four-window smoke" if smoke else "complete day"}; 2/10/30s profiles and neutral parity.')
    try:exec(compile(code,'continuation_engine','exec'),dict(__name__='__main__',__package__=__package__,
        __file__=str(old.HERE/'pf_step4_ladder_v3.py'),_DEST=str(dest),_STUDY=manager))
    finally:pd.read_parquet=original
    mids=pd.read_parquet(dest/'win'/old.stem('default')).mid
    hashes={}
    for kind in ('win','fills','requests'):
        p=dest/kind/old.stem('default');a=pd.read_parquet(p)
        b=pd.read_parquet(S.PRIOR/'runs/build'/day/kind/old.stem('default'))
        exact(a,b[b.mid.isin(mids)]);hashes[kind]=S.sha(p)
    a=pd.read_parquet(dest/'generations/default.parquet')
    b=pd.read_parquet(S.PRIOR/'runs/build'/day/'generations/default.parquet');b=b[b.mid.isin(mids)]
    cols=['id','day','decision','initial_size','t_send','t_act']
    from .pf_order_policy_inputs import FEATURES
    exact(a[cols+FEATURES],b[cols+FEATURES])
    ind=pd.read_parquet(dest/'independent_windows.parquet')
    reconcile(*(pd.read_parquet(dest/k/old.stem('default')) for k in ('win','fills','requests')),ind,a,len(mids))
    paired=pd.read_parquet(dest/'paired.parquet');pop=pd.read_parquet(dest/'population.parquet')
    if a.id.duplicated().any() or paired.id.duplicated().any() or set(a.id)!=set(pop.id):raise ValueError('population identity')
    if len(paired)!=int(pop.status.eq('forked').sum()) or len(paired)!=manager.reconciled:raise ValueError('unreconciled fork')
    for name in ('paired.parquet','population.parquet','generations/default.parquet','independent_windows.parquet'):hashes[name]=S.sha(dest/name)
    r=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),status='PASS',day=day,smoke=smoke,windows=len(mids),
        generations=len(a),fork_actions=len(paired),statuses=pop.status.value_counts().to_dict(),paired_statuses=paired.status.value_counts().to_dict(),
        keep_forks_reconciled=manager.reconciled,neutral_exact=True,hashes=hashes,source_sha=hashlib.sha256(code.encode()).hexdigest(),declaration_sha=S.sha(S.OUT/'declaration.json'))
    S.write(('smoke_' if smoke else 'build_')+day+'.json',r)
    S.progress('build_'+day,'complete',f'{len(mids)} windows; {len(a)} neutral generations; {len(paired)} paired actions. Exact neutral and every KEEP fork reconciled.')

if __name__=='__main__':run(sys.argv[1],len(sys.argv)>2 and sys.argv[2]=='smoke')
