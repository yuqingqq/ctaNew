"""Causal observed features and protected-boundary parquet access."""
import bisect
from pathlib import Path
import numpy as np
import pandas as pd
from .pf_order_policy_spec import ROOT,DAYS
from .pf_model_spec import SETS

MARKET=SETS['ALL']+['side','order_price','remaining','inventory','order_age_ms','other_reserved',
                    'guard_v','guard_margin','seed','calm','activity_ratio','d_bnimb250','d_pfimb250']
QUEUE=['queue_initial','queue_now','queue_min','queue_progress']
FEATURES=MARKET+QUEUE


class SafeReader:
    def __init__(self,day,read):
        if day not in DAYS:raise ValueError('protected day')
        self.day=day;self.read=read;self.mask=None
        lo=pd.Timestamp(day,tz='UTC');prev=(lo-pd.Timedelta(days=1)).strftime('%Y%m%d');nxt=(lo+pd.Timedelta(days=1)).strftime('%Y%m%d')
        protected={'20261028','20261029','20261030','20261104','20261105','20261106','20261109','20261110','20261111','20261112','20261113'}
        self.lo=int(lo.timestamp()) if prev in protected else None
        self.hi=int((lo+pd.Timedelta(days=1)).timestamp()) if nxt in protected else None

    def __call__(self,path,*args,**kw):
        path=str(path);filters=list(kw.pop('filters',[]) or [])
        if '/step1/' in path:
            d=path.split('/step1/')[1].split('/')[0]
            if d not in DAYS:raise ValueError('protected input directory')
        if path.endswith('legs_timed2.parquet'):
            ident=self.read(path,columns=['exec_s'])
            self.mask=np.ones(len(ident),bool)
            if self.lo is not None:filters.append(('exec_s','>=',self.lo));self.mask &= ident.exec_s.to_numpy()>=self.lo
            if self.hi is not None:filters.append(('exec_s','<',self.hi));self.mask &= ident.exec_s.to_numpy()<self.hi
        elif path.endswith('legs_tk.parquet'):
            a=self.read(path,*args,**kw)
            if self.mask is None or len(a)!=len(self.mask):raise ValueError('taker metadata identity')
            return a[self.mask].reset_index(drop=True)
        elif '/books/' in path:
            if self.lo is not None:filters.append(('t_ms','>=',self.lo*1000))
            if self.hi is not None:filters.append(('t_ms','<',self.hi*1000))
        return self.read(path,*args,filters=filters or None,**kw)


class Inputs:
    def __init__(self,day):
        if day not in DAYS:raise ValueError('protected feature day')
        lo=int(pd.Timestamp(day,tz='UTC').timestamp()*1000)
        path=ROOT/f'data/predict_fun/research/step5/{day}/rows_rl_d2_pk.parquet'
        r=pd.read_parquet(path,columns=['mid','tau']+SETS['ALL'],filters=[('tau','>=',lo),('tau','<',lo+86400000)])
        self.by={int(m):(g.tau.to_numpy(dtype=float),g[SETS['ALL']].to_numpy(dtype=np.float32)) for m,g in r.groupby('mid',sort=False)}


class Window:
    def __init__(self,inputs,mid,F,kf,ws):
        self.F=F;self.kf=np.asarray(kf);self.tf=np.array([f['t_ms'] for f in F]);self.ws=ws
        self.rt,self.x=inputs.by.get(int(mid),(np.empty(0),np.empty((0,len(SETS['ALL'])))))
        self.bp=np.array([[f[f'b{i}_p'] for i in range(1,6)] for f in F]);self.ap=np.array([[f[f'a{i}_p'] for i in range(1,6)] for f in F])
        self.bs=np.array([[f[f'b{i}_s'] for i in range(1,6)] for f in F]);self.ass=np.array([[f[f'a{i}_s'] for i in range(1,6)] for f in F])
        self.calms=np.zeros(len(self.rt),bool);self.ratio=np.full(len(self.rt),np.nan)
        self.db=np.full(len(self.rt),np.nan);self.dp=self.db.copy()
        if not len(self.rt) or not len(F):return
        old=np.searchsorted(self.rt,self.rt-250,side='right')-1
        oldbook=np.searchsorted(self.kf,self.rt-250,side='right')-1;book=np.searchsorted(self.kf,self.rt,side='right')-1
        ix={s:i for i,s in enumerate(SETS['ALL'])}
        a=self.x[:,ix['bn_ntr1000']].astype(float);valid=np.isfinite(a)
        sums=np.r_[0.,np.cumsum(np.where(valid,a,0.))];counts=np.r_[0,np.cumsum(valid)]
        lo=np.searchsorted(self.rt,self.rt-30000);n=np.arange(len(a));den=counts[n]-counts[lo]
        mean=np.divide(sums[n]-sums[lo],den,out=np.full(len(a),np.nan),where=den>=100)
        self.ratio=(1+a)/(1+mean)
        self.db=self.x[:,ix['bn_imb']]-self.x[old,ix['bn_imb']]
        self.dp=self.x[:,ix['pf_imb1']]-self.x[old,ix['pf_imb1']]
        fresh=(old>=0)&(book>=0)&(oldbook>=0)&(self.rt-self.rt[old]<=350)&(self.rt-self.kf[book]<=1000)&(self.rt-250-self.kf[oldbook]<=1000)
        changed=np.r_[True,(self.bp[1:,0]!=self.bp[:-1,0])|(self.ap[1:,0]!=self.ap[:-1,0])]
        last_change=np.maximum.accumulate(np.where(changed,self.kf,-np.inf))
        stable=(self.bp[book,0]==self.bp[oldbook,0])&(self.ap[book,0]==self.ap[oldbook,0])&(self.rt-last_change[book]>=250)
        depth=(self.bs[book,0]>.8*self.bs[oldbook,0])&(self.ass[book,0]>.8*self.ass[oldbook,0])
        fields=np.column_stack([self.ratio,self.db,self.x[:,ix['pf_imb1']],self.x[:,ix['bn_r250']],self.x[:,ix['bn_age']]])
        calm=fresh&stable&depth&np.isfinite(fields).all(axis=1)&(np.abs(fields[:,1])<.3)&(np.abs(fields[:,2])<1/3)&(np.abs(fields[:,3])<.5)&(fields[:,0]<2)&(fields[:,4]<=1000)
        goodbook=(self.bp[book,0]>=0)&(self.ap[book,0]>self.bp[book,0]);calm &= goodbook
        for j in range(4,len(calm)):
            self.calms[j]=bool(calm[j-4:j+1].all() and self.rt[j]-self.rt[j-4]<=200.001)

    def row(self,t):
        j=int(np.searchsorted(self.rt,t,side='right')-1)
        return j if j>=0 and t-self.rt[j]<=100 else -1

    def index(self,t,known=True):
        a=self.kf if known else self.tf;i=int(np.searchsorted(a,t,side='right')-1)
        return i if i>=0 and t-a[i]<=1000 else -1

    def calm(self,t):
        j=self.row(t);return j>=0 and bool(self.calms[j])

    def shown(self,i,side,p):
        if i<0:return np.nan
        pp,qq=(self.bp,self.bs) if side==0 else (self.ap,self.ass)
        match=pp[i]==p
        if match.any():return float(qq[i,match][0])
        last=pp[i,-1]
        if last>=0 and (p<last if side==0 else p>last):return np.nan
        return 0.

    def mid(self,t):
        i=self.index(t,False)
        if i<0 or self.bp[i,0]<0 or self.ap[i,0]<=self.bp[i,0]:return np.nan
        return float((self.bp[i,0]+self.ap[i,0])/2)

    def features(self,t,o,net,other,v,line):
        j=self.row(t);i=self.index(t);birth=self.index(o['t0'])
        if j<0 or i<0:return None
        now=self.shown(i,o['side'],o['pt']);initial=self.shown(birth,o['side'],o['pt'])
        vals=[self.shown(k,o['side'],o['pt']) for k in range(max(0,birth),i+1)];vals=[x for x in vals if np.isfinite(x)]
        if not vals or not np.isfinite(now) or not np.isfinite(initial):return None
        x=dict(zip(SETS['ALL'],map(float,self.x[j])))
        x.update(side=float(o['side']),order_price=float(o['pt']),remaining=float(o['size']-o.get('kq',0)),inventory=float(net),
                 order_age_ms=t-o['t0'],other_reserved=float(other),guard_v=float(v),guard_margin=float(v-line),seed=float(o.get('_seed',False)),
                 calm=float(self.calms[j]),activity_ratio=float(self.ratio[j]),d_bnimb250=float(self.db[j]),d_pfimb250=float(self.dp[j]),
                 queue_initial=initial,queue_now=now,queue_min=min(vals),queue_progress=initial-min(vals))
        return x


def micro_price(f,side):
    b,a,bs,ass=f['b1_p'],f['a1_p'],f['b1_s'],f['a1_s']
    if b<0 or a<=b:return None
    micro=(b*ass+a*bs)/max(bs+ass,1e-12);half=(a-b)/2
    p=int(np.floor(micro-half+1e-9)) if side==0 else int(np.ceil(micro+half-1e-9))
    if not (1<=p<=99) or (p<b or p>=a if side==0 else p>a or p<=b):return None
    touch=b if side==0 else a;nxt=f['b2_p'] if side==0 else f['a2_p']
    if p==touch and (nxt<0 or abs(touch-nxt)>=2):return None
    return p
