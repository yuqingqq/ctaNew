"""Frozen, outcome-blind matched-book diagnostic; no trading or fill simulation."""
from pathlib import Path
import collections
import datetime as dt
import gzip
import hashlib
import json
import subprocess
import numpy as np
import pandas as pd

ROOT=Path('/home/yuqing/ctaNew')
HERE=Path(__file__).resolve().parent
DEST=ROOT/'data/predict_fun/research/polymarket_leadlag_20261110'
REVIEW=HERE.parent/'reviews/polymarket_leadlag_20261110'
DAYS=['20261023','20261024','20261025','20261026','20261027','20261031','20261101','20261102','20261103']
HOURS=list(range(0,24,3))
LAGS=np.array([-50,-30,-20,-10,-5,-2,-1,0,1,2,5,10,20,30,50])
PHASES=[(5,60),(60,180),(180,240),(240,295)]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def day_ns(day):
    if day not in DAYS:raise ValueError('undeclared/protected date')
    return pd.Timestamp(day,tz='UTC').value


def metadata(rows,slugs):
    result={}
    for r in rows:
        slug=r.get('slug')
        if slug not in slugs:continue
        outcomes=r['outcomes'];outcomes=json.loads(outcomes) if isinstance(outcomes,str) else outcomes
        tokens=r['clobTokenIds'];tokens=json.loads(tokens) if isinstance(tokens,str) else tokens
        ws=int(slug.rsplit('-',1)[1])
        if outcomes!=['Up','Down'] or len(tokens)!=2 or len(set(tokens))!=2 or r['window_start']!=ws or r['window_end']!=ws+300:
            raise ValueError('market/outcome identity mismatch')
        if r['recv_ns']>ws*10**9:continue
        if slug not in result or r['recv_ns']<result[slug]['recv_ns']:
            result[slug]=dict(r,up_token=str(tokens[0]))
    return result


def setup():
    if DEST.exists():raise ValueError('refuse existing study')
    population=[];inputs={}
    for day in DAYS:
        p=ROOT/f'data/predict_fun/research/step1/{day}/registry.parquet';inputs[str(p)]=sha(p)
        reg=pd.read_parquet(p,columns=['mid','ws','we','up_index','feed','provider','symbol'])
        if reg.ws.duplicated().any():raise ValueError('duplicated Predict market start')
        for hour in HOURS:
            ws=int(day_ns(day)//10**9+hour*3600);z=reg[reg.ws==ws]
            if len(z)!=1 or int(z.iloc[0].we)!=ws+300 or list(z.iloc[0].up_index)!=[1]:
                raise ValueError('Predict market/window identity')
            mid=int(z.iloc[0].mid);slug=f'btc-updown-5m-{ws}'
            paths=dict(pm=str(ROOT/f'data/pm_5min/raw/{day}/{slug}.jsonl.gz'),
                pf=str(ROOT/f'data/predict_fun/research/step1/{day}/books/part_{day}_{hour:02d}.parquet'),
                bn=str(ROOT/f'data/mm_hf/raw/bookTicker/BTCUSDT/{day}_{hour:02d}.csv.gz'))
            for name in paths.values():
                if Path(name).is_file():inputs[name]=sha(name)
            population.append(dict(day=day,hour=hour,ws=ws,mid=mid,slug=slug,paths=paths,
                predict_contract={k:str(z.iloc[0][k]) for k in ('feed','provider','symbol')},
                missing_files=[k for k,v in paths.items() if not Path(v).is_file()]))
    slugs={x['slug'] for x in population}
    # Registry envelopes only; only selected, already-consumed identities are decoded.
    selected=[]
    with (ROOT/'data/pm_5min/markets.jsonl').open() as f:
        for line in f:
            if any(slug in line for slug in slugs):selected.append(json.loads(line))
    meta=metadata(selected,slugs)
    DEST.mkdir(parents=True);write(DEST/'pm_metadata.json',meta);inputs[str(DEST/'pm_metadata.json')]=sha(DEST/'pm_metadata.json')
    sources=[Path(__file__),HERE/'pf_pm_leadlag_stats.py',HERE/'tests/test_pm_leadlag.py']
    d=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        source_sha={str(p.relative_to(HERE.parents[1])):sha(p) for p in sources},input_sha=inputs,population=population,
        design='72 scheduled BTC windows: first5min of every third UTC hour,9 previously consumed dates. Missing windows remain statuses; no backfill. This samples days, not all288 windows/day.',
        unit='Neutral100ms diagnostic grid, not order actions, fills or independent samples. No execution or PnL claim.',
        clocks='Primary local receipt clock; separate source-clock diagnostic sorted by venue timestamp, not a causal executable clock. No offset fitted.',
        prices='Published Up BBO only; preserve invalid rows; strict two-sided 0<bid<ask<100, no cross-token reconstruction.',
        freshness_ms=[1000,5000],grid_ms=100,lag_ms=(LAGS*100).tolist(),phases_s=PHASES,
        correlation='Within-window/phase centered100ms mid changes; positive lag means PM change precedes Predict change. Both lag endpoints remain in same phase/window.',
        null=dict(draws=500,seed=20261110,method='circular shifts of PM changes within each window/phase by >=10s; family maximum of absolute positive-minus-negative-lag asymmetry across7lags x4phases=28, primary receipt/1s freshness only'),
        prediction='Expanding chronological ridge: first3 dates train, next6 test; primary1s future Predict mid change, 0.5s/5s descriptive. Baseline Predict past0.2/1/5s moves, Binance past0.3/1/5s moves and fixed current p(1-p)/sqrt(time-left) interactions, current Predict mid/spread/time/phase; add PM past0.2/1/5s moves. Standardize train only; fixed ridge penalty1. No tuning, no settlement labels. Same eligible rows for both models; zero-change forecast also scored.',
        inference='Descriptive sampled-window study. Report per-day stability and shifted-clock null; no confidence intervals or independent-validation claim from incomplete-day sampling.',
        contracts='Historical Polymarket metadata names BTC/USD TWAP. Predict registry and current docs identify its own reference round. Same UTC window is not identical strike/payoff; phases retained separately.',
        protections='No Nov4..6,Nov9..13 quote or outcome payloads; no live/frozen changes; single light process.')
    write(DEST/'declaration.json',d);write(REVIEW/'declaration.json',d)
    print('DECLARED',len(population),'scheduled windows;',len(meta),'known-before-start PM identities',flush=True)


def check():
    d=json.loads((DEST/'declaration.json').read_text())
    for name,h in d['source_sha'].items():
        if sha(HERE.parents[1]/name)!=h:raise ValueError('source changed: '+name)
    return d


def pm_events(obj,token,condition):
    out=[]
    for e in obj if isinstance(obj,list) else [obj]:
        if not isinstance(e,dict):raise ValueError('invalid PM event')
        if e.get('market') and e['market']!=condition:raise ValueError('wrong PM condition')
        if e.get('event_type')=='book' and str(e.get('asset_id'))==token:
            bids=[float(x['price'])*100 for x in e['bids'] if float(x['size'])>0]
            asks=[float(x['price'])*100 for x in e['asks'] if float(x['size'])>0]
            out.append((int(e['timestamp'])*10**6,max(bids,default=np.nan),min(asks,default=np.nan)))
        elif e.get('event_type')=='price_change':
            for x in e['price_changes']:
                if str(x['asset_id'])==token:
                    bid,ask=float(x['best_bid'])*100,float(x['best_ask'])*100
                    out.append((int(e['timestamp'])*10**6,np.nan if bid==0 else bid,np.nan if ask==100 else ask))
    return out


def read_pm(path,meta,ws):
    rows=[];counts=collections.Counter();last=0
    with gzip.open(path,'rt') as f:
        for line in f:
            if not line.strip():counts['blank']+=1;continue
            stamp,raw=line.split('\t',1);recv=int(stamp)
            if recv<last:raise ValueError('PM receipt regression')
            last=recv
            if recv>(ws+300)*10**9:break
            if recv<(ws-5)*10**9:continue
            counts['messages']+=1
            for source,bid,ask in pm_events(json.loads(raw),meta['up_token'],meta['conditionId']):
                rows.append((recv-ws*10**9,source-ws*10**9,bid,ask))
    return np.array(rows,dtype=float).reshape(-1,4),dict(counts)


def sample(rows,times,clock=0):
    """Return latest state, including invalid states, never backfill past a bad book."""
    if rows.ndim!=2 or rows.shape[1]!=4 or clock not in (0,1):raise ValueError('invalid book table')
    if len(rows)==0:return np.full((len(times),3),np.nan)
    if not np.isfinite(rows[:,:2]).all():raise ValueError('invalid clock')
    if (np.diff(rows[:,0])<0).any():raise ValueError('receipt regression')
    a=rows[np.argsort(rows[:,clock],kind='stable')];i=np.searchsorted(a[:,clock],times,side='right')-1
    z=a[np.clip(i,0,len(a)-1)];good=(i>=0)&(z[:,2]>0)&(z[:,3]<100)&(z[:,2]<z[:,3])
    age=(times-z[:,clock])/1e6
    return np.column_stack([np.where(good,(z[:,2]+z[:,3])/2,np.nan),np.where(good,z[:,3]-z[:,2],np.nan),np.where(i>=0,age,np.nan)])


def read_bn(path,ws,times):
    chunks=[]
    for a in pd.read_csv(path,header=None,usecols=[0,1,4,6],chunksize=100000):
        a=a.apply(pd.to_numeric,errors='raise');last=int(a[0].iloc[-1])
        a=a[(a[0]>=(ws-5)*10**9)&(a[0]<=(ws+300)*10**9)].copy()
        a[0]=a[0].astype('int64')-ws*10**9;a[1]=a[1].astype('int64')*10**6-ws*10**9
        chunks.append(a.to_numpy(dtype=float))
        if last>(ws+300)*10**9:break
    a=np.concatenate(chunks)
    if len(a)==0:return np.full(len(times),np.nan)
    if (np.diff(a[:,0])<0).any():raise ValueError('Binance receipt regression')
    idx=np.searchsorted(a[:,0],times,side='right')-1;z=a[np.clip(idx,0,len(a)-1)]
    age=(times-z[:,0])/1e6;good=(idx>=0)&(z[:,2]>0)&(z[:,3]>z[:,2])&(age<=1000)
    return np.where(good,(z[:,2]+z[:,3])/2,np.nan)


def build():
    d=check();meta=json.loads((DEST/'pm_metadata.json').read_text());statuses=[];paths={}
    (DEST/'grids').mkdir(exist_ok=False)
    for w in d['population']:
        status=dict(day=w['day'],ws=w['ws'],mid=w['mid'],slug=w['slug'],missing_files=w['missing_files'])
        if w['missing_files'] or w['slug'] not in meta:
            statuses.append(dict(status,status='missing_file_or_prestart_metadata'));continue
        for path in w['paths'].values():
            if sha(path)!=d['input_sha'][path]:raise ValueError('input changed')
        ws=w['ws'];rel=np.arange(3001)*.1;times=np.arange(3001,dtype='int64')*100000000
        f=pd.read_parquet(w['paths']['pf'],columns=['mid','recv_ns','t_ms','b1_p','a1_p']);f=f[f.mid==w['mid']].copy()
        f['recv_ns']=f['recv_ns'].astype('int64')-ws*10**9;f['t_ms']=f['t_ms'].astype('int64')*10**6-ws*10**9
        pf=f[['recv_ns','t_ms','b1_p','a1_p']].to_numpy(dtype=float)
        pm,counts=read_pm(w['paths']['pm'],meta[w['slug']],ws)
        if not len(pf) or not len(pm):statuses.append(dict(status,status='empty_book'));continue
        data=dict(rel=rel,bn=read_bn(w['paths']['bn'],ws,times))
        for name,a in [('pf',pf),('pm',pm)]:
            for ci,clock in enumerate(('recv','source')):
                z=sample(a,times,ci)
                for i,k in enumerate(('mid','spread','age')):data[f'{name}_{clock}_{k}']=z[:,i]
            lag=(a[:,0]-a[:,1])/1e6
            status[name+'_latency_ms']=dict(zip(('p01','p50','p99'),np.quantile(lag,[.01,.5,.99]).tolist()))
            status[name+'_source_regressions']=int((np.diff(a[:,1])<0).sum())
            status[name+'_bbo_rows']=len(a)
        out=DEST/'grids'/f"{w['day']}_{w['hour']:02d}.parquet";pd.DataFrame(data).to_parquet(out,index=False);paths[str(out)]=sha(out)
        statuses.append(dict(status,status='built',pm_parser=counts,grid=str(out)))
        print('BUILT',w['day'],w['hour'],flush=True)
    write(DEST/'build.json',dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),declaration_sha=sha(DEST/'declaration.json'),statuses=statuses,grids=paths))


if __name__=='__main__':
    import sys
    {'declare':setup,'build':build}[sys.argv[1]]()
