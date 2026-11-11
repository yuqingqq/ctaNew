"""Build neutral generation rows and same-state potential outcomes on consumed dates."""
import collections
import json
import numpy as np
import pandas as pd
from .pf_queue_model_spec import ROOT, OUT, DAYS, STARTS, AGES, CONFIGS, ACTIONS, SIZE, RISK_C, check, sha, write
from .pf_queue_model_sim import Tape, simulate, features, marks, reservation
from .pf_model_spec import SETS
from .pf_io import mem_check


def load_day(day):
    if day not in DAYS:raise ValueError('undeclared/protected day')
    s=ROOT/f'data/predict_fun/research/step1/{day}';s5=ROOT/f'data/predict_fun/research/step5/{day}'
    lo=int(pd.Timestamp(day,tz='UTC').value//10**9);hi=lo+86400
    reg=pd.read_parquet(s/'registry.parquet',filters=[('ws','>=',lo),('ws','<',hi)])
    if reg.ws.duplicated().any() or reg.mid.duplicated().any():raise ValueError('duplicate market identity')
    for x in reg.itertuples():
        u=np.asarray(x.up_index)
        if x.we!=x.ws+300 or u.dtype.kind not in ('i','u') or u.size!=1 or int(u.item())!=1:raise ValueError('market identity')
    ids=reg.mid.astype(int).tolist()
    bf=[('mid','in',ids),('t_ms','>=',lo*1000),('t_ms','<',hi*1000)]
    cols=['mid','t_ms','recv_ns']+[f'{s}{i}_{v}' for s in 'ba' for i in range(1,6) for v in 'ps']
    books=pd.concat([pd.read_parquet(p,columns=cols,filters=bf) for p in sorted((s/'books').glob('*.parquet'))],ignore_index=True)
    books.sort_values(['mid','t_ms','recv_ns'],kind='stable',inplace=True)
    # Read only identity/time columns before filtering quote payloads. Preserve legs_tk row alignment.
    ident=pd.read_parquet(s/'legs_timed2.parquet',columns=['mid','exec_s'])
    mask=ident.mid.isin(ids)&(ident.exec_s>=lo)&(ident.exec_s<hi)
    legs=pd.read_parquet(s/'legs_timed2.parquet',columns=['mid','exec_s','side','ptick','shares','t_ms','how','t_e2'],
        filters=[('mid','in',ids),('exec_s','>=',lo),('exec_s','<',hi)])
    if not np.array_equal(legs[['mid','exec_s']].values,ident.loc[mask,['mid','exec_s']].values):raise ValueError('filtered leg identity mismatch')
    tk=pd.read_parquet(s/'legs_tk.parquet',columns=['tk']).tk.to_numpy()
    if len(tk)!=len(ident):raise ValueError('taker identities not aligned')
    tk=tk[mask.to_numpy()];tk=np.where(tk>=0,tk,-1-np.arange(len(tk)))
    a=pd.Series(np.where(legs.how>0,legs.t_ms,np.nan)).groupby(tk,sort=False).transform('min').to_numpy()
    b=pd.Series(np.where(legs.how==0,legs.t_e2,np.nan)).groupby(tk,sort=False).transform('min').to_numpy()
    legs['anchor']=np.where(np.isfinite(a),a,b)
    untimed=int(legs.anchor.isna().sum());legs=legs[legs.anchor.notna()]
    r=pd.read_parquet(s5/'rows_rl_d2_pk.parquet',columns=list(dict.fromkeys(['mid','tau']+SETS['ALL'])),
                      filters=[('mid','in',ids),('tau','>=',lo*1000),('tau','<',hi*1000)])
    v=pd.read_parquet(s5/'preds_t60k.parquet',columns=['mid','tau','v_bid','v_ask'],
                      filters=[('mid','in',ids),('tau','>=',lo*1000),('tau','<',hi*1000)])
    if not np.array_equal(r[['mid','tau']].values,v[['mid','tau']].values):raise ValueError('feature/benchmark identity mismatch')
    r['v_bid']=v.v_bid;r['v_ask']=v.v_ask;del v,ident,tk
    clock=json.loads((s/'clock_rl_d2.json').read_text())
    if clock['d_dec']!=2:raise ValueError('clock era mismatch')
    bg={m:g for m,g in books.groupby('mid',sort=False)};lg={m:g for m,g in legs.groupby('mid',sort=False)}
    rg={m:g for m,g in r.groupby('mid',sort=False)}
    mem_check('queue data loaded')
    for row in reg.sort_values('ws').itertuples():
        b=bg.get(row.mid);r=rg.get(row.mid);l=lg.get(row.mid)
        if b is None or r is None:yield row,None,dict(untimed_day=untimed);continue
        ws=int(row.ws)*1000
        tf=b.t_ms.to_numpy(dtype='int64')-ws
        receipt=(b.recv_ns.to_numpy(dtype='int64')-ws*10**6)/1e6
        kn=np.maximum.accumulate(np.maximum(tf+13.2,receipt-clock['shift_ms'])+2)
        if l is None:
            tt=np.empty(0);ts=tp=tq=np.empty(0)
        else:
            tt=l.anchor.to_numpy()-ws;ts=l.side.to_numpy();tp=l.ptick.to_numpy();tq=l.shares.to_numpy()
            order=np.lexsort((np.where(ts==0,-tp,tp),ts,tt));tt,ts,tp,tq=[a[order] for a in (tt,ts,tp,tq)]
        tape=Tape(tf,kn,b[[f'b{i}_p' for i in range(1,6)]].to_numpy(),b[[f'a{i}_p' for i in range(1,6)]].to_numpy(),
            b[[f'b{i}_s' for i in range(1,6)]].to_numpy(),b[[f'a{i}_s' for i in range(1,6)]].to_numpy(),
            tt,ts,tp,tq,r.tau.to_numpy()-ws,r[SETS['ALL']].to_numpy(dtype=np.float32),r[['v_bid','v_ask']].to_numpy())
        tape.validate();yield row,tape,dict(untimed_day=untimed)


def admit(net,side,start=0):
    s=1 if side==0 else -1
    return not (s*net>=10-1e-9 or s*net+SIZE>25+1e-9 or (start>=180 and s*net+SIZE>1e-9))


def build(day):
    d=check()
    if day not in DAYS:raise ValueError('protected day')
    path=OUT/f'{day}.parquet';fp=OUT/f'{day}_fills.parquet'
    if path.exists() or fp.exists():raise ValueError('refuse overwrite')
    for p,h in d['input_sha'].items():
        if ('/'+day+'/') in p and sha(p)!=h:raise ValueError('input changed: '+p)
    rows=[];fills=[];counts=collections.Counter();windows=[];untimed=0
    for reg,tape,stat in load_day(day):
        windows.append(int(reg.ws));untimed=stat['untimed_day']
        neutral_net={name:0. for name in CONFIGS}
        for ep,start in enumerate(STARTS):
            send=start*1000.
            for name,c in CONFIGS.items():
                pair=[];i=tape.index(send,True) if tape is not None else -1
                initial_net=neutral_net[name]
                for side in (0,1):
                    age=AGES[(int(reg.mid)+ep+side)%len(AGES)];decision=send+c['place']+age
                    rid=f'{day}:{int(reg.mid)}:{start}:{side}:{name}'
                    row=dict(id=rid,day=day,mid=int(reg.mid),ws=int(reg.ws),episode=start,side=side,config=name,
                             decision=decision,status='missing_book',decision_status='missing_book',initial_net=initial_net,mark_t=min(240000.,decision+30000.),
                             outcome=float(reg.outcome=='Up') if reg.outcome in ('Up','Down') else np.nan)
                    row['stop_mid']=tape.midpoint(240000.) if tape is not None else np.nan
                    if pd.isna(reg.maker_fee_bps) or reg.maker_fee_bps!=0:raise ValueError('missing or nonzero maker fee')
                    if tape is None or not tape.book_ok(i):pair.append((row,None,None));continue
                    p=int(tape.bp[i,0] if side==0 else tape.ap[i,0]);row['price']=p
                    if (p if side==0 else 100-p)*SIZE<100:
                        row['status']=row['decision_status']='minimum_notional';pair.append((row,None,None));continue
                    stop=tape.hard_stop(side,p,send,send+5000.)
                    if stop<=send:
                        row['status']=row['decision_status']='initial_hard_guard';pair.append((row,None,None));continue
                    potentials={a:simulate(tape,side,p,send,decision,c,a,stop=stop) for a in ACTIONS}
                    neutral=potentials['keep'];base_prefix=[f for f in neutral['fills'] if f[0]<decision+c['cancel']]
                    if any([f for f in x['fills'] if f[0]<decision+c['cancel']]!=base_prefix for x in potentials.values()):raise ValueError('unavoidable fill prefix changed')
                    x,status=features(tape,side,p,send,decision,c,neutral,initial_net)
                    row['status']=row['decision_status']=status
                    if x is not None:row.update(x)
                    j=tape.row(decision);row['legacy_v']=float(tape.v[j,side]) if j>=0 else np.nan
                    metric={a:marks(tape,z,row['mark_t'],row['outcome']) for a,z in potentials.items()}
                    if any(not np.isfinite(v['value']) for v in metric.values()):row['status']='missing_common_mark'
                    for action,stats in metric.items():
                        row.update({action+'_'+k:v for k,v in stats.items()})
                        for a in AGES:row[f'{action}_reserved_{int(a)}']=reservation(potentials[action],send+c['place']+a,c)
                        for fill_index,(t,q,through,generation,anchor) in enumerate(potentials[action]['fills']):
                            fills.append(dict(id=rid,action=action,fill_index=fill_index,t=t,q=q,through=through,generation=generation,anchor=anchor,price=p,side=side))
                    pair.append((row,potentials,metric))
                # Reference path admits by known inventory and holds; admission itself is not outcome selected.
                admitted=[z[2] is not None and admit(initial_net,side,start)
                          for side,z in enumerate(pair)]
                for side,(row,potentials,metric) in enumerate(pair):
                    row['reference_admitted']=bool(admitted[side])
                    if metric is not None:
                        other=pair[1-side];s=1 if side==0 else -1
                        other_end=other[2]['keep']['net'] if admitted[1-side] else 0.
                        other_known=sum(q*(1 if 1-side==0 else -1) for t,q,_,_,_ in other[1]['keep']['fills'] if t+c['notice']<=row['decision']) if admitted[1-side] else 0.
                        own_known=sum(q*s for t,q,_,_,_ in potentials['keep']['fills'] if t+c['notice']<=row['decision'])
                        row['inventory']=initial_net+own_known+other_known
                        row['other_reserved']=reservation(other[1]['keep'],row['decision'],c) if admitted[1-side] else 0.
                        for a,m in metric.items():
                            row[a+'_utility']=m['value']-RISK_C*((initial_net+other_end+m['net'])**2-(initial_net+other_end)**2)
                    counts[row['status']]+=1;rows.append(row)
                neutral_net[name]+=sum(pair[s][2]['keep']['net'] for s in (0,1) if admitted[s])
                if abs(neutral_net[name])>25+1e-7:raise ValueError('reference capacity violated')
        if len(windows)%48==0:print('BUILD',day,len(windows),'rss_mib',mem_check('queue build'),flush=True)
    day0=int(pd.Timestamp(day,tz='UTC').timestamp());missing=sorted(set(range(day0,day0+86400,300))-set(windows))
    a=pd.DataFrame(rows);f=pd.DataFrame(fills)
    if a.id.duplicated().any():raise ValueError('duplicate action generation')
    a.to_parquet(path,index=False);f.to_parquet(fp,index=False)
    write(OUT/f'{day}_build.json',dict(day=day,windows=len(windows),missing_registry_windows=missing,rows=len(a),statuses=dict(counts),
        untimed_legs_excluded=untimed,outcomes_missing=int(a.outcome.isna().sum()),data_sha=sha(path),fills_sha=sha(fp)))
    print('BUILT',day,len(a),dict(counts),flush=True)


if __name__=='__main__':
    import sys
    build(sys.argv[1])
