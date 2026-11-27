"""Passive-policy projection of the frozen market-path loader, with an October control."""
from collections import Counter
import json
import os
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

from signals import BTC, OUT, HERE, DAYS, CONTROL, prev_day
os.environ['PF_RL_DDEC']='2'
import pf_clock_rl as clock

prior = runpy.run_path(str(HERE.parent/'urgency_near_selected_20261127/oct26.py'))
strong, R, U, ref, pilot, common = (prior[k] for k in ('strong','R','U','ref','pilot','common'))


def anchor(directory, reg):
    leg = pd.read_parquet(directory/'legs_timed2.parquet',
                          columns=['mid','exec_s','side','ptick','shares','how','t_ms','t_e2'])
    key = pd.read_parquet(directory/'legs_tk.parquet',columns=['tk']).tk.to_numpy()
    assert len(key)==len(leg)
    group = np.where(key>=0,key,-1-np.arange(len(key)))
    ws_anchor = leg.t_ms.where(leg.how>0).groupby(group).transform('min')
    book_anchor = leg.t_e2.where(leg.how==0).groupby(group).transform('min')
    leg['anc'] = ws_anchor.fillna(book_anchor)
    leg['ws'] = leg.mid.map(reg.ws.where(reg.outcome.isin(('Up','Down'))))*1000.
    base = (leg.anc>=leg.ws-2000)&(leg.anc<=leg.ws+61000)
    off = float(np.median((leg.anc-leg.exec_s*1000)[base]))
    assert np.isfinite(off)
    unt = leg.anc.isna()&(leg.exec_s*1000>=leg.ws-2000)&(leg.exec_s*1000<leg.ws+301000)
    leg.loc[unt,'anc'] = leg.loc[unt,'exec_s'].groupby(group[unt]).transform('min')*1000+off
    return leg[(leg.anc>=leg.ws-2000)&(leg.anc<=leg.ws+301000)],off


def worlds(day, directory, reg, control):
    legs,offset=anchor(directory,reg)
    paths=sorted((directory/'books').glob('part_*.parquet'))
    timings=pd.concat([pd.read_parquet(p,columns=['recv_ns','t_ms']) for p in paths],ignore_index=True)
    shift=clock.day_shift(timings.t_ms,timings.recv_ns)
    del timings
    if control:
        assert offset==R.MM.KP.Tape().offset(day)
        assert shift==json.loads((directory/'clock_rl_d2.json').read_text())['shift_ms']
    print(day,'clock shift',shift,'imputation',offset,flush=True)
    columns=['mid','recv_ns','t_ms']+[f'{s}{i}_{k}' for s in 'ba' for i in range(1,6) for k in 'ps']
    stem=lambda h:(pd.Timestamp(day,tz='UTC')+pd.Timedelta(hours=h)).strftime('%Y%m%d_%H')
    for hour, markets in reg.groupby('hour'):
        names={f'part_{stem(h)}.parquet' for h in (hour-1,hour,hour+1) if h<24}
        pieces=[]
        for p in paths:
            if p.name in names:
                pieces.append(pd.read_parquet(p,columns=columns,filters=[('mid','in',list(markets.index))]))
        books=pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame(columns=columns)
        for m,r in markets.iterrows():
            ws=float(r.ws)*1000
            b=books[(books.mid==m)&(books.t_ms>=ws-30000)&(books.t_ms<=ws+300000)].sort_values('t_ms',kind='stable')
            if r.outcome not in ('Up','Down') or not np.isfinite(r.start_price) or b.empty:
                yield m,'loader_excluded',None,{}
                continue
            if r.decimal_precision!=2:
                yield m,'precision_not_2',None,{}
                continue
            tf=b.t_ms.to_numpy(dtype=float)
            kf=np.maximum.accumulate(clock.pf_known(tf,b.recv_ns.to_numpy(),shift))
            tg=ws+np.arange(0,299800,50)
            prices=[b[[f'{s}{i}_p' for i in range(1,6)]].to_numpy(dtype=int) for s in 'ba']
            W=[]
            for side,s in enumerate('ba'):
                l=legs[(legs.mid==m)&(legs.side==side)]
                o=np.lexsort(((-1 if side==0 else 1)*l.ptick.to_numpy(),l.anc.to_numpy()))
                W.append(dict(ws=ws,tg=tg,tf=tf,kf=kf,jk=np.searchsorted(kf,tg,'right')-1,
                              P5=prices[side],opp=prices[1-side][:,0],
                              S5=b[[f'{s}{i}_s' for i in range(1,6)]].to_numpy(dtype=float),
                              anc=l.anc.to_numpy()[o],ptl=l.ptick.to_numpy()[o],shl=l.shares.to_numpy()[o]))
            path,status,info=R.X.M.market_path(W,ws)
            yield m,status,path,info


def run(day):
    assert day in (CONTROL,*DAYS)
    control=day==CONTROL
    if not control:
        assert (OUT/'control_pass.json').exists()
    strong['fixture']()
    directory=(R.MM.S1 if control else OUT/'step1')/day
    reg=pd.read_parquet(directory/'registry.parquet').set_index('mid',verify_integrity=True)
    reg['hour']=(reg.ws.astype('int64')%86400)//3600
    opening=R.BTC.day_ms(day)
    # Refuse duplicate/off-grid openings rather than quietly scoring a different population.
    expected_slots=set(range(opening//1000,opening//1000+86400,300))
    assert reg.ws.is_unique and set(reg.ws)<=expected_slots
    reg=reg.sort_values('ws')
    fit=U.S.read(R.HERE/'fit_v2.json')['cells']
    bind=U.S.read(ref['S']['L']['E'].REPLICATION/'result.json')
    assert R.sha(R.HERE/'fit_v2.json')==bind['fit_sha'] and R.sha(R.CENSUS)==bind['census_sha']
    if control:
        reference={r['mid']:r for r in U.S.read(R.CENSUS)['rows'] if r['day']==day}
        assert set(reference)==set(reg.index)
        selected=pd.read_csv(HERE.parent/'urgency_near_selected_20261127/markets.csv',float_precision='round_trip').set_index(['mid','policy','arm'])
        first=min(selected.index.get_level_values('mid'))
    sr=json.loads((OUT/f'grid_{day}.json').read_text())
    assert R.sha(BTC/f'btc_{day}.parquet')==sr['grid_sha'] and R.sha(OUT/f'pressure_{day}.parquet')==sr['pressure_sha']
    allowed=(prev_day(day),day)
    with patch.multiple(R.BTC,BTC_DAYS=allowed,PAD_LO_MS=opening-3600000,BTC_END_MS=opening+86400000):
        btc=R.BTC.load_btc(BTC,day,Counter())
    pressure=pd.read_parquet(OUT/f'pressure_{day}.parquet')
    bn=pressure.k_ms.to_numpy(),pressure.mid.to_numpy()
    edges=R.MM.TR.T.P.frozen_edges()
    records=[]
    census=[]
    for m,status,path,info in worlds(day,directory,reg,control):
        if control:
            assert status==reference[m]['status'],(m,status,reference[m]['status'])
        ws=1000.*reg.at[m,'ws']
        rec=dict(day=day,mid=m,opening_ms=ws,status=status)
        census.append(rec)
        if status!='complete': continue
        ident=R.MM.ident_of(day,m,ws,info,path)
        if control: assert ident['tape_id']==reference[m]['tape_id'],m
        rec['tape_id']=ident['tape_id']
        if control and m!=first: continue
        assert np.isfinite(reg.at[m,'end_price']) and reg.at[m,'maker_fee_bps']==0
        with patch.object(R.MM,'bn',lambda:bn):
            flags,z=U.S.signal_data(btc,info,ws,edges)
        left=(ws+300000-info['t_start']-R.BE.STEP*np.arange(len(z)))/1000.
        time,near=strong['curves'](left,z)
        args=(path,info,ident,flags,fit,R.VM.V.payout_of(reg.loc[m]))
        for arm in ('nominal','short'):
            for name,lim in [('original',None),('time_only',time),('near',near)]:
                row=ref['simulate'](arm,*args) if lim is None else pilot['simulate'](arm,args,lim)
                if control and name!='time_only':
                    for k,v in row.items():
                        assert abs(v-selected.at[(m,name,arm),k])<1e-6,(m,k,v)
                records.append(dict(day=day,mid=m,tape_id=ident['tape_id'],arm=arm,policy=name,**row))
    for ws in sorted(expected_slots-set(reg.ws)):
        census.append(dict(day=day,mid=None,opening_ms=ws*1000,status='registry_missing'))
    assert len(census)==288
    print(day,'census',dict(Counter(r['status'] for r in census)),flush=True)
    pd.DataFrame(census).to_csv(HERE/f'census_{day}.csv',index=False)
    if control:
        assert len(records)==6
        (OUT/'control_pass.json').write_text(json.dumps({'day':day,'markets':len(reg),'first_control':int(first),'sha':R.sha(HERE/f'census_{day}.csv')}))
    else:
        frame=pd.DataFrame(records)
        if len(frame):
            base=frame[frame.policy=='original']
            assert len(frame)==6*sum(r['status']=='complete' for r in census)
            for name in ('near','time_only'):
                common['validate'](frame[frame.policy==name],base)
            assert frame.query("arm=='short'").post_request_shares.eq(0).all()
        frame.to_csv(HERE/f'markets_{day}.csv',index=False)


if __name__=='__main__':
    run(sys.argv[1])
