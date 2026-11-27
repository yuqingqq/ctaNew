"""Feed the archived earlier dates through the unchanged event-pressure experiment."""
import gzip
import hashlib
import json
import runpy
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REV = HERE.parent
PRIOR = REV / 'event_pressure_cancel_20261127'
E = SimpleNamespace(**runpy.run_path(str(PRIOR/'run.py')))
H, R = E.H, E.R
OD = H.ref['OD']
EARLY = ('20261023','20261024','20261025','20261026','20261027','20261031','20261101','20261102','20261103')
FAR = REV/'urgency_far_extension_20261127'
STRONG = REV/'urgency_early_strong_20261127'
SELECTED = REV/'urgency_near_selected_20261127'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def baseline(day):
    if day in OD.SEG_OF:
        folder,policy=FAR,'previous_btc'
    elif day=='20261026':
        folder,policy=SELECTED,'near'
    else:
        folder,policy=STRONG,'btc_time'
    path=folder/('markets.csv' if folder==SELECTED else f'markets_{day}.csv')
    if folder!=SELECTED:
        assert sha(path)==read(folder/'result.json')['sha256'][path.name]
    else:
        assert sha(path)==OCT26_HASH
    rows=pd.read_csv(path,float_precision='round_trip').query('arm=="nominal" and policy==@policy')
    assert rows.day.eq(int(day)).all() and rows.mid.is_unique
    return rows.set_index('mid')


def census(day):
    if day in OD.SEG_OF:
        path=OD.OUT3/'summary.json'
        assert sha(path)=='f1b45f6555c707654acf337359e98d900b67cf4c1277312dfdd4d82c0b46c460'
        rows=read(path)['census']
    else:
        path=R.CENSUS
        bind=read(H.ref['S']['L']['E'].REPLICATION/'result.json')
        assert sha(path)==bind['census_sha']
        rows=read(path)['rows']
    return pd.DataFrame([r for r in rows if r['day']==day]).set_index('mid')


def btc_for(day):
    if day in OD.SEG_OF:
        grid=SimpleNamespace(**runpy.run_path(str(FAR/'grid.py')))
        days=[d for d in (H.prev_day(day),day) if d in grid.DAYS]
        for d in days:
            assert sha(grid.OUT/f'btc_{d}.parquet')==read(FAR/f'grid_{d}.json')['sha256']
        with patch.multiple(R.BTC,BTC_DAYS=tuple(days),PAD_LO_MS=R.BTC.day_ms(days[0]),BTC_END_MS=R.BTC.day_ms(day)+86400000):
            return R.BTC.load_btc(grid.OUT,day,Counter())
    folder=R.X.DATA/'maker_trace_v2_20261123/btc'
    for line in (H.strong['PRIOR']/'btc.sha256').read_text().splitlines():
        expected,name=line.split()
        assert sha(folder/Path(name).name)==expected
    return R.BTC.load_btc(folder,day,Counter())


def feed(day):
    if day in OD.SEG_OF:
        record=read(OD.RCPT/'bn_events.json')
        assert sha(OD.OUT3/'bn_changes.parquet')==record['out_sha256']
        k,mid=OD.seg_arrays(OD.SEG_OF[day])
    else:
        record=read(REV/'take_signal_20261125/bn_events.json')
        assert sha(R.MM.BN)==record['out_sha256']
        k,mid=R.MM.bn()
    opening=R.BTC.day_ms(day)
    next_day=(pd.Timestamp(day)+pd.Timedelta(days=1)).strftime('%Y%m%d')
    stamps=[]
    for stem,rec in record['files'].items():
        if not (stem.startswith(day+'_') or stem==H.prev_day(day)+'_23' or stem==next_day+'_00'):
            continue
        p=E.source.Roots().bn/f'{stem}.csv.gz'
        assert sha(p)==rec['sha256']
        with gzip.open(p,'rb') as f:
            while lines:=f.readlines(4<<20):
                arr,_=E.source.parse_bn(b''.join(lines))
                ok=((arr['recv_ns']>=1787579334881534478)&np.isfinite(arr['ask'])&
                    (arr['bid']>0)&(arr['ask']>arr['bid']))
                ts=arr['E_ms']+R.MM.B_IN-opening
                ok &= (ts>=0)&(ts<86400000)
                if ok.any():
                    stamps.append(np.unique(ts[ok]).astype(np.int32))
    updates=np.unique(np.concatenate(stamps)).astype(np.int64)+opening
    # Preserve the source segment's history, including missing history at its first tick.
    left=max(0,k.searchsorted(opening-1000)-1)
    right=k.searchsorted(opening+86400000)
    k,mid=k[left:right],mid[left:right]
    clock=k+R.MM.B_IN
    in_day=clock[(clock>=opening)&(clock<opening+86400000)]
    ix=updates.searchsorted(in_day)
    assert (ix<len(updates)).all() and np.array_equal(updates[ix],in_day)
    return updates,np.r_[-1.e16,clock],np.r_[np.nan,mid],k,mid


def run(day):
    assert day in EARLY
    archive=baseline(day)
    population=census(day)
    complete=population.query("status=='complete'")
    assert len(population)==288 and set(archive.index)==set(complete.index)
    if 'tape_id' in complete:
        assert archive.tape_id.to_dict()==complete.tape_id.to_dict()
    directory=R.MM.S1/day
    reg=pd.read_parquet(directory/'registry.parquet').set_index('mid')
    reg['hour']=(reg.ws.astype('int64')%86400)//3600
    assert reg.index.is_unique and set(reg.index)==set(population.index)
    btc=btc_for(day)
    updates,clock,mid,k,values=feed(day)
    edges=R.MM.TR.T.P.frozen_edges()
    fit_path=R.HERE/'fit_v2.json'
    assert sha(fit_path)==read(H.ref['S']['L']['E'].REPLICATION/'result.json')['fit_sha']
    fit=read(fit_path)['cells']
    records=[]
    seen=set()
    for m,status,path,info in H.worlds(day,directory,reg.sort_values('ws'),False):
        if day in OD.SEG_OF and reg.at[m,'ws']*1000+300000>=OD.FWD_MS:
            status='boundary_censored_forward_tail'
        assert m not in seen and status==population.at[m,'status'],(day,m,status,population.at[m,'status'])
        seen.add(m)
        if status!='complete':
            continue
        ws=1000.*reg.at[m,'ws']
        ident=R.MM.ident_of(day,m,ws,info,path)
        assert ident['tape_id']==archive.at[m,'tape_id'],(day,m,'tape identity')
        with patch.object(R.MM,'bn',lambda:(k,values)):
            flags,z=H.U.S.signal_data(btc,info,ws,edges)
        start,end=info['t_start'],info['end_rel']
        _,limits=H.strong['curves']((ws+300000-start-E.STEP*np.arange(len(z)))/1000,z)
        payout=R.VM.V.payout_of(reg.loc[m])
        path=dict(path,frames=np.asarray(path['frames']),trades=np.asarray(path['trades']))
        args=(path,info,ident,flags,fit,payout)
        times=E.events(updates,clock,mid,start,end,edges,flags)
        control,old=E.simulate(args,limits,[])
        metadata=['day','arm','policy','tape_id','active_ticks','missing_ticks','tighter_ticks','far_tighter_ticks']
        for col in archive.columns.difference(metadata):
            assert abs(control[col]-archive.at[m,col])<1e-6,(day,m,col,control[col],archive.at[m,col])
        candidate,new=E.simulate(args,limits,times)
        a,b=E.exposures(old,path),E.exposures(new,path)
        row=dict(day=day,mid=m,tape_id=ident['tape_id'])
        for label,metrics in [('grid',control),('event',candidate),('lost',E.valued(a-b,path,end,payout)),
                              ('added',E.valued(b-a,path,end,payout)),('grid_marks',E.valued(a,path,end,payout)),
                              ('event_marks',E.valued(b,path,end,payout))]:
            row.update({f'{label}_{key}':v for key,v in metrics.items()})
        records.append(row)
    assert seen==set(population.index) and {r['mid'] for r in records}==set(archive.index)
    f=pd.DataFrame(records)
    runpy.run_path(str(PRIOR/'score.py'))['validate'](f)
    f.to_csv(HERE/f'paired_{day}.csv',index=False)
    print(day,len(f),'verified market pairs',flush=True)


OCT26_HASH='1eaaec9283a9c8ff5dfea16b982d2a16687056437a32a6587eb93ad617a66c42'
if __name__=='__main__':
    run(sys.argv[1])
