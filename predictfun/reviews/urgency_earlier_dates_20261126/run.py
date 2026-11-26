"""Retrospective replay on all three previously unscored dates in the prepared unreserved census."""
import csv
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'urgency_later_dates_20261126'))
import replay as L
from verify import verify

R,U,OD=L.R,L.U,L.OD
DAYS=('20261023','20261024','20261025')


def check_csv(path, summary):
    with path.open() as f:
        rows=list(csv.DictReader(f))
    verify(rows,summary)
    rows[0]=dict(rows[0],settle=float(rows[0]['settle'])+1)
    try:
        verify(rows,summary)
    except ValueError:
        return
    raise AssertionError('corrupt settlement accepted')


def run():
    prev=U.S.read(L.E.REPLICATION/'result.json')
    fit=U.S.read(R.HERE/'fit_v2.json')
    assert R.sha(R.HERE/'fit_v2.json')==prev['fit_sha'] and fit['days']==list(DAYS[1:])
    assert R.sha(R.CENSUS)==prev['census_sha']==fit['census_sha']
    assert R.sha(L.E.PRIOR/'urgency.py')==prev['controller_sha']
    assert R.sha(Path(L.E.__file__))=='21b556a98b8c1826aa1ddee75c4554f72bda218cda746ccaa7aa8de28423bd97'
    assert R.sha(OD.OUT3/'bn_changes.parquet')==U.S.read(OD.RCPT/'bn_events.json')['out_sha256']
    assert R.sha(OD.OUT3/'summary.json')=='f1b45f6555c707654acf337359e98d900b67cf4c1277312dfdd4d82c0b46c460'
    inputs={tuple(f.removesuffix('.json').split('_')) for f in fit['inputs']}
    used=set()
    archive=U.S.read(OD.OUT3/'summary.json')
    old=U.S.read(R.CENSUS)['rows']
    src=R.MM.KP.Tape()
    edges=R.MM.TR.T.P.frozen_edges()
    daily={}
    # Positive parity on each cache family: earliest complete Oct 26 and Oct 31 market.
    for day in ('20261026','20261031',*DAYS):
        cache3=day in ('20261023','20261031')
        census=[r for r in (archive['census'] if cache3 else old) if r['day']==day]
        ids={r['mid'] for r in census if r['status']=='complete'}
        reg=R.MM.registry(day).set_index('mid')
        assert len(census)==len(reg)==288 and set(reg.index)=={r['mid'] for r in census}
        if cache3:
            records={r['mid']:r['sha256'] for r in archive['rows'] if r['day']==day and r['setting']=='STRICT58'}
            assert ids==set(records)
        else:
            tape_ids={r['mid']:r['tape_id'] for r in census if r['status']=='complete'}
        if day not in DAYS:
            m=min(ids,key=lambda m:(reg.loc[m,'ws'],m))
            ids={m}
            known=pd.read_csv((L.E.PRIOR/'markets.csv') if day=='20261026' else L.HERE/f'markets_{day}.csv',
                              float_precision='round_trip').set_index(['mid','arm'])
        rg=pd.read_parquet(R.MM.S1/day/'registry.parquet').set_index('mid')
        off=src.offset(day)
        seen=set()
        rows=[]
        for hour,g in reg.groupby('hour'):
            batch=sorted(ids & set(g.index))
            if not batch:
                continue
            for m,status,path,info in R.MM.worlds(day,batch,src,off,reg):
                assert status=='complete' and m in batch and m not in seen
                seen.add(m)
                ws=1000.*reg.at[m,'ws']
                ident=R.MM.ident_of(day,m,ws,info,path)
                if cache3:
                    ledger=OD.OUT3/f'{m}_STRICT58.json.gz'
                    assert R.sha(ledger)==records[m]
                    assert json.loads(gzip.decompress(ledger.read_bytes()))['ident']==ident
                    seg=OD.SEG_OF[day]
                    assert OD.SEG_BOUNDS[seg][0]<=ws<ws+300000<OD.SEG_BOUNDS[seg][1]
                    cache=OD.make_cache3(info['t_start'],info['end_rel'],seg)
                else:
                    assert tape_ids[m]==ident['tape_id']
                    cache=R.MM.make_cache(info['t_start'],info['end_rel'])
                payout=R.VM.V.payout_of(rg.loc[m])
                metrics=L.replay_market(path,info,ident,R.triggers(cache,edges),fit['cells'],payout)
                key=day,str(m)
                if day in DAYS and key in inputs:
                    used.add(key)
                for arm,r in metrics.items():
                    if day not in DAYS:
                        k='reference' if arm=='original' and day=='20261026' else arm
                        assert all(abs(v-known.loc[(m,k),f])<1e-6 for f,v in r.items())
                    else:
                        rows.append(dict(day=day,mid=m,tape_id=ident['tape_id'],arm=arm,capacity_fit_market=key in inputs,**r))
            if hour%6==0:
                print(day,hour,len(seen),flush=True)
        assert ids==seen
        if day in DAYS:
            frame=pd.DataFrame(rows)
            result=L.daily_summary(frame)
            result['coverage']=dict(Counter(r['status'] for r in census))
            result['capacity_fit_markets']=int(frame.query("arm=='urgency'").capacity_fit_market.sum())
            result['fit_overlap_breakdown']={str(flag):L.daily_summary(group) for flag,group in frame.groupby('capacity_fit_market')}
            path=HERE/f'markets_{day}.csv'
            frame.to_csv(path,index=False,mode='x')
            check_csv(path,result)
            daily[day]=result
        print('reconciled',day,len(seen),flush=True)
    assert used==inputs
    with (HERE/'result.json').open('x') as f:
        json.dump(dict(days=daily,direct_fit_overlap=len(used)),f,indent=2,allow_nan=False)
        f.write('\n')


if __name__=='__main__':
    run()
