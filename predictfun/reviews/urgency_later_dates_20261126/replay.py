"""Unchanged urgency controller on the four later dates of the existing v3 census."""
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'early_add_limit_20261126'))
import replay_v2 as E

U, R = E.U, E.R
OD = R.VM.V.F.OD
DAYS = ('20261031','20261101','20261102','20261103')
COUNTS = dict(zip(DAYS,(102,88,72,76)))
FIELDS = ('settle','turnover','residual','closed','end_pos','abs_end','shares','posts','all_pulls')


def daily_summary(frame):
    by_arm = {}
    for name,g in frame.groupby('arm'):
        assert g.mid.is_unique and g[list(FIELDS)].notna().all().all()
        assert ((g.turnover+g.residual-g.settle).abs()<1e-6).all()
        assert ((2*g.closed+g.abs_end-g.shares).abs()<1e-6).all()
        by_arm[name] = dict(markets=len(g),settle=float(g.settle.sum()),turnover=float(g.turnover.sum()),
                            residual=float(g.residual.sum()),matched=float(g.closed.sum()),
                            cents_per_matched=float(100*g.turnover.sum()/g.closed.sum()),
                            mean_abs_end=float(g.abs_end.mean()),p90_end=float(g.abs_end.quantile(.9)),
                            max_end=float(g.abs_end.max()),positive=int((g.settle>1e-6).sum()))
    paired=frame.pivot(index='mid',columns='arm',values='settle')
    assert paired.notna().all().all()
    delta=paired.urgency-paired.original
    gate=U.P.gate(by_arm['original'],by_arm['urgency'])
    return dict(arms=by_arm,gate=gate,passed=all(gate.values()),delta=float(delta.sum()),
                delta_without_best=float(delta.sum()-delta.max()),
                better=int((delta>1e-6).sum()),worse=int((delta< -1e-6).sum()))


def replay_market(path,info,ident,flags,model,payout):
    assert payout in (0,.5,1)
    args=path,info,ident,flags,model,payout
    original=U.D.simulate(*args,False)
    urgency=E.simulate(*args,False)
    return {name:{k:r[k] for k in FIELDS} for name,r in [('original',original),('urgency',urgency)]}


def run():
    previous=U.S.read(E.REPLICATION/'result.json')
    assert R.sha(E.PRIOR/'urgency.py')==previous['controller_sha']
    assert R.sha(R.HERE/'fit_v2.json')==previous['fit_sha']
    assert R.sha(Path(E.__file__))=='21b556a98b8c1826aa1ddee75c4554f72bda218cda746ccaa7aa8de28423bd97'
    assert R.sha(Path(OD.__file__))=='89420960a26c6fb601f9c76b1ce3f3def6f9a9e9532d211dfb7cdac14b18f334'
    assert R.sha(OD.OUT3/'bn_changes.parquet')==U.S.read(OD.RCPT/'bn_events.json')['out_sha256']
    assert R.sha(OD.OUT3/'summary.json')=='f1b45f6555c707654acf337359e98d900b67cf4c1277312dfdd4d82c0b46c460'
    v3=U.S.read(OD.OUT3/'summary.json')
    fit=U.S.read(R.HERE/'fit_v2.json')['cells']
    edges=R.MM.TR.T.P.frozen_edges()
    src=R.MM.KP.Tape()
    prior_day='20261027'
    assert R.sha(E.REPLICATION/'markets.csv')==E.CONTROL_HASHES[prior_day][0]
    old=pd.read_csv(E.REPLICATION/'markets.csv',float_precision='round_trip').set_index(['mid','arm'])
    day_results={}
    for day in (prior_day,*DAYS):
        reg=R.MM.registry(day).set_index('mid')
        rg=pd.read_parquet(R.MM.S1/day/'registry.parquet').set_index('mid')
        off=src.offset(day)
        if day==prior_day:
            market=min(old.index.get_level_values('mid'))
            assert R.sha(R.CENSUS)==previous['census_sha']
            ids={market}
            old_id=next(r['tape_id'] for r in U.S.read(R.CENSUS)['rows'] if r['day']==day and r['mid']==market)
        else:
            census=[r for r in v3['census'] if r['day']==day]
            assert len(census)==len(reg)==len(set(r['mid'] for r in census))==288
            assert set(reg.index)=={r['mid'] for r in census}
            ids={r['mid'] for r in census if r['status']=='complete'}
            archive={r['mid']:r['sha256'] for r in v3['rows'] if r['day']==day and r['setting']=='STRICT58'}
            assert ids==set(archive) and len(ids)==COUNTS[day]
        rows=[]
        seen=set()
        for hour,g in reg.groupby('hour'):
            mids=sorted(ids & set(g.index))
            if not mids:
                continue
            for m,status,path,info in R.MM.worlds(day,mids,src,off,reg):
                assert status=='complete' and m not in seen and m in mids
                seen.add(m)
                ws=1000.*reg.at[m,'ws']
                ident=R.MM.ident_of(day,m,ws,info,path)
                if day==prior_day:
                    assert ident['tape_id']==old_id
                    cache=R.MM.make_cache(info['t_start'],info['end_rel'])
                else:
                    assert OD.SEG_OF[day]=='B' and OD.SEG_BOUNDS['B'][0]<=ws and ws+300000<OD.FWD_MS
                    ledger=OD.OUT3/f'{m}_STRICT58.json.gz'
                    assert R.sha(ledger)==archive[m]
                    saved=json.loads(gzip.decompress(ledger.read_bytes()))
                    assert ident==saved['ident']
                    cache=OD.make_cache3(info['t_start'],info['end_rel'],'B')
                flags=R.triggers(cache,edges)
                metrics=replay_market(path,info,ident,flags,fit,R.VM.V.payout_of(rg.loc[m]))
                for arm,r in metrics.items():
                    if day==prior_day:
                        v=old.loc[(m,'reference' if arm=='original' else arm)]
                        assert all(abs(float(v[k])-q)<1e-6 for k,q in r.items()),(m,arm)
                    else:
                        rows.append(dict(day=day,mid=m,arm=arm,tape_id=ident['tape_id'],**r))
            if hour%4==0:
                print(day,'hour',hour,'completed',len(seen),flush=True)
        assert seen==ids
        if day!=prior_day:
            frame=pd.DataFrame(rows)
            summary=daily_summary(frame)
            summary['coverage']=dict(Counter(r['status'] for r in census))
            frame.to_csv(HERE/f'markets_{day}.csv',index=False,mode='x')
            day_results[day]=summary
        print(day,'complete',len(seen),flush=True)
    all_rows=pd.concat([pd.read_csv(HERE/f'markets_{d}.csv',float_precision='round_trip') for d in DAYS])
    result=dict(days=day_results,pooled=daily_summary(all_rows),
                positive_days=sum(day_results[d]['gate']['positive_settlement'] for d in DAYS),
                passed_days=sum(day_results[d]['passed'] for d in DAYS))
    with (HERE/'result.json').open('x') as f:
        json.dump(result,f,allow_nan=False,indent=2)
        f.write('\n')


if __name__=='__main__':
    run()
