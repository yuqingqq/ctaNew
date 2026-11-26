"""Paired, cap-scaled passive unwind on the already scored Tokyo backtest markets."""
import json
import math
import runpy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'urgency_cap150_eight_dates_20261126'
ref = runpy.run_path(str(BASE / 'run.py'))
R, U, OD = (ref[k] for k in ('R', 'U', 'OD'))
original_curve = U.S.curve
KEY = ['day', 'mid', 'tape_id', 'arm']


def taper(left, ratio=1.):
    original = original_curve(left, ratio)
    return original if left >= 150 else 0.15 * original


def baseline(day):
    path = BASE / f'markets_{day}.csv'
    expected = json.loads((BASE / 'verification.json').read_text())['sha256'][path.name]
    assert R.sha(path) == expected
    return pd.read_csv(path, float_precision='round_trip', dtype={'day':str})


def upper_decile(values):
    return float(values.nlargest(math.ceil(len(values) / 10)).mean())


def metrics(rows):
    return dict(markets=len(rows), mean_end=rows.abs_end.mean(), tail_end=upper_decile(rows.abs_end),
                at_cap=int((rows.abs_end >= 150-1e-6).sum()), mean_peak=rows.peak_position.mean(),
                max_peak=rows.peak_position.max(), turnover=rows.turnover.sum(), residual=rows.residual.sum(),
                settle=rows.settle.sum(), tail_loss=upper_decile(-rows.settle),
                positive_days=int((rows.groupby('day').settle.sum() > 0).sum()),
                closed=rows.closed.sum(), shares=rows.shares.sum())


def decision(old, new):
    less_inventory = new['mean_end'] < old['mean_end'] and new['tail_end'] < old['tail_end']
    passes = (less_inventory and new['tail_loss'] <= old['tail_loss'] and
              new['settle'] >= old['settle'] and new['positive_days'] >= old['positive_days'])
    return 'passes' if passes else ('risk_tradeoff' if less_inventory else 'no_inventory_improvement')


def validate(rows, old):
    assert len(rows) == len(old) and not rows.duplicated(KEY).any()
    assert set(map(tuple, rows[KEY].values)) == set(map(tuple, old[KEY].values))
    for a in (old, rows):
        assert np.isfinite(a[['abs_end','settle','residual','turnover','shares','closed','peak_position']]).all().all()
        assert (a.abs_end >= 0).all() and (a.peak_position >= a.abs_end-1e-6).all()
        assert (a.peak_position <= 150+1e-6).all()
        assert np.allclose(a.turnover+a.residual,a.settle,atol=1e-6,rtol=0)
        assert np.allclose(2*a.closed+a.abs_end,a.shares,atol=1e-6,rtol=0)


def score():
    tables = []
    for day in ref['DAYS']:
        old = baseline(day)
        new = pd.read_csv(HERE / f'markets_{day}.csv',float_precision='round_trip',dtype={'day':str})
        validate(new, old)
        tables.extend([old.assign(policy='current'), new.assign(policy='taper')])
    all_rows = pd.concat(tables)
    out = []
    for (arm,policy),group in all_rows.groupby(['arm','policy']):
        for day,g in [('pooled',group), *group.groupby('day')]:
            out.append(dict(arm=arm,policy=policy,day=day,**metrics(g)))
    table = pd.DataFrame(out)
    table.to_csv(HERE / 'comparison.csv',index=False)
    pooled = table[table.day=='pooled'].set_index(['arm','policy'])
    verdicts = {arm:decision(pooled.loc[arm,'current'],pooled.loc[arm,'taper']) for arm in ref['ARMS']}
    receipt = dict(verdicts=verdicts, candidate_passes=all(v=='passes' for v in verdicts.values()),
                   sha256={p.name:R.sha(p) for p in sorted(HERE.glob('*.csv'))})
    (HERE/'result.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(pooled.to_string(),verdicts,flush=True)


def fixture():
    assert [round(taper(t),4) for t in [190,150,149.2,90,48,30,0]] == [1000,1000,149,75,22.5,0,0]
    with patch.object(U.S,'curve',taper):
        assert U.over_budget(100,90) and U.over_budget(-100,90) and not U.over_budget(75,90)
    assert not U.over_budget(100,90)
    test = dict(mean_end=12,tail_end=60,tail_loss=18,settle=100,positive_days=6)
    good = dict(test,mean_end=10,tail_end=55)
    assert decision(test,good)=='passes' and decision(test,dict(good,settle=99))=='risk_tradeoff'
    assert decision(test,test)=='no_inventory_improvement'
    row = baseline(ref['DAYS'][0]).iloc[:2].copy()
    validate(row,row)
    broken = row.copy()
    broken['settle'] += 1
    try:
        validate(broken,row)
    except AssertionError:
        return
    raise AssertionError('corrupt ledger accepted')


def run():
    fixture()
    fit = U.S.read(R.HERE/'fit_v2.json')['cells']
    binding = U.S.read(ref['S']['L']['E'].REPLICATION/'result.json')
    assert R.sha(R.HERE/'fit_v2.json') == binding['fit_sha'] and R.sha(R.CENSUS)==binding['census_sha']
    assert R.sha(ref['S']['L']['E'].PRIOR/'urgency.py')==binding['controller_sha']
    assert R.sha(OD.OUT3/'bn_changes.parquet')==U.S.read(OD.RCPT/'bn_events.json')['out_sha256']
    tape = R.MM.KP.Tape()
    edges = R.MM.TR.T.P.frozen_edges()
    for day in ref['DAYS']:
        old = baseline(day)
        lookup = old.set_index(['mid','arm'],verify_integrity=True)
        ids = set(old.mid)
        assert len(old)==2*len(ids) and set(old.arm)==set(ref['ARMS'])
        reg = R.MM.registry(day).set_index('mid')
        payouts = pd.read_parquet(R.MM.S1/day/'registry.parquet').set_index('mid')
        seg = OD.SEG_OF.get(day)
        off = tape.offset(day)
        rows = []
        for _,hour in reg.groupby('hour'):
            batch = sorted(ids & set(hour.index))
            if not batch:
                continue
            for mid,status,path,info in R.MM.worlds(day,batch,tape,off,reg):
                ident = R.MM.ident_of(day,mid,1000.*reg.at[mid,'ws'],info,path)
                assert status=='complete' and ident['tape_id']==lookup.at[(mid,'nominal'),'tape_id']
                cache = (OD.make_cache3(info['t_start'],info['end_rel'],seg) if seg
                         else R.MM.make_cache(info['t_start'],info['end_rel']))
                args = (path,info,ident,R.triggers(cache,edges),fit,R.VM.V.payout_of(payouts.loc[mid]))
                for arm in ref['ARMS']:
                    if mid==min(ids):
                        check = ref['simulate'](arm,*args)
                        for k,v in check.items():
                            assert abs(v-lookup.at[(mid,arm),k]) < 1e-6,(mid,arm,k)
                    with patch.object(U.S,'curve',taper):
                        result = ref['simulate'](arm,*args)
                    rows.append(dict(day=day,mid=mid,tape_id=ident['tape_id'],arm=arm,**result))
        df = pd.DataFrame(rows)
        validate(df,old)
        df.to_csv(HERE/f'markets_{day}.csv',index=False)
        print(day,len(ids),'paired markets verified',flush=True)
    score()


if __name__=='__main__':
    run()
