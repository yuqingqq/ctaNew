"""Independent read-back on committed-size/cap controls, market populations, economics and gate."""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
DAYS=('20261023','20261024','20261025','20261027','20261031','20261101','20261102','20261103')
ARMS=('nominal','short')


def close(a,b,tol=1e-6):
    if not math.isclose(float(a),float(b),abs_tol=tol,rel_tol=0):
        raise ValueError(f'{a} != {b}')


def read(path):
    with path.open() as src:
        return list(csv.DictReader(src))


def check():
    close(1,1)
    try: close(1,1.001)
    except ValueError: pass
    else: raise AssertionError('corrupted control accepted')
    prior=read(HERE.parent/'tokyo_execution_comparison_20261126'/'cap_markets.csv')
    old={r['mid']:r for r in prior if r['cap']=='150'}
    control=read(HERE/'markets_20261026.csv')
    assert len(control)==len(old)==70 and {r['mid'] for r in control}==set(old)
    for r in control:
        assert r['arm']=='nominal'
        for k,v in old[r['mid']].items():
            if v and k not in ('mid','cap'): close(v,r[k])
    report=json.loads((HERE/'result.json').read_text())
    daily={(r['day'],r['arm']):r for r in read(HERE/'daily.csv')}
    assert len(daily)==len(DAYS)*len(ARMS)
    allrows=defaultdict(list)
    day_values=defaultdict(list)
    totals=Counter()
    hashes={}
    for d in DAYS:
        path=HERE/f'markets_{d}.csv'
        rows=read(path)
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        baseline=read(HERE.parent/'urgency_size50_20261126'/f'markets_{d}.csv')
        ids={x['mid'] for x in baseline if x['size']=='50'}
        assert len(rows)==len(ids)*2 and {(r['mid'],r['arm']) for r in rows}=={(m,a) for m in ids for a in ARMS}
        market_tape={}
        for r in rows:
            assert r['day']==d and r['tape_id']==market_tape.setdefault(r['mid'],r['tape_id'])
            assert float(r['peak_position'])<=150.000001
            close(float(r['turnover'])+float(r['residual']),r['settle'])
            close(float(r['gross_gain'])+float(r['gross_loss']),r['turnover'])
            close(float(r['win_sh'])+float(r['loss_sh'])+float(r['flat_sh']),r['closed'])
            close(2*float(r['closed'])+float(r['abs_end']),r['shares'])
            assert 0<=float(r['post_request_shares'])<=float(r['shares'])+1e-6
            if r['arm']=='short': close(0,r['post_request_shares'])
            allrows[r['arm']].append(r)
        for a in ARMS:
            sub=[r for r in rows if r['arm']==a]
            count=len(sub); ref=daily[d,a]
            close(count,ref['markets'])
            close(report['coverage'][d]['complete'],count)
            close(sum(report['coverage'][d].values()),288)
            for k in ('closed','settle','turnover','residual','shares','post_request_shares'):
                close(math.fsum(float(r[k]) for r in sub),ref[k])
            close(math.fsum(float(r['abs_end']) for r in sub)/count,ref['mean_abs_end'])
            close(float(ref['turnover'])*100/float(ref['closed']),ref['cents_per_matched'])
            day_values[a].append(float(ref['settle']))
            totals[a]+=count
    states=[]
    for a,rows in allrows.items():
        assert totals[a]==619
        summary=report['arms'][a]
        x=math.fsum(float(r['settle']) for r in rows)
        leave_day=x-max(day_values[a]); leave_market=x-max(float(r['settle']) for r in rows)
        npos=sum(v>0 for v in day_values[a])
        close(x,summary['settle']);close(leave_day,summary['without_best_day'])
        close(leave_market,summary['without_best_market']);close(npos,summary['positive_days'])
        states.append('stop' if min(x,leave_market,leave_day)<=0 else 'pass' if npos>=5 else 'inconclusive')
    expected='stop' if 'stop' in states else 'consider_prospective_validation' if set(states)=={'pass'} else 'inconclusive'
    assert expected==report['screen']
    (HERE/'verification.json').write_text(json.dumps(dict(control_markets=70,paired_markets=619,
        arms=2,screen=expected,sha256=hashes),indent=2)+'\n')
    print('Verified nominal control, market membership, both ledgers, daily results and screen:',expected)

if __name__=='__main__': check()
