"""Replay each size through the frozen urgency controller and count actual fill IDs."""
import io
import math
import sys
from collections import Counter
from pathlib import Path
from runpy import run_path
from subprocess import check_output
from unittest.mock import patch

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRIOR = HERE.parent / 'urgency_size100_20261126'
S = run_path(str(PRIOR / 'size_test.py'))
U, R, OD = S['U'], S['R'], S['OD']
SIZES = (50, 100, 220)


def ledger(fills, mark):
    fifo = R.X.fifo(fills, mark)
    closing = set()
    pairs = set()
    gross = [0., 0.]
    shares = [0., 0., 0.]
    for opened, closed, q in U.C.inventory_pairs(fills):
        closing.add(closed[1])
        pairs.add((opened[1], closed[1]))
        edge = (1 - 2 * opened[2]) * (closed[4] - opened[4])
        bucket = 0 if edge > 0 else 1 if edge < 0 else 2
        shares[bucket] += q
        if edge:
            gross[bucket] += q * edge / 100
    assert abs(sum(gross) * 100 - fifo['completed_c']) < 1e-5
    assert abs(sum(shares) - fifo['matched_shares']) < 1e-6
    return dict(closed=fifo['matched_shares'], turnover=fifo['completed_c']/100,
                residual=fifo['residual_c']/100, shares=sum(f[3] for f in fills),
                filled_orders=len({f[1] for f in fills}), fill_events=len(fills),
                closing_orders=len(closing), order_pairs=len(pairs), gross_gain=gross[0],
                gross_loss=gross[1], win_sh=shares[0], loss_sh=shares[1], flat_sh=shares[2])


def replay(size, path, info, ident, flags, fit, payout):
    assert size in SIZES and payout in (0, .5, 1)
    branch = U.UrgencyBranch(path, info['end_rel'], ident)
    branch.q0 = size
    policy = U.S.Smooth(branch, flags, fit, 'linear', {}, [])
    assert policy.q0 == size and policy.Qside == 440 and branch.cap == 1000 and branch.levels == 2
    with patch.object(R, 'max_new', S['size_bound'](size)):
        for tick in range(R.LB.n_ticks(branch)):
            R.LB.step(branch, tick, policy)
        R.LB.finish(branch)
    orders = branch.E.o
    S['order_check'](orders, size)
    assert len(policy.proposals) == len(orders)
    pos = 0.
    for _, gid, side, q, _, _ in branch.E.fills:
        assert gid in range(len(orders)) and q > 0
        pos += (1-2*side)*q
        assert abs(pos) <= 1000 + 1e-6
    mark = 100*payout
    _, end, settled = branch.E.value(mark)
    row = ledger(branch.E.fills, mark)
    row.update(settle=settled/100, end_pos=end, abs_end=abs(end), posts=len(orders),
               all_pulls=sum(o['req'] is not None for o in orders))
    assert abs(end-pos) < 1e-6
    assert abs(row['shares']-2*row['closed']-abs(end)) < 1e-6
    assert abs(row['settle']-row['turnover']-row['residual']) < 1e-6
    return row


def control(row, old):
    for field in S['FIELDS']:
        if not math.isclose(row[field], old[field], abs_tol=1e-6, rel_tol=0):
            raise ValueError(f"control mismatch at {field}: {row[field]} vs {old[field]}")


def fixture():
    fills = [(0,0,0,10,40,1), (1,1,0,5,42,2), (2,2,1,12,47,3),
             (3,2,1,8,39,4), (4,3,0,2,37,5)]
    row = ledger(fills, 50)
    assert (row['filled_orders'],row['fill_events'],row['closing_orders'],row['order_pairs'],row['closed']) == (4,5,2,3,17)
    assert abs(row['gross_gain']-.84) < 1e-9 and abs(row['gross_loss']+.09) < 1e-9
    assert S['size_bound'](50)(-200,0,1000,.5) == 50
    reference = dict.fromkeys(S['FIELDS'], 0.)
    control(reference, reference)
    try:
        control(dict(reference, closed=1), reference)
    except ValueError:
        return
    raise AssertionError("changed match quantity accepted")


def run():
    fixture()
    fit_file = R.HERE / 'fit_v2.json'
    binding = U.S.read(S['L']['E'].REPLICATION/'result.json')
    assert R.sha(fit_file) == binding['fit_sha'] and R.sha(R.CENSUS) == binding['census_sha']
    assert R.sha(S['L']['E'].PRIOR/'urgency.py') == binding['controller_sha']
    assert R.sha(OD.OUT3/'bn_changes.parquet') == U.S.read(OD.RCPT/'bn_events.json')['out_sha256']
    assert R.sha(OD.OUT3/'summary.json') == 'f1b45f6555c707654acf337359e98d900b67cf4c1277312dfdd4d82c0b46c460'
    fit = U.S.read(fit_file)['cells']
    raw = U.S.read(R.CENSUS)['rows']
    alt = U.S.read(OD.OUT3/'summary.json')['census']
    src = R.MM.KP.Tape()
    edges = R.MM.TR.T.P.frozen_edges()
    for day in (sys.argv[2:] or S['DAYS']):
        assert day in S['DAYS']
        previous = S['archived'](day)
        path = PRIOR/f'markets_{day}.csv'
        saved = check_output(['git','show',f'c2f5ff352:{path.relative_to(ROOT)}'], cwd=ROOT)
        old = pd.read_csv(io.BytesIO(saved),float_precision='round_trip').set_index(['size','mid'],verify_integrity=True)
        seg = OD.SEG_OF.get(day)
        census = [r for r in (alt if seg else raw) if r['day']==day]
        reg = R.MM.registry(day).set_index('mid')
        ids = {r['mid'] for r in census if r['status']=='complete'}
        assert len(reg)==len(census)==288 and set(reg.index)=={r['mid'] for r in census}
        assert ids == set(previous.index) and set(old.index)=={(n,m) for n in (100,220) for m in ids}
        expected = previous.tape_id.to_dict() if seg else {r['mid']:r['tape_id'] for r in census if r['status']=='complete'}
        payouts = pd.read_parquet(R.MM.S1/day/'registry.parquet').set_index('mid')
        off = src.offset(day)
        rows = []
        seen = set()
        for _, group in reg.groupby('hour'):
            batch = sorted(ids & set(group.index))
            if not batch:
                continue
            for m, status, market, info in R.MM.worlds(day,batch,src,off,reg):
                assert status=='complete' and m in batch and m not in seen
                seen.add(m)
                ws = 1000.*reg.at[m,'ws']
                ident = R.MM.ident_of(day,m,ws,info,market)
                assert ident['tape_id']==expected[m]
                if seg:
                    assert OD.SEG_BOUNDS[seg][0]<=ws<ws+300000<OD.SEG_BOUNDS[seg][1]
                    cache = OD.make_cache3(info['t_start'],info['end_rel'],seg)
                else:
                    cache = R.MM.make_cache(info['t_start'],info['end_rel'])
                flags = R.triggers(cache,edges)
                payout = R.VM.V.payout_of(payouts.loc[m])
                for n in SIZES:
                    row = replay(n,market,info,ident,flags,fit,payout)
                    if n!=50:
                        control(row, old.loc[n,m])
                    rows.append(dict(day=day,mid=m,size=n,**row))
        assert seen==ids
        pd.DataFrame(rows).to_csv(HERE/f'markets_{day}.csv',index=False,mode='x')
        print(day,len(ids),dict(Counter(r['status'] for r in census)),flush=True)


if __name__=='__main__':
    {'test':fixture,'run':run}[sys.argv[1]]()
