"""Two-day paired test of the upcoming inventory allowance on new quotes only."""
from collections import Counter
from pathlib import Path
import json
import math
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent/'inventory_urgency_20261125'
REPLICATION = HERE.parent/'urgency_replication_20261125'
sys.path.insert(0, str(PRIOR))
import urgency as U

R = U.R
DAYS = ('20261026', '20261027')
CONTROL_HASHES = {
    DAYS[0]: ('9adf210ca553fc2cd3be7f4b13b46553148d6354df8d30536a5a19b5379f9a49',
              '58bd8f97f3999a2a4e56ea8477b33ad9d2a3e587269b14514ab1cf8b11617e7c'),
    DAYS[1]: ('2a327e51f6d8b44282e75cf6734dccaf7262645ceb6ee51e8fb52720cb8db3dc',
              '71034d95d3ebfb9aa3ef458fc465f84f0d23b6e5f52b0aaab19389e60a96b796'),
}


class Ahead(U.S.Smooth):
    """EarlyUnwind.slot consults limit before any POST bookkeeping or venue minimum check."""
    sizing_new_quote = False

    def limit(self, g):
        normal = super().limit(g)
        left_ms = self.b.ident['opening_ms'] + 300000 - self.b.ident['start_abs_ms'] - g
        if self.sizing_new_quote and 60000 < left_ms <= 90000:
            return min(normal, U.S.curve(60))
        return normal

    def slot(self, view, rung, price):
        self.sizing_new_quote = True
        try:
            return super().slot(view, rung, price)
        finally:
            self.sizing_new_quote = False


def guard_size(left_ms, signed, reserved, size):
    if 60000 < left_ms <= 90000 and signed + reserved + size > 250 + R.EPS:
        raise ValueError('new proposal exceeds the upcoming budget')


def test():
    from types import SimpleNamespace
    policy = Ahead.__new__(Ahead)
    policy.schedule = 'linear'
    policy.b = SimpleNamespace(ident=dict(opening_ms=0, start_abs_ms=0))
    for left in (90001,90000,89999,60001,60000,59999):
        g = 300000-left
        for new in (False,True):
            policy.sizing_new_quote = new
            expected = 250 if new and 60000<left<=90000 else U.S.curve(left/1000)
            assert abs(policy.limit(g)-expected)<1e-8
    for side in (0,1):
        sign = 1-2*side
        same = 30
        for pos,qty in [(sign*210,10),(-sign*350,220)]:
            guard_size(75000,sign*pos,same,qty)
        try:
            guard_size(75000,210,same,11)
        except ValueError:
            pass
        else:
            raise AssertionError('ignored pending shares')
    print('Time boundaries, ordinary review limit, closing quantity and over-limit refusal passed.')


def simulate(path, info, ident, flags, model, payout, candidate):
    branch = U.UrgencyBranch(path, info['end_rel'], ident)
    cls = Ahead if candidate else U.S.Smooth
    policy = cls(branch, flags, model, 'linear', {}, [])
    expiry = ident['opening_ms']+300000-ident['start_abs_ms']
    for k in range(R.LB.n_ticks(branch)):
        R.LB.step(branch, k, policy)
    R.LB.finish(branch)
    fills = branch.E.fills
    ts = np.array([f[0] for f in fills])
    inventory = np.r_[0., np.cumsum([(1-2*f[2])*f[3] for f in fills])]
    _, end_pos, value = branch.E.value(100*payout)
    fifo = R.X.fifo(fills,100*payout)
    assert abs(end_pos-inventory[-1])<1e-6 and max(abs(inventory))<=1000+1e-6
    assert abs(expiry-branch.end-2000)<1e-6
    # Independently reconstruct client-known holdings and all unclosed reservations at
    # every submitted quote in the window, including earlier same-tick submissions.
    orders = branch.E.o
    closed = branch.closed
    assert all(o['gid']==i for i,o in enumerate(orders))
    recent_posts = 0
    still_reserved = 0
    for gid,o in enumerate(orders):
        g = o['submit']
        left = expiry-g
        if not 60000 < left <= 90000:
            continue
        side = o['side']
        recent_posts += 1
        if candidate:
            known = inventory[np.searchsorted(ts,g-R.BE.FILL_NOTICE,'right')]
            older = [i for i in range(gid) if orders[i]['side']==side and
                     (i not in closed or closed[i][1]>g)]
            qty_by_gid = Counter()
            for t,i,_,q,_,_ in fills:
                if t+R.BE.FILL_NOTICE>g:
                    break
                qty_by_gid[i]+=q
            reserved=sum(orders[i]['size']-qty_by_gid[i] for i in older)
            still_reserved+=any(orders[i]['req'] is not None and orders[i]['req']<=g for i in older)
            guard_size(left,(1-2*side)*known,reserved,o['size'])
    r=dict(settle=value/100, turnover=fifo['completed_c']/100, residual=fifo['residual_c']/100,
           closed=fifo['matched_shares'], abs_end=abs(end_pos), end_pos=end_pos,
           shares=sum(f[3] for f in fills), posts=len(orders),
           all_pulls=sum(o['req'] is not None for o in orders),
           abs_60s=abs(inventory[np.searchsorted(ts,expiry-60000,'right')]),
           window_posts=recent_posts)
    if candidate:
        r['window_posts_with_pending_cancel']=still_reserved
    assert math.isclose(r['settle'],r['turnover']+r['residual'],abs_tol=1e-6)
    assert math.isclose(r['shares'],2*r['closed']+r['abs_end'],abs_tol=1e-6)
    return r


def aggregate(g):
    totals={k:float(g[k].sum()) for k in ['settle','turnover','residual','closed','posts','all_pulls','window_posts']}
    return dict(**totals, mean_end=float(g.abs_end.mean()), p90_end=float(g.abs_end.quantile(.9)),
                mean_60=float(g.abs_60s.mean()), over_60=int((g.abs_60s>250+R.EPS).sum()),
                cents_per_matched=100*totals['turnover']/totals['closed'])


def score(df):
    result={}
    for day,rows in df.groupby('day'):
        groups={a:aggregate(g) for a,g in rows.groupby('arm')}
        c,b=groups['ahead'],groups['urgency']
        dif=rows.pivot(index='mid',columns='arm',values='settle')
        assert dif.notna().all().all()
        delta=dif.ahead-dif.urgency
        gate=dict(higher_settlement=c['settle']>b['settle'],lower_ending=c['mean_end']<b['mean_end'],
                  positive_completed=c['turnover']>0)
        result[day]=dict(**groups,gate=gate,passed=all(gate.values()),better=int((delta>1e-6).sum()),
                         worse=int((delta< -1e-6).sum()),equal=int((abs(delta)<=1e-6).sum()),
                         delta_ex_largest=float(delta.sum()-delta.max()))
    return result


def run():
    test()
    replication = U.S.read(REPLICATION/'result.json')
    fit_path = R.HERE/'fit_v2.json'
    assert R.sha(PRIOR/'urgency.py')==replication['controller_sha']
    assert R.sha(fit_path)==replication['fit_sha']
    fit=U.S.read(fit_path)
    assert R.sha(R.CENSUS)==fit['census_sha']==replication['census_sha']
    census=U.S.read(R.CENSUS)['rows']
    src=R.MM.KP.Tape()
    edges=R.MM.TR.T.P.frozen_edges()
    rows=[]
    for day,folder in zip(DAYS,(PRIOR,REPLICATION)):
        base=[]
        for filename,digest in zip(('markets.csv','inventory.csv'),CONTROL_HASHES[day]):
            path=folder/filename
            assert R.sha(path)==digest
            base.append(pd.read_csv(path,float_precision='round_trip').query("arm=='urgency'").set_index('mid'))
        idx={r['mid']:r['tape_id'] for r in census if r['day']==day and r['status']=='complete'}
        pop=[r for r in census if r['day']==day]
        reg=R.MM.registry(day).set_index('mid')
        assert set(reg.index)=={r['mid'] for r in pop} and len(reg)==len(pop)==288
        assert dict(Counter(r['status'] for r in pop))==replication['coverage'][day]
        assert all(set(table.index)==set(idx) and table.index.is_unique for table in base)
        rg=pd.read_parquet(R.MM.S1/day/'registry.parquet').set_index('mid')
        off=src.offset(day)
        seen=set()
        for hour,group in reg.groupby('hour'):
            ids=sorted(set(group.index)&idx.keys())
            if not ids:
                continue
            for m,status,path,info in R.MM.worlds(day,ids,src,off,reg):
                assert status=='complete' and m not in seen
                seen.add(m)
                ident=R.MM.ident_of(day,m,1000.*reg.loc[m,'ws'],info,path)
                assert ident['tape_id']==idx[m]
                flags=R.triggers(R.MM.make_cache(info['t_start'],info['end_rel']),edges)
                payout=R.VM.V.payout_of(rg.loc[m])
                assert payout in (0,1)
                for arm in ('urgency','ahead'):
                    r=simulate(path,info,ident,flags,fit['cells'],payout,arm=='ahead')
                    if arm=='urgency':
                        for k,v in r.items():
                            if k!='window_posts':
                                table=base[int(k=='abs_60s')]
                                assert abs(v-table.at[m,k])<1e-6,(m,k)
                    rows.append(dict(day=day,mid=m,arm=arm,**r))
            print(day,hour,len(seen),flush=True)
        assert seen==set(idx)
    df=pd.DataFrame(rows)
    results=score(df)
    results.update(passed_both_days=all(results[day]['passed'] for day in DAYS),
                   coverage=replication['coverage'],source_sha=R.sha(Path(__file__)))
    receipt=json.dumps(results,allow_nan=False,indent=2)
    df.to_csv(HERE/'markets.csv',index=False,mode='x')
    with (HERE/'result.json').open('x') as f:
        f.write(receipt+'\n')


if __name__=='__main__':
    {'test':test,'run':run}[sys.argv[1]]()
