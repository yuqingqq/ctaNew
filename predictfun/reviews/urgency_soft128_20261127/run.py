"""Size fresh quotes against a 128-share soft limit inside the Tokyo hard cap."""
import math
import json
import runpy
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pandas as pd

HERE = Path(__file__).resolve().parent
taper_test = runpy.run_path(str(HERE.parent/'urgency_inventory_taper_20261126'/'run.py'))
ref = taper_test['ref']
R, U, OD, parent = (ref[k] for k in ('R','U','OD','prior'))
DAYS, ARMS = ref['DAYS'], ref['ARMS']
Base = parent.Veto


SOFT_LIMIT = 150 - 22


class BufferedSize(Base):
    def __init__(self, *args):
        super().__init__(*args)
        self.activity = Counter()

    def slot(self, view, rung, price):
        opened, first = self.opened, self.first_side
        proposal = super().slot(view, rung, price)
        action, size, _ = proposal
        if action != 'POST':
            return proposal
        side = view['side']
        signed = (1-2*side)*view['position']
        reserved = view['A'] if side else view['B']
        available = math.floor(SOFT_LIMIT - signed - reserved + R.EPS)
        if size <= available:
            return proposal
        native_price = 100-price if side else price
        if available < 1 or available*native_price < 100-R.EPS:
            self.proposals.pop()
            self.opened, self.first_side = opened, first
            self.activity['buffer_wait'] += 1
            return 'WAIT', None, 'soft_room'
        g, s, capacity, level, _, unit = self.proposals[-1]
        self.proposals[-1] = (g,s,capacity,level,max(0.,signed+reserved+available),unit)
        self.activity['buffer_clip'] += 1
        return 'POST',available,'soft_size'


def fixture():
    def post(policy,view,rung,price):
        policy.opened,policy.first_side = 1,view['side']
        policy.proposals.append((0,view['side'],100,0,150,.5))
        return 'POST',50,'post'
    for side in (0,1):
        for signed,reserved,native,expected in [
            (78,0,50,50),(80,20,50,28),(100.2,0,50,27),(-80,20,50,50),
            (120,8,50,0),(120,7,50,0),(120,6,50,2)
        ]:
            obj = BufferedSize.__new__(BufferedSize)
            obj.opened,obj.first_side,obj.proposals,obj.activity=None,None,[],Counter()
            view=dict(side=side,position=signed*(1-2*side),
                      A=reserved if side else 99,B=99 if side else reserved)
            with patch.object(Base,'slot',post):
                action,q,_=obj.slot(view,1,100-native if side else native)
            assert (action,q)==(('POST',expected) if expected else ('WAIT',None))
            assert len(obj.proposals)==int(expected>0) and (obj.opened is None)==(expected==0)
            if 0<expected<50:
                assert obj.proposals[-1][4]==max(0.,signed+reserved+expected)
            with patch.object(Base,'slot',return_value=('WAIT',None,'tokyo_cap')):
                assert obj.slot(view,1,50)==('WAIT',None,'tokyo_cap')
    assert BufferedSize.review is Base.review and BufferedSize.live is Base.live
    ref['fixture']()


def simulate(arm,args):
    policies = []
    def build(*policy_args):
        pol = BufferedSize(*policy_args)
        policies.append(pol)
        return pol
    with patch.object(parent,'Veto',build):
        result = ref['simulate'](arm,*args)
    assert result['peak_position']<=SOFT_LIMIT+1e-6
    pol, = policies
    return dict(result,buffer_wait=pol.activity['buffer_wait'],buffer_clip=pol.activity['buffer_clip'])

def score():
    groups = []
    for day in DAYS:
        old = taper_test['baseline'](day)
        new = pd.read_csv(HERE/f'markets_{day}.csv',float_precision='round_trip',dtype={'day':str})
        taper_test['validate'](new,old)
        groups.extend((old.assign(policy='current'),new.assign(policy='soft128')))
    table = []
    for (arm,policy),group in pd.concat(groups).groupby(['arm','policy']):
        for day,g in [('pooled',group),*group.groupby('day')]:
            table.append(dict(day=day,arm=arm,policy=policy,**taper_test['metrics'](g)))
    scores = pd.DataFrame(table)
    scores.to_csv(HERE/'comparison.csv',index=False)
    pooled = scores[scores.day=='pooled'].set_index(['arm','policy'])
    verdicts = {a:taper_test['decision'](pooled.loc[a,'current'],pooled.loc[a,'soft128']) for a in ARMS}
    receipt = dict(verdicts=verdicts,candidate_passes=all(v=='passes' for v in verdicts.values()),
                   sha256={p.name:R.sha(p) for p in sorted(HERE.glob('*.csv'))})
    (HERE/'result.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(pooled.to_string(),verdicts,flush=True)


def run():
    fixture()
    fit = U.S.read(R.HERE/'fit_v2.json')['cells']
    bind = U.S.read(ref['S']['L']['E'].REPLICATION/'result.json')
    assert R.sha(R.HERE/'fit_v2.json')==bind['fit_sha'] and R.sha(R.CENSUS)==bind['census_sha']
    assert R.sha(ref['S']['L']['E'].PRIOR/'urgency.py')==bind['controller_sha']
    assert R.sha(OD.OUT3/'bn_changes.parquet')==U.S.read(OD.RCPT/'bn_events.json')['out_sha256']
    assert R.sha(OD.OUT3/'summary.json')=='f1b45f6555c707654acf337359e98d900b67cf4c1277312dfdd4d82c0b46c460'
    tape = R.MM.KP.Tape()
    edges = R.MM.TR.T.P.frozen_edges()
    for day in DAYS:
        old = taper_test['baseline'](day)
        lookup = old.set_index(['mid','arm'])
        ids = set(old.mid)
        assert lookup.index.is_unique and len(old)==2*len(ids)
        reg = R.MM.registry(day).set_index('mid')
        payouts = pd.read_parquet(R.MM.S1/day/'registry.parquet').set_index('mid')
        off,seg = tape.offset(day), OD.SEG_OF.get(day)
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
                for arm in ARMS:
                    def parity(row):
                        for k in row:
                            if k in lookup.columns:
                                assert abs(row[k]-lookup.at[(mid,arm),k])<1e-6,(mid,arm,k)
                    if mid==min(ids):
                        parity(ref['simulate'](arm,*args))
                    row = simulate(arm,args)
                    if row['buffer_wait']+row['buffer_clip']==0:
                        parity(row)
                    rows.append(dict(day=day,mid=mid,tape_id=ident['tape_id'],arm=arm,**row))
        df = pd.DataFrame(rows)
        taper_test['validate'](df,old)
        df.to_csv(HERE/f'markets_{day}.csv',index=False)
        print(day,len(ids),'markets reconciled',flush=True)
    score()


if __name__=='__main__':
    run()
