"""One late veto on fresh inventory-adding proposals, against the frozen Tokyo control."""
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


class NewOrderVeto(Base):
    def __init__(self, *args):
        super().__init__(*args)
        self.activity = Counter()
        self.expiry = self.b.ident['opening_ms'] + 300000 - self.b.ident['start_abs_ms']

    def slot(self, view, rung, price):
        opened, first = self.opened, self.first_side
        proposal = super().slot(view, rung, price)
        action, size, _ = proposal
        left = (self.expiry-view['t'])/1000
        if action != 'POST' or left >= 150:
            return proposal
        side = view['side']
        signed = (1-2*side)*view['position']
        if signed < -R.EPS:
            self.activity['reducing_pass'] += 1
            return proposal
        reserved = view['A'] if side else view['B']
        if signed + reserved + size <= taper_test['taper'](left)+R.EPS:
            return proposal
        self.opened, self.first_side = opened, first
        self.proposals.pop()
        self.activity['veto_ask' if side else 'veto_bid'] += 1
        return 'WAIT', None, 'late_adding_veto'


def fixture():
    def post(policy, view, rung, price):
        policy.opened, policy.first_side = 1, view['side']
        policy.proposals.append(50)
        return 'POST', 50, 'post'

    with patch.object(Base,'slot',post):
        for side in (0,1):
            for left, signed, reserved, veto in [
                (150,100,0,False),(90,25,0,False),(90,25,1,True),
                (90,0,30,True),(90,80,0,True),(30,0,0,True),(30,-1,100,False)
            ]:
                obj = NewOrderVeto.__new__(NewOrderVeto)
                obj.expiry, obj.opened, obj.first_side = 300000, None, None
                obj.proposals, obj.activity = [], Counter()
                v = dict(side=side,t=obj.expiry-1000*left,position=(1-2*side)*signed,
                         A=reserved if side else 150, B=150 if side else reserved)
                action = obj.slot(v,0,50)
                assert (action[0]=='WAIT') == veto
                assert len(obj.proposals)==int(not veto) and (obj.opened is None)==veto
                assert (obj.first_side is None)==veto and (veto or action==('POST',50,'post'))
    assert NewOrderVeto.review is Base.review and NewOrderVeto.live is Base.live
    ref['fixture']()


def simulate(arm,args):
    policies = []
    def build(*policy_args):
        pol = NewOrderVeto(*policy_args)
        policies.append(pol)
        return pol
    with patch.object(parent,'Veto',build):
        result = ref['simulate'](arm,*args)
    pol, = policies
    return dict(result, **{k:pol.activity[k] for k in ('veto_bid','veto_ask','reducing_pass')})


def score():
    groups = []
    for day in DAYS:
        old = taper_test['baseline'](day)
        new = pd.read_csv(HERE/f'markets_{day}.csv',float_precision='round_trip',dtype={'day':str})
        taper_test['validate'](new,old)
        groups.extend((old.assign(policy='current'),new.assign(policy='new_order_veto')))
    table = []
    for (arm,policy),group in pd.concat(groups).groupby(['arm','policy']):
        for day,g in [('pooled',group),*group.groupby('day')]:
            table.append(dict(day=day,arm=arm,policy=policy,**taper_test['metrics'](g)))
    scores = pd.DataFrame(table)
    scores.to_csv(HERE/'comparison.csv',index=False)
    pooled = scores[scores.day=='pooled'].set_index(['arm','policy'])
    verdicts = {a:taper_test['decision'](pooled.loc[a,'current'],pooled.loc[a,'new_order_veto']) for a in ARMS}
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
                    if row['veto_bid']+row['veto_ask']==0:
                        parity(row)
                    rows.append(dict(day=day,mid=mid,tape_id=ident['tape_id'],arm=arm,**row))
        df = pd.DataFrame(rows)
        taper_test['validate'](df,old)
        df.to_csv(HERE/f'markets_{day}.csv',index=False)
        print(day,len(ids),'markets reconciled',flush=True)
    score()


if __name__=='__main__':
    run()
