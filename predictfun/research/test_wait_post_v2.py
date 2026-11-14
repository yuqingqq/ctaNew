"""Causal waiting, economic target, chronology and acceptance falsifiers."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import pf_wait_post_v2 as M
import pf_entry_value as E
import pf_wait_post as OLD
from test_entry_value import context, Score
from test_order_harm_client50 import normalized


class FunctionScore:
    def __init__(self,fn):self.fn=fn
    def predict(self,x):return np.asarray([self.fn(z) for z in x])

class FittedScore:
    def predict(self,x):return x[:,0]
    def dump_model(self):return {'tree_info':[{'num_leaves':2}]}
    def save_model(self,path):Path(path).write_text('synthetic fitted head\n')


def teacher(fn):
    return M.Forecast({s:{'post':FunctionScore(fn)} for s in (0,1)},'synthetic-frozen-policy')


def first(W):
    views=[];M.run(W,lambda v:True,observe=lambda v,a:views.append(v))
    return next(v for v in views if v['status']=='eligible' and v['side']==0 and v['rung']==0)


def late(adverse=False):
    if not adverse:return context([(400.,49,30.)])
    import pf_joint_ladder as JL
    bid=[(0.,[(50,100),(49,100),(48,100),(47,100),(46,100)],52),(500.,[(40,100)],42)]
    ask=[(0.,[(52,100),(53,100),(54,100),(55,100),(56,100)],50),(500.,[(42,100)],40)]
    w={0:JL._ctx(0,bid,[(400.,49,30.)],t_end=1000.),1:JL._ctx(1,ask,[],t_end=1000.)}
    for v in w.values():v['vkl']=[-1.]*len(v['tg']);v['m']=1;v['tf_mff']=(v['tf'],np.array([51.,41.]))
    return w


class WaitPostV2Tests(unittest.TestCase):
    def test_neutral_keeper_fast_lane_execution_parity(self):
        for adverse in (False,True):
            for shift in M.CFGS:
                for delay in (0,4):
                    W=late(adverse)
                    old=E.run(copy.deepcopy(W),lambda v:v['kind']=='rest' or v['k']>=delay,shift)
                    new=M.run(copy.deepcopy(W),lambda v:v['k']>=delay,shift)
                    for field in ('fl','fills','orders','order_states','endpos','stat','terminal_time'):
                        self.assertEqual(normalized(old[field]),normalized(new[field]),field)
                    self.assertTrue(all(d[3]=='entry' for d in new['policy_decisions']))

    def test_fast_inputs_equal_original_and_preserve_waiting_state(self):
        W=late(True);v=first(W);tr=M.Tracker();tr.see(v)
        for k in (0,1,4,10):
            view=dict(v,k=k,t=k*50.,price=40 if k>=10 else 50,known_position=30.)
            x=M.inputs(W,view,tr)
            np.testing.assert_equal(x[:len(OLD.FEATURES)],OLD.features(W,view))
            self.assertEqual(x[-2],50.*k);self.assertEqual(x[-1],view['price']-50)
        future=copy.deepcopy(W);future[0]['S5'][1]*=10;future[0]['P5'][1]-=5
        np.testing.assert_equal(M.inputs(W,v,M.Tracker()),M.inputs(future,v,M.Tracker()))
        with self.assertRaises(ValueError):M.inputs(W,dict(v,kind='rest'),M.Tracker())

    def test_absolute_target_separates_old_zero_label_profit_and_loss(self):
        for adverse,positive in ((False,True),(True,False)):
            W=late(adverse);v=first(W)
            old=E.paired_probe(W,v)
            self.assertEqual(old['target_c'],0.)
            pair,x=M.paired(W,v,0.,M.Forecast())
            self.assertEqual(pair['status'],'eligible');self.assertEqual(pair['qwait'],0.)
            self.assertEqual(pair['qpost']>0,positive)
            self.assertEqual(M.post_rule(pair['qpost'],pair['qwait']),positive)
        self.assertTrue(M.post_rule(1.,1.));self.assertFalse(M.post_rule(-1.,-1.));self.assertFalse(M.post_rule(0.,0.))

    def test_wait_follows_same_model_and_can_wait_multiple_decisions(self):
        W=late();v=first(W);j=M.FEATURES.index('idle_age_ms')
        frozen=teacher(lambda x:1. if x[j]>=300. else -1.)
        r,probe,tr=M.fork(W,v,0.,frozen,'wait')
        self.assertEqual(r['order_states'][0,0][0]['ts'],300.)
        actions=[z for z in r['policy_decisions'] if z[1:5]==(0,0,'entry',0)]
        self.assertTrue(all(not z[-1] for z in actions if z[0]<6))
        # A teacher that never turns positive must never expire into a repost.
        no=M.fork(W,v,0.,teacher(lambda x:-1.),'wait')[0]
        self.assertFalse(no['order_states'][0,0])
        p,_=M.paired(W,v,0.,frozen)
        self.assertGreater(p['qwait'],0.);self.assertGreater(p['qpost'],0.)
        self.assertTrue(M.post_rule(p['qpost'],p['qwait']))

    def test_wait_value_captures_model_driven_better_later_entry(self):
        W=context([(95.,49,30.),(200.,39,30.)],adverse=True);v=first(W)
        j=M.FEATURES.index('price');d=M.FEATURES.index('rung');side=M.FEATURES.index('side_sign')
        frozen=teacher(lambda x:1. if x[j]<45. and x[d]==0 and x[side]>0 else -1.)
        pair,_=M.paired(W,v,0.,frozen)
        self.assertLess(pair['qpost'],0.)
        self.assertGreater(pair['qwait'],0.)
        self.assertFalse(M.post_rule(pair['qpost'],pair['qwait']))
        waited=M.fork(W,v,0.,frozen,'wait')[0]
        self.assertEqual(waited['order_states'][0,0][0]['ts'],100.)

    def test_absolute_value_accounts_for_displacing_better_deeper_fill(self):
        W=context([(95.,49,30.),(200.,39,30.)],adverse=True);v=first(W)
        j=M.FEATURES.index('price');frozen=teacher(lambda x:1. if x[j]<45. else -1.)
        pair,_=M.paired(W,v,0.,frozen)
        # Posting40 displaces the other rung's fill at39. Both fills have a
        # positive standalone markout, but admitting40 loses1c/share jointly.
        self.assertAlmostEqual(pair['qwait'],-1.)
        self.assertGreater(pair['value_wait'],0.)
        self.assertGreater(pair['value_never'],pair['value_wait'])

    def test_delayed_probe_has_shared_prefix_and_real_wait_age(self):
        W=context(end=1000.);v=first(W);p,x=M.paired(W,v,400.,M.Forecast())
        self.assertEqual(p['probe_t'],400.);self.assertEqual(x[-2],400.)
        for branch in ('post','wait','never'):
            r,probe,_=M.fork(W,v,400.,M.Forecast(),branch)
            self.assertTrue(all(not z[-1] for z in r['policy_decisions'] if z[1:5]==(0,0,'entry',0) and z[0]<8))
        with self.assertRaisesRegex(ValueError,'birth state'):M.paired(W,dict(v,known_position=999.),400.,M.Forecast())
        p,x=M.paired(W,v,5000.,M.Forecast());self.assertEqual(p['status'],'probe_not_reached');self.assertIsNone(x)

    def test_generation_population_deduplicated_and_statuses_retained(self):
        rows,st=M.build_window(context(),'20261023',1,M.Forecast(),all_sampled=True)
        f=pd.DataFrame(rows);self.assertEqual(len(f),12)
        self.assertFalse(f.duplicated(['day','market','side','rung','generation']).any())
        self.assertTrue(set(f.status)<={'eligible','probe_not_reached'})
        self.assertTrue((f[f.status=='eligible'].qpost==0).all())
        self.assertGreater((f.status=='probe_not_reached').sum(),0)

    def test_independent_side_heads_and_invalid_predictions(self):
        W=context();predictor=M.Forecast({0:{'post':Score(1.),'wait':Score(0.)},1:{'post':Score(1.),'wait':Score(2.)}},'fake')
        r=M.run(W,M.policy(W,predictor))
        self.assertTrue(r['order_states'][0,0]);self.assertFalse(r['order_states'][1,0])
        for p,w in ((np.nan,0),(1,np.inf),(-1,-2)):
            self.assertFalse(M.post_rule(p,w))

    def test_constant_heads_refused_positive_health_control(self):
        x=np.arange(100,dtype=float)[:,None];y=x[:,0]
        self.assertTrue(M.model_health(FittedScore(),x,y)['has_splits'])
        constant=FittedScore();constant.predict=lambda x:np.ones(len(x))
        with self.assertRaisesRegex(ValueError,'constant'):M.model_health(constant,x,y)
        constant=FittedScore();constant.dump_model=lambda:{'tree_info':[{'num_leaves':1}]}
        with self.assertRaisesRegex(ValueError,'constant'):M.model_health(constant,x,y)
        self.assertTrue(M.model_health(None,x,np.zeros(len(x)),True)['structural_zero'])
        with self.assertRaises(ValueError):M.model_health(None,x,y,True)

    def test_chronological_pipeline_preserves_raw_economic_targets(self):
        import lightgbm as lgb
        observed=[]
        def rows(out,kind,day,partial=False):
            self.assertIn(day,M.BOOT_DAYS if kind=='flat' else M.FOLDS[M.TESTS[0]])
            r=pd.DataFrame(np.zeros((1200,len(M.FEATURES))),columns=M.FEATURES)
            r.iloc[:,0]=np.arange(1200);r['side']=[0]*600+[1]*600;r['day']=day;r['status']='eligible'
            r['qpost']=np.where(r.side==0,25.,-25.);r['qwait']=np.where(r.side==0,15.,-15.)
            return r
        def dataset(x,y,feature_name):observed.append(y.copy());return (x,y)
        for test,nheads in ((None,2),(M.TESTS[0],4)):
            observed.clear()
            with tempfile.TemporaryDirectory() as tmp:
                days=M.BOOT_DAYS if test is None else M.FOLDS[test];kind='flat' if test is None else 'learned'
                for d in days:
                    p=Path(tmp)/M.row_stage(kind,d);p.mkdir();(p/'receipt.json').write_text('{}')
                if test is not None:
                    p=Path(tmp)/'bootstrap';p.mkdir();(p/'receipt.json').write_text('{}')
                with mock.patch.object(M,'load_rows',side_effect=rows),mock.patch.object(lgb,'Dataset',side_effect=dataset),mock.patch.object(lgb,'train',return_value=FittedScore()):
                    M.fit(tmp,test)
                self.assertEqual(len(observed),nheads)
                self.assertTrue(all(np.max(abs(y))>10 for y in observed))
                meta=E.read_json(Path(tmp)/('bootstrap' if test is None else 'fit_'+test)/'receipt.json')
                self.assertEqual(meta['train'],list(days))
                if test is not None:self.assertTrue(all(d<test for d in meta['train']))
        self.assertTrue(all(max(M.BOOT_DAYS)<min(days) for days in M.FOLDS.values()))

    def test_matched_local_null_detects_real_and_reversed_selection(self):
        r=pd.DataFrame(dict(day=['x']*400,side=[0]*400,rung=[0]*400,probe_t=np.arange(400)*50.,delta=[-1.]*200+[1.]*200,
                            qpost=[-1.]*200+[1.]*200,post_action=[False]*200+[True]*200))
        self.assertTrue(M.timing_null(r)['passes']);r.post_action=~r.post_action
        self.assertFalse(M.timing_null(r)['passes'])
        with self.assertRaises(ValueError):M.timing_null(r,20)
        r.post_action=True;self.assertFalse(M.timing_null(r)['passes'])

    def test_quota_controller_matches_placements_and_rejects_bad_quota(self):
        for q in ([0,0],[2,3]):
            choose,used=M.quota_policy(q,1,1.);r=M.run(context(),choose)
            actual=[sum(len(r['order_states'][s,d]) for d in range(M.NR)) for s in (0,1)]
            self.assertEqual(actual,q);self.assertEqual(used,q)
        with self.assertRaises(ValueError):M.quota_policy([1,-1],1,.5)
        choose,used=M.quota_policy([1000,1000],1,1.);M.run(context(),choose)
        self.assertNotEqual(used,[1000,1000])

    def test_full_screen_requires_timing_skill_even_with_positive_pnl(self):
        def data(day,out):
            rows=[]
            for arm in M.ARMS:
                for shift in M.CFGS:
                    rows.append(dict(day=day,market=1,arm=arm,shift=shift,pnl=2. if arm=='VALUE' else 1.,
                        pnl_without_young=2. if arm=='VALUE' else 1.,inventory_square_seconds=1. if arm=='VALUE' else 2.,
                        shares=1.,places=1,pulls=0,posts_bid=1,posts_ask=1,any_quote_steps=1,both_quote_steps=1,grid_steps=1))
            local=pd.DataFrame(dict(day=[day]*800,side=[0]*400+[1]*400,rung=[0]*800,probe_t=np.tile(np.arange(400)*50.,2),
                delta=([-1.]*200+[1.]*200)*2,qpost=([-1.]*200+[1.]*200)*2,post_action=([False]*200+[True]*200)*2))
            return pd.DataFrame(rows),local,dict(source_status={},counts={},excluded_markets=[])
        with mock.patch.object(M,'read_eval',side_effect=data):
            positive,_=M.screen('unused');self.assertTrue(positive['basic_pass'])
            with mock.patch.object(M,'timing_null',return_value={'passes':False}):
                wrong,_=M.screen('unused');self.assertFalse(wrong['basic_pass'])
                self.assertTrue(wrong['absolute_positive_m50'])

    def test_portfolio_null_checks_counts_identity_and_computed_economics(self):
        candidate=pd.DataFrame(dict(day=['x','x'],market=[1,2],posts_bid=[2,3],posts_ask=[1,4],pnl=[1.,1.],pnl_without_young=[1.,1.]))
        draws=pd.concat([candidate.assign(draw=i,pnl=-1.,pnl_without_young=-1.) for i in range(200)],ignore_index=True)
        self.assertTrue(all(c['passes'] for c in M.validate_null_draws(draws,candidate)))
        losing=candidate.assign(pnl=-3.,pnl_without_young=-3.)
        self.assertFalse(any(c['passes'] for c in M.validate_null_draws(draws,losing)))
        for mutation in ('count','identity','missing','duplicate'):
            bad=draws.copy()
            if mutation=='count':bad.loc[0,'posts_bid']+=1
            if mutation=='identity':bad.loc[bad.market==1,'market']=3
            if mutation=='missing':bad=bad.iloc[:-1]
            if mutation=='duplicate':bad=pd.concat([bad,bad.iloc[:1]],ignore_index=True)
            with self.assertRaises(ValueError):M.validate_null_draws(bad,candidate)

    def test_refuses_old_contract_protected_dates_and_partial_rows(self):
        with self.assertRaises(ValueError):M.require(OLD.stamp())
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/M.row_stage('flat',M.BOOT_DAYS[0]);p.mkdir();(p/'rows.parquet').write_text('partial')
            E.write_json(p/'receipt.json',dict(**M.stamp(),day=M.BOOT_DAYS[0],kind='flat',rows_sha256=M.digest(p/'rows.parquet'),partial=True,complete=False))
            with self.assertRaisesRegex(ValueError,'partial'):M.load_rows(tmp,'flat',M.BOOT_DAYS[0])
            with self.assertRaises(FileExistsError):M.stage(tmp,p.name)
        with self.assertRaises(ValueError):M.rows_day('20261109','unused','learned')


if __name__=='__main__':unittest.main()
