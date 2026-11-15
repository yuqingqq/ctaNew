"""Positive controls and known-bad refusals for the local entry instrument."""
import copy
import unittest
from unittest import mock
import numpy as np
import pandas as pd
import pf_entry_local as P
import pf_joint_ladder as J


def context():
    W={0:J._ctx(0,[(0.,[(50,100),(49,100),(48,100),(47,100),(46,100)],52)],[(200.,50,125.),(400.,45,50.)],t_end=1000.),
       1:J._ctx(1,[(0.,[(52,100),(53,100),(54,100),(55,100),(56,100)],50)],[],t_end=1000.)}
    for s,w in W.items():
        w['m']=1; w['ws']=0.; w['vkl']=[1.]*len(w['tg'])
        w['X']=np.zeros((len(w['tg']),len(P.K.G32)+5))
        w['X'][:,P.K.G32.index('sigma')]=.01
        w['X'][:,P.K.G32.index('pf_mid')]=51.
        w['tf_mff']=(np.array([0.,1200.,4000.,8000.]),np.array([51.,51.,52.,53.]))
    return W


class Checks(unittest.TestCase):
    def test_tranche_clock_sign_and_zeros(self):
        W=context()
        for w in W.values(): w['te']=7000.
        v=dict(t=0.,side=0,price=50,activation=40.)
        order=dict(ts=0.,ta=40.,pt=50,tr=float('inf'),ref=False,fl=[(200.,10.),(3000.,20.)])
        y=P.label(W,order,v,51.)
        self.assertEqual(y['status'],'eligible'); self.assertEqual(y['fill_fraction'],1.)
        self.assertAlmostEqual(y['local_value'],5/3)
        self.assertAlmostEqual(y['drift_value'],2/3)
        other=P.label(W,dict(order,fl=[]),v,51.)
        self.assertEqual(other['local_value'],0.); self.assertEqual(other['fill_fraction'],0.)
        ask=P.label(W,dict(order,pt=52),dict(v,side=1,price=52),51.)
        self.assertAlmostEqual(ask['local_value'],1/3)
        with self.assertRaisesRegex(ValueError,'identity'):
            P.label(W,dict(order,ts=1.),v,51.)
        with self.assertRaisesRegex(ValueError,'tranche'):
            P.label(W,dict(order,fl=[(20.,1.)]),v,51.)

    def test_horizon_status_and_future_mark_refusal(self):
        W=context(); v=dict(t=0.,side=0,price=50,activation=40.)
        o=dict(ts=0.,ta=40.,pt=50,tr=float('inf'),ref=False,fl=[])
        self.assertEqual(P.label(W,o,v,51.)['status'],'insufficient_horizon')
        for w in W.values(): w['te']=7000.
        o['fl']=[(5000.,10.)]
        self.assertEqual(P.label(W,o,v,51.)['status'],'missing_future_mark')

    def test_features_and_engine_reconciliation(self):
        W=context(); views=[]
        r=P.run(W,lambda v:True,observe=lambda v,a:views.append(v) if v['kind']=='entry' and v['status']=='eligible' else None)
        legacy=P.N.run(copy.deepcopy(W),lambda v:True)
        self.assertEqual(r['fl'],legacy['fl']); self.assertEqual(r['policy_decisions'],legacy['policy_decisions'])
        x,mid=P.vector(W,views[0]); self.assertEqual(len(x),len(P.FEATURES)); self.assertEqual(mid,51.)
        P.future_check(W,views[0],x)
        bad=dict(views[0],t=views[0]['t']+1.)
        with self.assertRaisesRegex(ValueError,'timestamp'): P.vector(W,bad)
        # Block every entry: the instrument must observe the action change.
        self.assertEqual(len(P.run(W,lambda v:False)['fl']),0)
        self.assertGreater(len(r['fl']),0)

    def test_unique_basis_and_no_future_dependency(self):
        t=np.array([0.,1.,2.,3.,4.,5.]); values=np.array([1.,1.,1.,2.,2.,2.])
        full=P.DATA.relay_basis(t,values,np.array([4.]),5)
        prefix=P.DATA.relay_basis(t[:5],values[:5],np.array([4.]),5)
        np.testing.assert_equal(full,prefix)
        # Planted overlap duplication biases the median and must be detected.
        dup=P.DATA.relay_basis(np.r_[t[:5],3.,4.][np.argsort(np.r_[t[:5],3.,4.])],
            np.r_[values[:5],2.,2.][np.argsort(np.r_[t[:5],3.,4.])],np.array([4.]),5)
        self.assertNotEqual(float(full[0]),float(dup[0]))

    def test_pre_read_boundary_guard(self):
        touched=[]
        def read(p,*a,**kw): touched.append(p); return None
        def load(day):
            self.assertTrue(P.K.SKIP_FOREIGN_PARTS); self.assertEqual(P.K.PREROLL_MS,30000.)
            pd.read_parquet('/synthetic/books/part_20261028_00.parquet')
        with mock.patch.object(pd,'read_parquet',read),mock.patch.object(P.K,'load_day',load):
            with self.assertRaisesRegex(ValueError,'before read'): P.DATA.load_day('20261027')
        self.assertEqual(touched,[])
        with self.assertRaisesRegex(ValueError,'undeclared'): P.DATA.load_day('20261104')

    def test_fair_assignment_precision_and_identity(self):
        x=pd.DataFrame(dict(tau=[0.,50.],gap=np.array([1.,2.],np.float32),basis_bps=np.array([3.,4.],np.float32)))
        c=([0.,50.],None,None,None,np.array([1.123456789,2.]),np.array([3.123456789,4.]))
        P.DATA.apply_correction(x,c)
        np.testing.assert_equal(x.gap.to_numpy(),c[4]); np.testing.assert_equal(x.basis_bps.to_numpy(),c[5])
        with self.assertRaisesRegex(ValueError,'timestamp'):
            P.DATA.apply_correction(x,([0.,51.],*c[1:]))

    def test_keeper_is_order_depth_specific(self):
        W=context(); arrays={(1,0):np.array([[.9,.1,.1,.1,.1,.1]]),(1,1):np.array([[.9,.1,.1,.1,.1,.1]])}
        cancel=P.stop_rule(W,arrays)
        self.assertTrue(cancel(dict(side=0,k=0,price=50)))
        self.assertTrue(cancel(dict(side=1,k=0,price=52)))
        self.assertFalse(cancel(dict(side=0,k=0,price=49)))
        self.assertFalse(cancel(dict(side=1,k=0,price=53)))
        self.assertFalse(cancel(dict(side=0,k=0,price=44)))
        self.assertFalse(P.stop_rule(W,{})(dict(side=0,k=0,price=50)))

    def test_queue_availability_positive_and_empty(self):
        W=context(); posted=P.run(W,lambda v:True); empty=P.run(W,lambda v:False)
        a=P.path_metrics(W,posted); b=P.path_metrics(W,empty)
        self.assertGreater(a['quote_live_ms'],0.); self.assertLessEqual(a['quote_live_ms'],a['quote_possible_ms'])
        self.assertEqual(b['quote_live_ms'],0.); self.assertEqual(b['inventory_squared_seconds'],0.)

    def test_matched_null_positive_and_bad_input(self):
        n=200
        f=pd.DataFrame(dict(day=['20261027']*n,market=np.arange(n),side=[0]*n,rung=[0]*n,generation=[0]*n,
            probe_t=[1000.]*n,sampling_weight=[2.]*n,post_action=np.arange(n)<100,local_value=np.where(np.arange(n)<100,1.,-1.)))
        self.assertTrue(P.matched(f)['passes'])
        self.assertFalse(P.matched(f.assign(post_action=~f.post_action))['passes'])
        with self.assertRaises(ValueError): P.matched(f,10)
        with self.assertRaises(ValueError): P.matched(pd.concat([f,f.iloc[:1]]))


if __name__=='__main__': unittest.main()
