"""Positive controls and known-bad refusals for queue continuation research."""
import copy
import unittest
import numpy as np
import pandas as pd
from predictfun.research.pf_model_spec import SETS
from predictfun.research.pf_queue_model_sim import Tape, simulate, features, marks, MARKET
from predictfun.research.pf_queue_model_spec import CONFIGS
from predictfun.research.pf_queue_model_data import admit, load_day
from predictfun.research.pf_queue_model_fit import fit, choices, rejoin_allowed, null_test
from predictfun.research.pf_queue_model_verify import verify_window,verify_selection


def tape(trades=(),queue=10.):
    t=np.arange(0.,70001.,50.);n=len(t)
    bid=np.tile(np.arange(49.,44.,-1),(n,1));ask=np.tile(np.arange(51.,56.),(n,1))
    x=np.zeros((n,len(SETS['ALL'])));x[:,SETS['ALL'].index('pf_mid')]=50.
    a=np.asarray(trades,float).reshape(-1,4)
    return Tape(t,t.copy(),bid,ask,np.full((n,5),queue),np.full((n,5),queue),
                a[:,0],a[:,1],a[:,2],a[:,3],t.copy(),x,np.ones((n,2)))


C=dict(rule='L',shift=0.,place=40.,cancel=22.,notice=100.)


class QueueControls(unittest.TestCase):
    def sim(self,t,action='keep',side=0,p=49.,decision=1040.,c=None):
        return simulate(t,side,p,0.,decision,c or C,action)

    def qty(self,r):return sum(f[1] for f in r['fills'])

    def test_queue_progress_and_rejoin_cost(self):
        t=tape([(500,0,49,10),(1500,0,49,5)])
        keep=self.sim(t);cancel=self.sim(t,'cancel');rejoin=self.sim(t,'rejoin')
        self.assertEqual([self.qty(r) for r in (keep,cancel,rejoin)],[5,0,0])
        self.assertGreater(marks(t,keep,4000,1)['value'],marks(t,cancel,4000,1)['value'])
        # Known-bad reset-to-back counterfactual must lose the preserved queue benefit.
        self.assertNotEqual(keep['fills'],rejoin['fills'])

    def test_toxic_keep_and_side_mirror(self):
        t=tape([(1500,0,49,20)])
        t.bp[t.tf>=2000]-=5;t.ap[t.tf>=2000]-=5
        self.assertLess(marks(t,self.sim(t),4000,0)['value'],marks(t,self.sim(t,'cancel'),4000,0)['value'])
        mirrored=tape([(1500,1,51,20)])
        self.assertEqual(self.qty(self.sim(mirrored,side=1,p=51)),5)
        self.assertEqual(marks(mirrored,self.sim(mirrored,side=1,p=51),4000,0)['value'],5)

    def test_cancel_latency_and_exact_boundary(self):
        t=tape([(1050,0,49,12),(1062,0,49,3)])
        self.assertEqual(self.qty(self.sim(t,'cancel')),2)
        self.assertEqual(self.qty(self.sim(t)),5)
        self.assertEqual(self.sim(t,'cancel')['orders'][0]['remove'],1062)

    def test_notice_and_rejoin_pending_reservation(self):
        t=tape([(1000,0,49,12),(1500,0,49,20)])
        r=self.sim(t,'rejoin');x,status=features(t,0,49,0,1040,C,self.sim(t))
        self.assertEqual(status,'eligible');self.assertEqual(x['known_remaining'],5)
        self.assertEqual(r['orders'][1]['size'],5);self.assertEqual(self.qty(r),7)
        rows=pd.DataFrame([dict(side=0,inventory=20,known_remaining=5,episode=5),
                           dict(side=1,inventory=5,known_remaining=5,episode=185),
                           dict(side=1,inventory=10,known_remaining=5,episode=185)])
        self.assertEqual(rejoin_allowed(rows).tolist(),[False,False,True])

    def test_postonly_and_pending_create_cancel(self):
        t=tape([(70,0,49,20)])
        t.x[:,SETS['ALL'].index('bn_r250')]=-1
        r=self.sim(t,'cancel');self.assertEqual(r['orders'][0]['remove'],95)
        self.assertEqual(self.qty(r),5)
        t=tape();t.ap[:]=48
        self.assertTrue(self.sim(t)['orders'][0]['refused'])

    def test_rejoin_minimum_notional(self):
        t=tape([(500,0,49,14),(1500,0,49,20)])
        r=self.sim(t,'rejoin');self.assertEqual(r['orders'][1]['size'],1)
        self.assertTrue(r['orders'][1]['refused']);self.assertEqual(self.qty(r),4)

    def test_L_M_bracket(self):
        t=tape([(1500,0,49,5)])
        t.bs[t.tf>=500,0]=0
        self.assertEqual(self.qty(self.sim(t)),0)
        self.assertEqual(self.qty(self.sim(t,c={**C,'rule':'M'})),5)

    def test_causal_features_no_latent_or_future_input(self):
        t=tape([(1000,0,49,12)])
        r=self.sim(t);a,status=features(t,0,49,0,1040,C,r)
        self.assertEqual(status,'eligible')
        t.x[t.rt>1040]=999;t.bs[t.known>1040]=999
        r['orders'][0]['ahead']=999;r['orders'][0]['initial']=999
        r['fills'].append((1100,3,False,0,1100))
        b,_=features(t,0,49,0,1040,C,r)
        for k in a:np.testing.assert_allclose(a[k],b[k],equal_nan=True)
        self.assertEqual(a['known_remaining'],5)

    def test_stale_invalid_missing_state(self):
        t=tape();t.bp[t.tf==4000,0]=99
        self.assertTrue(np.isnan(t.midpoint(4000)))
        self.assertTrue(np.isnan(t.midpoint(80000)))
        t.rt=t.rt[t.rt<100];t.x=t.x[:len(t.rt)]
        self.assertEqual(t.hard_stop(0,49,0,5000),200)

    def test_bad_clocks_and_protected_dates(self):
        t=tape();t.validate();t.tf[1]=-1
        with self.assertRaises(ValueError):t.validate()
        with self.assertRaises(ValueError):next(load_day('20261109'))
        with self.assertRaises(ValueError):fit(pd.DataFrame({'day':['20261024']}),'20261024',MARKET)
        with self.assertRaises(ValueError):null_test(pd.DataFrame(),199)

    def test_fixed_size_reduce_only_no_reversal(self):
        self.assertTrue(admit(5,1,185));self.assertFalse(admit(2,1,185))
        self.assertFalse(admit(-2,0,185));self.assertTrue(admit(-5,0,185))
        self.assertFalse(admit(10,0,5));self.assertTrue(admit(9,0,5))

    def test_artifact_checker_positive_and_falsifiers(self):
        f=pd.DataFrame([dict(t=1000.,side=0,q=5.,price=49.,id='a',action='keep',fill_index=0,through=False,generation=0,anchor=1000.)])
        r=dict(net=5.,cash_c=-245.,stop_value_c=5.,settle_value_c=255.,inventory_square_s=5975.,terminal_abs=5.,max_inventory=5.,shares=5.,fills=1)
        verify_window(r,f,50.,1.);verify_selection(f,f)
        with self.assertRaises(ValueError):verify_window({**r,'cash_c':-244},f,50.,1.)
        bad=f.copy();bad.loc[0,'price']=48
        with self.assertRaises(ValueError):verify_selection(bad,f)


if __name__=='__main__':unittest.main()
