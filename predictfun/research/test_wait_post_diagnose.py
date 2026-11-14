import copy
import unittest
import numpy as np
import pandas as pd
import pf_wait_post_diagnose as D
from test_wait_post_v2 import late,teacher
from test_order_harm_client50 import normalized


class DiagnosisTests(unittest.TestCase):
    def test_metric_positive_and_bad(self):
        y=np.arange(20,dtype=float)-9.5
        self.assertEqual(D.metric(y,y,0)['r2_vs_train_mean'],1)
        self.assertLess(D.metric(y,-y,0)['r2_vs_train_mean'],0)
        with self.assertRaises(ValueError):D.metric(y,np.full(20,np.nan),0)

    def test_tail_fires(self):
        self.assertEqual(D.tail(np.ones(100))['top1pct_squared_target_fraction'],.01)
        self.assertGreater(D.tail(np.r_[np.ones(99),1000.])['top1pct_squared_target_fraction'],.99)

    def test_support_known_and_unknown(self):
        d={k:np.zeros(6) for k in D.M.FEATURES};f=pd.DataFrame(d);s=D.Support(f);x=np.zeros(len(D.M.FEATURES))
        self.assertEqual(s.check(x),(6,[]));x[D.M.FEATURES.index('known_position')]=600
        self.assertEqual(s.check(x)[0],0);self.assertIn('known_position',s.check(x)[1])

    def test_guard_off_exact_and_on_changes_exit(self):
        W=late(True)
        for w in W.values():w['vkl']=[1.]*len(w['tg'])
        forecast=teacher(lambda x:1.)
        a=D.M.run(copy.deepcopy(W),D.M.policy(W,forecast));b=D.guard_run(copy.deepcopy(W),forecast)
        for key in ('fl','order_states','endpos','stat'):self.assertEqual(normalized(a[key]),normalized(b[key]))
        for w in W.values():w['vkl']=[-1.]*len(w['tg'])
        c=D.guard_run(W,forecast);self.assertGreater(c['stat']['pulls'][0],0)
        self.assertNotEqual(normalized(a['order_states']),normalized(c['order_states']))

    def test_attribution_and_refusal(self):
        W=late(True);r=D.M.run(W,lambda v:True);f,p=D.attribute(W,r)
        self.assertTrue(f);self.assertAlmostEqual(sum(x['fill_spread']+x['post_fill_move'] for x in f),D.E.metrics(W,r)['pnl'])
        self.assertGreater(p['both_live'],0)
        bad=copy.deepcopy(r);bad['order_states'][0,0][0]['fl'][0]=(bad['order_states'][0,0][0]['fl'][0][0],1.)
        with self.assertRaises(ValueError):D.attribute(W,bad)

    def test_saved_replay_refuses_bad_number(self):
        W=late();r=D.M.run(W,lambda v:True);m=D.E.metrics(W,r);D.reconcile(m,m)
        bad=dict(m,pnl=m['pnl']+1)
        with self.assertRaises(ValueError):D.reconcile(m,bad)

    def test_generation_dedup_and_bad_repeat(self):
        s=D.Support(pd.DataFrame({k:np.zeros(6) for k in D.M.FEATURES}));a=D.EntryAudit({0:s,1:s});x=np.zeros(len(D.M.FEATURES))
        v=dict(side=0,rung=0,generation=0,t=0,kind='entry');a.capture(v,x,(-1,0),False,'eligible');a.capture(dict(v,t=100),x,(1,0),True,'eligible')
        self.assertEqual(len(a.units),1);u=next(iter(a.units.values()));self.assertEqual(u['post_delay'],100);self.assertFalse(u['first_post'])
        with self.assertRaises(ValueError):a.capture(dict(v,t=150),x,(1,0),True,'eligible')


if __name__=='__main__':unittest.main()
