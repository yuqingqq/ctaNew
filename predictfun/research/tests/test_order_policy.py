"""Positive and falsifier controls for own-order continuation and calm entry."""
import copy
import types
import unittest
import numpy as np
import pandas as pd
from predictfun.research import pf_order_policy_spec as S
from predictfun.research.pf_order_policy_inputs import Window,SafeReader,FEATURES,MARKET,micro_price
from predictfun.research.pf_order_policy_hooks import Hook
from predictfun.research.pf_order_policy_shadow import extend
from predictfun.research.pf_order_policy_engine import source,replace
from predictfun.research.pf_order_policy_verify import reconcile,exact
from predictfun.research.pf_order_policy_fit import fit
from predictfun.research.pf_order_policy_report import null,bootstrap
from predictfun.research.pf_model_spec import SETS


def fixture():
    t=np.arange(0.,15001.,50.);x=np.zeros((len(t),len(SETS['ALL'])))
    x[:,SETS['ALL'].index('bn_ntr1000')]=10
    f=[]
    for tt in t:
        row={'t_ms':tt}
        for k in range(1,6):row.update({f'b{k}_p':50-k,f'a{k}_p':50+k,f'b{k}_s':100.,f'a{k}_s':100.})
        f.append(row)
    inputs=types.SimpleNamespace(by={1:(t,x)})
    return inputs,f,t


def order(**kw):
    o=dict(side=0,lane=0,pt=49.,rem=5.,size=5.,ahead=0.,ci=120,t0=6000.,kq=0.);o.update(kw);return o


def hook(mode='hold_rule',phase='build'):
    inputs,f,t=fixture();parent=types.SimpleNamespace(inputs=inputs,day='20261023',timing='X',phase=phase,model=None,windows=[],save=lambda m,r:None)
    h=Hook(parent,mode,1,0.,f,t,[],[],lambda t:(False,False),lambda t,s:.995,lambda t:1.)
    r=dict(side=0,lane=0,pt=49.,size=5.,t_send=5960.,t_act=6000.,cancelled=False)
    h.sent(r);o=order();h.activated(o,r)
    return h,o


class Controls(unittest.TestCase):
    def test_frozen_source_and_bad_anchor(self):
        for q in ('M','L'):compile(source(q),'instrumented','exec')
        with self.assertRaises(ValueError):replace('abcabc','abc','x')

    def test_calm_requires_history_and_known_inputs(self):
        inputs,f,t=fixture();w=Window(inputs,1,f,t,0.)
        self.assertFalse(w.calm(1000));self.assertTrue(w.calm(6500))
        inputs.by[1][1][:,SETS['ALL'].index('bn_age')]=np.nan
        self.assertFalse(Window(inputs,1,f,t,0.).calm(6500))

    def test_arithmetic_activity_and_future_invariance(self):
        inputs,f,t=fixture();x=inputs.by[1][1];col=SETS['ALL'].index('bn_ntr1000');j=130
        x[j,col]=30;w=Window(inputs,1,f,t,0.)
        self.assertAlmostEqual(w.ratio[j],31/11);self.assertFalse(w.calm(t[j]))
        before=w.calms[:j+1].copy();x[j+1:]=1000
        np.testing.assert_array_equal(Window(inputs,1,f,t,0.).calms[:j+1],before)

    def test_price_identity_and_thinning(self):
        inputs,f,t=fixture();f[128]['b1_p']=48.
        self.assertFalse(Window(inputs,1,f,t,0.).calm(6500))
        inputs,f,t=fixture();f[130]['b1_s']=79.
        self.assertFalse(Window(inputs,1,f,t,0.).calm(6500))

    def test_strict_marks_do_not_backfill(self):
        inputs,f,t=fixture();f[130]['a1_p']=f[130]['b1_p']
        w=Window(inputs,1,f,t,0.);self.assertTrue(np.isnan(w.mid(6500)))
        self.assertTrue(np.isnan(w.mid(20000)))

    def test_features_not_latent_queue_or_unnotified_fills(self):
        inputs,f,t=fixture();w=Window(inputs,1,f,t,0.);o=order()
        a=w.features(6500,o,0,0,.995,1);o['ahead']=999;o['rem']=0
        self.assertEqual(a,w.features(6500,o,0,0,.995,1))
        o['kq']=2;self.assertEqual(w.features(6500,o,0,0,.995,1)['remaining'],3)

    def test_preserved_queue_and_exact_cancel_boundary(self):
        inputs,f,t=fixture();w=Window(inputs,1,f,t,0.)
        trades=[(6522.,0,49.,5.,True),(6600.,0,49.,5.,True)]
        fs=extend(w,order(),[],[],trades,6522,6600,'L');self.assertEqual(sum(x[1] for x in fs),5)
        self.assertEqual(extend(w,order(ahead=100),[],[],trades,6522,6600,'L'),[])
        self.assertEqual(extend(w,order(),[],[],trades,6600,6600,'L'),[])

    def test_displayed_cap_stress_and_competing_pending_cancel(self):
        inputs,f,t=fixture()
        for row in f[125:]:row['b1_s']=0.
        w=Window(inputs,1,f,t,0.);trades=[(6600.,0,49.,5.,True)]
        self.assertEqual(sum(x[1] for x in extend(w,order(ahead=100),[],[],trades,6500,7000,'M')),5)
        self.assertEqual(extend(w,order(ahead=100),[],[],trades,6500,7000,'L'),[])
        pending=dict(side=0,lane=2,pt=50.,size=5.,t_act=6550.,t_rm=6700.,cancelled=True)
        self.assertEqual(extend(w,order(),[],[pending],trades,6500,7000,'L'),[])

    def test_competing_priority_volume_conserved(self):
        inputs,f,t=fixture();w=Window(inputs,1,f,t,0.)
        older=order(lane=2,t_end=6700.,t0=5000.)
        fs=extend(w,order(),[older],[],[(6600.,0,49.,8.,True)],6500,7000,'L')
        self.assertEqual(sum(x[1] for x in fs),3)

    def test_hold_one_generation_one_second_and_hard_stop(self):
        h,o=hook();p,size,keep=h.action(6500,0,None,0,o,False,0,[o]);self.assertEqual((p,size,keep),(49,5,True))
        self.assertEqual(h.action(6550,0,None,0,o,False,0,[o]),(49,5,False))
        self.assertEqual(h.action(7500,0,None,0,o,False,0,[o]),(None,0,False))
        self.assertEqual(h.action(7600,0,None,0,o,False,0,[o]),(None,0,False))
        h,o=hook();h.action(6500,0,None,0,o,False,0,[o]);h.guard=lambda t:(True,False)
        self.assertEqual(h.action(6550,0,None,0,o,False,0,[o]),(None,0,False))

    def test_size_pending_inventory_and_stale_refusals(self):
        for field,value in [('size',6),('lane',2)]:
            h,o=hook();o[field]=value
            self.assertFalse(h.action(6500,int(o['lane']),None,0,o,False,0,[o])[2])
        h,o=hook();self.assertFalse(h.action(6500,0,None,0,o,True,0,[o])[2])
        h,o=hook();self.assertFalse(h.action(6500,0,None,0,o,False,10,[o])[2])
        h,o=hook();h.w.x[:,SETS['ALL'].index('bn_age')]=1001
        self.assertFalse(h.action(6500,0,None,0,o,False,0,[o])[2])

    def test_seed_pending_fixed_size_and_expiry(self):
        h,o=hook('calm_seed');self.assertTrue(h.seed_flag(6500,0,None,.995,1));self.assertEqual(h.seed_size(6500,0,25),5)
        pending=dict(_seed=True,t_act=6540.)
        self.assertTrue(h.seed_flag(6520,0,pending,.995,1));self.assertEqual(h.seed_size(6520,0,25),5)
        self.assertFalse(h.seed_flag(7540,0,pending,.995,1))
        self.assertFalse(h.seed_flag(6500,0,None,.98,1))
        self.assertFalse(h.seed_flag(60000,0,None,.995,1))

    def test_filter_only_new_increasing(self):
        h,o=hook('calm_filter');self.assertFalse(h.allow_new(1000,0,0));self.assertTrue(h.allow_new(1000,0,-1))
        self.assertTrue(h.allow_new(6500,0,0))

    def test_label_latency_prefix_and_zero_outcomes(self):
        h,o=hook('reference');h.action(6500,0,None,0,o,False,0,[o]);h.cancelling(o,6522)
        r=h.records[o['_gid']];h.label(r,order(rem=0),[],[],6522);self.assertEqual(r['label'],0)
        h,o=hook('reference');h.action(6500,0,None,0,o,False,0,[o]);r=h.records[o['_gid']]
        h.trades=[(6521.,0,49.,5.,True),(6600.,0,49.,3.,True)];h.tt=np.array([6521.,6600.]);h.label(r,order(rem=3),[],[],6522)
        self.assertEqual(r['label_q'],3);self.assertAlmostEqual(r['label'],3-.09)

    def test_protected_reader_refusal(self):
        with self.assertRaises(ValueError):SafeReader('20261104',lambda *a,**k:None)
        safe=SafeReader('20261103',lambda *a,**k:None)
        with self.assertRaises(ValueError):safe('/data/step1/20261104/registry.parquet')

    def test_chronology_and_planted_incremental_signal(self):
        rng=np.random.default_rng(7);n=4000
        a=pd.DataFrame(rng.normal(size=(n,len(FEATURES))),columns=FEATURES);a['day']='20261023';a['label']=5*(a.queue_now>0)-2.5
        m=fit(a,'20261024',FEATURES);q=m.predict(a[FEATURES],num_threads=1)
        self.assertGreater(np.corrcoef(q,a.label)[0,1],.9)
        with self.assertRaises(ValueError):fit(a,'20261023',FEATURES)

    def test_reconciler_and_known_bad_artifact(self):
        w=pd.DataFrame([dict(mid=1,pnl=1.02,net_end=2.)]);f=pd.DataFrame([dict(mid=1,t=1.,side=0,pt=49.,q=2.,settle=1.02)])
        q=pd.DataFrame([dict(mid=1,kind='place')]);ind=pd.DataFrame([dict(mid=1,cash=-.98,inventory_square_s=239.1*4)])
        g=pd.DataFrame([dict(id='one',entry_q=2.,entry_settle=102.,status='eligible',label=0.,label_q=0.,postcancel_rem=3.,cancel_effective=2.,decision=1.,remaining=5.)])
        reconcile(w,f,q,ind,g,1)
        bad=f.copy();bad.loc[0,'pt']=50
        with self.assertRaises(ValueError):reconcile(w,bad,q,ind,g,1)
        with self.assertRaises(AssertionError):exact(f,bad)

    def test_matched_null_and_day_refusal(self):
        a=pd.DataFrame(dict(day=['d']*40,side=[0,1]*20,hour=[0]*40));k=np.arange(40)%4==0;v=k.astype(float)
        d=null(a,k,v,['day','side','hour']);self.assertEqual(d['kept'],10);self.assertLess(d['p_ge'],.05)
        with self.assertRaises(ValueError):bootstrap([1,2,3,4])


if __name__=='__main__':unittest.main()
