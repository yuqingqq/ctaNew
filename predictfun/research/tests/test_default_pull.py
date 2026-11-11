"""Positive controls and explicit bad-input/mutation refusals for default quoting."""
import types
import unittest
import numpy as np
import pandas as pd
from predictfun.research import pf_default_pull_hooks as H
from predictfun.research.pf_default_pull_shadow import paired_leg
from predictfun.research.pf_default_pull_engine import source
from predictfun.research.pf_default_pull_verify import actions
from predictfun.research.pf_own_order_warmup_v3 import CausalWindow,controls
from predictfun.research.pf_order_policy_engine import replace
from predictfun.research.pf_order_policy_inputs import SafeReader
from predictfun.research.pf_order_policy_verify import exact
from predictfun.research.pf_order_policy_report import null,bootstrap
from predictfun.research.tests.test_order_policy import fixture,order

def hook(mode='default',phase='build'):
    inputs,f,t=fixture();parent=types.SimpleNamespace(inputs=inputs,day='20261023',timing='X',phase=phase,models={},windows=[],save=lambda m,r:None)
    h=H.Hook(parent,mode,1,0.,f,t,[],[],lambda t:(False,False),lambda t,s:.01,lambda t:.05)
    r=dict(side=0,lane=0,pt=49.,size=5.,t_send=5960.,t_act=6000.,cancelled=False)
    h.sent(r);o=order();h.activated(o,r)
    return h,o

class Controls(unittest.TestCase):
    def test_sources_and_anchor_refusal(self):
        for q in ('M','L'):compile(source(q),'generated','exec')
        with self.assertRaises(ValueError):replace('xx','x','z')

    def test_warmup_future_mutation(self):controls()

    def test_pressure_unknown_and_direction(self):
        x=dict(side=0,pf_imb1=0.,bn_imb=0.,bn_flow1000=0.,d_bnimb250=0.,activity_ratio=1.)
        self.assertFalse(H.pressure(x,1));self.assertTrue(H.pressure(x,.7))
        x['bn_flow1000']=-1;x['activity_ratio']=3
        self.assertTrue(H.pressure(x,1));x['side']=1;self.assertFalse(H.pressure(x,1))
        x['activity_ratio']=np.nan;self.assertIsNone(H.pressure(x,1))

    def test_same_price_not_top_level_thinning(self):
        inputs,f,t=fixture();f[130]['b1_p']=50.;f[130]['b1_s']=10.;f[130]['b2_p']=49.;f[130]['b2_s']=100.
        w=CausalWindow(inputs,1,f,t,0.)
        self.assertEqual(H.thin_ratio(w,6500,0,49),1.)
        f[130]['b2_s']=70.;w=CausalWindow(inputs,1,f,t,0.)
        self.assertEqual(H.thin_ratio(w,6500,0,49),.7)
        self.assertTrue(np.isnan(H.thin_ratio(w,100,0,49)))

    def test_default_vs_guard_and_no_latent_features(self):
        h,o=hook();known={0:o,1:None,2:None,3:None};fl={i:[] for i in known}
        a=h.targets(6500,[(None,0)]*4,known,fl,[o],0)
        self.assertEqual(a[0],(49,5));o['rem']=0;o['ahead']=999
        self.assertEqual(a,h.targets(6500,[],known,fl,[o],0))
        h,o=hook('matched_guard');known[0]=o
        self.assertEqual(h.targets(6500,[],known,fl,[o],0)[0],(None,0))

    def test_reservations_and_partial_fill_retention(self):
        h,o=hook();known={0:o,1:None,2:None,3:None};fl={i:[] for i in known}
        o['kq']=2
        self.assertEqual(h.targets(6500,[],known,fl,[o],2)[0],(49,5))
        zombie=order(size=8,kq=0,t_end=6600)
        self.assertEqual(h.targets(6500,[],known,fl,[o,zombie],2)[0],(None,0))
        known[0]=None
        self.assertEqual(h.targets(6500,[],known,fl,[zombie],2)[0],(None,0))

    def test_probe_one_generation_no_outcome_selection(self):
        h,o=hook();known={0:o,1:None,2:None,3:None};fl={i:[] for i in known}
        h.records[o['_gid']]['probe_age']=100
        h.targets(6500,[],known,fl,[o],0);h.targets(6600,[],known,fl,[o],0)
        self.assertEqual(len(h.probes),1);self.assertEqual(h.records[o['_gid']]['decision'],6500)
        self.assertEqual(H.probe_age('abc'),H.probe_age('abc'))

    def test_model_threshold_and_cooldown(self):
        h,o=hook('queue',phase='eval');h.parent.models['queue']=types.SimpleNamespace(predict=lambda *a,**k:np.array([-1.]))
        known={0:o,1:None,2:None,3:None};fl={i:[] for i in known}
        self.assertEqual(h.targets(6500,[],known,fl,[o],0)[0],(None,0))
        self.assertEqual(h.cool[0],7500);known[0]=None
        self.assertEqual(h.targets(7000,[],known,fl,[],0)[0],(None,0))
        self.assertEqual(h.targets(7500,[],known,fl,[],0)[0],(49,5))

    def test_shadow_prefix_boundary_queue_reset(self):
        inputs,f,t=fixture();w=CausalWindow(inputs,1,f,t,0.)
        trades=[(6521.,0,49.,5.,True),(6522.,0,49.,2.,True),(7600.,0,49.,4.,True)]
        keep=paired_leg(w,order(),[],[],trades,6522,8500,'L')
        reset=paired_leg(w,order(),[],[],trades,6522,8500,'L',7540,5)
        self.assertEqual(sum(x[1] for x in keep),5);self.assertEqual(keep[0][0],6522)
        self.assertEqual(reset,[])
        # Planted reset with retained queue would fill4; the reset instrument must distinguish it.
        bad=paired_leg(w,order(),[],[],trades,7540,8500,'L')
        self.assertEqual(sum(x[1] for x in bad),4)
        with self.assertRaises(ValueError):paired_leg(w,order(),[],[],trades,6522,8500,'Z')

    def test_shadow_pending_cancel_priority(self):
        inputs,f,t=fixture();w=CausalWindow(inputs,1,f,t,0.)
        older=order(t_end=6700,t0=5000)
        fs=paired_leg(w,order(),[older],[],[(6600.,0,49.,8.,True)],6500,7000,'L')
        self.assertEqual(sum(x[1] for x in fs),3)
        bad=paired_leg(w,order(),[],[],[(6600.,0,49.,8.,True)],6500,7000,'L')
        self.assertEqual(sum(x[1] for x in bad),5)

    def test_artifact_identity_and_duplicate_refusal(self):
        a=pd.DataFrame([dict(id='x',status='eligible',decision=100.,t_act=0.,probe_age=100.,cancel_effective=122.,
            label_q=0.,postcancel_rem=5.,reset_q=0.,initial_size=5.,label=0.,keep_value=0.,reset_value=0.,
            d_bnimb250=np.nan,d_pfimb250=np.nan,t_rel=.1)])
        actions(a);exact(a,a.copy())
        with self.assertRaises(ValueError):actions(pd.concat([a,a],ignore_index=True))
        b=a.copy();b.loc[0,'label']=1
        with self.assertRaises(ValueError):actions(b)
        with self.assertRaises(AssertionError):exact(a,b)

    def test_null_and_day_units(self):
        a=pd.DataFrame({'day':['d']*200,'side':[0]*200});values=np.arange(200.)
        r=null(a,values>=100,values,['day','side'])
        self.assertEqual(r['draws'],500);self.assertEqual(r['kept'],100);self.assertLess(r['p_ge'],.01)
        with self.assertRaises(ValueError):null(a,values>100,np.full(200,np.nan),['day'])
        self.assertEqual(bootstrap([1]*6),[6,6])
        with self.assertRaises(ValueError):bootstrap([1]*4)

    def test_protected_day_refusal(self):
        SafeReader('20261103',lambda *a,**k:None)
        with self.assertRaises(ValueError):SafeReader('20261104',lambda *a,**k:None)

if __name__=='__main__':unittest.main()
