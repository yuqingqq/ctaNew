import unittest
from unittest.mock import patch
import pandas as pd
from predictfun.research import pf_late_engine as e
from predictfun.research.pf_late_run import pre60_equal,reconcile_metadata


class LateControls(unittest.TestCase):
    def test_early_exact_and_late_reservation(self):
        wants=[(None,0),(51,25),(None,0),(52,25)]
        self.assertEqual(e.plan(10,wants,[None]*4,[],'late_reserved',59.999),tuple(wants))
        self.assertEqual(e.plan(10,wants,[None]*4,[],'late_reserved',60),((None,0),(51,10),(None,0),(None,0)))
        with self.assertRaises(ValueError):e.assert_admission(10,1,11,[],'late_reserved',60)
        self.assertEqual(e.admission(10,1,25,[],'late_reserved',59.99),25)
        self.assertEqual(e.admission(-1,1,25,[],'late_reserved',60),0)
        with self.assertRaises(ValueError):e.assert_admission(-1,1,1,[],'late_reserved',60)

    def test_pending_cancels_and_retention(self):
        o=dict(side=1,pt=51,size=10,kq=8,rem=0,cancelled=True)
        self.assertEqual(e.admission(2,1,5,[o,o],'late_reserved',120),0)
        self.assertEqual(e.plan(2,[(None,0),(51,5)],[None,o],[o],'late_reserved',120),((None,0),(51,10)))
        with self.assertRaises(ValueError):e.plan(1,[(None,0),(51,1)],[],[],'bad',60)

    def test_cancel_history_boundary(self):
        ws=1790000000000.;ack=ws+59990;expiry=e.release(ack,ws+59968,100,'late_reserved')
        self.assertFalse(e.visible(expiry,ws+59999,ws,100,'late_reserved',ack))
        self.assertTrue(e.visible(expiry,ws+60000,ws,100,'late_reserved',ack))
        self.assertTrue(e.visible(expiry,ws+60089,ws,100,'late_reserved',ack))
        self.assertFalse(e.visible(expiry,ws+60091,ws,100,'late_reserved',ack))
        self.assertEqual(e.release(ack,ws,100,None),ack)
        with self.assertRaises(ValueError):e.visible(expiry,ws+59999,ws,100,'late_reserved')

    def test_target_and_price_engagement(self):
        self.assertEqual([e.target(t) for t in [60,150,240]],[10,5,0])
        self.assertEqual(e.price(6,1,52,49,52,150,'late_taper1'),51)
        self.assertEqual(e.price(-6,0,49,49,52,150,'late_taper1'),50)
        self.assertEqual(e.price(5,1,52,49,52,150,'late_taper1'),52)
        self.assertEqual(e.price(50,1,52,49,52,59.9,'late_taper1'),52)
        self.assertEqual(e.price(50,1,52,49,52,240,'late_taper1'),52)
        self.assertEqual(e.price(50,1,52,49,52,150,'late_reserved'),52)

    def test_guard_and_cross_refusal(self):
        self.assertIsNone(e.price(50,1,None,49,52,150,'late_taper1'))
        self.assertEqual(e.price(50,1,50,49,50,150,'late_taper1'),50)
        self.assertEqual(e.price(50,0,49,49,52,150,'late_taper1'),49)
        with self.assertRaises(ValueError):e.price(float('nan'),1,52,49,52,150,'late_taper1')

    def test_source_identity(self):
        compile(e.source(),'late','exec')
        with patch('predictfun.research.pf_late_engine.sha',return_value='wrong'):
            with self.assertRaises(ValueError):e.source()

    def test_prefix_checker(self):
        a=pd.DataFrame(dict(mid=[1,1],t=[59.,61.],q=[5.,7.]))
        b=a.copy();b.loc[1,'q']=8
        pre60_equal(a,b,'fills',None)
        b.loc[0,'q']=6
        with self.assertRaises(AssertionError):pre60_equal(a,b,'fills',None)
        with self.assertRaises(ValueError):pre60_equal(a,b,'bad',None)
        q=pd.DataFrame(dict(mid=[1,1],t_ms=[59000.,61000.],kind=['place','place']))
        r=q.copy();r.loc[1,'kind']='cancel'
        pre60_equal(q,r,'requests',pd.Series({1:0.}))
        r.loc[0,'kind']='cancel'
        with self.assertRaises(AssertionError):pre60_equal(q,r,'requests',pd.Series({1:0.}))

    def test_exact_fill_concession_and_refusal(self):
        intent=e.intent_meta('late_taper1',150000,0,1,(51,6),6,{1:dict(t=150000,price=51,base_price=52)})
        o={'late_meta':e.order_meta('late_taper1',150250,0,1,51,6,{1:intent})}
        row=e.fill_meta('late_taper1',1,0,151000,0,1,51,4,2,o)
        self.assertAlmostEqual(row['concession'],.04)
        self.assertEqual(row['reducing_q'],4)
        earlier=e.fill_meta('late_taper1',1,0,50000,0,0,50,6,6,
                            {'late_meta':dict(send_rel=49.,decision_rel=49.,decision_net=0.,known_net=0.,base_price=50,concession_ticks=0.)})
        row['index']=1
        f=pd.DataFrame([dict(mid=1,t=50.,side=0,pt=50,q=6),dict(mid=1,t=151.,side=1,pt=51,q=4)])
        z=pd.DataFrame([earlier,row]);reconcile_metadata(f,z)
        wrong=z.copy();wrong['net_before']+=10;wrong['net_after']+=10
        with self.assertRaises(ValueError):reconcile_metadata(f,wrong)
        z.loc[1,'t']+=.001
        with self.assertRaises(AssertionError):reconcile_metadata(f,z)
        with self.assertRaises(ValueError):e.order_meta('late_taper1',149999,0,1,51,6,{1:intent})
        with self.assertRaises(ValueError):e.intent_meta('late_taper1',150001,0,1,(51,6),6,{1:intent})
        with self.assertRaises(ValueError):e.fill_meta('late_taper1',1,0,151000,0,1,51,4,2,{})


if __name__=='__main__':
    unittest.main()
