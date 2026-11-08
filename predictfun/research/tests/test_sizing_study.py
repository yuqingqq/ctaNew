import unittest
from unittest.mock import patch
from predictfun.research.pf_sizing_engine import lots, headroom, size_state, source, cap_cannot_bind,capacity_cannot_clip
from predictfun.research.pf_sizing_features import source as features

class SizingControls(unittest.TestCase):
    def test_pending_cancels_reserve_capacity_and_not_net(self):
        orders=[dict(side=0,size=100,kq=20,cancelled=True),dict(side=0,size=50,kq=0)]
        self.assertEqual(headroom(5,0,orders),15)
        self.assertEqual(headroom(5,1,orders),155)
        self.assertEqual(min(25,headroom(5,0,orders)),15)

    def test_no_room_and_partial_fill(self):
        self.assertEqual(headroom(150,0,[]),0)
        self.assertEqual(headroom(-160,1,[]),0)
        self.assertEqual(headroom(-160,0,[]),310)

    def test_hysteresis_and_unsafe_value(self):
        self.assertFalse(size_state(False,.07,.0698))
        self.assertTrue(size_state(False,.08,.0698))
        self.assertTrue(size_state(True,.065,.0698))
        self.assertFalse(size_state(True,.05,.0698))
        self.assertFalse(size_state(True,None,.0698))

    def test_rounding_and_bad_quantity(self):
        self.assertEqual(lots(24.999999),24.99)
        self.assertEqual(lots(.009),0)
        self.assertEqual(lots(5.22),5.22)
        with self.assertRaises(ValueError):lots(float('nan'))
        with self.assertRaises(ValueError):headroom(0,2,[])

    def test_cap_reuse_refuses_binding_and_halted_paths(self):
        r=dict(min_worst=-180.,requests=[1])
        self.assertTrue(cap_cannot_bind(r,31.,150.,False))
        self.assertFalse(cap_cannot_bind(r,29.,150.,False))
        self.assertFalse(cap_cannot_bind(r,31.,150.,True))
        self.assertFalse(cap_cannot_bind(dict(min_worst=float('nan'),requests=[1]),0,150,False))
        self.assertTrue(capacity_cannot_clip(dict(live_ctr=dict(capacity_checks=3,would_clip_capacity=0),requests=[1])))
        self.assertFalse(capacity_cannot_clip(dict(live_ctr=dict(capacity_checks=3,would_clip_capacity=1),requests=[1])))
        self.assertFalse(capacity_cannot_clip(dict(live_ctr={},requests=[1])))

    def test_pinned_builders_and_known_bad_refusal(self):
        compile(source(),'engine','exec');compile(features(),'features','exec')
        with patch('predictfun.research.pf_sizing_engine.sha',return_value='wrong'):
            with self.assertRaises(ValueError):source()
        with patch('predictfun.research.pf_sizing_features.sha',return_value='wrong'):
            with self.assertRaises(ValueError):features()

    def test_reconciliation_fires_on_wrong_market_even_when_totals_equal(self):
        import pandas as pd
        from predictfun.research.pf_sizing_run import reconcile
        w=pd.DataFrame(dict(mid=[1,2],pnl=[1.,-1.]))
        f=pd.DataFrame(dict(mid=[1,2],settle=[1.,-1.]))
        reconcile(w,f)
        f['mid']=[2,1]
        with self.assertRaises(ValueError):reconcile(w,f)

    def test_probability_inputs_ignore_future_columns(self):
        import pandas as pd
        import numpy as np
        from predictfun.research.pf_settlement_study import matrix,causal_train
        r=pd.DataFrame(dict(day=['20261023','20261024'],outcome_seen_ns=[0,9999999999999999999],
          pf_mid=[50.,60.],t_rel=[60.,120.],y=[0.,1.],end_price=[2.,3.]))
        a=matrix(r,False);r['end_price']=1e15;r['y']=1-r.y
        np.testing.assert_array_equal(a,matrix(r,False))
        self.assertEqual(len(causal_train(r,'20261025')),1)

    def test_shadow_matched_control_can_detect_and_refuse(self):
        import pandas as pd
        from predictfun.research.pf_settlement_inventory import matched_random
        g=pd.DataFrame(dict(day=['20261101']*4,hour=[1]*4,anchor=[60]*4,side=[0]*4,
          market_bin=[2]*4,status=['eligible']*4,executable=[False,False,False,True],
          potential_delta=[-2.,-1.,1.,2.],delta_vs_hold=[0.,0.,0.,2.]))
        self.assertGreater(matched_random(g)['excess_vs_random'],.4)
        g.loc[3,'delta_vs_hold']=3.
        with self.assertRaises(ValueError):matched_random(g)

    def test_day_null_detects_signal_and_refuses_bad_populations(self):
        import numpy as np
        from predictfun.research.pf_settlement_study import paired_inference
        a=np.column_stack([np.ones(8),np.tile([-1.,1.],4)])
        result=paired_inference(a)
        self.assertLess(result[0]['p_max_t_family'],.05)
        self.assertEqual(result[1]['p_two_sided'],1.)
        with self.assertRaises(ValueError):paired_inference(a[:4])
        a[0,0]=np.nan
        with self.assertRaises(ValueError):paired_inference(a)

if __name__=='__main__':unittest.main()
