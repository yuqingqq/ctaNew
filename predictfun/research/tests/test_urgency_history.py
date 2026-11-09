import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from predictfun.research.pf_urgency_history import identity,pre180_equal,day_null,comparison


class HistoryControls(unittest.TestCase):
    def test_scope_and_policy_identity(self):
        identity('20261023','J')
        for d in ['20261022','20261027','20261109']:
            with self.assertRaises(ValueError):identity(d,'J')
        with patch('predictfun.research.pf_urgency_history.sha',return_value='wrong'):
            with self.assertRaises(ValueError):identity('20261023','J')

    def test_pre180_event_identity(self):
        b=pd.DataFrame(dict(mid=[1,1],t=[179.9,180.],q=[5.,5.]))
        a=b.copy();a.loc[1,'q']=10.
        pre180_equal(b,a,'fills',None)
        a.loc[0,'q']=10.
        with self.assertRaises(AssertionError):pre180_equal(b,a,'fills',None)
        q=pd.DataFrame(dict(mid=[1,1],t_ms=[1179999.,1180000.],size=[5.,5.]))
        a=q.copy();a.loc[1,'size']=10.
        pre180_equal(q,a,'requests',pd.Series({1:1000000.}))
        a.loc[0,'mid']=2
        with self.assertRaises(AssertionError):pre180_equal(q,a,'requests',pd.Series({1:1000000.,2:1000000.}))

    def test_day_null_positive_and_corruption(self):
        out=day_null(np.ones((10,2)))
        self.assertLess(out[0]['p_max_t_family'],.02)
        self.assertIsNone(day_null(np.ones((4,1)))[0]['interval'])
        for a in [np.array([1,2]),np.ones((1,2)),np.array([[1.],[float('nan')]])]:
            with self.assertRaises(ValueError):day_null(a)

    def test_comparison_matches_days(self):
        row=dict(day='20261023',markets=288,settle=1.,plus1=1.,mark_stop=1.,carry=0.,
                 reducing_shares=1.,reducing_mo1=1.,shares=1.,abs_net_end_mean=1.,requests={'place':1})
        self.assertEqual(comparison([row],[row])['delta']['settle'],0.)
        with self.assertRaises(ValueError):comparison([row],[dict(row,day='20261024')])
        with self.assertRaises(ValueError):comparison([row,row],[row,row])


if __name__=='__main__':unittest.main()
