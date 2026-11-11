import unittest
import pandas as pd
from predictfun.research.pf_order_policy_inputs import FEATURES
from predictfun.research.pf_own_order_fit_v2 import unique_actions

class Controls(unittest.TestCase):
    def rows(self):
        x={k:0. for k in FEATURES};x.update(day='20261023',mid=1,t_send=10.,t_act=50.,decision=100.,initial_size=5.,label=1.)
        return pd.DataFrame([dict(x,mode='reference'),dict(x,mode='calm_seed')])
    def test_duplicate_action_is_counted_once(self):
        a=unique_actions(self.rows());self.assertEqual(len(a),1);self.assertEqual(a.iloc[0]['mode'],'reference')
    def test_different_known_state_is_different_action(self):
        a=self.rows();a.loc[1,'inventory']=1.;self.assertEqual(len(unique_actions(a)),2)
    def test_conflicting_label_refuses(self):
        a=self.rows();a.loc[1,'label']=2.
        with self.assertRaises(ValueError):unique_actions(a)

if __name__=='__main__':unittest.main()
