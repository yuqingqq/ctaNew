import unittest
from unittest.mock import patch
import pandas as pd
from predictfun.research.pf_entry_engine import plan,room,admission,assert_admission,source
from predictfun.research.pf_entry_report import full_path


class EntryControls(unittest.TestCase):
    def test_shared_side_budget_and_uniform_control(self):
        w=[(50,25),(50,25),(49,25),(51,25)]
        self.assertEqual(plan(0,w,[None]*4,[],'reserved10'),((50,10),(50,10),(None,0),(None,0)))
        self.assertEqual(plan(0,w,[None]*4,[],'uniform10'),((50,10),(50,10),(49,10),(51,10)))

    def test_partial_fills_keep_original_identity(self):
        o=dict(side=0,pt=50,size=10.,kq=8.)
        self.assertEqual(plan(8,[(50,25)],[o],[o],'reserved10'),((50,10),))
        self.assertEqual(room(8,0,[o,o]),0.)
        self.assertEqual(plan(8,[(49,25)],[o],[o],'reserved10'),((None,0.),))

    def test_pending_reservations_and_minimum(self):
        o=dict(side=0,pt=49,size=8.,cancelled=True)
        self.assertEqual(plan(0,[(50,25)],[None],[o],'reserved10'),((50,2.),))
        self.assertEqual(plan(0,[(49,25)],[None],[o],'reserved10'),((None,0.),))
        self.assertEqual(admission(0,0,10,[o],'reserved10'),2.)
        assert_admission(0,0,2,[o],'reserved10')
        with self.assertRaises(ValueError):assert_admission(0,0,3,[o],'reserved10')

    def test_reduction_and_guard_withdrawal_unchanged(self):
        w=[(50,25),(51,25)];o=dict(side=0,pt=50,size=5.)
        self.assertEqual(plan(-8,w,[None,None],[],'reserved10')[0],(50,25))
        self.assertEqual(plan(8,w,[None,None],[],'reserved10')[1],(51,25))
        self.assertEqual(plan(0,[(None,0)],[o],[o],'reserved10'),((None,0),))
        self.assertEqual(admission(-5,0,25,[],'reserved10'),25.)

    def test_symmetry_and_no_actual_remaining_leak(self):
        o=dict(side=1,pt=50,size=8.,kq=0.,rem=0.)
        self.assertEqual(room(-1,1,[o]),1.)
        with self.assertRaises(ValueError):room(float('nan'),0,[])
        with self.assertRaises(ValueError):plan(0,[(50,5)],[None],[],'unknown')

    def test_pinned_source_and_refusal(self):
        compile(source(),'entry','exec')
        with patch('predictfun.research.pf_entry_engine.sha',return_value='bad'):
            with self.assertRaises(ValueError):source()

    def test_full_inventory_and_fifo_accounting(self):
        f=pd.DataFrame(dict(t=[10.,100.,200.],q=[10.,6.,7.],side=[0,1,1],pt=[50.,60.,40.]))
        r=full_path(f)
        self.assertAlmostEqual(r['inventory_square'],11500/300)
        self.assertAlmostEqual(r['mean_abs_inventory'],1600/300)
        self.assertAlmostEqual(r['paired_pnl'],.2)
        self.assertEqual(r['paired_shares'],10.)
        self.assertEqual(r['net_end'],-3.)
        self.assertEqual(full_path(f.iloc[:0])['inventory_square'],0.)
        with self.assertRaises(ValueError):full_path(f.iloc[::-1])
        f.loc[0,'side']=2
        with self.assertRaises(ValueError):full_path(f)


if __name__=='__main__':unittest.main()
