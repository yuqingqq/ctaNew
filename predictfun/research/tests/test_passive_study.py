import unittest
from unittest.mock import patch
from predictfun.research.pf_passive_engine import keep_size,urgent_price,adjust,source


class PassiveControls(unittest.TestCase):
    def test_partial_fill_retains_original_order_size(self):
        o=dict(side=1,pt=51,size=25.,kq=20.)
        self.assertEqual(keep_size(5,1,51,o,[o]),25.)
        self.assertEqual(adjust(5,1,51,5,100,49,51,o,[o],{'RKEEP'}),(51,25.))
        self.assertIsNone(keep_size(4,1,51,o,[o]))
        self.assertIsNone(keep_size(-5,1,51,o,[o]))

    def test_pending_cancels_count_and_same_identity_deduplicates(self):
        o=dict(side=0,pt=49,size=5.);pending=dict(side=0,pt=48,size=4.,cancelled=True)
        self.assertEqual(keep_size(-9,0,49,o,[o,o,pending]),5)
        self.assertIsNone(keep_size(-8,0,49,o,[o,pending]))
        with self.assertRaises(ValueError):keep_size(float('nan'),0,49,o,[])

    def test_phase_guard_and_price_change_do_not_retain(self):
        o=dict(side=1,pt=55,size=25.)
        self.assertEqual(adjust(30,1,None,0,190,45,55,o,[o],{'RKEEP'}),(None,0))
        self.assertEqual(adjust(30,1,55,5,59,45,55,o,[o],{'RKEEP'}),(55,5))
        self.assertIsNone(keep_size(30,1,54,o,[o]))
        self.assertEqual(adjust(30,1,55,25,100,45,55,o,[o],{'RKEEP','RSMALL'}),(55,5))

    def test_price_symmetry_passivity_inventory_and_time(self):
        self.assertEqual(urgent_price(10,1,55,45,55,180),54)
        self.assertEqual(urgent_price(-10,0,45,45,55,180),46)
        self.assertEqual(urgent_price(10,1,51,50,51,180),51)
        self.assertEqual(urgent_price(9,1,55,45,55,180),55)
        self.assertEqual(urgent_price(10,1,55,45,55,179.9),55)
        self.assertEqual(urgent_price(10,1,55,45,55,240),55)
        self.assertEqual(urgent_price(10,0,45,45,55,190),45)
        with self.assertRaises(ValueError):urgent_price(10,3,55,45,55,190)

    def test_pinned_source_and_wrong_version_refusal(self):
        compile(source(),'passive','exec')
        with patch('predictfun.research.pf_passive_engine.sha',return_value='wrong'):
            with self.assertRaises(ValueError):source()

    def test_entry_checker_detects_wrong_market_even_if_totals_equal(self):
        import pandas as pd
        from predictfun.research.pf_passive_run import require_prephase_equal
        b=pd.DataFrame(dict(mid=[1,2],t=[10.,20.],q=[1.,1.]))
        require_prephase_equal(b,b.copy(),'fills',None)
        c=b.copy();c['mid']=[2,1]
        with self.assertRaises(AssertionError):require_prephase_equal(b,c,'fills',None)


if __name__=='__main__':unittest.main()
