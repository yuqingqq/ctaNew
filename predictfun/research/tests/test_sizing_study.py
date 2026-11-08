import unittest
from unittest.mock import patch
from predictfun.research.pf_sizing_engine import lots, headroom, size_state, source
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

    def test_pinned_builders_and_known_bad_refusal(self):
        compile(source(),'engine','exec');compile(features(),'features','exec')
        with patch('predictfun.research.pf_sizing_engine.sha',return_value='wrong'):
            with self.assertRaises(ValueError):source()
        with patch('predictfun.research.pf_sizing_features.sha',return_value='wrong'):
            with self.assertRaises(ValueError):features()

if __name__=='__main__':unittest.main()
