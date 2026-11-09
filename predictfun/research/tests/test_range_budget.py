import unittest
from unittest.mock import patch
from predictfun.research import pf_range_engine as e


class RangeControls(unittest.TestCase):
    def test_cross_zero_within_range_and_refuse_excess(self):
        self.assertEqual(e.admission(10,1,25,[],'range10'),20)
        self.assertEqual(e.admission(-10,0,25,[],'range10'),20)
        e.assert_admission(10,1,20,[],'range10')
        with self.assertRaises(ValueError):e.assert_admission(10,1,20.01,[],'range10')
        e.assert_path(-10,'range10')
        with self.assertRaises(ValueError):e.assert_path(-10.01,'range10')

    def test_shared_lanes_and_pending_cancels(self):
        w=[(50,25),(51,25),(49,25),(52,25)]
        self.assertEqual(e.plan(5,w,[None]*4,[],'range10'),((50,5),(51,15),(None,0),(None,0)))
        o=dict(side=1,pt=51,size=15,kq=0,rem=0,cancelled=True)
        self.assertEqual(e.admission(5,1,25,[o,o],'range10'),0)
        self.assertEqual(e.admission(5,0,25,[o],'range10'),5)  # no opposite-side credit

    def test_remaining_quantity_retention_and_guard(self):
        o=dict(side=1,pt=51,size=20,kq=8)
        self.assertEqual(e.plan(2,[(None,0),(51,25)],[None,o],[o],'range10'),((None,0),(51,20)))
        self.assertEqual(e.plan(2,[(None,0),(52,25)],[None,o],[o],'range10'),((None,0),(None,0)))
        self.assertEqual(e.plan(2,[(None,0),(None,0)],[None,o],[o],'range10'),((None,0),(None,0)))

    def test_minimum_and_reporting_grace(self):
        self.assertEqual(e.plan(9,[(50,25)],[None],[],'range10'),((None,0),))
        self.assertEqual(e.release_time(22,0,100,'range10'),e.release_time(22,0,100,'both_reserved'))
        self.assertGreater(e.release_time(22,0,100,'range10'),122)
        with self.assertRaises(ValueError):e.admission(float('nan'),0,25,[],'range10')

    def test_control_behavior_and_source_identity(self):
        self.assertEqual(e.admission(10,1,25,[],'both_reserved'),10)
        self.assertEqual(e.admission(10,1,25,[],'reserved10'),25)
        compile(e.source(),'range','exec')
        with patch('predictfun.research.pf_range_engine.sha',return_value='bad'):
            with self.assertRaises(ValueError):e.source()


if __name__=='__main__':unittest.main()
