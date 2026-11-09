import unittest
from unittest.mock import patch
from predictfun.research import pf_inflight_engine as e


class InflightControls(unittest.TestCase):
    def test_aggregate_close_no_reverse(self):
        w=[(49,25),(51,25),(48,25),(52,25)]
        self.assertEqual(e.plan(10,w,[None]*4,[],'reduce_reserved'),((49,25),(51,10),(48,25),(None,0)))
        self.assertEqual(e.admission(10,1,25,[],'reduce_reserved'),10)
        with self.assertRaises(ValueError):e.assert_admission(10,1,11,[],'reduce_reserved')

    def test_pending_all_lanes_no_physical_leak(self):
        o=dict(side=1,pt=52,size=10,kq=0,rem=0,cancelled=True)
        self.assertEqual(e.room(10,1,[o,o],'reduce_reserved'),0)
        self.assertEqual(e.plan(10,[(None,0),(51,25)],[None]*2,[o],'reduce_reserved'),((None,0),(None,0)))
        o['kq']=4
        self.assertEqual(e.room(6,1,[o],'reduce_reserved'),0)

    def test_partial_retention_and_guard(self):
        o=dict(side=1,pt=51,size=10,kq=8)
        self.assertEqual(e.plan(2,[(None,0),(51,25)],[None,o],[o],'reduce_reserved'),((None,0),(51,10)))
        self.assertEqual(e.plan(2,[(None,0),(None,0)],[None,o],[o],'reduce_reserved'),((None,0),(None,0)))

    def test_combined_bound_and_symmetry(self):
        w=[(49,25),(51,25),(48,25),(52,25)]
        self.assertEqual(e.plan(0,w,[None]*4,[],'both_reserved'),((49,10),(51,10),(None,0),(None,0)))
        self.assertEqual(e.admission(-7,0,25,[],'both_reserved'),7)
        self.assertEqual(e.admission(7,1,25,[],'both_reserved'),7)
        self.assertEqual(e.admission(7,0,25,[],'both_reserved'),3)
        self.assertEqual(e.admission(-7,1,25,[],'both_reserved'),3)
        e.assert_path(10,'both_reserved')
        with self.assertRaises(ValueError):e.assert_path(10.01,'both_reserved')

    def test_cancel_fill_report_grace(self):
        release=e.release_time(22,0,100,'both_reserved')
        self.assertGreater(release,122)
        self.assertLess(21+100,release)  # fill just before cancel must be reported before release
        self.assertEqual(e.release_time(22,0,100,'reserved10'),22)
        with self.assertRaises(ValueError):e.release_time(22,0,-1,'both_reserved')

    def test_source_identity_and_old_dispatch(self):
        compile(e.source(),'inflight','exec')
        self.assertEqual(e.admission(10,1,25,[],'reserved10'),25)
        with patch('predictfun.research.pf_inflight_engine.sha',return_value='bad'):
            with self.assertRaises(ValueError):e.source()
        with self.assertRaises(ValueError):e.room(0,0,[],'bad')


if __name__=='__main__':unittest.main()
