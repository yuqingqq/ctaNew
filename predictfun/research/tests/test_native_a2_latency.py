import copy
import tempfile
import unittest
from pathlib import Path
from predictfun.research import pf_native_a2_latency as m
from predictfun.research.tests import test_native_a2_continuity as prior

class LatencyControls(unittest.TestCase):
    def test_generated_anchors_and_drift_refusal(self):
        source=(m.s.a.HERE/'native_a2_continuity.rs').read_text()
        out=m.generate(source);self.assertIn('latency_profiles_change_both',out)
        self.assertNotIn('accepted:t+40.',out)
        with self.assertRaises(ValueError):m.generate(source.replace('accepted:t+40.','accepted:t+41.'))

    def test_profile_lifecycles_and_corruption_refusal(self):
        for profile,(place,cancel,pending) in m.LATENCIES.items():
            w,r=prior.NativeA2Controls().sample();o=r['orders'][0]
            o.update(accepted=place,end=100+cancel,release=1100+cancel)
            m.verify_arm(w,r,profile)
            bad=copy.deepcopy(r);bad['orders'][0]['end']+=1
            with self.assertRaises(ValueError):m.verify_arm(w,bad,profile)
            bad=copy.deepcopy(r);bad['actions'][0]['input_index']+=1
            with self.assertRaises(ValueError):m.verify_arm(w,bad,profile)

    def test_pending_cancel_and_duplicate_refusal(self):
        w,r=prior.NativeA2Controls().sample();r['commands'][1]['t']=5.
        w['events'][1]['t']=5.;r['actions'][0].update(start=5.,end=5.)
        for profile,(place,cancel,pending) in m.LATENCIES.items():
            o=r['orders'][0];end=max(5+cancel,pending);o.update(accepted=place,cancel_sent=5.,end=end,release=end+1000)
            m.verify_arm(w,r,profile)
            bad=copy.deepcopy(r);bad['commands'].append(bad['commands'][0])
            with self.assertRaises(ValueError):m.verify_arm(w,bad,profile)

    def test_exact_base_positive_and_changed_artifact_refusal(self):
        with tempfile.TemporaryDirectory() as t:
            left=Path(t)/'left';right=Path(t)/'right';left.write_bytes(b'correct');right.write_bytes(b'correct')
            m.exact_base(left,right);right.write_bytes(b'changed')
            with self.assertRaises(ValueError):m.exact_base(left,right)

if __name__=='__main__':unittest.main()
