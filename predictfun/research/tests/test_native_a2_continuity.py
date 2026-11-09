import copy
import unittest
from predictfun.research.pf_native_a2_continuity import measure,verify_arm,instrument,ROOT,REL,command,quote

class NativeA2Controls(unittest.TestCase):
    def test_native_off_sentinel_and_invalid_quote_refusal(self):
        row={f'raw_{k}{l}':(-1 if k=='tick' else 0.) for l in range(4) for k in ['tick','size']}
        self.assertEqual(quote(row,'raw'),[[None,0.]]*4)
        row.update(raw_tick0=50,raw_size0=5.)
        self.assertEqual(quote(row,'raw')[0],[50,5.])
        row['raw_tick0']=-1
        with self.assertRaises(ValueError):quote(row,'raw')

    def test_source_anchor_positive_and_changed_refusal(self):
        source=command(['git','show',f'HEAD:{REL}'],cwd=ROOT)+'\n'
        self.assertIn('pub fn diagnostic_wants',instrument(source))
        with self.assertRaises(ValueError):instrument(source.replace('let wants = self.order_wants(&d.lanes);','let wants = changed();'))

    def sample(self):
        off=[[None,0.]]*4;on=[[50,5.]]+off[1:]
        w=dict(events=[dict(t=0.,kind=0,input_index=1,ordinal=0,raw=on,net=0.),dict(t=100.,kind=0,input_index=2,ordinal=1,raw=off,net=0.)])
        r=dict(arm='reference',orders=[dict(id=0,lane=0,tick=50,size=5.,sent=0.,accepted=40.,cancel_sent=100.,end=122.,release=1122.)],
            actions=[dict(index=1,input_index=2,ordinal=1,lane=0,start=100.,old_tick=50,status='reference_cancel',order_id=None,end=100.,reason='not_retained')],
            commands=[dict(kind='place',id=0,lane=0,tick=50,size=5.,t=0.,sent=True),dict(kind='cancel',id=0,lane=0,t=100.,sent=True)],
            counts=dict(places=1,cancels=1,throttled_locally=0),retry_times=0)
        return w,r

    def test_exact_intervals_and_corrupt_artifact_refusal(self):
        w,r=self.sample();verify_arm(w,r);m=measure(w,r)
        self.assertEqual(m['aligned_lane_ms'],60);self.assertEqual(m['elsewhere_lane_ms'],22)
        self.assertEqual(m['active_share_ms'],410);self.assertEqual(m['reserved_share_ms'],5610)
        bad=copy.deepcopy(r);bad['orders'][0]['release']=1121.
        with self.assertRaises(ValueError):verify_arm(w,bad)
        bad=copy.deepcopy(r);bad['actions'][0]['input_index']=3
        with self.assertRaises(ValueError):verify_arm(w,bad)
        bad=copy.deepcopy(r);bad['counts']['places']=2
        with self.assertRaises(ValueError):verify_arm(w,bad)

    def test_union_not_double_counted_and_opposite_sides_do_not_net(self):
        w,r=self.sample();r['orders'].append(dict(r['orders'][0],id=1,lane=4))
        m=measure(w,r);self.assertEqual(m['aligned_lane_ms'],60);self.assertEqual(m['active_share_ms'],820)
        r['orders'][1]['lane']=1;m=measure(w,r)
        self.assertEqual(m['active_range_extra_share_ms'],410)

if __name__=='__main__':unittest.main()
