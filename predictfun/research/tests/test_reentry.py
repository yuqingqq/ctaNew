import unittest
from predictfun.research.pf_reentry import ReentryGate
from predictfun.research.pf_quote_capture import QuotePlanner

OFF=[(None,0.)]*4
def want(tick=50,size=5):return [(tick,size)]+OFF[1:]


class ReentryControls(unittest.TestCase):
    def test_reference_and_zero_delay_are_exact(self):
        for mode in ['pass','reentry','all_entry']:
            g=ReentryGate(mode,0)
            for t,w in [(0,want()),(10,OFF),(20,want()),(21,want(size=25)),(30,want(51)),(40,OFF)]:
                self.assertEqual(g.apply(t,w),w)
    def test_confirm_only_rapid_same_price_reentry(self):
        g=ReentryGate('reentry',50)
        self.assertEqual(g.apply(0,want()),want())
        self.assertEqual(g.apply(100,OFF),OFF)
        self.assertEqual(g.apply(120,want()),OFF)
        self.assertEqual(g.apply(169,want()),OFF)
        self.assertEqual(g.apply(170,want()),want())
        self.assertEqual(g.apply(171,want(size=25)),want(size=25))
        self.assertEqual(g.apply(180,OFF),OFF)
        self.assertEqual(g.apply(281,want()),want())
    def test_new_price_does_not_wait_and_old_price_is_not_kept(self):
        g=ReentryGate('reentry',50);g.apply(0,want());g.apply(10,OFF)
        self.assertEqual(g.apply(20,want()),OFF)
        self.assertEqual(g.apply(30,want(51)),want(51))
        self.assertFalse(g.actions[-1]['rapid'])
    def test_short_reentry_is_withheld_and_pull_is_immediate(self):
        g=ReentryGate('reentry',50);g.apply(0,want());g.apply(10,OFF)
        g.apply(20,want());self.assertEqual(g.apply(40,OFF),OFF)
        a=g.finish(100);self.assertEqual(len(a),2)
        self.assertIsNone(a[1]['admitted_at']);self.assertEqual(a[1]['end_reason'],'pull')
    def test_cancel_ack_gate_releases_at_confirmation_not_grace(self):
        g=ReentryGate('cancel_ack');p=QuotePlanner(incremental=True)
        for t,w in [(0,want()),(100,OFF),(110,want())]:p.plan(t,g.apply(t,w,p.orders))
        self.assertEqual(p.stats['places'],1)
        self.assertEqual(g.apply(121,want(),p.orders),OFF)
        self.assertEqual(g.apply(122,want(),p.orders),want())
        p.plan(122,want());self.assertEqual(p.stats['places'],2)
        self.assertEqual(p.reserved(0),10)
    def test_cancel_pending_create_waits_and_ignores_other_lanes(self):
        g=ReentryGate('cancel_ack');p=QuotePlanner(incremental=True)
        for t,w in [(0,want()),(10,OFF),(20,want())]:p.plan(t,g.apply(t,w,p.orders))
        self.assertEqual(g.apply(94,want(),p.orders),OFF)
        self.assertEqual(g.apply(95,want(),p.orders),want())
        h=ReentryGate('cancel_ack');h.apply(0,[(None,0.),(50,5),*OFF[2:]]);h.apply(10,OFF)
        w=[(None,0.),(50,5),*OFF[2:]]
        self.assertEqual(h.apply(20,w,p.orders),w)
    def test_all_entry_cost_control_and_refusals(self):
        g=ReentryGate('all_entry',50)
        self.assertEqual(g.apply(0,want()),OFF);self.assertEqual(g.apply(50,want()),want())
        self.assertEqual(g.apply(60,want(51)),OFF)
        with self.assertRaises(ValueError):g.apply(59,want())
        with self.assertRaises(ValueError):ReentryGate('bad')
        with self.assertRaises(ValueError):ReentryGate(confirm_ms=float('nan'))
        with self.assertRaises(ValueError):ReentryGate().apply(0,[(None,5)]+OFF[1:])
    def test_causal_prefix_and_action_identity(self):
        prefix=[(0,want()),(10,OFF),(20,want()),(60,want())]
        a=ReentryGate('reentry',50);b=ReentryGate('reentry',50)
        self.assertEqual([a.apply(t,w) for t,w in prefix],[b.apply(t,w) for t,w in prefix])
        a.apply(65,OFF);b.apply(100,want())
        aa=a.finish(200);bb=b.finish(200)
        self.assertEqual([(r['action'],r['lane'],r['tick'],r['start']) for r in aa],[(r['action'],r['lane'],r['tick'],r['start']) for r in bb])
        self.assertIsNone(aa[-1]['admitted_at']);self.assertEqual(bb[-1]['admitted_at'],100)


if __name__=='__main__':unittest.main()
