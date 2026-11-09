import math
import unittest
from predictfun.research.pf_quote_capture import QuotePlanner,EntryGate,components

OFF=[(None,0.)]*4
def target(size,tick=50,lane=0):
    v=OFF.copy();v[lane]=(tick,size);return v


class QuoteControls(unittest.TestCase):
    def test_single_order_reference_resizes_both_ways(self):
        p=QuotePlanner(layered=False,grace_ms=0)
        p.plan(0,target(5));p.plan(100,target(25));p.plan(200,target(5))
        self.assertEqual([c['size'] for c in p.commands if c['kind']=='place'],[5,25,5])
        self.assertEqual(p.stats['cancels'],2)
        p.fill(p.orders[-1].id,3);p.plan(300,target(5))
        self.assertEqual(p.stats['places'],3)

    def test_size_switch_keeps_base_identity(self):
        p=QuotePlanner(grace_ms=0)
        p.plan(0,target(5));base=p.orders[0].id
        p.plan(100,target(25));p.plan(200,target(5))
        b=next(o for o in p.orders if o.id==base)
        self.assertIsNone(b.cancel_at)
        self.assertEqual([c['size'] for c in p.commands if c['kind']=='place'],[5,20])
        self.assertEqual([c['slot'] for c in p.commands if c['kind']=='cancel'],['extra'])
        self.assertEqual(p.stats['kept_base_size_switches'],2)

    def test_fixed_size_parity(self):
        a=QuotePlanner(layered=False);b=QuotePlanner(layered=True,base=25)
        for t,w in [(0,target(25)),(100,target(25)),(200,target(25,51)),(250,OFF),(400,target(25))]:
            a.plan(t,w);b.plan(t,w)
        self.assertEqual(a.commands,b.commands)

    def test_partial_base_and_topup_not_refilled(self):
        p=QuotePlanner(grace_ms=0);p.plan(0,target(5));base=p.orders[0].id
        p.fill(base,3);p.plan(100,target(5));self.assertEqual(p.stats['places'],1)
        p.plan(200,target(25));extra=next(o for o in p.orders if o.slot=='extra')
        self.assertEqual(extra.size,20);self.assertEqual(next(o for o in p.orders if o.id==base).rem,2)
        p.fill(extra.id,7);p.plan(300,target(25));self.assertEqual(p.stats['places'],2)
        with self.assertRaises(ValueError):p.fill(extra.id,20)

    def test_minimum_order_fallback(self):
        self.assertEqual(components(0,10,25),{'base':25})
        self.assertEqual(components(1,90,25),{'base':25})
        self.assertEqual(components(0,10,5),{})
        self.assertEqual(components(0,50,6),{'base':6})
        self.assertEqual(components(0,50,25),{'base':5,'extra':20})
        with self.assertRaises(ValueError):components(0,0,25)

    def test_pull_reprice_and_stop_immediate(self):
        p=QuotePlanner();p.plan(0,target(25));p.plan(100,target(25,51))
        self.assertTrue(all(o.cancel_at==122 for o in p.orders if o.tick==50))
        p.plan(110,OFF)
        self.assertTrue(all(o.cancel_at is not None for o in p.orders))
        p.plan(500,target(25),hard_stop=True)
        self.assertFalse(any(c['kind']=='place' and c['t']==500 for c in p.commands))

    def test_pending_create_and_cancel_remain_reserved(self):
        p=QuotePlanner(cap=25,grace_ms=1000)
        p.plan(0,target(25));p.plan(10,OFF);p.plan(100,target(25))
        self.assertEqual(p.reserved(0),25);self.assertEqual(p.stats['places'],2)
        self.assertGreater(p.refused['capacity'],0)
        p.plan(1095,target(25));self.assertEqual(p.stats['places'],4)
        self.assertEqual(p.reserved(0),25)

    def test_inflight_limit_counts_both_lanes_and_topups(self):
        p=QuotePlanner();w=target(25);w[2]=(49,25)
        p.plan(0,w)
        self.assertEqual(p.stats['places'],3);self.assertGreater(p.refused['inflight'],0)
        p.plan(50,w);self.assertEqual(p.stats['places'],4)

    def test_reduce_only_shared_room_and_partial_base(self):
        p=QuotePlanner(grace_ms=0);p.plan(0,target(5,lane=1));p.fill(p.orders[0].id,3)
        w=target(3,lane=1);w[3]=(51,3)
        p.plan(100,w,known_net=3,reduce_only=True)
        self.assertEqual(p.reserved(1),2);self.assertEqual(p.stats['places'],1)
        p.plan(200,target(25),known_net=3,reduce_only=True)
        self.assertFalse(any(o.lane%2==0 and o.cancel_at is None for o in p.orders))

    def test_gate_pass_through_and_strict_exit(self):
        z=EntryGate(0);self.assertEqual(z.apply(target(25),[math.nan,math.nan],.05,100,math.nan),target(25))
        p=EntryGate(.01)
        self.assertEqual(p.apply(target(25),[.055,.1],.05,0,0),OFF)
        self.assertEqual(p.apply(target(25),[.07,.1],.05,10,10),target(25))
        self.assertEqual(p.apply(target(25),[.055,.1],.05,20,20),target(25))
        self.assertEqual(p.apply(target(25),[.05,.1],.05,30,30),OFF)
        self.assertEqual(p.apply(target(25),[.07,.1],.05,200,90),OFF)
        self.assertEqual(p.apply(target(25),[.07,.1],.05,201,202),OFF)

    def test_gate_resets_on_guard_pull_and_reprice(self):
        p=EntryGate(.01);p.apply(target(25),[.07,.1],.05,0,0)
        self.assertEqual(p.apply(OFF,[1.,1.],.05,1,1),OFF)
        self.assertEqual(p.apply(target(25),[.055,.1],.05,2,2),OFF)
        p.apply(target(25),[.07,.1],.05,3,3)
        self.assertEqual(p.apply(target(25,51),[.055,.1],.05,4,4),OFF)

    def test_exact_order_time_integrals(self):
        p=QuotePlanner(place_ms=40,cancel_ms=20,grace_ms=100)
        p.plan(0,target(5));p.plan(100,OFF)
        x=p.integrate(0,200)
        self.assertEqual(x['quoted_lane_ms'],80)
        self.assertEqual(x['share_ms'],400)
        self.assertEqual(x['oldest_lane_age_ms2'],3200)
        self.assertEqual(x['reserved_share_ms'],1000)
        with self.assertRaises(ValueError):p.plan(99,OFF)


if __name__=='__main__':unittest.main()
