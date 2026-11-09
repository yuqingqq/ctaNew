import unittest
from predictfun.research.pf_inventory_grace import InventoryGrace,inventory_reason,reserved_side,exposure
from predictfun.research.pf_quote_capture import QuotePlanner
from predictfun.research.pf_soft_pull import SoftPull,lanes,OFF
from predictfun.research.tests.test_soft_pull import row


def r(on=True,net=-5.,**kw):
    d=row(on=on,**kw);d['net']=net
    return d


class InventoryGraceControls(unittest.TestCase):
    def make(self,mode='reserved',net=-5.):
        p=QuotePlanner(incremental=True);g=InventoryGrace(mode=mode);g.apply(0,r(net=net),p)
        return p,g

    def test_reducing_positive_and_flat_wrong_side_refusal(self):
        p,g=self.make();g.apply(100,r(False,v=.045),p);self.assertEqual(len(g.holds),1)
        for net in [0.,5.]:
            p,g=self.make(net=net);g.apply(100,r(False,net=net,v=.045),p)
            self.assertEqual(g.actions[0]['status'],'inventory_not_reducing');self.assertFalse(g.holds)

    def test_individual_overshoot_and_reserved_double_use(self):
        self.assertIsNone(inventory_reason('individual',-5.,0,5.,25.))
        self.assertEqual(inventory_reason('reserved',-5.,0,5.,25.),'inventory_reserved_overshoot')
        self.assertEqual(inventory_reason('individual',-4.99,0,5.,5.),'inventory_individual_overshoot')
        self.assertIsNone(inventory_reason('reserved',5.,1,5.,5.))
        for args in [('bad',5.,1,5.,5.),('reserved',float('nan'),0,5.,5.),('reserved',-5.,0,5.,0.)]:
            with self.assertRaises(ValueError):inventory_reason(*args)

    def test_pending_cancel_grace_and_pending_create_count(self):
        p=QuotePlanner(incremental=True);p.plan(0,[(50,5)]+OFF[1:]);p.cancel(p.orders[0],100,'test')
        p.plan(130,[(50,5)]+OFF[1:])
        self.assertEqual(reserved_side(p,0,140),10) # second create not accepted yet
        self.assertEqual(reserved_side(p,0,1121),10)
        self.assertEqual(reserved_side(p,0,1122),5) # release_at boundary, without cleanup
        self.assertEqual(reserved_side(p,1,140),0)

    def test_inventory_change_cancels_existing_hold(self):
        p,g=self.make();g.apply(100,r(False,v=.045),p)
        g.observe(125,r(False,net=0.,v=.045),p)
        self.assertEqual(p.orders[0].cancel_at,147);self.assertFalse(g.holds)
        self.assertEqual(g.actions[0]['end_reason'],'inventory_not_reducing')

    def test_other_lane_new_order_consumes_room_after_plan(self):
        p,g=self.make();d=r(False,v=.045);d['raw_tick2']=49;d['raw_size2']=5.
        g.apply(100,d,p)
        self.assertFalse(g.holds);self.assertEqual(g.actions[0]['end_reason'],'inventory_reserved_overshoot')
        self.assertEqual(g.actions[0]['retained_ms'],0)
        self.assertEqual(p.orders[0].cancel_at,122)

    def test_unrestricted_policy_matches_prior_commands(self):
        p,g=self.make(mode='all');q=QuotePlanner(incremental=True);h=SoftPull(.01);h.apply(0,r(),q)
        for t,d in [(100,r(False,v=.045)),(130,r(False,v=.045)),(150,r()),(200,r(False,v=.045)),(350,r(False,v=.045))]:
            for gate,planner in [(g,p),(h,q)]:gate.expire(t,planner);gate.apply(t,d,planner)
        self.assertEqual(p.commands,q.commands);self.assertEqual(p.stats,q.stats)

    def test_exact_direction_and_range_exposure(self):
        p=QuotePlanner(incremental=True);p.plan(0,[(50,5)]+OFF[1:])
        q=exposure(p,0,100,OFF,-5.)
        self.assertEqual(q['off_reducing_share_ms'],300)
        self.assertEqual(q['off_other_share_ms'],0)
        self.assertEqual(q['active_range_extra_share_ms'],0)
        q=exposure(p,0,100,OFF,5.)
        self.assertEqual(q['off_other_share_ms'],300)
        self.assertEqual(q['active_range_extra_share_ms'],300)
        self.assertEqual(q['reserved_range_extra_share_ms'],500)
        self.assertEqual(exposure(p,0,100,lanes(r()),5.)['off_other_share_ms'],0)

    def test_opposite_orders_do_not_net_and_cancellation_lifecycle(self):
        p=QuotePlanner(incremental=True);p.plan(0,[(50,5),(52,5)]+OFF[2:]);p.cancel(p.orders[0],100,'test')
        q=exposure(p,0,200,OFF,0.)
        self.assertEqual(q['active_range_extra_share_ms'],5*160)
        self.assertEqual(q['reserved_range_extra_share_ms'],5*200)
        self.assertEqual(q['off_other_share_ms'],5*(82+160))
        with self.assertRaises(ValueError):exposure(p,100,0,OFF,0.)


if __name__=='__main__':unittest.main()
