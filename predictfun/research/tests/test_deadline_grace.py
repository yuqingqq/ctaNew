import unittest
from predictfun.research.pf_deadline_grace import UpdateFirstGrace
from predictfun.research.pf_inventory_grace import InventoryGrace
from predictfun.research.pf_quote_capture import QuotePlanner
from predictfun.research.tests.test_inventory_grace import r


class DeadlineGraceControls(unittest.TestCase):
    def make(self,gate=None):
        g=gate or UpdateFirstGrace();p=QuotePlanner(incremental=True)
        g.apply(0,r(),p);g.apply(100,r(False,v=.045),p)
        return g,p

    def test_equal_clock_reopen_keeps_order(self):
        g,p=self.make();g.expire(150,p);g.apply(150,r(),p);g.finish_timestamp(150,p)
        self.assertEqual(p.stats['places'],1);self.assertEqual(p.stats['cancels'],0)
        self.assertEqual(g.actions[0]['retained_ms'],50);self.assertEqual(g.actions[0]['end_reason'],'same_price_reopen')
        old,q=self.make(InventoryGrace(.01,'all',50.));old.expire(150,q);old.apply(150,r(),q)
        self.assertEqual(q.stats['cancels'],1);self.assertEqual(q.stats['places'],2)

    def test_still_off_deadline_is_not_extended(self):
        g,p=self.make();g.expire(150,p);g.apply(150,r(False,v=.045),p);g.finish_timestamp(150,p)
        self.assertEqual(p.commands[-1]['t'],150);self.assertEqual(p.orders[0].cancel_at,172)
        self.assertEqual(g.actions[0]['retained_ms'],50)

    def test_no_event_at_deadline_and_later_reopen(self):
        g,p=self.make();g.expire(151,p);g.apply(151,r(),p)
        self.assertEqual(p.commands[-2]['kind'],'cancel');self.assertEqual(p.commands[-2]['t'],150)
        self.assertEqual(g.actions[0]['end_reason'],'deadline')

    def test_hard_guard_at_deadline_still_cancels(self):
        g,p=self.make();g.expire(150,p);g.observe(150,r(False,v=.045,held=True),p);g.finish_timestamp(150,p)
        self.assertEqual(p.orders[0].cancel_at,172);self.assertEqual(g.actions[0]['end_reason'],'held')

    def test_all_same_clock_events_are_processed(self):
        g,p=self.make();g.expire(150,p);g.observe(150,r(False,v=.045),p)
        g.apply(150,r(),p);g.finish_timestamp(150,p)
        self.assertEqual(p.stats['cancels'],0)


if __name__=='__main__':unittest.main()
