import unittest
import numpy as np
from predictfun.research.pf_quote_capture import QuotePlanner
from predictfun.research.pf_soft_pull import SoftPull,lanes,OFF
from predictfun.research.pf_soft_pull_study import price_label


def row(on=True,v=.06,binance=False,held=False,tick=50,hard=False,rung=False):
    d=dict(v_bid=v,v_ask=v,binance_bid=binance,binance_ask=binance,held=held,midpoint=51.,bid=50,ask=52,input_index=1)
    for prefix in ['raw','d0.005','d0.01','d0.02','d0.04']:
        for lane in range(4):d[f'{prefix}_tick{lane}']=-1;d[f'{prefix}_size{lane}']=0.
        if (on or prefix!='raw') and not hard:d[f'{prefix}_tick0']=tick;d[f'{prefix}_size0']=5.
    if rung:d['raw_tick2']=tick;d['raw_size2']=5.
    return d


class SoftPullControls(unittest.TestCase):
    def setup_policy(self):
        p=QuotePlanner(incremental=True);g=SoftPull(.01);g.apply(0,row(),p)
        return p,g

    def test_positive_retention_and_same_price_reopen(self):
        p,g=self.setup_policy();g.apply(100,row(False,.045),p)
        self.assertEqual(len(g.holds),1);self.assertIsNone(p.orders[0].cancel_at)
        g.apply(150,row(),p)
        self.assertEqual(p.stats['places'],1);self.assertEqual(p.stats['cancels'],0)
        self.assertEqual(g.actions[0]['end_reason'],'same_price_reopen')
        self.assertEqual(g.actions[0]['retained_ms'],50)

    def test_hard_guard_and_lower_line_probe_cancel_without_placement(self):
        for changes,reason in [(dict(binance=True),'binance'),(dict(held=True),'held'),(dict(v=.0392),'lower_line'),
            (dict(v=float('nan')),'missing_score'),(dict(tick=49),'native_price_size_guard'),(dict(hard=True),'native_price_size_guard')]:
            p,g=self.setup_policy();g.apply(100,row(False,.045),p)
            kw=dict(on=False,v=.045);kw.update(changes);g.observe(125,row(**kw),p)
            self.assertEqual(g.actions[0]['end_reason'],reason);self.assertEqual(p.orders[0].cancel_at,147)
            self.assertEqual(p.stats['places'],1)

    def test_exact_deadline_inside_gap(self):
        p,g=self.setup_policy();g.apply(100,row(False,.045),p);g.expire(500,p)
        self.assertEqual(p.commands[-1]['t'],200);self.assertEqual(p.orders[0].cancel_at,222)
        self.assertEqual(p.integrate(100,500)['share_ms'],5*122)
        self.assertEqual(g.actions[0]['retained_ms'],100)

    def test_no_new_order_no_pending_create_no_cancel_undo(self):
        p,g=self.setup_policy();g.apply(20,row(False,.045),p)
        self.assertEqual(g.actions[0]['status'],'no_accepted_small_base');self.assertEqual(p.stats['places'],1)
        g.apply(60,row(False,.045),p);self.assertEqual(p.stats['places'],1);self.assertFalse(g.holds)
        p,g=self.setup_policy();p.cancel(p.orders[0],90,'external');g.apply(100,row(False,.045),p)
        self.assertFalse(g.holds);self.assertEqual(p.orders[0].cancel_at,112)

    def test_large_order_and_rung_duplicate_refused(self):
        p,g=self.setup_policy();p.orders[0].size=25;p.orders[0].rem=25;g.apply(100,row(False,.045),p)
        self.assertFalse(g.holds)
        p,g=self.setup_policy();g.apply(100,row(False,.045,rung=True),p)
        self.assertEqual(g.actions[0]['status'],'duplicate_rung');self.assertFalse(g.holds)

    def test_zero_delta_exact_planner_parity(self):
        p=QuotePlanner(incremental=True);q=QuotePlanner(incremental=True);g=SoftPull(0)
        for t,r in [(0,row()),(100,row(False,.045)),(150,row()),(175,row(tick=49)),(200,row(False,.045))]:
            g.expire(t,p);g.apply(t,r,p);q.plan(t,lanes(r))
        self.assertEqual(p.commands,q.commands);self.assertEqual(p.stats,q.stats)

    def test_prefix_causality_and_bad_parameters(self):
        p,g=self.setup_policy();g.apply(100,row(False,.045),p)
        snapshot=list(p.commands);g.expire(150,p);self.assertEqual(snapshot,p.commands)
        for delta in [-.1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):SoftPull(delta)

    def test_price_labels_no_spread_and_stale_refusal(self):
        r=dict(start=100.,midpoint=50.,lane=0)
        book=np.array([[90.,50.],[1090.,49.],[5090.,52.]])
        self.assertEqual(price_label(r,book,1000),('valid',-1.))
        r['lane']=1;self.assertEqual(price_label(r,book,1000),('valid',1.))
        self.assertEqual(price_label(r,book,5000),('valid',-2.))
        self.assertEqual(price_label(r,book,2000)[0],'stale_frame')
        r['midpoint']=51.;self.assertEqual(price_label(r,book,1000)[0],'start_identity_ambiguous')


if __name__=='__main__':unittest.main()
