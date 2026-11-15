"""Synthetic falsifiers for future-only timing masks and controls for certificates."""
import copy
from dataclasses import replace
import json
import math
import resource
import unittest
from datetime import datetime, timezone

import numpy as np

import pf_stop_ev_accounting as A
import pf_stop_ev_accounting_check as T
import pf_stop_ev_timing_certificate as C


def add_leg(W, side, time, price, quantity):
    w=copy.deepcopy(W); x=w[side]
    legs=sorted([*zip(x['anc'],x['ptl'],x['shl']),(time,price,quantity)],key=lambda z:(z[0],-z[1] if side==0 else z[1]))
    x['anc']=np.array([a for a,_,_ in legs]);x['ptl']=[p for _,p,_ in legs];x['shl']=[q for _,_,q in legs]
    return w


def missing(W, side, price, *, lo=-math.inf, hi=math.inf, qty=10., reason='untimed', nr=1, size=30.):
    l=T.ledger(W,nr=nr,size=size)
    c=replace(l.coverage,expected_legs=l.coverage.expected_legs+1,
              missing=(A.MissingLeg('absent',lo,hi,reason),))
    l=T.ledger(W,result=l.result,coverage=c,nr=nr,size=size)
    return l,C.Certificate(l,[C.MissingInfo('absent',side,price,qty)])


class Checks(unittest.TestCase):
    def test_earlier_queue_drain_changes_later_value(self):
        W=T.world(((200.,50,5.),),queue=10.)
        l,c=missing(W,0,50,lo=100.,hi=100.)
        baseline=T.action(T.ledger(W),150.)['label']['value_cents']
        changed=T.action(T.ledger(add_leg(W,0,100.,50,10.)),150.)['label']['value_cents']
        self.assertEqual((baseline,changed),(0.,5.))
        self.assertFalse(100. >= 150.)  # future-only [t,t+H] mask misses it
        self.assertEqual(c.label((0,0,0),150.,day='toy')['status'],'uncertain_prior_reference')

    def test_earlier_fill_changes_order_birth(self):
        W=T.world(((400.,50,10.),))
        base=T.ledger(W); alt=T.ledger(add_leg(W,0,100.,50,30.))
        self.assertEqual(base.orders[0,0,0]['ta'],40.)
        self.assertEqual(alt.orders[0,0,1]['ta'],240.)
        _,c=missing(W,0,50,lo=100.,hi=100.,qty=30.)
        self.assertFalse(c.assess(0,300.,1200.)['certified'])

    def test_trade_at_better_price_changes_shared_volume(self):
        W=T.world(((400.,49,40.),)); l,c=missing(W,0,50,lo=250.,hi=250.,qty=30.,nr=2)
        base=T.action(T.ledger(W,nr=2),300.,rung=1)['label']['filled_quantity']
        alt=T.action(T.ledger(add_leg(W,0,250.,50,30.),nr=2),300.,rung=1)['label']['filled_quantity']
        self.assertEqual((base,alt),(10.,30.))
        self.assertGreater(50,49)  # cannot reach focal bid directly
        self.assertFalse(c.assess(0,300.,1200.)['certified'])

    def test_opposite_side_is_invariant_without_cap(self):
        for side in (0,1):
            p=50 if side==0 else 52; other=1-side; op=52 if other==1 else 50
            W=T.world(((200.,p,10.),),side=side)
            l,c=missing(W,other,op,qty=100.)
            row=c.label((side,0,0),50.,day='toy')
            self.assertEqual(row['status'],'eligible_timing_invariant')
            self.assertEqual(row['source_accounting']['missing_legs'],1)
            for t in (0.,39.,40.,60.,100.,199.,200.,900.):
                alt=T.action(T.ledger(add_leg(W,other,t,op,100.)),side=side)
                self.assertEqual(alt['label'],row['label'])
                self.assertEqual(alt['features'],row['features'])

    def test_unreachable_price_is_invariant_even_with_multiple_missing_legs(self):
        for side,p,unreachable in ((0,50,60),(1,52,40)):
            W=T.world(((200.,p,10.),),side=side); l,c=missing(W,side,unreachable,qty=100.)
            row=c.label((side,0,0),50.,day='toy')
            self.assertEqual(row['status'],'eligible_timing_invariant')
            changed=W
            for t in (40.,50.,100.,199.,200.,500.): changed=add_leg(changed,side,t,unreachable,100.)
            alt=T.action(T.ledger(changed),side=side)
            self.assertEqual(alt['label'],row['label'])

    def test_known_interval_can_be_irrelevant_but_empirical_untimed_interval_cannot(self):
        W=T.world()
        _,c=missing(W,0,50,lo=2000.,hi=3000.,reason='known_outside')
        self.assertTrue(c.assess(0,50.,1050.)['certified'])
        _,c=missing(W,0,50,lo=2000.,hi=3000.,reason='untimed')
        self.assertFalse(c.assess(0,50.,1050.)['certified'])

    def test_e2_modes_keep_qualifications(self):
        l=T.ledger(T.world(),info={(0,0):A.LegInfo('0:0','e2',250.)}); c=C.Certificate(l,[])
        self.assertEqual(c.label((0,0,0),50.,day='toy')['status'],'uncertain_prior_reference')
        row=c.label((0,0,0),50.,day='toy',estimated=True)
        self.assertEqual(row['status'],'eligible_estimated_timing')
        self.assertTrue(row['timing_certificate']['e2_interval_is_assumption'])
        self.assertEqual(c.label((1,0,0),50.,day='toy')['status'],'eligible_timing_invariant')

    def test_refuses_cap_metadata_and_bad_provenance(self):
        l=T.ledger(); l.result['local_reference']['mode']='hard';l.result['local_reference']['cap']=30.
        with self.assertRaisesRegex(ValueError,'uncapped'): C.Certificate(l,[])
        l,_=missing(T.world(),0,50)
        with self.assertRaisesRegex(ValueError,'reconcile'): C.Certificate(l,[])
        with self.assertRaisesRegex(ValueError,'metadata'): C.Certificate(l,[C.MissingInfo('absent',0,50,-1.)])

    def test_invalid_marks_are_not_rescued_by_certificate(self):
        l,c=missing(T.world(((800.,50,10.),)),1,52)
        l.W[0]['opp'][l.W[0]['tf']>=1000.]=-1
        self.assertEqual(c.label((0,0,0),50.,day='toy')['status'],'missing_or_stale_mark')


if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.
    print(json.dumps(dict(asof=datetime.now(timezone.utc).isoformat(),tests=r.testsRun,
                         failures=len(r.failures),errors=len(r.errors),passed=r.wasSuccessful() and peak<2048.,peak_rss_mib=peak)))
    raise SystemExit(0 if r.wasSuccessful() and peak<2048. else 1)
