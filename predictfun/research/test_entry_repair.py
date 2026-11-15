"""Synthetic controls for date filtering, sampling, queue sensitivity and size."""
import copy
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent))
import pf_entry_repair as P
import pf_joint_ladder as J


def context(legs=()):
    W={0:J._ctx(0,[(0.,[(50,100),(49,100),(48,100),(47,100),(46,100)],52)],list(legs),t_end=1000.),
       1:J._ctx(1,[(0.,[(52,100),(53,100),(54,100),(55,100),(56,100)],50)],[],t_end=1000.)}
    for w in W.values():
        w['m']=1;w['vkl']=[1.]*len(w['tg']);w['tf_mff']=(w['tf'],np.array([51.]))
    return W


def compact(r):
    return dict(fills=r['fills'],orders=r['orders'],decisions=r['policy_decisions'],stat=r['stat'])


class Checks(unittest.TestCase):
    def test_source_migration_is_narrow(self):
        meta=P.M.stamp();meta['sources']['pf_kp_s4.py']=P.OLD_LOADER_SHA
        P.require_migrated(meta)
        bad=copy.deepcopy(meta);bad['sources']['pf_wait_post_v2.py']='bad'
        with self.assertRaises(ValueError):P.require_migrated(bad)
        bad=copy.deepcopy(meta);bad['sources']['pf_kp_s4.py']='bad'
        with self.assertRaises(ValueError):P.require_migrated(bad)

    def test_boundaries_fail_before_read(self):
        self.assertTrue(P.allowed_part('part_20261026_23.parquet','20261027'))
        self.assertTrue(P.allowed_part('part_20261027_23.parquet','20261027'))
        self.assertFalse(P.allowed_part('part_20261028_00.parquet','20261027'))
        self.assertFalse(P.allowed_part('part_20261030_23.parquet','20261031'))
        self.assertFalse(P.allowed_part('part_20261104_00.parquet','20261103'))
        with self.assertRaises(ValueError):P.allowed_part('garbage.parquet','20261027')
        with self.assertRaises(ValueError):P.allowed_part('part_20261104_00.parquet','20261104')
        touched=[]
        def reader(path,**kw):touched.append(path);return object()
        def loader(day):
            self.assertTrue(P.K.SKIP_FOREIGN_PARTS)
            pd.read_parquet('/synthetic/books/part_20261027_00.parquet')
            pd.read_parquet('/synthetic/books/part_20261028_00.parquet')
        with mock.patch.object(pd,'read_parquet',reader),mock.patch.object(P.K,'load_day',loader):
            with self.assertRaisesRegex(ValueError,'before read'):P.load_day('20261027')
        self.assertEqual(touched,['/synthetic/books/part_20261027_00.parquet'])

    def test_sampling_preserves_subset_and_age(self):
        records=[];new=0;old=0
        for generation in range(1,10001):
            key=(0,0,generation);s=P.sampling('20261025',1,key);h=P.M.hash_key('20261025',1,key)
            self.assertEqual(s['requested_delay'],P.M.AGES[(h//32)%len(P.M.AGES)])
            self.assertFalse(s['old_selected'] and not s['selected'])
            new+=s['selected'];old+=s['old_selected'];records.append(dict(generation=generation,**s))
        self.assertTrue(2200<new<2800);self.assertTrue(220<old<420)
        f=pd.DataFrame(records);np.testing.assert_equal(P.weights(f),np.full(len(f),4.))
        for field,value in [('inclusion_probability',0.),('inclusion_probability',2.),('sampling_weight',32.),('sampling_version','old')]:
            bad=f.copy();bad.loc[0,field]=value
            with self.assertRaises(ValueError):P.weights(bad)
        with self.assertRaises(ValueError):P.weights(f.drop(columns='inclusion_probability').assign(inclusion_probability=np.nan))

    def test_original_lane_exact_parity(self):
        W=context([(200.,50,125.),(400.,45,50.),(700.,49,20.)])
        a=P.N.run(copy.deepcopy(W),lambda v:True);b=P.run(copy.deepcopy(W))
        self.assertEqual(compact(a),compact(b))
        self.assertEqual(P.N.E.terminal(W,a),P.N.E.terminal(W,b))

    def test_unknown_queue_control_and_trade_through(self):
        # Rung 5 is outside top five. A small at-price trade must NOT assume front.
        W=context([(250.,45,10.)]);choose=lambda v:v['side']==0 and v['rung']==5
        zero=P.run(copy.deepcopy(W),choose=choose);block=P.run(copy.deepcopy(W),queue='block',choose=choose)
        self.assertEqual(sum(f[2] for f in zero['fills'][0,5]),10.)
        self.assertEqual(block['fills'][0,5],[])
        self.assertTrue(zero['order_states'][0,5][0]['initial_queue_unknown'])
        sweep=P.run(context([(250.,44,10.)]),queue='block',choose=choose)
        self.assertEqual(sum(f[2] for f in sweep['fills'][0,5]),10.)
        with self.assertRaises(ValueError):P.run(context(),queue='invalid')

    def test_size_retains_same_order_through_partial_fills(self):
        # Sweeps avoid assuming an observed queue position; 30 is exhausted twice,
        # 100 retains the original order through all three events.
        W=context([(200.,49,30.),(500.,49,30.),(800.,49,30.)]);choose=lambda v:v['side']==0 and v['rung']==0
        small=P.run(copy.deepcopy(W),30.,choose=choose);large=P.run(copy.deepcopy(W),100.,choose=choose)
        self.assertGreater(len(small['order_states'][0,0]),len(large['order_states'][0,0]))
        first=large['order_states'][0,0][0];self.assertEqual(len(first['fl']),3)
        self.assertEqual(first['rem'],10.);self.assertEqual(first['ta'],40.)
        self.assertEqual(sum(q for _,q in first['fl']),90.)
        self.assertFalse(any(large['stat']['viol'].values()))

    def test_label_reconciliation_refuses_changed_target(self):
        f=pd.DataFrame([dict(qpost=2.,qwait=1.,delta=1.,local_post=1.,external_post=1.,local_delta=.25,external_delta=.75)])
        P.N.check_targets(f)
        f.loc[0,'delta']=2.
        with self.assertRaises(ValueError):P.N.check_targets(f)

    def test_weighted_baseline_matches_original(self):
        f=pd.DataFrame(dict(rung=[0]*120+[1]*120,price=[50]*240,generation=np.arange(240)%3,qpost=np.arange(240)/10))
        w=P.N.S.weights(f)
        np.testing.assert_equal(P.baseline(f,f,w,'qpost'),P.N.S.baseline(f,f,'qpost'))
        altered=P.baseline(f,f,np.ones(len(f)),'qpost')
        self.assertFalse(np.array_equal(altered,P.baseline(f,f,w,'qpost')))


if __name__=='__main__':unittest.main()
