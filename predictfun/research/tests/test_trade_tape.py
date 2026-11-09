import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
from predictfun.research import pf_trade_tape as t
from predictfun.research import pf_step4_ladder_identity as engine

H=lambda n:'0x'+f'{n:064x}'


def fixture():
    maker=dict(hash=H(1),amount=str(5*10**18),price=str(31*10**16),outcome={'indexSet':1},quoteType='Bid',signer=H(9))
    trade=dict(transactionHash=H(2),settlementId='settlement-1',market={'id':42},executedAt='2026-11-08T00:00:10.000Z',
               taker=dict(hash=H(3),outcome={'indexSet':1},quoteType='Ask'),priceExecuted=str(31*10**16),makers=[maker])
    return trade,maker


def events():
    return [t.status_event(H(1),'settlement-1',s,ns,ns+400_000_000,5_000_000,310_000)
            for s,ns in [('Matched',10**9),('Confirmed',11*10**9),('Confirmed',11*10**9)]]


class TradeTapeControls(unittest.TestCase):
    def test_full_hash_collision_and_taker_identity(self):
        x,m=fixture();a=t.rest_leg(x,m,10**9)
        m['hash']=H(4);b=t.rest_leg(x,m,10**9)
        self.assertEqual(a['maker_order_hash'][2:18],b['maker_order_hash'][2:18])
        self.assertNotEqual(a['leg_id'],b['leg_id'])
        x['taker']['hash']=H(5);d=t.rest_leg(x,m,10**9)
        self.assertNotEqual(b['leg_id'],d['leg_id'])
        self.assertEqual(t.transaction_groups(pd.DataFrame([a,b,d])).tolist(),[0,0,1])
        with self.assertRaises(ValueError):t.hash32('0x1234')

    def test_down_coordinate_and_no_receipt_rounding(self):
        x,m=fixture();m.update(price=str(69*10**16),outcome={'indexSet':2})
        a=t.rest_leg(x,m,1794174957481512201)
        self.assertEqual((a['side'],a['ptick']),(1,31))
        self.assertEqual(a['first_recv_ns'],1794174957481512201)
        m['price']=str(691*10**15)
        off=t.rest_leg(x,m,10**9)
        self.assertIsNone(off['ptick']);self.assertEqual(off['price_grid_status'],'off_cent_grid')
        self.assertEqual(off['token_price_wei'],m['price'])

    def test_duplicate_pages_do_not_add_fills_or_hide_conflicts(self):
        x,m=fixture();a=t.rest_leg(x,m,20);b=t.rest_leg(x,m,10)
        t.merge_leg(a,b)
        self.assertEqual(a['first_recv_ns'],10);self.assertEqual(a['last_recv_ns'],20)
        self.assertEqual(a['shares'],5);self.assertEqual(a['observation_count'],2)
        b['amount_wei']=str(6*10**18)
        with self.assertRaises(ValueError):t.merge_leg(a,b)

    def test_later_confirmation_never_overwrites_match(self):
        r=t.resolve_status(events())
        self.assertEqual((r['match_lo_ns'],r['match_hi_ns']),(10**9,2*10**9))
        self.assertEqual(r['status_changed_ns'],10*10**9)
        self.assertEqual(r['clock_status'],'matched_status_interval')
        q=t.resolve_status(events()[1:])
        self.assertEqual(q['clock_status'],'missing_matched_event');self.assertIsNone(q['match_lo_ns'])

    def test_conflicting_statuses_retained_as_status(self):
        e=events();e.append(t.status_event(H(1),'settlement-1','Matched',2*10**9,3*10**9,5_000_000,310_000))
        self.assertEqual(t.resolve_status(e)['clock_status'],'conflicting_matched_event')
        self.assertIsNone(t.resolve_status(e)['match_lo_ns'])
        e.append(t.status_event(H(1),'settlement-1','Failed',4*10**9,5*10**9,5_000_000,310_000))
        self.assertEqual(t.resolve_status(e)['clock_status'],'conflicting_terminal_status')

    def test_exact_join_no_other_leg_clock_substitution(self):
        x,m=fixture();a=t.rest_leg(x,m,10);m['hash']=H(4);b=t.rest_leg(x,m,10)
        r,used=t.attach_status([a,b],events())
        self.assertEqual(len(used),1)
        self.assertEqual(r[0]['clock_status'],'matched_status_interval')
        self.assertEqual(r[1]['clock_status'],'missing_matched_event')
        self.assertIsNone(r[1]['match_lo_ns'])
        bad=events();bad[0]['quantity_micros']=6_000_000
        with self.assertRaises(ValueError):t.attach_status([a,b],bad)
        with self.assertRaises(ValueError):t.attach_status([a,a],events())

    def test_identity_corruption_and_mixed_status_refused(self):
        x,m=fixture();a=t.rest_leg(x,m,10);a['leg_id']='bad'
        with self.assertRaises(ValueError):t.transaction_groups(pd.DataFrame([a]))
        with self.assertRaises(ValueError):t.transaction_groups(pd.DataFrame([{'mid':42,'t_ms':123}]))
        e=events();e[-1]['trade_id']='other'
        with self.assertRaises(ValueError):t.resolve_status(e)

    def test_replay_rejects_confirmation_and_coarse_time(self):
        x,m=fixture();a=t.rest_leg(x,m,10);rows,_=t.attach_status([a],events())
        with self.assertRaisesRegex(ValueError,'point replay requires'):t.replay_times(pd.DataFrame(rows))
        rows[0].update(t_ms=1234.,how=1,exec_s=1)
        with self.assertRaises(ValueError):t.replay_times(pd.DataFrame(rows))
        # Positive fixture of a separate matching-event adapter's point contract.
        rows[0].update(clock_status='matching_engine_event',match_time_ns=1500000000,match_precision_ns=1000000,
                       match_event_id='direct-event',match_source_sha256='a'*64)
        self.assertEqual(t.replay_times(pd.DataFrame(rows)).tolist(),[1500.])
        rows[0]['price_grid_status']='off_cent_grid'
        with self.assertRaisesRegex(ValueError,'off-grid'):t.replay_times(pd.DataFrame(rows))
        rows[0]['price_grid_status']='cent_grid'
        rows[0]['match_precision_ns']=10**9
        with self.assertRaises(ValueError):t.replay_times(pd.DataFrame(rows))

    def test_transaction_disagreement_is_not_resolved_by_minimum(self):
        x,m=fixture();a=t.rest_leg(x,m,10);m['hash']=H(4);b=t.rest_leg(x,m,10)
        for row,ns in [(a,1500000000),(b,1600000000)]:
            row.update(clock_status='matching_engine_event',match_time_ns=ns,match_precision_ns=1000000,
                       match_event_id='direct-event',match_source_sha256='a'*64)
        with self.assertRaisesRegex(ValueError,'conflicting exact'):t.replay_times(pd.DataFrame([a,b]))

    def test_manifest_hash_and_population_refusals(self):
        x,m=fixture();a=t.rest_leg(x,m,10)
        a.update(clock_status='matching_engine_event',match_time_ns=1500000000,match_precision_ns=1,
                 match_event_id='direct',match_source_sha256='a'*64)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);pd.DataFrame([a]).to_parquet(p/'legs.parquet',index=False)
            manifest=dict(schema=t.SCHEMA,day='20261108',legs=1,markets=[42],legs_sha256=hashlib.sha256((p/'legs.parquet').read_bytes()).hexdigest())
            (p/'manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(len(t.load_replay_tape(p,'20261108')[0]),1)
            manifest['legs']=2;(p/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):t.load_replay_tape(p,'20261108')
            manifest['legs']=1;manifest['legs_sha256']='0'*64;(p/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):t.load_replay_tape(p,'20261108')

    def test_engine_replaces_all_heuristic_paths(self):
        source=engine.engine_source();compile(source,'checked-engine','exec')
        self.assertNotIn("groupby([LT[c] for c in key]",source)
        self.assertNotIn('t0 - u * gap',source)
        self.assertNotIn('g.exec_s.values * 1000.0 + shift',source)
        self.assertIn('LT, TXSTAT = IDENTITY_TAPE',source)
        self.assertIn('tt = g.t_ms.values',source)
        with patch.object(engine,'ENGINE_SHA','bad'):
            with self.assertRaises(ValueError):engine.engine_source()

    def test_entrypoints_fail_before_data_or_outputs(self):
        root=engine.HERE.parents[1]
        clean={k:v for k,v in os.environ.items() if not k.startswith('PF_')}
        for mod in ['pf_step1_timing','pf_step1_booktime','pf_step4_ladder_identity']:
            r=subprocess.run([sys.executable,'-m','predictfun.research.'+mod,'20261108','tokyoreal','custom'],
                             cwd=root,env=clean,text=True,capture_output=True)
            self.assertNotEqual(r.returncode,0);self.assertIn('REFUSED:',r.stderr)
            self.assertNotIn('Traceback',r.stderr)
        for script in ['run_forward.sh','run_forward_v3.sh','run_c2_day.sh']:
            r=subprocess.run(['bash',str(engine.HERE/script)],cwd=root,env=clean,text=True,capture_output=True)
            self.assertNotEqual(r.returncode,0);self.assertIn('REFUSED:',r.stderr)


if __name__=='__main__':unittest.main()
