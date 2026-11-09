import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
from predictfun.research import pf_fill_clock as c


class ClockControls(unittest.TestCase):
    def test_hash_identity(self):
        self.assertEqual(c.hex_id([1]*32),'0x'+'01'*32)
        for v in [[1]*31,[256]*32,['1']*32]:
            with self.assertRaises(ValueError):c.hex_id(v)

    def test_bounds_and_acceptance_not_lower_bound(self):
        lo,hi=c.causal_bounds(100,180,250)
        self.assertEqual((lo,hi),(100,180))
        self.assertEqual(c.classify(120,lo,hi),'compatible')
        self.assertEqual(c.classify(90,lo,hi),'before_place')
        self.assertEqual(c.classify(181,lo,hi),'after_terminal_or_notice')
        self.assertEqual(c.classify(181,lo,hi,10),'compatible')
        with self.assertRaises(ValueError):c.causal_bounds(100,90,250)
        with self.assertRaises(ValueError):c.classify(float('nan'),lo,hi)

    def test_ambiguous_leg_refusal(self):
        a=pd.DataFrame(dict(Z=[120.,121.,120.],J=[110.,110.,110.],X=[70.,71.,70.]))
        self.assertEqual(c.unique_anchor([0],a)[1],'unique_leg')
        self.assertEqual(c.unique_anchor([0,2],a)[1],'identical_clock_candidates')
        self.assertEqual(c.unique_anchor([0,1],a),(None,'ambiguous_leg_clock'))
        self.assertEqual(c.unique_anchor([],a),(None,'leg_unmapped'))

    def test_bounded_reader_refuses_future_payload(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x'
            p.write_text(json.dumps({'frame':{'at':100,'input':'Tick'},'kind':'step'},separators=(',',':'))+'\n'+
                '{"frame":{"at":'+str(c.END_MS*1000000)+',BROKEN FUTURE PAYLOAD\n')
            st={};self.assertEqual(len(list(c.bounded_steps(p,st))),1)
            self.assertTrue(st['stopped_before_cutoff_payload'])
            p.write_text('{"frame":{"at":100,"input":"Tick"},"kind":"step"}\n{"frame":{"at":99,"input":"Tick"},"kind":"step"}\n')
            with self.assertRaises(ValueError):list(c.bounded_steps(p,{}))

    def test_source_refusal(self):
        compile(c.source_prefix(),'prefix','exec')
        with patch.object(c,'sha',return_value='bad'):
            with self.assertRaises(ValueError):c.source_prefix()

    def test_notice_dedup_and_conflict_refusal(self):
        e=pd.DataFrame([dict(run='v6',order_hash='h',trade_id='t',cid=1,q=5.,price=.5,at=t,
            status=s,reported_match_ms=100.) for t,s in [(120,'Matched'),(130,'Confirmed'),(140,'Confirmed')]])
        d=c.dedup_events(e)
        self.assertEqual(len(d),1);self.assertEqual(d.iloc[0].q,5);self.assertEqual(d.iloc[0].first_positive_notice,120)
        e.loc[1,'q']=6
        with self.assertRaises(ValueError):c.dedup_events(e)


if __name__=='__main__':unittest.main()
