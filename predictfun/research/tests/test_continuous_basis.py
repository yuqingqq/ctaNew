import gzip
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from predictfun.research import pf_basis_continuous as b
from predictfun.research import pf_step5_features_continuous as f


def bn(t, mid):
    return {'BinanceBook':dict(known_ms=t, bid=mid-1, ask=mid+1)}


def relay(t, price):
    return {'Relay':dict(known_ms=t, price=price)}


class ContinuousBasis(unittest.TestCase):
    def test_ties_missing_and_distinct_equal_prints(self):
        events = [relay(0,100), bn(1,100), relay(2,110), bn(2,110), relay(2,110), relay(2,110)]
        rows, counts, _, _ = b.samples(events)
        self.assertEqual(rows, [(2,1.1,3),(2,1.,5),(2,1.,6)])
        self.assertEqual((counts['relay_events'],counts['sampled'],counts['no_binance']),(4,3,1))
        self.assertNotEqual(rows[0][1], 1., 'future equal-time Binance cannot price an earlier relay')

    def test_boundary_and_chunks_do_not_duplicate(self):
        events = [e for i in range(160) for e in (bn(3_600_000-80_000+i*1000,100),relay(3_600_000-80_000+i*1000,100+i*.01))]
        one = b.samples(events)[0]
        for k in [1,17,10000]:
            chunks = [events[i:i+k] for i in range(0,len(events),k)]
            self.assertEqual(b.samples(e for c in chunks for e in c)[0], one)
        shifted = [{kind:{**v,'known_ms':v['known_ms']+120_000}} for event in events for kind,v in event.items()]
        self.assertEqual([(t-120_000,v,i) for t,v,i in b.samples(shifted)[0]],one)
        self.assertEqual(len(one),160)
        bad = one + one[70:80]
        self.assertNotEqual(len(bad),len(one), 'legacy carry creates extra samples')

    def test_invalid_evidence_refused(self):
        for events in [[bn(2,100),relay(1,100)], [relay(float('nan'),100)],
                       [{'RelaySample':dict(known_ms=1,ratio=1)}], [bn(0,100),relay(1,-1)]]:
            with self.assertRaises(ValueError): b.samples(events)

    def test_artifact_identity_and_source_patch(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'inputs.gz'; out=Path(td)/'samples'
            with gzip.open(p,'wt') as z:
                z.write(json.dumps({'protocol':'PFMM_LADDER_JOURNAL_V3'})+'\n')
                for event in [bn(0,100),relay(1,110)]: z.write(json.dumps(event)+'\n')
            with self.assertRaises(ValueError): b.build(p,'0'*64,out)
            b.build(p,b.sha(p),out)
            t,v=b.load(out)
            self.assertEqual((t.tolist(),v.tolist()),([1.],[1.1]))
            (out/'samples.npz').write_bytes(b'corrupt')
            with self.assertRaises(ValueError): b.load(out)
        self.assertIn(f.NEW,f.source())
        self.assertNotIn(f.OLD,f.source())
        with patch.object(f,'BASE_SHA','0'*64):
            with self.assertRaises(ValueError): f.source()
        with patch.dict(os.environ,{'PF_BASIS_SAMPLES':'unused','PF_ROWS_SUFFIX':''}):
            with self.assertRaises(ValueError): f.run()


if __name__ == '__main__': unittest.main()
