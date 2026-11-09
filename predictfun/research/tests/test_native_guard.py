import gzip
import tempfile
from pathlib import Path
import unittest
from predictfun.research.pf_native_guard import prefix_time,bound


class NativeGuardControls(unittest.TestCase):
    def test_timestamp_envelopes_and_unknown_refusal(self):
        self.assertEqual(prefix_time(b'{"Frame":{"market":1,"known_ms":12.5,"bids":[]}}','inputs'),12.5)
        self.assertIsNone(prefix_time(b'{"OpenWindow":{"market":1,"ws_s":10}}','inputs'))
        self.assertEqual(prefix_time(b'{"Request":{"Cancel":{"market":1,"t":22.5}}}','outputs'),22.5)
        with self.assertRaises(ValueError):prefix_time(b'{"FutureSchema":{"time":3}}','inputs')
    def test_future_payload_never_decoded_or_copied(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);src=p/'in.gz';dst=p/'out.gz'
            with gzip.open(src,'wb') as f:f.write(b'{"protocol":"test"}\n{"Advance":{"now_ms":99}}\n{"Frame":{"known_ms":100,FUTURE_INVALID_JSON\n')
            r=bound(src,dst,'inputs',100)
            self.assertTrue(r['stopped_before_future_payload']);self.assertEqual(r['records'],1)
            with gzip.open(dst,'rb') as f:self.assertEqual(f.read(),b'{"protocol":"test"}\n{"Advance":{"now_ms":99}}\n')
            with self.assertRaises(FileExistsError):bound(src,dst,'inputs',100)
    def test_truncated_in_scope_gzip_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);src=p/'in.gz'
            with gzip.open(src,'wb') as f:f.write(b'{"Decision":{"market":1,"t":99}}\n')
            src.write_bytes(src.read_bytes()[:-3])
            with self.assertRaises(EOFError):bound(src,p/'out.gz','outputs',100)


if __name__=='__main__':unittest.main()
