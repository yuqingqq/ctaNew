import gzip
from pathlib import Path
import tempfile
import unittest
from predictfun.research.pf_quote_capture_study import decision_rows


class ReaderControls(unittest.TestCase):
    def test_future_payload_never_decoded(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.gz'
            with gzip.open(p,'wb') as f:
                f.write(b'{"Decision":{"market":1,"t":1000.0,"lanes":[[50,5],[null,0],[null,0],[null,0]]}}\n')
                f.write(b'{"Decision":{"market":1,"t":1794178920000.0,BROKEN FUTURE\n')
            s={};rows=list(decision_rows(p,{1:1},s))
            self.assertEqual(len(rows),1);self.assertEqual(rows[0]['size0'],5)
            self.assertTrue(s['stopped_before_future_payload'])

    def test_bad_shape_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.gz'
            with gzip.open(p,'wb') as f:f.write(b'{"Decision":{"market":1,"t":1000.0,"lanes":[]}}\n')
            with self.assertRaises(ValueError):list(decision_rows(p,{1:1},{}))


if __name__=='__main__':unittest.main()
