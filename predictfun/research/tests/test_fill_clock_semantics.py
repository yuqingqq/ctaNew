import unittest
import pandas as pd
from predictfun.research import pf_fill_clock_semantics as s


class SemanticsControls(unittest.TestCase):
    def test_timestamp_drift_kept_by_exact_identity(self):
        events=pd.DataFrame([dict(order_hash='h',trade_id='t',q=5.,price=.5,status=st,at=at,reported_match_ms=ts)
            for st,at,ts in [('Matched',150.,0.),('Confirmed',10150.,10000.),('Confirmed',10250.,10000.)]])
        x=s.status_stamps(events,'h','t')
        self.assertEqual(x['Matched']['stamps'],[0.]);self.assertEqual(x['Confirmed']['stamps'],[10000.])
        self.assertEqual(x['Confirmed']['n'],2)
        with self.assertRaises(ValueError):s.status_stamps(events,'h','wrong')
        events.loc[1,'q']=6
        with self.assertRaises(ValueError):s.status_stamps(events,'h','t')

    def test_missing_status_is_not_imputed(self):
        events=pd.DataFrame([dict(order_hash='h',trade_id='t',q=5.,price=.5,status='Confirmed',at=150.,reported_match_ms=0.)])
        self.assertEqual(s.status_stamps(events,'h','t')['Matched'],dict(n=0,first_receipt=None,stamps=[]))

    def test_both_second_assumptions(self):
        self.assertTrue(s.overlap(1000,1200,1400))
        self.assertTrue(s.overlap(1000,700,800,True))
        self.assertFalse(s.overlap(1000,700,800))
        self.assertFalse(s.overlap(1000,2000,2100))
        with self.assertRaises(ValueError):s.overlap(1000,1400,1200)

    def test_constant_offset_impossibility(self):
        q=pd.DataFrame(dict(Z=[1000.,1000.],lower=[100.,1100.],upper=[200.,1200.],order_hash=['a','b']))
        self.assertFalse(s.constant_interval(q)['exists_at10ms'])
        self.assertTrue(s.constant_interval(q.iloc[:1])['exists'])
        q.loc[0,'Z']=float('nan')
        with self.assertRaises(ValueError):s.constant_interval(q)


if __name__=='__main__':unittest.main()
