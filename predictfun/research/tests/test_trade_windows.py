import unittest
import pandas as pd
from predictfun.research.pf_trade_windows import select_trade_windows


class TradeWindowControls(unittest.TestCase):
    def test_actual_match_time_selects_phase_and_retains_counts(self):
        # Confirmation is deliberately the same later value for every event.
        f=pd.DataFrame(dict(mid=[42]*4,t_ms=[999.,1000.,300999.,301000.],confirmation_ns=[999999999999]*4))
        reg=pd.DataFrame(dict(mid=[42],ws=[1.],we=[301.])).set_index('mid')
        result,counts=select_trade_windows(f,reg)
        self.assertEqual(result.index.tolist(),[1,2])
        self.assertEqual(counts,dict(before_window=1,in_window=2,after_window=1))
        self.assertEqual(sum(counts.values()),len(f))
        self.assertEqual(len(f),4)

    def test_bad_population_and_clock_refused(self):
        reg=pd.DataFrame(dict(mid=[42],ws=[1.],we=[301.])).set_index('mid')
        with self.assertRaises(ValueError):select_trade_windows(pd.DataFrame(dict(mid=[43],t_ms=[1000.])),reg)
        with self.assertRaises(ValueError):select_trade_windows(pd.DataFrame(dict(mid=[42],t_ms=[float('nan')])),reg)
        reg.loc[42,'we']=301.5
        with self.assertRaises(ValueError):select_trade_windows(pd.DataFrame(dict(mid=[42],t_ms=[1000.])),reg)


if __name__=='__main__':unittest.main()
