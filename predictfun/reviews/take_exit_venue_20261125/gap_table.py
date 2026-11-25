"""Gap from a candidate Binance signal to the Predict match (print), per competitor order: futures tick (last same-direction mid change
# before arrival A = print - 150, i.e. a tick they could have reacted to) and, for early orders, the futures tick that came after A.
Every order waits the 150 ms hold, so a signal at S that triggered a taker needs match - S > 150 ms. Output (write-once): gap_table.json.
"""
import json
import sys, numpy as np, pandas as pd
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import lead_venues as LV
E = pd.read_parquet(LV.OUT / 'lead_events.parquet', columns=['day', 'A', 'd', 'cls', 'tf_after'])
gap_late, gap_early_after = [], []
for day in LV.DAYS:
    bk, bm = LV.OD.seg_arrays(LV.OD.SEG_OF[day]); bk = bk.astype(float)
    X = E[E.day == day]
    for a, d, c, tfa in zip(X.A.values, X.d.values, X.cls.values, X.tf_after.values):
        M = a + 150.
        if c == 'late':
            j = int(np.searchsorted(bk, a, 'right') - 1)
            while j > 0 and not (d * (bm[j] - bm[j - 1]) > 0):
                j -= 1
            if j > 0 and a - bk[j] < 2000:
                gap_late.append(M - bk[j])
        elif np.isfinite(tfa):
            gap_early_after.append(M - tfa)
q = lambda x: [round(float(np.quantile(x, p))) for p in (.1, .25, .5, .75, .9)]
fn = LV.H / 'gap_table.json'
assert not fn.exists(), 'gap_table.json exists (write-once)'
LV.TR.dump(fn, dict(asof='2026-11-25', quantiles=[.1, .25, .5, .75, .9], late_tick_to_match_ms=q(gap_late), late_n=len(gap_late), early_next_tick_to_match_ms=q(gap_early_after), early_n=len(gap_early_after)))
print('late orders: futures tick -> Predict match (ms) p10/p25/p50/p75/p90:', q(gap_late), 'n', len(gap_late))
print('early orders: the futures tick that came after they sent -> Predict match (ms):', q(gap_early_after), 'n', len(gap_early_after))
