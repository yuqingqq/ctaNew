"""Reproduction predicates of settlement times v2 (sealed / forward hours never opened), declared before v2 read any data (2026-11-18,
session aea5bfb9). v1 = settlement_times_20261118 (its 20261027 output VOID: it opened the sealed 20261028_00 / _01 hours).
 S1 days 20261023..26 never reach an excluded hour: v2's hours read equal v1's, no hour excluded, and legs_set (t_set_ms, set_status) is
    identical to v1's row by row.
 S2 day 20261027: v2 excludes exactly 20261028_00 and 20261028_01 and reads every other v1 hour; every leg outside the status-5 rows has
    v1's t_set_ms and set_status; every status-5 leg's executedAt lies within 2 h before 10-28 00Z.
 S3 every v2 day's controls C1-C4 pass (its receipt).
Usage: python3 compare_settlement.py <out.txt>"""
import json
import sys

import numpy as np
import pandas as pd

D = '/home/yuqing/ctaNew/data/predict_fun/research/'
B_MS = 1793145600000.                                     # 2026-10-28 00:00Z


def day(d):
    a, b = (json.load(open(D + f'{v}/{d}/settlement.json')) for v in ('settlement_times_20261118', 'settlement_times_v2_20261118'))
    A, Bv = (pd.read_parquet(D + f'{v}/{d}/legs_set.parquet') for v in ('settlement_times_20261118', 'settlement_times_v2_20261118'))
    same_t = np.array_equal(A.t_set_ms.values, Bv.t_set_ms.values, equal_nan=True); same_s = np.array_equal(A.set_status.values, Bv.set_status.values)
    out = dict(day=d, controls=all(b['R']['controls'].values()), hours_v1=len(a['R']['hours']), hours_v2=len(b['R']['hours']), excluded=b['R'].get('hours_excluded'),
               rows=(len(A), len(Bv)), identical=bool(same_t and same_s), status_v2=b['R']['status'])
    if d == '20261027':
        k = Bv.set_status.values != 5; legs = pd.read_parquet(D + f'step1/{d}/legs_timed2.parquet', columns=['exec_s'])
        ex = legs.exec_s.values.astype(float)[~k] * 1000.
        out.update(outside_status5_identical=bool(np.array_equal(A.t_set_ms.values[k], Bv.t_set_ms.values[k], equal_nan=True) and np.array_equal(A.set_status.values[k], Bv.set_status.values[k])),
                   status5=int((~k).sum()), status5_within_2h=bool(((ex >= B_MS - 7200000.) & (ex < B_MS)).all()) if len(ex) else True,
                   v1_status_of_status5=pd.Series(A.set_status.values[~k]).value_counts().to_dict(),
                   reads_every_other_v1_hour=sorted(set(a['R']['hours']) - set(b['R']['hours'])) == ['20261028_00', '20261028_01'])
    return out


if __name__ == '__main__':
    R = {d: day(d) for d in ('20261023', '20261024', '20261025', '20261026', '20261027')}
    S1 = all(R[d]['identical'] and R[d]['excluded'] == [] and R[d]['hours_v1'] == R[d]['hours_v2'] for d in ('20261023', '20261024', '20261025', '20261026'))
    x = R['20261027']; S2 = x['excluded'] == ['20261028_00', '20261028_01'] and x['reads_every_other_v1_hour'] and x['outside_status5_identical'] and x['status5_within_2h']
    S3 = all(R[d]['controls'] for d in R)
    out = dict(S1=S1, S2=S2, S3=S3, ALL=S1 and S2 and S3, days=R)
    txt = json.dumps(out, indent=1, default=str); open(sys.argv[1], 'w').write(txt + '\n'); print(txt)
