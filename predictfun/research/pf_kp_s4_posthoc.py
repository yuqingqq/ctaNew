"""Keep-or-pull S4 post hoc (2026-11-12, after kp_s4.txt was read; no readings, nothing chosen here is validated on these days).

Why the S4 policy (pull when the predicted Delta < 0) loses to the guard trigger although its out-of-fold skill is higher: score the
same decision rule one step at a time on the never-pull rows it was trained for (S3's held-out rows, oof.parquet), against the
perfect one-step decision. If the one-step gain is positive there while the simulated policy loses, the loss comes from the policy's
own path (repeated pulls and re-entries), not from the predictions at the states the label describes.
  per day   rows with a label (y1 not NaN), share predicted < 0, share y1 < 0 / = 0 / > 0, precision of the pulls (share y1 < 0 among
            predicted < 0), one-step gain of the rule = mean over rows of -y1 x [pred < 0] (c/sh of our order per row), the perfect
            one-step gain = mean of max(-y1, 0), for KP, G (v only) and KPS
  bins      mean y1 by KP prediction quantile (pooled; bottom 1 / 5 / 10 / 20 / 50%, top 50%)
Output: kp_s4_posthoc.json / .txt in the S4 output directory. Usage (through run_light, PF_SLOT=claude): pf_kp_s4_posthoc.py
"""
import json
import os

import numpy as np
import pandas as pd

OUT4 = '/home/yuqing/ctaNew/data/predict_fun/research/kp_s4_20261112'


def one_step(y, p):
    pull = p < 0
    return dict(pred_neg=float(pull.mean()), prec=float((y[pull] < 0).mean()) if pull.any() else float('nan'),
                gain=float(np.mean(-y * pull)), mean_y_pulled=float(y[pull].mean()) if pull.any() else float('nan'))


def main():
    O = pd.read_parquet(f'{OUT4}/oof.parquet')
    n_all = len(O); O = O[np.isfinite(O.y1.values)]
    res = dict(rows_all=int(n_all), rows_labelled=int(len(O)), per_day={}, pooled={}, bins={})
    for d, x in list(O.groupby('day')) + [('pooled', O)]:
        y = x.y1.values.astype(float)
        r = dict(rows=int(len(x)), y_neg=float((y < 0).mean()), y_zero=float((y == 0).mean()), y_pos=float((y > 0).mean()),
                 mean_y=float(y.mean()), perfect_gain=float(np.mean(np.maximum(-y, 0.0))))
        for k, c in (('KP', 'p_kp'), ('G', 'p_g'), ('KPS', 'p_kps')):
            r[k] = one_step(y, x[c].values)
        if d == 'pooled':
            res['pooled'] = r
        else:
            res['per_day'][str(d)] = r
    y = O.y1.values.astype(float); p = O.p_kp.values
    qs = np.quantile(p, [0.01, 0.05, 0.10, 0.20, 0.50])
    for lab, m in (('bottom 1%', p <= qs[0]), ('bottom 5%', p <= qs[1]), ('bottom 10%', p <= qs[2]), ('bottom 20%', p <= qs[3]),
                   ('bottom 50%', p <= qs[4]), ('top 50%', p > qs[4])):
        res['bins'][lab] = dict(rows=int(m.sum()), p_max=float(p[m].max()), mean_y=float(y[m].mean()), y_neg=float((y[m] < 0).mean()))
    L = [f'Keep-or-pull S4 post hoc: the S4 rule (pull when predicted Delta < 0) scored one step at a time on the never-pull rows '
         f'(held-out, M 50, y1 = Delta at H 5 s, 1 s markout, c/sh). Rows {res["rows_labelled"]} labelled of {n_all}.', '',
         'day       rows   y<0   y=0   y>0  mean y | perfect | KP: pred<0  prec   gain | G: pred<0  prec   gain | KPS: pred<0   gain']
    for d, r in list(res['per_day'].items()) + [('pooled', res['pooled'])]:
        L.append(f'{d:8s} {r["rows"]:7d} {r["y_neg"]:.3f} {r["y_zero"]:.3f} {r["y_pos"]:.3f} {r["mean_y"]:+.3f} | {r["perfect_gain"]:+.3f}  |'
                 f'     {r["KP"]["pred_neg"]:.3f} {r["KP"]["prec"]:.3f} {r["KP"]["gain"]:+.3f} |    {r["G"]["pred_neg"]:.3f} {r["G"]["prec"]:.3f}'
                 f' {r["G"]["gain"]:+.3f} |       {r["KPS"]["pred_neg"]:.3f} {r["KPS"]["gain"]:+.3f}')
    L += ['', 'mean y1 by KP prediction quantile (pooled):']
    for lab, b in res['bins'].items():
        L.append(f'  {lab:11s} rows {b["rows"]:7d}  pred <= {b["p_max"]:+.3f}  mean y {b["mean_y"]:+.3f}  share y<0 {b["y_neg"]:.3f}')
    open(f'{OUT4}/kp_s4_posthoc.txt', 'w').write('\n'.join(L) + '\n')
    json.dump(res, open(f'{OUT4}/kp_s4_posthoc.json', 'w'), indent=1)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
