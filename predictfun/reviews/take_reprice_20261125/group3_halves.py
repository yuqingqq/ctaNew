"""Which half of the window: the top taker's fires, every move and CHASE_TAKES' model top 10 % by time left (USER 2026-11-27: "it looks our
top 10% picks look similar to top takers fires in the chart, and they mostly fire in former half of the window?"). POST HOC: the split at
150 s left was suggested after seeing GROUP3_PROFILE's chart; four consumed days; descriptive. Reads GROUP3_PROFILE's labels and CHASE_TAKES'
rows and predictions (its selections reproduced: a check).
Readings: the share of each set with > 150 s / > 240 s / < 60 s left; the share of moves it chases in each half; CHASE_TAKES' model top-10 %
actions per half: value per intended share on both books at the known best + 0 / 3 / 10c (1 s mid), fill, by day.
Output (write-once): group3_halves.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/group3_halves.py
"""
import json
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import chase_takes as CT  # noqa: E402
import group3_profile as GP  # noqa: E402

C, TR, TK = CT.C, CT.TR, CT.TK
HALF = 150.


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (600, 601))
    fn = H / 'group3_halves.json'
    C.require(not fn.exists(), 'group3_halves.json exists (write-once)')
    G = pd.concat([pd.read_parquet(GP.OUT / f'{d}.parquet') for d in CT.DAYS], ignore_index=True)
    G = G[G.status == 'ok']
    X, _, _ = CT.frame(CT.OUT)
    X = X.merge(pd.read_parquet(CT.OUT / 'predictions.parquet', columns=['day', 't', 'd', 'pred_1s']), on=['day', 't', 'd'], how='left', validate='1:1')
    d, lim, md, day = X.d.to_numpy(float), X.lim.to_numpy(float), X.md.to_numpy(float), X.day.to_numpy()
    e1 = X.pred_1s.to_numpy(float) - (d * (lim - md) + CT.fee(lim))
    key = X.key.to_numpy()
    on = CT.first_per_move(X.mv_k.to_numpy(float) == 1, key)
    t10 = CT.first_per_move(e1 >= np.quantile(e1, .9), key)
    CJ = json.loads((H / 'chase_takes.json').read_text())
    C.require(len(on) == CJ['selections']['every_move_onset']['actions'] and len(t10) == CJ['selections']['model1s_top10']['actions'], 'CHASE_TAKES\' selections not reproduced')
    tau = X.fv_tau_s.to_numpy(float)
    shares = lambda t: dict(n=int(len(t)), first_half=float(np.mean(t > HALF)), first_minute=float(np.mean(t > 240.)), last_minute=float(np.mean(t < 60.)))
    out = dict(asof='2026-11-27', post_hoc=True, half_s=HALF, sets={f'its_group{g}': shares(G[G.group == g].fv_tau_s.to_numpy(float)) for g in (3, 2, 1)})
    out['sets']['every_move_onset'] = shares(tau[on])
    out['sets']['our_model_top10'] = shares(tau[t10])
    ch = np.isfinite(X.tt_chase_lag_ms.to_numpy(float))
    fh = tau[on] > HALF
    out['chase_rate'] = dict(first_half=float(ch[on][fh].mean()), second_half=float(ch[on][~fh].mean()), moves_first_half=int(fh.sum()), moves_second_half=int((~fh).sum()))
    PA = X[[f'pA_{k}' for k in range(5)]].to_numpy(float); SA = X[[f'sA_{k}' for k in range(5)]].to_numpy(float)
    PB = X[[f'pB_{k}' for k in range(5)]].to_numpy(float); SB = X[[f'sB_{k}' for k in range(5)]].to_numpy(float)
    okB = np.isfinite(X.dtB.to_numpy(float))
    BK = {'A': (PA, SA), 'AB': TK.minbook(PA, SA, PB, SB)}
    mk = X.m1s.to_numpy(float)
    out['our_top10_by_half'] = {}
    for nm, m in (('first_half', tau[t10] > HALF), ('second_half', tau[t10] <= HALF)):
        o = {}
        for b in ('A', 'AB'):
            ii = t10[m]
            ii = ii[okB[ii]] if b == 'AB' else ii
            for s in (0., 3., 10.):
                fl, co, fe = TK.vwalk(BK[b][0][ii], BK[b][1][ii], d[ii], lim[ii] + s * d[ii])
                v = (d[ii] * (mk[ii] * fl - co) - fe) / CT.Q
                by = {dh: float(v[day[ii] == dh].mean()) for dh in CT.DAYS if (day[ii] == dh).any()}
                o[f'{b}_s{int(s)}'] = dict(actions=int(len(ii)), value_c=float(v.mean()), fill=float(fl.sum() / (CT.Q * len(ii))), by_day=by,
                                           days_positive=int(sum(x > 0 for x in by.values())))
        out['our_top10_by_half'][nm] = o
    TR.dump(fn, out)
    print(json.dumps(out, indent=1)[:3000])


if __name__ == '__main__':
    main()
