"""SCOPE CHECK for GROUP3_PROFILE / GROUP3_HALVES / CHASE_TAKES' top-taker diagnostics (found 2026-11-27 while answering the USER's "What's
the main dif between our top 10 percentile fires against top taker fires? Check all information we have"): its orders with exact arrival
times (fire_times_orders, the fire set) cover only the venue-complete markets, so 'chased' flags are zero elsewhere by construction and its
fires come from markets where BTC ended near its start. Consumed days; descriptive.
Readings: per day, its orders (every leg of its taker address grouped by taker order, winners_legs) and the share in fire-set markets; the
markets it trades; GROUP3_PROFILE's firing probability by its side's best price and by time left, and GROUP3_HALVES' first-half shares,
recomputed on the fire-set markets only, next to the all-market numbers.
Output (write-once): scope_fireset.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/scope_fireset.py
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

C, TR, WT = CT.C, CT.TR, CT.WT
TOP = '0x7cab1c46a5B0557008198b92697F36df855ac0AD'


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (600, 601))
    fn = H / 'scope_fireset.json'
    C.require(not fn.exists(), 'scope_fireset.json exists (write-once)')
    A = pd.read_parquet(WT.OUT / 'fire_times_orders.parquet', columns=['day', 'rank', 't_set', 'mid'])
    A = A[(A['rank'] == 1) & A.t_set.notna()]
    fm = set(zip(A.day, A.mid.astype(np.int64)))
    out = dict(asof='2026-11-27', fire_set_orders=int(len(A)), fire_set_markets=int(len(fm)), by_day={})
    for day in CT.DAYS:
        L = pd.read_parquet(WT.OUT / f'winners_legs_{day}.parquet', columns=['mid', 'taker', 'taker_hash'])
        o = L[L.taker == TOP].groupby('taker_hash').mid.first()
        out['by_day'][day] = dict(its_orders=int(len(o)), its_markets=int(o.nunique()), registry_markets=int(len(CT.MM.registry(day))),
                                  orders_in_fire_set_markets=float(np.mean([(day, int(m)) in fm for m in o])) if len(o) else None,
                                  fire_set_markets=int(sum(1 for dy, _ in fm if dy == day)))
    X, _, _ = CT.frame(CT.OUT)
    key = X.key.to_numpy()
    on = CT.first_per_move(X.mv_k.to_numpy(float) == 1, key)
    inm = np.array([(dy, int(mi)) in fm for dy, mi in zip(X.day.to_numpy()[on], X.mid.to_numpy()[on])])
    ch = np.isfinite(X.tt_chase_lag_ms.to_numpy(float))[on]
    C.require(not ch[~inm].any(), 'a chased move outside the fire-set markets: the flags are not what this check assumes')
    d, lim = X.d.to_numpy(float)[on], X.lim.to_numpy(float)[on]
    price = np.where(d > 0, lim, 100. - lim)
    tau = X.fv_tau_s.to_numpy(float)[on]
    out['moves'] = dict(all=int(len(on)), in_fire_set_markets=int(inm.sum()), chased_all=float(ch.mean()), chased_in_fire_set=float(ch[inm].mean()))
    for nm, v, edges in (('side_best_c', price, GP.EDGES['side_best_c']), ('time_left_s', tau, GP.EDGES['time_left_s'])):
        b = np.digitize(v, edges[1:-1])
        out[f'chase_rate_by_{nm}'] = [dict(lo=float(edges[i]), hi=float(edges[i + 1]), all_markets=float(ch[b == i].mean()) if (b == i).any() else None,
                                           fire_set_markets=float(ch[(b == i) & inm].mean()) if ((b == i) & inm).any() else None,
                                           moves_fire_set=int(((b == i) & inm).sum()), moves_all=int((b == i).sum())) for i in range(len(edges) - 1)]
    out['moves_first_half'] = dict(all_markets=float(np.mean(tau > 150.)), fire_set_markets=float(np.mean(tau[inm] > 150.)))
    out['side_price_median'] = dict(all_markets=float(np.median(price)), fire_set_markets=float(np.median(price[inm])))
    TR.dump(fn, out)
    print(json.dumps(out, indent=1)[:4000])


if __name__ == '__main__':
    main()
