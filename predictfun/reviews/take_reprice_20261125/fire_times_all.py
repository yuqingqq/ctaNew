"""The top taker's complete fire list from the match records (USER 2026-11-26: "Just list the top takers fires with time"; the first list,
top_taker_fire_times.csv, held only its orders in the 332 markets with a complete Predict recording and timed legs). From exit_times' scan
(every match record with the top taker as taker; settlement-id timed): one order = one taker order hash; a fire = its orders in one market
whose fire times (settlement - 150 ms) lie within fire_times' BURST_MS of the previous order's, the earliest fire time kept; arrival at Predict
= fire - 34 ms; bet = UP when it bought UP (or sold DOWN). Markets of the four days only (a next-day hour's legs dropped).
Outputs (write-once): smart_fires/top_taker_fire_times_all.csv (fire_id, utc_time, t_ms, bet), smart_fires/fire_times_all.json (counts).
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/fire_times_all.py [<rehearsal dir>]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import fire_times as FT  # noqa: E402

WT, C, TR = FT.WT, FT.C, FT.TR
LAG, HOLD = 34., 150.


def main():
    od = Path(sys.argv[1]) if len(sys.argv) > 1 else H
    fn, fj = od / 'smart_fires/top_taker_fire_times_all.csv', od / 'smart_fires/fire_times_all.json'
    C.require(not fn.exists() and not fj.exists(), 'an output exists (write-once)')
    L = pd.concat([pd.read_parquet(WT.OUT / f'exit_legs_{d}.parquet') for d in WT.DAYS], ignore_index=True)
    n_raw = len(L)
    L = L.sort_values('recv_ns', kind='stable').drop_duplicates(['tx', 'maker_hash'], keep='first')
    reg = pd.concat([WT.MM.registry(d) for d in WT.DAYS])
    L = L[(L.role == 'taker') & L.mid.isin(set(reg.mid.astype(int)))]
    O = L.groupby('taker_hash').agg(mid=('mid', 'first'), t_set=('t_set', 'min'), side=('side', 'first'), sides=('side', 'nunique'), legs=('side', 'size'))
    C.require((O.sides == 1).all(), 'a taker order with two sides')
    O['fire'] = O.t_set - HOLD
    O = O.sort_values(['mid', 'fire'], kind='stable').reset_index()
    O['burst'] = (~(O.groupby('mid').fire.diff() <= FT.BURST_MS)).cumsum()
    B = O.groupby('burst').agg(fire=('fire', 'min'), side=('side', 'first'), sides=('side', 'nunique'), orders=('taker_hash', 'size'), mid=('mid', 'first'))
    B['t_ms'] = (B.fire - LAG).round().astype('int64')
    B = B.sort_values('t_ms', kind='stable').reset_index(drop=True)
    B['fire_id'] = np.arange(len(B))
    B['utc_time'] = pd.to_datetime(B.t_ms, unit='ms', utc=True).dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
    B['bet'] = np.where(B.side > 0, 'UP', 'DOWN')
    vs = json.loads((WT.VR.OUT / 'summary.json').read_text())
    comp = {int(r['mid']) for r in vs['rows'] if r['day'] in WT.DAYS and r['status'] == 'complete'}
    fn.parent.mkdir(parents=True, exist_ok=True)
    B[['fire_id', 'utc_time', 't_ms', 'bet']].to_csv(fn, index=False)
    out = dict(asof='2026-11-26', legs_raw=int(n_raw), taker_legs=int(len(L)), orders=int(len(O)), fires=int(len(B)), fires_with_both_sides=int((B.sides > 1).sum()),
               markets=int(B.mid.nunique()), markets_of_the_days=int(len(reg)), complete_markets=int(len(comp)), orders_in_complete_markets=int(O.mid.isin(comp).sum()),
               fires_in_complete_markets=int(B.mid.isin(comp).sum()), by_day={k: int(v) for k, v in B.utc_time.str[:10].value_counts().sort_index().items()},
               file_bytes=int(fn.stat().st_size), burst_ms=float(FT.BURST_MS))
    TR.dump(fj, out)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
