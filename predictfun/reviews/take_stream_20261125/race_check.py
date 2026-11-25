"""Would our triggers have reached Predict before the takers who actually took the stale quotes? Diagnostic on four consumed days.
Competitor orders: take_exit_venue_20261125 lead_events (one per order; arrival at Predict A = websocket print - 150 ms; direction d; class
late = after a Binance futures tick our way, early = before any). Our triggers: triggers_<day>.parquet (stream / depl / depls), decision =
event + 1.5 ms feed + 2 ms; our arrival at Predict = decision + network (22 ms; +10 / +20 ms sensitivities). Per order and trigger set: the
earliest same-direction trigger whose arrival falls in [A - 300, A + 300] ms (the same move); we win if it arrives before A. The old grid
decision of the missed take (lead_events.decision) gives the comparator. Output (write-once): race_check.json.
Usage: PF_SLOT=claude predictfun/research/run_light.sh race_check -- python3 predictfun/reviews/take_stream_20261125/race_check.py
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import stream_replay as S  # noqa: E402

C, TR = S.C, S.TR
LEAD = S.X.DATA / 'take_exit_venue_lead_20261125' / 'lead_events.parquet'
NETS = (22., 32., 42.)
SETS = {'stream': ('stream',), 'depl': ('depl',), 'depls': ('depls',), 'stream+depls': ('stream', 'depls'), 'stream+depl': ('stream', 'depl')}
WIN = 300.


def main():
    fn = H / 'race_check.json'
    C.require(not fn.exists(), 'race_check.json exists (write-once)')
    E = pd.read_parquet(LEAD, columns=['day', 'A', 'd', 'cls', 'decision'])
    res = collections.defaultdict(lambda: collections.defaultdict(list))
    for day in sorted(E.day.unique()):
        T = pd.read_parquet(S.OUT / f'triggers_{day}.parquet')
        X = E[E.day == day]
        for set_name, kinds in SETS.items():
            sel = T[T.kind.isin(kinds)]
            for d in (1, -1):
                td = np.sort(sel[sel.dir == d].t_event.to_numpy(float) + S.FEED + S.DECIDE)
                Xd = X[X.d == d]
                for net in NETS:
                    arr = td + net
                    i0 = np.searchsorted(arr, Xd.A.values - WIN, 'left')
                    has = (i0 < len(arr)) & (arr[np.minimum(i0, len(arr) - 1)] <= Xd.A.values + WIN)
                    margin = np.where(has, Xd.A.values - arr[np.minimum(i0, len(arr) - 1)], np.nan)
                    for c, m in zip(Xd.cls.values, margin):
                        res[(set_name, net)][c].append(m)
        for net in NETS:                                                           # the old 50 ms grid decision of our missed take
            m = X.A.values - (X.decision.values + net)
            for c, v in zip(X.cls.values, m):
                res[('grid', net)][c].append(v)
    out = dict(asof='2026-11-25', orders=int(len(E)), by_class=dict(collections.Counter(E.cls)), window_ms=WIN, results={})
    for (set_name, net), byc in res.items():
        for c, v in byc.items():
            v = np.asarray(v, float); has = np.isfinite(v)
            out['results'][f'{set_name}|{int(net)}|{c}'] = dict(
                orders=int(len(v)), trigger_in_window=float(has.mean()), we_arrive_first=float(np.mean(has & (v > 0))),
                margin_ms_p25_p50_p75=[float(np.quantile(v[has], q)) for q in (.25, .5, .75)] if has.any() else None)
    TR.dump(fn, out)
    for k, x in sorted(out['results'].items()):
        print(f"{k:24s} trigger in window {x['trigger_in_window']:.0%}  we arrive first {x['we_arrive_first']:.0%}  margin p25/p50/p75 {x['margin_ms_p25_p50_p75']}")


if __name__ == '__main__':
    main()
