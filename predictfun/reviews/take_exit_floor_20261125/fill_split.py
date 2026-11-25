"""TAKE-EXIT FLOOR v1 diagnostic (descriptive, saved ledgers only): FLOOR round trips and maker-exit fills split by the exit order's
class at post time: `touch` (posted at the best price) or `floor` (raised above the touch to the break-even floor, i.e. resting away
from the touch, where the replay's optimistic queue rule matters most). Round trips use the same fee-carrying FIFO as account().
Output (write-once): fill_split.json.
Usage: PF_SLOT=claude predictfun/research/run_light.sh floor_fill_split -- python3 predictfun/reviews/take_exit_floor_20261125/fill_split.py
"""
import collections
import gzip
import json
import sys
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import floor_replay as F  # noqa: E402

EPS = F.EPS


def split(rec):
    cls = {int(x['gid']): ('floor' if abs(x['price'] - x['raw_best']) > EPS else 'touch') for x in rec['floor_log']}
    act = {int(o['gid']): float(o['act']) for o in rec['orders']}
    fee = F.fee_map(rec)
    lots = collections.deque()
    out = collections.Counter()
    ages = collections.defaultdict(list)
    for t, gid, s, q, p, it in rec['fills']:
        sg = 1. if s == 0 else -1.
        f = fee[int(gid)] if it < 0 else 0.
        fps = f / q
        k = 'take' if it < 0 else cls[int(gid)]
        if it >= 0:
            out[f'{k}_fill_shares'] += q
            out[f'{k}_fills'] += 1
            ages[k].append(t - act[int(gid)])
        n = q
        while n > EPS and lots and lots[0][0] != sg:
            ls, lq, lp, lf = lots[0]
            x = min(n, lq)
            out[f'{k}_rt_net_c'] += ls * x * (p - lp) - x * (lf + fps)
            out[f'{k}_rt_shares'] += x
            n -= x; lq -= x
            if lq < EPS:
                lots.popleft()
            else:
                lots[0] = (ls, lq, lp, lf)
        if n > EPS:
            lots.append((sg, n, p, fps))
    return out, ages


def main():
    fn = H / 'fill_split.json'
    assert not fn.exists(), 'fill_split.json exists (write-once)'
    sm = json.loads((F.OUT / 'summary.json').read_text())
    tot, ages = collections.Counter(), collections.defaultdict(list)
    for r in sm['rows']:
        rec = json.loads(gzip.decompress((F.OUT / f"{r['mid']}_FLOOR.json.gz").read_bytes()))
        o, a = split(rec)
        tot.update(o)
        for k, v in a.items():
            ages[k] += v
    rt_sum = sum(tot[f'{k}_rt_net_c'] for k in ('touch', 'floor', 'take')) / 100
    want = sum(r['floor']['rt_net_c'] for r in sm['rows']) / 100
    assert abs(rt_sum - want) < 1e-6, (rt_sum, want)
    res = dict(asof='2026-11-25', markets=len(sm['rows']), rt_net_usd_check=dict(split_sum=rt_sum, account_total=want),
               **{k: dict(fills=int(tot[f'{k}_fills']), fill_shares=tot[f'{k}_fill_shares'], rt_shares=tot[f'{k}_rt_shares'], rt_net_usd=tot[f'{k}_rt_net_c'] / 100,
                          rt_net_c_per_share=tot[f'{k}_rt_net_c'] / tot[f'{k}_rt_shares'] if tot[f'{k}_rt_shares'] > EPS else None,
                          fill_age_ms_p50=float(np.median(ages[k])) if ages[k] else None, fill_age_ms_p90=float(np.quantile(ages[k], .9)) if ages[k] else None)
                  for k in ('touch', 'floor', 'take')})
    F.TR.dump(fn, res)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
