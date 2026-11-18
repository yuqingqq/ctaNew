"""Reproduction predicates of scenario replay v6 (C12 age by trade-anchor source, C2 reading at M250), declared before v6 read any data
(2026-11-18, session aea5bfb9, the readiness loop's item 3). v6 adds columns and readings only; nothing v5 computed may move.
 R5a v6 vs v5 per day: every scenario's E, score, null, KEEP, PULLALL, fills, shares, eligible, fills_value identical (1e-6); by_age (E, score,
     fills_value, fills, shares per bin), no_young and cancel_pending identical (1e-6); every v5 reading string identical.
 R5b C12 partitions: in every scenario and age bin the four sources' fills sum to the bin's fills exactly and their E to the bin's E (1e-6).
 R5c control (j) ran on every committed order: the day receipt's base_matched = orders (= v5's), and Z0 CONTROLS = YES.
 R5d C2: each M250 reading is HOLDS-AT-M250 iff sign(value at M250) = sign(value at B) != 0, recomputed here from the reported values.
Usage: python3 compare.py <day> <out.txt>"""
import json
import sys

import numpy as np

D = '/home/yuqing/ctaNew/data/predict_fun/research/'
NUM = ('E', 'score', 'null', 'KEEP', 'PULLALL', 'fills', 'shares', 'eligible', 'fills_value')


def close(a, b):
    return abs(float(a) - float(b)) <= 1e-6


def main(day, path):
    a = json.load(open(D + f'keep_pull_scenarios_v5_20261118/scenarios_{day}.json')); b = json.load(open(D + f'keep_pull_scenarios_v6_20261118/scenarios_{day}.json'))
    ra, rb = (json.load(open(D + f'keep_pull_scenarios_{v}_20261118/{day}/receipt.json')) for v in ('v5', 'v6'))
    diff = []
    for nm, x in a['scenarios'].items():
        y = b['scenarios'].get(nm)
        if y is None:
            diff.append((nm, 'missing')); continue
        diff += [(nm, k) for k in NUM if not close(x[k], y[k])]
        diff += [(nm, 'by_age', lab, k) for lab in x['by_age'] for k in x['by_age'][lab] if not close(x['by_age'][lab][k], y['by_age'][lab][k])]
        diff += [(nm, 'no_young', k) for k in x['no_young'] if not close(x['no_young'][k], y['no_young'][k])]
        diff += [(nm, 'cancel_pending', k) for k in x['cancel_pending'] if not close(x['cancel_pending'][k], y['cancel_pending'][k])]
    rd = [(k, v, b['readings'].get(k)) for k, v in a['readings'].items() if b['readings'].get(k) != v]
    R5a = set(a['scenarios']) == set(b['scenarios']) and not diff and not rd
    part = [(nm, lab) for nm, y in b['scenarios'].items() for lab, z in y['by_age'].items()
            if sum(v['fills'] for v in y['by_age_src'][lab].values()) != z['fills'] or not close(sum(v['E'] for v in y['by_age_src'][lab].values()), z['E'])]
    R5b = not part and all('by_age_src' in y for y in b['scenarios'].values())
    R5c = rb['stat']['base_matched'] == rb['stat']['orders'] == ra['stat']['orders'] and b['readings']['Z0 CONTROLS'] == 'YES'
    B, M = b['scenarios']['M50|A58|L37|N100'], b['scenarios']['M250|A58|L37|N100']
    want = {'C2 E AT M250 vs M50': (B['E'], M['E']), 'C2 KEEP + SCORE AT M250 vs M50': (B['KEEP'] + B['score'], M['KEEP'] + M['score'])}
    c2 = {k: (b['readings'].get(k), 'HOLDS-AT-M250' if np.sign(v[0]) == np.sign(v[1]) and v[0] != 0 else 'SIGN-DIFFERS-AT-M250', v) for k, v in want.items()}
    R5d = all(g == w for g, w, _ in c2.values())
    out = dict(day=day, R5a=R5a, differences=diff[:20], reading_differences=rd, R5b=R5b, partition_failures=part[:20], R5c=R5c,
               orders=(ra['stat']['orders'], rb['stat']['orders'], rb['stat']['base_matched']), R5d=R5d, c2=c2)
    out['ALL'] = R5a and R5b and R5c and R5d
    txt = json.dumps(out, indent=1, default=str); open(path, 'w').write(txt + '\n'); print(txt)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
