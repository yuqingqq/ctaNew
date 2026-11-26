"""Readout of public_score3's points: how often each direct factor is above its normal rate at the score's top-1% moments, at all random
moments and at the top taker's fires (USER 2026-11-26: "show me the sauce of the score"; "we should use the direct factors"). The median
x-normal values of public_score3.json sit near the e floor at most moments (most 300 ms windows hold no event), so shares above normal are
read here. Top 1% = the held-out gradient-boosted v3 score's top 1% of each day's random moments. Declared: a rate factor above normal =
its log ratio > 0; venues: any of the five venues' price gap moved >= +0.5 bp its way over 300 ms; replacement below normal = its change < 0.
Output (write-once): score3_profile.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/score3_profile.py [dir]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import public_score3 as P3  # noqa: E402

C, TR, WT = P3.C, P3.TR, P3.WT
ROWS = dict(far_refill_above_normal=('x_far_new_r300', 'gt0'), far_cancels_above_normal=('x_far_cxl_r300', 'gt0'), near_posts_above_normal=('x_near_new_r300', 'gt0'),
            counter_trades_above_normal=('x_near_trade_r300', 'gt0'), trades_its_way_above_normal=('x_far_trade_r300', 'gt0'),
            far_replacement_below_normal=('x_far_replace_dev', 'lt0'), deep_far_removals_above_normal=('z_far_rem_r1s', 'gt0'),
            deep_near_additions_above_normal=('z_near_add_r1s', 'gt0'), any_venue_ahead_05bp=('venue', 'venue'))


def main():
    fn = (Path(sys.argv[1]) if len(sys.argv) > 1 else H) / 'score3_profile.json'
    C.require(not fn.exists(), 'score3_profile.json exists (write-once)')
    cols = ['day', 'kind', 's_gbm_v3'] + sorted({c for c, k in ROWS.values() if k != 'venue'}) + [f'u_{v}_gap_d300' for v in P3.VEN3]
    Z = pd.read_parquet(WT.OUT / 'public_score3_points.parquet', columns=cols)
    R, Fz = Z[Z.kind == 'random'], Z[Z.kind == 'fire']
    top = np.zeros(len(R), bool)
    for dd, g in R.groupby('day'):
        top[(R.day == dd).to_numpy() & (R.s_gbm_v3 >= np.quantile(g.s_gbm_v3, .99)).to_numpy()] = True
    G = dict(top1=R[top], all_random=R, fires=Fz)
    out = dict(asof='2026-11-26', n={k: int(len(v)) for k, v in G.items()}, rows={})
    for nm, (c, k) in ROWS.items():
        o = {}
        for g, X in G.items():
            if k == 'venue':
                M = X[[f'u_{v}_gap_d300' for v in P3.VEN3]].to_numpy(float)
                ok = np.isfinite(M).any(axis=1)
                o[g] = float((np.nan_to_num(M, nan=-9.) >= .5).any(axis=1)[ok].mean())
            else:
                x = X[c].to_numpy(float)
                x = x[np.isfinite(x)]
                o[g] = float((x > 0).mean() if k == 'gt0' else (x < 0).mean())
        out['rows'][nm] = o
    TR.dump(fn, out)
    for nm, o in out['rows'].items():
        print(f"{nm:34s} top1 {o['top1']:.3f} | all {o['all_random']:.3f} | fires {o['fires']:.3f}")


if __name__ == '__main__':
    main()
