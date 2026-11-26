"""Which factors are present at the smart takers' fires? (USER 2026-11-26: "Can we first measure the factors on smart takers fires, check
what are the most existing critical factors"). Readout on the four consumed days; no strategy change.
Groups: TOP1 = the top taker's fires with no move its way in the previous 300 ms and their nearby moments (public_score3's points and v3
factors, read); RANK2 and RANKS3_10 = a seeded sample of 20,000 bursts of the takers ranked 2-10 (fire_times; first fire per burst; arrival
= fire - 34 ms, fire_events' offset for TOP2_10), their nearby moments = the same times shifted -20 / -10 / +10 / +20 s (same direction),
>= 1.5 s from an hour boundary in an hour with a book file, kept when no move its way in the previous 300 ms; the v3 factors computed with
public_score's and public_score3's builders unchanged (reconciled as there: the prior move and the next move).
Per group and factor (declared before the run): AUC of the factor alone for fires against their nearby moments (direction = higher or lower
at fires); prevalence = the share of fires beyond the nearby 80th (or 20th, by direction) percentile against the nearby share beyond it;
for rate factors (log ratio against the moment's own 30 s normal) the share above normal at fires and nearby; medians; the fires' rate of a
Binance move its way within 300 ms. Factors ranked by |AUC - 0.5|.
Outputs (write-once): smart_factors.json; OUT/smart_factors_points.parquet (RANK2 / RANKS3_10 points with factors).
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/smart_factors.py \
       [<rehearsal dir> <day> ...]
"""
import json
import re
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import public_score as PS  # noqa: E402
import public_score3 as P3  # noqa: E402
import timeline2 as T2  # noqa: E402

C, TR, WT = P3.C, P3.TR, P3.WT
N_SAMPLE, SHIFTS2, SEED = 20000, (-20000., -10000., 10000., 20000.), 20261126
STATE = ['b_imb1', 'b_lfar1', 'b_lnear1', 'b_spread', 'd_imb20', 'd_lfarcush', 'p_tau', 'p_pf5']
RATE = re.compile(r'_(r100|r300|r1s)$')
LABEL = dict(b_imb1='best-level imbalance its way', b_lfar1='far-side size at the best (log)', b_lnear1='near-side size at the best (log)', b_spread='spread (ticks)',
             d_imb20='depth imbalance, 20 levels', d_lfarcush='far-side size within $2 behind the best (log)', p_tau='time left in the window', p_pf5='Predict price move its way, 5 s',
             x_far_new='far-side refills vs normal', x_far_cxl='far-side cancels vs normal', x_far_trade='trades its way vs normal', x_near_new='near-side new orders vs normal',
             x_near_cxl='near-side cancels vs normal', x_near_trade='trades against it vs normal', x_far_replace='far-side replacement (300 ms)',
             x_far_replace_dev='far-side replacement vs normal', x_near_replace='near-side replacement (300 ms)', x_near_replace_dev='near-side replacement vs normal',
             x_counter_share='share of trades against it (300 ms)', x_counter_share_dev='share of trades against it vs normal', x_activity_r300='book activity vs normal (300 ms)',
             x_imb_d100='imbalance change, 100 ms', x_imb_d300='imbalance change, 300 ms', x_far_r300='far-side size change, 300 ms', x_near_r300='near-side size change, 300 ms',
             z_far_add_r1s='deep far-side additions vs normal (1 s)', z_far_rem_r1s='deep far-side removals vs normal (1 s)', z_near_add_r1s='deep near-side additions vs normal (1 s)',
             z_near_rem_r1s='deep near-side removals vs normal (1 s)', z_far_rem5_r300='far levels 2-5 removals vs normal (300 ms)', z_near_add5_r300='near levels 2-5 additions vs normal (300 ms)',
             z_far_replace_dev='deep far-side replacement vs normal', z_imb_d300='depth imbalance change, 300 ms', z_imb_d1s='depth imbalance change, 1 s',
             q_net_share5='other takers\' net flow its way, 5 s', q_fill_r1s='Predict fill rate vs normal (1 s)')


def label(f):
    if f in LABEL:
        return LABEL[f]
    m = re.match(r'x_(far_new|far_cxl|far_trade|near_new|near_cxl|near_trade)_r(100|300)$', f)
    if m:
        return f"{LABEL['x_' + m.group(1)]} ({m.group(2)} ms)"
    m = re.match(r'u_(.+)_(gap_d300|gap_d5s|its_r300|ag_r300)$', f)
    if m:
        v, k = m.group(1).replace('_', ' '), m.group(2)
        return {'gap_d300': f'{v} price moved ahead of Binance, 300 ms', 'gap_d5s': f'{v} price moved ahead of Binance, 5 s',
                'its_r300': f'{v} trades its way vs normal (300 ms)', 'ag_r300': f'{v} trades against it vs normal (300 ms)'}[k]
    return f


def points(days):
    c = float(json.loads((H / 'fire_events.json').read_text())['groups']['TOP2_10']['arrival_offset_ms'])
    A = pd.read_parquet(WT.OUT / 'fire_times_orders.parquet', columns=['day', 'rank', 't_set', 'burst', 'fire', 'd'])
    A = A[A.t_set.notna() & (A['rank'] > 1)]
    B = A.sort_values(['burst', 'fire'], kind='stable').drop_duplicates('burst', keep='first').reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    B = B.iloc[np.sort(rng.choice(len(B), min(N_SAMPLE, len(B)), replace=False))].reset_index(drop=True)
    B = B[B.day.isin(days)].reset_index(drop=True)
    B['grp'] = np.where(B['rank'] == 2, 'RANK2', 'RANKS3_10')
    B['dec'] = np.arange(len(B))
    B['a'] = B.fire - c
    parts = [pd.DataFrame(dict(dec=B.dec, day=B.day, t=B.a, d=B.d.astype(float), grp=B.grp, shift_ms=0.))]
    for s in SHIFTS2:
        parts.append(pd.DataFrame(dict(dec=B.dec, day=B.day, t=B.a + s, d=B.d.astype(float), grp=B.grp, shift_ms=s)))
    P = pd.concat(parts, ignore_index=True)
    keep = np.zeros(len(P), bool)
    for day in days:
        m = (P.day == day).to_numpy()
        keep[m] = T2.good(day, P.t.to_numpy(float)[m])
    return P[keep].reset_index(drop=True), int(len(B))


def readout(F, N, cols):
    rows = []
    for f in cols:
        a, b = F[f].to_numpy(float), N[f].to_numpy(float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 30 or len(b) < 30 or (np.r_[a, b].std() == 0):
            continue
        auc = float(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b]))
        hi = auc >= .5
        thr = np.quantile(b, .8 if hi else .2)
        pf, pn = (float((a > thr).mean()), float((b > thr).mean())) if hi else (float((a < thr).mean()), float((b < thr).mean()))
        r = dict(factor=f, label=label(f), auc=auc, strength=abs(auc - .5), direction='higher at fires' if hi else 'lower at fires',
                 fires_beyond=pf, nearby_beyond=pn, lift=pf / pn if pn > 0 else None, fires_median=float(np.median(a)), nearby_median=float(np.median(b)),
                 n_fires=int(len(a)), n_nearby=int(len(b)))
        if RATE.search(f):
            r.update(fires_above_normal=float((a > 0).mean()), nearby_above_normal=float((b > 0).mean()))
        rows.append(r)
    return sorted(rows, key=lambda r: -r['strength'])


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (2400, 2401))
    od = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    fn, fp = (od or H) / 'smart_factors.json', (od or WT.OUT) / 'smart_factors_points.parquet'
    C.require(not fn.exists() and not fp.exists(), 'an output exists (write-once)')
    days = tuple(sys.argv[2:]) if len(sys.argv) > 2 else WT.DAYS
    C.require(all(x in WT.DAYS for x in days), 'not a declared day')
    P, n_bursts = points(days)
    parts = []
    for day in days:
        Pd = P[P.day == day].reset_index(drop=True)
        if not len(Pd):
            continue
        F1 = PS.day_features(day, Pd)
        for k in STATE + ['their_prior', 'next_ms']:
            Pd[k] = F1[k]
        for k, v in P3.day_v3(day, Pd).items():
            Pd[k] = v
        parts.append(Pd)
        print(day, len(Pd), flush=True)
    Z = pd.concat(parts, ignore_index=True)
    Z = Z[~Z.their_prior.astype(bool)].reset_index(drop=True)
    Z['y'] = (Z.next_ms <= 300.).astype(int)
    Z['fire'] = Z.shift_ms == 0.
    T1 = pd.read_parquet(WT.OUT / 'public_score3_points.parquet', columns=['day', 'kind', 'y'] + P3.V3)
    T1 = T1[T1.day.isin(days)]
    G = {'TOP1': (T1[T1.kind == 'fire'], T1[T1.kind == 'nearby'])}
    for g in ('RANK2', 'RANKS3_10'):
        G[g] = (Z[Z.fire & (Z.grp == g)], Z[~Z.fire & (Z.grp == g)])
    out = dict(asof='2026-11-26', days=list(days), sampled_bursts_ranks_2_10=n_bursts, shifts_ms_ranks_2_10=list(SHIFTS2), groups={})
    for g, (Fz, N) in G.items():
        out['groups'][g] = dict(n_fires=int(len(Fz)), n_nearby=int(len(N)), fires_move300=float(Fz.y.mean()) if len(Fz) else None,
                                nearby_move300=float(N.y.mean()) if len(N) else None, factors=readout(Fz, N, P3.V3) if len(Fz) and len(N) else [])
    Z.to_parquet(fp, index=False)
    TR.dump(fn, out)
    for g, o in out['groups'].items():
        print(f"== {g}: fires {o['n_fires']}, nearby {o['n_nearby']}, move300 fires {o['fires_move300']} nearby {o['nearby_move300']}")
        for r in o['factors'][:15]:
            print(f"   {r['auc']:.3f} {r['direction'][:6]:6s} beyond {r['fires_beyond']:.2f} vs {r['nearby_beyond']:.2f}  {r['label']}")


if __name__ == '__main__':
    main()
