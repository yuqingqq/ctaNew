"""READINGS of QUEUE-PRICE-STATE-AUDIT-v1 (declaration.md section READINGS; frozen with the builder before the run; session aea5bfb9, 2026-11-22).
Reads the frozen run's outputs (hash-checked against its audit.json), writes the receipt copies and the declared readings; selects and fits nothing.
Support (declared): a value is evaluable only when its own cohort has >= 20 filled states valid for it (the cells' h<h>_sparse flag, review
72a36de9 finding 2). An order needs all three queue bands evaluable; day / side consistency names its counts (days_evaluable, days_same_order)
and same_order_all_4_days is true only when all four declared days are evaluable and match. Point estimates only (4 UTC days < 5).
Usage (run_light): read_queue_price.py (a rehearsal sets QP_RUN_DIR / QP_READ_DIR elsewhere and QP_QUIET=1: no value is printed)"""
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path('/home/yuqing/ctaNew'); R = REPO / 'predictfun/reviews/ladder_queue_price_20261122'
D = Path(os.environ.get('QP_RUN_DIR', REPO / 'data/predict_fun/research/ladder_queue_price_v1_20261122'))
O = Path(os.environ.get('QP_READ_DIR', R / 'outputs')); QUIET = os.environ.get('QP_QUIET') == '1'
DAYS = ('20261024', '20261025', '20261026', '20261027'); BANDS = ('front', 'within_R', 'beyond_R'); IVS = ('I0', 'I1', 'I2', 'I3', 'union'); HK = (1000, 30000)
P = 'at_touch|shown_positive'


def need(cond, msg):
    if not cond:
        raise SystemExit(f'READINGS REFUSED: {msg}')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load():
    audit = json.loads((D / 'audit.json').read_text()); need(all(audit['outputs'][n] == sha(D / n) for n in ('states.csv', 'cells.csv')), 'an output differs from its audit.json hash')
    S = pd.read_csv(D / 'states.csv', dtype={'day': str}, float_precision='round_trip'); T = pd.read_csv(D / 'cells.csv', dtype={'day': str, 'side': str, 'band': str}, float_precision='round_trip')
    return audit, S, T


def pick(T, view, band, iv, day='all', side='all', coh='common'):
    g = T[(T.view == view) & (T.band == band) & (T.interval == iv) & (T.day == day) & (T.side == side) & (T.cohort == coh)]
    return g.iloc[0] if len(g) == 1 else None


def value(r, h):
    """(per share, evaluable) of one cell at horizon h."""
    if r is None:
        return np.nan, False
    return float(r[f'h{h}_per_share']), not bool(r[f'h{h}_sparse'])


def order(vals):
    """vals: {band: (per share, evaluable)} -> the ascending band order, or insufficient_support."""
    if not all(vals[b][1] for b in BANDS):
        return 'insufficient_support'
    return '<'.join(sorted(BANDS, key=lambda b: vals[b][0]))


def principal(T, iv, h, day='all', side='all'):
    return order({b: value(pick(T, 'price_display_queue', f'{P}|{b}', iv, day, side), h) for b in BANDS})


def pooled(T, iv, h):
    """The all-state queue order at the same support rule: the price_display_queue cells summed over price / display per queue band."""
    g = T[(T.view == 'price_display_queue') & (T.interval == iv) & (T.day == 'all') & (T.side == 'all') & (T.cohort == 'common')].copy(); g['qb'] = g.band.str.split('|').str[-1]; vals = {}
    for b in BANDS:
        x = g[g.qb == b]; q = float(x[f'h{h}_valid_shares'].sum()); vals[b] = (float(x[f'h{h}_net'].sum()) / q if q > 0 else np.nan, int(x[f'h{h}_valid_filled'].sum()) >= 20)
    return order(vals), {b: dict(per_share=vals[b][0], valid_filled=int(g[g.qb == b][f'h{h}_valid_filled'].sum())) for b in BANDS}


def consistency(T, iv, h):
    pool = principal(T, iv, h); days = [principal(T, iv, h, day=d) for d in DAYS]; sides = [principal(T, iv, h, side=s) for s in ('0', '1')]
    ev = [d for d in days if d != 'insufficient_support']; es = [s for s in sides if s != 'insufficient_support']
    return dict(pooled=pool, days=dict(zip(DAYS, days)), days_evaluable=len(ev), days_same_order=sum(d == pool for d in ev) if pool != 'insufficient_support' else 0,
                same_order_all_4_days=bool(pool != 'insufficient_support' and len(ev) == 4 and all(d == pool for d in ev)), sides=dict(zip(('0', '1'), sides)),
                sides_same_order=bool(pool != 'insufficient_support' and len(es) == 2 and all(s == pool for s in es)))


def table(T, view, bands=None, ivs=IVS):
    g = T[(T.view == view) & (T.day == 'all') & (T.side == 'all') & (T.cohort == 'common') & T.interval.isin(ivs)]
    g = g[g.band.isin(bands)] if bands is not None else g
    cols = ['band', 'interval', 'all_opportunities', 'all_at_risk', 'all_filled', 'all_shares', 'censored_from_common', 'all_t_to_end_median', 'all_t_to_end_lt30s']
    for h in (0, 1000, 5000, 30000):
        cols += [f'h{h}_valid', f'h{h}_valid_filled', f'h{h}_valid_shares', f'h{h}_any_fill_rate', f'h{h}_per_share', f'h{h}_per_opportunity', f'h{h}_sparse']
    cols += ['h1000_fill_positive', 'h1000_fill_negative']; return g[cols].reset_index(drop=True)


def main():
    audit, S, T = load(); O.mkdir(parents=True, exist_ok=True); need(not (O / 'readings.json').exists(), 'the readings exist (write-once)')
    for n in ('states.csv', 'cells.csv'):
        with open(O / f'{n}.gz', 'wb') as fh, gzip.GzipFile(fileobj=fh, mode='wb', mtime=0) as g:
            g.write((D / n).read_bytes())
        need(hashlib.sha256(gzip.decompress((O / f'{n}.gz').read_bytes())).hexdigest() == audit['outputs'][n], f'{n}: the receipt copy differs')
    (O / 'audit.json').write_bytes((D / 'audit.json').read_bytes())
    occ = {k: int(v) for k, v in S.groupby(['price_state', 'display', 'queue_band']).size().items()}; occ = {'|'.join(k): v for k, v in occ.items()}
    fr = S[S.queue_band == 'front']
    q2 = {iv: {str(h): consistency(T, iv, h) for h in HK} for iv in IVS}
    q3 = {iv: {str(h): dict(zip(('order', 'bands'), pooled(T, iv, h))) for h in HK} for iv in IVS}
    persists = {str(h): ('insufficient_support' if 'insufficient_support' in (q2['union'][str(h)]['pooled'], q3['union'][str(h)]['order'])
                         else bool(q2['union'][str(h)]['pooled'] == q3['union'][str(h)]['order'])) for h in HK}
    bd = T[(T.view == 'behind_depth_queue') & (T.band != 'outside')].copy(); bd['depth'] = bd.band.str.split('|').str[0]
    q5 = {}
    for dep in sorted(bd.depth.unique()):
        vals = {h: {b: value(pick(T, 'behind_depth_queue', f'{dep}|{b}', 'union'), h) for b in BANDS} for h in HK}
        q5[dep] = {str(h): dict(order=order(vals[h]), per_share={b: vals[h][b][0] for b in BANDS}, evaluable={b: vals[h][b][1] for b in BANDS}) for h in HK}
    pp = T[(T.view == 'principal_pressure') & (T.band != 'outside') & (T.interval == 'union') & (T.day == 'all') & (T.side == 'all') & (T.cohort == 'common')]
    census = {r.band: dict(states=int(r.all_opportunities), valid_filled_1s=int(r.h1000_valid_filled)) for r in pp.itertuples()}
    contr = 0
    for b in BANDS:
        ref = pick(T, 'principal_pressure', f'{b}|non_adverse', 'union')
        for alt in ('q5', 'positive_below_q5'):
            a = pick(T, 'principal_pressure', f'{b}|{alt}', 'union'); contr += int(value(a, 1000)[1] and value(ref, 1000)[1])
    cen = {b: dict(censored_from_common=int(r.censored_from_common), t_to_end_median=float(r.all_t_to_end_median), t_to_end_lt30s=int(r.all_t_to_end_lt30s), states=int(r.all_opportunities))
           for b in BANDS for r in [pick(T, 'price_display_queue', f'{P}|{b}', 'union')] if r is not None}
    va = S[S.venue_status == 'available']; vq = {}
    for g, x in va.groupby('price_display'):
        d = x.client_minus_venue; vq[g] = dict(n=int(len(d)), within_1=int((d.abs() < 1).sum()), below_by_1=int((d <= -1).sum()), above_by_1=int((d >= 1).sum()))
    rd = dict(asof_outputs=audit['asof'], states=int(len(S)), support_rule='valid filled states >= 20 on the value\'s own cohort',
              q1_occupancy=occ, q1_front=dict(states=int(len(fr)), **{f'{a}|{b}': int(((fr.price_state == a) & (fr.display == b)).sum()) for a in ('inside_best', 'at_touch', 'behind_best')
                                                                     for b in ('shown_zero', 'shown_positive', 'shown_unknown', 'shown_invalid')}),
              q2_principal=q2, q3_pooled=q3, q3_persists_at_touch_occupied=persists, q5_behind_depth=q5, q6_pressure_census=census, q6_contrasts_evaluable=contr,
              q7_principal_censoring=cen, q8_venue_by_price_display=vq, intervals_quoted='none (4 UTC days < 5): point estimates only')
    tabs = dict(t_principal=table(T, 'price_display_queue', [f'{P}|{b}' for b in BANDS]), t_price_display=table(T, 'price_display'),
                t_front_composition=table(T, 'price_display_queue', sorted(b for b in T.band.unique() if str(b).endswith('|front'))),
                t_behind_depth=table(T, 'behind_depth_queue', sorted(b for b in T.band.unique() if str(b).startswith('depth='))), t_pressure_census=table(T, 'principal_pressure', ivs=('union',)))
    for n, x in tabs.items():
        x.to_csv(O / f'{n}.csv', index=False)
    (O / 'readings.json').write_text(json.dumps(rd, indent=1, default=str) + '\n')
    print(f'READINGS WRITTEN: {len(tabs)} tables' if QUIET else json.dumps(rd, indent=1, default=str)[:8000])


if __name__ == '__main__':
    main()
