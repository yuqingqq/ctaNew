"""B18 (2026-11-11, USER: "can we check on other days as well ... do we now pull orders when top makers do in the cases like imbalance
happens, or flow bursts? Or we still have those bad fills"): on the 9 replay days, how much of each one's filled volume comes in
pressure moments, and what those fills are worth: the top makers (public tape) against our strategy (the v3 replay with the live guard,
REF, and with B16's qb guard). Read-only, descriptive.

Pressure classes, from the maker's side (sg = +1 bid, -1 ask), read at the fill time from the latest 50 ms row at or before it
(rows_rl_d2_pk, the live clock; B14's engineered features on the same rows), fixed before the first run:
  thin     our side is the thin side: sg x pf_imb1 <= -1/3 (our best level at most half the other's)
  thinning our side's best level shrank >= 20% in the last 250 ms (L1 size now <= 0.8 x its size 5 rows earlier, same window)
  burst    a Binance flow burst against us: bn_act_rel >= ln 2 (trades in the last 1 s >= 2x their trailing 30 s mean) and
           sg x bn_flow250 < 0 (net aggressive flow against our side)
  bnshift  Binance's best bid / ask imbalance moved against us: sg x d_bnimb_250 <= -0.3
  any      at least one of the four
Base rate: the share of live rows (t_rel < 240 s, both sides) in each class. Fills: tape = every timed maker leg in [ws, ws + 240 s)
(B12g's loader: one time per taker order; groups by the day's shares: top 5, ranks 6-20, rest); replay = the v3 arm's fills in REF
(b14rJ / b14rX) and qb (b16bJ / b16bX). Value: c/sh at 1 s and 60 s (tape: B12g's values; replay: 100 x mo1 / q, 100 x mo60 / q) -
the replay's scale is not the tape's (its fill model is generous, B1a), so values are compared within a group, and exposure (the
share of a group's filled shares that fall in a class) across groups.
Readings DECLARED BEFORE THE FIRST RUN (9 days; J for the replay):
 R180 exposure: REF's share of filled shares in 'any' against the top 5's, per day: MORE-EXPOSED if higher on >= 7 of 9 days;
      LESS-EXPOSED if lower on >= 7 of 9; else SIMILAR.
 R181 qb against REF, the same share: QB-AVOIDS-MORE if lower on >= 7 of 9 days; QB-AVOIDS-LESS if higher on >= 7 of 9; else SAME.
 R182 REF's fills in 'any' against REF's other fills, c/sh at 60 s, per day: PRESSURE-FILLS-WORSE if lower on >= 7 of 9 days and
      pooled lower; else NOT-WORSE. The same rule for the top 5 is reported beside (descriptive).
Controls (any failure -> no table): the class rules on a planted frame (signs per side, the 250 ms shrink within a window, a
burst needs both parts); every replay and tape fill finds a row within 100 ms (else a status, counted); the reading rules on
synthetic inputs.
Limits: whole-market tape fills are not orders we could place; the replay is our simulated behaviour under an unvalidated fill model;
thresholds are fixed choices, not fitted; the 9 days were read before (exploratory).
Usage: predictfun/research/run_light.sh b18_pressure_fills -- python3 -m predictfun.research.pf_b18_pressure_fills <out_dir>
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from . import pf_io as io
from .pf_b12g_horizons import load as tape_load, groups as tape_groups

DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102', '20261103']
S4, S5 = f'{io.OUT}/step4', f'{io.OUT}/step5'
B14X = f'{io.OUT}/b14_20261111'
ARMF = 'MICRO_G4_G2_G6_G5_G3_G1_PM90_GS_LD2_CAP10_XNA_XMIN1_XV_RO60_S240__mid_szAM-5-25-95.parquet'
RUNS = {'REF J': 'preds_t60k_b14rJ', 'REF X': 'preds_t60k_b14rX', 'qb J': 'preds_qb60km_b16bJ', 'qb X': 'preds_qb60km_b16bX'}
CLS = ('thin', 'thinning', 'burst', 'bnshift', 'any')
LN2 = float(np.log(2.0))


def classes(sg, imb1, lsize_now, lsize_250, act_rel, flow250, d_bnimb):
    """Boolean arrays per class for makers on side sg (+1 bid, -1 ask); sizes are log1p of our side's best-level size."""
    with np.errstate(invalid='ignore'):
        c = dict(thin=sg * imb1 <= -1.0 / 3.0,
                 thinning=np.expm1(lsize_now) <= 0.8 * np.expm1(lsize_250),
                 burst=(act_rel >= LN2) & (sg * flow250 < 0),
                 bnshift=sg * d_bnimb <= -0.3)
    for k in list(c):
        c[k] = np.where(np.isfinite(c[k].astype(float)), c[k], False).astype(bool)
    c['any'] = c['thin'] | c['thinning'] | c['burst'] | c['bnshift']
    return c


def selftest():
    sg = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    c = classes(sg, imb1=np.array([-0.5, 0.5, 0.2, -0.2, 0.0]), lsize_now=np.log1p(np.array([7.0, 7.0, 9.0, 10.0, 10.0])),
                lsize_250=np.log1p(np.array([10.0, 10.0, 10.0, 10.0, 10.0])), act_rel=np.array([1.0, 1.0, 1.0, 0.1, 1.0]),
                flow250=np.array([-5.0, 5.0, 5.0, 5.0, -5.0]), d_bnimb=np.array([-0.4, 0.4, 0.0, -0.4, 0.0]))
    ok = (c['thin'].tolist() == [True, True, False, False, False] and c['thinning'].tolist() == [True, True, False, False, False]
          and c['burst'].tolist() == [True, True, False, False, True] and c['bnshift'].tolist() == [True, True, False, False, False]
          and c['any'].tolist() == [True, True, False, False, True])
    r = (label_more([0.5] * 9, [0.3] * 9, ('MORE', 'LESS', 'SIMILAR')) == 'MORE' and label_more([0.1] * 9, [0.3] * 9, ('MORE', 'LESS', 'SIMILAR')) == 'LESS'
         and label_more([0.5, 0.1] * 4 + [0.3], [0.3] * 9, ('MORE', 'LESS', 'SIMILAR')) == 'SIMILAR')
    return ok and r


def label_more(a, b, names):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) != 9 or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return 'n/a'
    if (a > b).sum() >= 7:
        return names[0]
    if (a < b).sum() >= 7:
        return names[1]
    return names[2]


def day_rows(d):
    """Per market: tau grid and the class inputs for both sides (rows_rl_d2_pk + B14's engineered features, same row order)."""
    r = pd.read_parquet(f'{S5}/{d}/rows_rl_d2_pk.parquet', columns=['mid', 'tau', 't_rel', 'pf_mid', 'pf_imb1', 'pf_lbs', 'pf_las', 'bn_flow250'])
    e = pd.read_parquet(f'{B14X}/ex_{d}.parquet', columns=['mid', 'tau', 'bn_act_rel', 'd_bnimb_250'])
    if not ((e.mid.values == r.mid.values).all() and (e.tau.values == r.tau.values).all()):
        sys.exit(f'REFUSED: {d} engineered features not aligned with the rows')
    r['bn_act_rel'] = e.bn_act_rel.values; r['d_bnimb_250'] = e.d_bnimb_250.values; del e
    r = r.sort_values(['mid', 'tau'], kind='stable').reset_index(drop=True)
    g = r.groupby('mid', sort=False)
    r['lbs_250'] = g.pf_lbs.shift(5); r['las_250'] = g.pf_las.shift(5)
    return r


def lookup(R, mid, t):
    """Row index per fill: the latest row of the market with tau <= t and t - tau <= 100 ms; -1 otherwise."""
    out = np.full(len(t), -1, dtype=np.int64)
    starts = R.groupby('mid', sort=False).indices
    for m, idx in pd.Series(np.arange(len(t))).groupby(mid, sort=False).indices.items():
        ri = starts.get(m)
        if ri is None:
            continue
        tau = R.tau.values[ri]; k = np.searchsorted(tau, t[idx], side='right') - 1
        ok = (k >= 0) & (t[idx] - tau[np.clip(k, 0, None)] <= 100.0)
        out[idx[ok]] = ri[np.clip(k, 0, None)][ok]
    return out


def fill_classes(R, ix, side):
    sg = np.where(side == 0, 1.0, -1.0)
    ok = ix >= 0; j = np.clip(ix, 0, None)
    ln = np.where(side == 0, R.pf_lbs.values[j], R.pf_las.values[j]); l2 = np.where(side == 0, R.lbs_250.values[j], R.las_250.values[j])
    c = classes(sg, R.pf_imb1.values[j], ln, l2, R.bn_act_rel.values[j], R.bn_flow250.values[j], R.d_bnimb_250.values[j])
    return {k: v & ok for k, v in c.items()}, ok


def summarise(q, v1, v60, c, ok):
    w = np.where(ok, q, 0.0); tot = w.sum(); out = dict(shares=float(tot))
    for k in CLS:
        s = c[k]
        ws = w[s]; wn = w[~s & ok]
        out[k] = dict(share=float(ws.sum() / tot) if tot else float('nan'),
                      v1_in=float((ws * v1[s]).sum() / ws.sum()) if ws.sum() else float('nan'),
                      v60_in=float((ws * v60[s]).sum() / ws.sum()) if ws.sum() else float('nan'),
                      v1_out=float((wn * v1[~s & ok]).sum() / wn.sum()) if wn.sum() else float('nan'),
                      v60_out=float((wn * v60[~s & ok]).sum() / wn.sum()) if wn.sum() else float('nan'))
    return out


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    if not selftest():
        sys.exit('REFUSED: selftest')
    res = dict(days={}, statuses={}, base_rate={})
    pool = {}
    for d in DAYS:
        R = day_rows(d)
        live = ((R.t_rel < 240) & R.pf_mid.notna()).values
        base = {}
        for side in (0, 1):
            cc = classes(np.full(len(R), 1.0 if side == 0 else -1.0), R.pf_imb1.values, R.pf_lbs.values if side == 0 else R.pf_las.values,
                         R.lbs_250.values if side == 0 else R.las_250.values, R.bn_act_rel.values, R.bn_flow250.values, R.d_bnimb_250.values)
            for k in CLS:
                base.setdefault(k, []).append(cc[k][live])
        res['base_rate'][d] = {k: float(np.concatenate(v).mean()) for k, v in base.items()}
        T, vol = tape_load(f'{io.OUT}/step1/{d}'); g = tape_groups(T, vol)
        ix = lookup(R, T.mid.values, T.anc.values); c, ok = fill_classes(R, ix, T.side.values)
        res['statuses'][f'{d} tape'] = dict(fills=int(len(T)), no_row=int((~ok).sum()))
        day = {}
        for nm in ('top 5', '6-20', 'rest'):
            s = g == nm
            day[nm] = summarise(T.shares.values[s], T.v1.values[s], T.v60.values[s], {k: v[s] for k, v in c.items()}, ok[s])
            pool.setdefault(nm, []).append((T.shares.values[s], T.v1.values[s], T.v60.values[s], {k: v[s] for k, v in c.items()}, ok[s]))
        reg = pd.read_parquet(f'{io.OUT}/step1/{d}/registry.parquet', columns=['mid', 'ws']).set_index('mid').ws
        for nm, run in RUNS.items():
            F = pd.read_parquet(f'{S4}/ladder_tokyoreal_custom_e2_{run}/{d}/fills/{ARMF}')
            ta = F.mid.map(reg).values * 1000.0 + F.t.values * 1000.0
            ixr = lookup(R, F.mid.values, ta); cr, okr = fill_classes(R, ixr, F.side.values)
            res['statuses'][f'{d} {nm}'] = dict(fills=int(len(F)), no_row=int((~okr).sum()))
            v1 = 100.0 * F.mo1.values / F.q.values; v60 = 100.0 * F.mo60.values / F.q.values
            day[nm] = summarise(F.q.values, v1, v60, cr, okr)
            pool.setdefault(nm, []).append((F.q.values, v1, v60, cr, okr))
        res['days'][d] = day
        print(d, {nm: round(day[nm]['any']['share'], 3) for nm in day}, flush=True)
        del R, T; io.mem_check(d)
    res['pooled'] = {}
    for nm, parts in pool.items():
        q = np.concatenate([p[0] for p in parts]); v1 = np.concatenate([p[1] for p in parts]); v60 = np.concatenate([p[2] for p in parts])
        c = {k: np.concatenate([p[3][k] for p in parts]) for k in CLS}; ok = np.concatenate([p[4] for p in parts])
        res['pooled'][nm] = summarise(q, v1, v60, c, ok)
    a_ref = [res['days'][d]['REF J']['any']['share'] for d in DAYS]; a_top = [res['days'][d]['top 5']['any']['share'] for d in DAYS]
    a_qb = [res['days'][d]['qb J']['any']['share'] for d in DAYS]
    res['R180'] = label_more(a_ref, a_top, ('MORE-EXPOSED', 'LESS-EXPOSED', 'SIMILAR'))
    res['R181'] = label_more(a_qb, a_ref, ('QB-AVOIDS-LESS', 'QB-AVOIDS-MORE', 'SAME'))
    vin = [res['days'][d]['REF J']['any']['v60_in'] for d in DAYS]; vout = [res['days'][d]['REF J']['any']['v60_out'] for d in DAYS]
    pr = res['pooled']['REF J']['any']
    res['R182'] = 'PRESSURE-FILLS-WORSE' if (sum(a < b for a, b in zip(vin, vout)) >= 7 and pr['v60_in'] < pr['v60_out']) else 'NOT-WORSE'
    tin = [res['days'][d]['top 5']['any']['v60_in'] for d in DAYS]; tout = [res['days'][d]['top 5']['any']['v60_out'] for d in DAYS]
    res['R182_top5_beside'] = dict(days_worse=int(sum(a < b for a, b in zip(tin, tout))), pooled_in=res['pooled']['top 5']['any']['v60_in'],
                                   pooled_out=res['pooled']['top 5']['any']['v60_out'])
    json.dump(res, open(f'{out}/b18.json', 'w'), indent=1, default=float)
    L = [f"B18 pressure fills. R180 {res['R180']}; R181 {res['R181']}; R182 {res['R182']}", '',
         'base rate (share of live rows, both sides), mean of days: ' + ', '.join(f"{k} {np.mean([res['base_rate'][d][k] for d in DAYS]):.3f}" for k in CLS), '',
         'pooled 9 days: share of filled shares in each class | c/sh at 1 s / 60 s in the class (out of it)']
    for nm in ('top 5', '6-20', 'rest', 'REF J', 'qb J', 'REF X', 'qb X'):
        p = res['pooled'][nm]
        L.append(f"  {nm:6s} " + ' | '.join(f"{k} {p[k]['share']:.3f} ({p[k]['v1_in']:+.2f}/{p[k]['v60_in']:+.2f} vs {p[k]['v1_out']:+.2f}/{p[k]['v60_out']:+.2f})" for k in CLS))
    L.append('')
    L.append("per day, share of filled shares in 'any': " + '; '.join(f"{d[4:]} top5 {res['days'][d]['top 5']['any']['share']:.3f} REF {res['days'][d]['REF J']['any']['share']:.3f} qb {res['days'][d]['qb J']['any']['share']:.3f}" for d in DAYS))
    L.append('statuses (fills without a row within 100 ms): ' + json.dumps({k: v['no_row'] for k, v in res['statuses'].items() if v['no_row']}))
    open(f'{out}/b18.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
