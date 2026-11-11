"""B19a (2026-11-11, USER: "we do the analysis first, then check if the performance is better when we use a regime aware model.
Meanwhile, when pressure comes and market is in favor, the orders are resting there, so we capture those trends in favor?").
Descriptive. Three regimes per side and moment, from B18's classes and their mirror images (read at the moment from the latest 50 ms
row, live clock; thresholds as B18, fixed before the first run):
  adverse     any of B18's: our side thin (sg x pf_imb1 <= -1/3), our level thinning (-20% in 250 ms), a Binance burst against us
              (bn_act_rel >= ln 2 and sg x bn_flow250 < 0), Binance's book shift against us (sg x d_bnimb_250 <= -0.3)
  favourable  none of those, and any mirror: our side heavy (sg x pf_imb1 >= +1/3), the OTHER side's level thinning (-20% in 250 ms),
              a Binance burst our way (bn_act_rel >= ln 2 and sg x bn_flow250 > 0), Binance's book shift our way (sg x d_bnimb_250 >= +0.3)
  calm        neither (moments with both an adverse and a favourable sign count as adverse; their share is reported)
Population A (our side): the guard grid's rows (8 read days, every 500 ms of the live 0-240 s window, both sides) with
  - the live guard's out-of-fold value (B14's oof_grid base_*) and its gate (above its own per-minute 90th percentile on the grid rows,
    as B16's R160), and v3's touch-side proxy (MICRO at a 1-tick spread quotes the heavier side: sg x pf_imb1 >= 0);
  - our back-of-queue resting order's realised value from B16c's label (grid_ql2.parquet, rule M: ql 1 s, ql60 60 s, qlm combined),
    its fill rate and its value per filled share.
Population B (fills): the top makers on the public tape (B12g's loader, 0-240 s, the day's top 5 / ranks 6-20 / rest) and our v3
  replay with the live guard (REF, b14rJ), by the regime at the fill; c/sh at 1 s and 60 s (the replay's scale is not the tape's, B1a).
Readings DECLARED BEFORE THE FIRST RUN (8 read days for A; the 9 replay days for B):
 R200 where the guard quotes: CALM-FAVOURED if the gate's quote rate inside calm moments exceeds its rate inside adverse moments on
      >= 7 of 8 days; ADVERSE-FAVOURED if the reverse on >= 7 of 8; else NEITHER.
 R201 what our order at the back earns per moment (combined label, mean over all moments of the regime): CALM-PAYS if calm > adverse
      on >= 7 of 8 days and pooled; else NOT. FAVOURABLE-PAYS beside: favourable > calm on >= 7 of 8 days and pooled.
 R202 favourable fills: TOP-CAPTURES-MORE if the top 5's share of filled shares in favourable moments exceeds REF's on >= 7 of 9 days;
      WE-CAPTURE-MORE if the reverse on >= 7 of 9; else SIMILAR.
Controls (any failure -> no table): the classes on a planted frame (B18's adverse cases and their mirrors; a moment with both signs is
adverse); the label file's K1 report; grid rows find their 50 ms row exactly (same tau) or are counted; the reading rules on
synthetic inputs.
Limits: descriptive; the label is an order at the back each moment (resting continuously would move it up, which this does not
credit); tape fills are whole-market; the replay misses pressure adverse selection (B18); the 9 days were read before.
Usage: predictfun/research/run_light.sh b19a_regimes -- python3 -m predictfun.research.pf_b19a_regimes <out_dir>
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from . import pf_io as io
from . import pf_b14_pressure_guard as B
from .pf_b12g_horizons import load as tape_load, groups as tape_groups
from .pf_b18_pressure_fills import day_rows, lookup, LN2, ARMF

READ = B.READ
DAYS9 = B.DAYS
S4 = f'{io.OUT}/step4'
QL2 = f'{S4}/badfill_info_tx_tk/grid_ql2.parquet'
QL2_REPORT = f'{S4}/badfill_info_tx_tk/grid_ql2_report.json'
REG = ('calm', 'adverse', 'favourable')


def regimes(sg, imb1, own_now, own_250, oth_now, oth_250, act_rel, flow250, d_bnimb):
    """Regime per moment for makers on side sg; sizes are log1p of each side's best-level size now and 5 rows (250 ms) earlier."""
    with np.errstate(invalid='ignore'):
        burst = act_rel >= LN2
        adv = ((sg * imb1 <= -1.0 / 3.0) | (np.expm1(own_now) <= 0.8 * np.expm1(own_250)) | (burst & (sg * flow250 < 0)) | (sg * d_bnimb <= -0.3))
        fav = ((sg * imb1 >= 1.0 / 3.0) | (np.expm1(oth_now) <= 0.8 * np.expm1(oth_250)) | (burst & (sg * flow250 > 0)) | (sg * d_bnimb >= 0.3))
    adv = np.asarray(adv, bool); fav = np.asarray(fav, bool)
    r = np.where(adv, 1, np.where(fav, 2, 0))
    return r, adv & fav


def selftest():
    sg = np.array([1.0, 1.0, -1.0, 1.0, 1.0, 1.0])
    imb = np.array([-0.5, 0.5, 0.5, 0.0, 0.5, 0.0])
    own_now = np.log1p(np.array([10.0, 10.0, 10.0, 10.0, 7.0, 10.0])); own_250 = np.log1p(np.full(6, 10.0))
    oth_now = np.log1p(np.array([10.0, 10.0, 10.0, 7.0, 10.0, 10.0])); oth_250 = np.log1p(np.full(6, 10.0))
    act = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]); flow = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 5.0]); db = np.zeros(6)
    r, both = regimes(sg, imb, own_now, own_250, oth_now, oth_250, act, flow, db)
    ok = r.tolist() == [1, 2, 1, 2, 1, 2] and both.tolist() == [False, False, False, False, True, False]
    ok = ok and lab78([0.2] * 8, [0.1] * 8, ('A', 'B', 'C')) == 'A' and lab78([0.1] * 8, [0.2] * 8, ('A', 'B', 'C')) == 'B'
    ok = ok and lab78([0.2, 0.1] * 4, [0.15] * 8, ('A', 'B', 'C')) == 'C'
    return ok


def lab78(a, b, names, need=7):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return 'n/a'
    if (a > b).sum() >= need:
        return names[0]
    if (a < b).sum() >= need:
        return names[1]
    return names[2]


def row_regimes(R, ix, side):
    sg = np.where(side == 0, 1.0, -1.0); j = np.clip(ix, 0, None); ok = ix >= 0
    own_now = np.where(side == 0, R.pf_lbs.values[j], R.pf_las.values[j]); own_250 = np.where(side == 0, R.lbs_250.values[j], R.las_250.values[j])
    oth_now = np.where(side == 0, R.pf_las.values[j], R.pf_lbs.values[j]); oth_250 = np.where(side == 0, R.las_250.values[j], R.lbs_250.values[j])
    r, both = regimes(sg, R.pf_imb1.values[j], own_now, own_250, oth_now, oth_250, R.bn_act_rel.values[j], R.bn_flow250.values[j], R.d_bnimb_250.values[j])
    return np.where(ok, r, -1), both & ok, R.pf_imb1.values[j] * sg


def fill_split(q, v1, v60, reg):
    w = q.astype(float); tot = w[reg >= 0].sum(); out = dict(shares=float(tot))
    for i, nm in enumerate(REG):
        s = reg == i; ws = w[s]
        out[nm] = dict(share=float(ws.sum() / tot) if tot else float('nan'),
                       v1=float((ws * v1[s]).sum() / ws.sum()) if ws.sum() else float('nan'),
                       v60=float((ws * v60[s]).sum() / ws.sum()) if ws.sum() else float('nan'))
    return out


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    if not selftest():
        sys.exit('REFUSED: selftest')
    rq = json.load(open(QL2_REPORT))
    if rq.get('k1_equal') != rq.get('k1_rows') or rq.get('k1_rows', 0) < 1000:
        sys.exit('REFUSED: label K1')
    J = pd.read_parquet(f'{B.OUTD}/oof_grid.parquet', columns=['day', 'mid', 'tau', 't_rel', 'base_bid', 'base_ask'])
    Q = pd.read_parquet(QL2, columns=['day', 'mid', 'tau', 'ql_bid', 'ql_ask', 'ql60_bid', 'ql60_ask', 'qlm_bid', 'qlm_ask', 'qf_bid', 'qf_ask'])
    J = J.merge(Q, on=['day', 'mid', 'tau'], how='left', validate='one_to_one'); del Q
    mn = B.minute_of(J.t_rel.values); q90 = B.grid_q90(J.base_bid.values, J.base_ask.values, mn)
    res = dict(A={}, B={}, statuses={})
    A_days = {}
    for d in READ:
        R = day_rows(d); s_ = (J.day == d).values; Jd = J[s_]
        ix = lookup(R, Jd.mid.values, Jd.tau.values.astype(float))
        exact = (ix >= 0) & (R.tau.values[np.clip(ix, 0, None)] == Jd.tau.values)
        res['statuses'][d] = dict(grid_rows=int(len(Jd)), exact_row=int(exact.sum()))
        per = {}
        for side, nm in ((0, 'bid'), (1, 'ask')):
            reg, both, simb = row_regimes(R, np.where(exact, ix, -1), np.full(len(Jd), side))
            gate = B.gate_on(Jd[f'base_{nm}'].values, mn[s_], q90); micro = simb >= 0
            for k, v in dict(reg=reg, both=both, gate=gate, micro=micro, qlm=Jd[f'qlm_{nm}'].values, ql=Jd[f'ql_{nm}'].values,
                             ql60=Jd[f'ql60_{nm}'].values, qf=Jd[f'qf_{nm}'].values).items():
                per.setdefault(k, []).append(v)
        P = {k: np.concatenate(v) for k, v in per.items()}
        day = dict(both_share=float(P['both'].mean()))
        for i, nm in enumerate(REG):
            s = P['reg'] == i
            day[nm] = dict(time_share=float(s.mean()), gate_rate=float(P['gate'][s].mean()), v3_quote_rate=float((P['gate'] & P['micro'])[s].mean()),
                           qlm=float(np.nanmean(P['qlm'][s])), ql=float(np.nanmean(P['ql'][s])), ql60=float(np.nanmean(P['ql60'][s])),
                           fill_rate=float(np.mean(P['qf'][s] > 0)),
                           v_per_filled_sh=float(np.nansum(P['ql'][s] * 5.0) / max(np.nansum(P['qf'][s]), 1e-9)),
                           qlm_gated=float(np.nanmean(P['qlm'][s & P['gate']])) if (s & P['gate']).any() else float('nan'),
                           qlm_not_gated=float(np.nanmean(P['qlm'][s & ~P['gate']])) if (s & ~P['gate']).any() else float('nan'))
        day['quoted_split'] = {nm: float(((P['reg'] == i) & P['gate']).sum() / max(P['gate'].sum(), 1)) for i, nm in enumerate(REG)}
        A_days[d] = day; A_days[d]['_P'] = P
        print('A', d, {nm: (round(day[nm]['time_share'], 3), round(day[nm]['gate_rate'], 3), round(day[nm]['qlm'], 3)) for nm in REG}, flush=True)
        del R; io.mem_check(f'A {d}')
    PA = {k: np.concatenate([A_days[d]['_P'][k] for d in READ]) for k in A_days[READ[0]]['_P']}
    pooled = {}
    for i, nm in enumerate(REG):
        s = PA['reg'] == i
        pooled[nm] = dict(time_share=float(s.mean()), gate_rate=float(PA['gate'][s].mean()), v3_quote_rate=float((PA['gate'] & PA['micro'])[s].mean()),
                          qlm=float(np.nanmean(PA['qlm'][s])), ql=float(np.nanmean(PA['ql'][s])), ql60=float(np.nanmean(PA['ql60'][s])),
                          fill_rate=float(np.mean(PA['qf'][s] > 0)), v_per_filled_sh=float(np.nansum(PA['ql'][s] * 5.0) / max(np.nansum(PA['qf'][s]), 1e-9)),
                          qlm_gated=float(np.nanmean(PA['qlm'][s & PA['gate']])), qlm_not_gated=float(np.nanmean(PA['qlm'][s & ~PA['gate']])))
    pooled['quoted_split'] = {nm: float(((PA['reg'] == i) & PA['gate']).sum() / max(PA['gate'].sum(), 1)) for i, nm in enumerate(REG)}
    pooled['both_share'] = float(PA['both'].mean())
    for d in READ:
        A_days[d].pop('_P')
    res['A'] = dict(days=A_days, pooled=pooled)
    res['R200'] = lab78([A_days[d]['calm']['gate_rate'] for d in READ], [A_days[d]['adverse']['gate_rate'] for d in READ], ('CALM-FAVOURED', 'ADVERSE-FAVOURED', 'NEITHER'))
    calm_v = [A_days[d]['calm']['qlm'] for d in READ]; adv_v = [A_days[d]['adverse']['qlm'] for d in READ]; fav_v = [A_days[d]['favourable']['qlm'] for d in READ]
    res['R201'] = 'CALM-PAYS' if (sum(a > b for a, b in zip(calm_v, adv_v)) >= 7 and pooled['calm']['qlm'] > pooled['adverse']['qlm']) else 'NOT'
    res['R201_favourable'] = 'FAVOURABLE-PAYS' if (sum(a > b for a, b in zip(fav_v, calm_v)) >= 7 and pooled['favourable']['qlm'] > pooled['calm']['qlm']) else 'NOT'
    # B: fills by regime, tape (top 5 / 6-20 / rest) and our replay (REF J), 9 days
    Bd = {}; poolB = {}
    for d in DAYS9:
        R = day_rows(d)
        T, vol = tape_load(f'{io.OUT}/step1/{d}'); g = tape_groups(T, vol)
        ix = lookup(R, T.mid.values, T.anc.values); reg, _, _ = row_regimes(R, ix, T.side.values)
        Bd[d] = {}
        for nm in ('top 5', '6-20', 'rest'):
            s = g == nm
            Bd[d][nm] = fill_split(T.shares.values[s], T.v1.values[s], T.v60.values[s], reg[s])
            poolB.setdefault(nm, []).append((T.shares.values[s], T.v1.values[s], T.v60.values[s], reg[s]))
        regs = pd.read_parquet(f'{io.OUT}/step1/{d}/registry.parquet', columns=['mid', 'ws']).set_index('mid').ws
        F = pd.read_parquet(f'{S4}/ladder_tokyoreal_custom_e2_preds_t60k_b14rJ/{d}/fills/{ARMF}')
        ta = F.mid.map(regs).values * 1000.0 + F.t.values * 1000.0
        ixr = lookup(R, F.mid.values, ta); regr, _, _ = row_regimes(R, ixr, F.side.values)
        v1 = 100.0 * F.mo1.values / F.q.values; v60 = 100.0 * F.mo60.values / F.q.values
        Bd[d]['REF J'] = fill_split(F.q.values, v1, v60, regr); poolB.setdefault('REF J', []).append((F.q.values, v1, v60, regr))
        print('B', d, {nm: round(Bd[d][nm]['favourable']['share'], 3) for nm in Bd[d]}, flush=True)
        del R, T; io.mem_check(f'B {d}')
    res['B'] = dict(days=Bd, pooled={nm: fill_split(*[np.concatenate([p[i] for p in parts]) for i in range(4)]) for nm, parts in poolB.items()})
    res['R202'] = lab78([Bd[d]['top 5']['favourable']['share'] for d in DAYS9], [Bd[d]['REF J']['favourable']['share'] for d in DAYS9],
                        ('TOP-CAPTURES-MORE', 'WE-CAPTURE-MORE', 'SIMILAR'), need=7)
    json.dump(res, open(f'{out}/b19a.json', 'w'), indent=1, default=float)
    p = res['A']['pooled']
    L = [f"B19a regimes. R200 {res['R200']}; R201 {res['R201']} ({res['R201_favourable']}); R202 {res['R202']}", '',
         'A. our side, 8 read days pooled (grid moments, both sides):',
         '  regime       time   gate rate  v3 quote (gate+side)  order at back: combined / 1 s / 60 s per moment | fill rate | c/sh per filled share (1 s) | gated / not gated (combined)']
    for nm in REG:
        x = p[nm]
        L.append(f"  {nm:11s} {x['time_share']:.3f}  {x['gate_rate']:.3f}      {x['v3_quote_rate']:.3f}                {x['qlm']:+.3f} / {x['ql']:+.3f} / {x['ql60']:+.3f}"
                 f"   | {x['fill_rate']:.3f} | {x['v_per_filled_sh']:+.2f} | {x['qlm_gated']:+.3f} / {x['qlm_not_gated']:+.3f}")
    L.append(f"  quoted moments by regime: " + ', '.join(f"{k} {v:.3f}" for k, v in p['quoted_split'].items()) + f"; moments with both signs {p['both_share']:.3f}")
    L.append(''); L.append('B. fills by regime, 9 days pooled: share of filled shares | c/sh 1 s / 60 s')
    for nm, x in res['B']['pooled'].items():
        L.append(f"  {nm:6s} " + ' | '.join(f"{r} {x[r]['share']:.3f} ({x[r]['v1']:+.2f} / {x[r]['v60']:+.2f})" for r in REG))
    L.append('per day, favourable share top 5 / REF: ' + '; '.join(f"{d[4:]} {Bd[d]['top 5']['favourable']['share']:.3f} / {Bd[d]['REF J']['favourable']['share']:.3f}" for d in DAYS9))
    open(f'{out}/b19a.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
