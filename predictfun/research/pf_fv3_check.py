"""FV3, correct numbers on the FROZEN v3 engine (2026-11-12, session ce68ebe0). USER: "We can fix the errors and get some correct
numbers first." After the pipeline review (handoffs/20261112_pipeline_review.md): every replay "v3" so far was lane J-v3 (FINDINGS C11);
settlement values count tie windows as Up (P10); the order-path candidate (CS) was tested on J-v3 only.

Engine: pf_step4_ladder_v3.py (generated, frozen; never edited here), arm pm_ladder_v3 = pf_ladder_v3_build.ARM, run on the 9
consumed replay days only (10-23..27, 10-31, 11-01..03), with the v3 forward check's settings (run_forward_v3.sh) minus PF_FORWARD:
PF_RL_DDEC=2, PF_RL_LPLACE=40, PF_RL_LCANCEL=22, PF_RL_LPENDCANCEL=95, PF_RL_LFILL=100, PF_TIMING=e2, PF_TXTIME=ws, PF_TXKEY=tk,
PF_PREDS=preds_t60k; timing J = PF_TXJIT=frame; timing X = PF_TXSHIFT=-50 PF_QACT=max. The engine is event-driven (decisions at
every known Predict frame), fills our own orders best price first, uses frames strictly before each event and the policy's known view.
Runs (run_fv3.sh, tag ladder_tokyoreal_custom_e2_preds_t60k_<tag>):
  existing: i2J / i2X (the in-sample runs the v3 receipt checked byte-identical against the generated engine), v3 arm.
  fv3c40J / fv3c40X: control reruns of the baseline on 10-23 and 11-03 (must reproduce i2J / i2X's v3 fills exactly).
  fv3p20J / fv3p20X / fv3p10J / fv3p10X: order path PF_RL_LPLACE = 20 / 10 ms (the pending-cancel window stays 95 ms).
  fv3x250 / fv3x250p10: the 250 ms fill clock, PF_TXSHIFT=-250 PF_QACT=max, at LPLACE 40 and 10.
Corrections in the analysis: settlement of fills in tie windows (start = end price; the registry says Up) at 50 (P10): settle - sg x 50 x
q / 100. Checklist columns (FINDINGS §1): C1 order age at the fill (age_ms: < 100 / 100-500 / >= 500), without fills on orders < 100 ms
old, B1a-weighted (M 0 and M 50 precision); C12 by anchor source (timed = WS-timed vs E2); C3 cancel-pending (zombie) share; C2 the
250 ms runs; the first 60 s alone beside the full window (for comparison with J-v3).

Readings DECLARED BEFORE THE NEW RUNS (evaluated in code; day rule = >= 7 of 9 days):
 R330 YOUNG (frozen v3, i2J and i2X): the share of its $/day at 1 s from fills on orders < 100 ms old; YOUNG-DRIVEN if > 50% at both.
 R331 AT-250 (fv3x250): v3's $/day at 1 s > 0 by the day rule -> POSITIVE-AT-250, < 0 -> NEGATIVE-AT-250, else MIXED; the same for
      settlement (ties at 50) printed.
 R332 ORDER-PATH (fv3p20 and fv3p10 vs i2, at J and at X): w/o-young $/day at 1 s above the baseline by the day rule at both timings
      -> GAIN; below at both -> LOSS; else NOT-SHOWN; B1a-weighted and settlement printed beside.
 R333 PATH-AT-250 (fv3x250p10 vs fv3x250): w/o-young $/day at 1 s above by the day rule -> GAIN-AT-250 / LOSS-AT-250 / NOT-SHOWN.
Controls (any failure -> no table): the control reruns equal i2J / i2X's v3 fills row for row on both days; the tie correction touches
exactly the registry's tie windows (14 on the 9 days) and every bid fill there loses 0.5 $/sh, every ask fill gains 0.5; the age bins
add up to the totals; on non-tie windows a side-0 fill's settle has the sign of (outcome - price) (side convention check).
Usage: research/run_fv3.sh (the runs and the report); python3 predictfun/research/pf_fv3_check.py report
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

IO = '/home/yuqing/ctaNew/data/predict_fun/research'; S4 = f'{IO}/step4'; OUTF = f'{IO}/fv3_20261112'
DAYS9 = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102', '20261103']
CTRL_DAYS = ('20261023', '20261103')
ARMF = 'MICRO_G4_G2_G6_G5_G3_G1_PM90_GS_LD2_CAP10_XNA_XMIN1_XV_RO60_S240__mid_szAM-5-25-95.parquet'
TAG = lambda t: f'{S4}/ladder_tokyoreal_custom_e2_preds_t60k_{t}'
BINS = (('< 100 ms', -1e18, 100.0), ('100-500 ms', 100.0, 500.0), ('>= 500 ms', 500.0, 1e18))
B1A = {'M0': (0.08288286541654095, 0.5476125580415678, 0.46785435948727594), 'M50': (0.12337820978184168, 0.2858571490112695, 0.47665862368157524)}
RUNS = {'i2J': 'J base', 'i2X': 'X base', 'fv3p20J': 'J path 20', 'fv3p20X': 'X path 20', 'fv3p10J': 'J path 10', 'fv3p10X': 'X path 10',
        'fv3x250': 'X250 base', 'fv3x250p10': 'X250 path 10'}


def ties(day):
    r = pd.read_parquet(f'{IO}/step1/{day}/registry.parquet', columns=['mid', 'start_price', 'end_price', 'outcome'])
    return set(int(m) for m in r.mid[(r.start_price == r.end_price) & r.start_price.notna()]), r.set_index('mid').outcome.to_dict()


def load(tag, day):
    f = pd.read_parquet(f'{TAG(tag)}/{day}/fills/{ARMF}')
    tw, outc = ties(day)
    sg = np.where(f.side.values == 0, 1.0, -1.0); tie = f.mid.isin(tw).values
    f['settle_fix'] = f.settle.values - np.where(tie, sg * 50.0 * f.q.values / 100.0, 0.0); f['tie'] = tie
    f['abin'] = pd.cut(f.age_ms.values, [-1e18, 100.0, 500.0, 1e18], right=False, labels=[b[0] for b in BINS]).astype(str)
    return f, tw, outc


def summarise(f, sub=None):
    x = f if sub is None else f[sub]
    d = dict(fills=int(len(x)), q=float(x.q.sum()), mo1=float(x.mo1.sum()), settle=float(x.settle_fix.sum()), settle_raw=float(x.settle.sum()),
             zombie_q=float(x.q[x.zombie.astype(bool)].sum()), timed_q=float(x.q[x.timed.astype(bool)].sum()))
    for b in BINS:
        y = x[x.abin == b[0]]
        d[f'mo1|{b[0]}'] = float(y.mo1.sum()); d[f'q|{b[0]}'] = float(y.q.sum())
        for tm, nm in ((True, 'ws'), (False, 'e2')):
            z = y[y.timed.astype(bool) == tm]; d[f'mo1|{b[0]}|{nm}'] = float(z.mo1.sum()); d[f'q|{b[0]}|{nm}'] = float(z.q.sum())
    d['wo'] = d['mo1|100-500 ms'] + d['mo1|>= 500 ms']
    for k_, w in B1A.items():
        d[f'w_{k_}'] = sum(d[f'mo1|{b[0]}'] * p for b, p in zip(BINS, w))
    return d


def report():
    os.makedirs(OUTF, exist_ok=True); bad = {}; RES = dict(as_of=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), days=DAYS9)
    # controls: the rerun baseline equals the in-sample runs
    for tm in ('J', 'X'):
        for day in CTRL_DAYS:
            a = pd.read_parquet(f'{TAG("i2" + tm)}/{day}/fills/{ARMF}'); b = pd.read_parquet(f'{TAG("fv3c40" + tm)}/{day}/fills/{ARMF}')
            if len(a) != len(b) or not a.reset_index(drop=True).equals(b.reset_index(drop=True)):
                bad[f'control {tm} {day}'] = (len(a), len(b))
    D = {}; ntie = 0; sign_bad = 0
    for tag in RUNS:
        for day in DAYS9:
            f, tw, outc = load(tag, day)
            if tag == 'i2J':
                ntie += len(tw)
                nt = f[~f.tie & (f.side == 0)]
                yv = nt.mid.map(outc).map({'Up': 100.0, 'Down': 0.0}).values
                sign_bad += int(np.sum(np.sign(nt.settle.values) * np.sign(yv - nt.pt.values) < 0))
                tf = f[f.tie]; exp = np.where(tf.side == 0, -0.5, 0.5) * tf.q
                if not np.allclose((tf.settle_fix - tf.settle).values, exp.values):
                    bad['tie correction'] = 1
            D[(tag, day)] = dict(all=summarise(f), m1=summarise(f, f.t < 60.0))
            s = D[(tag, day)]['all']
            if abs(sum(s[f'mo1|{b[0]}'] for b in BINS) - s['mo1']) > 1e-6:
                bad[f'bins {tag} {day}'] = 1
    if ntie != 14:
        bad['tie windows'] = ntie
    if sign_bad:
        bad['side convention'] = sign_bad
    if bad:
        sys.exit(f'FV3 CONTROL FAILED -> no table: {bad}')
    L = [f'FV3 frozen pm_ladder_v3 on the 9 replay days; as of {RES["as_of"]}; controls passed (reruns reproduce i2J / i2X on {CTRL_DAYS}, '
         f'{ntie} tie windows corrected, side convention, bins)']
    dayv = lambda tag, k_, part='all': np.array([D[(tag, d)][part][k_] for d in DAYS9])
    for part, nm in (('all', 'whole window (0-240 s)'), ('m1', 'first 60 s')):
        L.append(''); L.append(f'{nm}: $/day at 1 s - all | < 100 ms | 100-500 ms | >= 500 ms | w/o young | B1a-weighted M0 / M50 | settle (ties 50; raw) | '
                               f'shares/day | E2-anchored share | pending share | days $1s > 0')
        for tag, lab in RUNS.items():
            p = {k_: dayv(tag, k_, part).sum() / 9.0 for k_ in D[(tag, DAYS9[0])][part]}
            L.append(f'  {lab:<13} {p["mo1"]:>7.2f} | {p["mo1|< 100 ms"]:>6.2f} | {p["mo1|100-500 ms"]:>6.2f} | {p["mo1|>= 500 ms"]:>6.2f} | {p["wo"]:>6.2f} | '
                     f'{p["w_M0"]:>5.2f} / {p["w_M50"]:>5.2f} | {p["settle"]:>7.2f} ({p["settle_raw"]:.2f}) | {p["q"]:>7.0f} | {1 - p["timed_q"] / max(p["q"], 1e-9):.0%} | '
                     f'{p["zombie_q"] / max(p["q"], 1e-9):.1%} | {int(np.sum(dayv(tag, "mo1", part) > 0))}')
            RES[f'{tag}|{part}'] = p
        if part == 'all':
            L.append('  young split by anchor source (whole window, base J): ' + ', '.join(
                f'{b[0]} WS {dayv("i2J", f"mo1|{b[0]}|ws").sum() / 9:.2f} / E2 {dayv("i2J", f"mo1|{b[0]}|e2").sum() / 9:.2f}' for b in BINS))
    sh = [dayv(t, 'mo1|< 100 ms').sum() / dayv(t, 'mo1').sum() for t in ('i2J', 'i2X')]
    RES['R330'] = dict(label='YOUNG-DRIVEN' if min(sh) > 0.5 else 'NOT-YOUNG-DRIVEN', share_J_X=sh)
    L.append(''); L.append(f'R330 YOUNG (frozen v3): {RES["R330"]["label"]}; share of $ at 1 s from orders < 100 ms: J {sh[0]:.0%}, X {sh[1]:.0%}')
    n_pos = int(np.sum(dayv('fv3x250', 'mo1') > 0)); n_neg = int(np.sum(dayv('fv3x250', 'mo1') < 0))
    RES['R331'] = dict(label='POSITIVE-AT-250' if n_pos >= 7 else ('NEGATIVE-AT-250' if n_neg >= 7 else 'MIXED'), days_pos=n_pos,
                       usd_day=float(dayv('fv3x250', 'mo1').mean()), settle_day=float(dayv('fv3x250', 'settle').mean()))
    L.append(f'R331 AT-250: {RES["R331"]["label"]}; days > 0 {n_pos}, $/day at 1 s {RES["R331"]["usd_day"]:.2f}, settlement {RES["R331"]["settle_day"]:.2f}')
    RES['R332'] = {}
    for p_ in ('20', '10'):
        up = [int(np.sum(dayv(f'fv3p{p_}{tm}', 'wo') > dayv(f'i2{tm}', 'wo'))) for tm in ('J', 'X')]
        dn = [int(np.sum(dayv(f'fv3p{p_}{tm}', 'wo') < dayv(f'i2{tm}', 'wo'))) for tm in ('J', 'X')]
        lab = 'GAIN' if min(up) >= 7 else ('LOSS' if min(dn) >= 7 else 'NOT-SHOWN')
        diff = [float((dayv(f'fv3p{p_}{tm}', 'wo') - dayv(f'i2{tm}', 'wo')).mean()) for tm in ('J', 'X')]
        RES['R332'][p_] = dict(label=lab, days_up=up, days_down=dn, diff_wo_J_X=diff)
        L.append(f'R332 ORDER-PATH {p_} ms vs 40: {lab}; days above at J / X {up}, below {dn}; mean diff w/o young J / X ' + ' / '.join(f'{x:+.2f}' for x in diff) + ' $/day')
    up = int(np.sum(dayv('fv3x250p10', 'wo') > dayv('fv3x250', 'wo'))); dn = int(np.sum(dayv('fv3x250p10', 'wo') < dayv('fv3x250', 'wo')))
    RES['R333'] = dict(label='GAIN-AT-250' if up >= 7 else ('LOSS-AT-250' if dn >= 7 else 'NOT-SHOWN'), days_up=up, days_down=dn)
    L.append(f'R333 PATH-AT-250 (10 vs 40 ms): {RES["R333"]["label"]}; days above {up}, below {dn}')
    L.append(''); L.append('per day $ at 1 s (whole window): ' + '; '.join(f'{RUNS[t]} ' + ' '.join(f'{x:.0f}' for x in dayv(t, 'mo1')) for t in ('i2J', 'i2X', 'fv3x250', 'fv3p10J', 'fv3p10X')))
    json.dump(RES, open(f'{OUTF}/fv3.json', 'w'), indent=1, default=float)
    open(f'{OUTF}/fv3.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'report':
        report()
    else:
        sys.exit(__doc__)
