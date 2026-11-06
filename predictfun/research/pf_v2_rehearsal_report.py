"""pm_ladder_v2 rehearsal on the week-2 read days (in-sample, point estimates; 4 days < 5: no interval). Per day and timing: candidate
v2 (walk-forward refit, pf_v2_refit + pf_step4_ladder_v2) beside v1 (round-i candidate) and the control (round-i today), mark@stop
(mtm_stop_usd), +1 s (mo1_usd), settle (gross_usd), shares; v2's window, thresholds, lines and minute-0 quote share (refit.json).
Not a like-for-like race: v1's values on 10-31..11-02 come from leave-one-day-out models trained on the other 7 read days (later days
included) and v1's lines are cut on all 8 days (the day itself included); v2 sees only earlier days.
Checks (refuse): each v2 run used preds_v2w7k and the per-day lines file of its own day; the refit window ends before the day.
Output step4/report_v2_rehearsal.json. Usage: python3 -m predictfun.research.pf_v2_rehearsal_report
"""
import json, sys
import numpy as np
from . import pf_io as io

DAYS = ['20261031', '20261101', '20261102', '20261103']; S4 = f'{io.OUT}/step4'
CAND = 'MICRO+G4+G2+G6+G5+G3+G1+PM90+GS+LD2+XNA+XMIN1+XV+S60/mid@AM:5:25:95'
CTRL = 'MICRO+G4+G2+G6+G5+G3+G1+V0+GS+XNA+XMIN1+XV+S60/mid@AV:5:25:0.06'
F = ('mtm_stop_usd', 'mo1_usd', 'gross_usd', 'shares', 'fills')


def arm(path, name):
    a = json.load(open(f'{path}/summary.json'))['arms'][name]
    return {k: a[k] for k in F}


def main():
    out = dict(note=__doc__.split('\n')[0], timings={}); bad = 0
    for tm, ref in (('J', 'lJi'), ('X', 'lXi')):
        rows = {}
        for d in DAYS:
            p2 = f'{S4}/ladder_tokyoreal_custom_e2_preds_v2w7k_v2r{tm}/{d}'; s2 = json.load(open(f'{p2}/summary.json'))
            env = s2.get('env', {}); rf = json.load(open(f'{io.OUT}/step5/{d}/v2w7/refit.json'))
            ok = env.get('PF_PREDS') == 'preds_v2w7k' and env.get('PF_LINES_FILE', '').format(day=d).endswith(f'step5/{d}/lines_v2w7.json') \
                and max(rf['window']) < d and rf['T'] == d
            bad += int(not ok)
            v1 = arm(f'{S4}/ladder_tokyoreal_custom_e2_preds_t60k_{ref}/{d}', CAND); c = arm(f'{S4}/ladder_tokyoreal_custom_e2_preds_t60k_{ref}/{d}', CTRL)
            v2 = arm(p2, CAND)
            rows[d] = dict(v2=v2, v1=v1, control=c, v2_minus_v1=round(v2['mtm_stop_usd'] - v1['mtm_stop_usd'], 1),
                           window=rf['window'], short_window=rf['short_window'], lines90_m0=rf['lines']['90'][0], lines95_m0=rf['lines']['95'][0],
                           thresholds=rf['thresholds'], minute0_share=rf['minute0_share'], checks_ok=ok)
        tot = {k: round(float(np.sum([rows[d][k]['mtm_stop_usd'] for d in DAYS])), 1) for k in ('v2', 'v1', 'control')}
        full = [d for d in DAYS if not rows[d]['short_window']]
        out['timings'][tm] = dict(per_day=rows, sum_4d=tot, sum_full_window_days={k: round(float(np.sum([rows[d][k]['mtm_stop_usd'] for d in full])), 1)
                                                                                  for k in ('v2', 'v1', 'control')}, full_window_days=full,
                                  v2_beats_v1_days=int(sum(rows[d]['v2_minus_v1'] > 0 for d in DAYS)))
        print(tm, 'sum 4 days', tot, 'full-window days', full, out['timings'][tm]['sum_full_window_days'], flush=True)
        for d in DAYS:
            r = rows[d]
            print(f"  {d} v2 {r['v2']['mtm_stop_usd']:+.1f} v1 {r['v1']['mtm_stop_usd']:+.1f} ctrl {r['control']['mtm_stop_usd']:+.1f} "
                  f"v2-v1 {r['v2_minus_v1']:+.1f} | v2 shares {r['v2']['shares']:.0f} v1 {r['v1']['shares']:.0f} | line90 {r['lines90_m0']} "
                  f"share>90 {r['minute0_share']['bid']['above_90']:.3f}/{r['minute0_share']['ask']['above_90']:.3f} win {len(r['window'])}d", flush=True)
    out['checks_pass'] = bad == 0
    io.write_json(f'{S4}/report_v2_rehearsal.json', out)
    print('CHECKS', 'PASS' if bad == 0 else f'FAIL {bad}')
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
