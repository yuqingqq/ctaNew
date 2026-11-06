"""Forward check report, days 11-04..11-08 (receipt predictfun/freeze/forward_20261104_08/RECEIPT.md; run by run_forward.sh).
Declared rules (mark@stop $ = summary mtm_stop_usd, no rebate), evaluated here as predicates:
  v1 PASS iff (a) sum_5d v1 > 0 at J and at X, and (b) sum_5d (v1 - control) > 0 at J and at X        (pm_ladder_v1 receipt, unchanged)
  v2 PASS iff (a) sum_5d v2 > 0 at J and at X, and (b) sum_5d (v2 - control) > 0 at J and at X
  ROLLING ADOPTED iff v2 PASS and sum_5d (v2 - v1) > 0 at J and at X
Beside (not part of any rule): per-day values, days > 0, t interval (4 df, 2.776), settle (gross_usd), +1 s (mo1_usd), shares, fills,
windows, v2's refit window / lines / minute-0 quote share, v1's minute-0 quote share at the fixed lines.
Checks (refuse before any value is printed): every run has PF_FORWARD=20261109, the declared PF_PREDS (and v2's per-day lines file),
the declared arms; v2's refit window is the 7 admissible days before T; logs/forward/code_sha256.txt exists.
Usage: python3 -m predictfun.research.pf_forward_report
"""
import json, os, sys
import numpy as np, pandas as pd
from . import pf_io as io

DAYS = ['20261104', '20261105', '20261106', '20261107', '20261108']; FWD = '20261109'; T4 = 2.776; S4 = f'{io.OUT}/step4'
CAND = 'MICRO+G4+G2+G6+G5+G3+G1+PM90+GS+LD2+XNA+XMIN1+XV+S60/mid@AM:5:25:95'
CTRL = 'MICRO+G4+G2+G6+G5+G3+G1+V0+GS+XNA+XMIN1+XV+S60/mid@AV:5:25:0.06'
F = ('mtm_stop_usd', 'gross_usd', 'mo1_usd', 'shares', 'fills', 'windows')
LINES_V1 = json.load(open(f'{S4}/minute_lines_t60k.json'))['lines']


def ci(x):
    x = np.asarray(x, float); m = x.mean(); h = T4 * x.std(ddof=1) / np.sqrt(len(x))
    return dict(sum=round(float(x.sum()), 1), mean=round(float(m), 1), lo=round(float(m - h), 1), hi=round(float(m + h), 1), days_pos=int((x > 0).sum()))


def share_above(d, preds, line90):
    p = pd.read_parquet(f'{io.OUT}/step5/{d}/{preds}.parquet', columns=['mid', 'tau', 'v_bid', 'v_ask'])
    ws = pd.read_parquet(f'{io.OUT}/step1/{d}/registry.parquet', columns=['mid', 'ws']).set_index('mid').ws
    rel = (p.tau.values - p.mid.map(ws).values * 1000.0) / 1000.0; m = (rel >= 0) & (rel < 60)
    return {s: round(float(np.mean(x[np.isfinite(x)] > line90)), 4) for s, x in (('bid', p.v_bid.values[m]), ('ask', p.v_ask.values[m]))}


def main():
    errs = []
    if not os.path.exists(f'{io.OUT}/logs/forward/code_sha256.txt'):
        errs.append('logs/forward/code_sha256.txt missing')
    V = {}
    for tm in ('J', 'X'):
        for d in DAYS:
            for which, tag, preds, arms in (('v1', f'fw1{tm}', 'preds_t60k', (CTRL, CAND)), ('v2', f'fw2{tm}', 'preds_v2w7k', (CAND,))):
                p = f'{S4}/ladder_tokyoreal_custom_e2_{preds}_{tag}/{d}/summary.json'
                if not os.path.exists(p):
                    errs.append(f'missing {p}'); continue
                s = json.load(open(p)); env = s.get('env', {})
                if env.get('PF_FORWARD') != FWD or env.get('PF_PREDS') != preds or sorted(s['arms']) != sorted(arms):
                    errs.append(f'{tm} {d} {which}: env / arms {env.get("PF_FORWARD")} {env.get("PF_PREDS")} {sorted(s["arms"])}')
                if which == 'v2' and env.get('PF_LINES_FILE', '').format(day=d) != f'{io.OUT}/step5/{d}/lines_v2w7.json':
                    errs.append(f'{tm} {d} v2 lines file {env.get("PF_LINES_FILE")}')
                if (tm == 'J') != (env.get('PF_TXJIT') == 'frame') or (tm == 'X') != (env.get('PF_TXSHIFT') == '-50' and env.get('PF_QACT') == 'max'):
                    errs.append(f'{tm} {d} {which}: timing env')
                for a in arms:
                    V[(tm, d, 'control' if a == CTRL else which)] = {k: s['arms'][a][k] for k in F}
    for d in DAYS:
        rf = json.load(open(f'{io.OUT}/step5/{d}/v2w7/refit.json'))
        if rf['T'] != d or rf['short_window'] or len(rf['window']) != 7 or max(rf['window']) >= d:
            errs.append(f'{d} refit window {rf["window"]}')
    if errs:
        sys.exit('REFUSED: ' + '; '.join(errs))
    out = dict(rules=__doc__.split('Beside')[0], timings={}, verdict={})
    sums = {}
    for tm in ('J', 'X'):
        g = lambda w: np.array([V[(tm, d, w)]['mtm_stop_usd'] for d in DAYS])
        v1, v2, c = g('v1'), g('v2'), g('control')
        sums[tm] = dict(v1=v1.sum(), v2=v2.sum(), v1_c=(v1 - c).sum(), v2_c=(v2 - c).sum(), v2_v1=(v2 - v1).sum())
        per = {d: {w: V[(tm, d, w)] for w in ('v1', 'v2', 'control')} for d in DAYS}
        out['timings'][tm] = dict(v1=ci(v1), v2=ci(v2), control=ci(c), v1_minus_control=ci(v1 - c), v2_minus_control=ci(v2 - c), v2_minus_v1=ci(v2 - v1),
                                  per_day=per)
    v1_pass = all(sums[t]['v1'] > 0 and sums[t]['v1_c'] > 0 for t in ('J', 'X'))
    v2_pass = all(sums[t]['v2'] > 0 and sums[t]['v2_c'] > 0 for t in ('J', 'X'))
    adopt = v2_pass and all(sums[t]['v2_v1'] > 0 for t in ('J', 'X'))
    out['verdict'] = dict(v1_PASS=v1_pass, v2_PASS=v2_pass, ROLLING_ADOPTED=adopt, sums={t: {k: round(float(v), 1) for k, v in sums[t].items()} for t in sums})
    out['quote_share_minute0'] = {d: dict(v1=share_above(d, 'preds_t60k', LINES_V1['90'][0]),
                                          v2=share_above(d, 'preds_v2w7k', json.load(open(f'{io.OUT}/step5/{d}/lines_v2w7.json'))['lines']['90'][0]),
                                          v2_line90=json.load(open(f'{io.OUT}/step5/{d}/lines_v2w7.json'))['lines']['90'][0]) for d in DAYS}
    io.write_json(f'{S4}/report_forward_20261104_08.json', out)
    print(json.dumps(out['verdict'], indent=1))
    for tm in ('J', 'X'):
        T = out['timings'][tm]
        print(tm, {k: T[k] for k in ('v1', 'v2', 'control', 'v1_minus_control', 'v2_minus_control', 'v2_minus_v1')})
    print('quote share minute 0', out['quote_share_minute0'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
