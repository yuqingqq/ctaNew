"""Report of the order-change causes of pm_ladder_v1 (run_causes.sh; pf_ladder_causes.py). Checks first (each refuses): every causes run's
candidate fills / win / requests parquet are byte-identical to round i (lJi / lXi; falsifier: the round-i pm arm differs) and the causes
file's attributed request count equals the run's request rows. Then per timing: requests / window by cause, lane and kind; the share of
requests whose cause is not the side's guard value; flip-backs; cancel ages. Day-clustered t intervals over the 8 read days (7 df);
11-03 beside. Output step4/report_causes_v1.json. Usage: python3 -m predictfun.research.pf_causes_report
"""
import collections, filecmp, json, sys
import numpy as np, pandas as pd
from . import pf_io as io

D8 = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']; BES = '20261103'
STEM = 'MICRO_G4_G2_G6_G5_G3_G1_PM90_GS_LD2_XNA_XMIN1_XV_S60__mid_szAM-5-25-95.parquet'
OTHER = 'MICRO_G4_G2_G6_G5_G3_G1_PM90_GS_XNA_XMIN1_XV_S60__mid_szAM-5-25-95.parquet'
S4 = f'{io.OUT}/step4'; T97 = 2.365


def ci(x):
    x = np.asarray(x, float); m = x.mean(); h = T97 * x.std(ddof=1) / np.sqrt(len(x))
    return dict(mean=round(m, 4), lo=round(m - h, 4), hi=round(m + h, 4), days_pos=int((x > 0).sum()))


def main():
    out = dict(note='frozen candidate pm_ladder_v1, round-i engine; causes per pf_ladder_causes.py docstring', checks={}, timings={})
    bad = 0
    for tm, ref in (('J', 'lJi'), ('X', 'lXi')):
        per = {}
        for d in D8 + [BES]:
            A = f'{S4}/ladder_tokyoreal_custom_e2_preds_t60k_{ref}/{d}'; B = f'{S4}/ladder_tokyoreal_custom_e2_preds_t60k_ca{tm}/{d}'
            same = all(filecmp.cmp(f'{A}/{k}/{STEM}', f'{B}/{k}/{STEM}', shallow=False) for k in ('fills', 'win', 'requests'))
            fz = any(filecmp.cmp(f'{A}/{k}/{OTHER}', f'{B}/{k}/{STEM}', shallow=False) for k in ('fills', 'win', 'requests'))
            C = json.load(open(f'{S4}/causes_v1/{tm}_{d}.json'))['counts']
            nq = len(pd.read_parquet(f'{B}/requests/{STEM}', columns=['mid']))
            attr = sum(v for k, v in C.items() if k.startswith('req|'))
            ok = same and not fz and C['requests'] == nq == attr and not any(k.endswith('|unattributed') for k in C if k.startswith('req|'))
            out['checks'][f'{tm}_{d}'] = dict(identical=same, falsifier_identical=fz, requests=nq, attributed=attr, ok=ok); bad += int(not ok)
            per[d] = C
        rows = []
        for d, C in per.items():
            w = C['windows']
            for k, v in C.items():
                p = k.split('|')
                if p[0] == 'req':
                    rows.append(dict(day=d, what='req', lane=p[1], kind=p[2], cause=p[3], n=v / w))
                elif p[0] == 'change':
                    rows.append(dict(day=d, what='change', lane=p[1], kind=p[2], cause=p[3], n=v / w))
                elif p[0] in ('flipback', 'flipback_pred', 'flipback_mixed'):
                    rows.append(dict(day=d, what=p[0], lane=p[1], kind='', cause=p[2], n=v / w))
                elif p[0] == 'cancel_age':
                    rows.append(dict(day=d, what='cancel_age', lane=p[1], kind=p[3], cause=p[2], n=v / w))
        R = pd.DataFrame(rows)
        piv = lambda q, by: q.groupby(['day'] + by).n.sum().unstack(by).fillna(0.0)
        req = R[R.what == 'req']
        tot = req.groupby('day').n.sum()
        by_cause = piv(req, ['cause']); by_lck = piv(req, ['lane', 'kind', 'cause'])
        nonpred = tot - by_cause.get('prediction', 0.0)
        share_np = nonpred / tot
        ch = R[R.what == 'change']; chc = piv(ch, ['lane', 'kind', 'cause'])
        fbs = {k: R[R.what == k].groupby('day').n.sum().reindex(tot.index).fillna(0.0) for k in ('flipback', 'flipback_pred', 'flipback_mixed')}
        fb = fbs['flipback']
        ca = R[R.what == 'cancel_age']; cag = piv(ca, ['cause', 'kind'])
        T = dict(requests_per_window=dict(read=ci(tot[D8]), d1103=round(float(tot[BES]), 2)),
                 nonprediction_share=dict(read=ci(share_np[D8]), d1103=round(float(share_np[BES]), 4)),
                 by_cause_per_window={c: dict(read=ci(by_cause[c][D8]), d1103=round(float(by_cause[c][BES]), 2), share=round(float(by_cause[c][D8].sum() / tot[D8].sum()), 4))
                                      for c in by_cause.columns},
                 by_lane_kind_cause_per_window={'|'.join(c): round(float(by_lck[c][D8].mean()), 3) for c in by_lck.columns},
                 changes_per_window={'|'.join(c): round(float(chc[c][D8].mean()), 3) for c in chc.columns},
                 flipbacks_per_window={k: dict(read=ci(x[D8]), d1103=round(float(x[BES]), 3)) for k, x in fbs.items()},
                 flipbacks_by={k: round(float(np.mean([per[d].get(k, 0) / per[d]['windows'] for d in D8])), 3)
                               for k in sorted({k for d in D8 for k in per[d] if k.startswith('flipback')})},
                 cancel_age_per_window={'|'.join(c): round(float(cag[c][D8].mean()), 3) for c in cag.columns},
                 windows_per_day={d: per[d]['windows'] for d in per}, d_events_per_window=round(float(np.mean([per[d]['d_events'] / per[d]['windows'] for d in D8])), 1),
                 d_events_with_change_share=round(float(np.mean([per[d]['d_events_with_change'] / per[d]['d_events'] for d in D8])), 4))
        out['timings'][tm] = T
        print(tm, 'requests/window', T['requests_per_window'], 'non-prediction share', T['nonprediction_share'], flush=True)
        for c, v in sorted(T['by_cause_per_window'].items(), key=lambda kv: -kv[1]['share']):
            print(f'   {c:24s} {v}', flush=True)
        print('  flipbacks/window', T['flipbacks_per_window'], T['flipbacks_by'], flush=True)
    out['all_checks_pass'] = bad == 0
    io.write_json(f'{S4}/report_causes_v1.json', out)
    print('CHECKS', 'PASS' if bad == 0 else f'FAIL {bad}', flush=True)
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
