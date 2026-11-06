"""D1 inventory mechanisms on pm_ladder_v1 (user 2026-11-06: "We test these mechanisms like skewing, reducing only after the end, etc,
run v1 with these to check the result"). Arms of run_inv.sh and their report. Engine: pf_step4_ladder_inv.py (pf_ladder_inv_build.py).
IN-SAMPLE, exploratory: the 9 round-i days (10-23..27, 10-31, 11-01..03) are consumed; v1's guard models / lines and the RL limit
(pf_inv_r60.py: 90th percentile of v1's own exposure) were fit on them. Nothing here validates a mechanism; a pick would need the
forward race (a superseding receipt before the 11-04..11-08 read).
Per arm and timing (J, X): settle $ (gross_usd: realised at settlement, what the live loss cap counts) and mark@stop $ (mtm_stop_usd;
RO arms are marked at their later stop, 120 / 240 s, so mark@stop is not like-for-like with v1 there), day sd, worst day, carry sd per
window (pooled), |net| at stop, fills, shares, +1 s c/sh, 9-day max drawdown of the settle path in start order, worst window, and the
$150 total-loss cap on the arm's own path (pf_cap_check: first refusal exact, halt on the uncapped path).
Differences vs v1 by day (settle, mark): mean, t interval (8 df, 2.306), days better; carry sd lower on how many days.
Usage: python3 -m predictfun.research.pf_inv_report arms main|ctrl     (prints PF_ARMS)
       python3 -m predictfun.research.pf_inv_report                    (report -> step4/report_inv_v1.json)
"""
import json, sys
import numpy as np, pandas as pd
from . import pf_io as io
from .pf_cap_check import first_refusal, first_halt, mdd, PER_DAY

DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102', '20261103']
CTRL_DAYS = ['20261024', '20261101']
S4 = f'{io.OUT}/step4'; T8 = 2.306
BASE = 'G4+G2+G6+G5+G3+G1+PM90+GS+LD2'; TAIL = 'XNA+XMIN1+XV'; SIZE = 'AM:5:25:95'


def arm(extra, stop='S60'):
    toks = BASE.split('+') + [x for x in extra.split('+') if x] + TAIL.split('+') + stop.split('+')
    return f"MICRO|{'+'.join(toks)}|mid|{SIZE}"


def name(spec):
    pol, rs, m, sz = spec.split('|')
    return f'{pol}+{rs}/{m}@{sz}'


def stem(spec):
    return name(spec).replace('/', '__').replace('+', '_').replace('@', '_sz').replace(':', '-')


def arms():
    L = json.load(open(f'{S4}/inv_r60.json'))['limit_usd']
    main = {'v1': arm(''), 'SK5': arm('SK5'), 'RX5': arm('RX5'), 'SK5+RX5': arm('SK5+RX5'), 'CAP25': arm('CAP25'), 'CAP10': arm('CAP10'),
            'RL': arm(f'RL{L}'), 'RO60-120': arm('', 'RO60+S120'), 'RO60-240': arm('', 'RO60+S240'), 'RO60-240+RX5': arm('RX5', 'RO60+S240'),
            'D1ALL': arm(f'SK5+RX5+RL{L}', 'RO60+S240')}
    ctrl = {'SK1000': arm('SK1000'), 'RX1000': arm('RX1000'), 'CAP1000': arm('CAP1000'), 'RL1000000': arm('RL1000000'),
            'RO300': arm('', 'RO300+S60'), 'SK5+RX5': arm('SK5+RX5')}   # last: the falsifier (engaged, must differ)
    engage = {'SK5': [('SK_shift',)], 'RX5': [('RX_exempt',)], 'SK5+RX5': [('SK_shift',), ('RX_exempt',)], 'CAP25': [('CAPN', 'LD_CAPN')],
              'CAP10': [('CAPN', 'LD_CAPN')], 'RL': [('RL', 'LD_RL')], 'RO60-120': [('RO', 'LD_RO')], 'RO60-240': [('RO', 'LD_RO')],
              'RO60-240+RX5': [('RO', 'LD_RO'), ('RX_exempt',)], 'D1ALL': [('SK_shift',), ('RX_exempt',), ('RL', 'LD_RL'), ('RO', 'LD_RO')]}
    return main, ctrl, engage, L


def run_dir(tag, d):
    return f'{S4}/ladder_tokyoreal_custom_e2_preds_t60k_{tag}/{d}'


def ci(x):
    x = np.asarray(x, float); m = x.mean(); h = T8 * x.std(ddof=1) / np.sqrt(len(x))
    return dict(mean=round(float(m), 1), lo=round(float(m - h), 1), hi=round(float(m + h), 1), days_pos=int((x > 0).sum()),
                excludes_0=bool(m - h > 0 or m + h < 0))


def main():
    MAIN, _, _, L = arms()
    chk = json.load(open(f'{S4}/inv_check.json'))
    if not chk.get('PASS'):
        sys.exit('REFUSED: step4/inv_check.json is not PASS')
    out = dict(as_of=pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%dT%H:%MZ'), days=DAYS, rl_limit_usd=L, arms={k: v for k, v in MAIN.items()}, timings={})
    for tm in ('J', 'X'):
        R = {}
        for lab, spec in MAIN.items():
            per = {}; W = []
            for d in DAYS:
                s = json.load(open(f'{run_dir("iv" + tm, d)}/summary.json'))['arms'][name(spec)]
                per[d] = s; W.append(pd.read_parquet(f'{run_dir("iv" + tm, d)}/win/{stem(spec)}.parquet'))
            W = pd.concat(W).sort_values('ws').reset_index(drop=True); p, mw = W.pnl.values, W.min_worst.values; n = len(W)
            st = np.array([per[d]['gross_usd'] for d in DAYS]); mk = np.array([per[d]['mtm_stop_usd'] for d in DAYS])
            ref = [first_refusal(p[s:s + PER_DAY], mw[s:s + PER_DAY]) for s in range(n - PER_DAY + 1)]
            hlt = [first_halt(p[s:s + PER_DAY]) for s in range(n - PER_DAY + 1)]
            sh = sum(per[d]['shares'] for d in DAYS)
            R[lab] = dict(settle=round(float(st.sum()), 1), settle_days_pos=int((st > 0).sum()), settle_day_sd=round(float(st.std(ddof=1)), 1),
                          settle_worst_day=round(float(st.min()), 1), mark=round(float(mk.sum()), 1), mark_days_pos=int((mk > 0).sum()),
                          carry_sd_window=round(float(W.carry_stop.std()), 2), settle_sd_window=round(float(W.pnl.std()), 2),
                          abs_net_stop_mean=round(float(W.net_end.abs().mean()), 1), abs_net_stop_p90=round(float(W.net_end.abs().quantile(0.9)), 1),
                          fills=int(sum(per[d]['fills'] for d in DAYS)), shares=round(float(sh), 0),
                          mo1_c_per_sh=round(100.0 * sum(per[d]['mo1_usd'] for d in DAYS) / max(sh, 1e-9), 3),
                          max_drawdown=round(mdd(p), 1), worst_window=round(float(p.min()), 1),
                          cap150_refusal_1d_pct=round(100.0 * np.mean([r is not None for r in ref]), 1),
                          cap150_halt_1d_pct=round(100.0 * np.mean([h is not None for h in hlt]), 1),
                          per_day=dict(settle=dict(zip(DAYS, st.round(1).tolist())), mark=dict(zip(DAYS, mk.round(1).tolist())),
                                       carry_sd=dict(zip(DAYS, [per[d]['carry_stop_sd'] for d in DAYS]))),
                          _st=st, _mk=mk)
        b = R['v1']
        for lab, r in R.items():
            if lab == 'v1':
                continue
            r['vs_v1'] = dict(settle=ci(r['_st'] - b['_st']), mark=ci(r['_mk'] - b['_mk']),
                              carry_sd_lower_days=int(sum(r['per_day']['carry_sd'][d] < b['per_day']['carry_sd'][d] for d in DAYS)),
                              settle_day_sd_ratio=round(r['settle_day_sd'] / b['settle_day_sd'], 3))
        for r in R.values():
            del r['_st'], r['_mk']
        out['timings'][tm] = R
    io.write_json(f'{S4}/report_inv_v1.json', out)
    for tm in ('J', 'X'):
        print(f'== {tm}  (settle / mark $ over 9 days; day sd; worst day; carry sd/window; |net| stop; max dd; cap150 refuse / halt % within 1 day)')
        for lab, r in out['timings'][tm].items():
            v = r.get('vs_v1')
            print(f"{lab:14s} settle {r['settle']:8.1f} ({r['settle_days_pos']}/9) mark {r['mark']:7.1f} daysd {r['settle_day_sd']:6.1f} worst {r['settle_worst_day']:7.1f} "
                  f"carrysd {r['carry_sd_window']:5.2f} net {r['abs_net_stop_mean']:5.1f} dd {r['max_drawdown']:7.1f} cap {r['cap150_refusal_1d_pct']:5.1f}/{r['cap150_halt_1d_pct']:5.1f}"
                  + (f" | vs v1 settle {v['settle']['mean']:+.1f} ({v['settle']['lo']:+.1f}..{v['settle']['hi']:+.1f}) mark {v['mark']['mean']:+.1f} "
                     f"({v['mark']['lo']:+.1f}..{v['mark']['hi']:+.1f}) carry-lower {v['carry_sd_lower_days']}/9" if v else ''))
    return 0


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == 'arms':
        m_, c_, _, _ = arms(); print(';'.join((m_ if sys.argv[2] == 'main' else c_).values())); sys.exit(0)
    sys.exit(main())
