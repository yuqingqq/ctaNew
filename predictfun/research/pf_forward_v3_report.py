"""Forward check report of pm_ladder_v3, days 11-09..11-13 (receipt predictfun/freeze/pm_ladder_v3/RECEIPT.md; run by run_forward_v3.sh).
Declared rules (settle $ = summary gross_usd: every arm held to settlement, no rebate; sums over the 5 days), evaluated here as predicates:
  v3 PASS       iff (a) sum v3 > 0 at J and at X, and (b) sum (v3 - control) > 0 at J and at X
  v3 PREFERRED  iff v3 PASS and (c) sum (v3 - v1) > 0 at J and at X, and (d) day sd of v3 < day sd of v1 at J and at X
Beside (not part of any rule): per-day values; days > 0; t interval (4 df, 2.776); +1 s $ (mo1_usd, like-for-like trading profit);
mark@stop (v3 at 240 s, v1 / control at 60 s: not like-for-like); worst day; max drawdown over the 5 days' windows; $150 cap halt
within 1 day (pf_cap_check); |net| at stop; fills; shares.
Checks (refuse before any value is printed): every run has PF_FORWARD=20261114, PF_PREDS=preds_t60k, the declared arms and timing env,
no PF_R60_FILE on v3 runs; logs/forward_v3/code_sha256.txt exists and lists the v3 engine at params.json's hash.
Controls (run first, refuse on failure): the rule predicates on synthetic days (one PASS + PREFERRED case, one case failing each of a-d).
rehearse: the same code on 5 READ days (v1 / control from round i lJi / lXi, v3 from i2J / i2X; IN-SAMPLE, not a check of anything but
the code path) -> step4/report_forward_v3_rehearsal.json; its falsifier: the forward env checks applied to these in-sample runs must refuse.
Usage: python3 -m predictfun.research.pf_forward_v3_report [rehearse]
"""
import json, os, sys
import numpy as np, pandas as pd
from . import pf_io as io
from .pf_inv_report import name, stem
from .pf_cap_check import first_halt, mdd, PER_DAY
from .pf_ladder_v3_build import ARM, V1, CTRL, OUT as PARAMS

FDAYS = ['20261109', '20261110', '20261111', '20261112', '20261113']; FWD = '20261114'
RDAYS = ['20261027', '20261031', '20261101', '20261102', '20261103']
T4 = 2.776; S4 = f'{io.OUT}/step4'
WHO = {'v3': ARM, 'v1': V1, 'control': CTRL}


def run_dir(tag, d):
    return f'{S4}/ladder_tokyoreal_custom_e2_preds_t60k_{tag}/{d}'


def tags(mode, tm):
    """(tag, arms that run must hold exactly, expected PF_FORWARD) per arm."""
    if mode == 'forward':
        return {'v1': (f'f3a{tm}', (CTRL, V1), FWD), 'control': (f'f3a{tm}', (CTRL, V1), FWD), 'v3': (f'f3b{tm}', (ARM,), FWD)}
    return {'v1': (f'l{tm}i', None, None), 'control': (f'l{tm}i', None, None), 'v3': (f'i2{tm}', None, None)}


def load(mode, days, expect=None):
    expect = expect or mode; errs = []; V = {}; W = {}
    if expect == 'forward':
        cs = f'{io.OUT}/logs/forward_v3/code_sha256.txt'
        eng = json.load(open(PARAMS))['engine']['sha256']
        if not os.path.exists(cs) or not any(l.startswith(eng) and l.rstrip().endswith('pf_step4_ladder_v3.py') for l in open(cs)):
            errs.append('logs/forward_v3/code_sha256.txt missing or without the v3 engine at the params.json hash')
    for tm in ('J', 'X'):
        T = tags(mode, tm); TE = tags(expect, tm)
        for who, (tag, _, _) in T.items():
            _, arms, fwd = TE[who]; ws = []
            for d in days:
                p = f'{run_dir(tag, d)}/summary.json'
                if not os.path.exists(p):
                    errs.append(f'missing {tm} {d} {who}'); continue
                s = json.load(open(p)); env = s.get('env', {})
                if env.get('PF_FORWARD') != fwd or env.get('PF_PREDS') != 'preds_t60k':
                    errs.append(f'{tm} {d} {who}: env PF_FORWARD {env.get("PF_FORWARD")} PF_PREDS {env.get("PF_PREDS")}')
                if arms is not None and sorted(s['arms']) != sorted(name(a) for a in arms):
                    errs.append(f'{tm} {d} {who}: arms {sorted(s["arms"])}')
                if name(WHO[who]) not in s['arms']:
                    errs.append(f'{tm} {d} {who}: arm absent'); continue
                if who == 'v3' and expect == 'forward' and 'PF_R60_FILE' in env:
                    errs.append(f'{tm} {d} v3: PF_R60_FILE set')
                if (tm == 'J') != (env.get('PF_TXJIT') == 'frame') or (tm == 'X') != (env.get('PF_TXSHIFT') == '-50' and env.get('PF_QACT') == 'max'):
                    errs.append(f'{tm} {d} {who}: timing env')
                V[(tm, d, who)] = s['arms'][name(WHO[who])]
                ws.append(pd.read_parquet(f'{run_dir(tag, d)}/win/{stem(WHO[who])}.parquet', columns=['ws', 'pnl', 'net_end']))
            if ws:
                W[(tm, who)] = pd.concat(ws).sort_values('ws').reset_index(drop=True)
    return V, W, errs


def verdict(S):
    """S[tm][who] = per-day settle arrays. The declared predicates."""
    pas = all(S[t]['v3'].sum() > 0 and (S[t]['v3'] - S[t]['control']).sum() > 0 for t in ('J', 'X'))
    pref = pas and all((S[t]['v3'] - S[t]['v1']).sum() > 0 and S[t]['v3'].std(ddof=1) < S[t]['v1'].std(ddof=1) for t in ('J', 'X'))
    return dict(v3_PASS=bool(pas), v3_PREFERRED=bool(pref))


def controls():
    base = {'v3': np.array([30., 10., -5., 20., 15.]), 'v1': np.array([60., -40., -30., 50., 10.]), 'control': np.array([5., -10., -20., 0., 5.])}
    mk = lambda **kw: {'J': dict(base), 'X': dict(base, **kw)}
    cases = [(mk(), dict(v3_PASS=True, v3_PREFERRED=True)),
             (mk(v3=np.array([-30., 10., -5., -20., 15.])), dict(v3_PASS=False, v3_PREFERRED=False)),                       # (a)
             (mk(control=np.array([50., 20., 0., 10., 20.])), dict(v3_PASS=False, v3_PREFERRED=False)),                     # (b)
             (mk(v1=np.array([30., 10., -5., 20., 16.])), dict(v3_PASS=True, v3_PREFERRED=False)),                         # (c)
             (mk(v1=np.array([10., 10., 10., 10., 20.])), dict(v3_PASS=True, v3_PREFERRED=False))]                         # (d)
    for i, (S, want) in enumerate(cases):
        got = verdict(S)
        if got != want:
            sys.exit(f'CONTROL {i} FAILED: {got} != {want}')
    return len(cases)


def ci(x):
    x = np.asarray(x, float); m = x.mean(); h = T4 * x.std(ddof=1) / np.sqrt(len(x))
    return dict(sum=round(float(x.sum()), 1), mean=round(float(m), 1), lo=round(float(m - h), 1), hi=round(float(m + h), 1),
                days_pos=int((x > 0).sum()), day_sd=round(float(x.std(ddof=1)), 1), worst_day=round(float(x.min()), 1))


def main(mode):
    nc = controls()
    days = FDAYS if mode == 'forward' else RDAYS
    V, W, errs = load(mode, days)
    if errs:
        sys.exit('REFUSED: ' + '; '.join(errs))
    out = dict(mode=mode, in_sample=mode != 'forward', days=days, rules=__doc__.split('Beside')[0], controls_passed=nc, timings={})
    if mode != 'forward':
        _, _, ferrs = load(mode, days, expect='forward')
        out['falsifier_forward_checks_refuse_in_sample_runs'] = len(ferrs)
        if not ferrs:
            sys.exit('FALSIFIER DID NOT FIRE: the forward env checks accepted in-sample runs')
    S = {}
    for tm in ('J', 'X'):
        g = lambda who, k: np.array([V[(tm, d, who)][k] for d in days], float)
        S[tm] = {w: g(w, 'gross_usd') for w in WHO}
        R = {}
        for w in WHO:
            p = W[(tm, w)].pnl.values
            hl = [first_halt(p[i:i + PER_DAY]) for i in range(max(len(p) - PER_DAY + 1, 0))]
            R[w] = dict(settle=ci(g(w, 'gross_usd')), plus1=ci(g(w, 'mo1_usd')), mark_at_own_stop=ci(g(w, 'mtm_stop_usd')),
                        max_drawdown=round(mdd(p), 1), cap150_halt_1d_pct=round(100.0 * np.mean([x is not None for x in hl]), 1) if hl else None,
                        abs_net_stop_mean=round(float(W[(tm, w)].net_end.abs().mean()), 2), fills=int(g(w, 'fills').sum()), shares=round(float(g(w, 'shares').sum()), 0),
                        per_day={d: {k: V[(tm, d, w)][k] for k in ('gross_usd', 'mo1_usd', 'mtm_stop_usd', 'fills', 'shares')} for d in days})
        for b in ('v1', 'control'):
            R[f'v3_minus_{b}'] = dict(settle=ci(g('v3', 'gross_usd') - g(b, 'gross_usd')), plus1=ci(g('v3', 'mo1_usd') - g(b, 'mo1_usd')))
        out['timings'][tm] = R
    out['verdict'] = dict(verdict(S), sums={t: dict(v3=round(float(S[t]['v3'].sum()), 1), v3_minus_control=round(float((S[t]['v3'] - S[t]['control']).sum()), 1),
                                                    v3_minus_v1=round(float((S[t]['v3'] - S[t]['v1']).sum()), 1),
                                                    sd_v3=round(float(S[t]['v3'].std(ddof=1)), 1), sd_v1=round(float(S[t]['v1'].std(ddof=1)), 1)) for t in S})
    io.write_json(f'{S4}/report_forward_v3_{"20261109_13" if mode == "forward" else "rehearsal"}.json', out)
    print(('IN-SAMPLE REHEARSAL (code path only) ' if mode != 'forward' else '') + json.dumps(out['verdict'], indent=1))
    for tm in ('J', 'X'):
        for w, r in out['timings'][tm].items():
            print(tm, f'{w:18s}', 'settle', r['settle'], 'plus1', r['plus1'])
    return 0


if __name__ == '__main__':
    sys.exit(main('rehearse' if sys.argv[1:] == ['rehearse'] else 'forward' if not sys.argv[1:] else sys.exit('usage: [rehearse]')))
