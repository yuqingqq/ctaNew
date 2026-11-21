"""INDEPENDENT CHECKS v3 of RECEIPT-LEAD-TIME-AUDIT-v1 (declaration.md; session aea5bfb9, 2026-11-21). SUPERSEDES v2 (b8642e6d, kept with its log),
which crashed before computing any check: pandas read the audit CSVs' day column as integers while the run records key days as strings; v3
reads day as text. Nothing else changed from v2, which SUPERSEDED audit_checks.py (v1, frozen in
b1167651, kept with its log): v1 crashed before writing any output (a Python list combined with a pandas Series in the states membership
filter), and its fill-timing cross-check compared the audit's GENERATION-censored +1 s sum (a generation with any fill lacking its +1 s mid
excluded) with the review's PER-FILL sum over valid fills - two declared conventions, not one number. v2 fixes the filter and checks the
counts / quantities against the review, the per-fill +1 s sums recomputed from the raw ledgers against the review's, and the audit's
generation-censored sums against an independent generation-level recount. Nothing else changed. Recomputed from the saved LADDER_REST run and
the audit outputs, without the audit module's code:
 A identity: every frozen file committed and byte-equal; the run's recorded sources = the 47d2eec1 blobs; every audit output hash;
 B membership: 96 statuses (19 complete); one generations_v2 row per submitted generation of the complete markets; one states row per saved
   snapshot of the complete markets, the kinds kept apart;
 C the review's independent numbers (reviews/automatic/20261121_ladder_rest_review evidence.json / diagnostics.json, read from the files):
   the [pending_first, live_first) window, the sweep rows, the live / first window denominators, the early requests by reason, recovery /
   reversal, the fills before a request and while cancel-pending, the exact both-side share, the all-horizon common subset and the net top-5
   loss share equal the audit's (1e-6; the share 1e-9);
 D the window labels of every 25th states row recomputed from the raw ledger by the window rule; bins.csv recomputed from states.csv and
   the frozen edges; the lead rule and the candidate recomputed from bins.csv;
 E the causal audit: 4 days, every sampled snapshot unchanged under future-only perturbation (market and order inputs), the market positive
   control firing at least once a day, the order control firing on every eligible snapshot;
 F resources: this round's CPU <= 0.5 core-hours, peaks <= 1,536 MiB.
falsify(): each corruption of a copy must fail its check while the clean evidence passes. Usage: audit_checks.py"""
import copy
import glob
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path('/home/yuqing/ctaNew'); R = REPO / 'predictfun/reviews/ladder_audit_20261121'; D = REPO / 'data/predict_fun/research/ladder_audit_v1_20261121'
RUN = REPO / 'data/predict_fun/research/ladder_rest_v1_20261121'; REV = REPO / 'predictfun/reviews/automatic/20261121_ladder_rest_review'
COMMITTED = ['predictfun/research/pf_ladder_audit.py', 'predictfun/research/pf_ladder_audit_selftest.py', 'predictfun/research/pf_branch_ladder_safe.py',
             'predictfun/research/pf_ladder_audit_mutants.py', 'predictfun/research/pf_branch_ladder_safe_mutants.py', 'predictfun/research/pf_ladder_rest_selftest_v2.py',
             'predictfun/research/pf_branch_ladder_mutants_v2.py', 'predictfun/research/pf_ladder_rest_mutants_v2.py', 'predictfun/reviews/ladder_audit_20261121/declaration.md',
             'predictfun/reviews/ladder_audit_20261121/launch_audit.sh', 'predictfun/reviews/ladder_audit_20261121/audit_checks.py',
             'predictfun/reviews/ladder_audit_20261121/audit_checks_v2.py', 'predictfun/reviews/ladder_audit_20261121/audit_checks_v3.py']
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
CSV = ('generations_v2', 'sweep_rows', 'recovery', 'durations', 'churn', 'states', 'bins', 'leads')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*a):
    r = subprocess.run(['git', '-C', str(REPO)] + list(a), capture_output=True)
    return r.stdout if r.returncode == 0 else None


def load():
    import gzip
    au = json.loads((D / 'audit.json').read_text()); E = dict(audit=au, recs=[], causal={})
    for nm in CSV:
        E[nm] = pd.read_csv(D / f'{nm}.csv', float_precision='round_trip', dtype={'day': str}); E[f'{nm}_sha'] = sha(D / f'{nm}.csv')
    man = json.loads((REPO / 'predictfun/reviews/ladder_rest_20261121/manifest.json').read_text()); E['days'] = man['days']; E['run_days'] = {}
    for day in man['days']:
        E['run_days'][day] = json.loads((RUN / 'days' / f'{day}.json').read_text())
        E['recs'] += [json.loads(x) for x in gzip.decompress((RUN / 'days' / f'{day}.jsonl.gz').read_bytes()).decode().splitlines() if x]
        p = D / f'causal_{day}.json'
        if p.exists():
            E['causal'][day] = json.loads(p.read_text())
    E['features'] = pd.concat([pd.read_parquet(RUN / 'days' / f'{d}_features.parquet') for d in man['days']], ignore_index=True)
    E['edges'] = json.loads((RUN / 'measure' / 'audit.json').read_text())['edges']
    E['rev_e'] = json.loads((REV / 'evidence.json').read_text()); E['rev_d'] = json.loads((REV / 'diagnostics.json').read_text())
    return E


def sg(s):
    return 1. if int(s) == 0 else -1.


def eff(d, o):
    return max(d + 37., float(o['submit']) + 95.) if d < float(o['act']) else d + 37.


def bin_of(x, e):
    if x is None or not np.isfinite(x):
        return 'missing'
    if x == 0.:
        return 'zero'
    return 'nonzero' if not e else f'q{1 + int(np.searchsorted(np.asarray(e), x, side="right"))}'


def close(a, b, tol=1e-6):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


def run_checks(E):  # noqa: C901
    au, cr = E['audit'], E['audit']['cross']; chk, det = {}, {}
    chk['A_committed'] = all(git('show', f'HEAD:{p}') == (REPO / p).read_bytes() for p in COMMITTED)
    srcs = E['run_days'][E['days'][0]]['sources']
    chk['A_run_sources'] = all(d['sources'] == srcs for d in E['run_days'].values()) and all(
        hashlib.sha256(git('show', f'47d2eec1:predictfun/research/{n}') or b'').hexdigest() == h for n, h in srcs.items())
    chk['A_outputs'] = all(au['outputs'].get(f'{nm}.csv') == E[f'{nm}_sha'] for nm in CSV)
    comp = [r for r in E['recs'] if r['status'] == 'complete']; st = {}
    for r in E['recs']:
        st[r['status']] = st.get(r['status'], 0) + 1
    chk['B_statuses'] = len(E['recs']) == 96 and st.get('complete') == 19 and au['statuses'] == st
    chk['B_generations'] = len(E['generations_v2']) == sum(len(r['ledger']['orders']) for r in comp)
    keys = {(r['day'], int(r['m'])) for r in comp}; F = E['features']; Fc = F[np.array([(d, int(m)) in keys for d, m in zip(F.day, F.m)]) & (F.gid >= 0).values]
    chk['B_states'] = len(E['states']) == len(Fc) and sorted(E['states'].kind.value_counts().items()) == sorted(Fc.kind.value_counts().items())
    rv, rd = E['rev_e'], E['rev_d']
    chk['C_locked_window'] = (cr['locked_window']['fills'] == rv['locked_window']['fills'] and cr['locked_window']['generations'] == rv['locked_window']['generations']
                              and close(cr['locked_window']['shares'], rv['locked_window']['shares']) and close(cr['locked_window']['end_value'], rv['locked_window']['end_value']))
    chk['C_sweeps'] = all(cr['sweeps'][k] == rd['sweeps'][k] for k in ('trade_grouped_fill_rows', 'time_grouped_fill_rows', 'time_grouped_but_not_same_trade_rows'))
    chk['C_window_denominators'] = all(cr['live_fill_denominators'][k][z] == rv['live_fill_denominators'][k][z] for k in ('first', 'live')
                                       for z in ('n', 'any_generation_fill', 'future_fill'))
    chk['C_early_requests'] = (cr['first_review']['requested_by_first_live'] == rd['first_review']['accepted_orders_requested_by_first_live_tick']
                               and cr['first_review']['by_reason'] == rd['first_review']['by_reason']
                               and cr['first_review']['policy_first_review_pulls'] == rd['first_review']['reported_policy_first_review_pulls'])
    chk['C_recovery'] = cr['inventory_modes']['recovered'] == rd['inventory_modes']['recovery_to_band_exits'] and cr['inventory_modes']['reversal'] == rd['inventory_modes']['direct_reversal_exits']
    ft = rd['fill_timing']
    chk['C_fill_timing_counts'] = all(cr['fill_timing'][k]['fills'] == ft[k]['fills'] and close(cr['fill_timing'][k]['q'], ft[k]['q']) for k in ('before_request', 'cancel_pending'))
    pf = {'before_request': [0., 0], 'cancel_pending': [0., 0]}; gc = {'before_request': 0., 'cancel_pending': 0.}
    for r in comp:
        L_ = r['ledger']
        for o in L_['orders']:
            fl = [(f, mm) for f, mm in zip(L_['fills'], L_['fill_mids']) if int(f[1]) == int(o['gid'])]
            parts = dict(before_request=[z for z in fl if o['req'] is None or float(z[0][0]) < float(o['req'])],
                         cancel_pending=[z for z in fl if o['req'] is not None and float(o['req']) <= float(z[0][0]) < float(o['remove'])])
            for k, zs in parts.items():
                for f, mm in zs:
                    if mm[6] is not None:
                        pf[k][0] += sg(f[2]) * float(f[3]) * (mm[6] - float(f[4])); pf[k][1] += 1
                if zs and all(mm[6] is not None for f, mm in zs):
                    gc[k] += sum(sg(f[2]) * float(f[3]) * (mm[6] - float(f[4])) for f, mm in zs)
    det['C_fill_timing'] = dict(per_fill={k: v[0] for k, v in pf.items()}, per_fill_valid={k: v[1] for k, v in pf.items()}, generation_censored=gc)
    chk['C_fill_timing_per_fill_vs_review'] = all(close(pf[k][0], ft[k]['mo1000']) and pf[k][1] == ft[k]['n1000'] for k in ('before_request', 'cancel_pending'))
    chk['C_fill_timing_generation_censored'] = all(close(gc[k], cr['fill_timing'][k]['mo1s']) for k in ('before_request', 'cancel_pending'))
    chk['C_both_exact'] = close(cr['durations']['both_exact'], rv['headline']['both_exact'], 1e-9)
    mh = rd['markout_headline']; ca = cr['common_markouts_all_horizons']
    chk['C_common_subset'] = (ca['fills'] == int(mh['common_fills']) and close(ca['q'], mh['common_q']) and close(ca['edge'], mh['common_edge']) and close(ca['mo50'], mh['common_mo50'])
                              and close(ca['mo1s'], mh['common_mo1000']))
    chk['C_net_top5'] = close(cr['net_concentration']['top5_share'], mh['top5_market_loss_share'], 1e-9)
    # D window labels (every 25th states row), bins and leads
    byk = {(r['day'], int(r['m'])): r for r in comp}; S = E['states']; bad_w = 0
    for x in S.iloc[::25].to_dict('records'):
        L_ = byk[(x['day'], int(x['m']))]['ledger']; o = [y for y in L_['orders'] if int(y['gid']) == int(x['gid'])][0]; mark = float(L_['mark'])
        fl = [(f, mm) for f, mm in zip(L_['fills'], L_['fill_mids']) if int(f[1]) == int(x['gid'])]
        e_ = float(o['act']) if x['kind'] == 'entry' else eff(float(x['t']), o); w = fl if x['kind'] == 'entry' else [z for z in fl if float(z[0][0]) >= e_]
        q = sum(float(f[3]) for f, mm in w); v = sum(sg(f[2]) * float(f[3]) * (mark - float(f[4])) for f, mm in w)
        cen = any(mm[6] is None for f, mm in w); m1 = None if cen else sum(sg(f[2]) * float(f[3]) * (mm[6] - float(f[4])) for f, mm in w)
        bad_w += int(not (close(x['win_q'], q) and close(x['win_end'], v) and bool(x['censored_1s']) == (cen and len(w) > 0) and (cen or close(x['win_mo1s'], m1))))
    det['D_windows_checked'] = len(S.iloc[::25]); chk['D_windows'] = bad_w == 0
    import pf_ladder_rest as M  # noqa: E402 (the dictionary only)
    sys.path.insert(0, str(REPO / 'predictfun/research'))
    rows = []
    for (kind, s), g in S.groupby(['kind', 'side']):
        for c in M.DICT_COLS:
            e = E['edges'].get(f'{kind}|{s}|{c}', []); b = [bin_of(v, e) for v in g[c].astype(float)]
            for (day, bn), h in g.assign(_b=b).groupby(['day', '_b']):
                u = h[((~h.censored_1s.astype(bool)) & h.win_mo1s.notna()) | (h.win_n == 0)]; y = u.win_mo1s.fillna(0.).astype(float)
                rows.append((kind, int(s), c, day, bn, len(h), len(u), float(y.sum())))
    BT = E['bins']; bt = {(r.kind, int(r.side), r.feature, str(r.day), r.bin): (int(r.n), int(r.uncensored), float(r.mo1s_sum)) for r in BT.itertuples()}
    chk['D_bins'] = len(rows) == len(BT) and all(k[:5] in bt and bt[k[:5]][0] == k[5] and bt[k[:5]][1] == k[6] and close(bt[k[:5]][2], k[7]) for k in rows)
    cand = None; best = None
    for kind in ('entry', 'first', 'live'):
        for c in M.DICT_COLS:
            sub = BT[(BT.kind == kind) & (BT.feature == c)]; cells = []
            for day in E['days']:
                for s in (0, 1):
                    a = sub[(sub.day.astype(str) == day) & (sub.side == s) & (sub.bin == 'q1')]; b = sub[(sub.day.astype(str) == day) & (sub.side == s) & (sub.bin == 'q5')]
                    cells.append(None if not (len(a) and len(b) and int(a.uncensored.iloc[0]) >= 10 and int(b.uncensored.iloc[0]) >= 10) else float(b.mo1s_mean.iloc[0]) - float(a.mo1s_mean.iloc[0]))
            if all(x is not None for x in cells) and len({np.sign(x) for x in cells}) == 1 and np.sign(cells[0]) != 0:
                lt = E['leads'][(E['leads'].kind == kind) & (E['leads'].feature == c)]; pooled = float(lt.pooled_q5_minus_q1.iloc[0])
                if best is None or abs(pooled) > best:
                    best, cand = abs(pooled), (kind, c)
    want = au['candidate']; det['D_candidate'] = dict(recomputed=cand, audit=[want.get('kind'), want.get('feature')] if want['answer'] == 'LEAD' else None)
    chk['D_leads'] = (cand is None and want['answer'] == 'NO_LEAD') or (cand is not None and want['answer'] == 'LEAD' and (want['kind'], want['feature']) == cand)
    lflag = {(r.kind, r.feature): bool(r.lead) for r in E['leads'].itertuples()}
    chk['D_lead_flags'] = cand is None or lflag.get(cand, False)
    # E causal
    cz = E['causal']; det['E'] = {d: {k: v for k, v in c.items() if k not in ('rows', 'sources')} for d, c in cz.items()}
    chk['E_causal'] = bool(len(cz) == 4 and all(c['n'] > 0 and c['market_same'] == c['n'] and c['order_same'] == c['n'] and c['market_fired'] >= 1
                                                and c['order_fired'] == c['order_fire_eligible'] for c in cz.values()))
    cpu, peak = 0., 0
    for p in glob.glob(str(R / 'logs' / '**' / '*.log'), recursive=True):
        for ln in Path(p).read_text().splitlines():
            m_ = RL.match(ln)
            if m_:
                cpu += float(m_.group(6)); peak = max(peak, int(m_.group(4)))
    det['F'] = dict(core_hours=cpu / 3600., peak_mib=peak); chk['F_resources'] = cpu / 3600. <= .5 and peak < 1536
    return dict(checks=chk, details=det, all_ok=bool(all(chk.values())))


CASES = {
    'C_locked_window': lambda E: E['audit']['cross']['locked_window'].update(fills=E['audit']['cross']['locked_window']['fills'] + 1),
    'C_sweeps': lambda E: E['audit']['cross']['sweeps'].update(trade_grouped_fill_rows=0),
    'B_generations': lambda E: E.update(generations_v2=E['generations_v2'].iloc[1:]),
    'D_windows': lambda E: E.update(states=E['states'].assign(win_q=E['states'].win_q + 1.)),
    'D_bins': lambda E: E.update(bins=E['bins'].assign(mo1s_sum=E['bins'].mo1s_sum + 1.)),
    'E_causal': lambda E: next(iter(E['causal'].values())).update(market_same=0),
    'C_both_exact': lambda E: E['audit']['cross']['durations'].update(both_exact=E['audit']['cross']['durations']['both_exact'] + 1e-6),
}


def falsify(E):
    base = run_checks(E); out = {}
    for name, fn in CASES.items():
        X_ = copy.deepcopy(E); fn(X_); r = run_checks(X_)
        out[name] = dict(clean=bool(base['checks'].get(name)), corrupted=bool(r['checks'].get(name)), applied=name in r['checks'])
    return dict(cases=out, falsifier_ok=bool(all(v['clean'] and not v['corrupted'] and v['applied'] for v in out.values())))


def main():
    sys.path.insert(0, str(REPO / 'predictfun/research')); E = load(); res = run_checks(E)
    (R / 'checks_v3.json').write_text(json.dumps(res, indent=1, default=str) + '\n'); fz = falsify(E); (R / 'checks_falsifier_v3.json').write_text(json.dumps(fz, indent=1) + '\n')
    print(json.dumps(dict(all_ok=res['all_ok'], failed=[k for k, v in res['checks'].items() if not v], falsifier_ok=fz['falsifier_ok'], details=res['details']), indent=1, default=str)[:3000])


if __name__ == '__main__':
    main()
