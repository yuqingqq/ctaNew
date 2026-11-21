"""INDEPENDENT CHECKS of ASK-RESERVATION-GUARD-v1 (declaration.md; session aea5bfb9, 2026-11-21). Recomputed from the stored evidence without the
guard module's measurement code:
 A identity: every frozen file committed; the rule sha256 in the manifest = every day record's; the edge = the frozen audit.json's last
   live|1|ctx_bid_reserved_shares edge = 200; sources; output hashes;
 B membership: one record per complete baseline market (19) with both arms; the pilot = the first two complete markets of each date;
 C the disabled control: every disabled ledger's fills, orders, requests and value equal the saved baseline record's (read from the
   baseline run's files), the flags agree;
 D the rule: every intervention on the ask side at float32(B) >= 200 with a guard reason; no guard reason on a bid; no gid requested twice;
 E accounting: each arm's value = cash + position x mark recomputed from its fills;
 F the measurement: arms.csv values / quantities recomputed from the records; the paired END and per-share readings recomputed from arms.csv;
   the selection control recomputed independently (selection set, strata, statistic, valid / total / excluded quantities, the seeded draws);
 G review f8c5e579 finding 2: the audit's stored bin means and censored counts recomputed from its state rows, every lead flag, the pooled
   ranking and the candidate re-derived (the probe's corruptions must fail here);
 H the corrected generation censor count (13 = generations with fills whose lifetime +1 s is missing; 0 <= c <= generations);
 I resources: this round <= 0.75 core-hours; peaks <= 1,536 MiB.
falsify(): each corruption of a copy must fail its check while the clean evidence passes. Usage: guard_checks.py"""
import copy
import glob
import gzip
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path('/home/yuqing/ctaNew'); R = REPO / 'predictfun/reviews/ladder_guard_20261121'; D = REPO / 'data/predict_fun/research/ladder_guard_v1_20261121'
RUN = REPO / 'data/predict_fun/research/ladder_rest_v1_20261121'; AUD = REPO / 'data/predict_fun/research/ladder_audit_v1_20261121'
COMMITTED = ['predictfun/research/pf_ladder_guard.py', 'predictfun/research/pf_ladder_guard_selftest.py', 'predictfun/research/pf_ladder_guard_mutants.py',
             'predictfun/reviews/ladder_guard_20261121/declaration.md', 'predictfun/reviews/ladder_guard_20261121/manifest.json', 'predictfun/reviews/ladder_guard_20261121/launch_guard.sh',
             'predictfun/reviews/ladder_guard_20261121/guard_checks.py', 'predictfun/reviews/ladder_guard_20261121/tools/pilot_projection_guard.py']
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
REM_BANDS, AGE_BANDS = (25., 50., 75.), (500., 2000., 10000.)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git_blob(rel):
    r = subprocess.run(['git', '-C', str(REPO), 'show', f'HEAD:{rel}'], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def gz(p):
    return [json.loads(x) for x in gzip.decompress(Path(p).read_bytes()).decode().splitlines() if x]


def load():
    gm = json.loads((R / 'manifest.json').read_text()); E = dict(gm=gm, days={}, recs=[], base={}, arrays={})
    for day in gm['days']:
        E['days'][day] = json.loads((D / 'days' / f'{day}.json').read_text()); E['recs'] += [dict(r, _day=day) for r in gz(D / 'days' / f'{day}.jsonl.gz')]
        E['base'].update({(day, int(r['m'])): r for r in gz(RUN / 'days' / f'{day}.jsonl.gz')})
    E['pilot_days'] = {day: json.loads((D / 'pilot' / f'{day}.json').read_text()) for day in gm['days'] if (D / 'pilot' / f'{day}.json').exists()}
    E['arms'] = pd.read_csv(D / 'measure' / 'arms.csv', dtype={'day': str}, float_precision='round_trip'); E['maudit'] = json.loads((D / 'measure' / 'audit.json').read_text())
    E['selection'] = json.loads((D / 'selection.json').read_text())
    E['states'] = pd.read_csv(AUD / 'states.csv', dtype={'day': str}, float_precision='round_trip'); E['bins'] = pd.read_csv(AUD / 'bins.csv', dtype={'day': str}, float_precision='round_trip')
    E['leads'] = pd.read_csv(AUD / 'leads.csv'); E['aaudit'] = json.loads((AUD / 'audit.json').read_text()); E['gens'] = pd.read_csv(AUD / 'generations_v2.csv', dtype={'day': str})
    E['edges'] = json.loads((RUN / 'measure' / 'audit.json').read_text())['edges']
    return E


def bin_of(x, e):
    if x is None or not np.isfinite(x):
        return 'missing'
    if x == 0.:
        return 'zero'
    return 'nonzero' if not e else f'q{1 + int(np.searchsorted(np.asarray(e), x, side="right"))}'


def band(x, edges):
    return 'missing' if x is None or not np.isfinite(x) else int(np.searchsorted(np.asarray(edges, float), float(x), side='right'))


def isign(x):
    return 'missing' if x is None or not np.isfinite(x) else (-1 if x < -1e-9 else (1 if x > 1e-9 else 0))


def close(a, b, tol=1e-6):
    return a is not None and b is not None and np.isfinite(float(a)) and np.isfinite(float(b)) and abs(float(a) - float(b)) <= tol


def selection_recompute(S, edges, draws, seed):
    g = S[(S.kind == 'live') & (S.side == 1)].copy(); e = edges['live|1|ctx_bid_reserved_shares']
    g['sel'] = [bin_of(float(np.float32(v)) if np.isfinite(v) else np.nan, e) == 'q5' for v in g.ctx_bid_reserved_shares.astype(float)]
    g['st'] = [json.dumps([d, int(m), band(r, REM_BANDS), band(a, AGE_BANDS), isign(i)]) for d, m, r, a, i in
               zip(g.day, g.m, g.fp_remaining.astype(float), g.ctx_generation_age_ms.astype(float), g.inv_known.astype(float))]
    g['unc'] = (~g.censored_1s.astype(bool)) & (g.win_mo1s.notna() | (g.win_n == 0)); g['y'] = g.win_mo1s.fillna(0.).astype(float)
    u = g[g.unc]; s = u[u.sel]; out = dict(selected=int(len(s)), states=int(len(u)), censored=int((~g.unc).sum()), J=float(s.y.mean()), valid_window_q=float(s.win_q.sum()),
                                          total_window_q=float(g[g.sel].win_q.sum()), excluded_window_q=float(g[g.sel & ~g.unc].win_q.sum()))
    strata = {k: (int(v.sel.sum()), int(len(v)), v.index.values) for k, v in u.groupby('st')}; rng = np.random.default_rng(seed); null = []
    for _ in range(int(draws)):
        pick = [i for k, (n_s, n, ix) in strata.items() if n_s > 0 for i in rng.choice(ix, size=n_s, replace=False)]; null.append(float(u.loc[pick, 'y'].mean()))
    out.update(null_p05=float(np.quantile(null, .05)), null_mean=float(np.mean(null)), movable_strata=sum(1 for n_s, n, _ in strata.values() if 0 < n_s < n))
    return out


def run_checks(E):  # noqa: C901
    gm = E['gm']; chk, det = {}, {}
    chk['A_committed'] = all(git_blob(p) == (REPO / p).read_bytes() for p in COMMITTED)
    sys.path.insert(0, str(REPO / 'predictfun/research')); import pf_ladder_guard as G
    chk['A_rule'] = gm['rule_sha256'] == G.rule_sha() and all(d['rule_sha256'] == gm['rule_sha256'] for d in list(E['days'].values()) + list(E['pilot_days'].values()))
    chk['A_edge'] = gm['edge'] == 200. == float(E['edges'][gm['edge_key']][-1])
    chk['A_sources'] = all(d['sources'] == G.sources() for d in E['days'].values())
    chk['A_outputs'] = all(sha(D / 'days' / f'{day}.jsonl.gz') == d['sha256']['gz'] and sha(D / 'days' / f'{day}_arrays.npz') == d['sha256']['npz'] for day, d in E['days'].items()) and all(
        E['maudit']['outputs'][n] == sha(D / 'measure' / n) for n in ('arms.csv', 'control.csv'))
    comp = {day: sorted(int(m) for (d, m), r in E['base'].items() if d == day and r['status'] == 'complete') for day in gm['days']}
    chk['B_membership'] = {d: sorted(int(m) for m in v) for d, v in gm['complete'].items()} == comp and sorted((r['_day'], int(r['m'])) for r in E['recs']) == sorted((d, m) for d, v in comp.items() for m in v)
    chk['B_pilot'] = all([int(m) for m in gm['pilot'][d]] == comp[d][:2] for d in gm['days']) and all(d_['members'] == [int(m) for m in gm['pilot'][day]] for day, d_ in E['pilot_days'].items())
    cbad = 0
    for r in E['recs']:
        b = E['base'][(r['_day'], int(r['m']))]['ledger']; d_ = r['disabled']
        cbad += int(not (d_['fills'] == b['fills'] and d_['orders'] == b['orders'] and d_['requests'] == b['requests'] and d_['value'] == b['value'] and r['control_all']))
    det['C_disabled_mismatch'] = cbad; chk['C_disabled_equals_baseline'] = cbad == 0
    dbad = 0
    for r in E['recs']:
        g_ = r['guard']; dbad += sum(1 for z in g_['guard_log'] if not (z[1] == 1 and z[5] >= 200. and z[8] in ('guard_pull', 'guard_withhold')))
        dbad += sum(1 for o in g_['orders'] if o['side'] == 0 and o['why'] == 'policy' and any(z[8] == 'guard_pull' and z[3] == o['gid'] for z in g_['guard_log']))
        rq = {}
        for e in g_['requests']:
            rq[e[2]] = rq.get(e[2], 0) + 1
        dbad += sum(1 for v in rq.values() if v > 1)
    det['D_rule_violations'] = dbad; chk['D_rule'] = dbad == 0
    ebad = 0
    for r in E['recs']:
        for arm in ('disabled', 'guard'):
            L_ = r[arm]; sg = lambda s: 1. if int(s) == 0 else -1.
            cash = -sum(sg(f[2]) * float(f[3]) * float(f[4]) for f in L_['fills']); pos = sum(sg(f[2]) * float(f[3]) for f in L_['fills'])
            ebad += int(not (close(cash, L_['cash']) and close(pos, L_['position']) and close(cash + pos * L_['mark'], L_['value'])))
    chk['E_accounting'] = ebad == 0
    A_ = E['arms']; fb = 0
    for r in E['recs']:
        for arm in ('disabled', 'guard'):
            row = A_[(A_.day == r['_day']) & (A_.m == int(r['m'])) & (A_.arm == arm)]
            fb += int(not (len(row) == 1 and close(row.value.iloc[0], r[arm]['value']) and int(row.fills.iloc[0]) == len(r[arm]['fills'])
                           and close(row.all_q.iloc[0], sum(float(f[3]) for f in r[arm]['fills']))))
    chk['F_arms'] = fb == 0
    gg, bb = A_[A_.arm == 'guard'].set_index(['day', 'm']), A_[A_.arm == 'disabled'].set_index(['day', 'm']); days = sorted(A_.day.unique())
    dEND = {d: float((gg.loc[d].value - bb.loc[d].value).sum()) for d in days}; pr = E['maudit']['predicates']
    cat = lambda v: 'IMPROVES_EVERY_DAY' if all(x > 0 for x in v) else 'WORSENS_EVERY_DAY' if all(x < 0 for x in v) else 'MIXED'
    chk['F_END_reading'] = pr['END_paired']['answer'] == cat(list(dEND.values())) and all(close(pr['END_paired']['by_day'][d], dEND[d]) for d in days)
    per = lambda X, d: float(X.loc[d].all_cut_mo1s.sum() / X.loc[d].all_cut_valid_q.sum())
    dps = {d: per(gg, d) - per(bb, d) for d in days}
    chk['F_per_share_reading'] = pr['mo1s_per_share']['all']['answer'] == cat(list(dps.values())) and all(close(pr['mo1s_per_share']['all']['by_day'][d], dps[d]) for d in days)
    sr = selection_recompute(E['states'], E['edges'], E['selection']['draws'], E['selection']['seed']); sel = E['selection']; det['F_selection'] = sr
    chk['F_selection'] = (sr['selected'] == sel['selected'] and sr['states'] == sel['states'] and close(sr['J'], sel['J']) and close(sr['null_p05'], sel['null_p05'])
                          and close(sr['null_mean'], sel['null_mean']) and sr['movable_strata'] == sel['movable_strata'] and pr['selection'] == sel['answer'])
    # G finding 2: stored bin means / censored counts recomputed from the state rows; flags, ranking and the candidate re-derived
    S = E['states']; recomputed = {}; import pf_ladder_rest as M
    for (kind, s), g in S.groupby(['kind', 'side']):
        for c in M.DICT_COLS:
            e = E['edges'].get(f'{kind}|{s}|{c}', []); b = [bin_of(v, e) for v in g[c].astype(float)]
            for (day, bn), h in g.assign(_b=b).groupby(['day', '_b']):
                u = h[((~h.censored_1s.astype(bool)) & h.win_mo1s.notna()) | (h.win_n == 0)]
                recomputed[(kind, int(s), c, str(day), bn)] = (int(h.censored_1s.astype(bool).sum()), int(len(u)), float(u.win_mo1s.fillna(0.).mean()) if len(u) else None)
    BT = E['bins']; mbad = cbad2 = 0
    for r in BT.itertuples():
        k = (r.kind, int(r.side), r.feature, str(r.day), r.bin); rc = recomputed.get(k)
        if rc is None:
            mbad += 1; cbad2 += 1; continue
        mbad += int(not ((rc[2] is None and pd.isna(r.mo1s_mean)) or close(rc[2], r.mo1s_mean, 1e-9))); cbad2 += int(rc[0] != int(r.censored) or rc[1] != int(r.uncensored))
    chk['G_bin_means'] = mbad == 0 and len(recomputed) == len(BT); chk['G_bin_censored'] = cbad2 == 0
    flags, pooled = {}, {}
    for kind in ('entry', 'first', 'live'):
        for c in M.DICT_COLS:
            cells = []
            for day in sorted(S.day.unique()):
                for s in (0, 1):
                    a, b5 = recomputed.get((kind, s, c, day, 'q1')), recomputed.get((kind, s, c, day, 'q5'))
                    cells.append(None if not (a and b5 and a[1] >= 10 and b5[1] >= 10) else b5[2] - a[2])
            g = S[S.kind == kind]; bins = [bin_of(v, E['edges'].get(f'{kind}|{s}|{c}', [])) for v, s in zip(g[c].astype(float), g.side)]
            un = g.assign(_b=bins); un = un[((~un.censored_1s.astype(bool)) & un.win_mo1s.notna()) | (un.win_n == 0)]; y = un.win_mo1s.fillna(0.)
            pooled[(kind, c)] = float(y[un._b == 'q5'].mean() - y[un._b == 'q1'].mean()) if (un._b == 'q5').any() and (un._b == 'q1').any() else None
            sg_ = {int(np.sign(x)) for x in cells if x is not None}; flags[(kind, c)] = bool(all(x is not None for x in cells) and len(sg_) == 1 and 0 not in sg_ and pooled[(kind, c)] is not None)
    LT = E['leads']; chk['G_lead_flags'] = all(flags[(r.kind, r.feature)] == bool(r.lead) for r in LT.itertuples()) and all(
        (pooled[(r.kind, r.feature)] is None and pd.isna(r.pooled_q5_minus_q1)) or close(pooled[(r.kind, r.feature)], r.pooled_q5_minus_q1, 1e-9) for r in LT.itertuples())
    leads = [k for k, v in flags.items() if v]; cand = sorted(leads, key=lambda k: (-abs(pooled[k]), ('entry', 'first', 'live').index(k[0]), M.DICT_COLS.index(k[1])))[0] if leads else None
    ac = E['aaudit']['candidate']
    chk['G_candidate'] = (cand is None and ac['answer'] == 'NO_LEAD') or (cand is not None and ac['answer'] == 'LEAD' and (ac['kind'], ac['feature']) == cand and close(ac['pooled_q5_minus_q1'], pooled[cand], 1e-9)
                                                                        and int(ac['sign']) == int(np.sign(pooled[cand])) and int(ac['n_leads']) == len(leads))
    gens = E['gens']; cc = int((gens.lifetime_mo1s.isna() & (gens.n_fills > 0)).sum()); det['H_censor_count'] = cc
    chk['H_censor_count'] = 0 <= cc <= len(gens) and E['selection']['generation_censor_count'] == cc
    chk['H_valid_terms'] = close(sr['valid_window_q'], sel['valid_window_q']) and close(sr['total_window_q'], sel['total_window_q']) and close(sr['excluded_window_q'], sel['excluded_window_q'])
    cpu, peak = 0., 0
    for p in glob.glob(str(R / 'logs' / '**' / '*.log'), recursive=True):
        for ln in Path(p).read_text().splitlines():
            m_ = RL.match(ln)
            if m_:
                cpu += float(m_.group(6)); peak = max(peak, int(m_.group(4)))
    det['I'] = dict(core_hours=cpu / 3600., peak_mib=peak); chk['I_resources'] = cpu / 3600. <= .75 and peak < 1536
    return dict(checks=chk, details=det, all_ok=bool(all(chk.values())))


def _first_logged(E):
    return next(r for r in E['recs'] if r['guard']['guard_log'])


CASES = {
    'D_rule': lambda E: _first_logged(E)['guard']['guard_log'][0].__setitem__(5, 150.),
    'C_disabled_equals_baseline': lambda E: E['recs'][0].update(control_all=False),
    'F_arms': lambda E: E.update(arms=E['arms'].assign(value=E['arms'].value + 1.)),
    'G_bin_means': lambda E: E.update(bins=E['bins'].assign(mo1s_mean=np.where((E['bins'].kind == 'live') & (E['bins'].feature == 'ctx_bid_reserved_shares') & (E['bins'].bin == 'q5'),
                                                                                E['bins'].mo1s_mean + 1e3, E['bins'].mo1s_mean))),
    'G_bin_censored': lambda E: E.update(bins=E['bins'].assign(censored=E['bins'].censored + 1)),
    'F_selection': lambda E: E['selection'].update(J=E['selection']['J'] + 1.),
    'H_valid_terms': lambda E: E['selection'].update(valid_window_q=E['selection']['valid_window_q'] + 1.),
}


def falsify(E):
    base = run_checks(E); out = {}
    for name, fn in CASES.items():
        X_ = copy.deepcopy(E); fn(X_); r = run_checks(X_)
        out[name] = dict(clean=bool(base['checks'].get(name)), corrupted=bool(r['checks'].get(name)), applied=name in r['checks'])
    return dict(cases=out, falsifier_ok=bool(all(v['clean'] and not v['corrupted'] and v['applied'] for v in out.values())))


def main():
    E = load(); res = run_checks(E); (R / 'checks.json').write_text(json.dumps(res, indent=1, default=str) + '\n'); fz = falsify(E)
    (R / 'checks_falsifier.json').write_text(json.dumps(fz, indent=1) + '\n')
    print(json.dumps(dict(all_ok=res['all_ok'], failed=[k for k, v in res['checks'].items() if not v], falsifier_ok=fz['falsifier_ok'], details=res['details']), indent=1, default=str)[:3000])


if __name__ == '__main__':
    main()
