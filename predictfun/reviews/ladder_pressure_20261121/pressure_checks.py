"""INDEPENDENT CHECKS of BINANCE-PRESSURE-GUARD-v1 (declaration.md; session aea5bfb9, 2026-11-21). Recomputed from the stored evidence: the day
records and arrays, the accepted baseline's records and feature rows, the audit's states and the frozen edge file. The pressure module is
imported for rule_sha() and its encodings only (REASONS, ST_CODES); no measurement code of it is used (review 2c5dbd53 finding 1).
 A identity: frozen files committed; the rule sha256 = the manifest's = every day record's; the manifest's edges = the frozen edge file's
   (its sha256); every record's sources = the manifest's; output hashes;
 B membership: one record per complete baseline market (19) with both arms; the pilot = the first two complete markets of each date;
 C the disabled control from the baseline's files: fills, orders, requests, value, fingerprint, receipt-known queue estimates, initial queue;
 D the rule from the saved cache arrays: every logged intervention at an ok tick with the cached float32 bits and ON by this checker's own
   q5 test (finite, > 0, >= the last frozen edge), a pressure reason, on its side; no pressure-arm KEEP (other than a fallback) or POST (or a
   cap / near-END WAIT after a POST) at an ON tick of its side; every pressure PULL / WAIT at an ON tick; no gid requested twice; no pressure
   reason in the disabled arm;
 E accounting: each arm's cash, position and value from its fills;
 F reconcile: every arms.csv field a reading uses recomputed from the fills, fill mids, orders and logs (value, quantities, END, edge / +50 /
   +100 ms sums with valid quantities and censor counts, the +1 s cutoff sums, good / bad, the END identity terms, rung splits, interventions);
   the END identity (eligible +1 s + movement + late = END = value) and guard - baseline = added - lost by this checker's own fill matching;
   the readings (END, by side, per share, participation, interpretation) from the recomputed metrics and this checker's selection answer;
   the selection control recomputed with the manifest's draws / seed / bands and its answer derived here;
 G integration: snapshot parity recomputed from the cache arrays against the baseline's saved feature rows (bits); the side sign from the
   arrays; per date the CAUSAL_N minimum-rank ok ticks recomputed and each causal entry unchanged / fired / equal to the cache;
 H queue: the initial queue of every non-refused order, a pull entry for every pressure pull, a first-review entry for every reviewed gid;
 I resources: this round <= 0.75 core-hours; peaks <= 1,536 MiB.
falsify(): each corruption of a copy must fail its family while the clean evidence passes; it includes the review's coordinated case (a
markout field and the stored reading moved together). Usage: pressure_checks.py | pressure_checks.py rehearse <dir> (the pre-run rehearsal on
the pilot copy: it prints the check booleans and the falsifier outcomes only; B_membership fails there by design)."""
import collections
import copy
import glob
import gzip
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path('/home/yuqing/ctaNew'); R = REPO / 'predictfun/reviews/ladder_pressure_20261121'; D = REPO / 'data/predict_fun/research/ladder_pressure_v1_20261121'
RUN = REPO / 'data/predict_fun/research/ladder_rest_v1_20261121'; AUD = REPO / 'data/predict_fun/research/ladder_audit_v1_20261121'
EDGE_FILE, EDGES_SHA = RUN / 'measure' / 'audit.json', 'ca688917cd7ddc7c40e94031bee646f4fc1339511bb7c48fbe5270797a38bf80'
COMMITTED = ['predictfun/research/pf_ladder_pressure.py', 'predictfun/research/pf_ladder_pressure_selftest.py', 'predictfun/research/pf_ladder_pressure_mutants.py',
             'predictfun/reviews/ladder_pressure_20261121/declaration.md', 'predictfun/reviews/ladder_pressure_20261121/manifest.json',
             'predictfun/reviews/ladder_pressure_20261121/launch_pressure.sh', 'predictfun/reviews/ladder_pressure_20261121/pressure_checks.py',
             'predictfun/reviews/ladder_pressure_20261121/tools/pilot_projection_pressure.py']
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
STEP, SG = 50., {0: 1., 1: -1.}
sys.path.insert(0, str(REPO / 'predictfun/research'))
import pf_ladder_pressure as P                            # noqa: E402  rule_sha and encodings only
import pf_ladder_rest as M                                # noqa: E402  encodings only (CMD_KINDS / CMD_ACTIONS)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git_blob(rel):
    r = subprocess.run(['git', '-C', str(REPO), 'show', f'HEAD:{rel}'], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def gz(p):
    return [json.loads(x) for x in gzip.decompress(Path(p).read_bytes()).decode().splitlines() if x]


def bits(v):
    return np.asarray([v], np.float32).view(np.uint32)[0]


def close(a, b, tol=1e-6):
    a, b = float(a), float(b)
    return (np.isnan(a) and np.isnan(b)) or abs(a - b) <= tol * max(1., abs(a), abs(b))


def on(x, last):
    """This checker's own q5 test: finite, positive and at least the last frozen edge (bin_of's q5 with four edges)."""
    return bool(np.isfinite(x) and x > 0. and x >= last)


def load():
    gm = json.loads((R / 'manifest.json').read_text()); E = dict(gm=gm, days={}, recs=[], base={}, arr={}, feats={})
    for day in gm['days']:
        E['days'][day] = json.loads((D / 'days' / f'{day}.json').read_text()); E['recs'] += [dict(r, _day=day) for r in gz(D / 'days' / f'{day}.jsonl.gz')]
        E['base'].update({(day, int(r['m'])): r for r in gz(RUN / 'days' / f'{day}.jsonl.gz')})
        z = np.load(D / 'days' / f'{day}_arrays.npz'); E['arr'][day] = {k: z[k] for k in z.files}
        E['feats'][day] = pd.read_parquet(RUN / 'days' / f'{day}_features.parquet', columns=['m', 'side', 't', 'status', 'adv_bn_r250'])
    E['pilot_days'] = {day: json.loads((D / 'pilot' / f'{day}.json').read_text()) for day in gm['days'] if (D / 'pilot' / f'{day}.json').exists()}
    E['arms'] = pd.read_csv(D / 'measure' / 'arms.csv', dtype={'day': str}, float_precision='round_trip')
    E['la'] = pd.read_csv(D / 'measure' / 'lost_added.csv', dtype={'day': str}, float_precision='round_trip')
    E['maudit'] = json.loads((D / 'measure' / 'audit.json').read_text()); E['selection'] = json.loads((D / 'selection.json').read_text())
    E['states'] = pd.read_csv(AUD / 'states.csv', dtype={'day': str}, float_precision='round_trip'); E['edges_raw'] = json.loads(EDGE_FILE.read_text())['edges']
    return E


def arr(E, day, m, name):
    return E['arr'][day][f'{m}__{name}']


# ---------------------------------------------------------------------------------------------------------------------- families
def fam_A(E):
    gm = E['gm']; c = {}
    c['A_committed'] = all(git_blob(p) == (REPO / p).read_bytes() for p in COMMITTED)
    c['A_rule'] = gm['rule_sha256'] == P.rule_sha() and all(d['rule_sha256'] == gm['rule_sha256'] for d in list(E['days'].values()) + list(E['pilot_days'].values()))
    c['A_edges'] = sha(EDGE_FILE) == EDGES_SHA == gm['edges_sha256'] and all(gm['edges'][str(s)] == [float(v) for v in E['edges_raw'][f'live|{s}|adv_bn_r250']] for s in (0, 1))
    c['A_sources'] = all(d['sources'] == gm['sources'] for d in list(E['days'].values()) + list(E['pilot_days'].values()))
    c['A_outputs'] = all(sha(D / 'days' / f'{day}.jsonl.gz') == d['sha256']['gz'] and sha(D / 'days' / f'{day}_arrays.npz') == d['sha256']['npz'] for day, d in E['days'].items()) and all(
        E['maudit']['outputs'][n] == sha(D / 'measure' / n) for n in ('arms.csv', 'control.csv', 'lost_added.csv', 'causal.csv'))
    return c, {}


def fam_B(E):
    gm = E['gm']; comp = {day: sorted(int(m) for (d, m), r in E['base'].items() if d == day and r['status'] == 'complete') for day in gm['days']}
    got = sorted((r['_day'], int(r['m'])) for r in E['recs'])
    c = dict(B_membership={d: sorted(int(m) for m in v) for d, v in gm['complete'].items()} == comp and got == sorted((d, m) for d, v in comp.items() for m in v)
             and all('disabled' in r and 'pressure' in r for r in E['recs']),
             B_pilot=all([int(m) for m in gm['pilot'][d]] == comp[d][:2] for d in gm['days']) and all(d_['members'] == [int(m) for m in gm['pilot'][day]] for day, d_ in E['pilot_days'].items()))
    return c, dict(complete={d: len(v) for d, v in comp.items()})


def fam_C(E):
    bad = 0
    for r in E['recs']:
        b = E['base'][(r['_day'], int(r['m']))]['ledger']; d_ = r['disabled']
        bad += int(not (d_['fills'] == b['fills'] and d_['orders'] == b['orders'] and d_['requests'] == b['requests'] and d_['value'] == b['value'] and d_['fingerprint'] == b['fingerprint']
                        and d_['queue_known'] == b['queue'] and {g: v[1] for g, v in d_['queue_venue']['ahead0'].items()} == b['ahead0'] and r['control_all']))
    return dict(C_disabled_equals_baseline=bad == 0), dict(C_mismatch=bad)


def fam_D(E):
    last = {s: float(E['gm']['edges'][str(s)][-1]) for s in (0, 1)}; KI, AI = M.CMD_KINDS, M.CMD_ACTIONS; RS = P.REASONS
    lb = cb = rb = db = 0; n_on_cmds = 0
    for r in E['recs']:
        day, m = r['_day'], int(r['m']); xs = {s: arr(E, day, m, f'cache__x{s}') for s in (0, 1)}; st = {s: arr(E, day, m, f'cache__status{s}') for s in (0, 1)}
        ON = {s: np.array([st[s][k] == 0 and on(float(xs[s][k]), last[s]) for k in range(len(xs[s]))], bool) for s in (0, 1)}
        for z in r['pressure']['guard_log']:
            k, s = int(round(z[0] / STEP)), int(z[1])
            lb += int(not (0 <= k < len(xs[s]) and st[s][k] == 0 and bits(z[5]) == bits(xs[s][k]) and on(float(z[5]), last[s]) and z[10] in ('pressure_pull', 'pressure_withhold')))
        cm = arr(E, day, m, 'pressure__commands')
        for c_ in cm:
            k, s, kind, act, why = int(round(c_[0] / STEP)), int(c_[1]), KI[int(c_[2])], AI[int(c_[7])], RS[int(c_[8])]
            onk = bool(ON[s][k]); n_on_cmds += int(onk)
            if kind == 'live' and act == 'KEEP' and why != 'fallback' and onk:
                cb += 1
            if kind == 'slot' and (act == 'POST' or (act == 'WAIT' and why in ('cap', 'near_end'))) and onk:
                cb += 1
            if why in ('pressure_pull', 'pressure_withhold') and not onk:
                cb += 1
        rq = collections.Counter(e[2] for e in r['pressure']['requests']); rb += sum(1 for v in rq.values() if v > 1)
        cd = arr(E, day, m, 'disabled__commands'); db += int(any(RS[int(c_[8])].startswith('pressure') for c_ in cd)) + len(r['disabled']['guard_log'])
    return dict(D_log=lb == 0, D_commands=cb == 0, D_requests=rb == 0, D_disabled_inert=db == 0), dict(D_log_bad=lb, D_command_escapes=cb, D_repeat_requests=rb, D_disabled_bad=db,
                                                                                                    D_commands_at_on_ticks=n_on_cmds)


def fam_E(E):
    bad = 0
    for r in E['recs']:
        for arm in ('disabled', 'pressure'):
            L_ = r[arm]; cash = -sum(SG[int(f[2])] * float(f[3]) * float(f[4]) for f in L_['fills']); pos = sum(SG[int(f[2])] * float(f[3]) for f in L_['fills'])
            bad += int(not (close(cash, L_['cash']) and close(pos, L_['position']) and close(cash + pos * L_['mark'], L_['value'])))
    return dict(E_accounting=bad == 0), {}


def recompute(rec, end):
    """This checker's arm metrics from one ledger (independent of the module's arm_metrics)."""
    mark = float(rec['mark']); rung = {int(o['gid']): int(round(float(o['rung']))) for o in rec['orders']}; out = collections.defaultdict(float)
    out['value'] = float(rec['value']); out['fills'] = float(len(rec['fills']))
    for f, mm in zip(rec['fills'], rec['fill_mids']):
        t, gid, s, q, p = float(f[0]), int(f[1]), int(f[2]), float(f[3]), float(f[4]); sg = SG[s]; e_ = sg * q * (mark - p)
        for nm in ('all', 'bid' if s == 0 else 'ask'):
            out[f'{nm}_q'] += q; out[f'{nm}_end'] += e_
            for h, i in (('edge', 0), ('mo50', 2), ('mo100', 4)):
                if mm[i] is None:
                    out[f'{nm}_{h}_censored'] += 1
                else:
                    out[f'{nm}_{h}'] += sg * q * (float(mm[i]) - p); out[f'{nm}_{h}_valid_q'] += q
            incut = t <= end - 1000.; el = incut and mm[6] is not None
            if incut:
                out[f'{nm}_cut_q'] += q; out[f'{nm}_cut_censored'] += float(mm[6] is None)
            if el:
                v = sg * q * (float(mm[6]) - p); out[f'{nm}_cut_mo1s'] += v; out[f'{nm}_cut_valid_q'] += q; out[f'{nm}_id_mo1s'] += v; out[f'{nm}_id_move'] += sg * q * (mark - float(mm[6]))
                out[f'{nm}_good_mo1s'] += max(v, 0.); out[f'{nm}_bad_mo1s'] += min(v, 0.)
            else:
                out[f'{nm}_id_late'] += e_
        out[f'rung{rung[gid]}_q'] += q; out[f'rung{rung[gid]}_end'] += e_
    for z in rec.get('guard_log', []):
        nm = 'bid' if int(z[1]) == 0 else 'ask'; out[f'{nm}_pulls'] += float(z[10] == 'pressure_pull'); out[f'{nm}_withholds'] += float(z[10] == 'pressure_withhold')
        out[f'{nm}_reducing_suppressed'] += float(z[8] < -1e-9 if int(z[1]) == 0 else z[8] > 1e-9)
    return out


RECON = [f'{nm}_{c}' for nm in ('all', 'bid', 'ask') for c in ('q', 'end', 'edge', 'edge_valid_q', 'edge_censored', 'mo50', 'mo50_valid_q', 'mo50_censored', 'mo100', 'mo100_valid_q',
                                                              'mo100_censored', 'cut_q', 'cut_mo1s', 'cut_valid_q', 'cut_censored', 'good_mo1s', 'bad_mo1s', 'id_mo1s', 'id_move', 'id_late')]
RECON += ['value', 'fills', 'rung0_q', 'rung0_end', 'rung1_q', 'rung1_end'] + [f'{nm}_{c}' for nm in ('bid', 'ask') for c in ('pulls', 'withholds', 'reducing_suppressed')]


def matched_split(fa, fb):
    key = lambda f: (round(float(f[0]), 6), int(f[2]), round(float(f[4]), 6), round(float(f[3]), 6)); rem = collections.Counter(map(key, fb)); out = []
    for f in fa:
        k = key(f); out.append(rem[k] <= 0); rem[k] -= 1 if rem[k] > 0 else 0
    return out


def selection_recompute(S, gm, edges_raw):
    nl = gm['null']; last = {s: float(edges_raw[f'live|{s}|adv_bn_r250'][-1]) for s in (0, 1)}; g = S[S.kind == 'live'].copy()
    band = lambda x, e: 'missing' if not np.isfinite(x) else int(np.searchsorted(np.asarray(e, float), float(x), side='right'))
    sign = lambda x: 'missing' if not np.isfinite(x) else (-1 if x < -1e-9 else (1 if x > 1e-9 else 0))
    g['sel'] = [on(float(np.float32(v)), last[int(s)]) if np.isfinite(v) else False for v, s in zip(g.adv_bn_r250.astype(float), g.side.astype(int))]
    b200 = ['missing' if not np.isfinite(v) else int(float(np.float32(v)) >= nl['b_edge']) for v in g.ctx_bid_reserved_shares.astype(float)]
    g['st'] = [json.dumps([d, int(m), int(s), band(r, nl['rem_bands']), band(a, nl['age_bands_ms']), sign(i), b]) for d, m, s, r, a, i, b in
               zip(g.day, g.m, g.side, g.fp_remaining.astype(float), g.ctx_generation_age_ms.astype(float), g.inv_known.astype(float), b200)]
    g['unc'] = (~g.censored_1s.astype(bool)) & (g.win_mo1s.notna() | (g.win_n == 0)); g['y'] = g.win_mo1s.fillna(0.).astype(float); u = g[g.unc]; s_ = u[u.sel]
    strata = {k: (int(v.sel.sum()), int(len(v)), v.index.values) for k, v in u.groupby('st')}; rng = np.random.default_rng(nl['seed']); null = []
    for _ in range(int(nl['draws'])):
        pick = [i for k, (ns, n, ix) in strata.items() if ns > 0 for i in rng.choice(ix, size=ns, replace=False)]; null.append(float(u.loc[pick, 'y'].mean()))
    J = float(s_.y.mean()); mv = [(ns, n) for ns, n, _ in strata.values() if 0 < ns < n]; p05 = float(np.quantile(null, .05))
    thin = len(mv) < nl['min_movable_strata'] or sum(ns for ns, n in mv) < nl['min_movable_rows']
    ans = 'INCONCLUSIVE' if thin else ('SELECTION_BEYOND_EXPOSURE' if J < p05 else 'NO_SELECTION_BEYOND_EXPOSURE')
    return dict(answer=ans, selected=int(len(s_)), states=int(len(u)), J=J, null_mean=float(np.mean(null)), null_p05=p05, movable_strata=len(mv),
                valid_window_q=float(s_.win_q.sum()), total_window_q=float(g[g.sel].win_q.sum()), excluded_window_q=float(g[g.sel & ~g.unc].win_q.sum()))


def fam_F(E):
    A_ = E['arms']; c, det = {}, {}; rb = ib = 0; rc = {}
    for r in E['recs']:
        end = float(r['path_info']['end_rel'])
        for arm in ('disabled', 'pressure'):
            x = recompute(r[arm], end); rc[(r['_day'], int(r['m']), arm)] = x; row = A_[(A_.day == r['_day']) & (A_.m == int(r['m'])) & (A_.arm == arm)]
            rb += int(len(row) != 1 or not all(close(row[k].iloc[0], x[k]) for k in RECON))
            ib += int(not (close(x['all_id_mo1s'] + x['all_id_move'] + x['all_id_late'], x['all_end']) and close(x['all_end'], x['value'])))
    c['F_reconcile'] = rb == 0; det['F_reconcile_bad'] = rb
    lb = 0; LA = E['la']
    for r in E['recs']:
        fb, fg = r['disabled']['fills'], r['pressure']['fills']; mark_b, mark_g = float(r['disabled']['mark']), float(r['pressure']['mark'])
        lo = [f for f, u in zip(fb, matched_split(fb, fg)) if u]; ad = [f for f, u in zip(fg, matched_split(fg, fb)) if u]
        en = lambda F_, mk: sum(SG[int(f[2])] * float(f[3]) * (mk - float(f[4])) for f in F_); row = LA[(LA.day == r['_day']) & (LA.m == int(r['m']))]
        lb += int(len(row) != 1 or not (int(row.lost_n.iloc[0]) == len(lo) and int(row.added_n.iloc[0]) == len(ad) and close(row.lost_end.iloc[0], en(lo, mark_b))
                                         and close(row.added_end.iloc[0], en(ad, mark_g)) and close(float(r['pressure']['value']) - float(r['disabled']['value']), en(ad, mark_g) - en(lo, mark_b))))
    c['F_identity'] = ib == 0 and lb == 0; det['F_identity_bad'] = ib + lb
    sr = selection_recompute(E['states'], E['gm'], E['edges_raw']); sel = E['selection']; det['F_selection'] = sr
    c['F_selection'] = (sr['answer'] == sel['answer'] and sel['seed'] == E['gm']['null']['seed'] and sel['draws'] == E['gm']['null']['draws'] and sr['selected'] == sel['selected']
                        and sr['states'] == sel['states'] and close(sr['J'], sel['J']) and close(sr['null_p05'], sel['null_p05']) and close(sr['null_mean'], sel['null_mean'])
                        and sr['movable_strata'] == sel['movable_strata'] and all(close(sr[k], sel[k]) for k in ('valid_window_q', 'total_window_q', 'excluded_window_q')))
    days = sorted({r['_day'] for r in E['recs']}); tot = lambda day, arm, k: sum(v[k] for (d, m, a), v in rc.items() if d == day and a == arm)
    cat = lambda v: 'IMPROVES_EVERY_DAY' if all(x > 0 for x in v) else 'WORSENS_EVERY_DAY' if all(x < 0 for x in v) else 'MIXED'
    dEND = {d: tot(d, 'pressure', 'value') - tot(d, 'disabled', 'value') for d in days}
    per = lambda d, arm, side: tot(d, arm, f'{side}_cut_mo1s') / tot(d, arm, f'{side}_cut_valid_q')
    dps = {side: {d: per(d, 'pressure', side) - per(d, 'disabled', side) for d in days} for side in ('all', 'bid', 'ask')}
    dq = {d: tot(d, 'pressure', 'all_q') - tot(d, 'disabled', 'all_q') for d in days}; pr = E['maudit']['predicates']
    better = cat(list(dEND.values())) == 'IMPROVES_EVERY_DAY' and cat(list(dps['all'].values())) == 'IMPROVES_EVERY_DAY' and sr['answer'] == 'SELECTION_BEYOND_EXPOSURE'
    less = all(v < 0 for v in dq.values()) and cat(list(dps['all'].values())) != 'IMPROVES_EVERY_DAY'
    c['F_readings'] = (pr['END_paired']['answer'] == cat(list(dEND.values())) and all(close(pr['END_paired']['by_day'][d], dEND[d]) for d in days)
                       and all(pr['mo1s_per_share'][s]['answer'] == cat(list(dps[s].values())) and all(close(pr['mo1s_per_share'][s]['by_day'][d], dps[s][d]) for d in days) for s in dps)
                       and all(close(pr['participation']['filled_q_delta_by_day'][d], dq[d]) for d in days) and pr['selection'] == sr['answer']
                       and pr['interpretation'] == ('better_selection' if better else 'quoting_less' if less else 'mixed'))
    det['F_readings'] = dict(END_by_day=dEND, per_share_all=dps['all'], interpretation=('better_selection' if better else 'quoting_less' if less else 'mixed'))
    return c, det


def fam_G(E):
    last = {s: float(E['gm']['edges'][str(s)][-1]) for s in (0, 1)}; pb = sb = 0; caches = collections.defaultdict(dict)
    for r in E['recs']:
        day, m = r['_day'], int(r['m']); xs = {s: arr(E, day, m, f'cache__x{s}') for s in (0, 1)}; st = {s: arr(E, day, m, f'cache__status{s}') for s in (0, 1)}; caches[day][m] = (xs, st)
        f = E['feats'][day]; f = f[(f.m == m) & (f.status == 'ok')]; n = eq = 0
        for s, t, v in zip(f.side, f.t, f.adv_bn_r250):
            k = int(round(float(t) / STEP)); n += 1; eq += int(abs(k * STEP - float(t)) <= 1e-6 and 0 <= k < len(xs[int(s)]) and st[int(s)][k] == 0 and bits(xs[int(s)][k]) == bits(v))
        pb += int(n != eq or r['cache']['parity']['rows'] != n or r['cache']['parity']['equal'] != eq)
        both = (st[0] == 0) & (st[1] == 0); sb += int(((xs[0][both] != -xs[1][both]).sum()) != 0)
    cb = 0; n_c = int(E['gm']['causal']['n_per_date'])
    for day, d_ in E['days'].items():
        cand = sorted((hashlib.sha256(f"{E['gm']['causal']['prefix']}{day}|{m}|{s}|{k}".encode()).hexdigest(), m, s, k) for m, (xs, st) in caches[day].items()
                      for s in (0, 1) for k in np.nonzero(st[s] == 0)[0].tolist())[:n_c]
        got = d_['causal']; cb += int(len(got) != n_c or [(c['rank'], int(c['m']), int(c['s']), int(c['k'])) for c in got] != [(a, int(b), int(c_), int(k)) for a, b, c_, k in cand])
        for c in got:
            xs, st = caches[day][int(c['m'])]
            cb += int(not (c['status'] == 'ok' and c['same'] and c['same_signal'] and c['fired'] and c['cached_equal'] and bits(c['x']) == bits(xs[int(c['s'])][int(c['k'])])
                           and c['on'] == on(float(c['x']), last[int(c['s'])])))
    return dict(G_parity=pb == 0, G_sign=sb == 0, G_causal=cb == 0), dict(G_parity_bad=pb, G_sign_bad=sb, G_causal_bad=cb)


def fam_H(E):
    bad = 0
    for r in E['recs']:
        for arm in ('disabled', 'pressure'):
            L_ = r[arm]; qv = L_['queue_venue']; cm = arr(E, r['_day'], int(r['m']), f'{arm}__commands')
            bad += sum(1 for o in L_['orders'] if not o['refused'] and str(o['gid']) not in qv['ahead0'])
            bad += sum(1 for z in L_['guard_log'] if z[10] == 'pressure_pull' and str(int(z[3])) not in qv['pull'])
            rev = {int(c_[3]) for c_ in cm if int(c_[2]) == M.CMD_KINDS.index('live') and int(c_[3]) >= 0 and P.REASONS[int(c_[8])] != 'fallback'}
            bad += sum(1 for g in rev if str(g) not in qv['first'])
    return dict(H_queue=bad == 0), dict(H_queue_bad=bad)


def fam_I(E, root=None):
    cpu, peak = 0., 0
    for p in glob.glob(str((root or R) / 'logs' / '**' / '*.log'), recursive=True):
        for ln in Path(p).read_text().splitlines():
            m_ = RL.match(ln)
            if m_:
                cpu += float(m_.group(6)); peak = max(peak, int(m_.group(4)))
    return dict(I_resources=cpu / 3600. <= .75 and peak < 1536), dict(I=dict(core_hours=cpu / 3600., peak_mib=peak))


FAMS = dict(A=fam_A, B=fam_B, C=fam_C, D=fam_D, E=fam_E, F=fam_F, G=fam_G, H=fam_H, I=fam_I)


def run_checks(E, fams=None):
    chk, det = {}, {}
    for k in (fams or FAMS):
        c, d = FAMS[k](E); chk.update(c); det.update(d)
    return dict(checks=chk, details=det, all_ok=bool(all(chk.values())))


# ---------------------------------------------------------------------------------------------------------------------- falsifier
def _first(E, pred):
    return next(r for r in E['recs'] if pred(r))


def _coordinated(E):
    """Review 2c5dbd53's case: a pressure-arm all_cut_mo1s moved by +10,000 and the stored per-share reading moved with it."""
    A_ = E['arms'].copy(); i = A_.index[A_.arm == 'pressure'][0]; A_.loc[i, 'all_cut_mo1s'] += 10000.; E['arms'] = A_
    g, b = A_[A_.arm == 'pressure'], A_[A_.arm == 'disabled']; pr = copy.deepcopy(E['maudit']['predicates'])
    for d in pr['mo1s_per_share']['all']['by_day']:
        pr['mo1s_per_share']['all']['by_day'][d] = float(g[g.day == d].all_cut_mo1s.sum() / g[g.day == d].all_cut_valid_q.sum() - b[b.day == d].all_cut_mo1s.sum() / b[b.day == d].all_cut_valid_q.sum())
    E['maudit'] = dict(E['maudit'], predicates=pr)


def _keep_escape(E):
    r = _first(E, lambda r: any(z[10] == 'pressure_pull' for z in r['pressure']['guard_log'])); key = f"{int(r['m'])}__pressure__commands"; day = r['_day']
    cm = E['arr'][day][key].copy(); i = int(np.nonzero(cm[:, 8] == P.REASONS.index('pressure_pull'))[0][0]); cm[i, 7] = M.CMD_ACTIONS.index('KEEP'); cm[i, 8] = P.REASONS.index('keep')
    E['arr'] = dict(E['arr']); E['arr'][day] = dict(E['arr'][day]); E['arr'][day][key] = cm


def _parity_bits(E):
    r = E['recs'][0]; day, m = r['_day'], int(r['m']); f = E['feats'][day]; f = f[(f.m == m) & (f.status == 'ok')].iloc[0]; key = f'{m}__cache__x{int(f.side)}'
    x = E['arr'][day][key].copy(); k = int(round(float(f.t) / STEP)); x[k] = np.float32(x[k] + 1.); E['arr'] = dict(E['arr']); E['arr'][day] = dict(E['arr'][day]); E['arr'][day][key] = x


def _rec(E, i, fn):
    E['recs'] = list(E['recs']); E['recs'][i] = copy.deepcopy(E['recs'][i]); fn(E['recs'][i])


def _log_x(E):
    i = next(j for j, r in enumerate(E['recs']) if r['pressure']['guard_log']); _rec(E, i, lambda r: r['pressure']['guard_log'][0].__setitem__(5, 0.1))


def _pull_missing(E):
    i = next(j for j, r in enumerate(E['recs']) if r['pressure']['queue_venue']['pull'])
    _rec(E, i, lambda r: r['pressure']['queue_venue']['pull'].pop(next(iter(r['pressure']['queue_venue']['pull']))))


def _fake_logs(E):
    E['_root'] = Path(tempfile.mkdtemp(dir=str(D))); (E['_root'] / 'logs').mkdir()
    (E['_root'] / 'logs' / 'fake.log').write_text('2026-11-21T00:00:00Z tag=fake rc=0 peak_rss_mib=2000 wall_s=3000.0 cpu_user_s=3000.0\n')


CASES = [
    ('coordinated_markout_and_reading', 'F', 'F_reconcile', _coordinated),
    ('logged_x_below_the_edge', 'D', 'D_log', _log_x),
    ('a_keep_at_an_on_tick', 'D', 'D_commands', _keep_escape),
    ('disabled_fill_dropped', 'C', 'C_disabled_equals_baseline', lambda E: _rec(E, 0, lambda r: r['disabled'].__setitem__('fills', r['disabled']['fills'][:-1]))),
    ('cash_changed', 'E', 'E_accounting', lambda E: _rec(E, 0, lambda r: r['pressure'].__setitem__('cash', r['pressure']['cash'] + 1.))),
    ('lost_end_changed', 'F', 'F_identity', lambda E: E.__setitem__('la', E['la'].assign(lost_end=E['la'].lost_end + np.where(E['la'].index == 0, 1., 0.)))),
    ('selection_seed_changed', 'F', 'F_selection', lambda E: E.__setitem__('selection', dict(E['selection'], seed=E['selection']['seed'] + 1))),
    ('stored_interpretation_changed', 'F', 'F_readings', lambda E: E.__setitem__('maudit', dict(E['maudit'], predicates=dict(E['maudit']['predicates'], interpretation='better_selection'
                                                                                                                              if E['maudit']['predicates']['interpretation'] != 'better_selection' else 'mixed')))),
    ('a_cache_value_changed', 'G', 'G_parity', _parity_bits),
    ('a_causal_entry_not_fired', 'G', 'G_causal', lambda E: E.__setitem__('days', dict(E['days'], **{sorted(E['days'])[0]: dict(E['days'][sorted(E['days'])[0]], causal=[
        dict(c, fired=False) if j == 0 else c for j, c in enumerate(E['days'][sorted(E['days'])[0]]['causal'])])}))),
    ('a_pull_without_its_queue_entry', 'H', 'H_queue', _pull_missing),
    ('a_member_dropped', 'B', 'B_membership', lambda E: E.__setitem__('recs', E['recs'][:-1])),
    ('a_record_rule_changed', 'A', 'A_rule', lambda E: E.__setitem__('days', dict(E['days'], **{sorted(E['days'])[0]: dict(E['days'][sorted(E['days'])[0]], rule_sha256='0' * 64)}))),
    ('resources_over_the_cap', 'I', 'I_resources', _fake_logs),
]


def falsify(E, base):
    out = {}
    for name, fam, target, fn in CASES:
        X_ = dict(E); fn(X_)
        try:
            c, _ = FAMS[fam](X_, X_['_root']) if fam == 'I' else FAMS[fam](X_)
        finally:
            if '_root' in X_:
                import shutil
                shutil.rmtree(X_['_root'], ignore_errors=True)
        out[name] = dict(family=fam, target=target, clean=bool(base['checks'][target]), refused=not c[target], flipped=sorted(k for k, v in c.items() if base['checks'].get(k) and not v))
    return dict(cases=out, falsifier_ok=bool(base['all_ok'] and all(v['clean'] and v['refused'] for v in out.values())))


def main():
    global D
    rehearse = len(sys.argv) > 2 and sys.argv[1] == 'rehearse'
    if rehearse:                                                                           # the pre-run rehearsal on the pilot copy: code paths, not acceptance
        D = Path(sys.argv[2])
    E = load(); res = run_checks(E); dest = D if rehearse else R; tag = '_rehearsal' if rehearse else ''
    (dest / f'checks{tag}.json').write_text(json.dumps(res, indent=1, default=str) + '\n'); fz = falsify(E, res)
    (dest / f'checks_falsifier{tag}.json').write_text(json.dumps(fz, indent=1) + '\n')
    if rehearse:
        print(json.dumps(dict(checks=res['checks'], cases={k: (v['clean'], v['refused']) for k, v in fz['cases'].items()}), indent=1)); return
    print(json.dumps(dict(all_ok=res['all_ok'], failed=[k for k, v in res['checks'].items() if not v], falsifier_ok=fz['falsifier_ok'],
                          cases={k: (v['refused'], v['flipped']) for k, v in fz['cases'].items()}, details=res['details']), indent=1, default=str)[:5000])


if __name__ == '__main__':
    main()
