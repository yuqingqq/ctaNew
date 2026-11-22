"""INDEPENDENT CHECKS of ASYNC-EXECUTION-SENSITIVITY-v1 (declaration.md; session aea5bfb9, 2026-11-22). Recomputed from the stored evidence with this
checker's own code (the module is not imported); recomputation happens once, the falsifier re-runs the comparisons.
 A identity: the frozen files committed; the parent pressure records (= the committed copies, gzip / npz bytes); the reused QUEUE-PRIORITY-
   SENSITIVITY-v1 records (= this manifest's sha256, gzip / npz = their records); this run's records; the measurement hashes;
 B membership: the 19 complete markets once each (pilot = the first two per date); the reused baselines cover the same 19;
 K parity: PARITY_BASELINE = the accepted pressure arm on every pilot market (fingerprint, fills, orders, requests, value, guard log, queue
   records, command array) - compared here, and the module's parity flags all true;
 S settings: each ASYNC record's settings and profile = activation 10 / live cancel 10 / pending 20 / desired_levels, the notices 100 / 100 / 0;
 E execution contract, per arm from its saved orders, fills and closures: activation = submit + the profile's placement; every removal = the
   profile's rule; at every POST the desired (ASYNC) or unclosed (BASELINE) same-side count < 2, no same-side unclosed order at that price, the
   size within the reservation room (cap 200 minus the notified position and the known remaining of every unclosed same-side order), no
   opposite unclosed order at a crossing price, activation before END;
 T treatment: FRONT_ASYNC's every accepted generation at ahead 0, front_set = accepted_new; CURRENT_ASYNC untreated;
 F fills: classes recomputed; own fills <= each public trade; M measurement: END, eligible +1 s, shares, the class table, the blocks table and
   the treatment census recomputed; P readings: the contrasts and the reading recomputed; I resources: batch <= 0.25, line <= 2.50, peak.
falsify(): eleven corruptions, each refused while the clean evidence passes. Usage: fa_checks.py [out]"""
import collections
import glob
import gzip
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path('/home/yuqing/ctaNew'); R = REPO / 'predictfun/reviews/ladder_async_fast_20261122'; D = REPO / 'data/predict_fun/research/ladder_async_fast_v1_20261122'
QD = REPO / 'data/predict_fun/research/ladder_queue_front_v1_20261122'; PRS = 'predictfun/reviews/ladder_pressure_20261121/outputs/days'
PDIR = REPO / 'data/predict_fun/research/ladder_pressure_v1_20261121/days'; SG = {0: 1., 1: -1.}; CAP, LEVELS, STEP = 200., 2, 50.
NEW = ('CURRENT_ASYNC_FAST10', 'FRONT_ASYNC_FAST10'); REUSE = dict(CURRENT_BASELINE='CURRENT_QUEUE', FRONT_BASELINE='FRONT_AT_ACTIVATION')
MEASURED = ('CURRENT_BASELINE', 'FRONT_BASELINE', 'CURRENT_ASYNC_FAST10', 'FRONT_ASYNC_FAST10')
PROF = dict(BASELINE=dict(place=58., cancel=37., pending=95., model='frozen'), ASYNC_FAST10=dict(place=10., cancel=10., pending=20., model='desired_levels'))
ARM_PROF = dict(CURRENT_BASELINE='BASELINE', FRONT_BASELINE='BASELINE', PARITY_BASELINE='BASELINE', CURRENT_ASYNC_FAST10='ASYNC_FAST10', FRONT_ASYNC_FAST10='ASYNC_FAST10')
COMMITTED = ['predictfun/research/pf_ladder_async_fast.py', 'predictfun/research/pf_ladder_async_fast_selftest.py', 'predictfun/research/pf_ladder_async_fast_mutants.py'] + \
    [f'predictfun/reviews/ladder_async_fast_20261122/{n}' for n in ('declaration.md', 'manifest.json', 'launch_async_fast.sh', 'fa_checks.py')]
CON = dict(EXEC_AT_CURRENT=(('CURRENT_ASYNC_FAST10', 1.), ('CURRENT_BASELINE', -1.)), EXEC_AT_FRONT=(('FRONT_ASYNC_FAST10', 1.), ('FRONT_BASELINE', -1.)),
           QUEUE_AT_FAST=(('FRONT_ASYNC_FAST10', 1.), ('CURRENT_ASYNC_FAST10', -1.)),
           INTERACTION=(('FRONT_ASYNC_FAST10', 1.), ('CURRENT_ASYNC_FAST10', -1.), ('FRONT_BASELINE', -1.), ('CURRENT_BASELINE', 1.)))
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')


def sha(b):
    return hashlib.sha256(b).hexdigest()


def blob(rel):
    r = subprocess.run(['git', '-C', str(REPO), 'show', f'HEAD:{rel}'], capture_output=True); return r.stdout if r.returncode == 0 else None


def rows(b):
    return [json.loads(x) for x in gzip.decompress(b).decode().splitlines() if x]


def close(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol * max(1., abs(float(a)), abs(float(b)))


def load(out=None):
    out = Path(out or D); gm = json.loads((R / 'manifest.json').read_text()); E = dict(out=out, gm=gm, src={}, runs={}, reused={}, parent={}, pz={}, z={})
    for d in gm['days']:
        rb = blob(f'{PRS}/{d}.json'); prec = json.loads(rb); pg, pn = (PDIR / f'{d}.jsonl.gz').read_bytes(), (PDIR / f'{d}_arrays.npz').read_bytes()
        E['src'][f'parent_{d}'] = bool((PDIR / f'{d}.json').read_bytes() == rb and sha(rb) == gm['parent']['day_records'][d] and sha(pg) == prec['sha256']['gz'] and sha(pn) == prec['sha256']['npz'])
        E['parent'].update({(d, int(r['m'])): r for r in rows(pg)}); E['pz'][d] = np.load(PDIR / f'{d}_arrays.npz')
        for part in ('pilot', 'rest'):
            q_ = QD / part / f'{d}.json'
            if f'{part}/{d}' in gm['reused_records']:
                qb = q_.read_bytes(); qr = json.loads(qb); qg = (QD / part / f'{d}.jsonl.gz').read_bytes(); qn = (QD / part / f'{d}_arrays.npz').read_bytes()
                E['src'][f'reused_{part}_{d}'] = bool(sha(qb) == gm['reused_records'][f'{part}/{d}'] and sha(qg) == qr['sha256']['gz'] and sha(qn) == qr['sha256']['npz'])
                E['reused'].update({(d, int(r['m'])): r for r in rows(qg)})
            p_ = out / part / f'{d}.json'
            if p_.exists():
                rec = json.loads(p_.read_text()); g_ = (out / part / f'{d}.jsonl.gz').read_bytes(); n_ = (out / part / f'{d}_arrays.npz').read_bytes()
                E['src'][f'run_{part}_{d}'] = bool(sha(g_) == rec['sha256']['gz'] and sha(n_) == rec['sha256']['npz'] and rec['profile_sha256'] == gm['profile_sha256'])
                for r in rows(g_):
                    E['runs'].setdefault((d, int(r['m'])), []).append(dict(r, part=part))
                E['z'][(d, part)] = np.load(out / part / f'{d}_arrays.npz')
    md = out / 'measure'; E['audit'] = json.loads((md / 'audit.json').read_text())
    for n in ('arms', 'parity', 'classes', 'treatment', 'blocks'):
        E[n] = pd.read_csv(md / f'{n}.csv', dtype={'day': str}, float_precision='round_trip')
    return E


def arm_recs(E, k):
    r, q = E['runs'][k][0], E['reused'][k]
    return dict(CURRENT_BASELINE=q['CURRENT_QUEUE'], FRONT_BASELINE=q['FRONT_AT_ACTIVATION'], CURRENT_ASYNC_FAST10=r['CURRENT_ASYNC_FAST10'], FRONT_ASYNC_FAST10=r['FRONT_ASYNC_FAST10'])


# ---------------------------------------------------------------------------------------------------------------------- this checker's quantities
def contract_errors(rec, prof, end):
    """The execution contract of one arm from its saved evidence (header) -> a list of violations."""
    pr = PROF[prof]; bad = []; od = sorted(rec['orders'], key=lambda o: int(o['gid'])); closed = {int(k): float(v[1]) for k, v in rec['closed'].items()}
    fl = collections.defaultdict(list)
    for f in rec['fills']:
        fl[int(f[1])].append((float(f[0]), float(f[3]), int(f[2])))
    known = lambda gid, g: sum(q for t, q, _ in fl[gid] if t + 100. <= g)
    allf = sorted((float(f[0]), SG[int(f[2])] * float(f[3])) for f in rec['fills'])
    for o in od:
        sub, act = float(o['submit']), float(o['act'])
        if not math.isclose(act, sub + pr['place'], abs_tol=1e-9):
            bad.append(f'activation {o["gid"]}')
        if o['req'] is not None:
            rq = float(o['req']); want = max(rq + pr['cancel'], sub + pr['pending']) if rq < act else rq + pr['cancel']
            if not math.isclose(float(o['remove']), want, abs_tol=1e-9):
                bad.append(f'removal {o["gid"]}')
    for j, o in enumerate(od):
        g, s, gid = float(o['submit']), int(o['side']), int(o['gid'])
        before = [x for x in od[:j] if float(x['submit']) <= g and closed.get(int(x['gid']), np.inf) > g]
        same = [x for x in before if int(x['side']) == s]; opp = [x for x in before if int(x['side']) != s]
        n_slot = len([x for x in same if (x['req'] is None or float(x['req']) > g)]) if pr['model'] == 'desired_levels' else len(same)
        I = sum(v for t, v in allf if t + 100. <= g); res = sum(float(x['size']) - known(int(x['gid']), g) for x in same)
        room = (CAP - I - res) if s == 0 else (CAP + I - res)
        if n_slot >= LEVELS:
            bad.append(f'slot {gid}')
        if any(float(x['price']) == float(o['price']) for x in same):
            bad.append(f'duplicate {gid}')
        if float(o['size']) > room + 1e-9:
            bad.append(f'room {gid}')
        if any((float(o['price']) >= float(x['price']) if s == 0 else float(o['price']) <= float(x['price'])) for x in opp):
            bad.append(f'own_cross {gid}')
        if g + pr['place'] >= end:
            bad.append(f'near_end {gid}')
    return bad


def end_value(rec):
    m = float(rec['mark']); return sum(SG[int(f[2])] * float(f[3]) * (m - float(f[4])) for f in rec['fills'])


def el_1s(rec, end):
    v = q = 0.
    for f, mm in zip(rec['fills'], rec['fill_mids']):
        if float(f[0]) <= end - 1000. and mm[6] is not None:
            v += SG[int(f[2])] * float(f[3]) * (float(mm[6]) - float(f[4])); q += float(f[3])
    return v, q


def my_blocks(rec, end):
    nt = int(math.ceil(end / STEP)); g = STEP * np.arange(nt); des = np.zeros((nt, 2)); ret = np.zeros((nt, 2)); closed = {int(k): float(v[1]) for k, v in rec['closed'].items()}
    for o in rec['orders']:
        s, sub, ct = int(o['side']), float(o['submit']), closed.get(int(o['gid']), np.inf); un = (g >= sub) & (g < ct)
        rt = un & (g >= float(o['req'])) if o['req'] is not None else np.zeros(nt, bool); des[:, s] += un & ~rt; ret[:, s] += rt
    cap_ret = sum(1 for t, s, w in rec['constraints'] if w == 'cap' and 0 <= int(round(float(t) / STEP)) < nt and ret[int(round(float(t) / STEP)), int(s)] > 0)
    return dict(desired_occupancy=float(np.minimum(des, LEVELS).sum() / (2 * LEVELS * nt)), retiring_ticks=float(ret.sum()), cap_while_retiring=cap_ret,
                **{w: sum(1 for c in rec['constraints'] if c[2] == w) for w in ('cap', 'near_end', 'own_cross_blocked', 'unavailable_price')})


def classify(vals):
    vals = list(vals); return 'POSITIVE_EVERY_DAY' if all(v > 0 for v in vals) else 'NEGATIVE_EVERY_DAY' if all(v < 0 for v in vals) else 'MIXED'


def mine(E):
    t, cl, bl = [], {}, {}
    for (d, m), lst in E['runs'].items():
        end = float(lst[0]['path_info']['end_rel']); ar = dict(BASELINE=E['parent'][(d, m)]['disabled'], **arm_recs(E, (d, m)))
        for arm, rec in ar.items():
            v1, q1 = el_1s(rec, end); t.append(dict(day=d, m=m, arm=arm, END=end_value(rec), mo1s=v1, el_q=q1, q=sum(float(f[3]) for f in rec['fills'])))
            if arm != 'BASELINE':
                bl[(d, m, arm)] = my_blocks(rec, end)
    T_ = pd.DataFrame(t); days = sorted(T_.day.unique()); tot = lambda a, d=None: T_[(T_.arm == a) & ((T_.day == d) if d else True)]; con = {}
    for nm, terms in CON.items():
        dE = {d: sum(w * tot(a, d).END.sum() for a, w in terms) for d in days}; d1 = {d: sum(w * tot(a, d).mo1s.sum() for a, w in terms) for d in days}
        eA, mA = classify(dE.values()), classify(d1.values())
        con[nm] = dict(END_answer=eA, mo1s_answer=mA, economic='IMPROVES' if eA == mA == 'POSITIVE_EVERY_DAY' else 'WORSENS' if eA == mA == 'NEGATIVE_EVERY_DAY' else 'MIXED',
                       END_total=sum(w * tot(a).END.sum() for a, w in terms), mo1s_total=sum(w * tot(a).mo1s.sum() for a, w in terms))
    nonneg = [a for a in MEASURED if tot(a).END.sum() >= 0]
    return dict(T=T_, blocks=bl, contrasts=con, reading='nonnegative_development_result' if nonneg else 'assumptions_do_not_rescue', nonneg=nonneg)


# ---------------------------------------------------------------------------------------------------------------------- families
def fam_A(E):
    c = dict(A_committed=all(blob(p) == (REPO / p).read_bytes() for p in COMMITTED), A_inputs=bool(E['src'] and all(E['src'].values())))
    md = E['out'] / 'measure'; c['A_outputs'] = all(E['audit']['outputs'][f'{n}.csv'] == sha((md / f'{n}.csv').read_bytes()) for n in ('arms', 'parity', 'classes', 'treatment', 'blocks'))
    return c, {}


def fam_B(E):
    gm = E['gm']; want = sorted((d, int(m)) for d in gm['days'] for m in gm['complete'][d]); got = sorted(E['runs'])
    placed = all(v[0]['part'] == ('pilot' if k[1] in gm['parts']['pilot'][k[0]] else 'rest') for k, v in E['runs'].items())
    return dict(B_membership=bool(got == want == sorted(E['reused']) and all(len(v) == 1 for v in E['runs'].values()) and placed and len(want) == 19)), dict(B=dict(markets=len(got)))


def fam_K(E):
    bad = []; pilots = [(d, int(m)) for d in E['gm']['days'] for m in E['gm']['parts']['pilot'][d]]
    for k in pilots:
        r = E['runs'].get(k, [None])[0]
        if r is None or 'PARITY_BASELINE' not in r:
            bad.append(f'{k} missing'); continue
        z, pp = r['PARITY_BASELINE'], E['parent'][k]['pressure']; za = E['z'][(k[0], 'pilot')][f'{k[1]}__PARITY_BASELINE__commands']; pa = E['pz'][k[0]][f'{k[1]}__pressure__commands']
        same = all(z[f] == pp[f] for f in ('fingerprint', 'fills', 'orders', 'requests', 'value', 'guard_log', 'queue_venue', 'queue_known')) and np.array_equal(za, pa, equal_nan=True)
        bad += [] if (same and r['parity_all'] and all(r['parity'].values())) else [f'{k}']
    return dict(K_parity=not bad and len(pilots) == 8), dict(K=dict(failed=bad))


def fam_S(E):
    bad = []
    for k, lst in E['runs'].items():
        for arm in NEW:
            st, pr = lst[0][arm]['settings'], lst[0][arm]['profile']
            ok_ = (st.get('activation_ms') == 10. and st.get('cancel_effect_ms') == 10. and st.get('pending_cancel_ms') == 20. and st.get('slot_model') == 'desired_levels'
                   and st.get('profile') == 'ASYNC_FAST10' and pr['name'] == 'ASYNC_FAST10' and pr['place'] == 10. and pr['cancel'] == 10. and pr['pending'] == 20.
                   and pr['fill_notice'] == 100. and pr['cancel_ack'] == 100. and pr['post_notice'] == 0.)
            bad += [] if ok_ else [f'{k} {arm}']
    return dict(S_settings=not bad), dict(S=dict(failed=bad))


_MEMO = {}


def contract_cached(rec, prof, end):
    key = (id(rec), prof, end)
    if key not in _MEMO:
        _MEMO[key] = (rec, contract_errors(rec, prof, end))                                # the record is kept alive with its result
    return _MEMO[key][1]


def fam_E(E):
    bad = []
    for k, lst in E['runs'].items():
        end = float(lst[0]['path_info']['end_rel'])
        for arm, rec in arm_recs(E, k).items():
            er = contract_cached(rec, ARM_PROF[arm], end); bad += [f'{k} {arm} {x}' for x in er[:3]]
    return dict(E_contract=not bad), dict(E=dict(failed=bad[:12], n=len(bad)))


def fam_T(E):
    bad = []
    for k, lst in E['runs'].items():
        for arm in NEW:
            rec = lst[0][arm]; acc = [v for v in rec['queue_venue']['ahead0'].values() if float(v[2]) > 0]; t = rec['treatment']
            ok_ = (all(float(v[1]) == 0. for v in acc) and t['front'] is True and t['front_set'] == t['accepted_new'] == len(acc)) if arm == 'FRONT_ASYNC_FAST10' else \
                (t['front'] is False and t['front_set'] == 0 and t['accepted_new'] == len(acc))
            bad += [] if ok_ else [f'{k} {arm}']
    return dict(T_treatment=not bad), dict(T=dict(failed=bad))


def fam_F(E):
    bad = []
    for k, lst in E['runs'].items():
        for arm in NEW:
            rec = lst[0][arm]; use = collections.Counter(); size = {}
            for f, c in zip(rec['fills'], rec['fill_class']):
                s, p, tp = int(f[2]), float(f[4]), float(c[1]); want = 'same_price' if tp == p else ('through' if (tp < p if s == 0 else tp > p) else 'invalid')
                bad += [] if (c[0] == want != 'invalid') else [f'{k} {arm} class']; use[int(f[5])] += float(f[3]); size[int(f[5])] = float(c[2])
            bad += [f'{k} {arm} conservation'] if (len(rec['fill_class']) != len(rec['fills']) or any(use[i] > size[i] + 1e-7 for i in use)) else []
    return dict(F_fills=not bad), dict(F=dict(failed=bad[:10]))


def fam_M(E):
    my = E['mine']; T_ = my['T']; bad = []
    J = T_.merge(E['arms'][['day', 'm', 'arm', 'value', 'all_id_mo1s', 'all_cut_valid_q', 'all_q']], on=['day', 'm', 'arm'], how='left')
    for r in J.itertuples(index=False):
        bad += [] if (close(r.END, r.value) and close(r.mo1s, r.all_id_mo1s) and close(r.el_q, r.all_cut_valid_q) and close(r.q, r.all_q)) else [f'arms {r.day} {r.m} {r.arm}']
    BL = E['blocks']
    for (d, m, arm), x in my['blocks'].items():
        g = BL[(BL.day == d) & (BL.m == m) & (BL.arm == arm)]; bad += [] if (len(g) == 1 and all(close(g.iloc[0][k], v) for k, v in x.items())) else [f'blocks {d} {m} {arm}']
    K_ = E['classes']
    for (d, m), lst in E['runs'].items():
        end = float(lst[0]['path_info']['end_rel'])
        for arm, rec in arm_recs(E, (d, m)).items():
            cnt = collections.Counter(int(f[5]) for f in rec['fills']); acc = collections.defaultdict(collections.Counter)
            for f, mm, c in zip(rec['fills'], rec['fill_mids'], rec['fill_class']):
                a = acc[c[0]]; a['fills'] += 1; a['q'] += float(f[3]); a['end'] += SG[int(f[2])] * float(f[3]) * (float(rec['mark']) - float(f[4])); a['shared_trade_fills'] += int(cnt[int(f[5])] > 1)
                if float(f[0]) <= end - 1000. and mm[6] is not None:
                    a['el_q'] += float(f[3]); a['mo1s'] += SG[int(f[2])] * float(f[3]) * (float(mm[6]) - float(f[4]))
            for cls in ('same_price', 'through'):
                g = K_[(K_.day == d) & (K_.m == m) & (K_.arm == arm) & (K_.cls == cls)]
                bad += [] if (len(g) == 1 and all(close(g.iloc[0][k], acc[cls].get(k, 0.)) for k in ('fills', 'q', 'end', 'el_q', 'mo1s', 'shared_trade_fills'))) else [f'class {d} {m} {arm} {cls}']
    return dict(M_measurement=not bad and len(J) == 5 * len(E['runs'])), dict(M=dict(failed=bad[:10]))


def fam_P(E):
    my, a = E['mine'], E['audit']['predicates']; bad = []
    for nm, x in my['contrasts'].items():
        y = a['contrasts'][nm]
        bad += [] if (y['END_answer'] == x['END_answer'] and y['mo1s_answer'] == x['mo1s_answer'] and y['economic'] == x['economic'] and close(y['END_total'], x['END_total'])
                      and close(y['mo1s_total'], x['mo1s_total'])) else [nm]
    bad += [] if (a['reading'] == my['reading'] and a['nonnegative_arms'] == my['nonneg']) else ['reading']
    return dict(P_readings=not bad), dict(P=dict(failed=bad))


def cpu(paths):
    tot, peak = 0., 0
    for p in paths:
        for ln in Path(p).read_text(errors='replace').splitlines():
            mm = RL.match(ln)
            if mm:
                tot += float(mm.group(6)); peak = max(peak, int(mm.group(4)))
    return tot / 3600., peak


def fam_I(E, root=None):
    gm = E['gm']; own, peak = cpu(glob.glob(str((root or R) / 'logs' / '**' / '*.log'), recursive=True))
    line, _ = cpu([p for n in gm['line_receipts'] for p in glob.glob(str(REPO / 'predictfun/reviews' / n / 'logs' / '**' / '*.log'), recursive=True)])
    rev, _ = cpu([p for n in gm['line_reviews'] for p in glob.glob(str(REPO / 'predictfun/reviews/automatic' / n / '*.log'))])
    return dict(I_resources=bool(own <= gm['caps']['batch_core_hours'] and peak < 1536 and own + line + rev <= gm['caps']['line_ceiling'])), \
        dict(I=dict(batch_core_hours=own, line_core_hours=own + line + rev, peak_mib=peak))


FAMS = dict(A=fam_A, B=fam_B, K=fam_K, S=fam_S, E=fam_E, T=fam_T, F=fam_F, M=fam_M, P=fam_P, I=fam_I)


def run_checks(E):
    chk, det = {}, {}
    for k in FAMS:
        c, d = FAMS[k](E); chk.update(c); det.update(d)
    return dict(checks=chk, details=det, all_ok=bool(all(chk.values())))


def _first(E, part='pilot'):
    return sorted(k for k, v in E['runs'].items() if v[0]['part'] == part)[0]


def _edit(E, fn, part='pilot'):
    k = _first(E, part); r = json.loads(json.dumps(E['runs'][k][0])); fn(r); runs = dict(E['runs']); runs[k] = [r]; E['runs'] = runs


def _fake(E):
    E['_root'] = Path(tempfile.mkdtemp(dir=str(E['out']))); (E['_root'] / 'logs').mkdir()
    (E['_root'] / 'logs' / 'fake.log').write_text('2026-11-22T00:00:00Z tag=fake rc=0 peak_rss_mib=100 wall_s=999.0 cpu_user_s=999.0\n')


def _dup(r):
    od = r['FRONT_ASYNC_FAST10']['orders']; a = next(o for o in od if o['side'] == 0); b = next(o for o in od if o['side'] == 0 and o['gid'] > a['gid'] and o['submit'] == a['submit'])
    b['price'] = a['price']


CASES = [
    ('an_output_hash_changed', 'A', 'A_outputs', lambda E: E.__setitem__('audit', dict(E['audit'], outputs=dict(E['audit']['outputs'], **{'arms.csv': '0' * 64})))),
    ('a_market_dropped', 'B', 'B_membership', lambda E: E.__setitem__('runs', {k: v for k, v in E['runs'].items() if k != _first(E)})),
    ('a_parity_fill_changed', 'K', 'K_parity', lambda E: _edit(E, lambda r: r['PARITY_BASELINE']['fills'][0].__setitem__(3, float(r['PARITY_BASELINE']['fills'][0][3]) + 1.))),
    ('an_async_setting_changed', 'S', 'S_settings', lambda E: _edit(E, lambda r: r['FRONT_ASYNC_FAST10']['settings'].__setitem__('activation_ms', 58.))),
    ('an_activation_shifted', 'E', 'E_contract', lambda E: _edit(E, lambda r: r['CURRENT_ASYNC_FAST10']['orders'][0].__setitem__('act', float(r['CURRENT_ASYNC_FAST10']['orders'][0]['submit']) + 58.))),
    ('a_size_over_the_room', 'E', 'E_contract', lambda E: _edit(E, lambda r: r['CURRENT_ASYNC_FAST10']['orders'][1].__setitem__('size', 400.))),
    ('a_duplicate_price', 'E', 'E_contract', lambda E: _edit(E, _dup)),
    ('a_front_generation_behind', 'T', 'T_treatment', lambda E: _edit(E, lambda r: next(v for v in r['FRONT_ASYNC_FAST10']['queue_venue']['ahead0'].values() if float(v[2]) > 0).__setitem__(1, 5.))),
    ('a_fill_class_flipped', 'F', 'F_fills', lambda E: _edit(E, lambda r: r['FRONT_ASYNC_FAST10']['fill_class'][0].__setitem__(0, 'through' if r['FRONT_ASYNC_FAST10']['fill_class'][0][0] == 'same_price'
                                                                                                                                     else 'same_price'))),
    ('an_arm_value_changed', 'M', 'M_measurement', lambda E: E.__setitem__('arms', E['arms'].assign(value=E['arms'].value + np.where(E['arms'].index == 0, 1., 0.)))),
    ('a_reading_changed', 'P', 'P_readings', lambda E: E.__setitem__('audit', dict(E['audit'], predicates=dict(E['audit']['predicates'], reading='nonnegative_development_result'
                                                                                                                if E['audit']['predicates']['reading'] != 'nonnegative_development_result' else 'assumptions_do_not_rescue')))),
    ('resources_over_the_cap', 'I', 'I_resources', _fake),
]


def falsify(E, base):
    out = {}
    for name, fam, target, fn in CASES:
        X_ = dict(E); fn(X_)
        try:
            c, _ = FAMS[fam](X_, X_['_root']) if fam == 'I' else FAMS[fam](X_)
        finally:
            if '_root' in X_:
                shutil.rmtree(X_['_root'], ignore_errors=True)
        out[name] = dict(family=fam, target=target, clean=bool(base['checks'][target]), refused=not c[target])
    return dict(cases=out, falsifier_ok=bool(base['all_ok'] and all(v['clean'] and v['refused'] for v in out.values())))


def main(argv):
    E = load(*(argv[:1] or [None])); E['mine'] = mine(E); r_ = run_checks(E); fz = falsify(E, r_)
    for n, obj in (('checks.json', r_), ('checks_falsifier.json', fz)):
        p = R / n; assert not p.exists(), f'{p} exists (write-once)'; p.write_text(json.dumps(obj, indent=1, default=str) + '\n')
    print(json.dumps(dict(all_ok=r_['all_ok'], failed=[k for k, v in r_['checks'].items() if not v], falsifier_ok=fz['falsifier_ok'],
                          cases={k: (v['clean'], v['refused']) for k, v in fz['cases'].items()}, details=r_['details']), indent=1, default=str)[:3500])


if __name__ == '__main__':
    main(sys.argv[1:])
