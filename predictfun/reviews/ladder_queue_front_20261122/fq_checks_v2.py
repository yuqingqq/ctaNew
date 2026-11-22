"""INDEPENDENT CHECKS v2 of QUEUE-PRIORITY-SENSITIVITY-v1 (declaration.md; session aea5bfb9, 2026-11-22), superseding fq_checks.py, whose falsifier helper
_edit crashed (dict(E['runs'], **{k: [r]}) with a tuple key: "keywords must be strings") before it wrote any verdict (logs/qf_checks.log). The only
change: _edit copies the dict and assigns the item; v2 is committed before it runs and writes checks_v2.json / checks_falsifier_v2.json. v1 text: Recomputed from the stored evidence (this round's
day records, gz rows and arrays; the pressure receipt's hash-bound records and arrays) with this checker's own code; the module is not imported.
 A identity: the frozen files committed; the parent day records (= the manifest's = the committed copies) and their gzip / npz bytes; this run's
   day records' gzip / npz bytes and treatment identity; the measurement outputs' hashes;
 B membership: the 19 complete markets each run once; the pilot = the first two complete markets of each date, the rest the others;
 K control: CURRENT_QUEUE = the accepted pressure arm on every market (fingerprint, fills, orders, requests, value, guard log, queue records,
   command array) - compared here, not read from the module's flags (which must also all be true);
 T treatment: FRONT's every accepted generation (remaining > 0) at ahead 0 after activation, front_set = accepted_new = the accepted count;
   CURRENT's front_set 0 and accepted_new = its accepted count;
 F fills: one class per fill, recomputed from the saved trade price / side; per public trade the own fills <= its size; no invalid class;
 M measurement: per arm and market the END (cash + position x mark from the fills = the record's value), the eligible +1 s total / shares and
   the filled shares against arms.csv; classes.csv recomputed; lost / added (the multiset convention) against lost_added.csv; treatment.csv;
 P readings: the declared predicates recomputed from this checker's totals and class table;
 I resources: this batch <= 0.25 core-hours, peak < 1,536 MiB, the line (its receipts, the reviews and this batch) <= 2.25.
falsify(): ten corruptions, each refused while the clean evidence passes. Usage: fq_checks.py [out]"""
import collections
import glob
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path('/home/yuqing/ctaNew'); R = REPO / 'predictfun/reviews/ladder_queue_front_20261122'; D = REPO / 'data/predict_fun/research/ladder_queue_front_v1_20261122'
PRS = 'predictfun/reviews/ladder_pressure_20261121/outputs/days'; PDIR = REPO / 'data/predict_fun/research/ladder_pressure_v1_20261121/days'
ARMS = ('CURRENT_QUEUE', 'FRONT_AT_ACTIVATION'); F_, C_ = 'FRONT_AT_ACTIVATION', 'CURRENT_QUEUE'; SG = {0: 1., 1: -1.}
COMMITTED = ['predictfun/research/pf_ladder_queue_front.py', 'predictfun/research/pf_ladder_queue_front_selftest.py', 'predictfun/research/pf_ladder_queue_front_mutants.py'] + \
    [f'predictfun/reviews/ladder_queue_front_20261122/{n}' for n in ('declaration.md', 'manifest.json', 'launch_queue_front.sh', 'fq_checks.py', 'fq_checks_v2.py')]
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
    out = Path(out or D); gm = json.loads((R / 'manifest.json').read_text()); E = dict(out=out, gm=gm, src={}, runs={}, parent={}, pz={}, z={}, recs={})
    for d in gm['days']:
        rb = blob(f'{PRS}/{d}.json'); prec = json.loads(rb); pg, pn = (PDIR / f'{d}.jsonl.gz').read_bytes(), (PDIR / f'{d}_arrays.npz').read_bytes()
        E['src'][f'parent_{d}'] = bool((PDIR / f'{d}.json').read_bytes() == rb and sha(rb) == gm['parent']['day_records'][d] and sha(pg) == prec['sha256']['gz'] and sha(pn) == prec['sha256']['npz'])
        E['parent'].update({(d, int(r['m'])): r for r in rows(pg)}); E['pz'][d] = np.load(PDIR / f'{d}_arrays.npz')
        for part in ('pilot', 'rest'):
            p_ = out / part / f'{d}.json'
            if not p_.exists():
                continue
            rec = json.loads(p_.read_text()); g_ = (out / part / f'{d}.jsonl.gz').read_bytes(); n_ = (out / part / f'{d}_arrays.npz').read_bytes(); E['recs'][(d, part)] = rec
            E['src'][f'run_{d}_{part}'] = bool(sha(g_) == rec['sha256']['gz'] and sha(n_) == rec['sha256']['npz'] and rec['treatment_sha256'] == gm['treatment_sha256'])
            for r in rows(g_):
                E['runs'].setdefault((d, int(r['m'])), []).append(dict(r, part=part))
            E['z'][(d, part)] = np.load(out / part / f'{d}_arrays.npz')
    md = out / 'measure'; E['audit'] = json.loads((md / 'audit.json').read_text())
    for n in ('arms', 'control', 'lost_added', 'classes', 'treatment'):
        E[n] = pd.read_csv(md / f'{n}.csv', dtype={'day': str}, float_precision='round_trip')
    return E


# ---------------------------------------------------------------------------------------------------------------------- this checker's quantities
def end_value(rec):
    m = float(rec['mark']); return sum(SG[int(f[2])] * float(f[3]) * (m - float(f[4])) for f in rec['fills'])


def el_1s(rec, end):
    v = q = 0.
    for f, mm in zip(rec['fills'], rec['fill_mids']):
        if float(f[0]) <= end - 1000. and mm[6] is not None:
            v += SG[int(f[2])] * float(f[3]) * (float(mm[6]) - float(f[4])); q += float(f[3])
    return v, q


def my_class(rec, end):
    out = {}; cnt = collections.Counter(int(f[5]) for f in rec['fills']); mark = float(rec['mark'])
    for f, mm, c in zip(rec['fills'], rec['fill_mids'], rec['fill_class']):
        k = out.setdefault(c[0], collections.Counter()); t, s, q, p = float(f[0]), int(f[2]), float(f[3]), float(f[4]); k['fills'] += 1; k['q'] += q; k['end'] += SG[s] * q * (mark - p)
        k['shared_trade_fills'] += int(cnt[int(f[5])] > 1)
        if t <= end - 1000. and mm[6] is not None:
            v = SG[s] * q * (float(mm[6]) - p); k['el_q'] += q; k['mo1s'] += v; k['pos1s'] += max(v, 0.); k['neg1s'] += min(v, 0.)
    return out


def my_lost_added(rb, rg):
    key = lambda f: (round(float(f[0]), 6), int(f[2]), round(float(f[4]), 6), round(float(f[3]), 6))

    def un(a, b):
        c = collections.Counter(map(key, b)); out = []
        for f in a:
            k = key(f); out.append(c[k] <= 0); c[k] -= 1 if c[k] > 0 else 0
        return [f for f, u in zip(a, out) if u]
    lo, ad = un(rb['fills'], rg['fills']), un(rg['fills'], rb['fills'])
    return dict(lost_n=len(lo), lost_q=sum(float(f[3]) for f in lo), added_n=len(ad), added_q=sum(float(f[3]) for f in ad), delta_value=float(rg['value']) - float(rb['value']))


def my_totals(E):
    """Per arm and market: END, eligible +1 s, eligible shares, filled shares (BASELINE from the parent's disabled arm)."""
    out = []
    for (d, m), lst in E['runs'].items():
        r = lst[0]; end = float(r['path_info']['end_rel'])
        for arm, rec in (('BASELINE', E['parent'][(d, m)]['disabled']),) + tuple((a, r[a]) for a in ARMS):
            v1, q1 = el_1s(rec, end); out.append(dict(day=d, m=m, arm=arm, END=end_value(rec), mo1s=v1, el_q=q1, q=sum(float(f[3]) for f in rec['fills'])))
    return pd.DataFrame(out)


def classify(vals):
    vals = list(vals); return 'POSITIVE_EVERY_DAY' if all(v > 0 for v in vals) else 'NEGATIVE_EVERY_DAY' if all(v < 0 for v in vals) else 'MIXED'


def my_predicates(T_, E):
    days = sorted(T_.day.unique()); tot = lambda a, d=None: T_[(T_.arm == a) & ((T_.day == d) if d else True)]
    dE = {d: tot(F_, d).END.sum() - tot(C_, d).END.sum() for d in days}; d1 = {d: tot(F_, d).mo1s.sum() - tot(C_, d).mo1s.sum() for d in days}
    eA, mA = classify(dE.values()), classify(d1.values()); econ = 'IMPROVES' if eA == mA == 'POSITIVE_EVERY_DAY' else 'WORSENS' if eA == mA == 'NEGATIVE_EVERY_DAY' else 'MIXED'
    dT = tot(F_).END.sum() - tot(C_).END.sum(); adds = bool(tot(F_).q.sum() > tot(C_).q.sum()); fe = tot(F_).END.sum()
    reading = ('queue_helps_policy_nonnegative' if econ == 'IMPROVES' and fe >= 0 else 'queue_helps_but_insufficient' if econ == 'IMPROVES'
               else 'front_priority_does_not_rescue' if adds and dT <= 0 else 'no_added_fills' if not adds else 'mixed')
    vol = val = 0.
    for cls in ('same_price', 'through'):
        qa = qb = va = vb = 0.
        for (d, m), lst in E['runs'].items():
            end = float(lst[0]['path_info']['end_rel']); a, b = my_class(lst[0][C_], end).get(cls, {}), my_class(lst[0][F_], end).get(cls, {})
            qa += a.get('el_q', 0.); va += a.get('mo1s', 0.); qb += b.get('el_q', 0.); vb += b.get('mo1s', 0.)
        pa = va / qa if qa > 0 else 0.; pb = vb / qb if qb > 0 else 0.; vol += (qb - qa) * pa; val += qb * (pb - pa)
    return dict(END_answer=eA, mo1s_answer=mA, economic=econ, reading=reading, adds_fills=adds, END_total=dT, recovered=dT / abs(tot(C_).END.sum()), volume=vol, value=val)


# ---------------------------------------------------------------------------------------------------------------------- families
def fam_A(E):
    c = dict(A_committed=all(blob(p) == (REPO / p).read_bytes() for p in COMMITTED), A_inputs=bool(E['src'] and all(E['src'].values())))
    md = E['out'] / 'measure'; c['A_outputs'] = all(E['audit']['outputs'][f'{n}.csv'] == sha((md / f'{n}.csv').read_bytes()) for n in ('arms', 'control', 'lost_added', 'classes', 'treatment'))
    return c, {}


def fam_B(E):
    gm = E['gm']; want = sorted((d, int(m)) for d in gm['days'] for m in gm['complete'][d]); got = sorted(E['runs'])
    once = all(len(v) == 1 for v in E['runs'].values()); parts_ok = all(gm['parts']['pilot'][d] == [int(m) for m in gm['complete'][d][:2]] and gm['parts']['rest'][d] == [int(m) for m in gm['complete'][d][2:]]
                                                                     for d in gm['days'])
    placed = all(v[0]['part'] == ('pilot' if k[1] in gm['parts']['pilot'][k[0]] else 'rest') for k, v in E['runs'].items())
    return dict(B_membership=bool(got == want and once and parts_ok and placed and len(want) == 19)), dict(B=dict(markets=len(got)))


def fam_K(E):
    bad = []
    for (d, m), lst in E['runs'].items():
        r = lst[0]; z, pp = r[C_], E['parent'][(d, m)]['pressure']; za = E['z'][(d, r['part'])][f'{m}__{C_}__commands']; pa = E['pz'][d][f'{m}__pressure__commands']
        same = all(z[k] == pp[k] for k in ('fingerprint', 'fills', 'orders', 'requests', 'value', 'guard_log', 'queue_venue', 'queue_known')) and np.array_equal(za, pa, equal_nan=True)
        bad += [] if (same and r['control_all'] and all(r['control'].values())) else [f'{d} {m}']
    return dict(K_control=not bad), dict(K=dict(markets_failed=bad))


def fam_T(E):
    bad = []
    for (d, m), lst in E['runs'].items():
        for arm in ARMS:
            rec = lst[0][arm]; q0 = rec['queue_venue']['ahead0']; acc = [v for v in q0.values() if float(v[2]) > 0]; t = rec['treatment']
            if arm == F_:
                ok_ = all(float(v[1]) == 0. for v in acc) and t['front'] is True and t['front_set'] == t['accepted_new'] == len(acc)
            else:
                ok_ = t['front'] is False and t['front_set'] == 0 and t['accepted_new'] == len(acc)
            bad += [] if ok_ else [f'{d} {m} {arm}']
    return dict(T_treatment=not bad), dict(T=dict(failed=bad))


def fam_F(E):
    bad = []
    for (d, m), lst in E['runs'].items():
        for arm in ARMS:
            rec = lst[0][arm]; use = collections.Counter(); size = {}
            if len(rec['fill_class']) != len(rec['fills']):
                bad.append(f'{d} {m} {arm} alignment'); continue
            for f, c in zip(rec['fills'], rec['fill_class']):
                s, p, tp = int(f[2]), float(f[4]), float(c[1]); want = 'same_price' if tp == p else ('through' if (tp < p if s == 0 else tp > p) else 'invalid')
                bad += [] if (c[0] == want and want != 'invalid') else [f'{d} {m} {arm} class']; use[int(f[5])] += float(f[3]); size[int(f[5])] = float(c[2])
            bad += [f'{d} {m} {arm} conservation'] if any(use[i] > size[i] + 1e-7 for i in use) else []
    return dict(F_fills=not bad), dict(F=dict(failed=bad[:10]))


def fam_M(E):
    T_ = E['mine_t']; A_ = E['arms']; bad = []
    J = T_.merge(A_[['day', 'm', 'arm', 'value', 'all_id_mo1s', 'all_cut_valid_q', 'all_q']], on=['day', 'm', 'arm'], how='left')
    for r in J.itertuples(index=False):
        bad += [] if (close(r.END, r.value) and close(r.mo1s, r.all_id_mo1s) and close(r.el_q, r.all_cut_valid_q) and close(r.q, r.all_q)) else [f'arms {r.day} {r.m} {r.arm}']
    K_ = E['classes']
    for (d, m), lst in E['runs'].items():
        end = float(lst[0]['path_info']['end_rel'])
        for arm in ARMS:
            mc = my_class(lst[0][arm], end)
            for cls in ('same_price', 'through'):
                g = K_[(K_.day == d) & (K_.m == m) & (K_.arm == arm) & (K_.cls == cls)]; x = mc.get(cls, collections.Counter())
                bad += [] if (len(g) == 1 and all(close(g.iloc[0][k], x.get(k, 0.)) for k in ('fills', 'q', 'end', 'el_q', 'mo1s', 'pos1s', 'neg1s', 'shared_trade_fills'))) else [f'class {d} {m} {arm} {cls}']
        la = my_lost_added(lst[0][C_], lst[0][F_]); g = E['lost_added'][(E['lost_added'].day == d) & (E['lost_added'].m == m)]
        bad += [] if (len(g) == 1 and all(close(g.iloc[0][k], v) for k, v in la.items()) and abs(float(g.iloc[0]['identity_gap'])) < 1e-6) else [f'lost_added {d} {m}']
        tr = E['treatment']
        for arm in ARMS:
            g = tr[(tr.day == d) & (tr.m == m) & (tr.arm == arm)]; t = lst[0][arm]['treatment']
            bad += [] if (len(g) == 1 and int(g.iloc[0]['front_set']) == t['front_set'] and int(g.iloc[0]['accepted_new']) == t['accepted_new']) else [f'treatment {d} {m} {arm}']
    return dict(M_measurement=not bad and len(J) == 3 * len(E['runs'])), dict(M=dict(failed=bad[:10]))


def fam_P(E):
    m, a = E['mine_p'], E['audit']['predicates']; bad = []
    bad += [] if (a['END']['answer'] == m['END_answer'] and a['mo1s']['answer'] == m['mo1s_answer'] and a['economic'] == m['economic'] and a['reading'] == m['reading']
                  and a['adds_fills'] == m['adds_fills']) else ['answers']
    bad += [] if (close(a['END']['total'], m['END_total']) and close(a['recovered'], m['recovered']) and close(a['mix']['volume'], m['volume']) and close(a['mix']['value'], m['value'])) else ['values']
    return dict(P_readings=not bad), dict(P=dict(failed=bad, mine=m))


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


FAMS = dict(A=fam_A, B=fam_B, K=fam_K, T=fam_T, F=fam_F, M=fam_M, P=fam_P, I=fam_I)


def run_checks(E):
    chk, det = {}, {}
    for k in FAMS:
        c, d = FAMS[k](E); chk.update(c); det.update(d)
    return dict(checks=chk, details=det, all_ok=bool(all(chk.values())))


def _first(E):
    return sorted(E['runs'])[0]


def _edit(E, fn):
    k = _first(E); r = json.loads(json.dumps(E['runs'][k][0])); fn(r); runs = dict(E['runs']); runs[k] = [r]; E['runs'] = runs


def _fake(E):
    E['_root'] = Path(tempfile.mkdtemp(dir=str(E['out']))); (E['_root'] / 'logs').mkdir()
    (E['_root'] / 'logs' / 'fake.log').write_text('2026-11-22T00:00:00Z tag=fake rc=0 peak_rss_mib=100 wall_s=999.0 cpu_user_s=999.0\n')


CASES = [
    ('an_output_hash_changed', 'A', 'A_outputs', lambda E: E.__setitem__('audit', dict(E['audit'], outputs=dict(E['audit']['outputs'], **{'arms.csv': '0' * 64})))),
    ('a_market_dropped', 'B', 'B_membership', lambda E: E.__setitem__('runs', {k: v for k, v in E['runs'].items() if k != _first(E)})),
    ('a_current_fill_changed', 'K', 'K_control', lambda E: _edit(E, lambda r: r[C_]['fills'][0].__setitem__(3, float(r[C_]['fills'][0][3]) + 1.))),
    ('a_front_generation_behind', 'T', 'T_treatment', lambda E: _edit(E, lambda r: next(v for v in r[F_]['queue_venue']['ahead0'].values() if float(v[2]) > 0).__setitem__(1, 5.))),
    ('a_fill_class_flipped', 'F', 'F_fills', lambda E: _edit(E, lambda r: r[F_]['fill_class'][0].__setitem__(0, 'through' if r[F_]['fill_class'][0][0] == 'same_price' else 'same_price'))),
    ('a_fill_over_its_trade', 'F', 'F_fills', lambda E: _edit(E, lambda r: r[F_]['fill_class'][0].__setitem__(2, 0.))),
    ('an_arm_value_changed', 'M', 'M_measurement', lambda E: E.__setitem__('arms', E['arms'].assign(value=E['arms'].value + np.where(E['arms'].index == 0, 1., 0.)))),
    ('a_class_value_changed', 'M', 'M_measurement', lambda E: E.__setitem__('classes', E['classes'].assign(mo1s=E['classes'].mo1s + np.where(E['classes'].index == 0, 1., 0.)))),
    ('a_reading_changed', 'P', 'P_readings', lambda E: E.__setitem__('audit', dict(E['audit'], predicates=dict(E['audit']['predicates'], reading='queue_helps_policy_nonnegative'
                                                                                                                if E['audit']['predicates']['reading'] != 'queue_helps_policy_nonnegative' else 'mixed')))),
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
    E = load(*(argv[:1] or [None])); E['mine_t'] = my_totals(E); E['mine_p'] = my_predicates(E['mine_t'], E); r_ = run_checks(E); fz = falsify(E, r_)
    for n, obj in (('checks_v2.json', r_), ('checks_falsifier_v2.json', fz)):
        p = R / n; assert not p.exists(), f'{p} exists (write-once)'; p.write_text(json.dumps(obj, indent=1, default=str) + '\n')
    print(json.dumps(dict(all_ok=r_['all_ok'], failed=[k for k, v in r_['checks'].items() if not v], falsifier_ok=fz['falsifier_ok'],
                          cases={k: (v['clean'], v['refused']) for k, v in fz['cases'].items()}, details=r_['details']), indent=1, default=str)[:3500])


if __name__ == '__main__':
    main(sys.argv[1:])
