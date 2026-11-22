"""INDEPENDENT CHECKS of LONG-CONTINUATION-PAIRED-VALUE-v1 (declaration.md; session aea5bfb9, 2026-11-22). Everything is recomputed from the saved
day ledgers and the ASYNC receipt's hash-bound pilot records with this checker's own code; the builder is not imported. Families:
 A identity: the frozen files committed; the census recomputed from the saved ASYNC pilot command arrays = census.csv (= its manifest sha256)
   and its selection; the day records' bytes; the measure outputs = the audit;
 B membership: pairs = the selected states once each, two arms each, a ledger per pair and a base per market;
 Y values: per pair Q of each arm (post-fork fills at the END mark + the carried inventory), Y, the focal / other split, the economic class,
   both arms' sequence classes and the mechanism;
 G gates: the forced action applied once per arm at equal contexts, pre-fork fills identical, the per-arm reconciliation, the queue reset
   recomputed from the saved activation books, no pressure pull on a lineage order, the unused-exemption control against the saved record;
 P parity: each market's base ledger = the saved ASYNC record (fills, orders, value, fingerprint);
 S summary: summary.csv's counts / sums per level and the declared reading;  I resources: batch + review reserve <= 0.40, peak < 1,536 MiB,
   line + batch + reserve <= 2.85.
M_fork (the carried mark) is not recomputed here: it cancels in Y; the builder's selftest checks its rule. falsify(): twelve corruptions,
each refused by its family while the clean evidence passes. Usage: lc_checks.py [out_dir]"""
import glob
import gzip
import hashlib
import io
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

REPO = Path('/home/yuqing/ctaNew'); R = REPO / 'predictfun/reviews/ladder_long_continuation_20261122'; D = REPO / 'data/predict_fun/research/ladder_long_continuation_v1_20261122'
AF = REPO / 'data/predict_fun/research/ladder_async_fast_v1_20261122'; ARM = 'CURRENT_ASYNC_FAST10'; TOL = 1e-6
COMMITTED = ['predictfun/research/pf_ladder_long_continuation.py', 'predictfun/research/pf_ladder_long_continuation_selftest.py', 'predictfun/research/pf_ladder_long_continuation_mutants.py'] + \
    [f'predictfun/reviews/ladder_long_continuation_20261122/{n}' for n in ('declaration.md', 'manifest.json', 'census.csv', 'launch_long_continuation.sh', 'lc_checks.py')]
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
SEQS = ('no_focal_fill', 'early_loss_later_benefit', 'early_loss_no_later_benefit', 'no_early_loss_later_benefit', 'no_early_loss_no_later_benefit')


def sha(b):
    return hashlib.sha256(b).hexdigest()


def blob(rel):
    r = subprocess.run(['git', '-C', str(REPO), 'show', f'HEAD:{rel}'], capture_output=True); return r.stdout if r.returncode == 0 else None


def close(a, b, tol=1e-6):
    a, b = float(a), float(b)
    return (math.isnan(a) and math.isnan(b)) or (not math.isnan(a) and not math.isnan(b) and abs(a - b) <= tol * max(1., abs(a), abs(b)))


def af_arrays(day):
    return np.load(AF / 'pilot' / f'{day}_arrays.npz')


def af_record(day, m):
    for ln in gzip.decompress((AF / 'pilot' / f'{day}.jsonl.gz').read_bytes()).decode().splitlines():
        r = json.loads(ln)
        if int(r['m']) == int(m):
            return r[ARM]
    return None


def load():
    gm = json.loads((R / 'manifest.json').read_text()); E = dict(gm=gm, days={}, led=[])
    for part, days in gm['parts'].items():
        for day in days:
            rec = json.loads((D / part / f'{day}.json').read_text()); E['days'][(part, day)] = rec
            E['led'] += [json.loads(x) for x in gzip.decompress((D / part / f'{day}_ledgers.jsonl.gz').read_bytes()).decode().splitlines()]
    md = D / 'measure'; E['audit'] = json.loads((md / 'audit.json').read_text())
    E['pairs'] = pd.read_csv(md / 'pairs.csv', dtype={'day': str}); E['sum'] = pd.read_csv(md / 'summary.csv', dtype={'day': str, 'side': str}); E['census'] = pd.read_csv(R / 'census.csv', dtype={'day': str})
    return E


def my_census(gm):
    rows = []
    for day, ms in gm['pilot_markets'].items():
        z = af_arrays(day)
        for m in ms:
            cm = z[f'{int(m)}__{ARM}__commands']
            for gid in sorted({int(x) for x in cm[(cm[:, 2] == 0) & (cm[:, 7] == 2) & (cm[:, 8] == 1), 3]}):
                x = cm[(cm[:, 2] == 0) & (cm[:, 7] == 2) & (cm[:, 8] == 1) & (cm[:, 3] == gid)]; i = int(np.argmin(x[:, 0]))
                rows.append((day, int(m), 'REVIEW', int(x[i, 1]), gid, float(x[i, 0]), sha(f'{ARM}|{day}|{int(m)}|REVIEW|{int(x[i, 1])}|{gid}'.encode())))
            for x in cm[(cm[:, 2] == 1) & (cm[:, 7] == 0) & (cm[:, 3] >= 0)]:
                rows.append((day, int(m), 'ADMISSION', int(x[1]), int(x[3]), float(x[0]), sha(f'{ARM}|{day}|{int(m)}|ADMISSION|{int(x[1])}|{float(x[0])}|{int(x[3])}'.encode())))
    cells = {}
    for r in rows:
        cells.setdefault(r[:4], []).append(r)
    sel = sorted(r[6] for v in cells.values() for r in sorted(v, key=lambda y: y[6])[:gm['per_cell']])
    return sorted(r[6] for r in rows), sel


def fam_A(E):
    gm = E['gm']; c = dict(A_committed=all(blob(p) == (REPO / p).read_bytes() for p in COMMITTED))
    allk, sel = my_census(gm); C_ = E['census']
    c['A_census'] = bool(sha((R / 'census.csv').read_bytes()) == gm['census_sha256'] and sorted(C_.key) == allk and sorted(C_[C_.selected].key) == sel == sorted(s['key'] for s in gm['selected']))
    ok = True
    for (part, day), rec in E['days'].items():
        ok &= all(rec['sha256'][k] == sha((D / part / f'{day}{suf}').read_bytes()) for k, suf in (('pairs', '_pairs.csv'), ('arms', '_arms.csv'), ('reps', '_reps.csv'), ('parity', '_parity.json'),
                                                                                           ('ledgers', '_ledgers.jsonl.gz')))
    c['A_days'] = bool(ok); c['A_audit'] = all(E['audit']['outputs'][n] == sha((D / 'measure' / n).read_bytes()) for n in E['audit']['outputs']); return c, {}


def pairs_led(E):
    return [x for x in E['led'] if x['kind'] != 'BASE']


def fam_B(E):
    keys = sorted(x['key'] for x in pairs_led(E)); want = sorted(s['key'] for s in E['gm']['selected']); bases = sorted((x['day'], int(x['m'])) for x in E['led'] if x['kind'] == 'BASE')
    mk = sorted((d, int(m)) for d, ms in E['gm']['pilot_markets'].items() for m in ms)
    ok = bool(keys == want and len(set(keys)) == len(keys) and sorted(E['pairs'].key) == want and bases == mk and all('retain' in x and 'alternative' in x for x in pairs_led(E)))
    return dict(B_membership=ok), dict(B=dict(pairs=len(keys), bases=len(bases)))


def cval(f, mark):
    return (1. if f[2] == 0 else -1.) * f[3] * (mark - f[4])


def my_arm(a):
    I0 = sum((1. if f[2] == 0 else -1.) * f[3] for f in a['pre_fills']); cash0 = -sum((1. if f[2] == 0 else -1.) * f[3] * f[4] for f in a['pre_fills'])
    post = sum(cval(f, a['mark']) for f in a['post_raw']); lin = set(a['lineage']); Q = post + (I0 * (a['mark'] - a['M_fork']) if a['M_fork'] is not None else float('nan'))
    ff = sorted([f for f in a['post_raw'] if f[1] == a['focal']], key=lambda f: f[0])
    if not ff:
        seq = 'no_focal_fill'
    else:
        t1 = ff[0][0]; early = sum(cval(f, a['mark']) for f in ff if f[0] == t1); later = sum(cval(f, a['mark']) for f in a['post_raw'] if f[1] in lin and f[0] > t1)
        seq = ('early_loss' if early < -TOL else 'no_early_loss') + '_' + ('later_benefit' if later > TOL else 'no_later_benefit')
    return dict(I0=I0, cash0=cash0, post=post, Q=Q, focal=sum(cval(f, a['mark']) for f in a['post_raw'] if f[1] in lin), other=sum(cval(f, a['mark']) for f in a['post_raw'] if f[1] not in lin),
                seq=seq, rec=bool(a['M_fork'] is None or abs(a['value'] - (cash0 + I0 * a['M_fork']) - Q) <= 1e-6 * max(1., abs(a['value']))))


def my_pairs(E):
    out = {}
    for x in pairs_led(E):
        r_, a_ = my_arm(x['retain']), my_arm(x['alternative']); Y = r_['Q'] - a_['Q'] if x['retain']['M_fork'] is not None else r_['post'] - a_['post']
        econ = 'retain_better' if Y > TOL else ('alternative_better' if Y < -TOL else 'tie')
        mech = ('retention_beneficial_with_early_loss' if r_['seq'].startswith('early_loss') else 'retention_beneficial_without_early_loss') if econ == 'retain_better' else econ
        out[x['key']] = dict(Y=Y, Y_post=r_['post'] - a_['post'], Y_focal=r_['focal'] - a_['focal'], Y_other=r_['other'] - a_['other'], econ=econ, mechanism=mech, seq_retain=r_['seq'],
                             seq_alternative=a_['seq'], rec=r_['rec'] and a_['rec'], Q_r=r_['Q'], Q_a=a_['Q'], Qs_r=x['retain']['Q'], Qs_a=x['alternative']['Q'])
    return out


def fam_Y(E):
    my = E['my']; bad = []
    for r in E['pairs'].to_dict('records'):
        x = my[r['key']]
        if not (all(close(r[k], x[k]) for k in ('Y', 'Y_post', 'Y_focal', 'Y_other')) and all(r[k] == x[k] for k in ('econ', 'mechanism', 'seq_retain', 'seq_alternative'))
                and close(x['Q_r'], x['Qs_r']) and close(x['Q_a'], x['Qs_a'])):
            bad.append(r['key'][:12])
    return dict(Y_values=not bad), dict(Y=dict(bad=bad[:5]))


def shown(book, side, price):
    pr, sz = book[:5], book[5:]; last = pr[4]
    if last >= 0 and (price < last if side == 0 else price > last):
        return 0.
    return float(sum(z for p, z in zip(pr, sz) if p >= 0 and p == price))


def fam_G(E):
    my = E['my']; bad = {k: [] for k in ('applied', 'pre_fork', 'reconciled', 'queue_reset', 'lineage_pull', 'unused_exemption')}; recs = {}; mine_ok = {}
    for x in pairs_led(E):
        R_, A_ = x['retain'], x['alternative']; k = x['key']; f0 = sum(len(v) for v in bad.values())
        if not (len(R_['applied']) == 1 and len(A_['applied']) == 1 and R_['applied'][0][4] == A_['applied'][0][4]):
            bad['applied'].append(k)
        if R_['pre_fills'] != A_['pre_fills']:
            bad['pre_fork'].append(k)
        if not my[k]['rec']:
            bad['reconciled'].append(k)
        if not all(r['refused'] or r['ahead0'] is None or abs(r['ahead0'] - shown(r['book'], r['side'], r['price'])) <= 1e-9 for a in (R_, A_) for r in a['replacements']):
            bad['queue_reset'].append(k)
        if any(int(g) in set(a['lineage']) for a in (R_, A_) for _, g in a['pp_log']):
            bad['lineage_pull'].append(k)
        if R_['suppressed'] == 0:
            key = (x['day'], int(x['m'])); recs.setdefault(key, af_record(*key)); fl = [list(f) for f in R_['pre_fills']] + [list(f) for f in R_['post_raw']]
            if sorted(map(tuple, fl)) != sorted(map(tuple, recs[key]['fills'])) or not close(R_['value'], recs[key]['value']):
                bad['unused_exemption'].append(k)
        mine_ok[k] = sum(len(v) for v in bad.values()) == f0
    st = dict(zip(E['pairs'].key, E['pairs'].status))
    return dict(G_gates=not any(bad.values()), G_status=bool(all((st[k] == 'ok') == v for k, v in mine_ok.items()))), dict(G={k: [x[:12] for x in v[:4]] for k, v in bad.items()})


def fam_P(E):
    bad = []
    for x in E['led']:
        if x['kind'] == 'BASE':
            r = af_record(x['day'], x['m']); b = x['base']
            if not (r is not None and b['fills'] == r['fills'] and b['orders'] == r['orders'] and b['value'] == r['value'] and b['fingerprint'] == r['fingerprint']):
                bad.append((x['day'], x['m']))
    return dict(P_parity=not bad), dict(P=dict(bad=bad))


def fam_S(E):
    P_ = E['pairs']; S_ = E['sum']; bad = []
    for r in S_.to_dict('records'):
        g = P_
        for k in ('kind', 'day', 'side'):
            if str(r[k]) != 'all':
                g = g[g[k].astype(str) == str(r[k])]
        o = g[g.status == 'ok']
        x = dict(pairs=len(g), ok=len(o), econ_retain_better=int((o.econ == 'retain_better').sum()), econ_alternative_better=int((o.econ == 'alternative_better').sum()),
                 econ_tie=int((o.econ == 'tie').sum()), Y_sum=float(o.Y.sum()), Y_focal_sum=float(o.Y_focal.sum()), Y_other_sum=float(o.Y_other.sum()), Y_abs_sum=float(o.Y.abs().sum()),
                 **{f'seq_{s}': int((o.seq_retain == s).sum()) for s in SEQS})
        if not all(close(r[k], v) for k, v in x.items()):
            bad.append((r['kind'], r['day'], r['side']))
    allr = S_[(S_.kind == 'all') & (S_.day == 'all')].iloc[0]; ok_ = P_[P_.status == 'ok']; vs, va = E['gm']['varied']['share'], E['gm']['varied']['abs_c']
    reliable = bool((P_.status == 'ok').all()) and bool(E['audit']['reading']['parity_all'])
    varied = bool(len(ok_) > 0 and (ok_.econ == 'retain_better').sum() >= vs * len(ok_) and (ok_.econ == 'alternative_better').sum() >= vs * len(ok_) and ok_.Y.abs().sum() >= va)
    reading = 'reliable_and_varied' if (reliable and varied) else ('reliable_not_varied' if reliable else 'not_reliable')
    return dict(S_summary=not bad and len(S_) > 0, S_reading=E['audit']['reading']['reading'] == reading and bool(allr.pairs == len(P_))), dict(S=dict(bad=bad[:4], reading=reading))


def cpu(paths):
    tot, peak = 0., 0
    for p in sorted(set(paths)):
        for ln in Path(p).read_text(errors='replace').splitlines():
            mm = RL.match(ln)
            if mm:
                tot += float(mm.group(6)); peak = max(peak, int(mm.group(4)))
    return tot / 3600., peak


def fam_I(E, root=None):
    gm = E['gm']; cp = gm['caps']; own, peak = cpu(glob.glob(str((root or R) / 'logs' / '**' / '*.log'), recursive=True))
    line, _ = cpu([p for n in gm['line_receipts'] for p in glob.glob(str(REPO / 'predictfun/reviews' / n / 'logs' / '**' / '*.log'), recursive=True)])
    rev, _ = cpu([p for n in gm['line_reviews'] for p in glob.glob(str(REPO / 'predictfun/reviews/automatic' / n / '*.log'))]); line += rev + cp['prior_unlogged_s'] / 3600.
    return dict(I_resources=bool(own + cp['review_reserve'] <= cp['batch_core_hours'] and peak < 1536 and line + own + cp['review_reserve'] <= cp['line_ceiling'])), \
        dict(I=dict(batch_core_hours=own, line_before=line, line_with_batch=line + own, with_reserve=line + own + cp['review_reserve'], peak_mib=peak))


FAMS = dict(A=fam_A, B=fam_B, Y=fam_Y, G=fam_G, P=fam_P, S=fam_S, I=fam_I)


def run_checks(E):
    chk, det = {}, {}
    for k in FAMS:
        c, d = FAMS[k](E); chk.update(c); det.update(d)
    return dict(checks=chk, details=det, all_ok=bool(all(chk.values())))


def _set_pair(E, col, fn):
    P_ = E['pairs'].copy(); P_.loc[0, col] = fn(P_.loc[0, col]); E['pairs'] = P_


def _mut(E, pred, mutate):
    L_ = json.loads(json.dumps(E['led']))
    for y in L_:
        if y['kind'] != 'BASE' and pred(y):
            mutate(y); break
    else:
        raise KeyError('no row to corrupt')
    E['led'] = L_


def _fake(E):
    E['_root'] = Path(tempfile.mkdtemp(dir=str(D))); (E['_root'] / 'logs').mkdir()
    (E['_root'] / 'logs' / 'fake.log').write_text('2026-11-22T00:00:00Z tag=fake rc=0 peak_rss_mib=100 wall_s=9999.0 cpu_user_s=9999.0\n')


CASES = [
    ('a_census_key_changed', 'A', 'A_census', lambda E: E.__setitem__('census', E['census'].assign(key=E['census'].key.where(E['census'].index != 0, '0' * 64)))),
    ('an_audit_hash_changed', 'A', 'A_audit', lambda E: E.__setitem__('audit', dict(E['audit'], outputs=dict(E['audit']['outputs'], **{'pairs.csv': '0' * 64})))),
    ('a_pair_dropped', 'B', 'B_membership', lambda E: E.__setitem__('led', [x for x in E['led'] if x['key'] != pairs_led(E)[0]['key']])),
    ('a_Y_changed', 'Y', 'Y_values', lambda E: _set_pair(E, 'Y', lambda v: v + 1.)),
    ('an_econ_class_flipped', 'Y', 'Y_values', lambda E: _set_pair(E, 'econ', lambda v: 'tie' if v != 'tie' else 'retain_better')),
    ('a_post_fill_changed', 'Y', 'Y_values', lambda E: _mut(E, lambda y: len(y['retain']['post_raw']) > 0, lambda y: y['retain']['post_raw'][0].__setitem__(3, y['retain']['post_raw'][0][3] + 1.))),
    ('a_second_application', 'G', 'G_gates', lambda E: _mut(E, lambda y: True, lambda y: y['retain']['applied'].append(y['retain']['applied'][0]))),
    ('an_inherited_queue', 'G', 'G_gates', lambda E: _mut(E, lambda y: any(not r['refused'] and r['ahead0'] is not None for r in y['alternative']['replacements']),
                                                      lambda y: next(r for r in y['alternative']['replacements'] if not r['refused'] and r['ahead0'] is not None).__setitem__(
                                                          'ahead0', next(r for r in y['alternative']['replacements'] if not r['refused'] and r['ahead0'] is not None)['ahead0'] + 7.))),
    ('a_lineage_pull', 'G', 'G_gates', lambda E: _mut(E, lambda y: len(y['alternative']['lineage']) > 0, lambda y: y['alternative']['pp_log'].append([0., y['alternative']['lineage'][0]]))),
    ('a_base_differs', 'P', 'P_parity', lambda E: E.__setitem__('led', [dict(x, base=dict(x['base'], value=x['base']['value'] + 1.)) if x['kind'] == 'BASE' else x for x in E['led']])),
    ('a_summary_count_changed', 'S', 'S_summary', lambda E: E.__setitem__('sum', E['sum'].assign(ok=E['sum'].ok + np.where(E['sum'].index == 0, 1, 0)))),
    ('resources_over_the_cap', 'I', 'I_resources', _fake),
]


def falsify(E, base):
    out = {}
    for name, fam, target, fn in CASES:
        X_ = dict(E); refused = False
        try:
            fn(X_)
            if fam in ('Y', 'S', 'G'):
                X_['my'] = my_pairs(X_)
            c, _ = FAMS[fam](X_, X_['_root']) if fam == 'I' else FAMS[fam](X_); refused = not c[target]
        except (IndexError, KeyError, StopIteration) as e_:
            print(f'falsifier case {name}: corruption not applied ({e_!r})')
        finally:
            if '_root' in X_:
                shutil.rmtree(X_['_root'], ignore_errors=True)
        out[name] = dict(family=fam, target=target, clean=bool(base['checks'][target]), refused=bool(refused))
    return dict(cases=out, falsifier_ok=bool(base['all_ok'] and all(v['clean'] and v['refused'] for v in out.values())))


def main(argv):
    E = load(); E['my'] = my_pairs(E); r_ = run_checks(E); fz = falsify(E, r_); rd = Path(argv[0]) if argv else R
    for n, obj in (('checks.json', r_), ('checks_falsifier.json', fz)):
        p = rd / n; assert not p.exists(), f'{p} exists (write-once)'; p.write_text(json.dumps(obj, indent=1, default=str) + '\n')
    print(json.dumps(dict(all_ok=r_['all_ok'], failed=[k for k, v in r_['checks'].items() if not v], falsifier_ok=fz['falsifier_ok'],
                          cases={k: (v['clean'], v['refused']) for k, v in fz['cases'].items()}, details=r_['details']), indent=1, default=str)[:4000])


if __name__ == '__main__':
    main(sys.argv[1:])
