"""INDEPENDENT CHECKS of LONG-VALUE-RETENTION-DISCRIMINATION-v1 (declaration.md; session aea5bfb9, 2026-11-22). Recomputed from the saved day ledgers,
pairs and the ASYNC receipt's hash-bound records with this checker's own code; the builder is not imported. Families:
 A identity: the frozen files committed; the census recomputed from the ASYNC command arrays = census.csv (= its manifest sha256) and its
   selection; the day records' bytes; the measure outputs = the audit;
 B membership: pairs = the selected states once each; a ledger per pair, a base per market;
 Y values: per arm Q from the saved post-fork fills and the carried inventory = the saved Q; Y = Q_retain - Q_alternative = the pairs' Y;
 G gates: the forced callback = the expected tuple (live, t, gid, KEEP / PULL) once per arm at equal contexts; pre-fork fills identical;
   reconciliation; the queue reset from the saved activation books; no lineage pressure pull; the unused-exemption control against the
   saved ASYNC record; both arms' features identical and equal to the pairs' columns; feature sanity (time left + t = END, rung = census,
   known remaining <= size, distance >= 0, binary flags);
 P parity: each market's base ledger = the saved ASYNC record;
 M model: an independent ridge refit (winsorize / standardize / clip / lambda = n) on the training ok pairs = model.json and the scored
   predictions; V_rule, the controls, per-day values, MSE, the matched draws (same seed, own loop) and the reading recomputed;
 I resources: batch (logs) + review reserve <= 1.25, peak < 1,536 MiB, line + batch + reserve <= 4.00.
falsify(): twelve corruptions, each refused by its family while the clean evidence passes. Usage: rd_checks.py [out_dir]"""
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

REPO = Path('/home/yuqing/ctaNew'); R = REPO / 'predictfun/reviews/ladder_retention_discrimination_20261122'; D = REPO / 'data/predict_fun/research/ladder_retention_discrimination_v1_20261122'
AF = REPO / 'data/predict_fun/research/ladder_async_fast_v1_20261122'; ARM = 'CURRENT_ASYNC_FAST10'; TOL = 1e-6
COMMITTED = ['predictfun/research/pf_ladder_retention_discrimination.py', 'predictfun/research/pf_ladder_retention_discrimination_selftest.py',
             'predictfun/research/pf_ladder_retention_discrimination_mutants.py'] + \
    [f'predictfun/reviews/ladder_retention_discrimination_20261122/{n}' for n in ('declaration.md', 'manifest.json', 'census.csv', 'launch_retention.sh', 'rd_checks.py')]
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')


def sha(b):
    return hashlib.sha256(b).hexdigest()


def blob(rel):
    r = subprocess.run(['git', '-C', str(REPO), 'show', f'HEAD:{rel}'], capture_output=True); return r.stdout if r.returncode == 0 else None


def close(a, b, tol=1e-6):
    a, b = float(a), float(b)
    return (math.isnan(a) and math.isnan(b)) or (not math.isnan(a) and not math.isnan(b) and abs(a - b) <= tol * max(1., abs(a), abs(b)))


def af_record(day, m, part):
    for ln in gzip.decompress((AF / part / f'{day}.jsonl.gz').read_bytes()).decode().splitlines():
        r = json.loads(ln)
        if int(r['m']) == int(m):
            return r
    return None


def load():
    gm = json.loads((R / 'manifest.json').read_text()); E = dict(gm=gm, days={}, led=[])
    for part, days in gm['parts'].items():
        for day in days:
            E['days'][(part, day)] = json.loads((D / part / f'{day}.json').read_text())
            E['led'] += [json.loads(x) for x in gzip.decompress((D / part / f'{day}_ledgers.jsonl.gz').read_bytes()).decode().splitlines()]
    md = D / 'measure'; E['audit'] = json.loads((md / 'audit.json').read_text()); E['model'] = json.loads((md / 'model.json').read_text())
    E['pairs'] = pd.read_csv(md / 'pairs.csv', dtype={'day': str}); E['scored'] = pd.read_csv(md / 'scored.csv', dtype={'day': str}); E['census'] = pd.read_csv(R / 'census.csv', dtype={'day': str})
    E['part_of'] = {(s['day'], int(s['m'])): s['part'] for s in gm['selected']}; return E


def my_census(gm):
    rows = []
    for d, pm in gm['markets'].items():
        for part in ('pilot', 'rest'):
            if not pm[part]:
                continue
            z = np.load(AF / part / f'{d}_arrays.npz')
            for m in pm[part]:
                cm = z[f'{int(m)}__{ARM}__commands']; lk = cm[(cm[:, 2] == 0) & (cm[:, 7] == 2) & (cm[:, 8] == 1)]
                for gid in sorted({int(x) for x in lk[:, 3]}):
                    x = lk[lk[:, 3] == gid]; i = int(np.argmin(x[:, 0])); s = int(x[i, 1])
                    rows.append((d, s, sha(f'{ARM}|{d}|{int(m)}|REVIEW|{s}|{gid}'.encode())))
    cells = {}
    for r in rows:
        cells.setdefault(r[:2], []).append(r[2])
    return sorted(r[2] for r in rows), sorted(k for v in cells.values() for k in sorted(v)[:gm['per_cell']])


def fam_A(E):
    gm = E['gm']; c = dict(A_committed=all(blob(p) == (REPO / p).read_bytes() for p in COMMITTED)); allk, sel = my_census(gm); C_ = E['census']
    c['A_census'] = bool(sha((R / 'census.csv').read_bytes()) == gm['census_sha256'] and sorted(C_.key) == allk and sorted(C_[C_.selected].key) == sel == sorted(s['key'] for s in gm['selected']))
    ok = True
    for (part, day), rec in E['days'].items():
        ok &= all(rec['sha256'][k] == sha((D / part / f'{day}{suf}').read_bytes()) for k, suf in (('pairs', '_pairs.csv'), ('parity', '_parity.json'), ('ledgers', '_ledgers.jsonl.gz')))
    c['A_days'] = bool(ok); c['A_audit'] = all(E['audit']['outputs'][n] == sha((D / 'measure' / n).read_bytes()) for n in E['audit']['outputs']); return c, {}


def pairs_led(E):
    return [x for x in E['led'] if x['kind'] == 'REVIEW']


def fam_B(E):
    keys = sorted(x['key'] for x in pairs_led(E)); want = sorted(s['key'] for s in E['gm']['selected'])
    bases = sorted((x['day'], int(x['m'])) for x in E['led'] if x['kind'] == 'BASE'); mk = sorted({(s['day'], int(s['m'])) for s in E['gm']['selected']})
    return dict(B_membership=bool(keys == want and len(set(keys)) == len(keys) and sorted(E['pairs'].key) == want and bases == mk)), dict(B=dict(pairs=len(keys), bases=len(bases)))


def qarm(a):
    sg = lambda s: 1. if int(s) == 0 else -1.; I0 = sum(sg(f[2]) * f[3] for f in a['pre_fills'])
    return sum(sg(f[2]) * f[3] * (a['mark'] - f[4]) for f in a['post_raw']) + I0 * (a['mark'] - a['M_fork'])


def fam_Y(E):
    P_ = E['pairs'].set_index('key'); bad = []
    for x in pairs_led(E):
        qr, qa = qarm(x['retain']), qarm(x['alternative'])
        if not (close(qr, x['retain']['Q']) and close(qa, x['alternative']['Q']) and close(qr - qa, P_.loc[x['key']].Y)):
            bad.append(x['key'][:10])
    return dict(Y_values=not bad), dict(Y=dict(bad=bad[:4]))


def shown(book, side, price):
    pr, sz = book[:5], book[5:]; last = pr[4]
    if last >= 0 and (price < last if side == 0 else price > last):
        return 0.
    return float(sum(z for p, z in zip(pr, sz) if p >= 0 and p == price))


def fam_G(E):
    P_ = E['pairs'].set_index('key'); C_ = E['census'].set_index('key'); bad = {k: [] for k in ('callback', 'pre_fork', 'reconciled', 'queue', 'lineage', 'unused', 'features', 'sanity')}
    recs, mine_ok = {}, {}
    for x in pairs_led(E):
        R_, A_ = x['retain'], x['alternative']; k = x['key']; f0 = sum(len(v) for v in bad.values()); r = P_.loc[k]
        if not (len(R_['applied']) == 1 and len(A_['applied']) == 1 and R_['applied'][0][:4] == ['live', x['t'], x['gid'], 'KEEP']
                and A_['applied'][0][:4] == ['live', x['t'], x['gid'], 'PULL'] and R_['applied'][0][4] == A_['applied'][0][4]):
            bad['callback'].append(k)
        if R_['pre_fills'] != A_['pre_fills']:
            bad['pre_fork'].append(k)
        sg = lambda s: 1. if int(s) == 0 else -1.
        for a in (R_, A_):
            I0 = sum(sg(f[2]) * f[3] for f in a['pre_fills']); c0 = -sum(sg(f[2]) * f[3] * f[4] for f in a['pre_fills'])
            if not abs(a['value'] - (c0 + I0 * a['M_fork']) - a['Q']) <= 1e-6 * max(1., abs(a['value'])):
                bad['reconciled'].append(k)
        if not all(rp['refused'] or rp['ahead0'] is None or abs(rp['ahead0'] - shown(rp['book'], rp['side'], rp['price'])) <= 1e-9 for a in (R_, A_) for rp in a['replacements']):
            bad['queue'].append(k)
        if any(int(g) in set(a['lineage']) for a in (R_, A_) for _, g in a['pp_log']):
            bad['lineage'].append(k)
        if R_['suppressed'] == 0:
            key = (x['day'], int(x['m'])); recs.setdefault(key, af_record(x['day'], x['m'], E['part_of'][key]))
            fl = sorted(map(tuple, R_['pre_fills'] + R_['post_raw'])); rf = sorted(map(tuple, recs[key][ARM]['fills']))
            if fl != rf or not close(R_['value'], recs[key][ARM]['value']):
                bad['unused'].append(k)
        fe = x['features']
        if not (fe is not None and fe == x['features_alt'] and all(close(r[f'f_{n}'], v) for n, v in fe.items())):
            bad['features'].append(k)
        else:
            key = (x['day'], int(x['m'])); recs.setdefault(key, af_record(x['day'], x['m'], E['part_of'][key])); end = float(recs[key]['path_info']['end_rel'])
            if not (close(fe['time_left_ms'] + x['t'], end) and fe['rung'] == C_.loc[k].rung and fe['known_rem'] <= fe['size'] + 1e-9 and fe['dist_ticks'] >= 0 and fe['age_ms'] > 0
                    and fe['pressure_ok'] in (0., 1.) and fe['restricted'] in (0., 1.)):
                bad['sanity'].append(k)
        mine_ok[k] = sum(len(v) for v in bad.values()) == f0
    st = dict(zip(E['pairs'].key, E['pairs'].status))
    return dict(G_gates=not any(bad.values()), G_status=bool(all((st[k] == 'ok') == v for k, v in mine_ok.items()))), dict(G={k: [x[:10] for x in v[:3]] for k, v in bad.items()})


def fam_P(E):
    bad = []
    for x in E['led']:
        if x['kind'] == 'BASE':
            r = af_record(x['day'], x['m'], E['part_of'][(x['day'], int(x['m']))])[ARM]; b = x['base']
            if not (b['fills'] == r['fills'] and b['orders'] == r['orders'] and b['value'] == r['value'] and b['fingerprint'] == r['fingerprint']):
                bad.append((x['day'], x['m']))
    return dict(P_parity=not bad), dict(P=dict(bad=bad))


def my_fit(X, y, cand):
    lo = np.quantile(X, cand['winsor_q'][0], axis=0); hi = np.quantile(X, cand['winsor_q'][1], axis=0); Xc = np.minimum(np.maximum(X, lo), hi)
    mu = Xc.mean(axis=0); sd = Xc.std(axis=0); sd[sd == 0] = 1.; Z = (Xc - mu) / sd
    yc = np.minimum(np.maximum(y, np.quantile(y, cand['y_clip_q'][0])), np.quantile(y, cand['y_clip_q'][1])); ym = yc.mean()
    w = np.linalg.lstsq(Z.T @ Z + cand['lambda_per_n'] * len(y) * np.eye(Z.shape[1]), Z.T @ (yc - ym), rcond=None)[0]
    return lambda Xn: ym + ((np.minimum(np.maximum(Xn, lo), hi) - mu) / sd) @ w, w, ym


def fam_M(E):
    gm = E['gm']; S_ = E['scored']; P_ = E['pairs']; cols = [f'f_{f}' for f in gm['features']]; ok = P_[P_.status == 'ok']
    Tr = ok[ok.day.isin(gm['train_days'])]; Ev = ok[ok.day.isin(gm['eval_days'])].copy(); fn, w, ym = my_fit(Tr[cols].values.astype(float), Tr.Y.values.astype(float), gm['candidate'])
    Ev['pred_my'] = fn(Ev[cols].values.astype(float)); se = S_[S_.split == 'eval'].set_index('key'); res = E['audit']['evaluation']
    c = dict(model=bool(np.allclose(w, E['model']['w'], rtol=1e-6, atol=1e-9) and close(ym, E['model']['ym'])),
             preds=bool(all(close(p, se.loc[k].pred) for k, p in zip(Ev.key, Ev.pred_my)) and len(se) == len(Ev)))
    ret = Ev.pred_my > 0; V = float(Ev.Y[ret].sum()); cr = bool(Tr.Y.mean() > 0); rng = np.random.default_rng(gm['draws']['seed']); dr = np.zeros(gm['draws']['n'])
    for _, g in Ev.groupby('day'):
        y = g.Y.values; kk = int((g.pred_my > 0).sum())
        for i in range(gm['draws']['n']):
            dr[i] += y[rng.choice(len(y), size=kk, replace=False)].sum() if kk else 0.
    p = (1 + (dr >= V).sum()) / (1 + len(dr)); per = {d: (float(g.Y[g.pred_my > 0].sum()), float(g.Y.sum())) for d, g in Ev.groupby('day')}
    c['metrics'] = bool(close(V, res['V_rule']) and close(Ev.Y.sum(), res['V_always_retain']) and close(Ev.Y.sum() if cr else 0., res['V_constant']) and close(p, res['p_draw'])
                        and close(((Ev.pred_my - Ev.Y) ** 2).mean(), res['mse_model']) and all(close(per[d][0], res['per_day'][d]['V_rule']) for d in per))
    rr = gm['reading']; n = len(Ev); ins = ((Tr.Y.abs() > TOL).sum() < rr['min_nontie'] or (Ev.Y.abs() > TOL).sum() < rr['min_nontie'] or ret.sum() < rr['min_actions']
                                        or ret.sum() > n - rr['min_actions'])
    ctrl = max(Ev.Y.sum(), 0., Ev.Y.sum() if cr else 0.); daily = all(v > 0 and v > max(t, 0., t if cr else 0.) for v, t in per.values()) and len(per) == len(gm['eval_days'])
    reading = 'INSUFFICIENT' if ins else ('useful_local_discrimination' if (V > 0 and V > ctrl and p <= rr['p_max'] and daily) else 'no_useful_discrimination')
    if not res.get('parity_all', False):
        reading = 'not_reliable'
    c['reading'] = reading == res['reading']
    return {f'M_{k}': v for k, v in c.items()}, dict(M=dict(reading=reading, V=V, p=p))


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
    rev, _ = cpu([p for n in gm['line_reviews'] for p in glob.glob(str(REPO / 'predictfun/reviews/automatic' / n / '*.log'))]); line += rev + gm['prior_unlogged_s'] / 3600.
    return dict(I_resources=bool(own + cp['review_reserve'] <= cp['batch_core_hours'] and peak < 1536 and line + own + cp['review_reserve'] <= cp['line_ceiling'])), \
        dict(I=dict(batch=own, line_before=line, line_with=line + own, with_reserve=line + own + cp['review_reserve'], peak_mib=peak))


FAMS = dict(A=fam_A, B=fam_B, Y=fam_Y, G=fam_G, P=fam_P, M=fam_M, I=fam_I)


def run_checks(E):
    chk, det = {}, {}
    for k in FAMS:
        c, d = FAMS[k](E); chk.update(c); det.update(d)
    return dict(checks=chk, details=det, all_ok=bool(all(chk.values())))


def _mut(E, pred, mutate):
    L_ = json.loads(json.dumps(E['led']))
    for y in L_:
        if y['kind'] == 'REVIEW' and pred(y):
            mutate(y); break
    else:
        raise KeyError('no row to corrupt')
    E['led'] = L_


def _df(E, key, col, fn, sel=None):
    X_ = E[key].copy(); i = X_.index[sel(X_)][0] if sel is not None else X_.index[0]; X_.loc[i, col] = fn(X_.loc[i, col]); E[key] = X_


def _fake(E):
    E['_root'] = Path(tempfile.mkdtemp(dir=str(D))); (E['_root'] / 'logs').mkdir()
    (E['_root'] / 'logs' / 'fake.log').write_text('2026-11-22T00:00:00Z tag=fake rc=0 peak_rss_mib=100 wall_s=9999.0 cpu_user_s=9999.0\n')


CASES = [
    ('a_census_key_changed', 'A', 'A_census', lambda E: E.__setitem__('census', E['census'].assign(key=E['census'].key.where(E['census'].index != 0, '0' * 64)))),
    ('an_audit_hash_changed', 'A', 'A_audit', lambda E: E.__setitem__('audit', dict(E['audit'], outputs=dict(E['audit']['outputs'], **{'pairs.csv': '0' * 64})))),
    ('a_pair_dropped', 'B', 'B_membership', lambda E: E.__setitem__('led', [x for x in E['led'] if x['key'] != pairs_led(E)[0]['key']])),
    ('a_post_fill_changed', 'Y', 'Y_values', lambda E: _mut(E, lambda y: len(y['retain']['post_raw']) > 0, lambda y: y['retain']['post_raw'][0].__setitem__(3, y['retain']['post_raw'][0][3] + 1.))),
    ('a_callback_action_changed', 'G', 'G_gates', lambda E: _mut(E, lambda y: True, lambda y: y['retain']['applied'][0].__setitem__(3, 'PULL'))),
    ('a_callback_time_changed', 'G', 'G_gates', lambda E: _mut(E, lambda y: True, lambda y: y['alternative']['applied'][0].__setitem__(1, y['t'] + 50.))),
    ('features_differ_between_arms', 'G', 'G_gates', lambda E: _mut(E, lambda y: y['features'] is not None, lambda y: y.__setitem__('features_alt', dict(y['features'], age_ms=y['features']['age_ms'] + 1.)))),
    ('a_lineage_pull', 'G', 'G_gates', lambda E: _mut(E, lambda y: len(y['alternative']['lineage']) > 0, lambda y: y['alternative']['pp_log'].append([0., y['alternative']['lineage'][0]]))),
    ('a_base_differs', 'P', 'P_parity', lambda E: E.__setitem__('led', [dict(x, base=dict(x['base'], value=x['base']['value'] + 1.)) if x['kind'] == 'BASE' else x for x in E['led']])),
    ('a_model_weight_changed', 'M', 'M_model', lambda E: E.__setitem__('model', dict(E['model'], w=[v + 1. for v in E['model']['w']]))),
    ('a_reading_flipped', 'M', 'M_reading', lambda E: E.__setitem__('audit', dict(E['audit'], evaluation=dict(E['audit']['evaluation'], reading='useful_local_discrimination' if
                                                                                                           E['audit']['evaluation']['reading'] != 'useful_local_discrimination' else 'INSUFFICIENT')))),
    ('resources_over_the_cap', 'I', 'I_resources', _fake),
]


def falsify(E, base):
    out = {}
    for name, fam, target, fn in CASES:
        X_ = dict(E); refused = False
        try:
            fn(X_); c, _ = FAMS[fam](X_, X_['_root']) if fam == 'I' else FAMS[fam](X_); refused = not c[target]
        except (IndexError, KeyError) as e_:
            print(f'falsifier case {name}: corruption not applied ({e_!r})')
        finally:
            if '_root' in X_:
                shutil.rmtree(X_['_root'], ignore_errors=True)
        out[name] = dict(family=fam, target=target, clean=bool(base['checks'][target]), refused=bool(refused))
    return dict(cases=out, falsifier_ok=bool(base['all_ok'] and all(v['clean'] and v['refused'] for v in out.values())))


def main(argv):
    E = load(); r_ = run_checks(E); fz = falsify(E, r_); rd = Path(argv[0]) if argv else R
    for n, obj in (('checks.json', r_), ('checks_falsifier.json', fz)):
        p = rd / n; assert not p.exists(), f'{p} exists (write-once)'; p.write_text(json.dumps(obj, indent=1, default=str) + '\n')
    print(json.dumps(dict(all_ok=r_['all_ok'], failed=[k for k, v in r_['checks'].items() if not v], falsifier_ok=fz['falsifier_ok'],
                          cases={k: (v['clean'], v['refused']) for k, v in fz['cases'].items()}, details=r_['details']), indent=1, default=str)[:3500])


if __name__ == '__main__':
    main(sys.argv[1:])
