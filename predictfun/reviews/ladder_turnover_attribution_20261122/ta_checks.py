"""INDEPENDENT CHECKS of PAIRED-TURNOVER-ATTRIBUTION-v1 (declaration.md; session aea5bfb9, 2026-11-22). Recomputed from the pilot's saved ledgers,
pairs and replacement rows with this checker's own code (a deque FIFO, tuple-multiset events, its own delay rule); the builder is not imported.
 A identity: the frozen files committed; the pilot day records = the manifest (their ledger bytes = the records); the pilot measure copies =
   its audit; the outputs = this audit;
 D decomposition: per arm matched closing profit, residual, matched shares and carried matched profit; matched + residual = the saved Q;
 Y per pair: the saved pilot Y = Q_retain - Q_alternative = dMatched + dResidual = the event-difference value; event counts;
 L delays: the corrected first-attempt / accepted delays from the replacement rows; the arms that differ from the pilot's stored field are
   exactly the review's twelve (replacement_delay_corrections.csv, committed at d799cb77), each at its corrected value;
 F focal: the per-arm focal fill-group classes (own END sign x round trip) and their sums;
 S summary rows and the computed day signs;  I resources: this follow-up <= 0.04 core-hours, peak < 1,536 MiB; the pilot batch + its 0.03
   review reserve + this + 0.02 <= 0.40; the line + those <= 2.85.
falsify(): ten corruptions, each refused by its family while the clean evidence passes. Usage: ta_checks.py [out_dir]"""
import collections
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

REPO = Path('/home/yuqing/ctaNew'); R = REPO / 'predictfun/reviews/ladder_turnover_attribution_20261122'; D = REPO / 'data/predict_fun/research/ladder_turnover_attribution_v1_20261122'
LC = REPO / 'data/predict_fun/research/ladder_long_continuation_v1_20261122'; LCR = REPO / 'predictfun/reviews/ladder_long_continuation_20261122'
REV = 'predictfun/reviews/automatic/20261122_long_continuation_review/replacement_delay_corrections.csv'; TOL = 1e-6
COMMITTED = ['predictfun/research/pf_ladder_turnover_attribution.py', 'predictfun/research/pf_ladder_turnover_attribution_selftest.py', 'predictfun/research/pf_ladder_turnover_attribution_mutants.py'] + \
    [f'predictfun/reviews/ladder_turnover_attribution_20261122/{n}' for n in ('declaration.md', 'manifest.json', 'ta_checks.py')]
OUTS = ('arms', 'pairs', 'lots', 'focal', 'delays', 'summary')
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')


def sha(b):
    return hashlib.sha256(b).hexdigest()


def blob(rel):
    r = subprocess.run(['git', '-C', str(REPO), 'show', f'HEAD:{rel}'], capture_output=True); return r.stdout if r.returncode == 0 else None


def close(a, b, tol=1e-6):
    a, b = float(a), float(b)
    return (math.isnan(a) and math.isnan(b)) or (not math.isnan(a) and not math.isnan(b) and abs(a - b) <= tol * max(1., abs(a), abs(b)))


def load():
    gm = json.loads((R / 'manifest.json').read_text()); gl = json.loads((LCR / 'manifest.json').read_text()); E = dict(gm=gm, src={}, led=[])
    for part, days in gl['parts'].items():
        for day in days:
            rb = (LC / part / f'{day}.json').read_bytes(); r = json.loads(rb); lb = (LC / part / f'{day}_ledgers.jsonl.gz').read_bytes()
            E['src'][f'{part}/{day}'] = bool(gm['lc_records'].get(f'{part}/{day}') == sha(rb) and r['sha256']['ledgers'] == sha(lb))
            E['led'] += [json.loads(x) for x in gzip.decompress(lb).decode().splitlines() if '"kind": "BASE"' not in x]
    au = json.loads((LC / 'measure' / 'audit.json').read_text()); E['src']['measure'] = all(au['outputs'][n] == sha((LC / 'measure' / n).read_bytes()) for n in ('pairs.csv', 'replacements.csv', 'arms.csv'))
    rd = lambda p: pd.read_csv(p, dtype={'day': str, 'side': str})
    E.update(lp=rd(LC / 'measure' / 'pairs.csv'), la=rd(LC / 'measure' / 'arms.csv'), lr=rd(LC / 'measure' / 'replacements.csv'), audit=json.loads((D / 'audit.json').read_text()))
    for n in OUTS:
        E[n] = rd(D / f'{n}.csv')
    cb = blob(REV); E['rev'] = pd.read_csv(io.BytesIO(cb), dtype={'day': str}) if cb is not None else pd.DataFrame(); return E


def fifo(a):
    """An independent FIFO: (matched, residual, matched shares, carried matched profit, closes as (open gid, open t, close gid, close t, profit))."""
    s_ = lambda side: 1. if int(side) == 0 else -1.; key = lambda f: (float(f[0]), int(f[5])); dq = collections.deque()
    for f in sorted(a['pre_fills'], key=key):
        s, q = s_(f[2]), float(f[3])
        while q > 1e-12 and dq and dq[0][0] != s:
            x = min(q, dq[0][1]); dq[0][1] -= x; q -= x
            if dq[0][1] <= 1e-12:
                dq.popleft()
        if q > 1e-12:
            dq.append([s, q, None, float(f[0]), int(f[1])])
    for lot in dq:
        lot[2] = float(a['M_fork'])
    m = mq = mc = 0.; cl = []
    for f in sorted(a['post_raw'], key=key):
        s, q, p = s_(f[2]), float(f[3]), float(f[4])
        while q > 1e-12 and dq and dq[0][0] != s:
            x = min(q, dq[0][1]); pr = x * (p - dq[0][2]) * dq[0][0]; m += pr; mq += x
            if len(dq[0]) == 5:
                mc += pr
            cl.append((dq[0][4], dq[0][3], int(f[1]), float(f[0]), pr, len(dq[0]) == 5)); dq[0][1] -= x; q -= x
            if dq[0][1] <= 1e-12:
                dq.popleft()
        if q > 1e-12:
            dq.append([s, q, p, float(f[0]), int(f[1]), 'new'])
    res = sum(l[1] * (float(a['mark']) - l[2]) * l[0] for l in dq)
    return m, res, mq, mc, cl


def fam_A(E):
    c = dict(A_committed=all(blob(p) == (REPO / p).read_bytes() for p in COMMITTED), A_inputs=bool(E['src'] and all(E['src'].values())))
    c['A_outputs'] = all(E['audit']['outputs'][f'{n}.csv'] == sha((D / f'{n}.csv').read_bytes()) for n in OUTS); return c, {}


def fam_D(E):
    A_ = E['arms'].set_index(['key', 'arm']); bad = []
    for x in E['led']:
        for name in ('retain', 'alternative'):
            m, res, mq, mc, _ = fifo(x[name]); r = A_.loc[(x['key'], name)]
            if not (close(m, r.matched) and close(res, r.residual) and close(mq, r.matched_q) and close(mc, r.matched_carried) and close(m + res, x[name]['Q'])):
                bad.append((x['key'][:10], name))
    return dict(D_decomposition=not bad and len(A_) == 2 * len(E['led'])), dict(D=dict(bad=bad[:4]))


def fam_Y(E):
    P_ = E['pairs'].set_index('key'); lp = E['lp'].set_index('key'); bad = []
    for x in E['led']:
        mr, rr, *_ = fifo(x['retain']); ma, ra, *_ = fifo(x['alternative']); Y = x['retain']['Q'] - x['alternative']['Q']
        ev = lambda a: collections.Counter((round(float(f[0]), 6), int(f[5]), int(f[2]), round(float(f[4]), 9), round(float(f[3]), 9)) for f in a['post_raw'])
        cr, ca = ev(x['retain']), ev(x['alternative']); vals = {}
        for a in (x['retain'], x['alternative']):
            for f in a['post_raw']:
                vals[(round(float(f[0]), 6), int(f[5]), int(f[2]), round(float(f[4]), 9), round(float(f[3]), 9))] = (1. if int(f[2]) == 0 else -1.) * float(f[3]) * (float(a['mark']) - float(f[4]))
        yv = sum(n * vals[k] for k, n in (cr - ca).items()) - sum(n * vals[k] for k, n in (ca - cr).items()); r = P_.loc[x['key']]
        if not (close(Y, lp.loc[x['key']].Y) and close(Y, r.Y) and close(mr - ma, r.dMatched) and close(rr - ra, r.dResidual) and close(yv, Y) and close(yv, r.Y_events)
                and int(sum((cr & ca).values())) == int(r.events_common) and int(sum((cr - ca).values())) == int(r.events_retain_only)):
            bad.append(x['key'][:10])
    return dict(Y_values=not bad and len(P_) == 64), dict(Y=dict(bad=bad[:4]))


def my_delay(rows, kind, arm, t):
    rows = sorted(rows, key=lambda r: (float(r['submit']), int(r['gid'])))
    keep = [r for r in rows if (float(r['submit']) > t if (kind == 'ADMISSION' and arm == 'retain') else float(r['submit']) >= t)]
    return (float(keep[0]['submit']) - t if keep else np.nan), next((float(r['submit']) - t for r in keep if not bool(r['refused'])), np.nan)


def fam_L(E):
    t = dict(zip(E['lp'].key, E['lp'].t)); kind = dict(zip(E['lp'].key, E['lp'].kind)); Dl = E['delays'].set_index(['key', 'arm']); bad = []; differ = set()
    st = E['la'].set_index(['key', 'arm']).first_rep_delay
    for (k, arm), g in E['lr'].groupby(['key', 'arm']):
        att, acc = my_delay(g.to_dict('records'), kind[k], arm, float(t[k])); r = Dl.loc[(k, arm)]
        if not (close(att, r.first_attempt_delay) and close(acc, r.first_accepted_delay)):
            bad.append((k[:10], arm))
        if not close(att, st.loc[(k, arm)]):
            differ.add((k, arm))
    rv = E['rev']; want = {(k, 'alternative') for k in rv.key} if len(rv) else set()
    at_ok = bool(len(rv) == 12 and all(close(Dl.loc[(k, 'alternative')].first_attempt_delay, c) for k, c in zip(rv.key, rv.correct)))
    return dict(L_delays=not bad and differ == want and at_ok), dict(L=dict(bad=bad[:4], differ=len(differ), review_rows=len(rv)))


def fam_F(E):
    my = collections.Counter(); sums = collections.defaultdict(float)
    for x in E['led']:
        for name in ('retain', 'alternative'):
            a = x[name]; lin = set(int(g) for g in a['lineage']); _, _, _, _, cl = fifo(a); M1 = float(a['mark']); grp = collections.defaultdict(list)
            for f in a['post_raw']:
                if int(f[1]) in lin:
                    grp[(float(f[0]), int(f[1]))].append(f)
            for (t, g), fs in grp.items():
                own = sum((1. if int(f[2]) == 0 else -1.) * float(f[3]) * (M1 - float(f[4])) for f in fs)
                hit = [c for c in cl if (c[2] == g and c[3] == t) or (c[0] == g and c[1] == t and not c[5])]; rt = sum(c[4] for c in hit)
                k = ('own_end_negative' if own < -TOL else ('own_end_positive' if own > TOL else 'own_end_zero')) + '_' + \
                    (('round_trip_profitable' if rt > TOL else ('round_trip_loss' if rt < -TOL else 'round_trip_flat')) if hit else 'round_trip_none')
                my[(x['kind'], name, k)] += 1; sums[(x['kind'], name, k, 'own')] += own; sums[(x['kind'], name, k, 'rt')] += rt
    F_ = E['focal']; ok = len(F_) == len(my)
    for r in F_.to_dict('records'):
        kk = (r['kind'], r['arm'], r['cls']); ok &= bool(my.get(kk) == int(r['n']) and close(sums[kk + ('own',)], r['own']) and close(sums[kk + ('rt',)], r['round_trip']))
    return dict(F_focal=bool(ok)), dict(F=dict(classes=len(my)))


def fam_S(E):
    P_ = E['pairs']; S_ = E['summary']; bad = []
    for r in S_.to_dict('records'):
        g = P_
        for k in ('kind', 'day', 'side'):
            if str(r[k]) != 'all':
                g = g[g[k].astype(str) == str(r[k])]
        x = dict(pairs=len(g), retain_better=int((g.Y > TOL).sum()), alternative_better=int((g.Y < -TOL).sum()), tie=int((g.Y.abs() <= TOL).sum()), Y_sum=g.Y.sum(),
                 dMatched_sum=g.dMatched.sum(), dResidual_sum=g.dResidual.sum(), dMatched_abs_sum=g.dMatched.abs().sum(), dResidual_abs_sum=g.dResidual.abs().sum())
        if not all(close(r[k], v) for k, v in x.items()):
            bad.append((r['kind'], r['day'], r['side']))
    ds = E['audit']['predicates']['day_signs']; mine = {}
    for k, g in P_.groupby('kind'):
        s = g.groupby('day').Y.sum(); mine[k] = 'POSITIVE_EVERY_DAY' if (s > TOL).all() else ('NEGATIVE_EVERY_DAY' if (s < -TOL).all() else 'MIXED')
    return dict(S_summary=not bad and len(S_) > 0, S_day_signs=all(ds[k]['sign'] == v for k, v in mine.items()) and set(ds) == set(mine)), dict(S=dict(bad=bad[:3], day_signs=mine))


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
    pilot, _ = cpu(glob.glob(str(LCR / 'logs' / '**' / '*.log'), recursive=True))
    line, _ = cpu([p for n in gm['line_receipts'] for p in glob.glob(str(REPO / 'predictfun/reviews' / n / 'logs' / '**' / '*.log'), recursive=True)])
    rev, _ = cpu([p for n in gm['line_reviews'] for p in glob.glob(str(REPO / 'predictfun/reviews/automatic' / n / '*.log'))]); line += rev + gm['prior_unlogged_s'] / 3600.
    batch = pilot + 0.03 + own + cp['review_reserve']
    return dict(I_resources=bool(own <= cp['followup_core_hours'] and peak < 1536 and batch <= cp['batch_core_hours'] and line + own + cp['review_reserve'] <= cp['line_ceiling'])), \
        dict(I=dict(followup_core_hours=own, pilot_batch=pilot, batch_with_reserves=batch, line_before=line, line_with=line + own, with_reserve=line + own + cp['review_reserve'], peak_mib=peak))


FAMS = dict(A=fam_A, D=fam_D, Y=fam_Y, L=fam_L, F=fam_F, S=fam_S, I=fam_I)


def run_checks(E):
    chk, det = {}, {}
    for k in FAMS:
        c, d = FAMS[k](E); chk.update(c); det.update(d)
    return dict(checks=chk, details=det, all_ok=bool(all(chk.values())))


def _df(E, key, col, fn, sel=None):
    X_ = E[key].copy(); i = X_.index[sel(X_)][0] if sel is not None else X_.index[0]; X_.loc[i, col] = fn(X_.loc[i, col]); E[key] = X_


def _fake(E):
    E['_root'] = Path(tempfile.mkdtemp(dir=str(D))); (E['_root'] / 'logs').mkdir()
    (E['_root'] / 'logs' / 'fake.log').write_text('2026-11-22T00:00:00Z tag=fake rc=0 peak_rss_mib=100 wall_s=999.0 cpu_user_s=999.0\n')


CASES = [
    ('an_output_hash_changed', 'A', 'A_outputs', lambda E: E.__setitem__('audit', dict(E['audit'], outputs=dict(E['audit']['outputs'], **{'pairs.csv': '0' * 64})))),
    ('a_matched_profit_changed', 'D', 'D_decomposition', lambda E: _df(E, 'arms', 'matched', lambda v: v + 1.)),
    ('a_residual_changed', 'D', 'D_decomposition', lambda E: _df(E, 'arms', 'residual', lambda v: v - 1.)),
    ('a_dMatched_changed', 'Y', 'Y_values', lambda E: _df(E, 'pairs', 'dMatched', lambda v: v + 1.)),
    ('an_event_count_changed', 'Y', 'Y_values', lambda E: _df(E, 'pairs', 'events_common', lambda v: v + 1)),
    ('a_delay_left_uncorrected', 'L', 'L_delays', lambda E: _df(E, 'delays', 'first_attempt_delay', lambda v: v + 750., sel=lambda d: (d.arm == 'alternative') & (d.kind == 'ADMISSION'))),
    ('a_focal_class_count_changed', 'F', 'F_focal', lambda E: _df(E, 'focal', 'n', lambda v: v + 1)),
    ('a_summary_sum_changed', 'S', 'S_summary', lambda E: _df(E, 'summary', 'dResidual_sum', lambda v: v + 1.)),
    ('a_day_sign_flipped', 'S', 'S_day_signs', lambda E: E.__setitem__('audit', dict(E['audit'], predicates=dict(E['audit']['predicates'], day_signs={
        k: dict(v, sign='POSITIVE_EVERY_DAY' if v['sign'] != 'POSITIVE_EVERY_DAY' else 'MIXED') for k, v in E['audit']['predicates']['day_signs'].items()})))),
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
