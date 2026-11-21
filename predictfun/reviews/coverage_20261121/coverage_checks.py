"""PI0-COVERAGE-v1: THE DECLARED CHECKS (2026-11-21, session aea5bfb9). Declared in the freeze, before the pilot and any label. Each check recomputes
with its own code (only the module's constants and committed_blob / committed_equal are imported) and is false when it cannot run.
 A identity: the module, selftest, harness, declaration, manifest, launcher, projection tool and this file are the committed ones; every
   partition / pilot record's rows and census bytes match it, its sources are the final files, its manifest the committed one.
 B population: per date the world sample recomputed from the raw orders file (one original per side; per UTC hour of t0 the PER_HOUR lowest
   sha256(DOMAIN|world|key)) equals the manifest's selected worlds and pi_world; the pilot is the PILOT_PER_DAY lowest ranks; the partitions
   hold every selected world once, inside their hour blocks.
 C queries: per partition the query sample recomputed from its census rows (the bound worlds only; cells by date x side x the world's hour x
   category; the PER_CELL lowest sha256(DOMAIN|query|key)) equals its rows: keys, N_cell, n_cell, pi_cell, pi_world, weight.
 D labels: every row with status ok: the target recomputed from its hold fills (at or after eff) equals it; eff is the hold activation
   (absent) or the captured modeled effect (review); every check true; a status other than ok names a false check or a world / reroll status.
 E determinism: the pilot's census rows of its worlds equal the build's census rows of the same worlds.
 F aggregates: forecasts, calibration, coverage, actions and all null draws recomputed equal the saved tables.
 G readings: the predicates recomputed equal audit.json's.
 H resources and suites: every run_light job under logs/ (and the source extension's) below 2,048 MiB, the round within 8 core-hours, every
   record's high-water <= 1,536 MiB; pf_mutant_summary.judge passes both harnesses' final records against the final sources.
FALSIFIER (falsify(), writes checks_falsifier.json): every check true on clean copies, the named check false after one corruption of a real
 copy: substituted rows bytes (A); a manifest world dropped (B); a row weight, rehashed (C); a row target value, rehashed (D); a build census
 row of a pilot world, rehashed (E); a forecasts MSE, an action J and a null draw (F, three cases); a reading (G); an over-budget job and a
 copied mutant record (H, two cases).
SELFTEST (selftest()): C, D and F on hand rows pass clean and fail after a corruption.
Usage (run_light): coverage_checks.py [selftest]  (no argument: the checks, then the falsifier)"""
import copy
import glob
import gzip
import hashlib
import io
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; RES = Path('/home/yuqing/ctaNew/predictfun/research'); REPO = RES.parents[1]; sys.path.insert(0, str(RES))
import pf_coverage as M                                  # noqa: E402  constants and committed_blob / committed_equal only

pd, RT = M.pd, M.RT
LOGS = HERE / 'logs'; SRC_LOGS = HERE / 'source' / 'logs'
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
REAL = dict(out=M.OUT, rec=REPO / M.RECEIPT, logs=LOGS, src_logs=SRC_LOGS, real=True)
STRC = {c: str for c in ('day', 'side', 'cat', 'population', 'rule', 'dim', 'cell', 'hbin', 'status')}


def ck(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def h(kind, key):
    return hashlib.sha256(f'{M.DOMAIN}|{kind}|{ck(key)}'.encode()).hexdigest()


def gz_rows(b):
    return [json.loads(x) for x in gzip.decompress(b).decode().splitlines() if x]


def close(a, b, tol=1e-9):
    an = a is None or (isinstance(a, float) and np.isnan(a)); bn = b is None or (isinstance(b, float) and np.isnan(b))
    if an or bn:
        return bool(an and bn)
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    return bool(abs(a - b) <= tol * max(1., abs(a), abs(b)))


def load(P):
    out, rec = Path(P['out']), Path(P['rec']); man = json.loads((rec / 'manifest.json').read_text()); parts, pilots = {}, {}
    for p in man['partitions']:
        k = int(p['part']); b = out / 'parts' / f'part_{k:03d}'
        if (Path(str(b) + '.json')).exists():
            parts[k] = (json.loads(Path(str(b) + '.json').read_text()), Path(str(b) + '_rows.jsonl.gz').read_bytes(), Path(str(b) + '_census.jsonl.gz').read_bytes())
    for day in man['days']:
        b = out / 'pilot' / f'pilot_{day}'
        if Path(str(b) + '.json').exists():
            pilots[day] = (json.loads(Path(str(b) + '.json').read_text()), Path(str(b) + '_rows.jsonl.gz').read_bytes(), Path(str(b) + '_census.jsonl.gz').read_bytes())
    rd = lambda n: pd.read_csv(out / f'{n}.csv', float_precision='round_trip', dtype=STRC) if (out / f'{n}.csv').exists() else None
    return dict(man=man, parts=parts, pilots=pilots, audit=json.loads((out / 'audit.json').read_text()) if (out / 'audit.json').exists() else None,
                FC=rd('forecasts'), CA=rd('calibration'), CV=rd('coverage'), AC=rd('actions'), ND=rd('null_draws'))


# ---------------------------------------------------------------------------------------------------------------------- recomputations
def worlds_from_orders(day, kp):
    O = pd.read_parquet(Path(kp) / day / 'orders.parquet', columns=['m', 't', 'side', 'oid']); out = []
    for (m, t), g in O.groupby(['m', 't'], sort=True):
        if sorted(g.side.astype(int)) == [0, 1]:
            t_ms = int(round(float(t))); out.append(dict(day=day, m=int(m), t_ms=t_ms, hour=int(t_ms // 3600000 % 24)))
    return out


def world_sample(W, per_hour):
    by = {}
    for w in W:
        by.setdefault(w['hour'], []).append(w)
    sel = {}
    for hr, ws in by.items():
        r = sorted(ws, key=lambda w: (h('world', dict(day=w['day'], m=w['m'], t_ms=w['t_ms'])), ck(dict(day=w['day'], m=w['m'], t_ms=w['t_ms'])))); n = min(per_hour, len(r))
        for w in r[:n]:
            sel[(w['day'], w['m'], w['t_ms'])] = dict(hour=hr, pi_world=n / len(r), rank=h('world', dict(day=w['day'], m=w['m'], t_ms=w['t_ms'])))
    return sel


def queries_from_census(cens, hours, pis):
    by = {}
    for r in cens:
        if r.get('cat') is not None:
            by.setdefault((r['day'], int(r['side']), hours[(r['day'], int(r['m']), int(r['t_ms']))], r['cat']), []).append(r)
    out = {}
    for cell, rs in by.items():
        key = lambda r: dict(day=r['day'], m=int(r['m']), t_ms=int(r['t_ms']), side=int(r['side']), t=int(r['t']), cat=r['cat'])
        o = sorted(rs, key=lambda r: (h('query', key(r)), ck(key(r)))); n = min(M.PER_CELL, len(o)); pi = n / len(o)
        for r in o[:n]:
            pw = pis[(r['day'], int(r['m']), int(r['t_ms']))]; out[ck(key(r))] = dict(N_cell=len(o), n_cell=n, pi_cell=pi, pi_world=pw, weight=1. / (pw * pi))
    return out


def target_of(fills, gid, side, mark, eff, ref):
    sg = 1. if int(side) == 0 else -1.; f = [x for x in fills if int(x[1]) == int(gid) and float(x[0]) >= float(eff)]
    q = sum(float(x[3]) for x in f); v = sum(sg * float(x[3]) * (float(mark) - float(x[4])) for x in f); qm = sum(sg * float(x[3]) * (float(mark) - float(ref)) for x in f)
    return q, v, qm


def wmean(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float); m = np.isfinite(x) & np.isfinite(w)
    return float((x[m] * w[m]).sum() / w[m].sum()) if m.any() and w[m].sum() > 0 else None


def label_frame(rows):
    out = []
    for r in rows:
        st, tg = r.get('stage') or {}, r.get('target') or {}; mo = st.get('moments') or {}; a = st.get('inputs')
        z = float(tg.get('q', 0.) > 1e-12) if tg else np.nan; tte = float(a[RT.FEATS.index('t_to_end')]) if a else np.nan
        hb = next((f'({lo:.0f},{hi:.0f}]' for lo, hi in M.HBINS if lo < tte <= hi), 'outside') if a else 'none'
        out.append(dict(day=r['day'], m=int(r['m']), t_ms=int(r['t_ms']), side=int(r['side']), cat=r['cat'], hour=int(r['hour']), weight=float(r['weight']), status=r['status'],
                        dec=st.get('dec_status'), P=mo.get('P'), Eq=mo.get('Eq'), Eqm=mo.get('Eqm'), V=mo.get('V'), edge=mo.get('edge'), q=tg.get('q'), value=tg.get('value'),
                        M=(tg.get('qm') if z == 1. else 0.) if tg else np.nan, z=z, hbin=hb))
    D = pd.DataFrame(out); D['ok'] = (D.status == 'ok') & D.V.notna() & (D.dec == 'model')
    return D


def forecast_rows(D):
    out = []; E = D[D.ok]
    for cat, g0 in E.groupby('cat', sort=True):
        pool = wmean(g0.z, g0.weight)
        views = [('all', 'all', g0)] + [(str(d), 'all', g) for d, g in g0.groupby('day', sort=True)] + [('all', str(s), g) for s, g in g0.groupby('side', sort=True)] + \
                [(str(d), str(s), g) for (d, s), g in g0.groupby(['day', 'side'], sort=True)]
        for day, side, g in views:
            w = g.weight; ev = g.V - g.value; em = g.Eqm - g.M
            out.append(dict(cat=cat, day=day, side=side, n=int(len(g)), fills=int((g.z > 0).sum()), weight=float(w.sum()), brier=wmean((g.P - g.z) ** 2, w),
                            brier_pooled=wmean((pool - g.z) ** 2, w), bias_V=wmean(ev, w), mse_V=wmean(ev ** 2, w), mse_zero=wmean(g.value ** 2, w),
                            mse_mech=wmean((g.edge * g.Eq - g.value) ** 2, w), bias_m=wmean(em, w), mse_m=wmean(em ** 2, w), mean_value=wmean(g.value, w)))
    return out


def action_rows(D, draws, seed):
    E = D[D.ok].reset_index(drop=True); out, nul = [], {}
    pops = [(c, (c,)) for c in M.CATS] + [('entry_combined', ('absent_POST', 'absent_WAIT')), ('review_combined', ('review_KEEP', 'review_PULL'))]
    for name, cats in pops:
        g = E[E.cat.isin(cats)].reset_index(drop=True)
        if not len(g):
            continue
        w, v = g.weight.to_numpy(float), g.value.to_numpy(float); W = float(w.sum())
        rule = np.array([(x > 0) if c.startswith('absent') else (x >= 0) for c, x in zip(g.cat, g.V.astype(float))])
        for r, m in (('none', np.zeros(len(g), bool)), ('all', np.ones(len(g), bool)), ('pi0_rule', rule)):
            out.append(dict(population=name, rule=r, n=int(len(g)), chosen=int(m.sum()), chosen_weight=float((w * m).sum()), J=float((w * m * v).sum() / W)))
        rng = np.random.default_rng(seed); keys = sorted(set(zip(g.day.astype(str), g.side.astype(int), g.hour.astype(int))), key=str)
        grp = [np.flatnonzero((g.day.astype(str) == d).to_numpy() & (g.side.astype(int) == s).to_numpy() & (g.hour.astype(int) == hh).to_numpy()) for d, s, hh in keys]
        ks = [int(rule[x].sum()) for x in grp]; J = []
        for _ in range(draws):
            m = np.zeros(len(g), bool)
            for x, k in zip(grp, ks):
                if k:
                    m[rng.choice(x, k, replace=False)] = True
            J.append(float((w * m * v).sum() / W))
        free = [(x, k) for x, k in zip(grp, ks) if 0 < k < len(x)]
        nul[name] = (dict(free_strata=len(free), free_rows=int(sum(len(x) for x, _ in free)), null_p95=float(np.percentile(J, 95)), J=float((w * rule * v).sum() / W)), J)
    return out, nul


# ---------------------------------------------------------------------------------------------------------------------- the checks
def check_A(P, D, R):
    ok = {}
    for k, (rec, rb, cb) in list(D['parts'].items()) + [(f'pilot_{d}', v) for d, v in D['pilots'].items()]:
        ok[str(k)] = bool(hashlib.sha256(rb).hexdigest() == rec['rows_sha256'] and hashlib.sha256(cb).hexdigest() == rec['census_sha256']
                          and (not P['real'] or (rec['sources'] == M.sources() and rec['manifest_sha256'] == M.C.sha(Path(P['rec']) / 'manifest.json'))))
    if P['real']:
        ok['committed'] = bool(all(M.committed_equal(f'predictfun/research/{n}', RES / n) for n in ('pf_coverage.py', 'pf_coverage_selftest.py', 'pf_coverage_mutants.py'))
                               and all(M.committed_equal(f'{M.RECEIPT}/{n}', Path(P['rec']) / n) for n in ('declaration.md', 'manifest.json', 'coverage_checks.py', 'launch_coverage.sh',
                                                                                                             'tools/pilot_projection_cov.py')))
    R['A'] = dict(n=len(ok), failures=[k for k, v in ok.items() if not v][:5])
    return bool(ok) and all(ok.values())


def check_B(P, D, R):
    man = D['man']; bad = []
    for day in man['days']:
        sel = world_sample(worlds_from_orders(day, man['sample'][day]['kp_dir']), man['per_hour']); ms = {(x['day'], int(x['m']), int(x['t_ms'])): x for x in man['sample'][day]['selected']}
        if set(sel) != set(ms) or not all(close(sel[k]['pi_world'], ms[k]['pi_world']) and int(sel[k]['hour']) == int(ms[k]['hour']) for k in sel):
            bad.append((day, 'selected')); continue
        pil = sorted(sel, key=lambda k: (sel[k]['rank'], ck(dict(day=k[0], m=k[1], t_ms=k[2]))))[:man['pilot_per_day']]
        if [ck(dict(day=k[0], m=k[1], t_ms=k[2])) for k in pil] != [ck(x) for x in man['sample'][day]['pilot']]:
            bad.append((day, 'pilot'))
    allsel = [(x['day'], int(x['m']), int(x['t_ms'])) for d in man['days'] for x in man['sample'][d]['selected']]
    mem = [(x['day'], int(x['m']), int(x['t_ms'])) for p in man['partitions'] for x in p['members']]
    hrs = {(x['day'], int(x['m']), int(x['t_ms'])): int(x['hour']) for d in man['days'] for x in man['sample'][d]['selected']}
    if sorted(mem) != sorted(allsel) or len(mem) != len(set(mem)) or not all(p['hours'][0] <= hrs[(x['day'], int(x['m']), int(x['t_ms']))] <= p['hours'][1] for p in man['partitions'] for x in p['members']):
        bad.append(('partitions',))
    R['B'] = dict(failures=bad[:5])
    return not bad


def _hours_pis(man):
    sel = {(x['day'], int(x['m']), int(x['t_ms'])): x for d in man['days'] for x in man['sample'][d]['selected']}
    return {k: int(v['hour']) for k, v in sel.items()}, {k: float(v['pi_world']) for k, v in sel.items()}


def check_C(P, D, R):
    hours, pis = _hours_pis(D['man']); bad = []
    for k, (rec, rb, cb) in D['parts'].items():
        okw = {kk for kk, st in rec['worlds'].items() if st == 'ok' and rec['binds'].get(kk, {}).get('binding_bad') == 0 and rec['binds'].get(kk, {}).get('causal_mismatch') == 0}
        cens = [r for r in gz_rows(cb) if ck([r['day'], int(r['m']), int(r['t_ms'])]) in okw]
        want = queries_from_census(cens, hours, pis); rows = gz_rows(rb)
        got = {ck(dict(day=r['day'], m=int(r['m']), t_ms=int(r['t_ms']), side=int(r['side']), t=int(r['t']), cat=r['cat'])): r for r in rows}
        if set(got) != set(want):
            bad.append((k, 'keys', len(got), len(want))); continue
        if not all(all(close(got[q][f], want[q][f]) for f in ('N_cell', 'n_cell', 'pi_cell', 'pi_world', 'weight')) for q in want):
            bad.append((k, 'fields'))
    R['C'] = dict(parts=len(D['parts']), failures=bad[:5])
    return bool(D['parts']) and not bad


def check_D(P, D, R):
    bad, n = [], 0
    for k, (rec, rb, cb) in D['parts'].items():
        for r in gz_rows(rb):
            if r['status'] != 'ok':
                ok_status = r['status'].startswith('world:') or r['status'] in ('reroll_mismatch', 'no_snapshot', 'no_stage_capture', 'hold_no_generation') or r['status'].startswith('hold_') \
                    or (r['status'].endswith('_failed') and r.get('checks', {}).get(r['status'][:-7]) is False)
                if not ok_status:
                    bad.append((k, 'status', r['status']))
                continue
            n += 1; st, tg, hd = r['stage'], r['target'], r['hold']; ref = float(st['inputs'][RT.FEATS.index('pf_mid')])
            q, v, qm = target_of(hd['fills'], hd['gid'], r['side'], r['mark'], tg['eff'], ref)
            eff_ok = close(tg['eff'], hd['act']) if r['cat'].startswith('absent') else close(tg['eff'], st['effect'])
            if not (close(q, tg['q']) and close(v, tg['value'], 1e-9) and close(qm, tg['qm'], 1e-9) and eff_ok and all(x is not False for x in r['checks'].values())
                    and close(r['weight'], 1. / (r['pi_world'] * r['pi_cell']))):
                bad.append((k, r['cat'], r['t']))
    R['D'] = dict(ok_rows=n, failures=bad[:5])
    return bool(n) and not bad


def check_E(P, D, R):
    full = {}
    for k, (rec, rb, cb) in D['parts'].items():
        for r in gz_rows(cb):
            full.setdefault((r['day'], int(r['m']), int(r['t_ms'])), []).append(r)
    bad, n = [], 0
    for day, (rec, rb, cb) in D['pilots'].items():
        pc = {}
        for r in gz_rows(cb):
            pc.setdefault((r['day'], int(r['m']), int(r['t_ms'])), []).append(r)
        for w, rs in pc.items():
            n += 1
            if full.get(w) != rs:
                bad.append((day, w))
    R['E'] = dict(worlds=n, failures=bad[:5])
    return bool(n) and not bad


def check_F(P, D, R, rows=None):
    rows = rows if rows is not None else [r for _, (rec, rb, cb) in sorted(D['parts'].items()) for r in gz_rows(rb)]
    L_ = label_frame(rows); FC = forecast_rows(L_); AC, NU = action_rows(L_, M.NULL_DRAWS, M.SEED); bad = []
    sv = {(r['cat'], str(r['day']), str(r['side'])): r for r in D['FC'].to_dict('records')}
    for r in FC:
        s = sv.get((r['cat'], r['day'], r['side']))
        if s is None or not all(close(r[c], s.get(c)) for c in r if c not in ('cat', 'day', 'side')):
            bad.append(('forecasts', r['cat'], r['day'], r['side'])); break
    sa = {(r['population'], r['rule']): r for r in D['AC'].to_dict('records')}
    for r in AC:
        s = sa.get((r['population'], r['rule']))
        if s is None or not all(close(r[c], s.get(c)) for c in ('n', 'chosen', 'chosen_weight', 'J')):
            bad.append(('actions', r['population'], r['rule'])); break
    for name, (s, J) in NU.items():
        g = D['ND'][D['ND'].population == name].sort_values('draw', key=lambda x: x.astype(int)); au = (D['audit'] or {}).get('null', {}).get(name, {})
        if len(g) != len(J) or not all(close(a, b, 1e-12) for a, b in zip(g.J.astype(float), J)) or not all(close(s[c], au.get(c)) for c in s):
            bad.append(('null', name)); break
    R['F'] = dict(failures=bad[:5])
    return not bad


def check_G(P, D, R, rows=None):
    rows = rows if rows is not None else [r for _, (rec, rb, cb) in sorted(D['parts'].items()) for r in gz_rows(rb)]
    L_ = label_frame(rows); AC, NU = action_rows(L_, M.NULL_DRAWS, M.SEED); a = {(r['population'], r['rule']): r for r in AC}; got = {}
    for name in ('entry_combined', 'review_combined'):
        if name not in NU:
            continue
        s = NU[name][0]; J = a[(name, 'pi0_rule')]['J']; inc = not (s['free_strata'] >= M.MIN_FREE_STRATA and s['free_rows'] >= M.MIN_FREE_ROWS)
        got[name] = 'INCONCLUSIVE' if inc else ('POSITIVE' if (J > 0 and J > a[(name, 'all')]['J'] and J > s['null_p95']) else 'NOT_POSITIVE')
    want = {k: v['answer'] for k, v in (D['audit'] or {}).get('predicates', {}).get('selects_positive_exposure', {}).items()}
    R['G'] = dict(recomputed=got, saved=want)
    return bool(got) and got == want


def resources(dirs):
    peaks, cpu, bad = [], 0., []
    for d in dirs:
        for p in glob.glob(str(Path(d) / '**' / '*.log'), recursive=True):
            for ln in Path(p).read_text().splitlines():
                m = RL.match(ln)
                if m:
                    peaks.append(int(m.group(4))); cpu += float(m.group(6)); bad += [m.group(2)] if int(m.group(4)) >= 2048 else []
    return dict(jobs=len(peaks), peak_mib=max(peaks) if peaks else None, core_hours=cpu / 3600., bad=bad, ok=bool(peaks and not bad and cpu / 3600. <= 8.))


def suites(d, harness, prefix):
    import importlib.util
    import pf_mutant_one as O_
    import pf_mutant_summary as S_
    spec = importlib.util.spec_from_file_location(harness[:-3], RES / harness); H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
    files = {'control': Path(d) / f'{prefix}_control.log', **{i: Path(d) / f'{prefix}_{i}.log' for i in range(len(H.MUTANTS))}}
    recs = {k: S_.last_record(p) for k, p in files.items()}
    one = all(p.exists() and sum(1 for ln in p.read_text().splitlines() if ln.startswith('{') and '"cls"' in ln) == 1 for p in files.values())
    extra = sorted({x.name for x in Path(d).glob(f'{prefix}_*.log')} - {p.name for p in files.values()})
    ok, bad = S_.judge(len(H.MUTANTS), recs, O_.sources(H), [m_[0] for m_ in H.MUTANTS])
    return dict(planted=len(H.MUTANTS), ok=bool(ok and one and not extra), failures=bad[:3], extra=extra)


def check_H(P, D, R):
    rs = resources([P['logs'], P['src_logs']]); hw = [float(rec.get('hwm_mib', 1e9)) for rec, _, _ in list(D['parts'].values()) + list(D['pilots'].values())]
    s1 = suites(Path(P['logs']) / 'mutants_cov', 'pf_coverage_mutants.py', 'cov'); s2 = suites(Path(P['src_logs']) / 'mutants_src', 'pf_source_extend_mutants.py', 'src')
    R['H'] = dict(resources=rs, hwm_max=max(hw) if hw else None, suites=[s1, s2], asof=M.C.stamp())
    return bool(rs['ok'] and hw and max(hw) <= 1536. and s1['ok'] and s2['ok'])


CHECKS = (('A_identity', check_A), ('B_population', check_B), ('C_queries', check_C), ('D_labels', check_D), ('E_determinism', check_E), ('F_aggregates', check_F),
          ('G_readings', check_G), ('H_resources_suites', check_H))


def run_checks(P, names=None):
    R, ckr = {}, {}
    try:
        D = load(P)
    except Exception as e:                                                                  # noqa: BLE001
        return {n: False for n, _ in CHECKS if names is None or n in names}, dict(load=repr(e)[:300])
    for n, f in CHECKS:
        if names is not None and n not in names:
            continue
        try:
            ckr[n] = bool(f(P, D, R))
        except Exception as e:                                                              # noqa: BLE001
            ckr[n] = False; R[n] = dict(crash=repr(e)[:300])
    return ckr, R


# ---------------------------------------------------------------------------------------------------------------------- the falsifier
def copy_tree(P, d):
    d = Path(d); (d / 'out').mkdir(parents=True); shutil.copytree(Path(P['out']) / 'parts', d / 'out' / 'parts'); shutil.copytree(Path(P['out']) / 'pilot', d / 'out' / 'pilot')
    for q in Path(P['out']).glob('*.*'):
        if q.is_file():
            shutil.copy(q, d / 'out' / q.name)
    (d / 'rec').mkdir(); shutil.copy(Path(P['rec']) / 'manifest.json', d / 'rec' / 'manifest.json'); shutil.copytree(P['logs'], d / 'logs'); shutil.copytree(P['src_logs'], d / 'src_logs')
    return dict(P, out=d / 'out', rec=d / 'rec', logs=d / 'logs', src_logs=d / 'src_logs', real=False)


def rewrite_rows(Q, fn, which='rows'):
    pd_ = Path(Q['out']) / 'parts'; k = sorted(int(p.name[5:8]) for p in pd_.glob('part_*.json') if p.name.count('_') == 1)[0]
    b = pd_ / f'part_{k:03d}_{which}.jsonl.gz'; rows = gz_rows(b.read_bytes()); ok = fn(rows); nb = M.LS.gz_bytes(rows); b.write_bytes(nb)
    rp = pd_ / f'part_{k:03d}.json'; rec = json.loads(rp.read_text()); rec[f'{which}_sha256'] = hashlib.sha256(nb).hexdigest(); rp.write_text(json.dumps(rec)); return bool(ok)


def corruptions():
    def a_sub(Q):
        pd_ = Path(Q['out']) / 'parts'; rs = sorted(pd_.glob('part_*_rows.jsonl.gz')); rs[0].write_bytes(rs[-1].read_bytes() if len(rs) > 1 else gzip.compress(b'{}\n', mtime=0)); return True

    def b_drop(Q):
        m = json.loads((Path(Q['rec']) / 'manifest.json').read_text()); d0 = m['days'][0]; m['sample'][d0]['selected'] = m['sample'][d0]['selected'][1:]
        (Path(Q['rec']) / 'manifest.json').write_text(json.dumps(m)); return True

    def c_weight(rows):
        r = next((x for x in rows if x.get('status') == 'ok'), None)
        if r is None:
            return False
        r['weight'] = float(r['weight']) * 2.; return True

    def d_value(rows):
        r = next((x for x in rows if x.get('status') == 'ok'), None)
        if r is None:
            return False
        r['target']['value'] = float(r['target']['value']) + 1.; return True

    def e_census(Q):
        pil = json.loads((Path(Q['rec']) / 'manifest.json').read_text())['sample']; want = {(x['day'], int(x['m']), int(x['t_ms'])) for d in pil for x in pil[d]['pilot']}
        pd_ = Path(Q['out']) / 'parts'
        for rp in sorted(pd_.glob('part_*_census.jsonl.gz')):
            rows = gz_rows(rp.read_bytes()); hit = next((r for r in rows if (r['day'], int(r['m']), int(r['t_ms'])) in want), None)
            if hit is not None:
                hit['room'] = float(hit['room']) + 1.; nb = M.LS.gz_bytes(rows); rp.write_bytes(nb); rj = pd_ / rp.name.replace('_census.jsonl.gz', '.json')
                rec = json.loads(rj.read_text()); rec['census_sha256'] = hashlib.sha256(nb).hexdigest(); rj.write_text(json.dumps(rec)); return True
        return False

    def csv_bump(name, col, d_, cond):
        def f(Q):
            p = Path(Q['out']) / f'{name}.csv'; df = pd.read_csv(p, float_precision='round_trip', dtype=STRC); idx = df.index[cond(df)]
            if not len(idx):
                return False
            df.loc[idx[0], col] = float(df.loc[idx[0], col]) + d_; df.to_csv(p, index=False); return True
        return f

    def g_flip(Q):
        p = Path(Q['out']) / 'audit.json'; j = json.loads(p.read_text()); s = j['predicates']['selects_positive_exposure']; k0 = sorted(s)[0]
        s[k0]['answer'] = 'POSITIVE' if s[k0]['answer'] != 'POSITIVE' else 'NOT_POSITIVE'; p.write_text(json.dumps(j)); return True

    def h_budget(Q):
        (Path(Q['logs']) / 'over_budget.log').write_text('2026-11-21T00:00:00Z tag=over_budget rc=0 peak_rss_mib=100 wall_s=29000 cpu_user_s=29000.0\n'); return True

    def h_suite(Q):
        d = Path(Q['logs']) / 'mutants_cov'; shutil.copy(d / 'cov_0.log', d / 'cov_1.log'); return True
    return [('A_substituted', 'A_identity', a_sub), ('B_drop', 'B_population', b_drop), ('C_weight', 'C_queries', lambda Q: rewrite_rows(Q, c_weight)),
            ('D_value', 'D_labels', lambda Q: rewrite_rows(Q, d_value)), ('E_census', 'E_determinism', e_census),
            ('F_mse', 'F_aggregates', csv_bump('forecasts', 'mse_V', 1., lambda df: (df.day == 'all') & (df.side == 'all'))),
            ('F_action', 'F_aggregates', csv_bump('actions', 'J', .1, lambda df: df.rule == 'pi0_rule')), ('F_draw', 'F_aggregates', csv_bump('null_draws', 'J', .01, lambda df: df.index == 0)),
            ('G_reading', 'G_readings', g_flip), ('H_budget', 'H_resources_suites', h_budget), ('H_suite', 'H_resources_suites', h_suite)]


def falsify():
    res = {}
    with tempfile.TemporaryDirectory(dir=str(M.OUT)) as td:
        for name, chk, fn in corruptions():
            Q = copy_tree(REAL, Path(td) / f'{name}_clean'); Qc = copy_tree(REAL, Path(td) / name)
            clean, _ = run_checks(Q, [chk]); ap = fn(Qc); bad, _ = run_checks(Qc, [chk])
            res[name] = dict(check=chk, clean=clean[chk], corrupted=bad[chk], applied=bool(ap))
            shutil.rmtree(Path(td) / f'{name}_clean', ignore_errors=True); shutil.rmtree(Path(td) / name, ignore_errors=True)
    ok = all(v['clean'] is True and v['corrupted'] is False and v['applied'] for v in res.values())
    (HERE / 'checks_falsifier.json').write_text(json.dumps(dict(cases=res, falsifier_ok=ok, asof=M.C.stamp()), indent=1, sort_keys=True) + '\n')
    print(json.dumps(dict(cases=res, falsifier_ok=ok))); return ok


def main():
    ckr, R = run_checks(REAL)
    (HERE / 'checks.json').write_text(json.dumps(dict(checks=ckr, all_ok=all(ckr.values()), details=R, asof=M.C.stamp()), indent=1, sort_keys=True, default=str) + '\n')
    print(json.dumps(dict(checks=ckr, all_ok=all(ckr.values())), indent=1)); return all(ckr.values())


# ---------------------------------------------------------------------------------------------------------------------- the selftest
def selftest():
    bad = []; rng = np.random.default_rng(5); rows = []
    for i in range(160):
        cat = M.CATS[(i // 2) % 4]; z = float(i % 3 == 0); V = (1. + i % 5) if cat in ('absent_POST', 'review_KEEP') else -(1. + i % 5); edge = .5; Eq = 3.; Mv = (-20. + i % 7) * z
        rows.append(dict(day=M.DAYS[i % 2], m=i // 4, t_ms=i, side=i % 2, t=0, cat=cat, hour=(i // 16) % 6, weight=1. + (i % 3), status='ok',
                         stage=dict(dec_status='model', moments=dict(P=.3, Eq=Eq, Eqm=V - edge * Eq, V=V, edge=edge), inputs=[100. + i] * len(RT.FEATS)),
                         target=dict(q=10. * z, value=edge * 10. * z + Mv, qm=Mv), N_cell=8, n_cell=4))
    Dm = M.frame(rows); FC = M.forecasts(Dm); AC, NU, DR = M.actions(Dm)
    D = dict(FC=pd.DataFrame(FC).astype({'day': str, 'side': str}), AC=pd.DataFrame(AC), ND=pd.DataFrame(DR), audit=dict(null=NU, predicates=M.predicates(FC, M.coverage(Dm), AC, NU)))
    if not check_F({}, D, {}, rows) or not check_G({}, D, {}, rows):
        bad.append('F / G false on the module\'s own hand tables')
    D2 = copy.deepcopy(D); D2['FC'].loc[0, 'mse_V'] = float(D2['FC'].loc[0, 'mse_V']) + 1.
    if check_F({}, D2, {}, rows):
        bad.append('F true after a changed MSE')
    D3 = copy.deepcopy(D); D3['ND'].loc[0, 'J'] = float(D3['ND'].loc[0, 'J']) + .01
    if check_F({}, D3, {}, rows):
        bad.append('F true after a changed null draw')
    D4 = copy.deepcopy(D); k0 = sorted(D4['audit']['predicates']['selects_positive_exposure'])[0]; s0 = D4['audit']['predicates']['selects_positive_exposure'][k0]
    s0['answer'] = 'POSITIVE' if s0['answer'] != 'POSITIVE' else 'NOT_POSITIVE'
    if check_G({}, D4, {}, rows):
        bad.append('G true after a flipped reading')
    hours = {('d', 1, 10): 3, ('d', 2, 20): 3}; pis = {('d', 1, 10): .5, ('d', 2, 20): .25}
    cens = [dict(day='d', m=m, t_ms=t0, side=0, t=t, cat='absent_WAIT') for (m, t0) in ((1, 10), (2, 20)) for t in range(0, 300, 50)]
    want = queries_from_census(cens, hours, pis); qs, _ = M.sample_queries([dict(r, gid=None, created_at=None) for r in cens], {k: dict(hour=v, pi_world=pis[k]) for k, v in hours.items()})
    gotk = {ck(dict(day=q['day'], m=q['m'], t_ms=q['t_ms'], side=q['side'], t=q['t'], cat=q['cat'])) for q in qs}
    if gotk != set(want) or not all(close(next(q for q in qs if ck(dict(day=q['day'], m=q['m'], t_ms=q['t_ms'], side=q['side'], t=q['t'], cat=q['cat'])) == k)['weight'], v['weight']) for k, v in want.items()):
        bad.append('queries_from_census differs from the module on hand rows')
    f = [[100., 3, 0, 10., 50., 0], [137., 3, 0, 5., 50., 1]]
    if target_of(f, 3, 0, 52., 137., 50.5) != (5., 10., 7.5) or target_of(f, 3, 1, 52., 100., 50.5)[1] != -(10. * 2. + 5. * 2.):
        bad.append('target_of: the inclusive effect or the ask sign')
    if bad:
        sys.exit('COVERAGE CHECKS SELFTEST FAILED: ' + '; '.join(bad))
    print('coverage checks selftest passed: F / G true on the module\'s hand tables and false after a changed MSE, null draw or reading; the query recomputation '
          'equals the module\'s; target_of on the inclusive effect and the ask sign')


if __name__ == '__main__':
    if sys.argv[1:] == ['selftest']:
        selftest()
    elif not sys.argv[1:]:
        a = main(); b = falsify(); sys.exit(0 if (a and b) else 1)
    else:
        sys.exit(__doc__)
