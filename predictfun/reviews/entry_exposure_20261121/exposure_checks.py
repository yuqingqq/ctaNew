"""ENTRY-EXPOSURE-v1: THE DECLARED CHECKS (review 3d8bf5ad; 2026-11-21, session aea5bfb9). Declared in the freeze, before any branch runs.
Each check recomputes from the saved partitions, the committed manifest and the outputs; a check that cannot run is false, with its reason.
 A identity: the data manifest is the one committed in HEAD; every partition record matches its bytes, the manifest and the final sources,
   holds two forks per member world (its rows are exactly the members x sides); forks.csv holds exactly the manifest's 768 forks once each;
   every manifest exclusion is carried unchanged.
 B reproduction: every manifest-evaluable fork ends ok with every branch check true (reproduced, keep baseline, prefix, KEEP reconciliation,
   initial gid, components, identity) and Y_PI0 equal to the stored Y; the failures are counted by status.
 C raw reconciliation (independent of the module's functions), on every ok fork from its own raw arms: value / cash / position from the
   fills; every fill inside its order's live window [act, remove) and <= END, on its order's side; the WAIT arm's fork decision is WAIT; the
   forced POST is unique in each POST arm and its gid is the recorded initial gid; V_initial, the filled quantity and the three components
   from the fills; Y_PI0, Y_KEEP_INITIAL, D_EXPOSURE, D_initial, D_focal_rest, D_opposite recomputed; forks.csv carries the same values; the
   KEEP arm never PULLs the initial generation and its shadow log is its overrides (gid0, shadow PULL); every arm's binding and causal probes
   are clean and the delivered-copy positive control fired.
 D summaries: summary.csv, reasons.csv and exposure.json's statuses recomputed here from forks.csv.
 E predicates: pf_entry_exposure.predicates on forks.csv equals exposure.json's.
 F resources: every run_light job under logs/ (recursive) peaked below 2,048 MiB, every partition's high-water <= 1,536 MiB; the round's CPU
   (recursive, every job) is within 2 core-hours.
 G suites: pf_mutant_summary.judge on every record of the exposure harness against the final sources; one record per file; no other file.
FALSIFIER (falsify()): every check on copies of its real inputs is true clean and false after one corruption, each applied on a real row: a
 dropped fork (A); a branch flag (B); compensating components D_initial + 1 / D_rest - 1 (C); a wrong but valid initial gid (C); one KEEP fill
 price (C); a summary mean (D); the diagnosis (E); an over-budget job (F); a copied mutant record (G).
SELFTEST (selftest(): synthetic forks through pf_entry_exposure.fork_package, before any real run): check C's row reconciliation passes every
 clean synthetic row and flags each of the three C corruptions.
Usage (run_light): exposure_checks.py [selftest]  (no argument: the checks, then the falsifier)"""
import copy
import glob
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; RES = Path('/home/yuqing/ctaNew/predictfun/research'); sys.path.insert(0, str(RES))
import pf_entry_exposure as M                            # noqa: E402

C, pd, LS = M.C, M.pd, M.LS
LOGS = HERE / 'logs'
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
ARMS = ('WAIT_PI0', 'POST_PI0', 'POST_KEEP_INITIAL')
FLAGS = ('reproduced', 'keep_baseline_equal', 'prefix_ok', 'keep_reconciled', 'initial_gid_equal', 'components_ok', 'identity_ok')
KEY = ['day', 'm', 't_ms', 'side']


def forks(out):
    F = pd.read_csv(Path(out) / 'forks.csv', float_precision='round_trip'); F['day'] = F.day.astype(str); return F


def parts(out):
    """Every partition's (k, record, rows), streamed."""
    man = json.loads((Path(out) / 'manifest.json').read_text())
    for k in range(len(man['parts'])):
        rec = json.loads((Path(out) / 'parts' / f'part_{k:03d}.json').read_text()); b = (Path(out) / 'parts' / f'part_{k:03d}.jsonl.gz').read_bytes()
        yield k, rec, b


def check_A(out, R):
    out = Path(out); man = json.loads((out / 'manifest.json').read_text()); bad = []
    if not M.committed_equal(M.RECEIPT_REL, out / 'manifest.json'):
        bad.append('the data manifest is not the committed one')
    msha = C.sha(out / 'manifest.json'); src = M.sources()
    for k, rec, b in parts(out):
        pt = man['parts'][k]; rows = LS.gz_rows(b); want = sorted((w['day'], int(w['m']), int(w['t_ms']), s) for w in pt['worlds'] for s in (0, 1))
        if hashlib.sha256(b).hexdigest() != rec['sha256'] or rec['manifest_sha256'] != msha or rec['sources'] != src or rec['rows'] != len(rows) \
                or sorted((str(r['day']), int(r['m']), int(r['t_ms']), int(r['side'])) for r in rows) != want:
            bad.append(f'part {k} identity / members')
    F = forks(out); mf = {(f['day'], int(f['m']), int(f['t_ms']), int(f['side'])): f for k_ in ('train', 'eval') for f in man['sets'][k_]['forks']}
    got = [(r.day, int(r.m), int(r.t_ms), int(r.side)) for r in F.itertuples()]
    if len(got) != len(set(got)) or set(got) != set(mf) or len(mf) != 768:
        bad.append('forks.csv is not the manifest\'s forks once each')
    st = dict(zip(got, F.status))
    if any(st.get(k_) != f['status'] for k_, f in mf.items() if f['status'] != 'ok'):
        bad.append('a manifest exclusion was not carried unchanged')
    js = json.loads((out / 'exposure.json').read_text())
    if js['manifest_sha256'] != msha or js['sources'] != src:
        bad.append('exposure.json identity')
    R['A'] = dict(bad=bad, forks=len(got), manifest_forks=len(mf))
    return not bad


def check_B(out, R):
    out = Path(out); man = json.loads((out / 'manifest.json').read_text()); F = forks(out)
    okm = {(f['day'], int(f['m']), int(f['t_ms']), int(f['side'])) for k_ in ('train', 'eval') for f in man['sets'][k_]['forks'] if f['status'] == 'ok'}
    G = F[[(r.day, int(r.m), int(r.t_ms), int(r.side)) in okm for r in F.itertuples()]]
    flags_ok = G[list(FLAGS)].astype(str).eq('True').all(axis=1) if len(G) else pd.Series(dtype=bool)
    same_y = (G.Y_PI0.astype(float) == G.Y_stored.astype(float)) if len(G) else pd.Series(dtype=bool)
    R['B'] = dict(evaluable=len(okm), rows=int(len(G)), by_status={str(k_): int(v) for k_, v in G.status.value_counts().items()}, flags_false=int((~flags_ok).sum()),
                  y_differs=int((~same_y).sum()))
    return bool(len(G) == len(okm) and len(G) and (G.status == 'ok').all() and flags_ok.all() and same_y.all())


def raw_row_bad(r):
    """The source-to-output reconciliation of one ok fork row from its own raw arms (header, C) -> its failures."""
    bad = []; s = int(r['side']); A = r['arms']; sg = lambda x: 1. if int(x) == 0 else -1.; val, comp = {}, {}
    for n in ARMS:
        a = A[n]; mark = float(a['mark']); fl = a['fill_list']; orders = {int(o['gid']): o for o in a['orders']}
        cash = -sum(sg(f[2]) * float(f[3]) * float(f[4]) for f in fl); pos = sum(sg(f[2]) * float(f[3]) for f in fl); val[n] = cash + pos * mark
        if abs(val[n] - float(a['value'])) > 1e-6 or abs(cash - float(a['cash'])) > 1e-6 or abs(pos - float(a['position'])) > 1e-9:
            bad.append(f'{n}: value / cash / position from the fills')
        for f in fl:
            o = orders.get(int(f[1]))
            if o is None or int(o['side']) != int(f[2]) or not (float(o['act']) <= float(f[0]) < float(o['remove']) and float(f[0]) <= float(a['end'])):
                bad.append(f'{n}: a fill outside its order\'s live window or side'); break
        if a['teacher']['binding_bad'] != 0 or a['teacher']['binding_checked'] < 1 or (a['causal'] or {}).get('mismatch', 1) != 0 or (a['causal'] or {}).get('parity_mismatch', 1) != 0:
            bad.append(f'{n}: binding / causal probes')
    dw = [d for d in A['WAIT_PI0']['decision_list'] if float(d[0]) == 0. and int(d[1]) == s]
    if len(dw) != 1 or dw[0][3] != 'WAIT':
        bad.append('WAIT_PI0: the fork decision is not WAIT')
    g0 = {}
    for n, b, cn in (('POST_PI0', 'init_PI0', 'comp_PI0'), ('POST_KEEP_INITIAL', 'init_KEEP', 'comp_KEEP')):
        dl = [d for d in A[n]['decision_list'] if float(d[0]) == 0. and int(d[1]) == s and d[3] == 'POST']
        if len(dl) != 1 or dl[0][4] is None:
            bad.append(f'{n}: not one forced POST'); continue
        g = int(dl[0][4]); g0[n] = g; mark = float(A[n]['mark'])
        if int(r[b]['gid0']) != g:
            bad.append(f'{n}: the recorded initial gid is not the forced POST\'s')
        c = dict(v_initial=0., v_focal_rest=0., v_opposite=0.); q0 = 0.
        for f in A[n]['fill_list']:
            v = sg(f[2]) * float(f[3]) * (mark - float(f[4]))
            if int(f[1]) == g:
                c['v_initial'] += v; q0 += float(f[3])
            elif int(f[2]) == s:
                c['v_focal_rest'] += v
            else:
                c['v_opposite'] += v
        comp[n] = c
        if abs(c['v_initial'] - float(r[b]['v_initial'])) > 1e-9 or abs(q0 - float(r[b]['filled'])) > 1e-9 or any(abs(c[k_] - float(r[cn][k_])) > 1e-9 for k_ in c):
            bad.append(f'{n}: V_initial / filled / components from the fills')
    if len(comp) == 2:
        cp, ck = comp['POST_PI0'], comp['POST_KEEP_INITIAL']
        want = dict(Y_PI0=val['POST_PI0'] - val['WAIT_PI0'], Y_KEEP_INITIAL=val['POST_KEEP_INITIAL'] - val['WAIT_PI0'], D_EXPOSURE=val['POST_KEEP_INITIAL'] - val['POST_PI0'],
                    D_initial=ck['v_initial'] - cp['v_initial'], D_focal_rest=ck['v_focal_rest'] - cp['v_focal_rest'], D_opposite=ck['v_opposite'] - cp['v_opposite'])
        want['D_rest'] = want['D_EXPOSURE'] - want['D_initial']
        if any(abs(float(r[k_]) - v) > 1e-6 for k_, v in want.items()):
            bad.append('the contrasts / components recomputed from the fills differ')
    gk = g0.get('POST_KEEP_INITIAL')
    if any(int(d[1]) == s and d[3] == 'PULL' and d[4] is not None and int(d[4]) == gk for d in A['POST_KEEP_INITIAL']['decision_list']):
        bad.append('KEEP: a PULL of the initial generation')
    if len(r['shadow']) != int(r['overrides']) or any(int(x['gid']) != gk or x['shadow_action'] != 'PULL' for x in r['shadow']):
        bad.append('KEEP: the shadow log is not its overrides')
    return bad


def check_C(out, R):
    F = forks(out).set_index(KEY); n = 0; bad = []; delivered = 0; cols = ('Y_PI0', 'Y_KEEP_INITIAL', 'D_EXPOSURE', 'D_initial', 'D_rest', 'D_focal_rest', 'D_opposite')
    for k, rec, b in parts(out):
        for r in LS.gz_rows(b):
            if r.get('status') != 'ok':
                continue
            n += 1; rb = raw_row_bad(r); delivered += sum(int((r['arms'][a_]['causal'] or {}).get('delivered_changed', 0)) for a_ in ARMS)
            fr = F.loc[(str(r['day']), int(r['m']), int(r['t_ms']), int(r['side']))]
            if any(abs(float(fr[c_]) - float(r[c_])) > 1e-9 for c_ in cols) or fr['status'] != r['status']:
                rb.append('forks.csv differs from the partition row')
            if rb and len(bad) < 10:
                bad.append(dict(part=k, key=(r['day'], r['m'], r['t_ms'], r['side']), bad=rb[:3]))
            elif rb:
                bad.append(None)
    R['C'] = dict(rows=n, failing=len(bad), examples=[x for x in bad if x][:5], delivered_changed=delivered)
    return bool(n and not bad and delivered > 0)


def check_D(out, R):
    F = forks(out); S = pd.read_csv(Path(out) / 'summary.csv', float_precision='round_trip'); W = pd.read_csv(Path(out) / 'reasons.csv'); js = json.loads((Path(out) / 'exposure.json').read_text())
    ok = F[F.status == 'ok']; bad = []; S['day'] = S.day.astype(str); W['day'] = W.day.astype(str)
    for (day, s), g in ok.groupby(['day', 'side']):
        row = S[(S.day == day) & (S.side == s)]
        if len(row) != 1:
            bad.append(f'{day}/{s}: no summary row'); continue
        row = row.iloc[0]; d = g.D_EXPOSURE.astype(float); ad = g[g.KEEP_filled > g.PI0_filled + 1e-9]
        mine = dict(forks=len(g), mean_D=d.mean(), sum_D=d.sum(), mean_D_initial=g.D_initial.mean(), mean_Y_PI0=g.Y_PI0.mean(), mean_Y_KEEP=g.Y_KEEP_INITIAL.mean(),
                    fill_prob_KEEP=(g.KEEP_filled > 1e-9).mean(), fill_prob_PI0=(g.PI0_filled > 1e-9).mean(), addl_fill_forks=len(ad), n_abs_D_gt30=(d.abs() > 30).sum(),
                    worst_D=d.min(), mean_vinit_KEEP=g.KEEP_v_initial.mean())
        if any(abs(float(row[k_]) - float(v)) > 1e-9 * max(1., abs(float(v))) for k_, v in mine.items()):
            bad.append(f'{day}/{s}: summary values')
        for b_ in ('PI0', 'KEEP'):
            want = {str(k_): int(v) for k_, v in g[f'{b_}_why'].value_counts().items()}
            got = {str(x.why): int(x.n) for x in W[(W.day == day) & (W.side == s) & (W.branch == b_)].itertuples()}
            if want != got:
                bad.append(f'{day}/{s}/{b_}: reasons')
    if len(S) != ok.groupby(['day', 'side']).ngroups:
        bad.append('summary rows')
    st = {(str(d_), str(s_)): int(n_) for (d_, s_), n_ in F.groupby(['day', 'status']).size().items()}
    if st != {(str(x['day']), str(x['status'])): int(x['n']) for x in js['statuses']}:
        bad.append('exposure.json statuses')
    R['D'] = dict(bad=bad)
    return not bad


def check_E(out, R):
    F = forks(out); js = json.loads((Path(out) / 'exposure.json').read_text()); mine = M.predicates(F); theirs = js['predicates']
    keys = ('diagnosis', 'more_execution', 'initial_carries', 'additional_fill_forks', 'extra_fills_harmful', 'forks')
    same = all(mine[k_] == theirs[k_] for k_ in keys) and {d: v['sign'] for d, v in mine['days'].items()} == {d: v['sign'] for d, v in theirs['days'].items()}
    R['E'] = dict(diagnosis=theirs.get('diagnosis'), recomputed=mine['diagnosis'], same=same)
    return bool(same)


def check_F(out, R):
    peaks, cpu, bad = [], 0., []
    for p in glob.glob(str(LOGS / '**' / '*.log'), recursive=True):
        for ln in Path(p).read_text().splitlines():
            m = RL.match(ln)
            if m:
                peaks.append(int(m.group(4))); cpu += float(m.group(6)); bad += [m.group(2)] if int(m.group(4)) >= 2048 else []
    hw = [float(rec.get('hwm_mib', 1e9)) for _, rec, _ in parts(out)]
    R['F'] = dict(jobs=len(peaks), peak_mib=max(peaks) if peaks else None, core_hours=cpu / 3600., bad=bad, part_hwm_max=max(hw) if hw else None)
    return bool(peaks and not bad and cpu / 3600. <= 2. and hw and max(hw) <= 1536.)


def check_G(R):
    import importlib.util
    import pf_mutant_one as O
    import pf_mutant_summary as S
    h = 'pf_entry_exposure_mutants.py'; spec = importlib.util.spec_from_file_location(h[:-3], RES / h); H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
    d = LOGS / 'mutants_exp'; files = {'control': d / 'mut_exp_control.log', **{i: d / f'mut_exp_{i}.log' for i in range(len(H.MUTANTS))}}
    recs = {k: S.last_record(p) for k, p in files.items()}
    one = all(p.exists() and sum(1 for ln in p.read_text().splitlines() if ln.startswith('{') and '"cls"' in ln) == 1 for p in files.values())
    extra = sorted({x.name for x in d.glob('mut_exp_*.log')} - {p.name for p in files.values()})
    ok, bad = S.judge(len(H.MUTANTS), recs, O.sources(H), [m_[0] for m_ in H.MUTANTS])
    R['G'] = dict(planted=len(H.MUTANTS), judge_ok=ok, failures=bad[:5], one_record_per_file=one, extra_files=extra)
    return bool(ok and one and not extra)


def main(out=M.OUT):
    out = Path(out); R = {}; ck = {}
    for k, f in (('A_identity', check_A), ('B_reproduction', check_B), ('C_raw', check_C), ('D_summaries', check_D), ('E_predicates', check_E), ('F_resources', check_F)):
        try:
            ck[k] = f(out, R)
        except Exception as e:                                                              # noqa: BLE001  (a check that cannot run is false)
            ck[k] = False; R[k[0]] = dict(error=repr(e)[:300])
    try:
        ck['G_suites'] = check_G(R)
    except Exception as e:                                                                  # noqa: BLE001
        ck['G_suites'] = False; R['G'] = dict(error=repr(e)[:300])
    J = dict(checks=ck, all_true=all(ck.values()), readings=R, source_sha256=C.sha(Path(__file__)), asof=C.stamp())
    (HERE / 'checks.json').write_text(json.dumps(J, indent=1, sort_keys=True, default=str) + '\n')
    print(json.dumps(dict(checks=ck, all_true=J['all_true']), indent=1))


def _corrupt_part(d, out, fn):
    """Replace one partition of copy d by a corrupted real row (fn(row) -> applied), its record's hash updated; the first row fn applies to."""
    for k, rec, b in parts(out):
        rows = LS.gz_rows(b)
        for i, r in enumerate(rows):
            if r.get('status') == 'ok' and fn(r):
                nb = LS.gz_bytes(rows); gp, rp = d / 'parts' / f'part_{k:03d}.jsonl.gz', d / 'parts' / f'part_{k:03d}.json'
                gp.unlink(); gp.write_bytes(nb); rp.unlink(); rp.write_text(json.dumps(dict(rec, sha256=hashlib.sha256(nb).hexdigest())))
                return True
    return False


def falsify(out=M.OUT):
    global LOGS
    out = Path(out); res = {}

    def safe(f, *a):
        try:
            return f(*a)
        except Exception:                                                                 # noqa: BLE001
            return False
    with tempfile.TemporaryDirectory(dir=str(out)) as td:
        td = Path(td)

        def copy_(name):
            d = td / name; (d / 'parts').mkdir(parents=True)
            for q in out.iterdir():
                if q.is_file():
                    shutil.copy(q, d / q.name)
            for q in (out / 'parts').iterdir():
                if q.is_file():
                    (d / 'parts' / q.name).symlink_to(q)
            return d

        def csv_fn(d, name, fn):
            D = pd.read_csv(d / name, float_precision='round_trip'); applied = fn(D); D.to_csv(d / name, index=False); return applied

        def drop(D):
            D.drop(index=D.index[D.status == 'ok'][0], inplace=True); return True
        def flag(D):
            D.loc[D.index[D.status == 'ok'][0], 'prefix_ok'] = False; return True
        def comp(D):
            i = D.index[D.status == 'ok'][0]; D.loc[i, 'D_initial'] += 1.; D.loc[i, 'D_rest'] -= 1.; return True
        def summ(D):
            D.loc[D.index[0], 'mean_D'] += 1.; return True

        def wrong_gid(r):
            g = int(r['init_KEEP']['gid0']); other = [int(o['gid']) for o in r['arms']['POST_KEEP_INITIAL']['orders'] if int(o['gid']) != g]
            if not other:
                return False
            r['init_KEEP']['gid0'] = other[0]; return True

        def fill_price(r):
            fl = r['arms']['POST_KEEP_INITIAL']['fill_list']
            if not fl:
                return False
            fl[0][4] = float(fl[0][4]) + 1.; return True
        clean = copy_('clean')
        for name, chk, fn, tgt in (('A', check_A, drop, 'forks.csv'), ('B', check_B, flag, 'forks.csv'), ('C_components', check_C, comp, 'forks.csv'),
                                   ('D', check_D, summ, 'summary.csv')):
            d = copy_(name); ap = csv_fn(d, tgt, fn); res[name] = dict(clean=safe(chk, clean, {}), corrupted=safe(chk, d, {}), applied=bool(ap))
        for name, fn in (('C_wrong_gid', wrong_gid), ('C_fill_price', fill_price)):
            d = copy_(name); ap = _corrupt_part(d, out, fn); res[name] = dict(clean=safe(check_C, clean, {}), corrupted=safe(check_C, d, {}), applied=bool(ap),
                                                                               a_still_true=safe(check_A, d, {}))
        d = copy_('E'); j = json.loads((d / 'exposure.json').read_text()); j['predicates']['diagnosis'] = 'flipped'; (d / 'exposure.json').write_text(json.dumps(j))
        res['E'] = dict(clean=safe(check_E, clean, {}), corrupted=safe(check_E, d, {}), applied=True)
        saved = LOGS
        try:
            h = td / 'H'; shutil.copytree(LOGS, h)
            LOGS = h; clF = safe(check_F, out, {})
            (h / 'over_budget.log').write_text('2026-11-21T00:00:00Z tag=over_budget rc=0 peak_rss_mib=100 wall_s=7300 cpu_user_s=7300.0\n')
            res['F'] = dict(clean=clF, corrupted=safe(check_F, out, {}), applied=True)
            h2 = td / 'H2'; shutil.copytree(saved, h2); LOGS = h2; clG = safe(check_G, {})
            shutil.copy(h2 / 'mutants_exp' / 'mut_exp_0.log', h2 / 'mutants_exp' / 'mut_exp_1.log')
            res['G'] = dict(clean=clG, corrupted=safe(check_G, {}), applied=True)
        finally:
            LOGS = saved
    ok = all(v['clean'] is True and v['corrupted'] is False and v['applied'] and v.get('a_still_true', True) for v in res.values())
    (HERE / 'checks_falsifier.json').write_text(json.dumps(dict(cases=res, falsifier_ok=ok, asof=C.stamp()), indent=1, sort_keys=True) + '\n')
    print(json.dumps(dict(cases=res, falsifier_ok=ok)))


def selftest():
    """Check C's row reconciliation on synthetic fork_package rows (header): clean rows pass; each corruption is flagged."""
    import pf_r2_labels_selftest as LT
    L, R2 = M.L, M.R2; bad = []; wroot = Path(os.environ['PF_TEACHER_SYN']); wj = json.loads((wroot / 'world.json').read_text()); day = '20261024'
    W = L.worlds_of_day(day, Path(wj['kp']), 2); _, prs, fair, depth, ff = L.world_inputs(day, W, R2.KP.Syn(), R2.X.syn_inputs, R2.X.syn_features())
    Mst, sst = LT.stubs(-12.), LT.OpenSupport(); rows = []
    for w in W:
        wb = L.world_bind(day, w, prs, fair, depth, ff)[0]
        if wb is None:
            continue
        for s in (0, 1):
            sel = dict(day=day, side=s, decision_ms=0, source='neutral', shadow_source='entry', offset_ms=0, kind='post')
            ctx = L.fork_for(sel, wb, Mst, sst); st = L.pair_bound(ctx, L.teacher_factory(Mst, sst, wb, every=M.AUDIT_EVERY), 'post')
            if st.get('status') != 'ok':
                continue
            f = dict(day=day, m=int(w['m']), t_ms=int(round(float(w['t']))), hour=0, side=s, oid=1, status='ok', sel=sel, fork_fingerprint=ctx['fingerprint'], Y=st['Y'],
                     arms={n: {h: a[h] for h in ('value', 'fills_sha256', 'decisions_sha256', 'requests_sha256')} for n, a in st['arms'].items()})
            rows.append(json.loads(json.dumps(M.fork_package(f, wb, Mst, sst), default=str)))
    okr = [r for r in rows if r['status'] == 'ok']
    if len(okr) < 2 or any(raw_row_bad(r) for r in okr):
        bad.append(f"clean synthetic rows: {len(okr)} ok, failures {[raw_row_bad(r)[:2] for r in okr][:3]}")
    def first(fn):
        for r in okr:
            x = copy.deepcopy(r)
            if fn(x):
                return x
        return None
    def wrong_gid(r):
        g = int(r['init_KEEP']['gid0']); other = [int(o['gid']) for o in r['arms']['POST_KEEP_INITIAL']['orders'] if int(o['gid']) != g]
        if not other:
            return False
        r['init_KEEP']['gid0'] = other[0]; return True
    def fill_price(r):
        fl = r['arms']['POST_KEEP_INITIAL']['fill_list']
        if not fl:
            return False
        fl[0][4] = float(fl[0][4]) + 1.; return True
    def comp(r):
        r['D_initial'] += 1.; r['D_rest'] -= 1.; return True
    for name, fn in (('compensating components', comp), ('wrong but valid gid', wrong_gid), ('a KEEP fill price', fill_price)):
        x = first(fn)
        if x is None:
            bad.append(f'{name}: no synthetic row to corrupt')
        elif not raw_row_bad(x):
            bad.append(f'{name}: not flagged')
    if bad:
        sys.exit('EXPOSURE CHECKS SELFTEST FAILED: ' + '; '.join(bad))
    print(f'exposure checks selftest passed: {len(okr)} clean synthetic rows reconcile; compensating components, a wrong but valid gid and a KEEP fill '
          'price are flagged')


if __name__ == '__main__':
    if sys.argv[1:] == ['selftest']:
        selftest()
    else:
        main(); falsify()
