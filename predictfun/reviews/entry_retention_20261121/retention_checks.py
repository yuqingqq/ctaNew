"""ENTRY-RETENTION-DISCRIMINATION-v1: THE DECLARED CHECKS (review 67a0d9f7; 2026-11-21, session aea5bfb9). Declared in the freeze, before any
measurement. Each check recomputes from the saved retention partitions, the saved exposure partitions and the committed manifest; a check
that cannot run is false, with its reason.
 A identity: the exposure manifest committed; every retention partition matches its bytes, the final sources, the manifest and its exposure
   partition's hash, and holds exactly its members x sides; stages.csv holds three rows per manifest fork; every exclusion carried.
 B prefix (independent of the rerun): on every ok fork, from the saved POST_PI0 ledger - every arm END 5,000 ms, one mark and the settings
   across arms and the held reference (review 73f6cfa6 item 3); the first live policy decision of gid0 (engine fallbacks excluded) is the first
   review, in time and action; the first PULL of gid0 is the first pull, its effect the engine's rule; the true callbacks up to the stop tick
   (decisions after the fork - the forced POST - fallbacks) are the recorded ones.
 C targets: the held initial fills equal the saved KEEP gid0 fills; every stage target (q, value, qm, severe) recomputed from the held fills
   at its effect with its own pf_mid; the unavoidable value; at the first pull the held value / quantity after the effect equal D_initial /
   the KEEP - PI0 initial quantity, and the POST_PI0 initial fills are the held ones before it.
 D forecasts: every recorded input vector hashes to its float32 bits; P, E[q], E[qm], V recomputed from the inputs by the frozen predictor
   (pf_r2_teacher.predict_rows) equal the recorded ones bit for bit and V = edge x E[q] + E[qm].
 E measurements: stages / calibration / bins / decisions / changes recomputed from the partitions equal the outputs; the pooled Brier and the
   MAE of V / zero / edge x E[q] per stage recomputed here from stages.csv.
 F predicates: the declared predicates recomputed equal retention.json's.
 G resources: every run_light job under logs/ (recursive) peaked below 2,048 MiB, every partition's high-water <= 1,536 MiB, the round's CPU
   within 1 core-hour.
 H suites: pf_mutant_summary.judge on every record of the retention harness against the final sources; one record per file; no other file.
FALSIFIER (falsify()): each check on copies of its real inputs is true clean and false after one corruption applied on a real row: a dropped
 fork (A); a first-review time and, separately, a missing callback and a wrong END in the saved exposure ledger (B); compensating target and
 unavoidable values and, separately, a wrong but valid held gid (C); one input value (D); a calibration value (E); the readings (F); an
 over-budget job (G); a copied mutant record (H).
Usage (run_light): retention_checks.py  (the checks, then the falsifier)"""
import glob
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; RES = Path('/home/yuqing/ctaNew/predictfun/research'); sys.path.insert(0, str(RES))
import pf_entry_retention as M                           # noqa: E402

C, pd, LS, T, L, BE = M.C, M.pd, M.LS, M.T, M.L, M.BE
LOGS = HERE / 'logs'
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
KEY = lambda r: (str(r['day']), int(r['m']), int(r['t_ms']), int(r['side']))
OUT_FILES = ('stages', 'calibration', 'bins', 'decisions', 'changes')


def parts(out):
    man = json.loads((M.EXPO / 'manifest.json').read_text())
    for k in range(len(man['parts'])):
        yield k, json.loads((Path(out) / 'parts' / f'part_{k:03d}.json').read_text()), (Path(out) / 'parts' / f'part_{k:03d}.jsonl.gz').read_bytes()


def expo_rows(expo, k):
    return {KEY(r): r for r in LS.gz_rows((Path(expo) / 'parts' / f'part_{k:03d}.jsonl.gz').read_bytes())}


def check_A(out, expo, R):
    out = Path(out); man = json.loads((M.EXPO / 'manifest.json').read_text()); bad = []; n = 0
    if not M.EX.committed_equal(M.EX.RECEIPT_REL, M.EXPO / 'manifest.json'):
        bad.append('the exposure manifest is not the committed one')
    msha = C.sha(M.EXPO / 'manifest.json'); src = M.sources(); mf = {(f['day'], f['m'], f['t_ms'], f['side']): f for k_ in ('train', 'eval') for f in man['sets'][k_]['forks']}
    for k, rec, b in parts(out):
        rows = LS.gz_rows(b); pt = man['parts'][k]; want = sorted((w['day'], int(w['m']), int(w['t_ms']), s) for w in pt['worlds'] for s in (0, 1))
        er = json.loads((Path(expo) / 'parts' / f'part_{k:03d}.json').read_text())
        if hashlib.sha256(b).hexdigest() != rec['sha256'] or rec['sources'] != src or rec['manifest_sha256'] != msha or rec['exposure_part_sha256'] != er['sha256'] \
                or rec['rows'] != len(rows) or sorted(KEY(r) for r in rows) != want:
            bad.append(f'part {k}: identity / members')
        for r in rows:
            n += 1; f = mf[KEY(r)]
            if f['status'] != 'ok' and r['status'] != f['status']:
                bad.append(f'part {k}: an exclusion not carried')
    S = pd.read_csv(Path(out) / 'stages.csv'); S['day'] = S.day.astype(str)
    if n != 768 or len(S) != 3 * 768 or S.groupby(['day', 'm', 't_ms', 'side']).size().ne(3).any():
        bad.append('not three stage rows for each of the 768 forks')
    R['A'] = dict(bad=bad[:5], forks=n, stage_rows=int(len(S)))
    return not bad


def first_policy(dl, fb, s, gid0, action=None):
    """The first live policy decision of gid0 on side s in a saved decision list (engine fallbacks excluded) -> (t, action) or None."""
    fbs = {(float(x[0]), int(x[1])) for x in fb}
    for d in dl:
        if int(d[1]) == s and d[2] == 'live' and d[4] is not None and int(d[4]) == gid0 and (float(d[0]), s) not in fbs and (action is None or d[3] == action):
            return float(d[0]), d[3]
    return None


def b_row(r, e):
    """Check B on one ok fork row r against its saved exposure row e -> the failures."""
    ap, ak, aw = e['arms']['POST_PI0'], e['arms']['POST_KEEP_INITIAL'], e['arms']['WAIT_PI0']; s, g0 = int(r['side']), int(r['gid0']); why = []
    if not (float(ap['end']) == float(ak['end']) == float(aw['end']) == M.END and float(ap['mark']) == float(ak['mark']) == float(aw['mark']) == float(r['held']['mark'])
            and ap['settings'] == ak['settings'] == aw['settings']):
        why.append('END / mark / settings')
    dl, fb = ap['decision_list'], ap['fallback_list']; fr_, fp_ = first_policy(dl, fb, s, g0), first_policy(dl, fb, s, g0, 'PULL')
    sr, sp = r['stages']['first_review'], r['stages']['first_pull']
    if (fr_ is None) != (sr.get('status') != 'ok') or (fr_ is not None and (fr_[0] != sr['t'] or fr_[1] != sr['action'])):
        why.append('first review')
    if (fp_ is None) != (sp.get('status') != 'ok') or (fp_ is not None and (fp_[0] != sp['t'] or float(sp['effect']) != fp_[0] + BE.CANCEL)):
        why.append('first pull')
    i0 = next((i for i, d in enumerate(dl) if float(d[0]) == 0. and int(d[1]) == s and d[3] == 'POST' and d[4] is not None and int(d[4]) == g0), None)
    stop = float(r['stop_ms'])
    if i0 is None:
        why.append('no forced POST')
    else:
        pre_fb = sum(1 for x in fb if any(float(x[0]) == float(d[0]) and int(x[1]) == int(d[1]) for d in dl[:i0]))
        calls = sum(1 for d in dl[i0 + 1:] if float(d[0]) <= stop) - (sum(1 for x in fb if float(x[0]) <= stop) - pre_fb); c = r['checks']
        if not (calls == c['callbacks'] == c['expected_callbacks'] and c['probe_checked'] == -(-c['callbacks'] // M.AUDIT_EVERY)):
            why.append('callbacks')
    return why


def c_row(r, e):
    """Check C on one ok fork row r against its saved exposure row e -> the failures."""
    s, g0 = int(r['side']), int(r['gid0']); sg = 1. if s == 0 else -1.; mark = float(r['held']['mark']); fh = r['held']['fills']; why = []; vals = {}
    if fh != [x for x in e['arms']['POST_KEEP_INITIAL']['fill_list'] if int(x[1]) == int(e['init_KEEP']['gid0'])] or int(r['held']['gid']) != g0 or g0 != int(e['init_KEEP']['gid0']):
        why.append('held fills / gid')
    for nm in M.STAGES:
        x = r['stages'][nm]
        if x.get('status') != 'ok':
            continue
        eff = float(r['held']['act']) if nm == 'entry' else float(x['effect']); ref = x['inputs'][M.FEATS.index('pf_mid')] if x.get('inputs') else np.nan
        fl = [f for f in fh if int(f[1]) == g0 and float(f[0]) >= eff]; q = sum(float(f[3]) for f in fl); v = sum(sg * float(f[3]) * (mark - float(f[4])) for f in fl)
        pq = float(fl[0][4]) if fl else np.nan; qm = v - (sg * (ref - pq) * q if q > 0 else 0.); tg = x['target']; vals[nm] = v
        tq = np.nan if tg['qm'] is None else float(tg['qm'])
        if not (abs(tg['q'] - q) <= 1e-9 and abs(tg['value'] - v) <= 1e-6 and (abs(tq - qm) <= 1e-6 or (np.isnan(qm) and np.isnan(tq)))
                and abs(tg['severe'] - max(-v - M.SEVERE, 0.)) <= 1e-6 and abs(float(tg['eff']) - eff) <= 1e-9):
            why.append(f'{nm} target')
        if nm != 'entry' and 'entry' in vals and abs(float(x['unavoidable']) - (vals['entry'] - v)) > 1e-6:
            why.append(f'{nm} unavoidable')
    sp = r['stages']['first_pull']
    if sp.get('status') == 'ok':
        eff = float(sp['effect']); fl = [f for f in fh if float(f[0]) >= eff]; kf, pf, di = float(e['init_KEEP']['filled']), float(e['init_PI0']['filled']), float(e['D_initial'])
        if not (abs(sum(sg * float(f[3]) * (mark - float(f[4])) for f in fl) - di) <= 1e-6 and abs(sum(float(f[3]) for f in fl) - (kf - pf)) <= 1e-9
                and [x for x in e['arms']['POST_PI0']['fill_list'] if int(x[1]) == g0] == [f for f in fh if float(f[0]) < eff]):
            why.append('first-pull reconciliation')
    return why


def d_row(r, Mo):
    """Check D on one ok fork row r with the frozen models -> the failures."""
    why = []; s = int(r['side'])
    for nm in M.STAGES:
        x = r['stages'][nm]
        if x.get('status') != 'ok' or not x.get('inputs') or (nm == 'first_pull' and x.get('same_as_review')):
            continue
        a = np.asarray(x['inputs'], np.float32)
        if hashlib.sha256(a.tobytes()).hexdigest() != x['inputs_sha256']:
            why.append(f'{nm} input hash')
        row = {c_: a[i] for i, c_ in enumerate(M.FEATS)}; row['price'] = float(x['price'])
        pr = T.predict_rows(Mo[(s, x['head'])], pd.DataFrame([row]), s); mo = x['moments']; f = lambda k_: float(np.asarray(pr[k_])[0])
        edge = (1. if s == 0 else -1.) * (float(a[M.FEATS.index('pf_mid')]) - float(x['price']))
        if not (f('P') == mo['P'] and f('Eq') == mo['Eq'] and f('Eqm') == mo['Eqm'] and f('V') == mo['V'] and edge == mo['edge']
                and abs(edge * mo['Eq'] + mo['Eqm'] - mo['V']) <= 1e-9 * max(1., abs(mo['V']))):
            why.append(f'{nm} moments')
    return why


def _rows_check(out, expo, R, key, fn, models=False):
    bad = []; n = 0; Mo = L.teacher_assets()[0] if models else None
    for k, rec, b in parts(out):
        ex = None if models else expo_rows(expo, k)
        for r in LS.gz_rows(b):
            if r['status'] != 'ok':
                continue
            n += 1; why = fn(r, Mo) if models else fn(r, ex[KEY(r)])
            if why and len(bad) < 10:
                bad.append(dict(part=k, key=KEY(r), why=why))
            elif why:
                bad.append(None)
    R[key] = dict(rows=n, failing=len(bad), examples=[x for x in bad if x][:5])
    return bool(n and not bad)


def check_B(out, expo, R):
    return _rows_check(out, expo, R, 'B', b_row)


def check_C(out, expo, R):
    return _rows_check(out, expo, R, 'C', c_row)


def check_D(out, expo, R):
    return _rows_check(out, expo, R, 'D', d_row, models=True)


def check_E(out, expo, R):
    out = Path(out); rows = [r for _, _, b in parts(out) for r in LS.gz_rows(b)]; S = M.stage_table(rows); bad = []
    for name, fn in (('stages', None), ('calibration', M.calibration), ('bins', M.bins), ('decisions', M.decisions), ('changes', M.changes)):
        mine = S if fn is None else pd.DataFrame(fn(S)); got = pd.read_csv(out / f'{name}.csv', float_precision='round_trip')
        mine = pd.read_csv(__import__('io').StringIO(mine.to_csv(index=False)), float_precision='round_trip')
        if list(mine.columns) != list(got.columns) or len(mine) != len(got) or not mine.astype(str).equals(got.astype(str)):
            bad.append(f'{name}.csv')
    St = pd.read_csv(out / 'stages.csv', float_precision='round_trip'); cal = pd.read_csv(out / 'calibration.csv', float_precision='round_trip')
    for nm in M.STAGES:
        E = St[(St.stage == nm) & (St.fork_status == 'ok') & (St.stage_status == 'ok') & (St.dec_status == 'model') & St.V.notna()]
        if not len(E):
            continue
        y = (E.q > 1e-12).astype(float); c = cal[(cal.stage == nm) & (cal.day.astype(str) == 'all')].iloc[0]
        mine = dict(brier=((E.P - y) ** 2).mean(), mae_V=(E.V - E.value).abs().mean(), mae_zero=E.value.abs().mean(), mae_mech=(E.edge * E.Eq - E.value).abs().mean())
        if any(abs(float(c[k_]) - float(v)) > 1e-9 * max(1., abs(float(v))) for k_, v in mine.items()):
            bad.append(f'{nm}: pooled metrics')
    R['E'] = dict(bad=bad)
    return not bad


def check_F(out, expo, R):
    rows = [r for _, _, b in parts(out) for r in LS.gz_rows(b)]; mine = M.predicates(M.stage_table(rows)); js = json.loads((Path(out) / 'retention.json').read_text())['predicates']
    same = json.loads(json.dumps(mine, sort_keys=True, default=str)) == json.loads(json.dumps(js, sort_keys=True, default=str))
    R['F'] = dict(readings=js.get('readings'), same=same)
    return bool(same)


def check_G(out, expo, R):
    peaks, cpu, bad = [], 0., []
    for p in glob.glob(str(LOGS / '**' / '*.log'), recursive=True):
        for ln in Path(p).read_text().splitlines():
            m = RL.match(ln)
            if m:
                peaks.append(int(m.group(4))); cpu += float(m.group(6)); bad += [m.group(2)] if int(m.group(4)) >= 2048 else []
    hw = [float(rec.get('hwm_mib', 1e9)) for _, rec, _ in parts(out)]
    R['G'] = dict(jobs=len(peaks), peak_mib=max(peaks) if peaks else None, core_hours=cpu / 3600., bad=bad, part_hwm_max=max(hw) if hw else None)
    return bool(peaks and not bad and cpu / 3600. <= 1. and hw and max(hw) <= 1536.)


def check_H(R):
    import importlib.util
    import pf_mutant_one as O
    import pf_mutant_summary as S
    h = 'pf_entry_retention_mutants.py'; spec = importlib.util.spec_from_file_location(h[:-3], RES / h); H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
    d = LOGS / 'mutants_ret'; files = {'control': d / 'mut_ret_control.log', **{i: d / f'mut_ret_{i}.log' for i in range(len(H.MUTANTS))}}
    recs = {k: S.last_record(p) for k, p in files.items()}
    one = all(p.exists() and sum(1 for ln in p.read_text().splitlines() if ln.startswith('{') and '"cls"' in ln) == 1 for p in files.values())
    extra = sorted({x.name for x in d.glob('mut_ret_*.log')} - {p.name for p in files.values()})
    ok, bad = S.judge(len(H.MUTANTS), recs, O.sources(H), [m_[0] for m_ in H.MUTANTS])
    R['H'] = dict(planted=len(H.MUTANTS), judge_ok=ok, failures=bad[:5], one_record_per_file=one, extra_files=extra)
    return bool(ok and one and not extra)


CHECKS = (('A_identity', check_A), ('B_prefix', check_B), ('C_targets', check_C), ('D_forecasts', check_D), ('E_measurements', check_E), ('F_predicates', check_F), ('G_resources', check_G))


def main(out=M.OUT, expo=M.EXPO):
    R = {}; ck = {}
    for k, f in CHECKS:
        try:
            ck[k] = f(out, expo, R)
        except Exception as e:                                                              # noqa: BLE001  (a check that cannot run is false)
            ck[k] = False; R[k[0]] = dict(error=repr(e)[:300])
    try:
        ck['H_suites'] = check_H(R)
    except Exception as e:                                                                  # noqa: BLE001
        ck['H_suites'] = False; R['H'] = dict(error=repr(e)[:300])
    J = dict(checks=ck, all_true=all(ck.values()), readings=R, source_sha256=C.sha(Path(__file__)), asof=C.stamp())
    (HERE / 'checks.json').write_text(json.dumps(J, indent=1, sort_keys=True, default=str) + '\n')
    print(json.dumps(dict(checks=ck, all_true=J['all_true']), indent=1))


def falsify(out=M.OUT, expo=M.EXPO):
    global LOGS
    out, expo = Path(out), Path(expo); res = {}

    def safe(f, *a):
        try:
            return f(*a)
        except Exception:                                                                 # noqa: BLE001
            return False
    with tempfile.TemporaryDirectory(dir=str(out)) as td:
        td = Path(td)

        def copy_(name, base):
            d = td / name; (d / 'parts').mkdir(parents=True)
            for q in base.iterdir():
                if q.is_file():
                    shutil.copy(q, d / q.name)
            for q in (base / 'parts').iterdir():
                if q.is_file():
                    (d / 'parts' / q.name).symlink_to(q.resolve())
            if (base / 'v2').exists():
                (d / 'v2').symlink_to((base / 'v2').resolve(), target_is_directory=True)
            return d

        def corrupt(d, fn):
            """Replace the first partition of copy d holding a row fn applies to; its record's hash updated."""
            for k in range(12):
                gp, rp = d / 'parts' / f'part_{k:03d}.jsonl.gz', d / 'parts' / f'part_{k:03d}.json'; rows = LS.gz_rows(gp.read_bytes())
                if any(fn(r) for r in rows if r.get('status') == 'ok'):
                    nb = LS.gz_bytes(rows); rec = json.loads(rp.read_text()); gp.unlink(); gp.write_bytes(nb); rp.unlink(); rp.write_text(json.dumps(dict(rec, sha256=hashlib.sha256(nb).hexdigest())))
                    return True
            return False
        once = lambda fn: (lambda st: (lambda r: (not st) and fn(r) and (st.append(1) or True)))([])

        def drop(r):
            return False
        def t_review(r):
            x = r['stages']['first_review']
            if x.get('status') != 'ok':
                return False
            x['t'] = float(x['t']) + 50.; return True
        def missing_cb(r):
            r['checks']['callbacks'] -= 1; r['checks']['expected_callbacks'] -= 1; return True
        def comp(r):
            x = r['stages']['first_review']
            if x.get('status') != 'ok':
                return False
            x['target']['value'] += 1.; x['unavoidable'] -= 1.; return True
        def wrong_gid(r):
            r['held']['gid'] = int(r['held']['gid']) + 1; return True
        def an_input(r):
            x = r['stages']['entry']
            if not x.get('inputs'):
                return False
            x['inputs'][M.FEATS.index('pf_mid')] += .5; return True
        def wrong_end(r):
            r['arms']['POST_PI0']['end'] = 6000.; return True
        clean, clean_e = copy_('clean', out), copy_('clean_expo', expo)
        for name, chk, fn, on in (('B_review_time', check_B, t_review, 'out'), ('B_missing_callback', check_B, missing_cb, 'out'), ('C_compensating', check_C, comp, 'out'),
                                  ('C_wrong_gid', check_C, wrong_gid, 'out'), ('D_input', check_D, an_input, 'out'), ('B_wrong_END', check_B, wrong_end, 'expo')):
            d = copy_(name, out if on == 'out' else expo); ap = corrupt(d, once(fn))
            args_c, args_x = (d, clean_e) if on == 'out' else (clean, d)
            res[name] = dict(clean=safe(chk, clean, clean_e, {}), corrupted=safe(chk, *args_c, {}) if on == 'out' else safe(chk, *args_x, {}), applied=bool(ap))
        d = copy_('A_drop', out); gp = d / 'parts' / 'part_000.jsonl.gz'; rows = LS.gz_rows(gp.read_bytes()); rp = d / 'parts' / 'part_000.json'; rec = json.loads(rp.read_text())
        nb = LS.gz_bytes(rows[:-1]); gp.unlink(); gp.write_bytes(nb); rp.unlink(); rp.write_text(json.dumps(dict(rec, sha256=hashlib.sha256(nb).hexdigest(), rows=len(rows) - 1)))
        res['A'] = dict(clean=safe(check_A, clean, clean_e, {}), corrupted=safe(check_A, d, clean_e, {}), applied=True)
        d = copy_('E', out); c_ = pd.read_csv(d / 'calibration.csv', float_precision='round_trip'); c_.loc[0, 'mae_V'] += 1.; c_.to_csv(d / 'calibration.csv', index=False)
        res['E'] = dict(clean=safe(check_E, clean, clean_e, {}), corrupted=safe(check_E, d, clean_e, {}), applied=True)
        d = copy_('F', out); j = json.loads((d / 'retention.json').read_text()); j['predicates']['readings'] = {k_: (not v) for k_, v in j['predicates']['readings'].items()}
        (d / 'retention.json').write_text(json.dumps(j)); res['F'] = dict(clean=safe(check_F, clean, clean_e, {}), corrupted=safe(check_F, d, clean_e, {}), applied=True)
        saved = LOGS
        try:
            h = td / 'G_logs'; shutil.copytree(saved, h); LOGS = h; clG = safe(check_G, out, expo, {})
            (h / 'over_budget.log').write_text('2026-11-21T00:00:00Z tag=over_budget rc=0 peak_rss_mib=100 wall_s=3700 cpu_user_s=3700.0\n')
            res['G'] = dict(clean=clG, corrupted=safe(check_G, out, expo, {}), applied=True)
            h2 = td / 'H_logs'; shutil.copytree(saved, h2); LOGS = h2; clH = safe(check_H, {})
            shutil.copy(h2 / 'mutants_ret' / 'mut_ret_0.log', h2 / 'mutants_ret' / 'mut_ret_1.log'); res['H'] = dict(clean=clH, corrupted=safe(check_H, {}), applied=True)
        finally:
            LOGS = saved
    ok = all(v['clean'] is True and v['corrupted'] is False and v['applied'] for v in res.values())
    (HERE / 'checks_falsifier.json').write_text(json.dumps(dict(cases=res, falsifier_ok=ok, asof=C.stamp()), indent=1, sort_keys=True) + '\n')
    print(json.dumps(dict(cases=res, falsifier_ok=ok)))


def selftest():
    """b_row / c_row / d_row on a synthetic fork_stages row and its exposure row: clean rows pass; each corruption is flagged."""
    import copy
    import os
    import pf_r2_labels_selftest as LT
    R2 = M.R2; EX = M.EX; bad = []; wj = json.loads((Path(os.environ['PF_TEACHER_SYN']) / 'world.json').read_text()); day = '20261024'
    W = L.worlds_of_day(day, Path(wj['kp']), 2); _, prs, fair, depth, ff = L.world_inputs(day, W, R2.KP.Syn(), R2.X.syn_inputs, R2.X.syn_features())
    Mst, sst = LT.stubs(-12.), LT.OpenSupport(); pair = None
    for w in W:
        wb = L.world_bind(day, w, prs, fair, depth, ff)[0]
        if wb is None:
            continue
        sel = dict(day=day, side=0, decision_ms=0, source='neutral', shadow_source='entry', offset_ms=0, kind='post')
        ctx = L.fork_for(sel, wb, Mst, sst); stp = L.pair_bound(ctx, L.teacher_factory(Mst, sst, wb, every=M.AUDIT_EVERY), 'post')
        if stp.get('status') != 'ok':
            continue
        f = dict(day=day, m=int(w['m']), t_ms=int(round(float(w['t']))), hour=0, side=0, oid=1, status='ok', sel=sel, fork_fingerprint=ctx['fingerprint'], Y=stp['Y'],
                 arms={n: {h: a[h] for h in ('value', 'fills_sha256', 'decisions_sha256', 'requests_sha256')} for n, a in stp['arms'].items()})
        e = json.loads(json.dumps(EX.fork_package(f, wb, Mst, sst), default=str)); r = json.loads(json.dumps(M.fork_stages(f, wb, Mst, sst, e), default=str)); pair = (r, e); break
    if pair is None or pair[0]['status'] != 'ok':
        sys.exit(f'RETENTION CHECKS SELFTEST FAILED: no clean synthetic row ({None if pair is None else pair[0]["status"]})')
    r, e = pair
    for nm, fn, args in (('B', b_row, (r, e)), ('C', c_row, (r, e)), ('D', d_row, (r, Mst))):
        if fn(*args):
            bad.append(f'{nm}: a clean synthetic row flagged ({fn(*args)})')
    def mut(fn):
        x, y = copy.deepcopy(r), copy.deepcopy(e); fn(x, y); return x, y
    cases = (('B', b_row, lambda x, y: x['stages']['first_review'].update(t=float(x['stages']['first_review']['t']) + 50.)),
             ('B', b_row, lambda x, y: x['checks'].update(callbacks=x['checks']['callbacks'] - 1, expected_callbacks=x['checks']['expected_callbacks'] - 1)),
             ('B', b_row, lambda x, y: y['arms']['POST_PI0'].update(end=6000.)),
             ('C', c_row, lambda x, y: (x['stages']['first_review']['target'].update(value=x['stages']['first_review']['target']['value'] + 1.),
                                        x['stages']['first_review'].update(unavoidable=x['stages']['first_review']['unavoidable'] - 1.))),
             ('C', c_row, lambda x, y: x['held'].update(gid=int(x['held']['gid']) + 1)),
             ('C', c_row, lambda x, y: y.update(D_initial=float(y['D_initial']) + 1.)),
             ('D', d_row, lambda x, y: x['stages']['entry']['inputs'].__setitem__(M.FEATS.index('pf_mid'), x['stages']['entry']['inputs'][M.FEATS.index('pf_mid')] + .5)))
    for i, (nm, fn, f_) in enumerate(cases):
        x, y = mut(f_)
        if not (fn(x, Mst) if nm == 'D' else fn(x, y)):
            bad.append(f'{nm} case {i}: a corruption not flagged')
    if r['stages']['first_pull'].get('status') != 'ok':
        bad.append('the synthetic row has no first pull (the reconciliation case is vacuous)')
    if bad:
        sys.exit('RETENTION CHECKS SELFTEST FAILED: ' + '; '.join(bad))
    print(f'retention checks selftest passed: a clean synthetic row passes B / C / D; {len(cases)} corruptions are flagged (review time, a missing callback, a wrong END, '
          'compensating target / unavoidable, a wrong but valid gid, D_initial, one input)')


if __name__ == '__main__':
    if sys.argv[1:] == ['selftest']:
        selftest()
    else:
        main(); falsify()
