"""PI0-ADMISSION-EXPOSURE-v1: THE DECLARED CHECKS (review 99fb165d; 2026-11-21, session aea5bfb9). Declared in the freeze, before any branch
runs. Each check recomputes from the saved partitions, the committed R4E traces and the committed manifest; a check that cannot run is false.
 A identity: the manifest is the committed one and its selection is recomputed from the committed R4E records (census, ranks, cells,
   weights summing to the census); every partition matches its bytes, the final sources, the manifest and its R4E record, and holds exactly
   its selected generations.
 B arms (independent of the rerun): on every ok row, against the saved R4E PI0 trace - the selected decision is the trace's POST of gid; END
   and the mark are the trace's; the KEEP arm equals the trace up to its first override (the first difference is KEEP for the trace's PULL of
   gid at the logged override tick; fills to it equal) or entirely without one; both ledgers recompute from their fills; R_factual is the
   trace value, D_EXPOSURE = R_keep - R_factual, and D_selected / D_focal_rest / D_opposite recompute from both arms' fills; the KEEP arm's
   binding and causal probes are clean.
 C targets: the held fills equal the KEEP arm's selected-gid fills; every stage target recomputed from the held fills at its effect; the
   unavoidable value; at the first pull the held value after the effect equals D_selected and the trace's selected fills are the held ones
   before it.
 D forecasts: input bit hashes; P, E[q], E[qm], V recomputed by the frozen predictor, bit-equal; V = edge x E[q] + E[qm].
 E measurements: generations / stages / economics / forecasts / bins / transitions / crosstab / retention_v2 recomputed equal the outputs.
 F predicates: recomputed equal admission.json's.
 G resources: every run_light job under logs/ (recursive) below 2,048 MiB, partition high-water <= 1,536 MiB, the round within 1 core-hour.
 H suites: pf_mutant_summary.judge on every record of the admission harness against the final sources; one record per file; no other file.
FALSIFIER (falsify(), writes checks_falsifier.json): each check true on clean copies and false after one corruption on a real row: a dropped
 generation (A); a KEEP decision before the override, and separately a factual value and a wrong END (B); compensating target / unavoidable
 values and a wrong but valid held gid (C); one input (D); an output value (E); the readings (F); an over-budget job (G); a copied mutant
 record (H). SELFTEST (selftest()): b_row / c_row / d_row on synthetic world_rows output pass clean and flag each corruption.
Usage (run_light): admission_checks.py [selftest]  (no argument: the checks, then the falsifier)"""
import copy
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
import pf_pi0_admission as M                             # noqa: E402

C, pd, LS, T, L, BE, RT = M.C, M.pd, M.LS, M.T, M.L, M.BE, M.RT
LOGS = HERE / 'logs'
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
WKEY = lambda r: (str(r['day']), int(r['m']), int(r['t_ms']))
GKEY = lambda r: (str(r['day']), int(r['m']), int(r['t_ms']), int(r['side']), int(r['gid']), int(r['decision_ms']))
OUTS = ('generations', 'stages', 'economics', 'forecasts', 'bins', 'transitions', 'crosstab', 'retention_v2')


def parts(out):
    man = json.loads((Path(out) / 'manifest.json').read_text())
    for k in man['parts']:
        yield k, json.loads((Path(out) / 'parts' / f'part_{k:03d}.json').read_text()), (Path(out) / 'parts' / f'part_{k:03d}.jsonl.gz').read_bytes()


_TR = {}


def traces():
    if not _TR:
        for k, rc, rows in M.r4e_parts():
            for r in rows:
                if r['status'] == 'ok':
                    _TR[WKEY(r)] = r['episodes']['PI0']['trace']
    return _TR


def ledger_bad(a, mark):
    sg = lambda x: 1. if int(x) == 0 else -1.; fl = a['fill_list']
    cash = -sum(sg(f[2]) * float(f[3]) * float(f[4]) for f in fl); pos = sum(sg(f[2]) * float(f[3]) for f in fl)
    return not (abs(cash + pos * mark - float(a['value'])) <= 1e-6 and abs(cash - float(a['cash'])) <= 1e-6 and abs(pos - float(a['position'])) <= 1e-9)


def comps(fl, gid, s, mark):
    out = dict(sel=0., rest=0., opp=0.)
    for f in fl:
        v = (1. if int(f[2]) == 0 else -1.) * float(f[3]) * (mark - float(f[4]))
        out['sel' if int(f[1]) == gid else 'rest' if int(f[2]) == s else 'opp'] += v
    return out


def b_row(r, tr):
    why = []; s, gid, t = int(r['side']), int(r['gid']), float(r['decision_ms']); dl = tr['decision_list']; ka = r['keep_arm']; mark = float(tr['mark'])
    if len([d for d in dl if float(d[0]) == t and int(d[1]) == s and d[2] == 'absent' and d[3] == 'POST' and d[4] is not None and int(d[4]) == gid]) != 1:
        why.append('not the trace\'s POST of gid')
    if not (float(ka['end']) == float(tr['end']) == M.END and float(ka['mark']) == mark):
        why.append('END / mark')
    dk = ka['decision_list']; i = next((j for j, (a, b) in enumerate(zip(dk, dl)) if a != b), None)
    if int(r['overrides']) > 0:
        if i is None or not (dk[i][3] == 'KEEP' and dl[i][3] == 'PULL' and int(dk[i][4]) == gid == int(dl[i][4]) and float(dk[i][0]) == float(r['shadow'][0]['t'])):
            why.append('prefix')
        elif [f for f in ka['fill_list'] if float(f[0]) <= float(dk[i][0])] != [f for f in tr['fill_list'] if float(f[0]) <= float(dk[i][0])]:
            why.append('prefix fills')
    elif dk != dl or ka['fill_list'] != tr['fill_list']:
        why.append('a no-override arm differs from the trace')
    if ledger_bad(ka, mark) or ledger_bad(tr, mark):
        why.append('ledgers from fills')
    if not (abs(float(r['R_factual']) - float(tr['value'])) <= 1e-9 and abs(float(r['R_keep']) - float(ka['value'])) <= 1e-9
            and abs(float(r['D_EXPOSURE']) - (float(r['R_keep']) - float(r['R_factual']))) <= 1e-9):
        why.append('values')
    cf, ck = comps(tr['fill_list'], gid, s, mark), comps(ka['fill_list'], gid, s, mark)
    if any(abs(float(r[n]) - (ck[k] - cf[k])) > 1e-6 for n, k in (('D_selected', 'sel'), ('D_focal_rest', 'rest'), ('D_opposite', 'opp'))):
        why.append('components from fills')
    if ka['teacher']['binding_bad'] != 0 or (ka['causal'] or {}).get('mismatch', 1) != 0 or (ka['causal'] or {}).get('parity_mismatch', 1) != 0:
        why.append('binding / causal')
    return why


def c_row(r, tr):
    why = []; s, gid = int(r['side']), int(r['gid']); sg = 1. if s == 0 else -1.; mark = float(r['held']['mark']); fh = r['held']['fills']; vals = {}
    if fh != [x for x in r['keep_arm']['fill_list'] if int(x[1]) == gid] or int(r['held']['gid']) != gid:
        why.append('held fills / gid')
    for nm in RT.STAGES:
        x = r['stages'][nm]
        if x.get('status') != 'ok':
            continue
        eff = float(r['held']['act']) if nm == 'entry' else float(x['effect']); ref = x['inputs'][RT.FEATS.index('pf_mid')] if x.get('inputs') else np.nan
        fl = [f for f in fh if int(f[1]) == gid and float(f[0]) >= eff]; q = sum(float(f[3]) for f in fl); v = sum(sg * float(f[3]) * (mark - float(f[4])) for f in fl)
        pq = float(fl[0][4]) if fl else np.nan; qm = v - (sg * (ref - pq) * q if q > 0 else 0.); tg = x['target']; vals[nm] = v; tq = np.nan if tg['qm'] is None else float(tg['qm'])
        if not (abs(tg['q'] - q) <= 1e-9 and abs(tg['value'] - v) <= 1e-6 and (abs(tq - qm) <= 1e-6 or (np.isnan(qm) and np.isnan(tq))) and abs(float(tg['eff']) - eff) <= 1e-9):
            why.append(f'{nm} target')
        if nm != 'entry' and 'entry' in vals and abs(float(x['unavoidable']) - (vals['entry'] - v)) > 1e-6:
            why.append(f'{nm} unavoidable')
    sp = r['stages']['first_pull']
    if sp.get('status') == 'ok':
        eff = float(sp['effect']); fl = [f for f in fh if float(f[0]) >= eff]
        if not (abs(sum(sg * float(f[3]) * (mark - float(f[4])) for f in fl) - float(r['D_selected'])) <= 1e-6
                and [x for x in tr['fill_list'] if int(x[1]) == gid] == [f for f in fh if float(f[0]) < eff]):
            why.append('first-pull reconciliation')
    return why


def d_row(r, Mo):
    why = []; s = int(r['side'])
    for nm in RT.STAGES:
        x = r['stages'][nm]
        if x.get('status') != 'ok' or not x.get('inputs') or (nm == 'first_pull' and x.get('same_as_review')):
            continue
        a = np.asarray(x['inputs'], np.float32)
        if hashlib.sha256(a.tobytes()).hexdigest() != x['inputs_sha256']:
            why.append(f'{nm} input hash')
        row = {c_: a[i] for i, c_ in enumerate(RT.FEATS)}; row['price'] = float(x['price'])
        pr = T.predict_rows(Mo[(s, x['head'])], pd.DataFrame([row]), s); mo = x['moments']; f = lambda k_: float(np.asarray(pr[k_])[0])
        edge = (1. if s == 0 else -1.) * (float(a[RT.FEATS.index('pf_mid')]) - float(x['price']))
        if not (f('P') == mo['P'] and f('Eq') == mo['Eq'] and f('Eqm') == mo['Eqm'] and f('V') == mo['V'] and edge == mo['edge']
                and abs(edge * mo['Eq'] + mo['Eqm'] - mo['V']) <= 1e-9 * max(1., abs(mo['V']))):
            why.append(f'{nm} moments')
    return why


def _rows_check(out, R, key, fn, models=False):
    bad = []; n = 0; Mo = L.teacher_assets()[0] if models else None; TR = None if models else traces()
    for k, rec, b in parts(out):
        for r in LS.gz_rows(b):
            if r['status'] != 'ok':
                continue
            n += 1; why = fn(r, Mo) if models else fn(r, TR[WKEY(r)])
            if why and len(bad) < 10:
                bad.append(dict(part=k, key=GKEY(r), why=why))
            elif why:
                bad.append(None)
    R[key] = dict(rows=n, failing=len(bad), examples=[x for x in bad if x][:5])
    return bool(n and not bad)


def check_A(out, R):
    out = Path(out); man = json.loads((out / 'manifest.json').read_text()); bad = []
    if not M.committed_equal(f'{M.RECEIPT}/manifest.json', out / 'manifest.json'):
        bad.append('the manifest is not the committed one')
    gens, worlds = M.census(M.r4e_parts()); sel, cells = M.select(gens)
    if [GKEY(g) for g in sel] != [GKEY(g) for g in man['selected']] or [g['weight'] for g in sel] != [g['weight'] for g in man['selected']] \
            or abs(sum(g['weight'] for g in sel) - len(gens)) > 1e-9 or man['census'] != len(gens):
        bad.append('the selection is not recomputed from the committed R4E records')
    msha = C.sha(out / 'manifest.json'); src = M.sources(); n = 0
    for k, rec, b in parts(out):
        rows = LS.gz_rows(b); want = sorted(GKEY(g) for g in man['selected'] if int(g['part']) == int(k)); n += len(rows)
        if hashlib.sha256(b).hexdigest() != rec['sha256'] or rec['sources'] != src or rec['manifest_sha256'] != msha or rec['r4e_record_sha256'] != C.sha(M.R4E_REC / f'part_{k:03d}.json') \
                or sorted(GKEY(r) for r in rows) != want:
            bad.append(f'part {k}: identity / members')
    if n != len(man['selected']):
        bad.append('not every selected generation has a row')
    R['A'] = dict(bad=bad[:5], rows=n, selected=len(man['selected']), census=len(gens))
    return not bad


def check_B(out, R):
    return _rows_check(out, R, 'B', b_row)


def check_C(out, R):
    return _rows_check(out, R, 'C', c_row)


def check_D(out, R):
    return _rows_check(out, R, 'D', d_row, models=True)


def check_E(out, R):
    out = Path(out); rows = [r for _, _, b in parts(out) for r in LS.gz_rows(b)]; G = M.gen_table(rows); S = M.stage_rows(rows); bad = []
    CT, nj, nmiss = M.crosstab(); RV, RF = M.retention_v2()
    for name, df in (('generations', G), ('stages', S), ('economics', pd.DataFrame(M.economics(G))), ('forecasts', pd.DataFrame(M.forecast_metrics(S) + M.forecast_metrics(S, weighted=False))),
                     ('bins', pd.DataFrame(M.bins(S))), ('transitions', pd.DataFrame(M.transitions(S))), ('crosstab', pd.DataFrame(CT)), ('retention_v2', pd.DataFrame(RV))):
        mine = pd.read_csv(__import__('io').StringIO(df.to_csv(index=False)), float_precision='round_trip'); got = pd.read_csv(out / f'{name}.csv', float_precision='round_trip')
        if list(mine.columns) != list(got.columns) or len(mine) != len(got) or not mine.astype(str).equals(got.astype(str)):
            bad.append(f'{name}.csv')
    R['E'] = dict(bad=bad)
    return not bad


def check_F(out, R):
    out = Path(out); rows = [r for _, _, b in parts(out) for r in LS.gz_rows(b)]; RV, _ = M.retention_v2()
    mine = M.predicates(M.gen_table(rows), M.stage_rows(rows), RV); js = json.loads((out / 'admission.json').read_text())['predicates']
    same = json.loads(json.dumps(mine, sort_keys=True, default=str)) == json.loads(json.dumps(js, sort_keys=True, default=str))
    R['F'] = dict(readings=js.get('readings'), same=same)
    return bool(same)


def check_G(out, R):
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
    h = 'pf_pi0_admission_mutants.py'; spec = importlib.util.spec_from_file_location(h[:-3], RES / h); H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
    d = LOGS / 'mutants_adm'; files = {'control': d / 'mut_adm_control.log', **{i: d / f'mut_adm_{i}.log' for i in range(len(H.MUTANTS))}}
    recs = {k: S.last_record(p) for k, p in files.items()}
    one = all(p.exists() and sum(1 for ln in p.read_text().splitlines() if ln.startswith('{') and '"cls"' in ln) == 1 for p in files.values())
    extra = sorted({x.name for x in d.glob('mut_adm_*.log')} - {p.name for p in files.values()})
    ok, bad = S.judge(len(H.MUTANTS), recs, O.sources(H), [m_[0] for m_ in H.MUTANTS])
    R['H'] = dict(planted=len(H.MUTANTS), judge_ok=ok, failures=bad[:5], one_record_per_file=one, extra_files=extra)
    return bool(ok and one and not extra)


CHECKS = (('A_identity', check_A), ('B_arms', check_B), ('C_targets', check_C), ('D_forecasts', check_D), ('E_measurements', check_E), ('F_predicates', check_F), ('G_resources', check_G))


def main(out=M.OUT):
    R = {}; ck = {}
    for k, f in CHECKS:
        try:
            ck[k] = f(out, R)
        except Exception as e:                                                              # noqa: BLE001  (a check that cannot run is false)
            ck[k] = False; R[k[0]] = dict(error=repr(e)[:300])
    try:
        ck['H_suites'] = check_H(R)
    except Exception as e:                                                                  # noqa: BLE001
        ck['H_suites'] = False; R['H'] = dict(error=repr(e)[:300])
    J = dict(checks=ck, all_true=all(ck.values()), readings=R, source_sha256=C.sha(Path(__file__)), asof=C.stamp())
    (HERE / 'checks.json').write_text(json.dumps(J, indent=1, sort_keys=True, default=str) + '\n')
    print(json.dumps(dict(checks=ck, all_true=J['all_true']), indent=1))


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
                    (d / 'parts' / q.name).symlink_to(q.resolve())
            return d

        def corrupt(d, fn):
            man = json.loads((d / 'manifest.json').read_text())
            for k in man['parts']:
                gp, rp = d / 'parts' / f'part_{k:03d}.jsonl.gz', d / 'parts' / f'part_{k:03d}.json'; rows = LS.gz_rows(gp.read_bytes())
                if any(fn(r) for r in rows if r.get('status') == 'ok'):
                    nb = LS.gz_bytes(rows); rec = json.loads(rp.read_text()); gp.unlink(); gp.write_bytes(nb); rp.unlink(); rp.write_text(json.dumps(dict(rec, sha256=hashlib.sha256(nb).hexdigest())))
                    return True
            return False

        def keep_dec(r):
            if int(r['overrides']) < 1:
                return False
            dl = r['keep_arm']['decision_list']; i = next((j for j, d in enumerate(dl) if float(d[0]) < float(r['shadow'][0]['t']) and float(d[0]) > float(r['decision_ms'])), None)
            if i is None:
                return False
            dl[i][3] = 'X'; return True
        def fact_value(r):
            r['R_factual'] = float(r['R_factual']) + 1.; r['D_EXPOSURE'] = float(r['D_EXPOSURE']) - 1.; r['D_selected'] = float(r['D_selected']) - 1.; return True
        def wrong_end(r):
            r['keep_arm']['end'] = 6000.; return True
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
            x['inputs'][RT.FEATS.index('pf_mid')] += .5; return True
        clean = copy_('clean')
        for name, chk, fn in (('B_keep_decision', check_B, keep_dec), ('B_factual_value', check_B, fact_value), ('B_wrong_END', check_B, wrong_end),
                              ('C_compensating', check_C, comp), ('C_wrong_gid', check_C, wrong_gid), ('D_input', check_D, an_input)):
            d = copy_(name); ap = corrupt(d, fn); res[name] = dict(clean=safe(chk, clean, {}), corrupted=safe(chk, d, {}), applied=bool(ap))
        d = copy_('A_drop'); man = json.loads((d / 'manifest.json').read_text()); k0 = man['parts'][0]; gp = d / 'parts' / f'part_{k0:03d}.jsonl.gz'; rows = LS.gz_rows(gp.read_bytes())
        rp = d / 'parts' / f'part_{k0:03d}.json'; rec = json.loads(rp.read_text()); nb = LS.gz_bytes(rows[:-1]); gp.unlink(); gp.write_bytes(nb); rp.unlink()
        rp.write_text(json.dumps(dict(rec, sha256=hashlib.sha256(nb).hexdigest(), rows=len(rows) - 1))); res['A'] = dict(clean=safe(check_A, clean, {}), corrupted=safe(check_A, d, {}), applied=True)
        d = copy_('E'); e_ = pd.read_csv(d / 'economics.csv', float_precision='round_trip'); e_.loc[0, 'wmean_D'] = float(e_.loc[0, 'wmean_D']) + 1.; e_.to_csv(d / 'economics.csv', index=False)
        res['E'] = dict(clean=safe(check_E, clean, {}), corrupted=safe(check_E, d, {}), applied=True)
        d = copy_('F'); j = json.loads((d / 'admission.json').read_text()); j['predicates']['readings'] = {k_: (not v) for k_, v in j['predicates']['readings'].items()}
        (d / 'admission.json').write_text(json.dumps(j)); res['F'] = dict(clean=safe(check_F, clean, {}), corrupted=safe(check_F, d, {}), applied=True)
        saved = LOGS
        try:
            h = td / 'G_logs'; shutil.copytree(saved, h); LOGS = h; clG = safe(check_G, out, {})
            (h / 'over_budget.log').write_text('2026-11-21T00:00:00Z tag=over_budget rc=0 peak_rss_mib=100 wall_s=3700 cpu_user_s=3700.0\n')
            res['G'] = dict(clean=clG, corrupted=safe(check_G, out, {}), applied=True)
            h2 = td / 'H_logs'; shutil.copytree(saved, h2); LOGS = h2; clH = safe(check_H, {})
            shutil.copy(h2 / 'mutants_adm' / 'mut_adm_0.log', h2 / 'mutants_adm' / 'mut_adm_1.log'); res['H'] = dict(clean=clH, corrupted=safe(check_H, {}), applied=True)
        finally:
            LOGS = saved
    ok = all(v['clean'] is True and v['corrupted'] is False and v['applied'] for v in res.values())
    (HERE / 'checks_falsifier.json').write_text(json.dumps(dict(cases=res, falsifier_ok=ok, asof=C.stamp()), indent=1, sort_keys=True) + '\n')
    print(json.dumps(dict(cases=res, falsifier_ok=ok)))


def selftest():
    """b_row / c_row / d_row on synthetic world_rows output (header)."""
    import os
    import pf_r2_labels_selftest as LT
    R2 = M.R2; bad = []; wj = json.loads((Path(os.environ['PF_TEACHER_SYN']) / 'world.json').read_text()); day = '20261024'
    W = L.worlds_of_day(day, Path(wj['kp']), 2); _, prs, fair, depth, ff = L.world_inputs(day, W, R2.KP.Syn(), R2.X.syn_inputs, R2.X.syn_features())
    Mst, sst = LT.stubs(-12.), LT.OpenSupport(); pair = None
    for w in W:
        wb = L.world_bind(day, w, prs, fair, depth, ff)[0]
        if wb is None:
            continue
        probe, cst = T.causal_probe(35); pol = M.A.Teacher0(Mst, sst); pol.causal = probe; B = BE.Branch(wb['path'], 0, ident=R2.episode_ident(day, w['o'][0])); pol.bind(B, wb)
        for k in range(M.NT):
            BE.step(B, k, pol)
        Lx = BE.finish(B); tr = dict(fill_list=RT.jl(B.E.fills), decision_list=RT.jl(B.decisions), fills_sha256=L._h(Lx['fills']), decisions_sha256=L._h(Lx['decisions']),
                                     value=float(Lx['value']), cash=float(Lx['cash']), position=float(Lx['position']), teacher=dict(calls=len(pol.log)), end=float(B.end),
                                     settings=dict(B.settings), mark=float(B.mark))
        tms = int(round(float(w['t']))); posts = [d for d in tr['decision_list'] if d[2] == 'absent' and d[3] == 'POST']
        for d in posts:
            g = dict(day=day, m=int(w['m']), t_ms=tms, side=int(d[1]), gid=int(d[4]), decision_ms=int(round(float(d[0]))), weight=1., n_cell=1, cell=list(M.cell_of(tms, float(d[0]), int(d[1]))))
            rr = json.loads(json.dumps(M.world_rows(day, dict(episodes=dict(PI0=dict(trace=tr))), wb, w, Mst, sst, [g])[0], default=str))
            if rr and rr[0]['status'] == 'ok' and rr[0]['overrides'] > 0 and rr[0]['stages']['first_pull'].get('status') == 'ok':
                pair = (rr[0], tr); break
        if pair:
            break
    if pair is None:
        sys.exit('ADMISSION CHECKS SELFTEST FAILED: no synthetic generation with an override and a first pull')
    r, tr = pair
    for nm, fn, arg in (('B', b_row, tr), ('C', c_row, tr), ('D', d_row, Mst)):
        if fn(r, arg):
            bad.append(f'{nm}: a clean synthetic row flagged ({fn(r, arg)})')
    cases = (('B', b_row, lambda x, y: x['keep_arm']['decision_list'][next(j for j, d in enumerate(x['keep_arm']['decision_list']) if float(d[0]) > float(x['decision_ms']))].__setitem__(3, 'X')),
             ('B', b_row, lambda x, y: x.update(R_factual=float(x['R_factual']) + 1., D_EXPOSURE=float(x['D_EXPOSURE']) - 1., D_selected=float(x['D_selected']) - 1.)),
             ('B', b_row, lambda x, y: x['keep_arm'].update(end=6000.)),
             ('C', c_row, lambda x, y: (x['stages']['first_review']['target'].update(value=x['stages']['first_review']['target']['value'] + 1.),
                                        x['stages']['first_review'].update(unavoidable=x['stages']['first_review']['unavoidable'] - 1.))),
             ('C', c_row, lambda x, y: x['held'].update(gid=int(x['held']['gid']) + 1)),
             ('C', c_row, lambda x, y: x.update(D_selected=float(x['D_selected']) + 1.)),
             ('D', d_row, lambda x, y: x['stages']['entry']['inputs'].__setitem__(RT.FEATS.index('pf_mid'), x['stages']['entry']['inputs'][RT.FEATS.index('pf_mid')] + .5)))
    for i, (nm, fn, f_) in enumerate(cases):
        x, y = copy.deepcopy(r), copy.deepcopy(tr); f_(x, y)
        if not (fn(x, Mst) if nm == 'D' else fn(x, y)):
            bad.append(f'{nm} case {i}: a corruption not flagged')
    if bad:
        sys.exit('ADMISSION CHECKS SELFTEST FAILED: ' + '; '.join(bad))
    print(f'admission checks selftest passed: a clean synthetic row passes B / C / D; {len(cases)} corruptions are flagged (a KEEP decision, a factual value, a wrong END, '
          'compensating target / unavoidable, a wrong but valid gid, D_selected, one input)')


if __name__ == '__main__':
    if sys.argv[1:] == ['selftest']:
        selftest()
    else:
        main(); falsify()
