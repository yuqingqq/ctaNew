"""R4E ADV2-ENTRY-SMOOTH-v1: THE ROUND'S DECLARED CHECKS (review 371aab56; 2026-11-21, session aea5bfb9). Declared in the freeze, before the
fit. Every check recomputes from the retained artifacts; a stored boolean is never the evidence. A check that cannot run is false, with its reason.
 A fit: fit.json has the two SMOOTH heads, PARAMS = pf_adv2.PARAMS with exactly min_data_in_leaf 1,000 (the frozen hash), 50 rounds; R3's
   extraction hash equals its committed receipt copy; each model file has its recorded hash and, reloaded, reproduces its in-sample weighted
   error on R3's POST training rows (which equal R3's extraction manifest).
 B order: the receipt copy fit/fit.json is in a commit that is an ancestor of HEAD, committed before the evaluation's asof and every replay
   part's asof; the evaluation and every part record name that fit.json's hash.
 C reuse: the evaluation's rows are R3's: its predictions file and R3's have the same POST row keys, labels, weights and RICH predictions,
   compared here from the two files; R3's accepted evaluation dataset is the declared one.
 D readings: every day x side x candidate has the groups all / POST / WAIT / fallback and POST / WAIT screens in the declared set; each
   (day, side, candidate) calibration's w_sum totals its supported rows' weight; the strata cover the declared dimensions; the predicates,
   recomputed here from the readings, screens and control, equal the stored ones.
 E control: per candidate the (day, side) and (day) scopes are present and finite; 500 draws, seed 20261121.
 F replay: every partition's members are R3's; every world's status is R3's; each ok world has the four policies, each episode a trace that
   passes pf_r2_label_sample._ledger_bad and _arm_calls_bad (the required callbacks recomputed from its decision log), zero causal / parity
   mismatches, the probe at every 35th call, cap_ok and value = the trace's; N0 is 0 with no order; the three controls equal R3's episodes
   (value, decision and fill hashes, read here from R3's parts).
 G reached: one entry per ok world and side; each ok pair passes pf_r2_label_sample._pair_bad against the frozen evaluation plan with no
   failure (END, mark, fill times, context, Y, ledgers, pre-fork fills, fingerprint, required callbacks, audit rate, causal fork inputs), its
   fork inputs equal the recorded decision's, and both heads recomputed here on its saved inputs give the stored predictions; only ok,
   no_eligible_decision and the frozen pair's missing_mark appear.
 H suites: pf_mutant_summary.judge on every record of the three harnesses against the final sources; one record line per file; no other file.
 I resources: every run_light job of the round peaked below 2,048 MiB; the round's CPU is within 8 core-hours.
FALSIFIER (rule 15; falsify()): C, D, F, G and H on copies of their real inputs are true clean and false after one corruption each (a label in
 the predictions; a stored predicate; a fill in an episode trace; a control episode's value; an arm fill of a reached pair; a copied record).
Usage (run_light): r4e_checks.py  (the checks, then the falsifier)"""
import glob
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; RES = Path('/home/yuqing/ctaNew/predictfun/research'); sys.path.insert(0, str(RES))
import pf_adv2_entry as N                                # noqa: E402
import pf_adv2_entry_eval as NE                          # noqa: E402
import pf_adv2_entry_replay as NR                        # noqa: E402

A, AE, AR, LS, L = N.A, NE.AE, NR.AR, NR.LS, NR.L
C, pd, T = A.C, A.pd, A.T
LOGS = HERE / 'logs'; REPO = RES.parent.parent
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
VERDICTS = ('PASS', 'FAIL', 'NOT_EVALUABLE', 'INSUFFICIENT_SUPPORT')


def git(*a):
    return subprocess.run(['git', '-C', str(REPO), *a], capture_output=True, text=True)


def check_A(out, R):
    fz = json.loads((out / 'fit.json').read_text()); E = N.r3_train(); lg = A.lgb(); bad = []
    for (s, h) in N.HEADS:
        k = f'{N.NAME} {s} {h}'; p = N.model_path(out, s)
        if C.sha(p) != fz['models'][k]['sha256']:
            bad.append(f'{k} hash'); continue
        b = lg.Booster(model_file=str(p)); pr = N.predict(b, E[(s, h)]['X']); wn = A.norm(E[(s, h)]['w'])
        if abs(float(np.sum(wn * (pr - E[(s, h)]['y']) ** 2) / np.sum(wn)) - fz['models'][k]['insample_wmse']) > 1e-9 or b.num_trees() != N.ROUNDS or fz['models'][k]['rows'] != len(E[(s, h)]['y']):
            bad.append(f'{k} reload / rows / trees')
    diff = {k: (A.PARAMS.get(k), fz['params'].get(k)) for k in set(A.PARAMS) | set(fz['params']) if A.PARAMS.get(k) != fz['params'].get(k)}
    R['A'] = dict(models=len(fz['models']), bad=bad, delta=diff)
    return bool(len(fz['models']) == 2 and not bad and diff == {'min_data_in_leaf': (200, 1000)} and fz['rounds'] == 50 and fz['params_sha256'] == N.PARAMS_SHA == A.sha_of(fz['params'])
                and fz['r3_extract_sha256'] == C.sha(N.R3_OUT / 'train' / 'extract.json') and N.committed_equal(N.R3_RECEIPT / 'fit' / 'extract.json', N.R3_OUT / 'train' / 'extract.json'))


def check_B(out, R):
    rel = str(NE.RECEIPT / 'fit' / 'fit.json'); log = git('log', '--format=%H %cI', '--diff-filter=A', '--', rel).stdout.split()
    ev = json.loads((out / 'eval' / 'eval.json').read_text()); recs = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(NR.OUT / 'parts' / 'part_*[0-9].json')))]
    first = log[-2:] if len(log) >= 2 else [None, None]; anc = first[0] is not None and git('merge-base', '--is-ancestor', first[0], 'HEAD').returncode == 0
    fsha = C.sha(out / 'fit.json'); t0 = pd.Timestamp(first[1]) if first[1] else None
    before = t0 is not None and t0 < pd.Timestamp(ev['asof']) and all(t0 < pd.Timestamp(r['asof']) for r in recs)
    R['B'] = dict(commit=first[0], committed_at=first[1], eval_asof=ev['asof'], parts=len(recs))
    return bool(anc and N.committed_equal(rel, out / 'fit.json') and before and recs and ev['fit_sha256'] == fsha and all(r['fit_sha256'] == fsha for r in recs))


def check_C(out, R):
    ev = json.loads((out / 'eval' / 'eval.json').read_text()); D = pd.read_csv(out / 'eval' / 'predictions.csv.gz'); r3 = pd.read_csv(N.REPO / NE.R3_PRED)
    idn = NE.r3_identity(D.assign(day=D.day.astype(str)), r3); acc = json.loads((AE.EVAL_DIR / 'accepted.json').read_text())
    R['C'] = dict(recomputed=idn, stored=ev['identity'])
    return bool(all(v for k, v in idn.items() if k not in ('rows', 'r3_rows')) and acc['dataset_sha256'] == NE.R3_DATASET and ev['r3_dataset_sha256'] == NE.R3_DATASET)


def check_D(out, R):
    ev = out / 'eval'; rd, sc, cal = pd.read_csv(ev / 'readings.csv'), pd.read_csv(ev / 'screens.csv'), pd.read_csv(ev / 'calibration.csv')
    st, pr, ct = pd.read_csv(ev / 'strata.csv'), pd.read_csv(ev / 'predicates.csv'), pd.read_csv(ev / 'control.csv'); D = pd.read_csv(ev / 'predictions.csv.gz'); miss = []
    for (day, s), _ in D.groupby(['day', 'side']):
        for c in NE.CANDS:
            if set(rd[(rd.day == day) & (rd.side == s) & (rd.candidate == c)].group) != {'all', 'POST', 'WAIT', 'fallback'}:
                miss.append(f'{day} {s} {c} groups')
            v = sc[(sc.day == day) & (sc.side == s) & (sc.candidate == c)]
            if set(v.group) != {'POST', 'WAIT'} or not set(v.verdict) <= set(VERDICTS):
                miss.append(f'{day} {s} {c} screens')
            g = D[(D.day == day) & (D.side == s) & (D.support == 'supported')]; w_ = cal[(cal.day == day) & (cal.side == s) & (cal.candidate == c)].w_sum.sum()
            if abs(w_ - g.w.sum()) > 1e-6 * max(1., g.w.sum()):
                miss.append(f'{day} {s} {c} calibration weight')
    rec = pd.DataFrame(NE.predicates(rd.assign(day=rd.day.astype(str)).to_dict('records'), sc.assign(day=sc.day.astype(str)).to_dict('records'),
                                     ct[ct.side.notna()].assign(day=ct.day.astype(str), side=lambda x: x.side.astype(int)).to_dict('records')))
    cols = ['admissions', 'status', 'post_value_positive', 'screen_pass', 'above_draw_mean', 'above_draw_p95']
    a = rec.sort_values(['day', 'side', 'candidate']).reset_index(drop=True)[cols]; b = pr.assign(day=pr.day.astype(str)).sort_values(['day', 'side', 'candidate']).reset_index(drop=True)[cols]
    same = bool(len(a) == len(b) and all((a[c].astype(str).values == b[c].astype(str).values).all() for c in cols))
    R['D'] = dict(missing=miss[:10], predicates_recomputed_equal=same, strata_dims=sorted(set(st.dim)))
    return bool(not miss and same and set(st.dim) == set(NE.STRATA) and len(pr) == 2 * 2 * len(NE.CANDS))


def check_E(out, R):
    ct = pd.read_csv(out / 'eval' / 'control.csv'); ok = []
    for c in NE.CANDS:
        g = ct[ct.candidate == c]; ok.append(len(g[g.side.notna()]) == 4 and len(g[g.side.isna()]) == 2 and np.isfinite(g[['realized', 'lift', 'draw_mean', 'draw_p95']].values).all())
    R['E'] = dict(rows=len(ct), draws=AE.DRAWS, seed=AE.SEED)
    return bool(all(ok) and AE.DRAWS == 500 and AE.SEED == 20261121)


def part_rows(k, out=None):
    out = Path(out) if out else NR.OUT; rec = json.loads((out / 'parts' / f'part_{k:03d}.json').read_text()); b = (out / 'parts' / f'part_{k:03d}.jsonl.gz').read_bytes()
    return rec, b, LS.gz_rows(b)


def check_F(out, R, rout=None):
    plan = json.loads((AE.EVAL_DIR / 'plan.json').read_text()); parts = AR.population(plan); bad, worlds, eps = [], 0, 0
    for k, (day, members) in enumerate(parts):
        rec, b, rows = part_rows(k, rout); r3m, r3w = NR.r3_reference(k)
        if hashlib.sha256(b).hexdigest() != rec['sha256'] or rec['members'] != r3m or [LS.wkey(r) for r in rows] != [LS.wkey(x) for x in members]:
            bad.append(f'part {k} bytes / members'); continue
        for r in rows:
            worlds += 1; ref = r3w.get((int(r['m']), int(r['t_ms'])))
            if ref is None or ref['status'] != r['status']:
                bad.append(f"{r['m']} {r['t_ms']} status vs R3"); continue
            if r['status'] != 'ok':
                continue
            if set(r['episodes']) != set(N.POLICIES):
                bad.append(f"{r['m']} {r['t_ms']} policies"); continue
            for name, e in r['episodes'].items():
                eps += 1; tr = e.get('trace') or {}; cz = tr.get('causal') or {}
                ok = (tr and LS._ledger_bad(tr) is None and LS._arm_calls_bad(tr, -1., 0, 0) is None and cz.get('mismatch', 1) == 0 and cz.get('parity_mismatch', 1) == 0
                      and cz.get('checked') == -(-int(tr['teacher']['calls']) // NR.AUDIT_EVERY) and e.get('cap_ok') is True and e.get('ledger_ok') is True
                      and e.get('value') == tr.get('value') and abs(tr['cash'] + tr['position'] * tr['mark'] - tr['value']) <= 1e-6)
                if name == 'N0':
                    ok = ok and e.get('value') == 0. and e.get('generations') == 0
                if name in NR.CONTROLS:
                    x = (ref.get('episodes') or {}).get(name) or {}
                    ok = ok and x.get('value') == tr.get('value') and x.get('decisions_sha256') == tr.get('decisions_sha256') and x.get('fills_sha256') == tr.get('fills_sha256')
                if not ok:
                    bad.append(f"{r['m']} {r['t_ms']} {name}")
    R['F'] = dict(population=sum(len(m) for _, m in parts), worlds=worlds, episodes=eps, bad=bad[:10], n_bad=len(bad))
    return bool(worlds == sum(len(m) for _, m in parts) and not bad and eps > 0)


def check_G(out, R, rout=None):
    plan = json.loads((AE.EVAL_DIR / 'plan.json').read_text()); parts = AR.population(plan); bad, cnt = [], {}
    fz, smooth, rich, _ = N.frozen()
    for k, (day, members) in enumerate(parts):
        rec, b, rows = part_rows(k, rout)
        for r in rows:
            if r['status'] != 'ok':
                continue
            if sorted(x['side'] for x in r['reached']) != [0, 1]:
                bad.append(f"{r['m']} {r['t_ms']} entries")
            for x in r['reached']:
                st = x['status']; cnt[st] = cnt.get(st, 0) + 1
                if st == 'no_eligible_decision':
                    ok = x.get('eligible') == 0
                elif st == 'pair_not_ok':
                    ok = (x.get('result') or {}).get('status') == 'missing_mark'
                elif st == 'ok':
                    xr = np.asarray([x['fork']['inputs']], np.float32); s = int(x['side'])
                    ok = (not LS._pair_bad(plan, day, r, x) and x.get('inputs_equal_recorded') is True and x.get('eligible', 0) > 0
                          and float(N.predict(smooth[(s, 'post')], xr)[0]) == x['pred'][N.NAME] and float(A.predict(rich['RICH'][(s, 'post')], 'RICH', xr)[0]) == x['pred']['RICH'])
                else:
                    ok = False
                if not ok:
                    bad.append(f"{r['m']} {r['t_ms']} side {x['side']} {st}")
    R['G'] = dict(statuses=cnt, bad=bad[:10], n_bad=len(bad))
    return bool(not bad and cnt.get('ok', 0) > 0 and sum(cnt.values()) <= 768)


def check_H(R):
    import importlib.util
    import pf_mutant_one as O
    import pf_mutant_summary as S
    out = {}
    for tag in ('entry', 'entry_eval', 'entry_replay'):
        h = f'pf_adv2_{tag}_mutants.py'; spec = importlib.util.spec_from_file_location(h[:-3], RES / h); H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
        d = LOGS / f'mutants_{tag}'; files = {'control': d / f'mut_{tag}_control.log', **{i: d / f'mut_{tag}_{i}.log' for i in range(len(H.MUTANTS))}}
        recs = {k: S.last_record(p) for k, p in files.items()}
        one = all(p.exists() and sum(1 for ln in p.read_text().splitlines() if ln.startswith('{') and '"cls"' in ln) == 1 for p in files.values())
        extra = sorted({x.name for x in d.glob(f'mut_{tag}_*.log')} - {p.name for p in files.values()})
        ok, bad = S.judge(len(H.MUTANTS), recs, O.sources(H), [m_[0] for m_ in H.MUTANTS])
        out[tag] = dict(planted=len(H.MUTANTS), caught=sum(1 for i in range(len(H.MUTANTS)) if (recs[i] or {}).get('cls') == 'CAUGHT'), judge_ok=ok, failures=bad[:5],
                        one_record_per_file=one, extra_files=extra)
    R['H'] = out
    return bool(all(v['judge_ok'] and v['caught'] == v['planted'] and v['one_record_per_file'] and not v['extra_files'] for v in out.values()))


def check_I(R):
    peaks, cpu, bad = [], 0., []
    for p in glob.glob(str(LOGS / '**' / '*.log'), recursive=True):
        for ln in Path(p).read_text().splitlines():
            m = RL.match(ln)
            if m:
                peaks.append(int(m.group(4))); cpu += float(m.group(6))
                if int(m.group(4)) >= 2048:
                    bad.append(m.group(2))
    R['I'] = dict(jobs=len(peaks), peak_mib=max(peaks) if peaks else None, core_hours=cpu / 3600., bad=bad)
    return bool(peaks and not bad and cpu / 3600. <= 8.)


def main(out=N.OUT):
    out = Path(out); R = {}; ck = {}
    for k, f in (('A_fit', check_A), ('B_order', check_B), ('C_reuse', check_C), ('D_readings', check_D), ('E_control', check_E), ('F_replay', check_F),
                 ('G_reached', check_G)):
        try:
            ck[k] = f(out, R)
        except Exception as e:                                                            # a check that cannot run is false, with its reason
            ck[k] = False; R[k[0]] = dict(error=repr(e)[:300])
    for k, f in (('H_suites', check_H), ('I_resources', check_I)):
        try:
            ck[k] = f(R)
        except Exception as e:
            ck[k] = False; R[k[0]] = dict(error=repr(e)[:300])
    J = dict(checks=ck, all_true=all(ck.values()), readings=R, source_sha256=C.sha(Path(__file__)), asof=C.stamp())
    (HERE / 'checks.json').write_text(json.dumps(J, indent=1, sort_keys=True, default=str) + '\n')
    print(json.dumps(dict(checks=ck, all_true=J['all_true']), indent=1))


def falsify(out=N.OUT):
    """The falsifier of the header (temporary copies inside the data directory; the real artifacts are never written)."""
    global LOGS
    out = Path(out); res = {}

    def safe(f, *a):
        try:
            return f(*a)
        except Exception:                                                                 # noqa: BLE001  (a corrupted copy that cannot be read is false)
            return False
    with tempfile.TemporaryDirectory(dir=str(out)) as td:
        td = Path(td)
        o = td / 'C'; (o / 'eval').mkdir(parents=True); shutil.copy(out / 'fit.json', o / 'fit.json')
        for q in (out / 'eval').iterdir():
            shutil.copy(q, o / 'eval' / q.name)
        clean = safe(check_C, o, {}); D = pd.read_csv(o / 'eval' / 'predictions.csv.gz'); D.loc[D.index[0], 'Y'] = D.loc[D.index[0], 'Y'] + 1.
        D.to_csv(o / 'eval' / 'predictions.csv.gz', index=False); res['C'] = dict(clean=clean, corrupted=safe(check_C, o, {}))
        for q in (out / 'eval').iterdir():
            shutil.copy(q, o / 'eval' / q.name)
        clean = safe(check_D, o, {}); P = pd.read_csv(o / 'eval' / 'predicates.csv'); P.loc[P.index[0], 'screen_pass'] = not bool(P.loc[P.index[0], 'screen_pass'])
        P.to_csv(o / 'eval' / 'predicates.csv', index=False); res['D'] = dict(clean=clean, corrupted=safe(check_D, o, {}))

        def part_copy(name, fn):
            g = td / name / 'parts'; g.mkdir(parents=True)
            for q in (NR.OUT / 'parts').glob('part_*'):
                shutil.copy(q, g / q.name)
            if fn is not None:
                rec = json.loads((g / 'part_000.json').read_text()); rows = LS.gz_rows((g / 'part_000.jsonl.gz').read_bytes()); fn(rows)
                b = LS.gz_bytes(rows); (g / 'part_000.jsonl.gz').write_bytes(b); rec['sha256'] = hashlib.sha256(b).hexdigest(); (g / 'part_000.json').write_text(json.dumps(rec))
            return td / name

        def first_ok(rows):
            return next(r for r in rows if r['status'] == 'ok')

        def fill_bump(rows):
            tr = next(e['trace'] for e in first_ok(rows)['episodes'].values() if e['trace']['fill_list']); tr['fill_list'][0][3] = float(tr['fill_list'][0][3]) + 1.
        def ctrl_value(rows):
            first_ok(rows)['episodes']['PI0']['trace']['value'] += 1e-6; first_ok(rows)['episodes']['PI0']['value'] += 1e-6
        def arm_fill(rows):
            for r in rows:
                for x in r.get('reached', []):
                    if x['status'] == 'ok':
                        for a_ in x['result']['arms'].values():
                            if a_['fill_list']:
                                a_['fill_list'][0][3] = float(a_['fill_list'][0][3]) + 1.; return
            raise ValueError('no reached arm with a fill to corrupt')
        clean_dir = part_copy('clean', None)
        res['F'] = dict(clean=safe(check_F, out, {}, clean_dir), corrupted_trace_fill=safe(check_F, out, {}, part_copy('Ffill', fill_bump)),
                        corrupted_control_value=safe(check_F, out, {}, part_copy('Fval', ctrl_value)))
        res['G'] = dict(clean=safe(check_G, out, {}, clean_dir), corrupted=safe(check_G, out, {}, part_copy('Garm', arm_fill)))
        h = td / 'H'
        for tag in ('entry', 'entry_eval', 'entry_replay'):
            shutil.copytree(LOGS / f'mutants_{tag}', h / f'mutants_{tag}')
        saved = LOGS
        try:
            LOGS = h; clean = safe(check_H, {}); shutil.copy(h / 'mutants_entry_eval' / 'mut_entry_eval_0.log', h / 'mutants_entry_eval' / 'mut_entry_eval_1.log')
            res['H'] = dict(clean=clean, corrupted=safe(check_H, {}))
        finally:
            LOGS = saved
    ok = all(v['clean'] is True and all(x is False for k, x in v.items() if k != 'clean') for v in res.values())
    (HERE / 'checks_falsifier.json').write_text(json.dumps(dict(cases=res, falsifier_ok=ok, asof=C.stamp()), indent=1, sort_keys=True) + '\n')
    print(json.dumps(dict(cases=res, falsifier_ok=ok)))


if __name__ == '__main__':
    main(); falsify()
