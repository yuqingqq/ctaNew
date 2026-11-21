"""R3 ADV2-RICH v1: THE ROUND'S DECLARED CHECKS (review 315979d2; 2026-11-21, session aea5bfb9). Declared in freeze 2, before any fit and
before any Oct 26-27 label is read. Every check is a predicate on the artifacts; the receipt copies its outputs by code (r3_tables.py).
 A extraction: train/extract.json names the accepted R2-F dataset and days; every head's X / y / w bytes and keys recompute to its hashes;
   the rows are every ok row of the dataset less the statused exclusions (the rows plus the excluded ok rows = the dataset's ok count).
 B fit: fit.json binds the extraction, the support, the resolved parameters (hash), 200 rounds and the three input lists; 12 model files,
   each file's sha256 as recorded, each reloaded predicting bit-equal to the stored predictions on its training rows (recomputed here).
 C order: the receipt copy fit/fit.json is in a commit that is an ancestor of HEAD and was committed before the evaluation's asof; the
   evaluation and the replay records name that fit.json's hash.
 D sample: the evaluation sample's accepted.json names the frozen plan 8619eec5; eval.json names its dataset hash; the statuses are counted.
 E readings: for every day x side x head x candidate the groups all / both actions / fallback are in readings.csv; every action group has a
   screen verdict in {PASS, FAIL, NOT_EVALUABLE, INSUFFICIENT_SUPPORT} (no dropped group); calibration rows exist for every candidate;
   coverage: every dimension of every evaluation block sums to the block's rows (group all), and of the training coverage to the
   extraction's rows per side / head.
 F control: per candidate the (day, head) and (day) scopes are present, finite, with 500 draws declared in the module.
 G replay: every population world has a status; every ok world has the 8 policies; every episode's ledger reconciles with zero binding
   failures and zero causal / parity mismatches, cap_ok, one churn record per reviewed generation; RICH carries its exposure counts; N0 is
   exactly 0; every ok world has exactly one reached entry per side / head, each ok / missing_mark / no_eligible_decision (no
   fork_mismatch), reconciled and bound when ok, with a positive inclusion count unless no_eligible_decision.
 H suites (review 315979d2 item 3, run here, not read from summary text): for each of the four harnesses pf_mutant_summary.judge on every
   record file against the harness's own names and pf_mutant_one.sources of the files on disk now (control PASSED, every index CAUGHT
   under its own index and name, one source set equal to the final sources); exactly one record line per file; no other record file.
 I resources: every job of the round (every run_light line under logs/, the mutant jobs included) peaked below 2,048 MiB; the round's
   CPU is within the 40 core-hour ceiling.
FALSIFIER (rule 15; falsify(), run after the checks): E, G and H on copies of their real inputs must be true, and false after one
 corruption each - a coverage row dropped; one episode's cap_ok false (its part rewritten with a matching record hash); one mutant record
 copied over another index's file. Written to checks_falsifier.json.
Usage (run_light): r3_checks.py [out]  (the checks, then the falsifier)"""
import glob
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; RES = Path('/home/yuqing/ctaNew/predictfun/research'); sys.path.insert(0, str(RES))
import pf_adv2 as A                                      # noqa: E402
import pf_adv2_eval as AE                                # noqa: E402
import pf_adv2_replay as AR                              # noqa: E402

C, pd, T = A.C, A.pd, A.T
LOGS = HERE / 'logs'; REPO = RES.parent.parent
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
VERDICTS = ('PASS', 'FAIL', 'NOT_EVALUABLE', 'INSUFFICIENT_SUPPORT')


def check_A(out, R):
    ex = json.loads((out / 'train' / 'extract.json').read_text()); E = A.load_train(out); m = A.manifest_of(E, ex['statuses'])
    acc = json.loads((A.TRAIN['dir'] / 'accepted.json').read_text()); ok_all = sum(v for k, v in acc['statuses'].items() if k.endswith(' ok'))
    excl = sum(v for k, v in ex['statuses'].items() if k.endswith(' crosses_day_boundary'))
    rows = sum(v['rows'] for v in ex['heads'].values()); R['A'] = dict(rows=rows, dataset_ok=ok_all, crossing=excl)
    return bool(ex['dataset_sha256'] == A.TRAIN['dataset_sha256'] == acc['dataset_sha256'] and ex['days'] == list(A.TRAIN['days']) and m['heads'] == ex['heads']
                and rows + excl == ok_all)


def check_B(out, R):
    fz = json.loads((out / 'fit.json').read_text()); E = A.load_train(out); lg = A.lgb(); bad = []
    for cand in A.CANDS:
        for (s, h) in A.HEADS:
            k = f'{cand} {s} {h}'; p = A.model_path(out, cand, s, h)
            if C.sha(p) != fz['models'][k]['sha256']:
                bad.append(f'{k} hash'); continue
            b = lg.Booster(model_file=str(p)); pr = A.predict(b, cand, E[(s, h)]['X']); wn = A.norm(E[(s, h)]['w'])
            if abs(float(np.sum(wn * (pr - E[(s, h)]['y']) ** 2) / np.sum(wn)) - fz['models'][k]['insample_wmse']) > 1e-9 or b.num_trees() > A.ROUNDS:
                bad.append(f'{k} reload')
    R['B'] = dict(models=len(fz['models']), bad=bad)
    return bool(len(fz['models']) == 12 and not bad and fz['params_sha256'] == A.PARAMS_SHA and fz['rounds'] == 200 and fz['extract_sha256'] == C.sha(out / 'train' / 'extract.json')
                and fz['support_sha256'] == C.sha(out / 'train' / 'support.json') and all(fz['inputs'][c]['sha256'] == A.INPUT_SHA[c] for c in A.CANDS))


def git(*a):
    return subprocess.run(['git', '-C', str(REPO), *a], capture_output=True, text=True)


def check_C(out, R):
    rel = str(AE.RECEIPT / 'fit' / 'fit.json'); log = git('log', '--format=%H %cI', '--diff-filter=A', '--', rel).stdout.split()
    ev = json.loads((out / 'eval' / 'eval.json').read_text()); recs = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(AR.OUT / 'parts' / 'part_*[0-9].json')))]
    first = log[-2:] if len(log) >= 2 else [None, None]; anc = first[0] is not None and git('merge-base', '--is-ancestor', first[0], 'HEAD').returncode == 0
    same = AE.committed(rel, out / 'fit.json'); fsha = C.sha(out / 'fit.json')
    before = first[1] is not None and pd.Timestamp(first[1]) < pd.Timestamp(ev['asof'])
    R['C'] = dict(commit=first[0], committed_at=first[1], eval_asof=ev['asof'], replay_parts=len(recs))
    return bool(anc and same and before and ev['fit_sha256'] == fsha and all(r['fit_sha256'] == fsha for r in recs) and recs)


def check_D(out, R):
    acc = json.loads((AE.EVAL_DIR / 'accepted.json').read_text()); ev = json.loads((out / 'eval' / 'eval.json').read_text())
    R['D'] = dict(statuses=acc['statuses'], pairs=acc['pairs'], dataset_sha256=acc['dataset_sha256'])
    return bool(acc['accepted'] is True and acc['plan_sha256'] == AE.EVAL_PLAN_SHA and ev['eval_dataset_sha256'] == acc['dataset_sha256'] and ev['statuses'])


def check_E(out, R):
    rd = pd.read_csv(out / 'eval' / 'readings.csv'); sc = pd.read_csv(out / 'eval' / 'screens.csv'); cal = pd.read_csv(out / 'eval' / 'calibration.csv')
    miss = []
    for day in json.loads((AE.EVAL_DIR / 'plan.json').read_text())['days']:
        for s in (0, 1):
            for h in ('post', 'keep'):
                for c in A.CANDS:
                    have = set(rd[(rd.day.astype(str) == day) & (rd.side == s) & (rd['head'] == h) & (rd.candidate == c)].group)
                    need = {'all', 'fallback'} | set(AE.GROUPS[h])
                    miss += [f'{day} {s} {h} {c} {g}' for g in need - have]
                    sv = sc[(sc.day.astype(str) == day) & (sc.side == s) & (sc['head'] == h) & (sc.candidate == c)]
                    if set(sv.group) != set(AE.GROUPS[h]) or not set(sv.verdict) <= set(VERDICTS):
                        miss.append(f'{day} {s} {h} {c} screens')
    cov, cvt = pd.read_csv(out / 'eval' / 'coverage.csv'), pd.read_csv(out / 'eval' / 'coverage_train.csv')
    rr = rd[(rd.group == 'all') & (rd.candidate == 'RICH')].set_index(['day', 'side', 'head']).rows
    cs = cov[cov.group == 'all'].groupby(['day', 'side', 'head', 'dim']).rows.sum(); cbad = [f'{k} {dm}' for k, n in rr.items() for dm in AE.COV_DIMS if int(cs.get((*k, dm), -1)) != int(n)]
    ex = json.loads((out / 'train' / 'extract.json').read_text()); ts = cvt.groupby(['side', 'head', 'dim']).rows.sum()
    cbad += [f'train {k} {dm}' for k, v in ex['heads'].items() for dm in AE.COV_DIMS if int(ts.get((int(k.split()[0]), k.split()[1], dm), -1)) != int(v['rows'])]
    R['E'] = dict(missing=miss[:10], screens=sc.verdict.value_counts().to_dict(), coverage_bad=cbad[:10])
    return bool(not miss and set(cal.candidate) == set(A.CANDS) and not cbad and len(rr) > 0)


def check_F(out, R):
    ct = pd.read_csv(out / 'eval' / 'control.csv'); ok = []
    for c in A.CANDS:
        g = ct[ct.candidate == c]; ok.append(len(g[g['head'].notna()]) == 4 and len(g[g['head'].isna()]) == 2 and np.isfinite(g[['realized', 'lift', 'draw_mean']].values).all())
    R['F'] = dict(rows=len(ct), draws=AE.DRAWS, seed=AE.SEED)
    return bool(all(ok) and AE.DRAWS == 500 and AE.SEED == 20261121)


def check_G(out, R):
    plan = json.loads((AE.EVAL_DIR / 'plan.json').read_text()); parts = AR.population(plan); n_pop = sum(len(ws) for _, ws in parts); worlds, bad = 0, []
    reached = dict(ok=0, missing_mark=0, no_eligible_decision=0, other=0)
    for k in range(len(parts)):
        rec = json.loads((AR.OUT / 'parts' / f'part_{k:03d}.json').read_text()); b = (AR.OUT / 'parts' / f'part_{k:03d}.jsonl.gz').read_bytes()
        if hashlib.sha256(b).hexdigest() != rec['sha256']:
            bad.append(f'part {k} bytes'); continue
        for r in A.LS.gz_rows(b):
            worlds += 1
            if r['status'] != 'ok':
                continue
            if set(r['episodes']) != set(A.POLICIES):
                bad.append(f"{r['m']} {r['t_ms']} policies")
            for name, e in r['episodes'].items():
                cz = e.get('causal') or {}
                if not (e.get('ledger_ok') is True and e.get('binding_bad') == 0 and cz.get('mismatch', 0) == 0 and cz.get('parity_mismatch', 0) == 0
                        and e.get('cap_ok') is True and len(e.get('churn', [None])) == e.get('reviewed_generations') and (name != 'RICH' or isinstance(e.get('exposure'), dict))):
                    bad.append(f"{r['m']} {r['t_ms']} {name}")
                if name == 'N0' and (e.get('value') != 0. or e.get('generations') != 0):
                    bad.append(f"{r['m']} {r['t_ms']} N0")
            if sorted((x['key']['side'], x['key']['head']) for x in r['reached']) != sorted((s, h) for s, h in A.HEADS):
                bad.append(f"{r['m']} {r['t_ms']} reached entries")
            for x in r['reached']:
                st = x.get('status'); reached[st if st in reached else 'other'] += 1
                if st == 'ok' and not (x.get('reconciled') is True and x.get('binding_bad') == 0 and x.get('binding_checked', 0) > 0):
                    bad.append(f"{r['m']} {r['t_ms']} reached pair")
                if (x.get('eligible', 0) > 0) != (st != 'no_eligible_decision'):
                    bad.append(f"{r['m']} {r['t_ms']} reached inclusion count")
    R['G'] = dict(population=n_pop, worlds=worlds, bad=bad[:10], reached=reached)
    return bool(worlds == n_pop and not bad and reached['other'] == 0)


def check_H(R):
    import importlib.util
    import pf_mutant_one as O
    import pf_mutant_summary as S
    out = {}
    for tag, h in (('sample', 'pf_adv2_sample_mutants.py'), ('adv2', 'pf_adv2_mutants.py'), ('eval', 'pf_adv2_eval_mutants.py'), ('replay', 'pf_adv2_replay_mutants.py')):
        spec = importlib.util.spec_from_file_location(h[:-3], RES / h); H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
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
    for p in glob.glob(str(LOGS / '**' / '*.log'), recursive=True):                   # the mutant jobs' logs included
        for ln in Path(p).read_text().splitlines():
            m = RL.match(ln)
            if m:
                peaks.append(int(m.group(4))); cpu += float(m.group(6))
                if int(m.group(4)) >= 2048:
                    bad.append(m.group(2))
    R['I'] = dict(jobs=len(peaks), peak_mib=max(peaks) if peaks else None, core_hours=cpu / 3600., bad=bad)
    return bool(peaks and not bad and cpu / 3600. <= 40.)


def main(out=A.OUT):
    out = Path(out); R = {}; ck = {}
    for k, f in (('A_extraction', check_A), ('B_fit', check_B), ('C_order', check_C), ('D_sample', check_D), ('E_readings', check_E), ('F_control', check_F),
                 ('G_replay', check_G)):
        try:
            ck[k] = f(out, R)
        except Exception as e:                                                            # a check that cannot run is false, with its reason
            ck[k] = False; R[k[0]] = dict(error=repr(e)[:300])
    for k, f in (('H_suites', check_H), ('I_resources', check_I)):
        try:
            ck[k] = f(R)
        except Exception as e:
            ck[k] = False; R[k[0]] = dict(error=repr(e)[:300])
    J = dict(checks=ck, all_true=all(ck.values()), readings=R, asof=C.stamp())
    (HERE / 'checks.json').write_text(json.dumps(J, indent=1, sort_keys=True, default=str) + '\n')
    print(json.dumps(dict(checks=ck, all_true=J['all_true']), indent=1))


def falsify(out=A.OUT):
    """The falsifier of the header (temporary copies inside the data directory; the real artifacts are never written)."""
    global LOGS
    import shutil
    import tempfile
    out = Path(out); res = {}
    with tempfile.TemporaryDirectory(dir=str(out)) as td:
        td = Path(td); o = td / 'E'; (o / 'eval').mkdir(parents=True); (o / 'train').mkdir()
        for n in ('readings.csv', 'screens.csv', 'calibration.csv', 'coverage.csv', 'coverage_train.csv'):
            shutil.copy(out / 'eval' / n, o / 'eval' / n)
        shutil.copy(out / 'train' / 'extract.json', o / 'train' / 'extract.json'); clean = check_E(o, {})
        cv = pd.read_csv(o / 'eval' / 'coverage.csv'); cv.drop(index=cv.index[0]).to_csv(o / 'eval' / 'coverage.csv', index=False)
        res['E'] = dict(clean=clean, corrupted=check_E(o, {}))
        g = td / 'G' / 'parts'; g.mkdir(parents=True)
        for q in (AR.OUT / 'parts').glob('part_*'):
            shutil.copy(q, g / q.name)
        saved = AR.OUT
        try:
            AR.OUT = td / 'G'; clean = check_G(out, {})
            rec = json.loads((g / 'part_000.json').read_text()); rows = A.LS.gz_rows((g / 'part_000.jsonl.gz').read_bytes())
            next(r for r in rows if r['status'] == 'ok')['episodes']['RICH']['cap_ok'] = False
            b = A.LS.gz_bytes(rows); (g / 'part_000.jsonl.gz').write_bytes(b); rec['sha256'] = hashlib.sha256(b).hexdigest(); (g / 'part_000.json').write_text(json.dumps(rec))
            res['G'] = dict(clean=clean, corrupted=check_G(out, {}))
        finally:
            AR.OUT = saved
        h = td / 'H'
        for tag in ('sample', 'adv2', 'eval', 'replay'):
            shutil.copytree(LOGS / f'mutants_{tag}', h / f'mutants_{tag}')
        saved = LOGS
        try:
            LOGS = h; clean = check_H({}); shutil.copy(h / 'mutants_eval' / 'mut_eval_0.log', h / 'mutants_eval' / 'mut_eval_1.log')
            res['H'] = dict(clean=clean, corrupted=check_H({}))
        finally:
            LOGS = saved
    ok = all(v['clean'] is True and v['corrupted'] is False for v in res.values())
    (HERE / 'checks_falsifier.json').write_text(json.dumps(dict(cases=res, falsifier_ok=ok, asof=C.stamp()), indent=1, sort_keys=True) + '\n')
    print(json.dumps(dict(cases=res, falsifier_ok=ok)))


if __name__ == '__main__':
    main(*sys.argv[1:]); falsify(*sys.argv[1:])
