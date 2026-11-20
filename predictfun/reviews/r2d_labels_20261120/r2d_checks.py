"""The R2-D receipt's computed checks, declared 2026-11-20 before the bounded build's outputs were read (review 8cc1dc49's next priority).
CONTRACT PREDICATES (must all hold; each computed):
 (A) the manifest:
     - builds A and B have byte-identical manifests and candidate files; the candidates load against the manifest's sha256 and number its
       candidate_rows; the manifest's freeze is pi0_50_v1's;
     - each day has its 32 prefix worlds, each with a status;
     - at most 16 selected per day / side / kind and 128 in all; every selected candidate ok; every selected pi0 fork equals its census
       rollout state;
     - no candidate row carries an outcome field.
 (B) persisted / resumed identity: every selected pair has a valid record in A and in B; B's records equal A's except the timing fields;
     B's first pairs pass stopped (STOPPED) and its second completed (PAIRS_DONE).
 (C) the labels:
     - every ok pair reconciled (both arms' ledgers, Y = R_a - R_b, the pre-fork history);
     - every arm teacher's binding checked > 0 with 0 failures;
     - the arm causal probes: checked > 0 in total, with 0 mismatches and 0 parity mismatches;
     - every non-ok pair a counted status.
 (D) resources: every pair <= 120 s and <= 1,536 MiB high-water, in A and B; no undisposed stop in A or B; every R2-D job's run_light peak
     <= 1,536 MiB.
 (E) the real-row controls (controls.json) all hold;
 (F) the adapter's mutation suite (both shards: every mutant of the harness caught by a check) passed on EXACTLY the adapter sources the
     builds recorded (sources_at_build.sha256), which is the condition of the speculative overlap. v2: the number of mutants is parsed
     from the harness. v1 compared with a stale literal 26, written before the candidates file added 3 mutants (29 in all); its output
     is kept as checks_v1_stale_count.json.
DECLARED READINGS (reported; never gates, never tuned on):
 - per day / side / kind: the prefix's candidates by source and status; the selected by source; the labeled pairs by status;
 - the arm teachers' calls by status and action, per arm name (POST / WAIT / KEEP / PULL), and their fallback share;
 - Y (cents) per day / side / kind: n, mean, median, min, max. This is an integration readout of a bounded, source-balanced sample, not a
   skill or value claim;
 - seconds per pair (mean / max) and two projections: labeling every ok candidate of the 32-world prefix, and of a full day (the prefix's
   ok candidates per world x the day's worlds).
Usage: r2d_checks.py <A> <B>; writes checks.json and the CSV copies beside this script."""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
import pf_r2_labels as L                                  # noqa: E402

HERE = Path(__file__).resolve().parent; A, B = Path(sys.argv[1]), Path(sys.argv[2]); C, pd = L.C, L.pd; res = {}; read = {}
man = json.loads((A / 'manifest.json').read_text()); msha = C.sha(A / 'manifest.json'); sel = man['selected']; cand = pd.DataFrame(L.load_candidates(A))
W = pd.DataFrame(man['worlds']); per = pd.DataFrame(sel)
cnt = per.groupby(['day', 'side', 'kind']).size() if len(per) else pd.Series(dtype=int)
res['A_manifest'] = bool((A / 'manifest.json').read_bytes() == (B / 'manifest.json').read_bytes() and man['freeze_sha256'] == L.FREEZE_SHA
                         and (A / man['candidates_file']).read_bytes() == (B / man['candidates_file']).read_bytes() and len(cand) == man['candidate_rows']
                         and all((W.day == d).sum() == man['prefix'] for d in man['days']) and W.status.notna().all()
                         and len(sel) <= 128 and (cnt <= man['cap']).all() and all(r['status'] == 'ok' for r in sel)
                         and all(r['fork_equals_census'] is True for r in sel if r['source'] == 'pi0') and not (set(cand.columns) & set(L.OUTCOMES)))
strip = lambda p: {k: v for k, v in json.loads(p.read_text()).items() if k not in ('asof', 'elapsed_s', 'hwm_mib', 'breach')}
recA = {L.rname(r): json.loads((A / 'pairs' / L.rname(r)).read_text()) for r in sel if L.valid_record(A / 'pairs', r, msha)}
recB_ok = all(L.valid_record(B / 'pairs', r, msha) for r in sel)
same = recB_ok and all(strip(A / 'pairs' / L.rname(r)) == strip(B / 'pairs' / L.rname(r)) for r in sel)
lg = lambda n: (HERE / 'logs' / n).read_text() if (HERE / 'logs' / n).exists() else ''
res['B_resume_identity'] = bool(len(recA) == len(sel) and same and re.search(r'^STOPPED$', lg('r2d_B_pairs1.log'), re.M) and 'PAIRS_DONE' not in lg('r2d_B_pairs1.log')
                                and re.search(r'^PAIRS_DONE$', lg('r2d_B_pairs2.log'), re.M))
ok_recs = [r_ for r_ in recA.values() if r_['result'].get('status') == 'ok']
arm_t = [a['teacher'] for r_ in ok_recs for a in r_['result']['arms'].values()]
czs = [c for r_ in ok_recs for c in r_['causal']]
res['C_labels'] = bool(ok_recs and all(r_['result']['reconciled'] and r_['result']['common_prefork_identical'] for r_ in ok_recs)
                       and all(t['binding_checked'] > 0 and t['binding_bad'] == 0 for t in arm_t) and sum(c['checked'] for c in czs) > 0
                       and all(c['mismatch'] == 0 and c['parity_mismatch'] == 0 for c in czs))
recs_all = list(recA.values()) + [json.loads((B / 'pairs' / L.rname(r)).read_text()) for r in sel if (B / 'pairs' / L.rname(r)).exists()]
und = [L.held_stops(d_, msha)['undisposed'] for d_ in (A, B)]
peaks = {p.name: [int(x) for x in re.findall(r'peak_rss_mib=(\d+)', p.read_text())] for p in sorted((HERE / 'logs').glob('r2d_*.log'))}
res['D_resources'] = bool(all(r_['elapsed_s'] <= L.PAIR_SECONDS and r_['hwm_mib'] <= L.PEAK_MIB for r_ in recs_all) and und == [[], []]
                          and peaks and all(v and max(v) <= 1536 for v in peaks.values()))
ctl = json.loads((HERE / 'controls.json').read_text()) if (HERE / 'controls.json').exists() else {}
res['E_controls'] = bool(ctl.get('all_true') is True and ctl.get('manifest_sha256') == msha)
SC = Path('/tmp/claude-1001/-home-yuqing-ctaNew/aea5bfb9-4ea4-41d0-bf73-bb98d818e3d4/scratchpad/teacher_logs')
built = {Path(l.split('  ')[1]).name: l.split('  ')[0][:16] for l in (HERE / 'sources_at_build.sha256').read_text().splitlines() if l.strip()}
suite = dict(zip(('pf_r2_labels.py', 'pf_r2_labels_selftest.py', 'pf_r2_labels_mutants.py'), (SC / 'labels_sources_at_mutants.txt').read_text().split()))
logs_m = [(SC / f'labels_mutants_{k}.log').read_text() if (SC / f'labels_mutants_{k}.log').exists() else '' for k in (0, 1)]
n_caught = sum(int(m.group(1)) for x in logs_m for m in [re.search(r'passed: the unchanged copy passes its selftest, (\d+) / \d+ planted', x)] if m)
import ast                                                                                    # noqa: E402
hsrc = (Path(L.__file__).resolve().parent / 'pf_r2_labels_mutants.py').read_text(); n_mut = len(ast.literal_eval(hsrc[hsrc.index('MUTANTS = [') + 10:hsrc.index(']\nRUN = ') + 1]))
res['F_suite_on_build_sources'] = bool(all(f'(shard {k}/2) passed' in logs_m[k] for k in (0, 1)) and n_caught == n_mut > 0 and all(built.get(n) == h for n, h in suite.items()))
read['mutants'] = dict(harness=n_mut, caught=n_caught)
# ---- readings
read['candidates'] = {f'{d} {s} {k} {src} {st}': int(n) for (d, s, k, src, st), n in cand.groupby(['day', 'side', 'kind', 'source', 'status']).size().items()}
read['selected'] = {f'{d} {s} {k} {src}': int(n) for (d, s, k, src), n in per.groupby(['day', 'side', 'kind', 'source']).size().items()} if len(per) else {}
read['pair_status'] = pd.Series([r_['result'].get('status') for r_ in recA.values()]).value_counts().to_dict()
arms = []
for r_ in ok_recs:
    for nm, a in r_['result']['arms'].items():
        t = a['teacher']; fb = sum(v for k_, v in t['by_status'].items() if k_ not in ('model', 'cap_blocked', 'near_end'))
        arms.append(dict(day=r_['candidate']['day'], side=r_['candidate']['side'], kind=r_['candidate']['kind'], source=r_['candidate']['source'], arm=nm,
                         calls=t['calls'], fallback=fb, **{f's_{k_}': v for k_, v in t['by_status'].items()}, **{f'a_{k_}': v for k_, v in t['by_action'].items()}))
Ar = pd.DataFrame(arms).fillna(0); Ar.to_csv(HERE / 'arms.csv', index=False)
read['arm_fallback_share'] = {f'{a}': float(g.fallback.sum() / max(g.calls.sum(), 1)) for a, g in Ar.groupby('arm')} if len(Ar) else {}
Y = pd.DataFrame([dict(day=r_['candidate']['day'], side=r_['candidate']['side'], kind=r_['candidate']['kind'], source=r_['candidate']['source'], Y=r_['result']['Y'],
                       elapsed_s=r_['elapsed_s'], hwm_mib=r_['hwm_mib']) for r_ in ok_recs])
Y.to_csv(HERE / 'labels_summary.csv', index=False)
read['Y'] = {f'{d} {s} {k}': dict(n=int(len(g)), mean=float(g.Y.mean()), median=float(g.Y.median()), min=float(g.Y.min()), max=float(g.Y.max()))
             for (d, s, k), g in Y.groupby(['day', 'side', 'kind'])} if len(Y) else {}
el = [r_['elapsed_s'] for r_ in recA.values()]; okc = cand[(cand.status == 'ok') & cand.kind.isin(L.KINDS)]
nw = {d: int(len(L.day_orders(d, Path(json.loads(Path(L.T.REAL['population']).read_text())['inputs']['kp_dir'])).groupby(['m', 't']))) for d in man['days']}
read['seconds_per_pair'] = dict(mean=float(np.mean(el)), max=float(np.max(el)), n=len(el))
read['projection_h'] = {d: dict(prefix_ok_candidates=int((okc.day == d).sum()), prefix_hours=float((okc.day == d).sum() * np.mean(el) / 3600.),
                                day_worlds=nw[d], full_day_hours=float((okc.day == d).sum() / man['prefix'] * nw[d] * np.mean(el) / 3600.)) for d in man['days']}
read['peaks_mib'] = {k: max(v) for k, v in peaks.items() if v}
out = dict(asof=C.stamp(), manifest_sha256=msha, predicates=res, all_true=all(res.values()), readings=read)
(HERE / 'checks.json').write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(dict(predicates=res, all_true=all(res.values())))); print(json.dumps(read, default=str)[:4000]); sys.exit(0 if all(res.values()) else 1)
