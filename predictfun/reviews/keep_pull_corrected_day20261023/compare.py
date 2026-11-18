"""Declared BEFORE the corrected one-day outputs were read (2026-11-18, session aea5bfb9): reproduction predicates for the corrected run
(known v5, ledger v5, scenarios v4 on consumed Oct 23) against the earlier artifacts. Expected: (a) known v5 (the module's own pipeline from
Nov 8's tables) reproduces the readings recomputed from the saved v4 tables; (b) the ledger's non-ST results equal v4's exactly (the S1 / S2 /
S3 / EV code paths did not change); (c) every scenario except the ST timeline equals v3's exactly, and STlo is absent. Differences are
listed, never absorbed. Run via run_light (PF_SLOT=claude): python3 compare.py <out_dir>"""
import json
import sys
from pathlib import Path

R = Path('/home/yuqing/ctaNew/data/predict_fun/research'); REV = Path('/home/yuqing/ctaNew/predictfun/reviews')
out = Path(sys.argv[1]); L = []; P = {}
ld = lambda p: json.loads(Path(p).read_text())
# (a) known v5 vs the preview recomputed from the saved v4 tables
k5 = ld(R / 'keep_pull_exec_windows_known_v5_20261118/known.json'); pv = ld(REV / 'settlement_lag_fix_check_20261118/known_readings_corrected.json')
dk = {k: (pv['readings'].get(k), k5['readings'].get(k)) for k in set(pv['readings']) | set(k5['readings']) if pv['readings'].get(k) != k5['readings'].get(k)}
P['a_known_v5_reproduces_preview'] = not dk; L.append(f"(a) known v5 readings vs the saved-table preview: {'IDENTICAL' if not dk else 'DIFFER ' + json.dumps(dk)}")
L.append(f"    known v5 KNOWN VERDICT: {k5['readings']['KNOWN VERDICT']}")
# (b) ledger v5 vs v4 on Oct 23: non-ST parts must be identical
l4 = ld(R / 'keep_pull_exec_windows_v4_20261118_20261023/exec_windows.json'); l5 = ld(R / 'keep_pull_exec_windows_v5_20261118_20261023/exec_windows.json')
r4, r5 = l4['R'], l5['R']; same, diff = [], []
for part in ('V3', 'bounds', 'xtab', 'decisions', 'coverage', 'disagreement'):
    for k in sorted(set(r4.get(part, {})) | set(r5.get(part, {}))):
        st = k.startswith('ST')
        (diff if r4[part].get(k) != r5[part].get(k) else same).append((part, k, st))
nonst_diff = [(p, k) for p, k, st in diff if not st]
P['b_ledger_non_st_identical'] = not nonst_diff
for k in ('fills', 'eligible', 'counted', 'events', 'legs', 'collapsed_E', 'score', 'sim_removed', 'sim_kept'):
    if r4.get(k) != r5.get(k):
        nonst_diff.append(('R', k)); P['b_ledger_non_st_identical'] = False
P['b_v0_controls'] = l5['readings'].get('V0 CONTROLS')
L.append(f"(b) ledger v5 vs v4: {len(same)} identical parts; non-ST differences {nonst_diff or 'NONE'}; ST parts that changed {sorted(set(k for p, k, st in diff if st))}")
L.append(f"    v5 st_lag_ms {r5.get('st_lag_ms')} role {r5.get('st_lag_role')}; V0 CONTROLS {P['b_v0_controls']}")
# (c) scenarios v4 vs v3 on Oct 23
s3 = ld(R / 'keep_pull_scenarios_v3_20261118/scenarios_20261023.json')['scenarios']; s4 = ld(R / 'keep_pull_scenarios_v4_20261118/scenarios_20261023.json')['scenarios']
F = ('KEEP', 'score', 'null', 'E', 'PULLALL', 'fills', 'shares', 'eligible', 'held', 'orders_filled', 'pulls', 'discovered', 'lost')
chg = {n: {f: (s3[n].get(f), s4[n].get(f)) for f in F if s3[n].get(f) != s4[n].get(f)} for n in s4 if n in s3}
chg = {n: v for n, v in chg.items() if v}
P['c_stlo_absent'] = not any(n.startswith('STlo') for n in s4); P['c_scenarios_v4'] = len(s4)
P['c_only_st_timeline_changed'] = all(n.startswith('ST') for n in chg); P['c_new_names'] = sorted(set(s4) - set(s3)); P['c_dropped_names'] = sorted(set(s3) - set(s4))
L.append(f"(c) scenarios v4 ({len(s4)}) vs v3 ({len(s3)}): dropped {P['c_dropped_names']}, new {P['c_new_names']}; changed {json.dumps(chg) if chg else 'NONE'}")
L.append('PREDICATES: ' + json.dumps(P))
out.mkdir(parents=True, exist_ok=True); (out / 'compare.txt').write_text('\n'.join(L) + '\n'); (out / 'compare.json').write_text(json.dumps(dict(predicates=P, known_diff=dk, ledger_non_st_diff=nonst_diff, scenario_changes=chg), indent=1, default=str))
print('\n'.join(L))
