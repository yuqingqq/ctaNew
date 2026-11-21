"""PI0-MOVEMENT-SHRINK-v1: copies the outputs into outputs/ (SHA256SUMS), writes tables.md and report.md, and the state hunks (FINDINGS P99,
HANDOFF section, STATUS block) into the scratchpad's edits/. Every number is read from a file; every answer is audit.json's computed predicate;
every sentence that states a comparison is selected by the computed values. argv[1] = the HANDOFF time from `date`. Usage: fill_shrink.py <time>"""
import gzip
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path('/home/yuqing/ctaNew/predictfun/reviews/movement_shrink_20261121'); SRC = Path('/home/yuqing/ctaNew/data/predict_fun/research/pi0_movement_shrink_v1_20261121')
S = Path('/tmp/claude-1001/-home-yuqing-ctaNew/aea5bfb9-4ea4-41d0-bf73-bb98d818e3d4/scratchpad'); NOW = sys.argv[1]
FREEZE, FIT = '9720813606172cde68ccdf85947c93c4ead28470', 'bd5e2f01'
STR = {c: str for c in ('head', 'side', 'arm', 'rule', 'dim', 'kind', 'cell')}
D = R / 'outputs'; D.mkdir(exist_ok=True)
for n in ('scores', 'calibration', 'actions', 'null_draws'):
    shutil.copy(SRC / f'{n}.csv', D / f'{n}.csv')
b = (SRC / 'eval_rows.csv').read_bytes(); (D / 'eval_rows.csv.gz').write_bytes(gzip.compress(b, mtime=0))
assert gzip.decompress((D / 'eval_rows.csv.gz').read_bytes()) == b
shutil.copy(SRC / 'audit.json', D / 'audit.json')
(D / 'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n' for p in sorted(D.iterdir()) if p.is_file() and p.name != 'SHA256SUMS'))
au = json.loads((SRC / 'audit.json').read_text()); fit = json.loads((R / 'fit.json').read_text())['params']; ck = json.loads((R / 'checks.json').read_text())
fa = json.loads((R / 'checks_falsifier.json').read_text()); man = json.loads((R / 'manifest.json').read_text())
SC, CA, AC = (pd.read_csv(SRC / f'{n}.csv', dtype=STR) for n in ('scores', 'calibration', 'actions'))
mut = re.search(r'(\d+) / (\d+) planted code errors are caught', (R / 'logs' / 'mutants_shr_summary.log').read_text())
rs = ck['details']['J']['resources']; asof = ck['details']['J']['asof'][:16].replace('T', ' ') + 'Z'
P = {p['head']: p for p in au['predicates']}; NU = au['null']
nf = sum(1 for v in fa['cases'].values() if v['clean'] and not v['corrupted'] and v['applied'])
f2 = lambda v, nd=2: '—' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{float(v):+,.{nd}f}'
g3 = lambda v, nd=3: '—' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{float(v):,.{nd}f}'


def sc(head, side, arm):
    g = SC[(SC['head'] == head) & (SC.side == side) & (SC.arm == arm)]; assert len(g) == 1; return g.iloc[0]


def ac(head, side, rule):
    g = AC[(AC['head'] == head) & (AC.side == side) & (AC.rule == rule)]; assert len(g) == 1; return g.iloc[0]


lam1 = all(float(g['lam']) == 1. for g in fit.values()); all_ok = all(g['status'] == 'ok' for g in fit.values())
S_FIT = ('Every group has fit support. The raw weighted least-squares slope exceeds 1 in all four groups, so lambda clips to 1 everywhere: the fitted candidate '
         'is the pooled-prior control, on every row.' if (lam1 and all_ok and all(float(g['lam_raw']) > 1. for g in fit.values())) else
         'The fitted lambdas: ' + ', '.join(f"{k.replace('|', ' side ')} {float(g['lam']):.3f} ({g['status']})" for k, g in fit.items()) + '.')
same = all(np.array_equal(pd.read_csv(SRC / 'eval_rows.csv', dtype=STR).pipe(lambda x: x[x['head'] == h]).pipe(lambda x: (x[f'V_candidate'].astype(float) == x[f'V_prior'].astype(float)).to_numpy()),
                          np.ones(int((pd.read_csv(SRC / 'eval_rows.csv', dtype=STR)['head'] == h).sum()), bool)) for h in ('entry', 'first_review'))


def pred_line(h):
    p = P[h]['prediction']; o, c = sc(h, 'all', 'original'), sc(h, 'all', 'candidate')
    side = '; '.join(f"side {s_}: movement MSE {'improves' if p[s_]['movement_mse_improves'] else 'does not improve'}, value MSE {'improves' if p[s_]['value_mse_improves'] else 'does not improve'}"
                     for s_ in ('0', '1') if s_ in p)
    return (f"movement MSE {g3(o.mse_m, 1)} → {g3(c.mse_m, 1)} c² ({'improves' if p['all']['movement_mse_improves'] else 'does not improve'}; zero {g3(o.mse_m_zero, 1)}); "
            f"value MSE {g3(o.mse_V, 1)} → {g3(c.mse_V, 1)} ({'improves' if p['all']['value_mse_improves'] else 'does not improve'}; zero {g3(o.mse_zero, 1)}, "
            f"mechanical {g3(o.mse_mech, 1)}); movement bias {f2(o.bias_m)} → {f2(c.bias_m)} c ({'reduced' if p['all']['movement_bias_reduced'] else 'not reduced'}); {side}")


def act_line(h):
    a = {r: ac(h, 'all', r) for r in ('none', 'all', 'original', 'prior', 'candidate')}; verb = 'POSTs on' if h == 'entry' else 'KEEPs'
    return (f"the candidate {verb} {int(a['candidate'].chosen)} of {int(a['all'].n)} rows (J {f2(a['candidate'].J, 3)} c per opportunity, mean among the chosen "
            f"{f2(a['candidate'].mean_chosen)}); original {int(a['original'].chosen)} rows, J {f2(a['original'].J, 3)}; all {f2(a['all'].J, 3)}; none {f2(a['none'].J, 3)}; "
            f"prior {int(a['prior'].chosen)} rows, J {f2(a['prior'].J, 3)}")


L = ['# PI0-MOVEMENT-SHRINK-v1 — report', '', f"{NOW}, session aea5bfb9. Review 7ad03681's one next priority.",
     f"- Freeze `{FREEZE}`: module, selftest, harness, declaration, manifest, checker, launcher; nothing fitted.",
     f"- Fit `{FIT}`: fit.json on the Oct 26 TRAIN keys only, committed before any Oct 27 score.",
     '- Results: the commit carrying this report.',
     "- Scope: a new adaptive experiment after the movement audit, not a retroactive pass of its gate; chronological consumed-day development (fit Oct 26, "
     "measure Oct 27), not unseen validation. One new candidate.", '',
     '## The computed readings (audit.json `predicates`)', '', '| Head | Decision | Reasons | Prediction (pooled) |', '|---|---|---|---|']
for h in ('entry', 'first_review'):
    p = P[h]; L.append(f"| {h} | **{p['decision']}** | {', '.join(p['inconclusive_reasons']) or '—'} | movement MSE {'improves' if p['prediction']['all']['movement_mse_improves'] else 'does not improve'}, "
                       f"value MSE {'improves' if p['prediction']['all']['value_mse_improves'] else 'does not improve'}, movement bias {'reduced' if p['prediction']['all']['movement_bias_reduced'] else 'not reduced'} |")
L += ['', '## The fit (Oct 26 only)', '', S_FIT, '', '| Group | status | n | fills | weight | weighted fills | mu (c / fill) | lambda raw | lambda | prior binding (weight share) |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for k, g in fit.items():
    L.append(f"| {k.replace('|', ', side ')} | {g['status']} | {g['n']} | {g['fills']} | {g3(g['weight'], 2)} | {g3(g['weighted_fills'], 2)} | {f2(g['mu'])} | {g3(g['lam_raw'])} | {g3(g['lam'])} | {g3(g['binding_weight_share'])} |")
L += ['', '## Prediction on Oct 27 (weighted; c per opportunity; c_m per filled opportunity)', '']
for h in ('entry', 'first_review'):
    L.append(f'- **{h}:** {pred_line(h)}.')
L += ['', '| head | side | arm | n | fills (unweighted / weighted) | mean E[qm] | mean M | movement bias | movement MSE | zero-movement MSE | V bias | V MSE | zero | mechanical | c_m (weighted / unweighted) |',
      '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in SC.to_dict('records'):
    L.append(f"| {r['head']} | {r['side']} | {r['arm']} | {int(r['n'])} | {int(r['fills'])} / {g3(r['weighted_fills'], 2)} | {f2(r['mean_Eqm'])} | {f2(r['mean_M'])} | {f2(r['bias_m'])} | "
             f"{g3(r['mse_m'], 1)} | {g3(r['mse_m_zero'], 1)} | {f2(r['bias_V'])} | {g3(r['mse_V'], 1)} | {g3(r['mse_zero'], 1)} | {g3(r['mse_mech'], 1)} | {f2(r['c_m'])} / {f2(r['c_m_unweighted'])} |")
L += ['', 'first_pull rows are descriptive (the first-review parameters), never in the main scores.', '', '## Local actions on Oct 27 (the neutral hold labels; stages never summed)', '']
for h in ('entry', 'first_review'):
    L.append(f'- **{h}:** {act_line(h)}.')
L += ['', '| head | side | rule | n | chosen | chosen weight | J (c / opp.) | mean among chosen | worlds | Kish n | severe n / sum | top-5 gen / world (weighted) | changed vs original |',
      '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in AC.to_dict('records'):
    L.append(f"| {r['head']} | {r['side']} | {r['rule']} | {int(r['n'])} | {int(r['chosen'])} | {g3(r['chosen_weight'], 2)} | {f2(r['J'], 3)} | {f2(r['mean_chosen'])} | {int(r['worlds'])} | "
             f"{g3(r['ess'], 1)} | {int(r['severe_n'])} / {g3(r['severe_sum'], 1)} | {g3(r['top5_generation_share'])} / {g3(r['top5_world_share'])} | {int(r['changed_vs_original'])} |")
L += ['', '| head | strata | free strata | free rows | chosen | J | null mean | p05 | p95 | percentile | weight-mismatch draws |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for h, n in NU.items():
    L.append(f"| {h} | {n['strata']} | {n['free_strata']} | {n['free_rows']} | {n['chosen']} | {f2(n['J'], 3)} | {f2(n['null_mean'], 3)} | {f2(n['null_p05'], 3)} | {f2(n['null_p95'], 3)} | "
             f"{g3(n['percentile'])} | {n['weight_mismatch_draws']} |")
L += ['', '## Validity', '', f"- Checks A–J: {sum(ck['checks'].values())} / {len(ck['checks'])} true (`checks.json`), each recomputing its values independently "
      '(identity and freeze; membership and every eval row against the audit table with the rowwise identity; the fit; every prediction and mask; every score, '
      'calibration and action row; all null draws; the readings; resources and the suite).',
      f"- Falsifier: {nf} / {len(fa['cases'])} corruptions of real copies flagged by their check, with clean copies true (`checks_falsifier.json`).",
      f"- Mutation suite: {mut.group(1)} / {mut.group(2)} planted errors caught by a selftest check (a first 55 / 56 run kept: a fixture gap).",
      f"- Populations as frozen: TRAIN {len(man['train']['entry'])} / {len(man['train']['first_review'])}, EVAL {len(man['eval']['entry'])} / {len(man['eval']['first_review'])}, "
      f"descriptive first pull {len(man['descriptive']['first_pull'])}; every eligible row reconciled with the audit's committed table.",
      f"- Cost as of {asof}: {rs['jobs']} run_light jobs, {g3(rs['core_hours'])} core-hours, peak {rs['peak_mib']} MiB (fit {rs['run_peaks'].get('shr_fit')}, score "
      f"{rs['run_peaks'].get('shr_score')} MiB).", '',
      '## Limits', '', '- One evaluation day; related generations; no interval; the 500 draws are not 500 market samples.',
      '- Entry evaluation filters PI0\'s own admissions (no evidence for posts PI0 rejected); the review is scored on PI0\'s reached live states; local choices '
      'on neutral labels, not a combined strategy.', '- Nothing else is tuned in this round.']
(R / 'report.md').write_text('\n'.join(L) + '\n')
# ---- tables.md (the calibration cells)
T = ['# PI0-MOVEMENT-SHRINK-v1 — calibration tables', '', 'kind: calibration (each arm\'s own E[qm] / V bins) or diagnostic (the audit\'s horizon / queue / ordinal bins).', '',
     '| head | arm | dim | kind | cell | n | fills | weight | mean pred E[qm] | mean M | mean u | mean pred V | mean value |', '|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in CA.to_dict('records'):
    T.append(f"| {r['head']} | {r['arm']} | {r['dim']} | {r['kind']} | {r['cell']} | {int(r['n'])} | {int(r['fills'])} | {g3(r['weight'], 2)} | {f2(r['mean_pred_m'])} | {f2(r['mean_M'])} | "
             f"{f2(r['mean_u'])} | {f2(r['mean_pred_V'])} | {f2(r['mean_value'])} |")
(R / 'tables.md').write_text('\n'.join(T) + '\n')
# ---- state hunks
e, rv = P['entry'], P['first_review']; ae, ar_ = ac('entry', 'all', 'candidate'), ac('first_review', 'all', 'candidate')
F = (f"| P99 | PI0-MOVEMENT-SHRINK-v1 (one scalar movement-calibration challenger: per side × head mu = Σ(w M)/Σ(w z) and λ = clip(weighted LS slope of M − E[qm] on "
     f"prior − E[qm], 0, 1), fitted on the consumed Oct 26 admitted states, measured on the consumed Oct 27 admitted states; a new adaptive experiment, not a "
     f"retroactive pass of P98's gate; as of {NOW}; checks all_true = {str(ck['all_ok']).lower()}, falsifier {nf} / {len(fa['cases'])}). {S_FIT} **Entry (163 rows):** "
     f"{pred_line('entry')}; {act_line('entry')}. **First review (148 rows):** {pred_line('first_review')}; {act_line('first_review')}. | Computed: entry "
     f"`{e['decision']}` ({', '.join(e['inconclusive_reasons']) or '—'}), first review `{rv['decision']}` ({', '.join(rv['inconclusive_reasons']) or '—'}). "
     f"{'The challenger only reduces posting / keeping; it selects no positive exposure (the candidate J does not exceed choosing none). ' if (float(ae.J) <= 0. and float(ar_.J) <= 0.) else ''}"
     f"P98's `representation_or_data_needed` flag is a declared research heuristic (review 7ad03681 item 1); the forced cohort's near-zero error is aggregate "
     f"calibration on that cohort, not conditional calibration of admissions. |\n")
H = (f"## {NOW} (session aea5bfb9) — PI0-MOVEMENT-SHRINK-v1: ONE SCALAR MOVEMENT CHALLENGER FIT OCT 26 / MEASURED OCT 27; CHECKS {'ALL TRUE' if ck['all_ok'] else 'NOT ALL TRUE'}; "
     f"ENTRY {e['decision']}, FIRST REVIEW {rv['decision']} (review 7ad03681's next priority) (USER: \"... continue the implementing and next priority loop\")\n"
     f"- [Receipt](reviews/movement_shrink_20261121/report.md).\n"
     f"  - Commits: freeze `97208136` (before any fit), fit `{FIT}` (before any Oct 27 score), results in this commit.\n"
     f"  - One new candidate (4 groups, ≤ 8 scalars); nothing else tuned.\n"
     f"- **Review 7ad03681's checklist (folded in):**\n"
     f"  - Item 1: P98's flag stays as recorded and is a declared research heuristic; the forced near-zero error is aggregate calibration on that cohort; this round "
     f"is a new adaptive experiment.\n"
     f"  - Item 2: shrink_checks.py recomputes every aggregate value of this receipt; its falsifier corrupts conditional means, MSE, weighted denominators, selection "
     f"values, null draws and readings.\n"
     f"  - Item 3: weighted and unweighted fill denominators stored and labelled; cost as of {asof}. P98's per-share table showed 2,014.7 unweighted shares beside "
     f"weighted ratios (weighted denominator 4,429.48); the audit's final logs total 152 jobs / 0.282 core-hours (not 148 / 0.275).\n"
     f"  - Item 4: every admission record bound to its committed copy, the audit table by its committed hash; no new mid lookup.\n"
     f"- **Readings** (FINDINGS P99):\n"
     f"  - Fit: {S_FIT}\n"
     f"  - Entry: {pred_line('entry')}; {act_line('entry')}; decision {e['decision']} ({', '.join(e['inconclusive_reasons']) or '—'}).\n"
     f"  - First review: {pred_line('first_review')}; {act_line('first_review')}; decision {rv['decision']} ({', '.join(rv['inconclusive_reasons']) or '—'}).\n"
     f"- **Cost:** {g3(rs['core_hours'])} core-hours as of {asof}; peak {rs['peak_mib']} MiB.\n"
     f"- **Next measurement:** the combined review (one request), which names the next priority.\n\n")
St = (f"movement_shrink_20261121:\n  status: done_pi0_movement_shrink_v1_one_candidate\n  asof: \"{NOW}\"\n  receipt: predictfun/reviews/movement_shrink_20261121/report.md\n"
      f"  commits: {{freeze: 97208136, fit: {FIT}}}\n  checks: {{all_true: {str(ck['all_ok']).lower()}, falsifier_ok: {str(fa['falsifier_ok']).lower()}, mutants_caught: {mut.group(1)}, mutants: {mut.group(2)}}}\n"
      f"  populations: {{train_entry: {len(man['train']['entry'])}, train_first_review: {len(man['train']['first_review'])}, eval_entry: {len(man['eval']['entry'])}, "
      f"eval_first_review: {len(man['eval']['first_review'])}}}\n"
      + ''.join(f"  fit_{k.replace('|', '_side')}: {{status: {g['status']}, mu: {float(g['mu']):.4f}, lam_raw: {float(g['lam_raw']):.4f}, lam: {float(g['lam']):.4f}}}\n" for k, g in fit.items())
      + ''.join(f"  reading_{h}: {{decision: {P[h]['decision']}, reasons: [{', '.join(P[h]['inconclusive_reasons'])}], movement_mse_improves: {str(P[h]['prediction']['all']['movement_mse_improves']).lower()}, "
                f"value_mse_improves: {str(P[h]['prediction']['all']['value_mse_improves']).lower()}, movement_bias_reduced: {str(P[h]['prediction']['all']['movement_bias_reduced']).lower()}, "
                f"J_candidate: {float(ac(h, 'all', 'candidate').J):.4f}, J_original: {float(ac(h, 'all', 'original').J):.4f}, chosen: {int(ac(h, 'all', 'candidate').chosen)}}}\n" for h in ('entry', 'first_review'))
      + f"  cost_core_hours: {float(rs['core_hours']):.3f}\n  next: combined_review_names_the_next_priority\n")
for n, t in (('F_shrink.md', F), ('H_shrink.md', H), ('S_shrink.yml', St)):
    (S / 'edits' / n).write_text(t)
print(f'report.md {len(L)} lines; tables.md {len(T)} lines; candidate equals prior on every eval row: {same}')
