"""PI0-COVERAGE-v1: copies the outputs into outputs/ (SHA256SUMS), writes report.md and tables.md, and the state hunks (FINDINGS P101, HANDOFF section,
STATUS block) into the scratchpad's edits/. Every number is read from a file; every answer is audit.json's computed predicate; every comparative
sentence is selected by computed values. argv[1] = the time from `date`. Usage: fill_coverage.py <time>"""
import glob
import gzip
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path('/home/yuqing/ctaNew/predictfun/reviews/coverage_20261121'); SRC = Path('/home/yuqing/ctaNew/data/predict_fun/research/pi0_coverage_v1_20261121')
S = Path('/tmp/claude-1001/-home-yuqing-ctaNew/aea5bfb9-4ea4-41d0-bf73-bb98d818e3d4/scratchpad'); NOW = sys.argv[1]
GO, SRC_FREEZE, FREEZE = '946536ad', 'aad6a8b6', '8a990d5f'
STR = {c: str for c in ('day', 'side', 'cat', 'population', 'rule', 'dim', 'cell', 'hbin', 'status', 'state')}
D = R / 'outputs'; D.mkdir(exist_ok=True)
for n in ('forecasts', 'calibration', 'coverage', 'actions', 'census_statuses', 'reference'):
    shutil.copy(SRC / f'{n}.csv', D / f'{n}.csv')
for n in ('labels', 'null_draws'):
    b = (SRC / f'{n}.csv').read_bytes(); (D / f'{n}.csv.gz').write_bytes(gzip.compress(b, mtime=0)); assert gzip.decompress((D / f'{n}.csv.gz').read_bytes()) == b
shutil.copy(SRC / 'audit.json', D / 'audit.json'); (D / 'parts').mkdir(exist_ok=True); (D / 'pilot').mkdir(exist_ok=True)
for p in sorted((SRC / 'parts').glob('part_*.json')) + sorted((SRC / 'pilot').glob('pilot_*.json')):
    if not p.name.endswith('_status.json'):
        shutil.copy(p, D / p.parent.name / p.name)
(D / 'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(D)}\n' for p in sorted(D.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS'))
rd = lambda n: pd.read_csv(SRC / f'{n}.csv', float_precision='round_trip', dtype=STR)
FC, CA, CV, AC, CS, REF = (rd(n) for n in ('forecasts', 'calibration', 'coverage', 'actions', 'census_statuses', 'reference'))
au = json.loads((SRC / 'audit.json').read_text()); ck = json.loads((R / 'checks.json').read_text()); fa = json.loads((R / 'checks_falsifier.json').read_text())
pj = json.loads((R / 'logs' / 'pilot_projection.json').read_text()); man = json.loads((R / 'manifest.json').read_text())
P = au['predicates']; NU = au['null']; H = ck['details'].get('H', {}); rs = H.get('resources', {})
mut = {k: re.search(r'(\d+) / (\d+) planted code errors are caught', (p).read_text()) for k, p in (('cov', R / 'logs' / 'mutants_cov_summary.log'), ('src', R / 'source' / 'logs' / 'mutants_src_summary.log'))}
f2 = lambda v, nd=2: '—' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{float(v):+,.{nd}f}'
g3 = lambda v, nd=3: '—' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{float(v):,.{nd}f}'
nf = sum(1 for v in fa['cases'].values() if v['clean'] and not v['corrupted'] and v['applied'])


def tbl(df, cols, heads=None, fmt=None):
    heads = heads or cols; fmt = fmt or {}; out = ['| ' + ' | '.join(heads) + ' |', '|' + '|'.join('---' for _ in heads) + '|']
    for r in df.to_dict('records'):
        out.append('| ' + ' | '.join((fmt[c](r[c]) if c in fmt else ('—' if r[c] is None or (isinstance(r[c], float) and np.isnan(r[c])) else str(r[c]))) for c in cols) + ' |')
    return '\n'.join(out)


pool = FC[(FC.day == 'all') & (FC.side == 'all')]
num = {c: (lambda v, c=c: g3(v, 1) if c.startswith('mse') else (f2(v) if c.startswith(('bias', 'c_m', 'mean')) else g3(v))) for c in FC.columns}
for c in ('n', 'fills', 'worlds', 'severe_n'):
    num[c] = lambda v: str(int(float(v)))
sel = P['selects_positive_exposure']; conc = P['concentration']
S_SEL = '; '.join(f"{k}: **{v['answer']}** (J rule {f2(v['J_rule'], 3)}, all {f2(v['J_all'], 3)}, none {f2(v['J_none'], 3)}, null p95 {f2(v['null_p95'], 3)}, "
                  f"free strata {v['free_strata']}, free rows {v['free_rows']}{'; ' + ', '.join(v['reasons']) if v['reasons'] else ''})" for k, v in sorted(sel.items()))
S_CONC = '; '.join(f"{k}: value error {'concentrated' if v['value_error_concentrated'] else 'not concentrated'} (top-5 world share {g3(v['top5_world_err'])}), "
                   f"severe {'concentrated' if v['severe_concentrated'] else 'not concentrated'} ({g3(v['top5_world_severe'])}), largest date share {g3(v.get('max_date_share_err'))}"
                   for k, v in sorted(conc.items()))
stat = pd.DataFrame([dict(status=k, n=v) for k, v in au['statuses'].items()])
L = ['# PI0-COVERAGE-v1 — report', '', f'{NOW}, session aea5bfb9. Review 216ff603\'s next priority (next_priority.md).',
     f'- USER go `{GO}` (13:41Z "Approve go") and the preflight choice "Extend source, all 6" (14:2xZ).',
     f'- Source extension: freeze `{SRC_FREEZE}`; the Oct 24 control reproduced the 11-17 outputs exactly; Oct 31 and Nov 1-3 built.',
     f'- Coverage freeze `{FREEZE}`: builder, selftest, harness, checks, manifest, pilot rule, launcher, before the pilot and any label.',
     '- Results: the commit carrying this report. Development data on consumed dates; no fit, no candidate, no validation, no strategy P&L.', '',
     '## Computed readings (audit.json `predicates`)', '', f'- **Selects positive local exposure (PI0\'s unchanged sign rule):** {S_SEL}.',
     f'- **Concentration:** {S_CONC}.', f"- **Covered cells** (>= 30 labeled rows and >= 10 fills per source x side x horizon bin): {len(P['covered'])}; "
     f"**lacking support**: {len(P['lacking_support'])} (lists in audit.json).", '',
     '## Population and statuses', '', f"- Manifest: {sum(len(man['sample'][d]['selected']) for d in man['days'])} worlds on {len(man['days'])} dates, {len(man['partitions'])} partitions; "
     f"parts done {len(au['parts_done'])}, missing {len(au['parts_missing'])}.", f"- Labeled rows {au['rows']}, eligible (ok, model moments) {au['eligible']}.", '',
     tbl(stat, ['status', 'n']), '', '### Census statuses (every decision of the factual episodes)', '',
     tbl(CS.groupby(['state', 'status'], as_index=False)['n'].sum().sort_values('n', ascending=False), ['state', 'status', 'n']), '',
     '## Forecasts of the unchanged PI0 moments (weighted by 1 / (pi_world pi_cell); c per opportunity; c_m per filled opportunity)', '',
     tbl(pool, ['cat', 'n', 'fills', 'fill_rate', 'mean_P', 'brier', 'brier_pooled', 'mean_V', 'mean_value', 'bias_V', 'mse_V', 'mse_zero', 'mse_mech', 'bias_m', 'mse_m', 'mse_m_zero',
                'c_m', 'severe_n', 'severe_sum'], fmt=num), '', '### By date', '',
     tbl(FC[(FC.side == 'all') & (FC.day != 'all')], ['cat', 'day', 'n', 'fills', 'mean_V', 'mean_value', 'bias_V', 'mse_V', 'mse_zero', 'bias_m', 'c_m', 'severe_n'], fmt=num), '',
     '### By side', '', tbl(FC[(FC.day == 'all') & (FC.side != 'all')], ['cat', 'side', 'n', 'fills', 'mean_V', 'mean_value', 'bias_V', 'mse_V', 'mse_zero', 'bias_m', 'c_m'], fmt=num), '',
     '### Concentration (weighted / unweighted top-5 shares; largest date share)', '',
     tbl(pool, ['cat', 'top5_gen_err', 'top5_world_err', 'top5_gen_err_unweighted', 'top5_world_err_unweighted', 'top5_world_severe', 'max_date_share_err', 'max_date_share_severe'], fmt=num), '',
     '## Local actions (neutral-hold labels; stages never summed; not PI0 P&L)', '',
     tbl(AC, ['population', 'rule', 'n', 'chosen', 'chosen_weight', 'J', 'mean_chosen', 'degenerate'], fmt={'J': lambda v: f2(v, 3), 'mean_chosen': f2, 'chosen_weight': lambda v: g3(v, 1)}), '',
     tbl(pd.DataFrame([dict(population=k, **v) for k, v in NU.items()]), ['population', 'strata', 'free_strata', 'free_rows', 'chosen', 'J', 'null_mean', 'null_p95', 'percentile',
                                                                      'chosen_weight_ratio_mean'], fmt={c: (lambda v: f2(v, 3)) for c in ('J', 'null_mean', 'null_p95')}), '',
     '## The Oct 26-27 admitted cohort (PI0-ADMISSION; a separate reference, never pooled)', '',
     tbl(REF[REF.side == 'all'], ['cat', 'day', 'n', 'fills', 'mean_V', 'mean_value', 'bias_V', 'mse_V', 'mse_zero', 'bias_m', 'c_m'], fmt=num), '',
     '## Validity', '', f"- Checks A-H: {sum(ck['checks'].values())} / {len(ck['checks'])} true (`checks.json`); falsifier {nf} / {len(fa['cases'])} (`checks_falsifier.json`).",
     f"- Mutation suites: builder {mut['cov'].group(1)} / {mut['cov'].group(2)}, source wrapper {mut['src'].group(1)} / {mut['src'].group(2)}, each caught by a selftest check.",
     f"- Pilot projection (write-once): {g3(pj.get('projected_core_hours'))} core-hours projected, pilot {g3(pj.get('pilot_core_hours'))} core-hours, run_rest {str(pj.get('run_rest')).lower()}.",
     f"- Cost as of {H.get('asof', '?')[:16].replace('T', ' ')}Z: {rs.get('jobs')} run_light jobs, {g3(rs.get('core_hours'))} core-hours (incl. the source extension), peak {rs.get('peak_mib')} MiB; "
     f"record high-water {g3(H.get('hwm_max'), 0)} MiB.", '', '## Limits', '',
     '- Development data on consumed dates; related generations within worlds; no interval; the 500 draws are not market samples.',
     '- Entry sources are PI0\'s factual POSTs and WAITs at its own reached states; the review source is PI0\'s first live reviews; local neutral-hold values.',
     '- The cell hour is the world\'s stratum hour (decisions lie within 5 s of the world start).']
(R / 'report.md').write_text('\n'.join(L) + '\n')
T = ['# PI0-COVERAGE-v1 — tables', '', '## Coverage (date x side x category x horizon bin)', '', tbl(CV, ['day', 'side', 'cat', 'hbin', 'rows', 'eligible', 'fills', 'statuses']), '',
     '## Calibration and diagnostic bins', '', tbl(CA, [c for c in CA.columns], fmt={c: g3 for c in ('mean_P', 'fill_rate', 'mean_V', 'mean_value', 'mean_Eqm', 'mean_M') if c in CA.columns}), '',
     '## Forecasts by date x side', '', tbl(FC[(FC.day != 'all') & (FC.side != 'all')], ['cat', 'day', 'side', 'n', 'fills', 'bias_V', 'mse_V', 'mse_zero', 'bias_m', 'c_m'], fmt=num)]
(R / 'tables.md').write_text('\n'.join(T) + '\n')
row = lambda c: pool[pool.cat == c].iloc[0] if (pool.cat == c).any() else None
pr = '; '.join(f"{c}: n {int(row(c).n)}, fills {int(row(c).fills)}, mean V {f2(row(c).mean_V)} vs value {f2(row(c).mean_value)} c, MSE(V) {g3(row(c).mse_V, 1)} vs zero {g3(row(c).mse_zero, 1)}"
               for c in ('absent_POST', 'absent_WAIT', 'review_KEEP', 'review_PULL') if row(c) is not None)
F = (f"| P101 | PI0-COVERAGE-v1 (neutral-exposure labels on six consumed dates - Oct 24-25, Oct 31, Nov 1-3, the last four with the frozen source chain "
     f"extended by USER choice - under the frozen PI0 and neutral-hold target; {sum(len(man['sample'][d]['selected']) for d in man['days'])} worlds, {au['rows']} labeled queries "
     f"({au['eligible']} eligible), inclusion-weighted; no fit; as of {NOW}; checks all_true = {str(ck['all_ok']).lower()}, falsifier {nf} / {len(fa['cases'])}). {pr}. | "
     f"Computed: {S_SEL}. Concentration: {S_CONC}. Covered cells {len(P['covered'])}, lacking {len(P['lacking_support'])}. Development data; not validation; local values are not PI0 P&L. |\n")
Hh = (f"## {NOW} (session aea5bfb9) — PI0-COVERAGE-v1: SIX-DATE NEUTRAL-EXPOSURE LABELS UNDER THE FROZEN PI0; CHECKS {'ALL TRUE' if ck['all_ok'] else 'NOT ALL TRUE'} (review 216ff603's next priority; USER go)\n"
      f"- [Receipt](reviews/coverage_20261121/report.md). Commits: go `{GO}`, source freeze `{SRC_FREEZE}`, coverage freeze `{FREEZE}`, results in this commit.\n"
      f"- **Readings** (FINDINGS P101): {S_SEL}.\n  - Concentration: {S_CONC}.\n  - Covered cells {len(P['covered'])}, lacking support {len(P['lacking_support'])}.\n"
      f"  - Pooled forecasts: {pr}.\n"
      f"- **Validity:** checks A-H {sum(ck['checks'].values())} / {len(ck['checks'])}; falsifier {nf} / {len(fa['cases'])}; mutants {mut['cov'].group(1)} / {mut['cov'].group(2)} and "
      f"{mut['src'].group(1)} / {mut['src'].group(2)}; the projection {g3(pj.get('projected_core_hours'))} core-hours.\n"
      f"- **Cost:** {g3(rs.get('core_hours'))} core-hours as of the checks; peak {rs.get('peak_mib')} MiB.\n"
      f"- **Next measurement:** the combined review (one request), which names the next priority.\n\n")
St = (f"coverage_20261121:\n  status: done_pi0_coverage_v1_no_fit\n  asof: \"{NOW}\"\n  receipt: predictfun/reviews/coverage_20261121/report.md\n"
      f"  commits: {{go: {GO}, source_freeze: {SRC_FREEZE}, freeze: {FREEZE}}}\n  checks: {{all_true: {str(ck['all_ok']).lower()}, falsifier_ok: {str(fa['falsifier_ok']).lower()}}}\n"
      f"  population: {{worlds: {sum(len(man['sample'][d]['selected']) for d in man['days'])}, partitions: {len(man['partitions'])}, labeled: {au['rows']}, eligible: {au['eligible']}}}\n"
      + ''.join(f"  selects_positive_{k}: {{answer: {v['answer']}, J_rule: {float(v['J_rule']):.4f}, J_all: {float(v['J_all']):.4f}, null_p95: {float(v['null_p95']):.4f}}}\n" for k, v in sorted(sel.items()))
      + f"  covered_cells: {len(P['covered'])}\n  lacking_cells: {len(P['lacking_support'])}\n  cost_core_hours: {float(rs.get('core_hours') or 0):.3f}\n  next: combined_review_names_the_next_priority\n")
for n, t in (('F_cov.md', F), ('H_cov.md', Hh), ('S_cov.yml', St)):
    (S / 'edits' / n).write_text(t)
print(f'report.md {len(L)} lines, tables.md {len(T)} lines')
