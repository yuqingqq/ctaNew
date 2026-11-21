"""The PI0-MOVEMENT-CALIBRATION-AUDIT-v1 state hunks (FINDINGS P98, HANDOFF section, STATUS block), every number read from the outputs, every
answer from audit.json's computed predicates and every reading sentence selected by the computed recommendation (declaration.md's map).
The HANDOFF time comes from `date` (argv[1]). Writes edits/F_movement.md, edits/H_movement.md, edits/S_movement.yml in the scratchpad."""
import json
import re
import sys
from pathlib import Path

import pandas as pd

S = Path('/tmp/claude-1001/-home-yuqing-ctaNew/aea5bfb9-4ea4-41d0-bf73-bb98d818e3d4/scratchpad'); NOW = sys.argv[1]
R = Path('/home/yuqing/ctaNew/predictfun/reviews/movement_audit_20261121'); D = R / 'outputs'
STR = {c: str for c in ('day', 'side', 'cell', 'view', 'admission', 'stage', 'cohort', 'subset', 'dim', 'quantity', 'weighting')}
au = json.loads((D / 'audit.json').read_text()); ck = json.loads((R / 'checks.json').read_text()); fa = json.loads((R / 'checks_falsifier.json').read_text())
MO, PP, CO = (pd.read_csv(D / f'{n}.csv', dtype=STR) for n in ('moments', 'prepost', 'concentration'))
P = {p['stage']: p for p in au['predicates']}; e = P['entry']; ctl = au['control']; rs = ck['details']['I']['resources']
mut = re.search(r'(\d+) / (\d+) planted code errors are caught', (R / 'logs' / 'mutants_mov_summary.log').read_text())
one = lambda df: df.iloc[0] if len(df) == 1 else (_ for _ in ()).throw(SystemExit(f'not one row: {len(df)}'))
ae = one(MO[(MO.cohort == 'admitted') & (MO.stage == 'entry') & (MO.view == 'all')]); fo = one(MO[(MO.cohort == 'forced') & (MO.stage == 'entry') & (MO.view == 'overlap')])
pa = one(PP[(PP.cohort == 'admitted') & (PP.stage == 'entry') & (PP.view == 'all')])
cw = one(CO[(CO.quantity == 'movement_error_entry') & (CO.weighting == 'weighted')])
SENT = dict(representation_or_data_needed='a small chronologically fitted calibration candidate is not supported by these cells; more representative data or a '
                                          'different target representation is needed (declaration.md)',
            calibration_candidate_defensible='a small chronologically fitted calibration candidate could be declared next (declaration.md)',
            inconclusive='the audit does not decide between a calibration candidate and a representation / data change (declaration.md)')
all_true = bool(ck['all_ok']); fok = bool(fa['falsifier_ok']); n_f = sum(1 for v in fa['cases'].values() if v['clean'] and not v['corrupted'] and v['applied'])
cells = ', '.join(f"Oct {k[6:8]} {'bid' if k.endswith('/0') else 'ask'} {v:+.2f}" for k, v in e['i']['cells'].items())
ans = lambda p: f"(i) {p['i']['answer']}, (ii) {p['ii']['answer']}, (iii) {p['iii']['answer']}, (iv) {p['iv']['answer']} → `{p['recommendation']}`"
rec = e['recommendation']; others = {nm: P[nm]['recommendation'] for nm in ('first_review', 'first_pull')}
same = all(v == rec for v in others.values())
F = (f"| P98 | PI0-MOVEMENT-CALIBRATION-AUDIT-v1 (PI0's frozen movement head on the {au['membership']['admitted_rows']} admitted generations, inclusion-weighted, Oct 26–27, "
     f"against the {au['membership']['forced_rows']} forced forks as an unweighted neutral reference, Oct 24–27; exact rowwise decomposition, declared decision-state strata, "
     f"realized fill timing and a pre / post-fill split from fill-time mids; no fit; as of {au['asof'][:16].replace('T', ' ')}Z; checks all_true = {str(all_true).lower()}, "
     f"falsifier {n_f} / {len(fa['cases'])}). **Admitted entry:** V − value {ae.bias_V:+.3f} c = edge × (E[q] − q) {ae.qty_term:+.3f} + E[qm] − M {ae.mov_term:+.3f}, of "
     f"which probability factor {ae.mov_prob:+.3f} and conditional head {ae.mov_cond:+.3f}. On fills the head forecasts {ae.mean_mhat_fill:+.2f} c per fill against "
     f"{ae.mean_M_fill:+.2f} realized (c = {ae.c_m:+.2f}, {int(ae.fills)} fills); the forced Oct 26–27 reference forecasts {fo.mean_mhat_fill:+.2f} against "
     f"{fo.mean_M_fill:+.2f} (c = {fo.c_m:+.2f}, {int(fo.fills)} fills). Per filled share the admitted movement is {pa.total_per_share:+.3f} c, of which "
     f"{pa.post_per_share:+.3f} post-fill. u = E[qm] − M by day × side: {cells} c; weighted top-5 world share {cw.top5_world_share:.3f}. Largest decision-state "
     f"range of c {max(v for v in e['ii']['decision_ranges'].values() if v is not None):.1f} c, delay range {e['ii']['delay_range']:.1f} c. | Computed (entry): {ans(e)}"
     f"{'; first review and first pull give the same recommendation' if same else '; first review ' + others['first_review'] + ', first pull ' + others['first_pull']}. "
     f"Declared meaning: {SENT[rec]}. |\n")
H = (f"## {NOW} (session aea5bfb9) — PI0-MOVEMENT-CALIBRATION-AUDIT-v1: THE MOVEMENT HEAD'S DECOMPOSITION, STRATA, FILL TIMING AND PRE / POST-FILL SPLIT; CHECKS "
     f"{'ALL TRUE' if all_true else 'NOT ALL TRUE'}; RECOMMENDATION {rec.upper()} (review 5a8ee6f5's next priority) (USER: \"... continue the implementing and next priority loop\")\n"
     f"- [Receipt](reviews/movement_audit_20261121/report.md).\n"
     f"  - Commits: freeze `03243e5b` (before any table), results in this commit.\n"
     f"  - One diagnostic; zero fitted candidates.\n"
     f"- **Review 5a8ee6f5's checklist (folded in):**\n"
     f"  - Finding 1: the bookTicker label (`new_bookticker`); the three clocks and units documented.\n"
     f"  - Finding 2: explicit unconditional / conditional / per-share units; qm is decision-to-END movement.\n"
     f"  - Finding 3: weighted and unweighted concentration; signed day × side contributions.\n"
     f"  - Finding 4: exact-key weight / n_cell / cell / part binding (module, check B, falsifier B_weight); the control's hour from each review's own decision time "
     f"({ctl['cell_equal_admission']} / {ctl['reviews_with_time']} equal the admission cell).\n"
     f"- **Readings** (FINDINGS P98):\n"
     f"  - Entry: {ans(e)}.\n"
     f"  - First review: {ans(P['first_review'])}; first pull: {ans(P['first_pull'])}.\n"
     f"  - Conditional head on fills: admitted c {ae.c_m:+.2f} (forecast {ae.mean_mhat_fill:+.2f}, realized {ae.mean_M_fill:+.2f} c per fill); forced Oct 26–27 c "
     f"{fo.c_m:+.2f} (forecast {fo.mean_mhat_fill:+.2f}, realized {fo.mean_M_fill:+.2f}).\n"
     f"  - Declared meaning: {SENT[rec]}.\n"
     f"- **Validity:** checks A–I {sum(ck['checks'].values())} / {len(ck['checks'])}; falsifier {n_f} / {len(fa['cases'])}; mutants {mut.group(1)} / {mut.group(2)}; "
     f"the review's independent numbers reproduce (check H).\n"
     f"- **Cost:** {rs['core_hours']:.3f} core-hours of the 1 ceiling; peak {rs['peak_mib']} MiB.\n"
     f"- **Next measurement:** the combined review (one request), which names the next priority.\n\n")
St = (f"movement_audit_20261121:\n"
      f"  status: done_pi0_movement_calibration_audit_v1_no_fit\n"
      f"  asof: \"{NOW}\"\n"
      f"  receipt: predictfun/reviews/movement_audit_20261121/report.md\n"
      f"  commits: {{freeze: 03243e5b}}\n"
      f"  checks: {{all_true: {str(all_true).lower()}, falsifier_ok: {str(fok).lower()}, mutants_caught: {mut.group(1)}, mutants: {mut.group(2)}}}\n"
      f"  cohorts: {{admitted: {au['membership']['admitted_rows']}, forced: {au['membership']['forced_rows']}}}\n"
      f"  entry_values: {{bias_V: {ae.bias_V:.4f}, mov_prob: {ae.mov_prob:.4f}, mov_cond: {ae.mov_cond:.4f}, c_admitted: {ae.c_m:.4f}, c_forced_overlap: {fo.c_m:.4f}, "
      f"post_fill_per_share: {pa.post_per_share:.4f}, total_per_share: {pa.total_per_share:.4f}}}\n"
      + ''.join(f"  reading_{nm}: {{i: {P[nm]['i']['answer']}, ii: {P[nm]['ii']['answer']}, iii: {P[nm]['iii']['answer']}, iv: {P[nm]['iv']['answer']}, "
                f"recommendation: {P[nm]['recommendation']}}}\n" for nm in ('entry', 'first_review', 'first_pull'))
      + f"  control: {{selected: {ctl['n_sel']}, eligible: {ctl['n']}, percentile: {ctl['percentile']}, cell_equal_admission: {ctl['cell_equal_admission']}}}\n"
      f"  cost_core_hours: {rs['core_hours']:.3f}\n"
      f"  next: combined_review_names_the_next_priority\n")
for n, t in (('F_movement.md', F), ('H_movement.md', H), ('S_movement.yml', St)):
    (S / 'edits' / n).write_text(t)
print(F); print(H); print(St)
