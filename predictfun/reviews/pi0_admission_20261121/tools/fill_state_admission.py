"""The PI0-ADMISSION-EXPOSURE-v1 state hunks (FINDINGS P97, HANDOFF section, STATUS block) from fill_admission.py's values
(admission_values.json). The HANDOFF time comes from `date` (argv[1]). Writes edits/F_admission.md, edits/H_admission.md, edits/S_admission.yml."""
import json
import sys
from pathlib import Path

S = Path('/tmp/claude-1001/-home-yuqing-ctaNew/aea5bfb9-4ea4-41d0-bf73-bb98d818e3d4/scratchpad'); NOW = sys.argv[1]
v = json.loads((S / 'admission_values.json').read_text()); P = v['predicates']; e = v['pooled']; sel = P['review_selection']
days = '; '.join(f"Oct {d[-2:]} {x['wmean_D']:+.3f} c ({x['n']})" for d, x in sorted(P['days'].items()))
reads = '; '.join(f"`{k}`: {v['sentences'][k]}" for k in v['holds'])
F = (f"| P97 | PI0-ADMISSION-EXPOSURE-v1 (the same keep-the-order contrast on {v['generations']} of the 689 POSTs PI0 itself made in the R4E replay, consumed Oct 26–27, four per "
     f"decision-UTC cell with inclusion weights; no fit; as of {v['asof'][:16].replace('T', ' ')}Z; checks all_true = {str(v['all_true']).lower()}). **Weighted mean D_EXPOSURE "
     f"= R(KEEP_SELECTED) − R(PI0) per day:** {days}; pooled {e['wmean_D']:+.3f} c (selected generation {e['wmean_D_selected']:+.3f}); P(fill) {e['factual_fill_prob']:.3f} → "
     f"{e['keep_fill_prob']:.3f}. First-review PULL selection: {v['review_sentence']} RETENTION v2 (supersedes P96's MAE-gated `forecasts_fail`): "
     f"{' '.join(v['retention_sentences'])} | {reads} |\n")
H = (f"## {NOW} (session aea5bfb9) — PI0-ADMISSION-EXPOSURE-v1: LONGER EXPOSURE OF PI0'S OWN ADMISSIONS; CHECKS {'ALL TRUE' if v['all_true'] else 'NOT ALL TRUE'}; "
     f"READING {', '.join(v['holds']).upper()} (review 99fb165d's next priority) (USER: \"... continue the implementing and next priority loop\")\n"
     f"- [Receipt](reviews/pi0_admission_20261121/report.md).\n"
     f"  - Commits: freeze `6b14953f` (before any branch ran), results in this commit.\n"
     f"  - One diagnostic; zero fitted candidates.\n"
     f"- **Review 99fb165d's checklist (folded in):**\n"
     f"  - P1: the MAE gate is superseded by RETENTION v2 (MSE / calibration); the perfect-mean mixture is a control.\n"
     f"  - P2: both-modeled masks; new observations are separated from aging; \"not demonstrated on this forced sample\".\n"
     f"  - P3: the pilot record is write-once.\n"
     f"  - The cross-tab of D_EXPOSURE by shadow admission is reported.\n"
     f"- **Readings** (FINDINGS P97):\n"
     f"  - Weighted mean D_EXPOSURE per day: {days}.\n"
     f"  - First review: {v['review_sentence']}\n"
     f"  - RETENTION v2: {' '.join(v['retention_sentences'])}\n"
     + ''.join(f"  - Declared reading `{k}`: {v['sentences'][k]}\n" for k in v['holds'])
     + f"- **Cost:** {v['core_hours']:.3f} core-hours of the 1 ceiling; peak {v['peak_mib']} MiB.\n"
     f"- **Next measurement:** the combined review (one request), which names the next priority.\n\n")
St = (f"pi0_admission_20261121:\n"
      f"  status: done_pi0_admission_exposure_v1_no_fit\n"
      f"  asof: \"{NOW}\"\n"
      f"  receipt: predictfun/reviews/pi0_admission_20261121/report.md\n"
      f"  commits: {{freeze: 6b14953f}}\n"
      f"  checks: {{all_true: {str(v['all_true']).lower()}, falsifier_ok: {str(v['falsifier_ok']).lower()}}}\n"
      f"  generations: {{census: 689, selected: {v['generations']}}}\n"
      + ''.join(f"  day_{d}: {{generations: {x['n']}, wmean_D: {x['wmean_D']:.4f}, sign: {x['sign']}}}\n" for d, x in sorted(P['days'].items()))
      + f"  gain_forks: {P['gain_forks']}\n"
      f"  review_selection: {{selected: {sel['n_sel']}, eligible: {sel['n']}, percentile: {sel['percentile']}, free_strata: {sel['free_strata']}, thin: {str(sel['thin']).lower()}, "
      f"discriminates: {str(P['review_discriminates']).lower()}}}\n"
      f"  retention_v2: {{entry_mse_beats_zero: {str(P['retention_v2']['entry_mse_beats_zero']).lower()}, review_mse_beats_zero: {str(P['retention_v2']['review_mse_beats_zero']).lower()}}}\n"
      f"  readings: {{{', '.join(f'{k}: {str(x).lower()}' for k, x in P['readings'].items())}}}\n"
      f"  cost_core_hours: {v['core_hours']:.3f}\n"
      f"  next: combined_review_names_the_next_priority\n")
for n, t in (('F_admission.md', F), ('H_admission.md', H), ('S_admission.yml', St)):
    (S / 'edits' / n).write_text(t)
print(F); print(H); print(St)
