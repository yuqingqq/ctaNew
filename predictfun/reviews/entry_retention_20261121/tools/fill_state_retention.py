"""The ENTRY-RETENTION-DISCRIMINATION-v1 state hunks (FINDINGS P96, HANDOFF section, STATUS block) from fill_retention.py's values
(retention_values.json). The HANDOFF time comes from `date` (argv[1]). Writes edits/F_retention.md, edits/H_retention.md, edits/S_retention.yml."""
import json
import sys
from pathlib import Path

S = Path('/tmp/claude-1001/-home-yuqing-ctaNew/aea5bfb9-4ea4-41d0-bf73-bb98d818e3d4/scratchpad'); NOW = sys.argv[1]
v = json.loads((S / 'retention_values.json').read_text()); P = v['predicates']; sel = P['selection']; c = v['calibration']; ca = v['changes']
excl = ', '.join(f'{n} {k}' for k, n in sorted(v['statuses'].items()) if k != 'ok') or 'none'
stage = lambda nm, lab: (f"{lab}: {sel[nm]['n_sel']} of {sel[nm]['n']} selected, mean value {sel[nm]['mean_sel']:+.2f} c vs matched null {sel[nm]['null_mean']:+.2f} c, "
                         f"percentile {sel[nm]['percentile']:.3f}{' (thin)' if sel[nm]['thin'] else ''}; MAE V {c[nm]['mae_V']:.2f} vs zero {c[nm]['mae_zero']:.2f} vs edge×E[q] "
                         f"{c[nm]['mae_mech']:.2f} c") if sel[nm]['mean_sel'] is not None and nm in c else f'{lab}: no selection'
reads = '; '.join(f"`{k}`: {v['sentences'][k]}" for k in v['holds'])
F = (f"| P96 | ENTRY-RETENTION-DISCRIMINATION-v1 (PI0's frozen forecasts at the shadow entry, the first review and the first pull of ENTRY-EXPOSURE-v1's forced "
     f"initial orders, against matched neutral-hold targets; no fit; consumed Oct 24–27; {v['forks']:,} forks, {v['ok']:,} ok, excluded {excl}; as of "
     f"{v['asof'][:16].replace('T', ' ')}Z; checks all_true = {str(v['all_true']).lower()}). {stage('entry', 'Entry (would_WAIT)')}. {stage('first_review', 'First review (PULL)')}. "
     + (f"Entry → review: V {ca['mean_V_entry']:+.2f} → {ca['mean_V_review']:+.2f} c, flip share {ca['flip_share']:.3f}. " if ca else '')
     + f"Restates P95 (review 73f6cfa6 item 1): suppressing every discretionary PULL of the forced initial generation loses on these forced forks; that does not show "
     f"the first pull was necessary. | {reads} |\n")
H = (f"## {NOW} (session aea5bfb9) — ENTRY-RETENTION-DISCRIMINATION-v1: PI0'S FORECASTS AT ENTRY / FIRST REVIEW / FIRST PULL AGAINST MATCHED TARGETS; CHECKS "
     f"{'ALL TRUE' if v['all_true'] else 'NOT ALL TRUE'}; READINGS {', '.join(v['holds']).upper()} (review 67a0d9f7's next priority) "
     f"(USER: \"... continue the implementing and next priority loop\")\n"
     f"- [Receipt](reviews/entry_retention_20261121/report.md).\n"
     f"  - Commits: freeze `2ac6e084` (before any measurement), results in this commit.\n"
     f"  - One diagnostic; zero fitted candidates.\n"
     f"- **Review 67a0d9f7's checklist (all nonblocking, folded in):**\n"
     f"  - 1: the exposure result is restated (every discretionary pull suppressed; not the first pull's necessity). Shadow admission, first review and first "
     f"pull are reported separately.\n"
     f"  - 2: forecast components are measured against matched targets, with full 107-input vectors.\n"
     f"  - 3: END / mark / settings and the callback and probe denominators are in check B, with wrong-END and missing-callback falsifiers.\n"
     f"  - 4: stage capture runs in fn (actual callbacks only).\n"
     f"  - 5: the END value is named end_value, and the fill-time execution edge is added.\n"
     f"- **Codex churn note `1b6e7799`:** its request (full 107-input vectors and moment / value components at submission and first review) is implemented here, "
     f"at entry, the first review and the first pull.\n"
     f"- **Readings** (FINDINGS P96):\n"
     f"  - {stage('entry', 'Entry (would_WAIT)')}.\n"
     f"  - {stage('first_review', 'First review (PULL)')}.\n"
     + ''.join(f"  - Declared reading `{k}`: {v['sentences'][k]}\n" for k in v['holds'])
     + f"- **Cost:** {v['core_hours']:.3f} core-hours of the 1 ceiling; peak {v['peak_mib']} MiB.\n"
     f"- **Next measurement:** the combined review (one request), which names the next priority.\n\n")
St = (f"entry_retention_20261121:\n"
      f"  status: done_entry_retention_discrimination_v1_no_fit\n"
      f"  asof: \"{NOW}\"\n"
      f"  receipt: predictfun/reviews/entry_retention_20261121/report.md\n"
      f"  commits: {{freeze: 2ac6e084}}\n"
      f"  checks: {{all_true: {str(v['all_true']).lower()}, falsifier_ok: {str(v['falsifier_ok']).lower()}}}\n"
      f"  forks: {{manifest: {v['forks']}, ok: {v['ok']}}}\n"
      + ''.join(f"  {nm}: {{selected: {sel[nm]['n_sel']}, eligible: {sel[nm]['n']}, percentile: {sel[nm]['percentile']:.4f}, thin: {str(sel[nm]['thin']).lower()}, "
                f"discriminates: {str(P['discriminates'][nm]).lower()}, value_fails: {str(P['value_fails'][nm]).lower()}}}\n" for nm in ('entry', 'first_review') if sel[nm]['percentile'] is not None)
      + f"  flip_share: {P['flip_share']}\n"
      f"  readings: {{{', '.join(f'{k}: {str(x).lower()}' for k, x in P['readings'].items())}}}\n"
      f"  cost_core_hours: {v['core_hours']:.3f}\n"
      f"  next: combined_review_names_the_next_priority\n")
for n, t in (('F_retention.md', F), ('H_retention.md', H), ('S_retention.yml', St)):
    (S / 'edits' / n).write_text(t)
print(F); print(H); print(St)
