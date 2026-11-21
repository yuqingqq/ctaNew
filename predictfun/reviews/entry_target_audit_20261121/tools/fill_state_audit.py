"""The ENTRY-TARGET-AUDIT-v1 state hunks (FINDINGS P94, HANDOFF section, STATUS block) from fill_audit.py's values (audit_values.json). The
HANDOFF time comes from `date` (argv[1]). Writes edits/F_audit.md, edits/H_audit.md, edits/S_audit.yml and prints them."""
import json
import sys
from pathlib import Path

S = Path('/tmp/claude-1001/-home-yuqing-ctaNew/aea5bfb9-4ea4-41d0-bf73-bb98d818e3d4/scratchpad'); NOW = sys.argv[1]
v = json.loads((S / 'audit_values.json').read_text()); P = v['predicates']
fmt = lambda p: (f"meaningful share {p['share_meaningful']:.3f}, initial no-fill share {p['init_no_fill']:.3f}, Σw|Y| / Σw|V_initial| "
                 f"{(p['abs_Y'] / p['abs_v_initial']) if p['abs_v_initial'] else float('nan'):.3f}, diagnosis {p['diagnosis']}")
F = (f"| P94 | ENTRY-TARGET-AUDIT-v1 (the existing POST / WAIT labels decomposed; no fit, no new label, no replay; {v['rows']:,} POST rows of R2-F's Oct 24–25 "
     f"and R3's Oct 26–27 accepted datasets with every status, plus R4E's {v['reached_rows']:,} reached pairs; as of {v['asof'][:16].replace('T', ' ')}Z; checks "
     f"all_true = {str(v['all_true']).lower()}). Y = V_initial (the forced generation's fills at the END mark) + D_focal + D_opposite holds on every accounted row. "
     f"**Oct 24–25 (in-sample):** {fmt(P['train'])}. **Oct 26–27 (adaptive):** {fmt(P['eval'])}. The sets {'agree' if v['agree'] else 'disagree'}. "
     f"The frozen models' POST groups carry a higher mean V_initial than their WAIT groups in {v['rank']['SMOOTH'][0]} of {v['rank']['SMOOTH'][1]} cells (SMOOTH) "
     f"and {v['rank']['RICH'][0]} of {v['rank']['RICH'][1]} (RICH). | {v['recommendation']} |\n")
H = (f"## {NOW} (session aea5bfb9) — ENTRY-TARGET-AUDIT-v1: THE EXISTING POST / WAIT LABELS DECOMPOSED (NO FIT, NO NEW LABEL); CHECKS "
     f"{'ALL TRUE' if v['all_true'] else 'NOT ALL TRUE'}; DIAGNOSIS {P['eval']['diagnosis'].upper()} ON OCT 26–27, {P['train']['diagnosis'].upper()} ON OCT 24–25 "
     f"(review d6d5d55e's next priority) (USER: \"... continue the implementing and next priority loop\")\n"
     f"- [Receipt](reviews/entry_target_audit_20261121/report.md).\n"
     f"  - Commits: freeze `811fd7b3`, amendment `6157b1a0` (every output under its own name, before any output), results in this commit.\n"
     f"  - Tuning of the failed direct POST − WAIT regressors (R3 RICH, R4E SMOOTH) is stopped.\n"
     f"- **Review d6d5d55e's checklist:**\n"
     f"  - 1: the smoothing experiment is closed as failed, and this audit is the next step.\n"
     f"  - 2: R4E's first-review and predicates tables are renamed per leg and per population in this receipt.\n"
     f"  - 3: the replay checker is not reused here, so the review's verified workaround stands; a future replay checker adds END / settings / "
     f"fill-time / common-mark predicates.\n"
     f"- **Readings** (FINDINGS P94):\n"
     f"  - Oct 24–25: {fmt(P['train'])}.\n"
     f"  - Oct 26–27: {fmt(P['eval'])}.\n"
     f"  - Recommendation (declared rule): {v['recommendation']}\n"
     f"- **Cost:** {v['core_hours']:.3f} core-hours of the 1 ceiling.\n"
     f"- **Next measurement:** the combined review (one request), which names the next priority.\n\n")
St = (f"entry_target_audit_20261121:\n"
      f"  status: done_entry_target_audit_v1_existing_labels_no_fit\n"
      f"  asof: \"{NOW}\"\n"
      f"  receipt: predictfun/reviews/entry_target_audit_20261121/report.md\n"
      f"  commits: {{freeze: 811fd7b3, amendment_distinct_outputs: 6157b1a0}}\n"
      f"  checks: {{all_true: {str(v['all_true']).lower()}, falsifier_ok: {str(v['falsifier_ok']).lower()}}}\n"
      + ''.join(f"  {k}: {{share_meaningful: {p['share_meaningful']:.4f}, init_no_fill: {p['init_no_fill']:.4f}, abs_v_initial: {p['abs_v_initial']:.1f}, abs_Y: {p['abs_Y']:.1f}, "
                f"rare_execution: {str(p['rare_execution']).lower()}, cancellation: {str(p['cancellation']).lower()}, diagnosis: {p['diagnosis']}}}\n" for k, p in sorted(P.items()))
      + f"  sets_agree: {str(v['agree']).lower()}\n"
      f"  rows: {{post_rows: {v['rows']}, reached_pairs: {v['reached_rows']}}}\n"
      f"  cost_core_hours: {v['core_hours']:.3f}\n"
      f"  next: combined_review_names_the_next_priority\n")
for n, t in (('F_audit.md', F), ('H_audit.md', H), ('S_audit.yml', St)):
    (S / 'edits' / n).write_text(t)
print(F); print(H); print(St)
