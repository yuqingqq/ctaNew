"""The ENTRY-EXPOSURE-v1 state hunks (FINDINGS P95, HANDOFF section, STATUS block) from fill_exposure.py's values (exposure_values.json). The
HANDOFF time comes from `date` (argv[1]). Writes edits/F_exposure.md, edits/H_exposure.md, edits/S_exposure.yml and prints them."""
import json
import sys
from pathlib import Path

S = Path('/tmp/claude-1001/-home-yuqing-ctaNew/aea5bfb9-4ea4-41d0-bf73-bb98d818e3d4/scratchpad'); NOW = sys.argv[1]
v = json.loads((S / 'exposure_values.json').read_text()); P = v['predicates']; d = P['days']
RC = Path('/home/yuqing/ctaNew/predictfun/reviews/entry_exposure_20261121'); J1 = json.loads((RC / 'checks.json').read_text()); J2 = json.loads((RC / 'checks_v2.json').read_text())
f1 = [k for k, x in sorted(J1['checks'].items()) if not x]; f2 = [k for k, x in sorted(J2['checks'].items()) if not x]; n_c = J1['readings']['C']['failing']
days = '; '.join(f"Oct {k[-2:]} {x['mean_D']:+.3f} c ({x['forks']} forks)" for k, x in sorted(d.items()))
excl = ', '.join(f'{n} {k}' for k, n in sorted(v['statuses'].items()) if k != 'ok') or 'none'
cost = (f"mean D_initial over them {P['mean_D_initial_additional']:+.3f} c, {P['share_D_initial_neg_additional']:.3f} negative"
        if P['mean_D_initial_additional'] is not None else 'none')
F = (f"| P95 | ENTRY-EXPOSURE-v1 (the same original-time neutral mandatory POST under WAIT_PI0 / POST_PI0 / POST_KEEP_INITIAL; consumed Oct 24–27, "
     f"4 hash-ranked worlds per UTC hour; {v['forks']:,} forks, {v['ok']:,} ok, excluded {excl}; as of {v['asof'][:16].replace('T', ' ')}Z; checks all_true = "
     f"{str(v['all_true']).lower()} under checks v3 after two in-band supersessions: freeze 2b `5ece0c98`, where the KEEP wrapper's override counter also "
     f"counted the causal probe's decide calls on {n_c} forks, so the override readings now come from the decision log; and freeze 2c `46a2eabc`, a NaN-aware "
     f"checker comparison. No value changed). PI0's stored pair reproduced on every evaluable fork: {str(v['checks'].get('B_reproduction')).lower()}. **Mean D_EXPOSURE = "
     f"R(KEEP) − R(POST_PI0), both sides per day:** {days}. P(initial fill) PI0 {P['fill_prob_PI0']:.4f} → KEEP {P['fill_prob_KEEP']:.4f}; "
     f"{P['additional_fill_forks']} additional-fill forks ({cost}). Σ D_initial {P['sum_D_initial']:+.1f} c, Σ D_rest {P['sum_D_rest']:+.1f} c. "
     f"**Diagnosis `{P['diagnosis']}`.** Supersedes P94's \"rare execution only\" wording: changing Y to V_initial alone would retain the observed sparsity "
     f"under PI0 (review 3d8bf5ad item 1). | {v['diagnosis_sentence']} |\n")
H = (f"## {NOW} (session aea5bfb9) — ENTRY-EXPOSURE-v1: THE SAME INITIAL POST UNDER PI0 VERSUS KEEPING IT; CHECKS "
     f"{'ALL TRUE' if v['all_true'] else 'NOT ALL TRUE'}; DIAGNOSIS {P['diagnosis'].upper()} (review 3d8bf5ad's next priority) "
     f"(USER: \"... continue the implementing and next priority loop\")\n"
     f"- [Receipt](reviews/entry_exposure_20261121/report.md).\n"
     f"  - Commits: freeze `6ca42d25` (before any branch ran); freeze 2b `5ece0c98` (override readings from the decision log, which the frozen check C "
     f"required); freeze 2c `46a2eabc` (a NaN-aware checker comparison); results in this commit. No value changed.\n"
     f"  - One continuation contrast enters the ledger. It is not a fitted candidate and not a hold timer.\n"
     f"- **Review 3d8bf5ad's checklist (all nonblocking, folded in):**\n"
     f"  - 1: the \"rare execution only\" wording is superseded: changing Y to V_initial alone would retain the observed sparsity under PI0.\n"
     f"  - 2: the `cancellation` flag is named an aggregate attenuation screen; the receipt shows the rowwise offset measure.\n"
     f"  - 3: the new checker reconciles every fork from its raw arms, with a compensating-component control and a wrong-but-valid gid control.\n"
     f"  - 4: core-hours use the recursive aggregation; the audit's top-5 share is labelled \"unweighted rows\".\n"
     f"- **Readings** (FINDINGS P95):\n"
     f"  - Mean D_EXPOSURE per day: {days}.\n"
     f"  - P(initial fill) PI0 {P['fill_prob_PI0']:.4f} → KEEP {P['fill_prob_KEEP']:.4f}; {P['additional_fill_forks']} additional-fill forks ({cost}).\n"
     f"  - Declared reading: {v['diagnosis_sentence']}\n"
     f"- **Cost:** {v['core_hours']:.3f} core-hours of the 2 ceiling; peak {v['peak_mib']} MiB.\n"
     f"- **Next measurement:** the combined review (one request), which names the next priority.\n\n")
St = (f"entry_exposure_20261121:\n"
      f"  status: done_entry_exposure_v1_keep_initial_vs_pi0\n"
      f"  asof: \"{NOW}\"\n"
      f"  receipt: predictfun/reviews/entry_exposure_20261121/report.md\n"
      f"  commits: {{freeze: 6ca42d25, freeze_2b_override_readings: 5ece0c98, freeze_2c_checker_nan: 46a2eabc}}\n"
      f"  checks: {{version: v3, all_true: {str(v['all_true']).lower()}, falsifier_ok: {str(v['falsifier_ok']).lower()}, v1_false: [{', '.join(f1)}], v2_false: [{', '.join(f2)}]}}\n"
      f"  forks: {{manifest: {v['forks']}, ok: {v['ok']}}}\n"
      f"  diagnosis: {P['diagnosis']}\n"
      + ''.join(f"  day_{k}: {{forks: {x['forks']}, mean_D: {x['mean_D']:.4f}, sign: {x['sign']}}}\n" for k, x in sorted(d.items()))
      + f"  pooled: {{fill_prob_PI0: {P['fill_prob_PI0']:.4f}, fill_prob_KEEP: {P['fill_prob_KEEP']:.4f}, additional_fill_forks: {P['additional_fill_forks']}, "
      f"more_execution: {str(P['more_execution']).lower()}, extra_fills_harmful: {str(P['extra_fills_harmful']).lower()}, initial_carries: {str(P['initial_carries']).lower()}}}\n"
      f"  cost_core_hours: {v['core_hours']:.3f}\n"
      f"  next: combined_review_names_the_next_priority\n")
for n, t in (('F_exposure.md', F), ('H_exposure.md', H), ('S_exposure.yml', St)):
    (S / 'edits' / n).write_text(t)
print(F); print(H); print(St)
