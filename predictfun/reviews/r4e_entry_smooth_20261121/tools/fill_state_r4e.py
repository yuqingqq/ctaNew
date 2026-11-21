"""The R4E state hunks: the FINDINGS P93 row and P92's corrected row (review 371aab56 item 2), the HANDOFF section and the STATUS block, from
fill_r4e.py's computed values (r4e_values.json). Sentences are chosen by computed predicates; the HANDOFF time comes from `date` (argv[1]).
Writes edits/F_r4e.md, edits/F_p92_fix.md, edits/H_r4e.md, edits/S_r4e.yml and prints them."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

S = Path('/tmp/claude-1001/-home-yuqing-ctaNew/aea5bfb9-4ea4-41d0-bf73-bb98d818e3d4/scratchpad'); NOW = sys.argv[1]
v = json.loads((S / 'r4e_values.json').read_text()); C_ = pd.DataFrame(v['cells_table']); rep = v['replay']; days = sorted(rep)
ps, prr = v['pred_SMOOTH'], v['pred_RICH']
rs = (float(C_.ratio_smooth.min()), float(C_.ratio_smooth.max())); rr = (float(C_.ratio_rich.min()), float(C_.ratio_rich.max()))
no_skill = ps['demonstrated'] == 0
fewer_posts = all(rep[d]['generations']['SMOOTH'] < rep[d]['generations']['ENTRY_ONLY'] for d in days)
above_entry = [rep[d]['smooth_vs_entry'][0] > 0 for d in days if rep[d]['smooth_vs_entry']]
below_pi0 = [rep[d]['smooth_vs_pi0'][0] < 0 for d in days if rep[d]['smooth_vs_pi0']]
controls_exact = all(n == rep[d]['worlds'] for d in days for n in rep[d]['r3_match'].values())
fmt = lambda x: f'{x:+,.0f}'
rline = '; '.join(f"{d[4:6]}-{d[6:]}: SMOOTH − ENTRY_ONLY {fmt(rep[d]['smooth_vs_entry'][0])} c ({rep[d]['smooth_vs_entry'][1]} above / {rep[d]['smooth_vs_entry'][2]} below), "
                  f"SMOOTH − PI0 {fmt(rep[d]['smooth_vs_pi0'][0])} c, ENTRY_ONLY − PI0 {fmt(rep[d]['entry_vs_pi0'][0])} c; generations {rep[d]['generations']['SMOOTH']:,} "
                  f"against {rep[d]['generations']['ENTRY_ONLY']:,} ({rep[d]['worlds']} worlds)" for d in days)
reached = v['reached']; rb = sum(1 for x in reached if x['smooth_below_rich']); rz = sum(1 for x in reached if x['smooth_below_zero'])
impl = ' '.join((["No demonstrated entry skill: the declared conjunction (selected POST value > 0, a PASS screen, above the draws' p95) holds in no day × side cell."] if no_skill else
                 [f"The declared entry demonstration holds in {ps['demonstrated']} of {ps['cells']} day × side cells."])
                + ([f"The smoother package moves the POST error toward the zero baseline (MSE / zero {rs[0]:.3f}–{rs[1]:.3f} against RICH's {rr[0]:.3f}–{rr[1]:.3f}) and posts less, without positive selected value."]
                   if v['below_rich'] == v['cells'] and fewer_posts and ps['post_value_positive'] == 0 else [])
                + ['A further entry candidate is a new declared one (multiplicity); two consumed days carry no interval.'])
F = (f"| P93 | R4E ADV2-ENTRY-SMOOTH-v1 (one declared smoother POST package: min_data_in_leaf 1,000, 50 rounds; PI0 retention; fitted on consumed Oct 24–25; "
     f"evaluated on R3's {v['identity']['rows']:,} Oct 26–27 POST rows, equal to R3's in keys / labels / weights / RICH predictions; the 384-world own-state "
     f"replay; as of {v['asof'][:16].replace('T', ' ')}Z; checks all_true = {str(v['all_true']).lower()}). **Prediction:** SMOOTH MSE / zero "
     f"{rs[0]:.3f}–{rs[1]:.3f} (below zero in {v['below_zero']} of {v['cells']}, below RICH in {v['below_rich']}; RICH {rr[0]:.3f}–{rr[1]:.3f}). **Declared entry "
     f"predicates** (per day × side): selected POST value > 0 in {ps['post_value_positive']}, screen PASS in {ps['screen_pass']}, above the draw mean in "
     f"{ps['above_draw_mean']} and p95 in {ps['above_draw_p95']}; the conjunction in {ps['demonstrated']} of {ps['cells']} ({ps['admissions']:,} admissions against "
     f"RICH's {prr['admissions']:,}). **Local value:** above never posting in {v['value_above_never']} of {v['cells']}, above RICH in {v['value_above_rich']}. "
     f"**Replay (diagnostic):** {rline}. The re-run controls equal R3's episodes in every world: {str(controls_exact).lower()}. **Reached SMOOTH entry states** "
     f"(both heads on identical inputs): SMOOTH below RICH on {rb} and below zero on {rz} of {len(reached)} days. Cost {v['round_core_hours']:.2f} core-hours, peak "
     f"{v['round_peak']:,} MiB. | {impl} |\n")
head_find = subprocess.run(['git', '-C', '/home/yuqing/ctaNew', 'show', 'HEAD:predictfun/FINDINGS.md'], capture_output=True, text=True).stdout
p92 = next(ln for ln in head_find.splitlines() if ln.startswith('| P92 |'))
old = 'on those, the mean prediction is +1.43 c at the post and -1.78 c at the review'
assert p92.count(old) == 1, 'P92 phrase not found'
p92_fix = p92.replace(old, 'on those, the POST advantage at the post averages +1.43 c and the KEEP advantage at the review -1.78 c: two different targets, not a '
                           'same-target reversal (review 371aab56 item 2; the hybrid losses are not an additive decomposition)') + '\n'
tags = [t for t, c in (('NO DEMONSTRATED ENTRY SKILL', no_skill), ('FEWER POSTS THAN ENTRY_ONLY', fewer_posts),
                       ('SMOOTH ABOVE ENTRY_ONLY IN THE REPLAY ON BOTH DAYS', all(above_entry) and len(above_entry) == len(days)),
                       ('SMOOTH BELOW PI0 ON BOTH DAYS', all(below_pi0) and len(below_pi0) == len(days))) if c]
H = (f"## {NOW} (session aea5bfb9) — R4E: ADV2-ENTRY-SMOOTH-v1, ONE DECLARED SMOOTHER ENTRY CANDIDATE ON THE EXISTING CONSUMED OCT 26–27 LABELS AND THE "
     f"384-WORLD REPLAY; CHECKS {'ALL TRUE' if v['all_true'] else 'NOT ALL TRUE'}{'; ' + ', '.join(tags) if tags else ''} (review 371aab56's next priority) "
     f"(USER: \"... continue the implementing and next priority loop\"; \"As long as the results are reliable, we can accelerate the process\")\n"
     f"- [Receipt](reviews/r4e_entry_smooth_20261121/report.md).\n"
     f"  - Commits: freeze `602fdfc5`, model hashes `eaea1161` (before any evaluation), freeze amendment `5c2ec2ca` (the reuse check's exact CSV "
     f"reader, before any evaluation output), freeze 2b `c69ea5c6` (the replay summary v2 and the falsifier v2), results in this commit.\n"
     f"  - **Adaptive consumed-day development:** these days informed the design. No interval, no validation, no refit, no sweep.\n"
     f"- **Review 371aab56's checklist, all handled in this round:**\n"
     f"  - 1, entry selection: this experiment.\n"
     f"  - 2: the first-review readings name the two targets. FINDINGS P92's wording is corrected here, and hybrid losses are not summed.\n"
     f"  - 3: strata by generation / source / horizon / other side; both heads on identical reached states.\n"
     f"  - 4: full traces for every episode and reached pair, recomputed by checks F and G with the frozen label validator; the controls' "
     f"reproduction of R3 is read from R3's parts.\n"
     f"  - 5: calibration w_sum pooling, and R3's pooled table recomputed by weights.\n"
     f"  - Operational: part identity on reuse, complete membership in the summary, a separate launcher.\n"
     f"- **Readings** (FINDINGS P93): SMOOTH MSE / zero {rs[0]:.3f}–{rs[1]:.3f} (RICH {rr[0]:.3f}–{rr[1]:.3f}). The declared entry conjunction holds in "
     f"{ps['demonstrated']} of {ps['cells']} cells. Replay: {rline}.\n"
     f"- **Deviations:**\n"
     f"  - The first evaluation refused at the reuse check, because pandas' default CSV float parser is not exact. The reader was amended "
     f"before any output.\n"
     f"  - Replay mutation run 1 caught {v['run1_replay'][0]} / {v['run1_replay'][1]} (a harness replacement did not compile).\n"
     f"  - Two frozen writers failed after the run and are superseded in freeze 2b. The replay summary used a column the episodes keep in their "
     f"traces; v2 proved the four CSVs byte-identical. The falsifier looked for a fill in one world without fills; v2 searches every partition and "
     f"records each corruption as applied. The checks A–I themselves are unchanged.\n"
     f"- **Cost:** {v['round_core_hours']:.2f} core-hours of the 8 ceiling; peak {v['round_peak']:,} MiB.\n"
     f"- **Next measurement:** the combined review (one request), which names the next priority.\n\n")
St = (f"r4e_entry_smooth_20261121:\n"
      f"  status: done_r4e_adv2_entry_smooth_v1_existing_labels_and_bounded_replay\n"
      f"  diagnosis: {{no_demonstrated_entry_skill: {str(no_skill).lower()}, fewer_posts_than_entry_only: {str(fewer_posts).lower()}, controls_equal_r3: {str(controls_exact).lower()}}}\n"
      f"  asof: \"{NOW}\"\n"
      f"  receipt: predictfun/reviews/r4e_entry_smooth_20261121/report.md\n"
      f"  commits: {{freeze: 602fdfc5, model_hashes: eaea1161, freeze_amendment: 5c2ec2ca, freeze2b_summary_falsifier_v2: c69ea5c6}}\n"
      f"  candidate: {{name: ADV2-ENTRY-SMOOTH-v1, delta: min_data_in_leaf_1000, rounds: 50, retention: frozen_pi0, comparator: ENTRY_ONLY, multiplicity: one_more_primary_two_heads}}\n"
      f"  checks: {{all_true: {str(v['all_true']).lower()}, falsifier_ok: {str(v['falsifier_ok']).lower()}}}\n"
      f"  suites: {{entry: {v['suites']['entry'][0]}/{v['suites']['entry'][1]}, entry_eval: {v['suites']['entry_eval'][0]}/{v['suites']['entry_eval'][1]}, "
      f"entry_replay: {v['suites']['entry_replay'][0]}/{v['suites']['entry_replay'][1]}}}\n"
      f"  prediction: {{smooth_mse_over_zero: [{rs[0]:.4f}, {rs[1]:.4f}], rich_mse_over_zero: [{rr[0]:.4f}, {rr[1]:.4f}], below_zero: {v['below_zero']}/{v['cells']}, below_rich: {v['below_rich']}/{v['cells']}}}\n"
      f"  predicates_smooth: {{post_value_positive: {ps['post_value_positive']}, screen_pass: {ps['screen_pass']}, above_draw_mean: {ps['above_draw_mean']}, above_draw_p95: {ps['above_draw_p95']}, "
      f"demonstrated: {ps['demonstrated']}, admissions: {ps['admissions']}, rich_admissions: {prr['admissions']}}}\n"
      + ''.join(f"  replay_{d}: {{worlds: {rep[d]['worlds']}, smooth_minus_entry_only_c: {rep[d]['smooth_vs_entry'][0]:.1f}, smooth_minus_pi0_c: {rep[d]['smooth_vs_pi0'][0]:.1f}, "
                f"entry_only_minus_pi0_c: {rep[d]['entry_vs_pi0'][0]:.1f}}}\n" for d in days)
      + f"  reached: {{smooth_below_rich_days: {rb}, smooth_below_zero_days: {rz}, days: {len(reached)}}}\n"
      f"  cost: {{round_core_hours: {v['round_core_hours']:.2f}, peak_mib: {v['round_peak']}}}\n"
      f"  next: combined_review_names_the_next_priority\n")
(S / 'edits').mkdir(exist_ok=True)
for n, t in (('F_r4e.md', F), ('F_p92_fix.md', p92_fix), ('H_r4e.md', H), ('S_r4e.yml', St)):
    (S / 'edits' / n).write_text(t)
print(F); print(p92_fix[:400]); print(H); print(St)
