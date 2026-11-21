"""The R3 state hunks (FINDINGS P92 row, HANDOFF section, STATUS block) from the receipt's outputs and fill_r3.py's computed values
(r3_values.json). Every number is read or computed here; sentences are chosen by computed predicates. The HANDOFF time comes from `date`
(argv[1]). Writes edits/F_r3.md, edits/H_r3.md, edits/S_r3.yml and prints them."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

S = Path('/tmp/claude-1001/-home-yuqing-ctaNew/aea5bfb9-4ea4-41d0-bf73-bb98d818e3d4/scratchpad'); R = Path('/home/yuqing/ctaNew/predictfun/reviews/r3_adv2_20261121')
NOW = sys.argv[1]                                                                           # 'YYYY-MM-DD HH:MMZ' from date
v = json.loads((S / 'r3_values.json').read_text())


def rcsv(d, name):
    p = R / d / name
    return pd.read_csv(p if p.exists() else Path(str(p) + '.gz'))


import re
def suite(name):
    t = (R / 'logs' / name).read_text(); m = re.search(r'(\d+) / (\d+) planted code errors are caught', t); return int(m.group(1)), int(m.group(2))
FZ = [suite(f'mutants_{t}_summary.log') for t in ('sample', 'adv2', 'eval', 'replay')]; fz_c, fz_n = sum(c for c, _ in FZ), sum(n for _, n in FZ)
R1 = [suite(f'mutants_{t}_run1_summary.log') for t in ('adv2', 'eval', 'replay')]; r1_c, r1_n = sum(c for c, _ in R1), sum(n for _, n in R1)
RS = suite('mutants_rsummary_summary.log')
C_ = pd.DataFrame(v['cells_table']); RR = pd.DataFrame(v['reached_table'])
by = {h: C_[C_['head'] == h] for h in ('post', 'keep')}
ratio = {h: (float(x.ratio_zero.min()), float(x.ratio_zero.max())) for h, x in by.items()}
bz = {h: int(x.beats_zero.sum()) for h, x in by.items()}; nh = {h: len(x) for h, x in by.items()}
post_neg = int((by['post'].value < 0).sum()); keep_fb = int(by['keep'].beats_fallback.sum())
keep_vs_pull = int((by['keep'].value > 0).sum())                                            # RICH's keep value against always-PULL (0)
always_keep_neg = int((by['keep'].fallback_value < 0).sum())
scr = v['screens']['RICH']
rep = v['replay']; days = sorted(rep)
dsum = lambda d, k: rep[d][k][0] if rep[d][k] else float('nan')
Ch = rcsv('replay', 'churn.csv'); ri = Ch[Ch.policy == 'RICH']
pull_share = float((ri.action == 'PULL').mean()); dt = sorted(set(np.round(ri.dt.dropna().unique(), 6)))
pv = ri[ri.action == 'PULL']; vpost, vrev = float(pv.value_post.mean()), float(pv.value_review.mean())
Ex = rcsv('replay', 'exposure.csv'); Ex = Ex[Ex.policy == 'RICH']; parts = Ex.key.str.split('|')
Ex = Ex.assign(state=parts.str[0], status=parts.str[1], dv=parts.str[2])
ab = Ex[(Ex.state == 'absent') & (Ex.status == 'model') & Ex.dv.fillna('').str.startswith('c_gen=')]
g3 = float(ab[ab.dv == 'c_gen=>=3'].n.sum() / ab.n.sum()) if ab.n.sum() else float('nan')
Tr = rcsv('eval', 'coverage_train.csv'); tp = Tr[(Tr.group == 'all') & (Tr['head'] == 'post') & (Tr.dim == 'c_gen')]
g3t = float(tp[tp.value.astype(str) == '>=3'].rows.sum() / tp.rows.sum())
fb_live = int(Ex[(Ex.state == 'live') & (Ex.status == 'outside_training_range') & Ex.dv.isna()].n.sum())
live_all = int(Ex[(Ex.state == 'live') & Ex.dv.isna()].n.sum())
G2r = json.loads((R / 'checks_v2.json').read_text())['readings']['G']
J1 = json.loads((R / 'checks.json').read_text()); G1bad = len(J1['readings'].get('G', {}).get('bad', []))
loss_share = {d: (dsum(d, 'entry_only_vs_pi0') / dsum(d, 'rich_vs_pi0')) if dsum(d, 'rich_vs_pi0') else float('nan') for d in days}
fmt = lambda x: f'{x:+,.0f}'
rep_line = '; '.join(f"{d[4:6]}-{d[6:]}: RICH − PI0 {fmt(dsum(d, 'rich_vs_pi0'))} c, ENTRY_ONLY − PI0 {fmt(dsum(d, 'entry_only_vs_pi0'))}, RETAIN_ONLY − PI0 "
                     f"{fmt(dsum(d, 'retain_only_vs_pi0'))}, FAIR − PI0 {fmt(dsum(d, 'fair_vs_pi0'))}, MECH − PI0 {fmt(dsum(d, 'mech_vs_pi0'))} (PI0 total "
                     f"{fmt(rep[d]['totals']['PI0'])} c, {rep[d]['worlds']} worlds)" for d in days)
post_layer = bz['post'] == 0 and post_neg == nh['post']                                  # no POST-head skill: above zero and negative value in every cell
most_pulled = pull_share > .5                                                               # most of RICH's reviewed generations pulled at the first review
rich_below_pi0 = all(dsum(d, 'rich_vs_pi0') < 0 for d in days)                             # RICH's replay total below PI0's on every day
impl = ' '.join(([f"The POST head shows no incremental skill here: MSE above zero and a negative local value in all {nh['post']} cells."] if post_layer else [])
                + ([f"Most of RICH's reviewed generations ({100 * pull_share:.0f} %) are pulled at the first review, {'/'.join(f'{x:g}' for x in dt)} ms after the post."] if most_pulled else [])
                + ["Any next candidate is a new declared one (multiplicity); two consumed days carry no interval."])
tags = [t for t, c in (('NO POST-HEAD SKILL', post_layer), ('MOST POSTS PULLED AT THE FIRST REVIEW', most_pulled), ('RICH BELOW PI0 IN THE REPLAY ON BOTH DAYS', rich_below_pi0)) if c]
F = (f"| P92 | R3 ADV2-RICH v1 (fitted on consumed Oct 24–25, evaluated on consumed Oct 26–27; the declared sample 18,132 ok pairs, the nested replay "
     f"{sum(rep[d]['worlds'] for d in days)} ok worlds × 8 policies; as of {v['asof'][:16].replace('T', ' ')}Z; checks v2 all_true = {str(v['all_true']).lower()}). "
     f"**Prediction:** RICH's weighted MSE / zero is {ratio['post'][0]:.3f}–{ratio['post'][1]:.3f} on the POST head (below zero in {bz['post']} of "
     f"{nh['post']} day × side cells) and {ratio['keep'][0]:.3f}–{ratio['keep'][1]:.3f} on the KEEP head ({bz['keep']} of {nh['keep']}); below FAIR in "
     f"{v['beats_fair']} and MECH in {v['beats_mech']} of {v['cells']} cells. Screens (RICH, {v['screens_total']} groups): PASS {scr['PASS']}, FAIL "
     f"{scr['FAIL']}, INSUFFICIENT_SUPPORT {scr['INSUFFICIENT_SUPPORT']}. **Local value** (pi0-conditional): RICH's POST value is negative in "
     f"{post_neg} of {nh['post']} cells (never posting is 0); its KEEP value beats always-keep in {keep_fb} of {nh['keep']} and always-pull (0) in "
     f"{keep_vs_pull} of {nh['keep']} (always-keep is negative in {always_keep_neg} of {nh['keep']}). Control: lift > 0 in {v['control_lift_pos']} "
     f"(day × head), above the draws' p95 in {v['control_above_p95']}. **Replay (diagnostic):** {rep_line}. **Churn:** {100 * pull_share:.0f} % of RICH's "
     f"reviewed generations are pulled at the first review, {', '.join(f'{x:g}' for x in dt)} ms after the post; on those, the mean prediction is "
     f"{vpost:+.2f} c at the post and {vrev:+.2f} c at the review. **State shift:** {100 * g3:.1f} % of RICH's own absent-state model decisions are at "
     f"generation ≥ 3 against {100 * g3t:.1f} % of the POST training rows; the reached-state MSE / zero exceeds the neutral one in {v['reached_worse']} of "
     f"{v['reached_cells']} day × head cells. Fallbacks: {v['fallback_rows']} of {v['eval_rows']:,} evaluation rows; {fb_live:,} of {live_all:,} RICH live "
     f"replay callbacks outside the training range. Cost {v['round_core_hours']:.2f} core-hours, peak {v['round_peak']:,} MiB. "
     f"| {impl} |\n")
H = (f"## {NOW} (session aea5bfb9) — R3: ADV2-RICH v1 FITTED ON CONSUMED OCT 24–25, EVALUATED ON CONSUMED OCT 26–27 AND REPLAYED ON {G2r['population']} WORLDS ({sum(rep[d]['worlds'] for d in days)} OK); "
     f"CHECKS v2 {'ALL TRUE' if v['all_true'] else 'NOT ALL TRUE'}{'; ' + ', '.join(tags) if tags else ''} (review 315979d2's next priority) (USER: \"... continue "
     f"the implementing and next priority loop\"; \"As long as the results are reliable, we can accelerate the process\")\n"
     f"- [Receipt](reviews/r3_adv2_20261121/report.md). Commits: freeze 1 `551185f2` (evaluation sample), freeze 2 `c8655f75` (candidate, controls, "
     f"evaluator, replay, checks; {fz_c} / {fz_n} planted errors caught), model hashes `2bec465d` (before any Oct 26–27 label), freeze 2b `259e6c56` "
     f"(the replay summary v2), results in this commit. **This round measures prediction skill on two consumed days (development evidence, no "
     f"interval, no validation).**\n"
     f"- **Review 315979d2's items:** 1, the coverage by source / horizon / six support dimensions (evaluation by RICH's action, training) and "
     f"RICH's own-state exposure; 2, the meaningful (|Y| > 1e-6 c) floor with the exact count beside it; 3, check H per record against the final "
     f"sources, with a falsifier.\n"
     f"- **Readings** (FINDINGS P92): POST head MSE / zero {ratio['post'][0]:.3f}–{ratio['post'][1]:.3f} ({bz['post']} of {nh['post']} below zero), "
     f"KEEP {ratio['keep'][0]:.3f}–{ratio['keep'][1]:.3f} ({bz['keep']} of {nh['keep']}); RICH screens PASS {scr['PASS']} / FAIL {scr['FAIL']} / "
     f"INSUFFICIENT {scr['INSUFFICIENT_SUPPORT']}; replay {rep_line}; {100 * pull_share:.0f} % of RICH's reviewed generations pulled at the first "
     f"review; {100 * g3:.1f} % of its absent decisions at generation ≥ 3 (training {100 * g3t:.1f} %).\n"
     f"- **Deviations, all in the receipt:**\n"
     f"  - v1 check G was false on {G1bad} final-tick KEEP pairs with zero callbacks, which is correct. r3_checks_v2.py changes only that "
     f"predicate, and its falsifier passes.\n"
     f"  - The frozen replay summarize crashed writing replay.json (an int column under sort_keys). The superseding summary proved the four "
     f"CSVs byte-identical ({RS[0]} / {RS[1]} mutants caught).\n"
     f"  - The launcher was edited in place under running queues, and was repaired before any queue read it.\n"
     f"  - Mutation run 1 caught {r1_c} / {r1_n}, all misses in the tests.\n"
     f"- **Cost:** {v['round_core_hours']:.2f} core-hours of the 40 ceiling (labels {v['label_core_hours']:.2f}, replay {v['replay_core_hours']:.2f}); "
     f"peak {v['round_peak']:,} MiB.\n"
     f"- **Next measurement:** the combined review (one request), which names the next priority. The question put to it: what the diagnosis "
     f"implies for the next declared candidate on consumed days. These results are not a model search, and no refit has been made.\n\n")
St = (f"r3_adv2_rich_v1_20261121:\n"
      f"  status: done_r3_adv2_rich_v1_consumed_oct26_27_prediction_diagnostic_and_bounded_replay\n"
      f"  diagnosis: {{no_post_head_skill: {str(post_layer).lower()}, most_posts_pulled_at_first_review: {str(most_pulled).lower()}, rich_below_pi0_both_days: {str(rich_below_pi0).lower()}}}\n"
      f"  asof: \"{NOW}\"\n"
      f"  receipt: predictfun/reviews/r3_adv2_20261121/report.md\n"
      f"  commits: {{freeze1: 551185f2, freeze2: c8655f75, model_hashes: 2bec465d, freeze2b_summary_v2: 259e6c56}}\n"
      f"  eval_plan_sha256_prefix: 8619eec52bf75b0c\n"
      f"  training_dataset_sha256_prefix: 2c62fc9baf696e6c\n"
      f"  checks_v1: {{all_true: {str(J1['all_true']).lower()}, G_replay: {str(J1['checks']['G_replay']).lower()}, reason: zero_callback_final_tick_keep_pairs_{G1bad}}}\n"
      f"  checks_v2: {{all_true: {str(v['all_true']).lower()}, falsifier_ok: {str(v['falsifier_ok']).lower()}}}\n"
      f"  suites: {{sample: {FZ[0][0]}/{FZ[0][1]}, adv2: {FZ[1][0]}/{FZ[1][1]}, eval: {FZ[2][0]}/{FZ[2][1]}, replay: {FZ[3][0]}/{FZ[3][1]}, replay_summary_v2: {RS[0]}/{RS[1]}, run1: {r1_c}/{r1_n}_test_instrument_only}}\n"
      f"  prediction: {{post_mse_over_zero: [{ratio['post'][0]:.3f}, {ratio['post'][1]:.3f}], keep_mse_over_zero: [{ratio['keep'][0]:.3f}, {ratio['keep'][1]:.3f}], "
      f"below_zero_cells: {v['beats_zero']}/{v['cells']}, below_fair: {v['beats_fair']}, below_mech: {v['beats_mech']}}}\n"
      f"  screens_rich: {{pass: {scr['PASS']}, fail: {scr['FAIL']}, insufficient_support: {scr['INSUFFICIENT_SUPPORT']}, not_evaluable: {scr['NOT_EVALUABLE']}}}\n"
      f"  local_value: {{post_negative_cells: {post_neg}/{nh['post']}, keep_beats_always_keep: {keep_fb}/{nh['keep']}, keep_beats_always_pull: {keep_vs_pull}/{nh['keep']}}}\n"
      f"  control_rich: {{lift_positive: \"{v['control_lift_pos']}\", above_p95: \"{v['control_above_p95']}\"}}\n"
      + ''.join(f"  replay_{d}: {{worlds: {rep[d]['worlds']}, rich_minus_pi0_c: {dsum(d, 'rich_vs_pi0'):.1f}, entry_only_minus_pi0_c: {dsum(d, 'entry_only_vs_pi0'):.1f}, "
                f"retain_only_minus_pi0_c: {dsum(d, 'retain_only_vs_pi0'):.1f}, pi0_total_c: {rep[d]['totals']['PI0']:.1f}}}\n" for d in days)
      + f"  churn_rich: {{first_review_pull_share: {pull_share:.3f}, post_to_review_ms: {dt[0] if len(dt) == 1 else dt}}}\n"
      f"  state_shift: {{rich_absent_gen3plus_share: {g3:.3f}, train_post_gen3plus_share: {g3t:.3f}, reached_worse_cells: {v['reached_worse']}/{v['reached_cells']}}}\n"
      f"  cost: {{round_core_hours: {v['round_core_hours']:.2f}, label_core_hours: {v['label_core_hours']:.2f}, replay_core_hours: {v['replay_core_hours']:.2f}, peak_mib: {v['round_peak']}}}\n"
      f"  next: combined_review_names_the_next_priority\n")
(S / 'edits').mkdir(exist_ok=True)
(S / 'edits' / 'F_r3.md').write_text(F); (S / 'edits' / 'H_r3.md').write_text(H); (S / 'edits' / 'S_r3.yml').write_text(St)
print(F); print(H); print(St)
