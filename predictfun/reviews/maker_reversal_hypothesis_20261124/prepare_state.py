"""Additive state for the USER-directed hypothesis verification packet."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent
block='''
<!-- REVERSAL_HYPOTHESIS_20261124_CODEX -->
## 2026-11-24 — USER-directed hypothesis-first action verification

[Report](reviews/maker_reversal_hypothesis_20261124/report.md). USER asks to hypothesize actions from conditions, then independently verify, without a full controller rebuild. Do not infer a cancel timestamp from BTC and then use BTC agreement as proof. One frozen H1: favourable-to-adverse BTC reversal -> withdrawal of that side within 5s; inherited 5s/1bp signal, neutral market-second detection, D0/D1000, wrong-side and 500 timing-placebo controls. No tuned variant or population cancellation-rate claim.

43 reversals over 11,400 market-side seconds. Same fixed 88 signed endings: 40 conditional withdrawals / 41 exhausted / 7 unresolved. D0: 39 withdrawal windows evaluable, four timing-only matches vs three wrong-side; strict post-before-reversal matches zero at both delays. Frozen reading hypothesis_not_demonstrated, not evidence makers never react. Post-run independently declared lower-bound join: three matches remain possible (reversal inside posting window; two G2, one G4); one reversal precedes earliest posting and is not an already-resting reaction under those bounds. D1000 retains two possible plus that early case. Exhausted endings have one timing overlap: proximity alone does not prove pull. No private action or continuous rest certified.

All 704 case/profile rows, 43 events and 5,000 draw rows reconcile; seven real-copy corruptions refused. Source814f8fb5d12543c910355467a0c243b5afba57ab; post-run chronology declaration4014fcf5f. About .0010 coreh, task1.9143/2.00, line5.2250/5.30, .05 reserve; mt_hyp counted once. No network/new days/model/live work or push.

ONE next priority: independently tighten post-before-reversal for the three possible chains using saved book/fill constraints, retaining all40 candidate and41 exhausted control coverage. Never use the signal to force the timestamp; preserve grades and ambiguity, no threshold search if unsupported. <=.015 coreh with projection gate within current USER direction/caps, no new USER decision. This replaces the generic all40-window refinement with a specific hypothesis verification link.
<!-- /REVERSAL_HYPOTHESIS_20261124_CODEX -->

'''
status='''reversal_hypothesis_20261124_codex:
  report: predictfun/reviews/maker_reversal_hypothesis_20261124/report.md
  source_freeze: 814f8fb5d12543c910355467a0c243b5afba57ab
  scope: hypothesis_first_saved_data_action_verification
  hypothesis: adverse_BTC_reversal_then_withdrawal_within_5s
  consumed_days: [20261024, 20261025, 20261026, 20261027]
  signal_reversals: 43
  fixed_signed_endings: 88
  conditional_withdrawals: 40
  evaluable_withdrawals: 39
  timing_only_matches_D0: 4
  wrong_side_matches_D0: 3
  strict_post_before_reversal_matches: 0
  post_run_chronology: {possible_inside_post_window: 3, reversal_before_earliest_post: 1}
  reading: hypothesis_not_demonstrated
  private_action_identified: false
  population_cancel_probability_estimated: false
  independent_checks_pass: true
  next_priority: independent_post_before_reversal_constraints_for_three_possible_chains
  next_within_current_user_scope: true
  next_max_core_hours: 0.015
  task_core_hours: 1.9143
  line_core_hours: 5.2250
  task_cap: 2.00
  line_cap: 5.30
  standing_reserve: 0.05
'''
blocks={p:block for p in ['predictfun/HANDOFF.md','predictfun/FINDINGS.md']}
blocks['predictfun/STATUS.yml']=status
blocks['predictfun/reviews/maker_trace_20261122/PENDING.md']=block.replace('(reviews/','(../')
(P/'state_blocks.json').write_text(json.dumps(blocks,indent=2)+'\n')
