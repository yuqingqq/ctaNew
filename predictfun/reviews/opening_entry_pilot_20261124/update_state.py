"""Stage this receipt's state additions without staging others' dirty hunks."""
import json,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[2]
c=json.loads((H/'checks.json').read_text());budget=json.loads((H/'resources.json').read_text())
subprocess.run(['git','diff','--cached','--quiet'],cwd=R,check=True)
lead='''<!-- OPENING_ENTRY_PILOT_20261124_CODEX -->
## 2026-11-24 — USER-corrected opening-only entry replay

[Receipt](reviews/opening_entry_pilot_20261124/report.md), freeze b1898f0b4. This supersedes all-entry filtering as the active task: USER only wants market-open entries changed. Gate until and through the first actual submission tick; every subsequent tick permanently uses the original entry policy. No rearm after fill/cancel/refusal. Same inventory correction in baseline and candidate; frozen historical source untouched.

Three complete consumed markets checked; disabled wrapper matches the saved fixed baseline;23577 later slot callbacks independently reproduce the original sizing/pressure decisions. Four scope corruptions refused. Full-market USD default -> opening-only:3702960 +4.28 -> +5.41;3789484 -17.40 -> +9.73;3877472 -136.46 -> -136.46. Pooled -149.58 -> -121.32, still losing. The earlier all-entry +21.27 is NOT this result.

3789484: initial bid batch550ms; original entries restored600ms; first-minute value -22.08 -> +2.08; full-market filled shares3859.7 ->3977.7 and both-side grid participation43.2% ->41.1%. Thus this narrower change helps the discussed example without the old all-entry suppression. Selected examples only; no validation/general entry-skill claim. Other first-batch times29.05s and0.85s are signal-triggered, not fixed timers.

ONE next priority: saved-ledger decomposition of the remaining losing market's first completed cycles and later turnover, comparing opening-only with fixed default. Keep opening scope fixed; no automatic all-entry filtering/threshold tuning/new replay. Packet upper163.99/180 CPU seconds, unchanged task2.05/line5.35 caps and .05h reserve. Other sessions' working changes preserved; no push/live use.
<!-- /OPENING_ENTRY_PILOT_20261124_CODEX -->

'''
finding='''<!-- OPENING_ENTRY_PILOT_20261124_CODEX -->
## 2026-11-24 — opening-only scope correction materially changes the answer

[Opening-only receipt](reviews/opening_entry_pilot_20261124/report.md). USER intended only the initial quote batch; assistant's prior all-entry gate was overbroad. New freeze b1898f0b4 gates only through first submission tick, then permanently delegates original entries.23577 later callbacks checked; disabled parity and accounting pass; four scope corruptions refused. Prior receipt remains valid for its different estimand, not as the answer to the USER's opening-only question.

Corrected-inventory baseline -149.58USD pooled -> opening-only -121.32USD, unlike the all-entry +21.27.3789484 improves -17.40 -> +9.73 full-market and -22.08 -> +2.08 first minute; filled shares slightly increase and both-side participation remains close.3702960 improves+1.14;3877472 value unchanged. Opening decisions help the named example but do not rescue the whole three-market policy. No new data, fit or validation; resources and next priority in HANDOFF.
<!-- /OPENING_ENTRY_PILOT_20261124_CODEX -->

'''
state=f'''opening_entry_pilot_20261124_codex:
  report: predictfun/reviews/opening_entry_pilot_20261124/report.md
  freeze: b1898f0b4
  stage: opening_only_replay_checked_still_pooled_negative
  scope: first_actual_submission_tick_inclusive_then_original_entries_forever
  later_admission_filter_active: false
  markets: [3702960, 3789484, 3877472]
  later_slot_callbacks_verified: {c['later_slot_calls_verified']}
  scope_corruptions_refused: 4
  disabled_full_market_parity: 1
  reused_fixed_baselines: 3
  baseline_total_usd: {c['pooled_baseline_c']/100}
  opening_only_total_usd: {c['pooled_candidate_c']/100}
  better_markets: {c['better_markets']}
  positive_markets: {c['positive_markets']}
  validation: false
  next_priority: saved_ledger_remaining_losing_cycles_without_scope_expansion
  packet_upper_cpu_s: {budget['packet_upper_cpu_s']}
  packet_cap_cpu_s: 180
  task_cap_coreh: 2.05
  line_cap_coreh: 5.35
  standing_reserve_coreh: 0.05
'''
for rel,block in [('predictfun/HANDOFF.md',lead),('predictfun/FINDINGS.md',finding),('predictfun/STATUS.yml',state)]:
 p=R/rel;working=p.read_text();base=subprocess.check_output(['git','show','HEAD:'+rel],cwd=R).decode()
 assert 'OPENING_ENTRY_PILOT_20261124_CODEX' not in working and 'opening_entry_pilot_20261124_codex:' not in working
 p.write_text(block+working)
 oid=subprocess.check_output(['git','hash-object','-w','--stdin'],cwd=R,input=(block+base).encode()).decode().strip()
 subprocess.run(['git','update-index','--cacheinfo','100644',oid,rel],cwd=R,check=True)
print('New blocks staged; existing unstaged changes preserved')
