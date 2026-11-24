"""Prepend and stage ONLY this packet's state blocks; preserve other dirty hunks."""
from pathlib import Path
import subprocess
R=Path('/home/yuqing/ctaNew')
subprocess.run(['git','diff','--cached','--quiet'],cwd=R,check=True)
blocks={
'predictfun/HANDOFF.md':'''<!-- ENTRY_GATE_PILOT_20261124_CODEX -->
## 2026-11-24 — USER-directed directional entry replay completed

[Receipt](reviews/entry_gate_pilot_20261124/report.md), freeze fa9d035d2. Latest USER asks to set entry rule and replay; used only remaining180-CPU-second dashboard allowance, no cap increase/new days. Three fixed consumed markets, ORIGINAL / FIXED_DEFAULT / DIRECTIONAL_ENTRY, nine complete arms. Original tape and execution parity3/3; independent result checks9/9 and seven corruption refusals. Frozen strategy files unchanged; experiment wrapper corrects the sign-crossing hysteresis in both comparison arms.

One unfitted rule gates new risk on the accepted opposite-side Binance q5 pressure (favourable direction), otherwise permits only unreserved reduction of client-known inventory. Live cancellation rules unchanged; no timer, no new thresholds or model. Full END USD fixed-default -> entry:3702960 +4.28 -> -2.99;3789484 -17.40 -> +31.92;3877472 -136.46 -> -7.66. Pooled -149.58 -> +21.27, completed-turnover delta +146.47 and residual +24.39. Filled shares -79.6%; only1/3 candidate markets positive. Selected examples, no validation/selection-beyond-activity claim.

Opening3789484: first quote550ms/bid only; first-minute value -22.08 -> +0.05, only12 matched shares. Avoids the cheap initial asks but does not reproduce maker3's turnover. Old inventory bug accidentally suppressed losing trading; the fix's effect is separated from entry.

ONE next priority: freeze an activity-matched entry control before tuning this rule, comparing complete turnover and residual on the same markets. No automatic follow-on replay: it needs a bounded resource plan. Packet upper139.43/180 CPU seconds including publication reserve; task/line caps2.05/5.35 and .05h standing reserve unchanged. No push/live use.
<!-- /ENTRY_GATE_PILOT_20261124_CODEX -->

''',
'predictfun/FINDINGS.md':'''<!-- ENTRY_GATE_PILOT_20261124_CODEX -->
## 2026-11-24 — entry gate helps two examples but largely suppresses activity

[Frozen three-market replay and receipt](reviews/entry_gate_pilot_20261124/report.md), fa9d035d2. Both comparison arms share the inventory sign-crossing correction; original saved ledgers reproduce exactly. An unfitted favourable-Binance admission gate gives +21.27USD pooled versus -149.58 fixed default, but only1/3 markets positive and filled shares79.6% lower. Completed FIFO delta+146.47USD, residual delta+24.39. No activity-matched null, so not a demonstrated signal-selection or validation result.

Our discussed3789484 opening no longer posts cheap asks at0ms: first bid at550ms; first-minute value+0.05 instead of-22.08, with only12 matched shares. Maker3's larger profitable early turnover remains unexplained by this rule. The original broken inventory restriction accidentally limited losing trading, especially3877472; do not preserve the bug because its numeric P&L was higher. Full data/code/identity checks and seven planted corruptions pass. No frozen implementation edited; resource and ownership details in HANDOFF.
<!-- /ENTRY_GATE_PILOT_20261124_CODEX -->

''',
'predictfun/STATUS.yml':'''directional_entry_pilot_20261124_codex:
  report: predictfun/reviews/entry_gate_pilot_20261124/report.md
  freeze: fa9d035d2
  stage: three_consumed_market_replay_checked_mixed_activity_confounded
  markets: [3702960, 3789484, 3877472]
  arms_checked: 9
  saved_original_parity: 3
  corruptions_refused: 7
  new_candidates: 1
  historical_strategy_files_changed: false
  inventory_fix_shared_by_comparison_arms: true
  entry_delta_usd: 170.8524795646827
  fixed_default_total_usd: -149.58122334849957
  candidate_total_usd: 21.27125621618313
  candidate_positive_markets: 1
  lower_filled_volume_fraction_approx: 0.796
  selection_beyond_exposure_tested: false
  validation: false
  next_priority: activity_matched_entry_control_before_tuning
  packet_upper_cpu_s: 139.43
  packet_cap_cpu_s: 180
  task_cap_coreh: 2.05
  line_cap_coreh: 5.35
  standing_reserve_coreh: 0.05
'''}
for rel,block in blocks.items():
 p=R/rel;working=p.read_text();base=subprocess.check_output(['git','show','HEAD:'+rel],cwd=R).decode()
 assert 'ENTRY_GATE_PILOT_20261124_CODEX' not in working and 'directional_entry_pilot_20261124_codex:' not in working
 p.write_text(block+working)
 oid=subprocess.check_output(['git','hash-object','-w','--stdin'],cwd=R,input=(block+base).encode()).decode().strip()
 subprocess.run(['git','update-index','--cacheinfo','100644',oid,rel],cwd=R,check=True)
print('Own blocks staged; pre-existing working changes preserved')
