"""Publish the USER's normalization/relation-first plan; no new measurement."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent
block='''
<!-- INDIRECT_RELATIONS_20261124_CODEX -->
## 2026-11-24 — USER: normalize, enumerate indirect relations, then connect events

[Plan and 20-relation catalog](reviews/maker_indirect_relations_20261124/plan.md). Design only, no new empirical result. Supersedes narrowing only the three reversal examples: preserve all 88 fixed signed orders; three cases are annotations after blind reconstruction. Normalize exact identity/native budget/share and pending USD units/YES side/clock intervals/coverage. Existing code already separates displayed shares from pending values; verify historical raw conventions and avoid converting twice, not a new unit-bug claim.

Relation families: immutable hash and cumulative fills; exhaustion; match/pending/settlement/release; first/later fill bounds; isolated additions and joint capacity; unexplained removal; price/time priority; reappearance; expiry/cancellation; replacement/replenishment/ladder overlap; inventory/turnover; market-condition hypotheses; queue progression. Observed links, conditional constraints, scenario links and behavioural hypotheses remain separate. Quantity conservation and joint allocation prevent counting one change several times. Never use BTC to manufacture the action timestamp later used to test BTC.

ONE next packet: normalization + quantity/pending/book-capacity relations R01-R10 across all88, reusing signed evidence (40 conditional withdrawals,41 exhausted controls,7 unresolved), <=.015 coreh all-in with count/projection gate. Bind complete adapter/checks before measurement; retain all statuses/alternative histories. Old/new intervals and coverage, contradictions, negative-control false pulls, narrowing and assumptions are the deliverables. Later rules remain catalogued where unavailable; no minor-cleanup loop, no full condition rebuild. Current same-day authorization/caps suffice; no USER decision needed within them. Last result ledger task~1.9143/2.00, line~5.2250/5.30 with .05reserve; mt_relplan publication counted once. No new data/model/live activity or push.
<!-- /INDIRECT_RELATIONS_20261124_CODEX -->

'''
status='''indirect_relations_20261124_codex:
  plan: predictfun/reviews/maker_indirect_relations_20261124/plan.md
  stage: design_catalog_complete_measurement_pending
  relations_catalogued: 20
  order: normalization_then_identity_quantity_linkage_then_action_hypotheses
  population: fixed_88_signed_orders
  selected_reversal_examples_are_population: false
  next_priority: normalized_quantity_pending_book_relations_R01_R10
  new_empirical_results: false
  next_within_current_user_scope: true
  next_max_core_hours: 0.015
  task_cap: 2.00
  line_cap: 5.30
  standing_reserve: 0.05
'''
blocks={p:block for p in ['predictfun/HANDOFF.md','predictfun/FINDINGS.md']}
blocks['predictfun/STATUS.yml']=status
blocks['predictfun/reviews/maker_trace_20261122/PENDING.md']=block.replace('(reviews/','(../')
(P/'state_blocks.json').write_text(json.dumps(blocks,indent=2)+'\n')
