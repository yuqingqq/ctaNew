"""Publish descriptive verified outputs; no new selection/behavioural predicate."""
import csv,gzip,json,re,collections,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[2];O=H/'outputs'
r=json.loads((O/'result.json').read_text());c=json.loads((H/'checks.json').read_text());v=json.loads((H/'readout.json').read_text());assert c['pass_all']
def rows(p):
 with (gzip.open if str(p).endswith('.gz') else open)(p,'rt') as f:return list(csv.DictReader(f))
b=rows(O/'chain_bounds.csv');o=rows(O/'order_summary.csv');led=rows(O/'order_ledger.csv');bk=rows(O/'book_edges.csv.gz');os={x['order_hash']:x for x in o}
mr=json.loads((ROOT/'predictfun/reviews/maker_terminal_chain_20261124/manifest.json').read_text())['legs'];groups={(x['day'],str(x['mid']),x['sid'],os[x['mk_hash']]['side'],os[x['mk_hash']]['tick']) for x in mr}
counts=collections.Counter((x['day'],x['mid'],x['sid'],x['side'],x['tick']) for x in bk if x['kind']=='match_reflection_candidate');gc=collections.Counter('none' if counts[k]==0 else ('one' if counts[k]==1 else 'multiple') for k in groups)
kind=collections.Counter(x['kind'] for x in bk);bygroup={g:{'n':sum(x['group']==g for x in b),'narrowed':sum(x['group']==g and x['narrowed']=='True' for x in b)} for g in sorted({x['group'] for x in b})}
spent=0
for line in (ROOT/'predictfun/research/run_light.log').read_text().splitlines():
 if re.search(r' tag=mt_qty_\S+ ',line):spent+=float(re.search(r' cpu_user_s=([\d.]+)',line).group(1))
extra=dict(record_groups=len(groups),match_candidates_per_group=dict(gc),book_edge_kinds=dict(kind),refined_by_source_group=bygroup,cpu_seconds_as_of_report=spent)
(H/'descriptive.json').write_text(json.dumps(extra,indent=2)+'\n')
report=f'''# QUANTITY-LINKAGE-v1 — built and independently verified

2026-11-24. USER: normalize and link indirect events across all88 signed orders, then test their trading logic. This implements the first executable packet of [the relation catalog](../maker_indirect_relations_20261124/plan.md), R01–R10. Scope:88 signed hashes,152 distinct raw execution events,18 markets, consumed Oct24–27. As of **{r['as_of_utc']}**. Descriptive reconstruction, not a representative cancellation rate, trading rule, validation or profitability result.

**Result:20/88 orders gain narrower conditional intervals.** Ten posting lower bounds tighten only under the stronger full-original-size scenario;17/40 previously conditional ending windows tighten. These overlap. The weaker observed-future-fill constraint improves0/79 eligible posting windows. No private pull is identified. The original41 exhausted controls remain exhausted;47 positive signed remainders do not mean live quotes. All88 are retained, with9 orders refused for accepted refinements and79 conditional.

Source freeze **6ffe353ce**; final build source **{r['source_commit']}**. Runtime adapters were committed before use (178a93dc9, cce7250a8,37b985866,1408b402a); verification adapter5b607e6b9. Source/input hashes are in sources.json, fast_sources*.json, recovery_sources.json and inputs.json. Frozen files were not edited. All original failures/refusals are retained.

## What now connects

| Relation | Verified coverage / result | What it establishes |
|---|---|---|
| R01–R02 signed budget and exhaustion |88 hashes /152 raw events;41 exhausted,47 incomplete |Original maker/taker amounts are immutable. Each successful fill consumes the same native making budget. A later partial fill does not reset the original order size. |
| R03 book/pending match reflections |88 orders have candidates;{len(groups)} record/side/price groups: {gc.get('none',0)} none, {gc.get('one',0)} one, {gc.get('multiple',0)} multiple |Candidate links only. A depth drop and pending growth can represent one match. Shared evidence is retained without allocating it repeatedly as independent volume. |
| R04 failed settlement/release |88 unknown |Successful receipts lack a complete attributed failure stream. Pending decreases cannot distinguish settlement, failure release or other changes. |
| R05–R06 first/later executions |All88 linked;25 hashes have multiple recorded fills,63 one |Execution windows and cumulative quantities. Repeated executions alone do not establish continuous resting or preserved priority. |
| R07 additions/restorations |88 have possible anonymous additions |No maker-attributed posting upper bound is obtained by matching a quantity to a book change. |
| R08 capacity bounds |20 narrower;59 no gain;9 refused |Conditional timing improvement, with A/B assumptions separate below. |
| R09 shared capacity |0 selected joint-demand conflicts;606 match-candidate edges compete with other records for a shared book pair |No contradiction under this loose selected-sample envelope; no unique allocation or completeness claim. |
| R10 unexplained depth reduction |78 orders' ending windows contain candidates;10 do not |Unexplained by potentially matching recorded fills, not proof of this maker's pull. Other orders, reservations, failures, unseen executions and masked changes remain alternatives. |

The link tables contain5,236 shared book edges and880 order/relation status rows. The exact budget ledger retains native integer units; REST and chain representations are reconciled rather than added as separate fills. Signed native side/price and the accepted YES mapping are checked. Settlement order is labelled as such and is never substituted for match order.

## Timing improvement and its assumptions

| Bound | Eligible n | Narrowed | Median width before → after | Maximum reduction |
|---|---:|---:|---:|---:|
| Posting A: observed future fills must fit |79|0|3.860 →3.860s|0s|
| Posting B: entire original signed size must fit |79|10|3.860 →3.054s|155.755s|
| Conditional ending, after all known fill upper bounds |40|17|3.161 →3.0685s|17.275s|

Both posting scenarios require full display/continuous rest/no reentry. B additionally assumes complete prior execution history and full initial displayed size. A is weaker; B must never be reported as an A result. Posting tests use only frames before the earliest hard first-match lower bound, so a post-match disappearance cannot falsely become a late-post lower bound. The posting upper bounds are unchanged. Narrowed lower endpoints are open at their witness frame.

Ending refinements require complete execution coverage and full display of the remaining order, use only frames after all observed fills' upper bounds, and retain the earlier conditional grade. They bound unavailability under the scenario; they do not prove cancel versus expiry/invalidation. Positive signed remainder is an upper bound, not observed available shares. Exact amounts do not imply complete private history.

Displayed shares and aggregate pending USD remain separate. Capacity uses displayed shares plus both sides' pending USD divided by the YES price, a deliberately loose upper envelope across possible display conventions. The two pending sides may reflect the same match: the sum is **not** executed volume. Per-order pending allocation and whether historical display includes reservations remain unidentified. This packet did not find a proven normalization defect in the historical payload.

The post-link annotation covers all4 timing-only reversal matches at the two previously declared delays, including the three posting-compatible examples:14 A/B annotation rows. Their chronology is unchanged. Three orders still permit posting either before or after reversal; the fourth's earliest conditional posting is after it. We have not established a causal reversal → pull chain.

## Verification and limits

[Checks](checks.json) pass on all152 raw receipt events,88 orders,880 relation rows,550,102 normalized points and5,236 book edges. The independent checker does not import the builder. It rehashes original inputs, rechecks signed hashes/assets/amounts against raw receipt logs, recomputes cell units/clocks/coherent pairs from raw frame/level files, recomputes every accepted/candidate interval and witness, joint-demand conflicts, relation counts, and primary summaries.21 synthetic controls pass;10 actual corruptions are refused (original-size reset, double accounting, wrong remainder, duplicate event, omitted order, invented timing/pull/grade).

Recovered cached cells are verified against raw inputs; no partial-check verdict is accepted. V2/V3 pilots reproduce every v1 inferred field and every normalized value exactly. Native pending convention is an explicit historical assumption; the independent raw recount verifies the declared conversion, not the venue's private accounting convention. Missing/invalid levels are never made zero. There is no complete private cancel/failure feed and no unique owner allocation for anonymous changes. The source sample itself was previously selected; these counts do not generalize to all maker decisions.

Small unclaimed checker gaps are deferred: candidate-edge narrative/attribution strings and the R10 certain-quantity auxiliary column are not independently rederived field for field. Their quantitative edge membership, possible quantities, drops, primary statuses and conditional bounds are checked. No conclusion here relies on those auxiliary columns. No new tune or cleanup-only round is required.

## Resource history

The first three pilot projections refused the fixed reserve gate (72.10,50.44,46.12s); caching unchanged pandas columns and removing unused valid level rows preserved output parity. The unchanged final projection plus actual prior charge fit the54s packet, replacing its fixed9s reserve. The full process then hit its20s CPU watchdog;16 normalized market files were retained and hash-bound. Recovery recomputed every order inference and completed all18 markets. The first full checker hit its16s watchdog; its complete rerun passed in14.86 process CPU seconds after eliminating an unnecessary whole-table string conversion. Every failed attempt is charged.

The verification-only local allowance was amended to .022 coreh under the USER's standing CPU-top-up authorization; task2.00, line5.30 and .05 reserve stayed unchanged. Charged packet cost at report generation: **{spent:.2f}s / {spent/3600:.5f}coreh**; publication is added in resource_final.json. No new data/network/day/model/replay/live use or push. Earlier conservative runtime estimates were inadequate for cell multiplicity; future jobs should checkpoint per market from the start and budget normalized cell count, not frame count alone.

## ONE next priority

**Test availability at competing executions and successor timing on these same88 chains (R11–R16), with pending reservations and exhaustion controls carried forward.** Reuse the present quantities and old/new bounds. At each strictly later same-price/through-price execution by another order, ask whether the old hash could still have been available or could already have been reserved/exhausted; refuse tie/overlap cases. Join existing same-maker successors with their full posting windows to separate sequential replenishment, overlap/ladder, and unresolved replacement. Keep scenarios rather than force a private action.

Freeze those action intervals before annotating BTC/inventory. The useful deliverable is a population table of surviving action chains and the conditions that are before/inside/after each action interval. No additional BTC threshold search, model fit or new days. This serves the USER's action/logic-chain goal; broader data collection or live evidence is outside this packet. A bounded saved-data follow-up can use existing authorization only after a count/runtime gate fits the remaining task/line allowance; otherwise ask for the specific additional resource.
'''
(H/'report.md').write_text(report)
block='''\n<!-- QUANTITY_LINKAGE_20261124_CODEX -->
## 2026-11-24 — normalized indirect linkage built and verified

[Report](reviews/maker_quantity_linkage_20261124/report.md). USER build/verify instruction fulfilled for the first R01–R10 packet: all88 signed hashes/152 raw events,18 markets on consumed Oct24–27. Full independent check passes (550,102 normalized points,5,236 book edges,880 relation rows);21 synthetic controls and10 corruption refusals. Partial fills consume one immutable signed budget;25 hashes have repeated fills.41 exhausted controls,40 conditional withdrawal candidates,7 unresolved remain distinct.

20 orders gain conditional timing information: posting B10/79 (full original size/coverage/full display premise); weaker future-fill posting A0/79; ending17/40. Median post width3.860→3.054s only in B; ending3.161→3.0685s. Posting upper bounds unchanged; no private pulls identified. Three reversal-compatible orders still have uncertain post-before-reversal chronology; fourth timing-only hit is before earliest post. No new behavioural/profitability result. Failed settlement stream, pending per-order allocation, private withdrawals remain unknown.

All resource refusals and watchdog stops retained/charged. Recovery reuses hash-bound normalized cells; independent checker verifies every cell against raw input. Verification-only packet allowance .022coreh under standing CPU authorization, unchanged task2.00/line5.30/.05reserve; final ledger in packet. Future jobs checkpoint markets and project cell multiplicity. No external messaging/push.

ONE next priority: availability at competing executions and successor timing (R11–R16) across same88; distinguish reserved/exhausted/conditionally absent and sequential replenishment vs overlapping ladder. Freeze physical bounds before condition annotation. Count/runtime gate within remaining authorization; no tuned variant/new days/live use. Minor unused auxiliary edge fields deferred.
<!-- /QUANTITY_LINKAGE_20261124_CODEX -->

'''
status='''quantity_linkage_20261124_codex:
  report: predictfun/reviews/maker_quantity_linkage_20261124/report.md
  stage: normalized_R01_R10_built_independently_verified
  orders: 88
  raw_events: 152
  normalized_points_checked: 550102
  orders_with_conditional_refinement: 20
  posting_weaker_A_refinements: 0
  posting_original_size_B_refinements: 10
  conditional_ending_refinements: 17
  private_pulls_identified: 0
  exhausted_controls: 41
  conditional_withdrawal_candidates: 40
  unresolved_orders: 7
  reversal_chronology_changed: false
  next_priority: same_88_availability_at_competing_executions_and_successor_timing_R11_R16
  no_new_model_profitability_or_validation_claim: true
  local_packet_cap_coreh: 0.022
  task_cap_coreh: 2.00
  line_cap_coreh: 5.30
  standing_reserve_coreh: 0.05
'''
blocks={'predictfun/HANDOFF.md':block,'predictfun/FINDINGS.md':block,'predictfun/STATUS.yml':status,'predictfun/reviews/maker_trace_20261122/PENDING.md':block.replace('(reviews/maker_quantity','(../maker_quantity')}
(H/'state_blocks.json').write_text(json.dumps(blocks,indent=2)+'\n')
print(json.dumps(extra,indent=2))
