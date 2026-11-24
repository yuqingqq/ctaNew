"""Receipt and handoff from checked action/state/reference artifacts."""
import collections,csv,json,statistics,subprocess,shutil
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[2]
def read(p):return list(csv.DictReader(p.open()))
def flag(x):return x=='True'
rr=read(H/'outputs/orders.csv');ordered=read(H/'outputs/ordered_chains.csv');c=json.loads((H/'census.json').read_text());q=json.loads((H/'checks.json').read_text());rq=json.loads((H/'reference_checks.json').read_text());ref=read(H/'outputs/reference_comparison.csv');lookup={(r['lag_ms'],r['field'],r['category']):r for r in ref};view=json.loads((H/'readout.json').read_text());assert q['pass_all'] and rq['pass_all']
source=subprocess.check_output(['git','rev-parse','520f7227f'],cwd=R,text=True).strip();amend=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()
rows=['| Condition over the successor posting window | Observed D0 | Matched market-time D0 | Observed lag1s | Matched market-time lag1s |','|---|---:|---:|---:|---:|']
for field,cat,label in [('btc_side','flat','BTC flat'),('btc_side','against','BTC against quote side'),('inventory_side','offsetting','Quote side offsets reconstructed inventory'),('inventory_side','adding','Quote side adds to reconstructed inventory'),('spread','tight','Spread stays below two ticks'),('own_depth','deep','Own touch depth stays above 200 shares')]:
 a=lookup[('0',field,cat)];b=lookup[('1000',field,cat)];rows.append(f"| {label} | {int(a['observed_count'])}/30 ({float(a['observed_share']):.1%}) | {float(a['expected_uniform_count']):.2f}/30 ({float(a['expected_uniform_share']):.1%}) | {int(b['observed_count'])}/30 ({float(b['observed_share']):.1%}) | {float(b['expected_uniform_count']):.2f}/30 ({float(b['expected_uniform_share']):.1%}) |")
next_priority='One action-choice study on the full already accepted next-order census: classify each observed same-side successor as same-price, more-aggressive-price or less-aggressive-price using identity/price only; retain posting-window grades. Compare these observed action types, with all feasible prior-state values, matching maker/market, starting price position and time remaining. This avoids inventing WAITs from missing fills. Freeze the action population and a support/runtime gate before condition results; if support is thin, stop without fitting a rule. A full run must fit the remaining resource allowance or receive a USER-approved top-up; no new days or live use are authorized here.'
text=f'''# Action first, then market conditions — completed

2026-11-24; as of {c['as_of_utc']}. [Physical-stage contract](contract.md), [reference amendment](reference_contract.md). Primary freeze **{source}**, reference freeze **{amend}**. All four dates Oct24–27 are consumed development days. Zero new model candidates, no validation/profitability claim, no network or simulator change.

The new packet expands the signed action population from88 to **{c['orders']} hashes** using already saved transaction receipts. It reconstructs actions without reading market features, writes and hashes that table, and only then annotates the possible action windows. Independent checks verify that state mapping leaves physical labels unchanged.

## Reconstructed actions

- **58 hashes new to the88-order packet:** 37 exhausted,9 with positive remaining budget,12 with incomplete recorded history.44 of these58 have usable posting windows and complete pure-maker history. These are new reconstructed cases, not new market days or newly collected data.
- **All146:** 78 exact signed-budget exhaustions,56 positive remainders,12 incomplete histories. Among the56 remainders,40 retain the old conditional-unavailability bounds;16 have no identified ending. A positive remainder is not a cancellation.
- **30 successors follow an ending under the time bounds:** 22 after exhaustion (7 same-price refill,15 reprice) and8 after inherited conditional unavailability (1 refill,7 reprice). Nine of these30 come from the added hashes:2 refill and7 reprice, all after exhaustion.
- Other successor statuses:47 overlapping/ambiguous,32 unavailable timing,37 no recorded successor. None of these proves WAIT or no posting.

**Important comparison convention:** this packet uses the pre-existing `post_fill.next_hash` candidate for every order. The earlier88-chain packet used the pull-table successor. Therefore30 versus the previous7 is **not** an improvement from new information alone;21 of the30 involve the old88 under the different, explicitly frozen link convention. Nine are additional cases. All sequences remain candidate relationships rather than identified replacement intent.

Exact rows: [orders.csv](outputs/orders.csv), [all30 ordered chains](outputs/ordered_chains.csv). The30 links span {view['ordered_markets']} markets, with day counts {view['ordered_days']}. Related orders share transactions and markets. The cache was acquired around earlier filled/terminal examples; this is not a random sample of maker decisions.

## Mapping the actions to conditions

Every post, ending and successor window has D0 and lag1s annotations: **876 rows**, including unknown/refused cases. Whole feasible windows are retained, not a midpoint picked to fit a story. Inventory is reconstructed venue exposure, not a private client's known inventory. The saved one-second grid can miss intrasecond changes.

In the30 ordered successor windows,23 remain BTC-flat at both clocks. To avoid treating a common market state as an action trigger, a separate adaptive amendment froze an equal-window market-time benchmark before computing it: same maker/market/side, same sampled duration and same elapsed-market phase. It enumerates12,127 reference windows and generates500 diagnostic draws per case/clock. The exact uniform expectation, rather than a selected draw, is the comparison below.

{chr(10).join(rows)}

All48 category/clock rows, including mixed and unknown categories, are in [reference_comparison.csv](outputs/reference_comparison.csv). These references are **market times, not observed non-actions**. They may contain unseen/unfilled posts. Only market identity, side, duration and elapsed phase are matched; the table does not control all other state variables or queue position. Draws are implementation/reference diagnostics, not day-cluster confidence intervals or p-values.

The check changes the interpretation: calm BTC is common in the matched market time too, so the high raw calm count does not establish a calm-BTC repost trigger. Offsetting inventory is modestly more common around the observed successor actions, with a smaller excess after lagging the state; it remains a hypothesis. Tight spread is more frequent than the phase-matched benchmark in these successor windows. This concerns maintenance/replenishment after a prior order, and must not be conflated with the older anonymous fresh-level/wider-spread population. No per-field difference is a demonstrated causal or profitable rule.

## Whole chain and remaining identification boundary

```mermaid
flowchart LR
 Q[Signed quote: original size and price] --> F[Recorded partial fills consume the same budget]
 F --> E[Budget exhausted]
 F --> U[Positive remainder: ending conditional or unknown]
 E --> N[Next observed order: refill or reprice candidate]
 U -. ending and sequence assumptions .-> N
 N --> C[Prior-state range: BTC, inventory, spread, depth]
 F --> I[Reconstructed inventory and later maker/taker activity]
```

This strengthens the mechanically identifiable depletion→replenishment/reprice chain. It does not recover a complete private posting/pulling policy. Exact private submission/cancellation commands, no-action exposure, uninterrupted queue priority and per-order pending allocation remain unknown. We therefore did **not** run or claim a posting/non-posting propensity test; its control-observability gate failed. Expanding fill-selected cases cannot by itself fix that gate.

## Verification, resource and deferred work

14 initial synthetic groups pass. Independent checks reparse218 raw signed fill events, reconcile241 saved feed legs, all146 action rows and876 state rows, and refuse9 planted corruptions. The market-time reference has four boundary/category groups and a separate scalar enumeration of every reference window; five real-copy corruptions are refused. The same accepted ABI decoder is reused for hashing/calldata; raw event accounting and action/state reconstruction are independently checked. No failed run in this round. No frozen predecessor file changed.

The raw-to-derived hashes are in census.json; compact evidence copies are included under evidence/. Sources and receipts are bound in manifest.json and reference_bindings.json; final hashes in SHA256SUMS. Packet cap .004core-hours (14.4CPU seconds), unchanged task2.00/line5.30 with .05reserve; measured all-attempt usage is in resource_final.json. Existing unrelated changes are preserved; local commits only.

Minor source-format cleanup is deferred. The material limit is action observability and selection, not a failed implementation check. Proven exhaustion, conditional unavailability and incomplete history remain distinct throughout.

## ONE next priority

{next_priority}
'''
(H/'report.md').write_text(text)
# Preserve checked derived evidence with unchanged bytes, below the repository size limit.
(H/'evidence').mkdir(exist_ok=True)
for name in ('decoded_orders.json','decoded_events.csv','members.json','life.json'):
 p=R/'data/predict_fun/research/maker_action_population_20261124'/name;assert p.stat().st_size<1024*1024;shutil.copyfile(p,H/'evidence'/name)
body=f'''
<!-- ACTION_POPULATION_20261124_CODEX -->
## 2026-11-24 — action-first expansion and subsequent condition mapping

[Report](reviews/maker_action_population_20261124/report.md). USER proceed instruction fulfilled:146 signed hashes,58 new to the original88;44 new usable posting chains. All146:78 exhausted /56 positive remainders /12 incomplete histories. Independent raw checks218 events/241 feed legs/146 actions/876 contexts,9 falsifiers;14 synthetic groups. Physical table frozen before state mapping and unchanged afterward.

30 successors ordered under full windows,9 from added hashes;22 exhaustion→successor (7 refill/15 reprice),8 inherited conditional ending→successor (1 refill/7 reprice). Uses accepted post_fill.next_hash, unlike the earlier pull-table successor, so30 vs7 is NOT a pure coverage improvement.47 ambiguous/32 unavailable/37 no observed successor. No private pulls or WAITs inferred.

Equal-window phase-matched market-time reference:12,127 windows,500 diagnostic draws/case-clock, all categories retained, independent scalar check/five falsifiers. BTC-flat observed23/30 vs24.61 expected (lag24 vs24.62): raw calm count is not a repost trigger. Offsetting inventory15 vs12.97 expected (lag14 vs12.93); exploratory, not an identified rule. Tight spread26 vs22.78 (lag25 vs22.74), a successor population distinct from fresh-level posting. References can contain unseen actions; posting/nonposting control gate remains unavailable. No model, profit, validation or significance claim.

ONE next priority: {next_priority}
<!-- /ACTION_POPULATION_20261124_CODEX -->

'''
blocks={f'predictfun/{n}':body for n in ('HANDOFF.md','FINDINGS.md')};blocks['predictfun/reviews/maker_trace_20261122/PENDING.md']=body.replace('(reviews/','(../');blocks['predictfun/STATUS.yml']='''action_population_20261124_codex:
  report: predictfun/reviews/maker_action_population_20261124/report.md
  stage: expanded_actions_then_state_mapping_verified
  signed_orders: 146
  added_orders: 58
  new_usable_posting_chains: 44
  exhausted: 78
  positive_remainder_not_pull: 56
  incomplete_histories: 12
  ordered_successors: 30
  newly_added_ordered_successors: 9
  successor_rule: accepted_post_fill_next_hash
  counts_comparable_to_prior_pull_successor_rule: false
  context_rows: 876
  reference_windows: 12127
  matched_nonactions_available: false
  new_trigger_or_profitability_claim: false
  next_priority: full_observed_next_order_action_choice_census_with_support_gate
  packet_cap_coreh: 0.004
  task_cap_coreh: 2.00
  line_cap_coreh: 5.30
  standing_reserve_coreh: 0.05
''';(H/'state_blocks.json').write_text(json.dumps(blocks,indent=2)+'\n');print('checked report, evidence copies and owned handoff prepared')
