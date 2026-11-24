"""Present checked outputs and stage additive state without other-session changes."""
import collections,csv,gzip,hashlib,importlib.util,json,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[2]
def read(p):
 with (gzip.open if str(p).endswith('.gz') else open)(p,'rt') as f:return list(csv.DictReader(f))
def git(*a,data=None):return subprocess.check_output(['git','-C',str(R),*a],input=data)
def sha(b):return hashlib.sha256(b).hexdigest()
spec=importlib.util.spec_from_file_location('cpu',R/'predictfun/reviews/maker_trace_20261122/tools/cpu_ledger.py');cpu=importlib.util.module_from_spec(spec);spec.loader.exec_module(cpu);ledger=cpu.ledger();cost=sum(r['cpu_s'] for r in ledger if r['tag'].startswith('mt_choices_'));task=sum(r['cpu_s'] for r in ledger)/3600+cpu.EXTRA_H
assert cost+.15<=5.04 and task+.15/3600+.05<=2 and cpu.LINE_BEFORE+task+.15/3600+.05<=5.3
for p,h in json.loads((H/'manifest.json').read_text())['files'].items():assert sha((R/p).read_bytes())==h,p
assert json.loads((H/'checks.json').read_text())['all_true']
a=read(H/'actions.csv.gz');p=read(H/'partial_cases.csv');s=read(H/'matched.csv');result=json.loads((H/'result.json').read_text());valid=[r for r in a if r['status']=='ok'];counts=collections.Counter(r['price_action'] for r in valid);examples=[r for r in p if r['successor_relation']=='sequential_after_under_bounds'];assert len(examples)==8
old={r['order_hash']:r for r in read(R/'predictfun/reviews/maker_action_population_20261124/outputs/ordered_chains.csv')}
rows=[]
for r in examples:
 o=old[r['order_hash']];rows.append(f"| {r['day']} / {r['mid']} / #{r['rank']} | {r['side']} {r['old_tick']}→{r['new_tick']}¢ | {float(r['filled_shares']):.3f} / {float(r['original_shares']):g} | {float(r['gap_min_s']):.3f}–{float(r['gap_max_s']):.3f}s | {r['btc_D0']} / {r['btc_lag']} | {r['inventory_D0']} / {r['inventory_lag']} | `{r['order_hash'][0:12]}` |")
report=f'''# Concrete partial-order chains and next-quote choices

As of {result['as_of_utc']}. Source freeze **551932c10**; comparator-only checker supersession **a5b947041**. Full source SHA: {git('rev-parse','HEAD').decode().strip()}. Consumed Oct24–27, top-three makers in the accepted fill-linked order census. No new days, tape replay, fit, validation or private action log.

We can reconstruct partial action chains and test explanations, but this round establishes no general BTC or inventory trigger. The useful result is that both lower and higher subsequent quotes occur while the old signed order still has unfilled budget; exhaustion alone cannot describe these conditional chains. This does not identify the private cancel command or prove the new order replaced the old one.

## Full observed-action census

4,149 source hashes give 3,286 distinct linked successor actions after de-duplication. There are 1,097 conditionally ordered posting windows, 1,361 overlapping/ambiguous windows and 828 refused windows. The 1,097 ordered actions split into **{counts['less']} less-aggressive prices, {counts['same']} unchanged prices, {counts['more']} more-aggressive prices**. This is a fill-observed population: it omits orders that never fill, and is not a posting/nonposting decision census. G2/G4 posting windows remain conditional/scenario bounds.

Comparing prices relative to the contemporaneous touch gives 15 definitely farther-back,54 definitely closer,8 exactly unchanged and1,020 ambiguous ranges among the1,097. A lower bid or higher ask by itself cannot distinguish following a moving market from deliberately retreating. These are cross-window bounds, not exact decision instants. The offset outcome tests whether retreat is identifiable, not whether all ambiguous cases were truly non-retreats.

## Every ordered partial-budget chain

All56 signed positive-remainder cases are retained in partial_cases.csv:8 sequential under inherited ending bounds,23 ambiguous,21 unavailable and4 with no observed successor. The8 sequential cases divide into4 less-aggressive,3 more-aggressive and1 same-price quote. Every old ending here is conditional unavailability; none is a privately observed cancellation. Remaining signed budget is an upper bound, not proof the quantity was resting. No successor in this eight has a verified original signed size.

The gap is from the inferred old ending to the next posting window. BTC is signed to the order side; inventory “offsetting” means that side would reduce the reconstructed signed exposure. Both current-window and1s-lag classifications are shown, including mixed categories. All8 are listed, not hand-selected winners.

| Date / market / maker | Later quote | Old filled / original shares | Conditional gap | BTC D0 / lag | Inventory D0 / lag | Old hash prefix |
|---|---|---:|---:|---|---|---|
{chr(10).join(rows)}

Three concrete hypotheses emerge from these records:

1. **Reprice lower while still wanting buys.** Maker#1, Oct26 market3789484:50-share bid46¢,10 filled; conditional ending269.035–271.500s; next bid32¢ posted272.947–274.993s after market start. BTC is against the bid in both clock views; buying offsets a short position. Compatible with repricing the desired buy downward as conditions deteriorate. It does not prove a BTC-triggered cancel, or movement farther behind the contemporaneous touch.
2. **Quote more aggressively on the inventory-reducing side.** Maker#2, Oct24 market3704004:50-share bid62¢,22.182 filled; conditional ending19.530–24.642s; next bid66¢ at31.762–33.603s. BTC is flat and buying offsets short exposure in both views. Compatible with continuing to work a buy while short, but a rising market could also explain the price change.
3. **Same-price new order after a partial ending.** Maker#3, Oct25 market3734864:10-share bid31¢,1.449 filled; conditional ending120.521–124.724s; another signed hash at31¢ at127.773–133.319s. BTC is flat and buying adds exposure in both views. This is a candidate renewal with unexhausted old budget; it does not establish depletion, a desired queue reset or the new size.

The independent 146-hash packet also contains22 ordered successors after exact budget exhaustion (7 same-price /15 repriced). That is a separate, stronger physical exhaustion category. Multiple fills under one hash consume that hash's fixed original budget; a different hash is a different signed order. Public records do not prove continuous queue residence between fills.

## Tests of general logic

Eight comparisons were declared before computation:two exposures × four price/offset outcomes,500 conditional draws each, fixed support and multiplicity rules. Only observed quotes are compared with observed quotes; no absent fill is called WAIT.

- BTC-against versus flat:only3 against/4 flat cases in3 common strata/2 markets after identity,phase,old-touch,inventory and timing matching. **Insufficient support** for a general reversal rule. We did not loosen matching after this result.
- Inventory-adding versus offsetting:41/34 quotes in31 strata/12 markets meet the declared count floor. Less-aggressive-price difference **−4.89 percentage points**, raw exploratory p=.760; same-price +1.15pp,p=1;more-aggressive +3.74pp,p=.804. Day signs disagree. **No demonstrated inventory-driven repricing rule.** This concerns price choice conditional on another observed quote, not whether inventory affects sizes, takes or the decision to quote at all.
- None of the8 tests meets the declared exploratory-pattern rule. Exact day values,all null draws and every status are committed. The permutation is a descriptive conditional allocation; related actions and unmeasured confounding preclude causal claims. These consumed days supply no validation claim.

## What the evidence supports now

Observed skeleton: signed order → partial/exhausted executions → sometimes a bounded conditional ending → next observed same-side quote at the same or a different price. Price changes occur in both directions, including on the exposure-reducing side. Market-following, adverse-flow protection and inventory management are plausible explanations for particular chains; a universal controller has not been recovered. A useful strategy rule cannot yet be promoted from these examples. This packet measures neither queue-profit causality nor whether remaining at the front pays for early losses.

The largest limitation is timing/ownership identification, not a missing classifier:most posting-offset intervals span several feasible states, and unseen/cancelled-before-fill orders are missing. Inherited unavailability and G4 scenarios remain assumptions. Reconstructed holdings are not private client-known inventory; no proof of the maker's motivation is claimed.

## Verification and disposition

Source identities,canonical successor membership,all prices/statuses/conditions/offset bounds,all scalar matched rates/null draws and all56 partial cases were independently recomputed from hash-bound accepted artifacts. Eight real-output corruptions were refused. Synthetic controls include a known association and no-association fixture. The first checker failed on Python bool-as-int parsing;v2 changes only boolean comparison,with explicit positive/negative controls. Measurement and frozen predicates are unchanged;failed logs remain.

Measured packet CPU before this publishing job:{cost:.2f}s;conservative publishing allowance .15s;packet cap5.04s. Task before publishing {task:.7f}coreh,line {cpu.LINE_BEFORE+task:.7f};both still fit task2.00/line5.30 plus .05 standing reserve. The final ledger receipt records this job after it closes. No implementation or other-session file is changed.

ONE next priority, if further compute is authorized:an **ending-anchored market-following versus retreat audit** on the56 signed partial-budget cases, comparing their old price against the touch/fair-value ranges at the inferred ending and at the next post, with overlapping and unidentified histories retained. The present post-to-post offset ranges cannot answer that narrower question. Do not fit a policy, weaken matching or claim queue advantage first. This is existing consumed evidence; it needs a bounded resource top-up if it exceeds the remaining allowance. No new days or live use are implied.
'''
(H/'report.md').write_text(report)
block='''\n<!-- PARTIAL_CHOICES_20261124_CODEX -->
## 2026-11-24 — concrete partial chains and observed quote choices

[Report](reviews/maker_partial_choices_20261124/report.md). USER requests concrete partial decisions fulfilled. Full accepted4149 hashes→3286 deduplicated successors;1097 conditionally ordered posting windows. All56 signed positive remainders retained:8 conditional ending→later quote (4 less-aggressive/3 more/1 same),23 ambiguous,21 unavailable,4 no link. Report prints all8 with original/filled amounts,prices,timing and D0/lag contexts. No private pull or successor size inferred.

Frozen matched action-choice test:BTC3vs4 in2markets is insufficient;inventory41vs34 in12markets has mixed day signs and no demonstrated repricing association. None of8 predeclared tests supports a general trigger. Post-to-post offset ranges:15 definite retreats/54 advances/8 unchanged/1020 ambiguous. Do not mistake absolute price changes for a deliberately more defensive quote. Underlying post bounds include G4 scenarios; observed quotes are fill-selected.

Independent identities/physical rows/matched rates/null draws/56-case recount passed;8 falsifiers. Boolean-only checker supersession disclosed;outputs unchanged. No model,new day,replay,queue-value or profitability claim. ONE next priority:ending-anchored market-following-versus-retreat audit on these56 partial cases,not weaker matching or a fitted controller. Must fit remaining allowance or receive a bounded CPU top-up;new days/live not authorized by this packet.
<!-- /PARTIAL_CHOICES_20261124_CODEX -->
\n'''
blocks={name:block for name in ['predictfun/HANDOFF.md','predictfun/FINDINGS.md']};blocks['predictfun/reviews/maker_trace_20261122/PENDING.md']=block.replace('(reviews/','(../');blocks['predictfun/STATUS.yml']='''partial_choices_20261124_codex:
  report: predictfun/reviews/maker_partial_choices_20261124/report.md
  stage: checked_partial_chains_and_observed_quote_choices
  source_orders: 4149
  deduplicated_successors: 3286
  conditionally_ordered_posts: 1097
  signed_partial_cases: 56
  ordered_conditional_end_successors: 8
  demonstrated_general_triggers: 0
  private_cancellations_identified: 0
  new_model_or_profitability_claim: false
  next_priority: ending_anchored_market_following_vs_retreat_audit_with_resource_gate
''';(H/'state_blocks.json').write_text(json.dumps(blocks,indent=2)+'\n')
assert not git('diff','--cached','--name-only').strip();scope={'parent':git('rev-parse','HEAD').decode().strip(),'preserved_dirty':{},'owned_state':{}}
for name in git('diff','--name-only','-z','HEAD').decode().split('\0'):
 if name and name not in blocks:
  f=R/name;scope['preserved_dirty'][name]=sha(f.read_bytes()) if f.exists() else None
for name,block in blocks.items():
 path=R/name;work=path.read_bytes();base=git('show','HEAD:'+name);own=block.encode();assert b'PARTIAL_CHOICES_20261124_CODEX' not in work and b'partial_choices_20261124_codex:' not in work
 def add(old):
  if name.endswith('HANDOFF.md'):
   anchor=b'<!-- /STANDING_BATCH_RULE_20261120_USER -->\n';assert old.count(anchor)==1;return old.replace(anchor,anchor+own,1)
  if name.endswith('STATUS.yml'):
   first,rest=old.split(b'\n',1);return first+b'\n'+own+rest
  return own+old
 path.write_bytes(add(work));staged=add(base);blob=git('hash-object','-w','--stdin',data=staged).decode().strip();git('update-index','--add','--cacheinfo','100644,'+blob+','+name);assert path.read_bytes().replace(own,b'',1)==work and staged.replace(own,b'',1)==base;scope['owned_state'][name]={'preexisting_work_sha':sha(work),'staged_sha':sha(staged),'post_work_sha':sha(path.read_bytes())}
for name,digest in scope['preserved_dirty'].items():
 f=R/name;assert (sha(f.read_bytes()) if f.exists() else None)==digest,name
(H/'commit_scope.json').write_text(json.dumps(scope,indent=2)+'\n');assert git('rev-parse','HEAD').decode().strip()==scope['parent'];print('report and owned state staged; all prior dirty changes preserved')
print(dict(valid_price_actions=dict(counts),cost_seconds=cost,task_before_finalizer=task,line_before_finalizer=cpu.LINE_BEFORE+task))
