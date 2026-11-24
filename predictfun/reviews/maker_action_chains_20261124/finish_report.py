"""Presentation and owned handoff from checked outputs; no new fit or label."""
import csv,json,hashlib,subprocess,importlib.util
from collections import Counter
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[2]
def read(p):return list(csv.DictReader(p.open()))
c=read(H/'outputs/chains.csv');q=read(H/'outputs/quote_relative.csv');qr={(r['order_hash'],r['lag_ms']):r for r in q};same=[]
for r in c:
 a,b=qr[(r['order_hash'],'0')],qr[(r['order_hash'],'1000')]
 status=a['comparison'] if a['comparison']==b['comparison'] else 'clock_sensitive'
 same.append(dict(order_hash=r['order_hash'],group=r['group'],sequence=r['successor_relation'],both_clocks=status))
table=[]
for group in ('all','exhausted','conditional_withdrawal','unresolved'):
 rows=[r for r in same if group=='all' or r['group']==group];table.append(dict(group=group,n=len(rows),counts=dict(Counter(r['both_clocks'] for r in rows))))
(H/'quote_readout.json').write_text(json.dumps(dict(both_clocks=table,ordered=[dict(r,**{f'comparison_{z}':qr[(r['order_hash'],z)]['comparison'] for z in ('0','1000')}) for r in same if r['sequence']=='sequential_after_under_bounds']),indent=2)+'\n')
p=H/'logic_chains.md';s=p.read_text();s=s.replace('toward_spread','more aggressive than old quote').replace('away_from_spread','less aggressive than old quote').replace('| Toward spread | Away from spread |','| More aggressive price | Less aggressive price |')
s=s.replace('Repricing direction is geometric (toward/away from spread at the old price), not a demonstrated response to BTC or a measured improvement in economic value.','The price-direction labels compare the two quotes only. A lower bid or higher ask can simply follow a moving market. They do not measure distance from the contemporaneous spread; the separate comparison below does.')
old='The next useful test is one **declared falsifiable chain hypothesis**, using the new action classes and all relevant controls, rather than another broad indicator search. Start with **depletion → same-side refill/reprice**, because exhaustion is observed. Treat “withdraw on reversal” as a separate weaker hypothesis until timing is identifiable. A same-format expansion is warranted only if it adds independent signed orders with observable successor timing; enlarging anonymous book samples alone does not close the identity gap.'
new='The next useful measurement is a **coverage-gated expansion of signed order chains**, aimed at testing depletion → same-side refill/reprice against partial-fill controls. First count unused existing top-maker hashes with signed terms, complete fill membership and usable successor windows; freeze selection without BTC, inventory or profit outcomes. If the census cannot add timing-identifiable chains, stop. Any full build must fit a declared resource budget; new source acquisition or a cap increase needs separate authorization. Another anonymous book sample does not solve maker identity.'
assert old in s;s=s.replace(old,new)
s+='''
## Does repricing follow the market, or retreat from it?

This extension was declared after the primary chain counts, before measuring the following quantities. It compares every feasible sampled ending state with every feasible sampled successor-post state, at current and lagged clocks. Values are cent ticks relative to the own-side best; all combinations are retained. The saved one-second grid is a proxy, not continuous observation of a private decision.

| Scope | Usable pairs | Less aggressive relative to touch, D0 / lag1s | More aggressive, D0 / lag1s | Timing ambiguous, D0 / lag1s |
|---|---:|---:|---:|---:|
| Exhausted orders | 19 | 2 / 2 | 3 / 5 | 14 / 12 |
| Conditional endings | 23 | 3 / 4 | 0 / 0 | 20 / 19 |
| Total | 42 | 5 / 6 | 3 / 5 | 34 / 31 |

Seven further successor windows are refused; 39 orders have no recorded successor. These are not counted as stable prices or absent actions. For conditional endings, 22 of the 26 linked quotes have a less aggressive *absolute price*, but only three of the 23 usable window comparisons consistently retreat relative to the touch at D0. **We cannot treat the price change alone as a deliberate risk retreat.**

All seven time-ordered chains, with the relative-touch comparison:

| Old hash | D0 | Lag1s |
|---|---|---|
'''
for r in c:
 if r['successor_relation']=='sequential_after_under_bounds':s+='| '+r['order_hash'][:12]+' | '+qr[(r['order_hash'],'0')]['comparison'].replace('_',' ')+' | '+qr[(r['order_hash'],'1000')]['comparison'].replace('_',' ')+' |\n'
s+='\nExact rows and statuses: [quote_relative.csv](outputs/quote_relative.csv). Independent scalar reconstruction matches all176 order/clock rows; five planted corruptions are refused. These remain descriptive chain estimates on consumed days, with no claimed new profitable rule.\n'
p.write_text(s)
next_priority='Coverage-gated expansion of signed order chains to test observed depletion → same-side refill/reprice against partial-fill controls: census existing unused top-maker hashes and complete event membership first; freeze outcome-blind membership only if enough new successor windows can be ordered. Stop if the timing/identity gate fails. Preparation can use current saved data; full computation needs a bounded freeze within the remaining resource allowance, or a USER-approved top-up. New acquisition/days/live use are outside this packet.'
report='''# AVAILABILITY-AND-ACTION-CHAINS-v1 — completed receipt

2026-11-24. USER: “Yes can you proceed to get the logic/action chains.” The readable deliverable is [logic_chains.md](logic_chains.md), including the population condition/action map and every time-ordered successor, not a chosen profitable example.

Primary source freeze: **46f83815e98568bcf2ec599c9385ffd246bdc17a**. Parent accepted quantity packet: **5f6fa5a98641f014bf9e73c8d16d68d9fc5b3126**. The market-relative extension was frozen separately in **QUOTE_COMMIT**, after the primary counts and before its measurement. Contracts, source/input hashes, failed preflight and all logs are retained.

## What is now connected

All 88 fixed signed hashes / 18 markets on consumed Oct24–27; primary as-of2026-11-24T04:18:56Z. Original immutable budgets and observed fills connect to conditional posting/ending bounds, existing exact-maker successor hashes, market-state ranges and later takes. 25 orders have multiple recorded fills; original budget is consumed rather than renewed by each partial fill. Endings remain 41 exhausted / 40 conditional-unavailable / 7 unresolved. This sample was drawn from a prior terminal, fill-selected cohort; its percentages are not unconditional maker action rates.

- Seven successors can be ordered after the old ending under the saved bounds: three exhausted→same-price refill, two exhausted→reprice and two conditional ending→reprice. Thirty-five successor windows overlap, seven are unusable and39 have no recorded successor. Never equate later first fill with new submission.
- Two conditional ending upper bounds tighten by8,480ms and2,250ms. No posting bound improves; no private cancel is identified. Every exhausted order retains the exhaustion control.
- 29,944 competing-record/order combinations are assessed with exact side/actor identity, hard execution-time windows and bracketing pending-quantity envelopes. Counts are repeated evidence within orders, not independent events. The between-fill test has **zero timing-separated opportunities**, so its zero interruptions is uninformative about continuous queue priority.
- 528 interval/clock context rows use the full saved one-second as-of state ranges. Later observed takes produce81 order references to68 unique take records. They are not automatically exits of those orders or reductions of client-known inventory.
- All49 existing successor links are printed apart from their sequence grade. Conditional endings have22 less-aggressive absolute-price successors out of26 links; exhausted orders also have13 of20. Therefore the seven ordered examples cannot establish a clean general depletion-vs-retreat policy.

## The additional market-relative distinction

The original `toward_spread`/`away_from_spread` enums mean only higher/lower own-side price. Their user-facing labels are corrected here; frozen primary values stay unchanged. The descriptive extension reuses the already bound `take_logic_v2` book panel (`kok` is book validity; `bok` is BTC validity), checks probability-to-cent conversion and carries all window combinations, without assuming ambiguous successors followed an ending. Signed quote change equals signed best-price change plus relative-offset change exactly.

42 usable pairs: at D0,5 are always less aggressive relative to touch,3 more aggressive and34 timing ambiguous. With1s lag these are6/5/31. Conditional endings alone are3/0/20 at D0 across23 usable pairs. Seven refused pairs and39 absent links remain statuses. We therefore cannot infer that the observed lower bids/higher asks generally constitute deliberate retreat rather than market following. Nor do ambiguous windows prove market following. The one-second proxy can miss intrasecond changes.

## Verified repeated logic versus remaining gaps

The prior neutral-population map remains the source for repeated condition/action associations:17,100 maker-seconds,11,400 market-side seconds and3,796 retreat episodes. Wider spreads associate with fresh levels; thin depth has different placement and retreat-response associations; passive-fill bursts precede more opposing takes in D0 but the tighter matched effect weakens with1s lag; large inventory associates more clearly with restrained same-direction taking than forced liquidation. No robust BTC-reversal→specific-maker-cancel trigger is identified. Quote creation is mostly anonymous. None of these associations is promoted to a profitable controller.

The actionable working chain is opportunity/quote → same signed budget filled in pieces → exhaustion or conditional ending → refill/reprice/overlap alternative → reconstructed inventory and observed later actions. Exact command time, original size of unsigned successors, hidden reservations, uninterrupted queue tenure and intent remain unidentified. This packet closes joins and preserves alternatives; it does not invent those links.

## Verification and limitations, bundled

Primary selftest23 groups; independent checker re-derives every competing evidence row, every88 chain, all528 context rows, every take link and summary. Ten real-copy corruptions are refused, including false pull, false repost, future context and omitted identity. Extension selftest6 groups; separate scalar checker matches all176 rows and rejects five corruptions. Positive unmodified controls pass. No model fitting, simulator change, new days, intervals or profitability conclusion.

NONBLOCKING interpretation limitations: the signed sample is small and fill/terminal-selected; most private timestamps are intervals; some inference needs full-history/full-display/priority/no-hidden-reservation assumptions; per-record pending envelopes do not identify per-order reservation allocation. Same-hash repeats alone cannot certify uninterrupted rest. The original direction enums are handled by the verified market-relative extension, not renamed inside frozen data. No outcome-invalidating issue remains open for the stated conditional result.

Preflight attempt1 refused the arbitrary150MB source-byte cap; before outcomes it was raised to512MiB while decoding only selected markets/columns. No sample or outcome changed. All failed and successful jobs are charged. Independent checks passed on their first runs. `archive.json` binds two <1MiB compressed parts of the29,944-row evidence; their concatenated CSV (one header) recompresses byte-identically to the original gzip. The original >1MiB file is local/ignored, not committed. Every committed evidence file is bound in SHA256SUMS.

Resource: packet cap .006core-hours (21.6CPU seconds), unchanged task2.00/line5.30 with .05reserve. `resource_final.json` records measured all-attempt usage and a finalization reserve. Only additive owned state is staged, preserving other-session work. No push or external messaging.

## ONE next priority

NEXT_PRIORITY
'''
report=report.replace('QUOTE_COMMIT',subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()).replace('NEXT_PRIORITY',next_priority);(H/'report.md').write_text(report)
body='''
<!-- ACTION_CHAINS_20261124_CODEX -->
## 2026-11-24 — action chains reconstructed and independently checked

[Full chain map](reviews/maker_action_chains_20261124/logic_chains.md); [receipt](reviews/maker_action_chains_20261124/report.md). USER continuation fulfilled on all88 signed orders /18 consumed markets. Seven successors ordered after ending under bounds (3 depleted→refill,2 depleted→reprice,2 conditional ending→reprice),35 ambiguous,7 refused,39 absent. Two additional conditional ending bounds tighten;0 private pulls.25 repeated-fill hashes consume one signed budget. Between-fill competing test hasZERO separated opportunities; no claim of continuous queue retention.81 later-take references/68 distinct actions are not order-exit attribution.

Independent check:29,944 competing evidence rows,88 chains,528 contexts;23 synthetic groups/10 corruptions. Declared market-relative extension:42 usable links, D0 5 less-aggressive/3 more-aggressive/34 ambiguous; lag6/5/31. Conditional-ending links3/0/20 at D0. Six synthetic groups,176 independent row checks,5 corruptions. Original direction enums compare prices only; do not infer a retreat from lower bid/higher ask alone. Keep prior neutral condition/action associations and identity gaps; no new profitability/controller claim.

NEXT_PRIORITY
<!-- /ACTION_CHAINS_20261124_CODEX -->

'''.replace('NEXT_PRIORITY','ONE next priority: '+next_priority)
blocks={f'predictfun/{n}':body for n in ('HANDOFF.md','FINDINGS.md')};blocks['predictfun/reviews/maker_trace_20261122/PENDING.md']=body.replace('(reviews/','(../')
blocks['predictfun/STATUS.yml']='''action_chains_20261124_codex:
  report: predictfun/reviews/maker_action_chains_20261124/report.md
  readable_chain_map: predictfun/reviews/maker_action_chains_20261124/logic_chains.md
  stage: conditional_action_chains_verified
  orders: 88
  markets: 18
  ordered_successors: 7
  ambiguous_successors: 35
  unusable_successor_windows: 7
  no_recorded_successor: 39
  new_conditional_ending_bounds: 2
  between_fill_separated_opportunities: 0
  private_pulls_identified: 0
  primary_check_pass: true
  market_relative_check_pass: true
  recovered_complete_private_policy: false
  next_priority: coverage_gated_signed_chain_expansion_depletion_vs_partial_controls
  packet_cap_coreh: 0.006
  task_cap_coreh: 2.00
  line_cap_coreh: 5.30
  standing_reserve_coreh: 0.05
'''
(H/'state_blocks.json').write_text(json.dumps(blocks,indent=2)+'\n')
print(json.dumps(table,indent=2));print('report and owned handoff prepared')
