# Hypothesis-first action reconstruction: reversal → withdrawal

2026-11-24. USER: make explicit assumptions about makers' reactions to market conditions, then verify them to bridge missing actions. This packet implements the first such hypothesis on saved evidence. It does not require a full condition/controller rebuild.

Source freeze **814f8fb5d12543c910355467a0c243b5afba57ab**. Measurement as-of **2026-11-24T03:10:27Z**, consumed Oct 24–27. [Frozen contract](contract.md), [computed result](outputs/result.json), [independent checks](checks.json), [all cases](outputs/case_alignment.csv).

## Method and hypothesis

We hypothesize an action chain, independently reconstruct its observable components, and classify whether the chain is supported, merely possible, contradicted under the assumptions, or unresolved. The hypothesis must not be used to create the very cancellation timestamp that supposedly verifies it.

**H1: a BTC reversal against an already-resting quote is followed by withdrawal of that side within five seconds.** On the saved one-second grid, the side-aligned trailing five-second return changes from >= +1 bp at t−5s to <= −1 bp at t, with the previous sample not already adverse. Bid and ask directions are opposite. These are inherited signal units/thresholds, a newly declared transition, and one declared response window; no threshold was selected from this result. D=0 and a one-second availability-delay sensitivity are retained. This is not measured private latency.

We first detect reversals over every saved market-side-second, independently of endings. We then join the fixed signed-order sample: 40 conditional withdrawals, 41 exhausted quantities, seven unresolved. The existing post and ending bounds remain unchanged. Timing-only overlap can suggest a chain; stricter alignment requires the reversal after the posting upper bound, which establishes post-before-reversal under the inherited premises. Continuous resting and a private cancel are still unobserved.

## Result

There are **43 grid-defined reversals** across the 19 markets: 2 / 2 / 33 / 6 by day. The ending sample is fixed, small and filled-order-selected, not all live orders exposed to those reversals.

| Withdrawal comparison | D0 | Availability delayed 1s |
|---|---:|---:|
| Evaluable conditional endings | 39 / 40 | 39 / 40 |
| Timing-only overlap with correct-side reversal | 4 / 39 | 3 / 39 |
| Timing-only overlap with wrong-side reversal | 3 / 39 | 3 / 39 |
| Correct-side match with post-before-reversal established by posting upper bound | 0 / 39 | 0 / 39 |

One withdrawal window is outside complete context coverage. Seven unresolved signed endings remain unresolved and never enter a reaction denominator. The exhausted comparator has one timing-only overlap at each delay (1/40 evaluable at D0; 1/41 at D1000), demonstrating why proximity to a reversal alone cannot label an ending as a pull. Exhausted versus incomplete cohorts are not exchangeable matches; these rates are not a causal contrast.

The frozen 500 within-market/side/minute timing placebos use the same windows and posting constraints. For the strict comparison, their mean overlap rates are 0.47% / 0.19%, with a p95 of 2.56% at both delays; observed zero provides no enrichment. These are empirical timing-placebo distributions, not population p-values. The timing-only four matches do not have a newly computed null; do not claim they exceed chance. The computed reading is **hypothesis_not_demonstrated**.

**This does not mean makers never react to reversals.** The posting upper bound can be later than the actual post, and the finite grid/response definition can miss other reactions. It means this narrow hypothesis is not established by this fixed sample. The count adequacy gate passes for the number of ending windows, but it does not supply observed live exposure at every reversal or make the sample representative. Do not train or infer a general cancellation rule from it.

## What the additional chronology bridge establishes

After seeing the four timing-only matches and zero strict matches, [a separately committed chronology readout](chronology_note.md) joined the original posting LOWER bound. It changes no threshold, ending, primary check or reading. All hit rows are in [chronology.csv](outputs/chronology.csv); it is post-run descriptive accounting of interval order.

- **Three of the four D0 matches remain plausible:** the reversal falls inside their posting window. A post→reversal→withdrawal chain is possible, but posting after the reversal is possible too. Two have G2 bounded-conditional post estimates; one has G4 scenario-only estimates.
- **One match is not an already-resting response under these bounds:** its reversal is at market second 172, while earliest posting is second 174.610. It may be a different action chain (posting after a reversal), not evidence of pulling an existing quote in reaction to it.
- With the one-second availability sensitivity, two of those plausible matches remain, plus the same before-earliest-post case. No private action is certified.

For orientation, one of the plausible G2 cases has post window 107.315–115.287s, adverse reversal observed at 113.000s, and ending window 113.287–116.909s, with signed quantity 220 and observed fills 100.4894 shares. It is an inspectable candidate chain, not a selected example proving a population rule. Every member of the fixed cohort remains in the outputs.

## Accomplishment and limitations

We have converted “perhaps they pull on reversal” into a declared, reproducible prediction of timing and side, checked it independently, removed one misleading near-time association, and isolated three possible chains whose missing link is specifically post-before-reversal. This adds a targeted identification question to the previous timing estimates. It does not replace them with a guessed cancel time.

The remaining limitation is observational: signed remaining quantity and disappearance make a withdrawal conditional on complete fills/no-reentry/display assumptions; estimated posting intervals do not always establish that a resting order existed at the trigger. Aggregate tape cannot certify a private cancellation or intent, and never-filled orders are absent. No profitability, model skill, queue-front or untouched-day validation claim is made.

## Verification and resource

Synthetic controls cover genuine reversal versus persistent adverse state, bid/ask sign, future-only changes, missing context, response boundaries, overlapping intervals, post ordering, exhaustion, preserved placebo counts, and a positive reading plus failing controls. Independent checking reconciles all 704 case/profile rows, all 43 signal events and 5,000 placebo rows. Seven planted corruptions are refused: omitted case, claimed private pull, wrong anchor clock, shifted window, changed match, exhaustion relabelled as withdrawal and signal-feature corruption. All checks pass on the first real run; prior packets and other-session changes are preserved.

Every Python process used PF_SLOT=codex/run_light. [Ledger](logs/ledger.log): about **0.0010 core-hours** for this packet (3.61 CPU seconds before the final ledger), against 0.015; task **1.9143/2.00**, line **5.2250/5.30**, with 0.05 reserve still fitting. Peak 131 MiB. Saved panel/CSV inputs only, no new tape, day, network, model, simulator or live action. Local commits only.

## ONE next priority

**Independently tighten the post-before-reversal link for the three plausible chains using their saved book and fill constraints, keeping all 40 withdrawal candidates and 41 exhausted controls in the coverage accounting.** Do not use the reversal itself to narrow a posting or cancellation interval. Carry the G2/G4 distinction; report whether the bounds establish pre-existing rest, rule it out, or remain ambiguous. This is a targeted hypothesis verification packet, not another horizon/threshold variant or a full condition rebuild. Stop refinement if the saved evidence cannot distinguish the alternatives; retain “possible” without turning it into a trading rule. A <=0.015 core-hour saved-data packet fits the current authorization/caps and does not need a new USER decision. New evidence/access, new days, live use or higher caps remain separate decisions.
