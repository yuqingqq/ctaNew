# New-results review: useful priority, cancellation and the entry clock

Review cutoff **2026-11-12 10:40 UTC**. Read the new B21/B21b, keep-or-pull S1/S4, full default-pull portfolio, pull-quality, independent/joint fill models and QR queue-race results. This review preserves their original outputs. It adds one deterministic synthetic clock audit; it does not fit a model, replay historical markets, inspect protected payloads or change live code.

**The evidence supports a more specific objective: acquire priority early enough, retain it when it offers worthwhile fills, and withdraw before avoidable harm. It has not produced a profitable default-rest replacement. There is also a newly verified causality defect in the fast-entry speed experiment, so its 30 ms result is not an execution target yet.**

## 1. The latest models reproduced the churn problem

The complete default-pull portfolio result is as of **05:42:48 UTC**, six consumed dates (Oct 26, 27, 31; Nov 1–3), **1,728 windows per policy/scenario**. The matched guard has the same single-layer size and inventory reservations as the new policies, making it the more useful policy comparison than the larger frozen ladder.

| Policy versus matched guard | Primary marked delta; common / missing windows | Stress marked delta; common / missing windows | Inventory-square time change, primary / stress |
|---|---|---|---|
| Pressure rules | +$29.98; 1,557 / 171 | -$105.76; 1,514 / 214 | -12.38% / +16.87% |
| Learned queue model | -$592.27; 1,508 / 220 | -$1,100.40; 1,528 / 200 | +80.13% / +26.32% |

These are paired changes on common windows with valid stop marks, not per-day returns. All six learned-policy daily deltas are negative in both scenarios. Pressure's primary estimate is uncertain and reverses under stress; it is not a promoted winner. Stress changes several execution assumptions together, so it does not identify a single latency cause.

The primary queue policy model-pulls **199,713 of 222,617 submitted generations**, with reported median simulated lifetime **132 ms**. These are eventual pulls per generation, not independent per-decision probabilities or elapsed-time presence. The model can repeatedly rescore an order, then re-enter after a timer without evidence that the danger cleared. This directly supports the user's concern: calling the architecture default quoting did not give it useful persistence.

S4 independently reaches a similar policy failure on nine consumed days/1,099,880 neutral rows, but uses a different first-minute, five-second-label simulator. Its numbers should not be pooled with the six-day full-portfolio study. Both studies show that local prediction skill versus random actions is insufficient evidence for a profitable sequential policy.

## 2. More residence and a faster order path were insufficient

B21, as of **02:25 UTC**, uses **220 consumed Nov 8 windows, first minute only**, one five-share order per side. Several signal arms remain present for 56–92% of simulated side-time, yet all eight default-rest arms lose at all three fill-clock assumptions. B21b removes own placement/cancellation/fill-notice latency on that same cohort and still does not rescue those arms. Its information/feed delay remains unchanged.

This rejects the tested combination of default residence and those triggers. It does not prove every early-resting policy is unprofitable, nor that faster information would have no value. Zero order-path delay is an optimistic execution assumption, not a mathematical upper bound on P&L: faster entry can admit additional adverse fills.

The perfect-foresight S1 policy is likewise a diagnostic. It demonstrates favorable action selection within the simulator when future outcomes are known. It does not establish that today's observable inputs can predict those outcomes, or that its greedy policy is a global profit bound.

## 3. QR changes the priority of the queue diagnosis

QR's saved result is as of **10:30:49 UTC**. Its nine-day replay decomposition covers **133,683 v3 orders at the best at activation**, with 44,279 off-best orders outside that decomposition; refusal/censoring statuses are retained. Its separate Nov 8 live-order analysis covers **13,616 at-best accepted orders in 220 windows**, out of 30,221 accepted orders examined (2,506 outside the first minute, 14,099 off-best).

Under its primary queue assumptions, approximately **41% replay / 44% live** of estimated queue ahead is assigned to joining after the earliest assumed feasible arrival, versus **15% / 14%** to same-price replacement chains. The live first-order acceptance delay from a level becoming best has median **386 ms**. These support investigating entry timing alongside persistence; reducing cancels alone cannot remove a queue already ahead of the first order.

These percentages are a **model-based decomposition**, including for actual live order timestamps. The fixed 97 ms boundary, ghost queues, snapshot allowance, matching of replacements and selected at-best population determine the attribution. They are not identified causal P&L contributions or evidence about individual competitors' measured latency. In particular, they do not prove PM90 alone caused the late arrivals.

The distinction worth preserving is **priority acquired early versus priority obtained when others leave**. A small queue ahead can accompany a harmful sweep. We should not reward age or apparent proximity to the front mechanically; evaluate the subsequent fill value from the same decision state.

## 4. New finding: QR's fast-entry clock uses future price information

The exact source at `a3f93fff`, `pf_qr_queue_race.py`, constructs birth-entry schedules as:

```python
send_time = birth_stamp + requested_live_delay - L_PLACE
# L_PLACE remains 40 ms
```

It assigns the new birth-frame price to that scheduled decision. With a **30 ms** requested live delay, submission is **10 ms before the birth**. The same issue affects the 15 ms arm; the 0 ms arm was already described as an idealized bound. The 30 ms arm was presented as a prospective reaction speed, which its unchanged order path cannot implement.

New checker `pf_qr_clock_review.py`, source committed at **9053263d before its run**, uses QR's actual birth detector and simulator. Two synthetic books are identical through 95 ms; at 100 ms one develops a best bid of 51 and the other 52. The supposedly 30 ms policy submits at 90 ms with price 51 versus 52, becoming live at 130 ms. Changing the future therefore changes the earlier action. This is a confirmed causal-clock failure, not an assumption about profitability.

The existing QR synthetic suite and all four mutations still pass. The added future-input test flags the problem; a 150 ms schedule preserves the pre-birth action prefix, and the independent availability checker accepts the causal schedule and refuses the future-reading and unsorted schedules. Receipt: `reviews/results_review_20261112/qr_clock.json`, as of **10:38:34 UTC**.

This qualifies the **actionable fast-entry performance interpretation**, including the claim that the 30 ms arm establishes a profitable speed target. No historical bias size or direction has been estimated. QR's part A/C queue decomposition is a separate calculation and is not invalidated by this scheduling defect. The observed snapshot spacing also remains a descriptive fact; it does not establish a universal latency threshold or sub-snapshot queue order.

A corrected speed study must separately specify when the birth becomes observable, when the decision is made, and when the order reaches the venue. If placement is accelerated, change that leg explicitly; never move a price-dependent decision before the information exists. Guard inputs and exits need the corresponding causal clocks. Preserve the original receipts and publish corrected results as a superseding version.

## 5. Better fill-rate estimates have not solved economic selection

The joint-outcome study, as of **10:11:07 UTC**, evaluates **40,059 neutral eligible actions on six consumed days**. It now produces coherent probability/quantity/value forecasts, and calibration brings aggregate KEEP fill rate to **27.49% predicted versus 27.05% observed**. That is a useful repair under the simulator's labels, not a live fill-model validation.

The economic decision is still wrong in aggregate on the **5,290 selected keeps**: predicted KEEP advantage **+$19.89 equivalent**, realized **-$27.62**. These are overlapping local counterfactual utilities, not portfolio dollars. At matched pull counts, the candidate is **-$11.93 equivalent versus guard-margin ranking**, and does not improve on the direct model. No sequential portfolio has been run for this new joint candidate. Better marginal fill frequency does not establish that the model distinguishes favorable from adverse fills.

Queue proxies do add information about existing-order fill opportunity. The evidence therefore does not justify simply dropping order state because one prior model used it badly. The unresolved quantity is the expected net value conditional on those fills and the alternative policy.

## Research implication

Keep the agreed objective of early quoting and useful priority; the old guard remains a benchmark, not an automatically restored architecture. Hold size/layer count fixed while addressing two distinct requirements:

1. **A causal acquisition experiment.** Measure arrival at newly observable levels with feasible feed/decision/place clocks. Separate entry timing from pull policy so that each effect is identifiable. The current fast-entry results cannot select a deployment latency.
2. **An aligned KEEP/PULL/recovery comparison.** The label and replay must use the same re-entry condition, price and delay. An order should not be repeatedly reintroduced merely because a timer expired while the hazard persists. Count unavoidable cancellation-prefix fills on both alternatives.

A short danger horizon can target avoidable imminent losses, but it should not erase the value of retained priority beyond that horizon. The current short local labels give zero value when neither branch fills before the cutoff even if queue progress differs; the 1s/5s markouts value fills already selected inside that cutoff and do not extend it. Test a declared continuation horizon or a separately validated tail value, without adding an arbitrary queue bonus or counting future fills twice. This limitation is structural; its contribution to the observed losses remains unmeasured.

The current 250 ms danger proposal also deserves explicit action-unit and latency checks: nonoverlapping time bins are not automatically distinct order generations, a pending create may have a different effective-cancel delay, and future-data invariance must hold for the whole pipeline. A frozen market-only trigger can still flicker; recovery gating removes forced timer re-entry but does not guarantee zero churn. These are design qualifications, not newly measured effects.

Live own-order match/cancel timing remains the common calibration gap. All economic results above use consumed history and imperfect queue/fill assumptions. No new strategy is ready for promotion on this review.

## Verification and saved state

Verified the complete B21, S4 and QR SHA256 manifests, the full-portfolio result against its independent audit hash, and the joint-model result/review against its publication receipt. Inspected the actual QR schedule and full-portfolio label/reset code. The only new numerical job was the 0.37 s synthetic clock audit (106 MiB peak); no historical result was recomputed. Original studies, models and reports are unchanged. The machine review receipt explicitly names the QR artifact and the causal interpretation it supersedes.
