# MATCHED-MARKET-TURNOVER-GAP-v1 — independent review

Date: 2026-11-23 UTC. Full pinned target: **55625bfa6ce8cf44002c9cc23b8f4b2d79a82700**.
Range: `8f2a2e753..55625bfa6ce8cf44002c9cc23b8f4b2d79a82700`.
Source reviewed in detached checkout `/tmp/pf-gap-review-55625bfa`. Development files and frozen receipts were not edited. Our design notes in this range are inherited instructions, not new experimental results.

## Verdict

**Accept `v2/matched_gap_r1` as a descriptive comparison under its declared clocks and accounting. No new result-blocking defect found.** The carry repair is complete on this population. Both output CSVs reproduce byte for byte; the answer JSON, source hashes and input hashes agree. A separate deque/FIFO calculation agrees on all **208 active party × market × view rows**, maximum absolute deviation **7.28e-12** across the tested USD/share fields.

The supported result is a **completed-turnover gap on the matched markets**. The comparison does not identify which posting, retention, taking, inventory or simulator mechanism caused it. In particular, the report's assertion that changing inventory management cannot close the gap is unsupported.

Population/as-of: 19 previously consumed BOOK markets, Oct 24–27, 2026; corrected artifact built 2026-11-23 07:56:59–07:57:01 UTC. Rank 1/top 3/top 10 comparisons have 19 both-active markets, rank 2 has 11, rank 3 has 17. Four days, point estimates only. No training, policy replay, new data or untouched validation in this review.

## Consolidated actionable findings

1. **P2 — NONBLOCKING for the queued estimation packet: the inventory prescription exceeds the decomposition.**

   Evidence: `predictfun/reviews/maker_trace_20261122/v2/report_matched_gap.md:35-36`; `tools/matched_gap.py:769-820`.

   The frozen `our_turnover_loses` condition is simply negative own completed-close value and positive benchmark completed-close value. It does not test an inventory intervention. FIFO completed and residual components are accounting classifications of the realized paths; inventory management can change entries, exits, quantities and which positions close. A profitable residual mark contribution also does not establish lower inventory risk or better inventory management.

   **Bundle an in-band wording correction:** “On these saved paths, our completed closes lose while the benchmark's completed closes earn; our residual mark contribution offsets some losses. Which control decisions cause the turnover gap is unmeasured.” Replace “residual exposure is smaller and better” with the measured residual **P&L contribution**. Withdraw “Fixing our exposure would not close it.” No rerun or inventory-policy experiment is required for this correction.

2. **P2 — NONBLOCKING for descriptive use: put the comparison's clock and execution qualifications beside the answer, not only in its contract/source.**

   Evidence: `report_matched_gap.md:15-18,34-38`; `tools/matched_gap.py:12-32,464-505,556-594,633-690`; `pf_mt_econ.py:404-409,579-585`.

   The same numerical interval and END mark are correctly used. They are **not identical physical execution clocks**: our fill clock is the simulator's assumed trade-anchor-minus-50-ms time, while theirs is the UUID settlement upper bound. In the 20,932 status-ok top-10 BOOK leg rows inspected, **69 have a retained lower spread anchor before start and settlement at/after start; 20 span END**. These are row counts, not distinct transactions or proof of the actual crossing time. Another 92 have no earlier anchor. There are no recorded top-10 legs exactly on start/END and no own fill whose engine time plus 50 ms passes END. Thus the inclusive-boundary implementation is correct, but physical-window membership and subsecond FIFO timing are not established by it.

   Also disclose: our passive fills come from the existing replay's queue/public-flow assumptions and start flat; the benchmark has recorded maker and taker fills, different sizes, participation and holdings. Theirs pays recorded fees converted at fill price, ours has the engine's zero maker fee; no assumed rebate is credited. This is an explicitly declared cost convention, not exact cash-plus-fee-token wealth at END. Revaluing the in-window fee-token position at END would change total value by approximately **−$0.97 / +$1.51 / +$0.04 / +$0.58 / +$2.39** for rank 1 / rank 2 / rank 3 / top 3 / top 10, respectively. It is small here and does not warrant rebuilding this receipt.

   **Bundle into the same correction:** call this “same-market, same numerical interval/mark, simulated-versus-recorded comparison”; explicitly retain the assumed clocks and fee convention. In the queued estimator, carry boundary-sensitive records and timing alternatives as such. Do not infer precise posting/cancel times, causal queue advantage, or achievable profit from this table. No blanket simulator fix is demanded by these qualifications.

No additional fix-only round is requested. The two items change interpretation, not the accepted saved numbers.

## What was verified

**Interval, price and accounting.** Inclusive `[start, END]` was settled in amendment 1 before measurement and matches the engine's endpoint rule. The adapter checks start/end/path identity, rejects out-of-episode own fills, converts own prices/marks from cents to USD, and reproduces own cash, position and value exactly. Both parties use the recorded own END mark. Opening marks only rebase pre-start carried positions. FIFO is per address; pooled parties sum members and never match inventory between different makers. Real fees are prorated between closes and residual legs; self-trades stay outside FIFO with their own gross/fee lines. No markout or resolution horizon was substituted for END.

**Carry repair.** `their_member` / `combine` now account for carried lots of a member with no in-window activity when that pooled party is active; the direct party check sums carried positions over every member. The cross-view check verifies carry minus zero equals recorded opening position times the END/open mark difference. The synthetic suite plants the original omission and rejects it. Independent accounting includes these members as well. Only top-10 carry economics change relative to the retained defective run: the net correction is **+$21.877**, with offsetting large market-level changes (e.g. −$710.89 and −$704.54 in two Oct 24 markets). Do not treat a small pooled correction as evidence that the original defect was harmless. Rank 1–3 have no pre-start fills under the declared settlement-clock assignment, so their two views coincide. The first run remains defective provenance; use `matched_gap_r1`.

**Readings.** The frozen first-match precedence and five-both-active-market screen are implemented as declared, including exposure-gap precedence when both turnover signs agree. Independent recomputation reproduces every per-market classification. Pooled `our_turnover_loses` holds for all five benchmark parties; this pooled secondary reading does not mean every market has that reading. The report's per-market counts and rounded answer table agree with the corrected outputs.

**Role mix.** Own completed closes are all maker→maker. The saved role table also shows positive benchmark maker→maker completed-close net: **+4.49 / +4.45 / +4.06 c/share** for ranks 1/2/3 (20,316 / 14,928 / 23,149 matched shares, rounded), **+2.68 c/share** for top 10 (107,187 shares), versus our **−1.90 c/share** (39,698 shares) across all 19 markets. Each benchmark uses its both-active market set stated above. This is a descriptive cross-tab of the already frozen output, not a new test or a causal passive-exit prescription. The role mix alone therefore does not remove the observed gap; sizes, prices, timing, selection and simulated fills remain different. Unclosed inventory remains in the residual and must stay beside these selected completed-close statistics.

## Prior findings and scope

- Prior review `8f2a2e753` exit-role causality, opposing-side versus actual absolute reduction, identifiability-fixture wording and `good_turnover_lost_to_residual` interpretation: **correctly superseded** in `v2/report_interpretation_v3.md`. The published inventory recount matches the earlier verified numbers. No economics rebuild was claimed.
- Prior latent receipt-earlier-than-UUID lifecycle case: **open, explicitly carried**, absent from the prior accepted saved population. Before the estimator admits such an input, apply a correct clock status/refusal or a bounded documented workaround. This is not a reason to rerun the accepted comparison.
- Earlier fill-relative features and outcome-selected decision-grid limitations remain. Relabelling suffices for descriptive historical examples; it does not create an unconditional action dataset or predictive evidence.
- The existing simulation itself was not revalidated in this saved-accounting review. No claim of transferable maker behavior or live profitability follows from passing checks.

## Bounded verification and provenance

- Pinned `matched_gap_selftest.py`: **393/393**, including carried-only omission, fee/accounting and boundary controls; [selftest.log](selftest.log).
- [check.py](check.py): no-write reconstruction of the accepted saved inputs, byte-exact CSV/answer/hash comparison, independent deque accounting, frozen-reading recomputation, old-versus-corrected comparison, fee-token and boundary diagnostics. Positive hand-computed close and negative missing-carry/fee-sign controls included. [evidence.json](evidence.json), [check.log](check.log).
- [summarize.py](summarize.py) / [summary.json](summary.json): compact accounting, role, clock and resource evidence. The old/new row-diff list also flags empty-string versus CSV-NaN `market_reason` representation; it is not an economic change. Summary total changes select finite, nonzero deltas.
- Two review-instrument first attempts failed before writing their results: tuple/list concatenation in the optional extra-column read, and attempting `abs(None)` on inactive-row deltas. Their logs are retained; the narrow instrumentation fixes did not change the tested development module or its outputs.
- All research checks used `PF_SLOT=codex` through the absolute `run_light.sh`. No other slot, full replay, raw tape experiment or fit was used. The detached source and reproduced source hashes bind this review to the full target above.
- Conservatively charge **0.010 core-hours** for this review, including failed checks and staging/verification; measured analytical user CPU is about 7.4 s. At the ledger snapshot, task spent before this charge is **1.7080 h**, line **5.0187 h**. Adding this charge and the entire queued packet's **0.035 h including its own review** yields **1.7530 / 1.85 task h**, **5.0637 / 5.25 line h**. Update the live ledger for concurrent work; do not charge the already booked prior-review allowances again.

## ONE next priority

**Complete the already user-authorized bounded MAKER-ORDER-ESTIMATOR / decision-logic packet**, incorporating the two wording corrections above in that same development round. Follow the existing [timing plan](../20261123_maker_decision_timing/plan.md), [logic amendment](../20261123_maker_decision_logic/plan.md) and [estimator requirement](../20261123_maker_order_estimator/plan.md); do not launch another model or policy variant from this comparison.

The useful output is a small set of predetermined connected histories: estimated original post size/price/window → partial fills and remaining queue → keep or estimated pull amount/window → competing repost/overlapping-ladder histories → subsequent opposite-side trades/takes and inventory. State which evidence distinguishes each history, where parsimony selects a representative, and where alternatives remain tied. Attach the longer-horizon full-book turnover/residual economics only after fixing the histories; do not choose latent commands to maximize apparent profit or anticipation. Use the present gap to focus the explanation on how orders enter and complete profitable turnover, while retaining inventory decisions as a possible mechanism.

Keep the fixed top-three/19-BOOK population, deterministic **≤96** detailed cases, existing own-order/synthetic truth controls, identical-public-tape exhaustion-versus-cancel and repost-versus-preexisting-ladder controls, no new data/fit/replay, and **≤0.035 core-hours all-in including 0.010 review reserve**. Freeze inference rules and selection before the measurement; begin with the smallest useful prototype and honor the projection gate. One combined acceptance request, with all material issues bundled. If public evidence leaves a mechanism unidentified, quantify and show the competing histories once rather than spending repeated rounds forcing precision.

**No additional USER decision is needed within that existing authorization and remaining ledger.** More resources, new days, untouched validation or live use still require the USER.

**Largest evidenced limitation:** a sound matched accounting gap has been found, but the comparison still contrasts simulated fills with recorded fills under different clock/execution assumptions. It has not identified the actual maker decision mechanism responsible for the better completed turnover.
