# R4E combined review — reliable negative entry result; audit the target before another fit

Reviewed 2026-11-21, evidence checked through 07:35 UTC.

**Full target SHA: `141e820e4b736c6a8c4f27b0a0dbf53789e95e89`.**
Range: `371aab568277dea8ed15c0e386b8c07e7bacd340..141e820e4b736c6a8c4f27b0a0dbf53789e95e89`.
Pinned source: detached `/tmp/pf-review-141e820e`. Development owns implementation. This review changes only review evidence and review-owned state inserts; no push.

**Accept R4E as a reliable negative development experiment. No BLOCKING finding for its main readings. Neither ADV2 entry candidate has demonstrated useful entry selection. Do not promote SMOOTH or test it on protected days.** This is not a general certification of the simulator or proof that market information cannot predict entry value.

The one next priority is **ENTRY-TARGET-AUDIT-v1: decompose the existing POST/WAIT labels and fill traces before choosing a different entry target or decision unit**. Stop further tuning of this direct 50 ms advantage-regression line while doing that bounded measurement. No additional USER approval is needed for the contract below. New label dates or untouched-day validation require the USER; neither is recommended now.

## What reproduced

| Measurement | Independently checked result |
|---|---|
| Existing evaluation population | 12,088 POST rows; R3 keys, labels, weights and RICH predictions exactly equal |
| New model error / zero | 1.003989, 0.998134, 1.000104, 0.995667 for Oct26 bid/ask, Oct27 bid/ask; lower than RICH in 4/4, lower than zero in 2/4 |
| Entry selection | 4,122 admissions; positive selected mean 0/4, POST screen PASS 0/4, above matched p95 0/4 |
| Weighted local policy value | −0.4712, −0.0025, −0.3859, −0.2306 c per sampled opportunity; below never-POST in 4/4 |
| Replay population | 384 planned, 380 ok: Oct26 188, Oct27 192; 3 missing-endpoint and 1 stale-future-book world retained as statuses |
| SMOOTH − ENTRY_ONLY | +163.586 c / +883.715 c, Oct26 / Oct27 |
| SMOOTH − PI0 | −2,206.629 c / −1,525.359 c; SMOOTH itself −2,471.698 c / −1,493.786 c |
| Replay generations | 7,252 SMOOTH versus 9,356 ENTRY_ONLY, 22.49% fewer |
| Churn with PI0 retention | SMOOTH pulls at the first live review in 3,325/3,386 reviewed generations on Oct26 and 3,491/3,555 on Oct27; fewer entries did not resolve rapid pulling |
| Reached comparisons | 760 ok POST pairs, 29 with abs(Y)>1e−6 c; both models reloaded on identical stored inputs; SMOOTH beats zero on neither day |
| Reproduction controls | N0, PI0 and ENTRY_ONLY equal R3 by value, decisions and fills in all 1,140 policy/world comparisons |

All quantities are gross cents with the frozen zero-cost convention and the original five-second END mark. These are **adaptive, consumed-day** measurements: Oct24–25 training, Oct26–27 development evaluation. No day-level interval, settlement-profitability claim, untouched validation or full R4 qualification follows. Improvements relative to a losing policy are not positive entry skill.

## Consolidated actionable checklist

Blocking means materially invalidating the present readings, not merely preventing model promotion. All items below are **NONBLOCKING for accepting this negative measurement**. Fold cheap corrections into the next measurement; do not start another cleanup-only cycle.

1. **P1 — NONBLOCKING: the entry model still fails the economic selection test.** The smoothing package at `pf_adv2_entry.py:33–47` reduces error but all four selected-POST means remain negative (`r4e_entry_smooth_20261121/report.md:95–112`; `pf_adv2_entry_eval.py:129–141`). On reached states it beats zero on neither day (`report.md:269–279`). First-review pulling still dominates with PI0 retention (`report.md:240–255`); compare it with ENTRY_ONLY's identical retention, not R3 RICH's different KEEP model. Fewer orders and smaller losses do not distinguish useful timing from reduced exposure. The paired replay fairly measures the *whole package* versus ENTRY_ONLY; it does not isolate why the improvement occurred. **Action:** close the current smoothing experiment as failed, retain all candidates/multiplicity, and perform the target audit below. No threshold mining from the favorable calibration bin, extra regularization sweep, or unsupported claim that entry prediction is impossible.

2. **P2 — NONBLOCKING: the label's narrow marginal question remains unresolved, and a few table names still mix estimands.** `pf_r2_labels.py:191–207` forces POST versus WAIT only at the fork and resumes PI0 in both arms on the following ticks. Y is a *whole-branch* END-value difference, not the standalone value of the new order, and WAIT may later enter. Therefore a near-zero label does not by itself mean the market state lacks economic information. Conversely, calling this a bad target requires evidence; two failed regressors do not establish that.

   The R4E table generator still calls the first-review score a “KEEP advantage” (`r4e_tables.py:171–179`, rendered receipt `report.md:240–255`). In SMOOTH and ENTRY_ONLY, entry uses paired advantage A and retention uses PI0's J-mapped **V_retain**, not KEEP−PULL; PI0's own entry is **V_post**, not POST−WAIT (`pf_adv2.py:306–331`, `pf_r2_teacher.py:27–44,406–423`, `pf_adv2_replay.py:235–254`). The raw columns already preserve `by_post` / `by_review`; no data rerun is needed. **Action:** name the score and policy per leg. Keep the corrected statement that these are different targets, and do not subtract their scores as a temporal forecast change.

   Also label the joined predicates table's `realized` column **mandatory-anchor control value**, separately from **all-sampled selected-POST weighted mean** (`pf_adv2_entry_eval.py:134–141`; receipt `report.md:80–91`). Their populations differ: the Oct26 ask control value +0.0636 c and negative selected-POST mean are not contradictory. This is a presentation clarification, not an arithmetic defect.

3. **P3 — NONBLOCKING: replay check F has a narrower contract than the retained trace permits.** It recomputes cash/position, callback counts and value but does not check episode END, settings, fill-time bounds or equality of marks across policies (`r4e_checks.py:122–149`). A rehashed copy with SMOOTH's trace END changed to 4,999 still passes F; see [receipt_checks.json](receipt_checks.json). This is a real checker gap, **not a demonstrated error in this run**: the independent review checked all 1,520 episode traces and found END=5,000, settings equal to the frozen plan, common finite marks, fills inside [0, END], and recomputed values correct. The reached-pair validator already checks these fields (`pf_r2_label_sample.py:483–527`). **Action:** add these inexpensive episode predicates and single-field falsifiers if that checker is reused; otherwise retain this review's verified workaround. No relabel/replay or separate fix round is required.

## Prior review and amendment closure

| Prior requirement / deviation | Disposition |
|---|---|
| One smoothing candidate, two POST heads, unchanged rows/features/weights, PI0 retention | Closed. Leaf minimum 1,000 and 50 rounds are the only declared changes; two new model hashes and all training extraction hashes match. No new KEEP model. |
| Different targets; hybrid losses not additive | Main correction closed, including FINDINGS P92. Remaining PI0 score nomenclature is item 2 above; no change to reported arithmetic. |
| State strata and both models on identical reached inputs | Closed. Strata/coverage tables reproduced and all 760 common-input predictions checked. Sparse evidence remains a scientific limitation, not a failed implementation. |
| Full episode and reached-arm traces | Closed. All fills, decision logs, sizes, callback/audit counters and label identities checked; item 3 records the remaining checker boundary. |
| Calibration pooled by weights | Closed. All bins carry w_sum; tables reproduce. R3's corrected pooled RICH table corresponds to the same predictions. |
| Part identity, exact membership, separate launcher | Closed for this run. All 12 parts' source/model/plan identities and ordered memberships match. No shared launcher reuse in this round. |
| Parser amendment | Accepted. `5c2ec2ca1aa496591c95d22034547e95ddb03e1b` changes only the evaluator's round-trip CSV reader and check C's readers, plus its declared test/doc changes. The exact equality predicate is unchanged; no evaluation output precedes the amendment. |
| Summary writer correction | Accepted. `c69ea5c68738e4453821dceba5027f3deb8058f9` replaces the absent top-level binding column with trace-derived binding totals. All four regenerated CSVs are byte-identical; replay.json reproduces. The v2 synthetic test executes the writer and rejects altered CSV/part bytes. |
| Falsifier correction | Accepted. v2 searches for an actual fill, records application, and fails unapplied corruptions. Clean copies pass, all declared corruptions fail. An additional round-trip-preserving label falsifier confirms rejection is caused by the changed label, rather than float-parser noise. |

Chronology checked: candidate freeze `602fdfc5c3441e5c58b39dc4ffd01e4b541daf1b` at 06:55:41; model commit `eaea1161dc39e8048655ebcd775fbf6004b18014` at 06:56:07; parser amendment at 06:58:57; evaluation as-of 07:00:08; writer amendment at 07:22:55; result commit at 07:24:19 UTC. The A–I predicates are unchanged **after the declared parser amendment**. The earlier failed evaluation, summary and falsifier runs remain provenance; they are not presented as successful runs.

## Independent checks and limits

[verify.py](verify.py) imports the pinned worktree; each execution ran through `PF_SLOT=codex /home/yuqing/ctaNew/predictfun/research/run_light.sh` with an absolute script path and the existing venv interpreter.

- [metadata.json](metadata.json), with [metadata2.log](metadata2.log): 36 output hashes, both model hashes, training extraction identity, unchanged engine/teacher/labels/R3 sources, the exact reader-only amendment, and final-source mutation records: 16 + 15 + 19 + 8. Copied-record falsifiers fail.
- [evaluation.json](evaluation.json): streamed every saved evaluation part, checked the accepted dataset hash/part hashes, rebuilt all 12,088 POST predictions, all seven tables and every 500-draw matched control. Independent weighted-error/value formulas agree.
- [replay.json](replay.json): all 12 parts; 1,520 episode ledgers; 660 recorded fills across the four policies; 254,405 required episode callbacks; 7,891 episode causal checks and 5,255 parity checks, no failures; 760 reached pairs, 5,379 causal and 5,077 parity checks, no failures. Raw fills reconcile cash, inventory and END value. All 1,140 R3 controls and four CSV byte comparisons match.
- [synthetic.json](synthetic.json): evaluator, replay and replacement-summary synthetic controls pass. Stub heads/cached synthetic tape only. The candidate selftest contains fits, so it was **not rerun**; its source and final-source records were reviewed instead.
- [receipt_checks.json](receipt_checks.json): A–I reproduce, all real-artifact falsifiers pass, and the END-counterexample documents item 3. No saved development artifact was modified.

Five successful bounded jobs took 102.49 seconds total wall time, peak 349 MiB. An initial review-only assertion incorrectly compared check C to the pre-parser freeze and failed; [metadata.log](metadata.log) preserves it. Corrected to explicitly verify only the declared reader changes and byte identity after that amendment. Including this failed review job: 103.96 seconds. No fits, label builds, real-tape replays, full experiment or protected-day access were run. Previously reviewed tape/engine assumptions are inherited; this is not a fresh exchange-simulator validity study. Passing these checks establishes faithful measurement, not profitability.

## ONE next priority: ENTRY-TARGET-AUDIT-v1

**Decision:** stop further tuning/promotion of the current direct POST−WAIT regressors. Use existing traces to choose whether the next entry experiment needs a different target or decision unit. Do not spend more label days or an untouched test on either failed candidate. This is one bounded diagnostic batch, not another model candidate or a fix-only round.

Development's first reviewable deliverable is a frozen audit builder, a small synthetic reconciliation fixture and **one combined measured receipt**, under this contract:

1. **Population and identities.** Stream every accepted POST row from R2-F's Oct24–25 training dataset `2c62fc9baf696e6cd8913c5c294dea2c9513640714ae34b413dc37ef8c0acb2e` and R3's Oct26–27 dataset `902526da20ecea3c224dab403135f701ac47386a72b0779723d36077b2553572`. Preserve all statuses and sampled zeros; use the frozen row keys/weights. Join existing predictions by exact keys, not row position. Include R4E's 760 reached pairs as a **separate world/side-balanced diagnostic**; do not mix them with the weighted neutral sample. Report mandatory and uniform-extra picks separately as well as the declared weighted total. No outcome-selected training or fresh tape access.

2. **Explain zero and nonzero labels with an exhaustive accounting partition.** After cancelling the identical pre-fork fills, distinguish: neither arm fills; economically identical post-fork fills; different fills with the same cash and inventory; different cash/inventory offset by the common END mark; and meaningful unequal values. Canonical economic-fill comparison may discard arm-local gid/time only for the explicitly named economic-equivalence diagnostic; retain the timed fill comparison beside it. Report exact zeros and abs(Y)≤1e−6 c arithmetic residues separately. An equal payoff does **not** prove the latent states reconverged. Unknown/unclassifiable evidence is an explicit status.

3. **Decompose the forced entry's contribution without inventing a counterfactual.** In the POST arm, locate the forced generation by the fork tick, focal side and the stored POST decision's gid; require a unique match. Define `V_initial` as that generation's fills valued at the existing common END mark. Define `D_following` as all other POST-arm fill values minus all WAIT-arm fill values, splitting later focal generations and the opposite side. Verify **Y = V_initial + D_following** on every row to the frozen ledger tolerance. Prefork values cancel. Never join gids across arms, and never call `V_initial` a deployable entry advantage: it omits replacement/opportunity costs. Report WAIT's next actual POST delay/no later POST and the initial generation's fills/no fills. Refusal, queue convergence or precise removal causes may only be asserted if the stored evidence distinguishes them; no-fill is not proof of refusal. Do not rerun tape merely to fill those optional fields.

4. **Useful outputs, not another tuning exercise.** Per day and side, report row/original/world counts, weight totals, the taxonomy shares, positive/negative/meaningful counts, weighted means and absolute-value totals of Y, V_initial and D_following, their offsetting contribution, and tail concentration. Use the already declared source, remaining-horizon, generation and other-side strata; show sparse counts rather than claims. Cross-tab the frozen predictions/actions against these components, with no learned thresholds or feature search. Separate in-sample Oct24–25 from adaptive Oct26–27. Do not fit on components, declare a profitable component, or replace Y with it in this round.

5. **Acceptance and controls.** Freeze the audit code/definitions before reading its new outputs. Verify dataset/part identities; every POST row appears once; exclusions reconcile; no row is dropped for Y=0 or missing predictions; all accounting identities hold. Synthetic cases must cover no fills, delayed WAIT entry with the same economic fill at another time/gid, initial gain offset by later loss, initial loss offset by later gain, and cross-side effects. Include positive controls and altered side/quantity/mark/gid-membership refusals. Bundle item-2 nomenclature corrections and any cheap item-3 checker reuse improvement. No intermediate review required.

6. **Bound and stopping point.** One core through the existing approved launcher; stream one partition at a time; planning ceiling **one core-hour**, 1,536 MiB checkpoint and 2,048 MiB hard RSS limits. Stop with explicit incomplete status if exceeded, not a favorable subset. No model fit, new label, changed continuation, changed END, cost/latency alteration, or real policy replay in this batch. Deliver hashes, counts, reconciliation results and a concrete target recommendation together.

The audit must answer **whether the observed difficulty is mostly rare incremental execution, cancellation of initial value by continuation, or both**. It must permit “not identified” rather than force a story. If the initial-generation component is itself mostly uninformative, relabeling it will not solve the problem. If continuation dominates, a larger decision unit is worth declaring next, but its alternative action and opportunity costs must be specified before building any labels. A richer hurdle/risk model, different continuation, or new target is a **new candidate**, never a retroactive repair of R3/R4E.

**Approval boundary:** this existing-data audit and stopping further tuning of these failed candidates need no additional USER approval. A later, concretely declared consumed-day target/model experiment can stay within the standing development loop and its agreed budget; this report does not yet launch it. **Additional label dates, protected/untouched-day access (including Oct28), or expanded resources need the USER's explicit scope/budget decision.** No untouched validation of the current candidates is warranted; eventual validation requires a separately frozen full pipeline and at least five complete untouched UTC days. This recommendation does not abandon all market-making or all entry research.
