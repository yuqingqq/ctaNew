# COMPETING-TRANSITION-AND-CONTINUATION v1 — combined review

2026-11-23 UTC. **Full pinned target: 4eab1efe2aea01d8b7e8c7a530604537d1a87ee8**.
Range: `a91f6bb52..4eab1efe2aea01d8b7e8c7a530604537d1a87ee8`.
Source: detached checkout `/tmp/pf-competing-review-4eab1efe`. Review only; no implementation or frozen receipt changed. The previously authorized component reviewers inspected code statically; the root ran the bounded checks in the codex slot.

**Result: accept the descriptive measurements and exact-leg accounting. No material computational defect requiring a rerun was found.** The report needs the scoped interpretation corrections below. They can be published together from existing evidence; no further tuned public-snapshot variant or minor-fix testing cycle is warranted.

Population/as-of: 3,796 retreat episodes on 19 BOOK markets, Oct 24–27, four consumed days; the pinned receipts completed on 2026-11-23. There are 999 same-side responses, 89 scenario-linked response orders, 23 with exact close attribution, and six separately selected example chains. These are related observations, not independent validation cases. No new data, model, strategy replay, untouched-day test or profitability inference.

## Ranked, consolidated findings

Paths below are relative to `predictfun/reviews/maker_trace_20261122/`. A BLOCKING label applies to the named inference, not the accepted numerical tables.

### 1. P2 — BLOCKING for exact-restoration claims: the measured outcome is broader

**Evidence:** `tools/competing_transition.py:96-107,613-631,652-661`; `tools/continuation_econ.py:313-315`; `v2/report_logic_chains_v6.md:34-46,91,109-113`.

The risk differences use **all same-side responses**: 753 restoring + 139 partial + 107 overshoot = **999**. They do not use exact restoration alone. Consequently “restoration +14.8 points near END,” “restoration −9.7 points when thin,” and “restored in 26%” name a narrower target than the code measures. Exact restoration is 753 / 3,796, about 19.8%; the combined response probability is 26.3%.

The continuation sample has the same distinction. An independent, explicitly post-hoc population check finds:

| Available candidate subtype | Orders | Entry-close net through END |
|---|---:|---:|
| Exact restoring | 21 | −$1.5149 |
| Overshoot | 2 | −$8.1384 |
| All available response candidates | 23 | −$9.6533 |

The subtype recount explains why attaching the pooled loss to exact restorations can materially mislead the mechanism story. It is not a newly declared subgroup test or a new candidate.

**Disposition:** call the measured event **same-side improvement/response**, and state the economics as **pooled exact FIFO entry-close attribution of candidate response orders**. Keep the three response subtypes explicit. No new outcome fit or subgroup experiment is needed to correct the wording. Creator identity, causal response and a counterfactual quoting policy remain unidentified; a small observed aggregate loss does not establish that restoring quotes generally cannot earn profit.

### 2. P2 — NONBLOCKING for the tables: the BTC summary contradicts its computed matched reading

**Evidence:** `competing_transition.py:643-755`; report `:44,112`; saved `competing_transition/{rd.csv,support.csv,claims.json}`.

BTC-against is **adequate_direction_consistent**, with matched risk differences **+3.104 points D0 / +3.142 points LAG**, positive on all four days at both clocks. The report's detailed row preserves this correctly; “BTC and fair-value conditions give no consistent direction” does not.

The unmatched D0 comparison is negative: 22.19% versus 26.57%. It compares 392 exposed and 3,146 reference episodes. The matched common-support population is **336 exposed / 767 reference**, with harmonic stratum weighting. This is a different population and weighting scheme, so the sign reversal is not a calculation error and neither estimand may silently replace the other.

The largest contributing market supplies **59.7% of net matched RD**, but **21.0% of absolute market contributions**. “About 60% of matched RD” is numerically correct if the net denominator is named; it is not 60% of observations or of absolute evidence. The concentration and composition sensitivity justify caution, not changing a declared consistent reading into “inconsistent.” No causal BTC trigger has been established.

**Disposition:** say “BTC-with and fair-gap are inconsistent; BTC-against is sign-consistent on the matched population, with composition and concentration caveats.” Keep D0/LAG, matched/unmatched populations and the net-contribution denominator beside it. No new significance test or threshold search.

### 3. P2 — NONBLOCKING for closing this line: identify the limit of these records, not all possible public evidence

**Evidence:** report `:119,123-128`; `competing_transition.py:747-755,838-865`; contract §§2–3 and stop rule.

The justified conclusion is that **these saved aggregate records leave most maker identities, post/cancel times and queue histories tied across compatible explanations**. This is a limit of the available information, not proof that every public source or future order-level feed is incapable of identifying them. Additional days of the same format could test observable associations or supply more rare cases; they cannot reconstruct an absent owner-tagged cancellation message simply by increasing sample size.

The frozen executable aggregate stop predicate is `all_thin_or_inconsistent`: it is false because three contrasts are consistent. The implementation also conservatively requires the 3-of-4-day sign rule at **both** D0 and LAG; that decision was declared before computation. The report's statement about the aggregate predicate is correct. Clarify that the two inconsistent contrasts end their branches, and partial consistency does not authorize another variant or turn anonymous transitions into top-maker decisions.

**Disposition:** close this particular same-data identification loop. Preserve the useful observable associations. A new lifecycle-evidence task needs a USER decision; untouched days and own-order records answer different questions and should not be offered as interchangeable cures for missing top-maker identities.

### 4. P3 — NONBLOCKING: label the continuation scopes and terminal-mark convention

**Evidence:** `continuation_econ.py:21-29,242-256,481-577,581-649`; report `:72-85`.

- The headline close table is for the **89 response candidates / 23 available orders**. The **375 takes, 12 FIFO close keys and 113.8 shares** are for their **union with the six example chains**. Independently recounting the response-only scope gives **215 distinct takes, nine close keys, 91.5657 shares**; chains alone give 204 takes, three close keys, 22.2184 shares. The union correctly de-duplicates overlap. Name those scopes explicitly so “exits of these orders” is not read as referring only to the preceding response table.
- Residual is final FIFO `rq` valued at the inherited economic terminal mark, whose observation is at or before `we − 2 s`; the execution cutoff is `we`. It is not generally the same as inventory open at that exact mark time. There are zero post-END legs/closes in this packet, so no demonstrated current accounting error results. Preserve that convention when saying “residual at END.”
- The 10 s / 60 s close windows are **entry-fill-to-close settlement-clock durations**, not time since the hypothesized posting decision. The two horizons are cumulative. Their equality with all within-market closes here is a property of this sample, not a general equivalence.

These are presentation/estimand corrections. No broad rebuild is justified.

## Accepted implementation and closure of a91f6bb52

**Outcomes.** All 3,796 unique episodes independently reconcile. The mutually exclusive map keeps competitors: other improvement 1,853, other retreat 69; same restoring/partial/overshoot 753/139/107; both up/down/narrow/widen 227/220/226/2; six two-side subtypes remain unknown; 184 further retreats, seven horizons, three empty-side censors. Two-side subtype joins verify the exact frame pair and refuse disagreement. No unsupported subtype is guessed. Censoring stays separate from observed competing transitions.

**Matched statistics.** Independently reconstructed **260 pooled/day/side/market RD rows**, including censor bounds, agree. The harmonic weight is `n1*n0/(n1+n0)` on strata containing both groups. The lower/upper RD correctly combine opposite ends of each group's censor bounds. All ten clock×contrast count screens reproduce using mixed-stratum episode/response/competitor counts and the five-market rule; no all-population count is substituted. All five combined readings reproduce. All contrasts are reported; there is no post-hoc favorable-feature selection or p-value promotion. Time-left omits its own matching bin. This is observational common-support standardization, not causal adjustment for all possible confounders.

**Timing.** The preceding review's frame-gap correction is adopted: lower `max(0, response_lo − retreat_hi)`, upper `response_hi − retreat_lo`, next-frame fraction and observer endpoint gap remain separate. Their medians are not a fitted reaction-latency distribution. The fair-gap caveat correctly distinguishes earlier BTC reference from the post-retreat best.

**Continuation.** Exact maker-leg and entry/exit-role keys are used, clean status/non-self filtering is explicit, takes and closes respect END, and repeated link/order/close/take keys are de-duplicated. Candidate response price and side must match; otherwise the link is contextual. Chain 3 remains outside-window context. The previous four reuse guards are closed, including the earlier post-END counterexample class.

Independent accounting verifies all **29 available orders**: 23 response orders plus six chains. It reproduces shares, net close values for every reported horizon, earlier-inventory closes, residual marks and fee allocation; 174 clean legs are unique. For response candidates: 19 closes / 156.7651 shares / **+$2.0442** within 10 s; 23 closes / 213.9006 shares / **−$9.6533** through END; top-three entry closes **−$12.0969**; earlier-inventory closes **−$21.7656**; available residual **−$7.0970**, unavailable-cohort residual **−$25.9999**. Fees are included through the inherited convention. There is no reason to recalculate all FIFO histories.

The 66 unavailable exact links remain explicit. Exact accounting means recorded fills can be attributed to FIFO legs; it does not upgrade the scenario link from public creation to maker intent. Likewise the frozen TAKE/PULL historical associations are unchanged by this round; they are not newly validated here.

## Verification, resource charge and limitations

- Pinned suites **81/81 competing transition + 51/51 continuation** pass. These are implementation checks, not model skill or profitability evidence.
- **21 output/source hashes** agree; used economic, close, take and episode inputs also match pinned identities. Large files were streamed for hashes and read by selected columns/keys. No raw tape, full study rerun, fitting, new day or strategy simulation.
- [check_saved.py](check_saved.py), [evidence.json](evidence.json), [check_saved.log](check_saved.log), [check_take_scopes.py](check_take_scopes.py) and [take_scopes.json](take_scopes.json) hold the independent recounts and controls. Changed-RD and duplicate-close controls fire; the pinned suites include positive and known-bad cases.
- Review-instrument attempts are preserved. V1 assumed a parquet outcome file instead of the declared CSV; V2 used a case-sensitive address filter and encountered an empty-frame indexing failure; V3 attempted a unique join including blank unknown hashes. The final instrument uses the declared CSV, lower-case identity matching, typed empty masks and the candidate-only key population. These were review-tool mistakes, not development findings.
- Measured checks before staging used about **9.35 CPU seconds**, including failed attempts; peak **710 MiB**. Charge this entire review **0.010 core-hours once**, within the requested 0.015. The ledger before that charge is **1.8973 task / 5.2080 line**; afterward **1.9073 / 5.2180**, or **1.9573 / 5.2680** with the standing 0.05 reserve, within 2.00 / 5.30. Do not also add these codex logs as `mt_` compute.

## ONE next priority — an attributed order-lifecycle evidence gate, requiring the USER

**Do not launch another tuned variant on these snapshots.** The next step serving the stated top-maker logic goal is to obtain an evidence source that can distinguish the competing histories, before building another estimator. This review supplies the concrete preparation contract; acquisition and any new measurement remain unstarted.

1. **Required evidence:** one maker-attributed lifecycle stream with stable order IDs joinable to fill hashes, covering add/accept, partial fills, cancel/amend/replace and terminal status. Include unfilled and cancelled orders, original/remainder size, price/side, exchange sequence or event timestamps, and synchronized local receipt time plus book/BTC context. A fill-only export is insufficient. Identify missing event types explicitly rather than synthesizing them.
2. **Scope to approve:** one complete, predetermined capture period for a consenting account or an existing authorized dataset. Prefer already-recorded data; no live orders or exchange adapter work. For a claim about top makers, the lifecycle evidence must actually concern one of those makers or provide a defensible maker identity join. Own-order logs can validate reconstruction accuracy and timing ambiguity **only for our actions**; they cannot reveal a competitor's intent. Anonymous later days are a validation population for the observable associations, not a replacement for this evidence.
3. **Bounded gate after approval:** inventory the source and freeze a small sample before reading outcomes. Reconstruct the public-view chain blind to the private lifecycle, then reveal ground truth and count correct/false post windows, cancel-versus-exhaustion, replace links, sizes and queue-state bounds, with abstentions and unfilled orders in denominators. Exact queue rank must remain unclaimed unless venue sequence/queue evidence supports it. Use positive/known-bad cases and stop if the source lacks the necessary events. No profitability/model experiment is part of this gate.
4. **Decision needed:** USER authorization to use/provide that new identifying source and approve its scope/budget before any data work. A proposed first gate is <=0.030 core-hours all-in, subject to the actual source size and ledger at approval. No new-source access, collection, new-day consumption or live action is authorized by this review. If no such source is available, close this top-maker command-identification line and retain the present associations; a change to counterfactual strategy research would be a separate USER decision.

**Largest evidenced limitation:** correct public-fill accounting and stable aggregate transition associations still leave the private maker action chain unidentified. The missing information is the lifecycle-to-maker link, not an arithmetic fix or another adjustment to the same support threshold.
