# MAKER-TRACE-RECONSTRUCTION-v1 — independent review

2026-11-23. Review-only; no implementation, model, simulator or development-test changes.

**Pinned target: `1af85c5691d5ca4c2f9565d94a13cd9680688d3c`.** Range `5927971d..1af85c5691d5ca4c2f9565d94a13cd9680688d3c`, including the build contract and its pre-run amendments. Source inspected in detached checkout `/tmp/pf-mt-review-1af85c56`; data read from the original repository. Governing scope: `8b39f9e8` and `9ff1e197`.

**Concurrent authorization update:** after the numerical review, HEAD advanced to `9da9c2e61`, recording USER approval for the lifecycle follow-up and an **ex-ante top-three decision-path study**, task cap **1.85h**, line **5.25h**. I read that amendment to settle the next priority; this report still reviews the pinned v1 implementation/results, not the new study's implementation. Existing changes are preserved.

## Disposition

**Accept the main descriptive economics and the limited R1/R2 results. Correct the economic interpretation before deriving a quoting rule. No evidence here justifies rebuilding the extraction or the entire lifecycle.** The seven failed checks are primarily mismatches between the checker and the frozen reconstruction conventions, not seven demonstrated corruptions of the trading ledger. Their adjudication is below; targeted checks are not a claim that the original checker now passes 39/39.

The independent top-ten sum is **+$35,586.00094**, comprising **+$130,789.27564 matched turnover −$72,443.93423 residual −$22,759.34047 fees**. Maker-leg gross/net is +$7,508.74004; taker gross is +$50,836.60136 before those fees. The all-address net-plus-fee residual is −$1.46e−11. These reproduce the reported rounded totals. All **335 input hashes** recorded by the report match the saved files. Frozen source hashes match after applying both declared amendments.

R1 remains the declared low-contradiction priority screen, not proof that queue priority explains the makers' profit. R2 reproduces **86 filled own orders / 99 fills / 849.867323 shares**, with zero recorded defects in each scenario. Neither result identifies unfilled orders, exact cancellation decisions or a profitable policy for us. Oct 24–27 are four consumed development days; Oct 23 supplies ex-ante identities and Nov 8 supplies the outcome-selected own-order control. No untouched validation was used.

## Consolidated findings, ranked

### 1. P1 — BLOCKING for the quoting conclusion: R4 mixes settlement profit with short-horizon adverse selection

Evidence: `reviews/maker_trace_20261122/report.md:75`, `:79`, `:99`; `tools/claims.py:57`, `:59`, `:66`; `research/pf_mt_econ.py:529`, `:540`, `:548`, `:585`, `:605`.

The depth, sweep and phase figures are **net P&L at resolution per filled share**. The taker row is a **gross 1-second markout**. The headline table does not name this distinction, and “Top-of-book quoting does not pay / lose after adverse selection” goes beyond those computed predicates. On the identical rows with a valid 1-second mark, the review gets:

| Population / role | Shares with valid 1s mark | Gross 1s c/share | Net 1s c/share, fees valued at that mark | Resolution net on these same rows, c/share |
|---|---:|---:|---:|---:|
| Ex-ante top ten, maker at d0 | 10,714,209 | +0.3633 | +0.3633 | −0.1573 |
| Other makers, maker at d0 | 25,198,524 | +0.1599 | +0.1599 | −0.1381 |
| Ex-ante top ten, taker | 3,802,656 | +1.2080 | +0.6009 | +0.7150 |
| Other makers, taker | 12,836,786 | +0.1555 | **−0.4245** | +1.0100 |

These are post-hoc review diagnostics, not a new declared candidate or validation. In particular, the sign difference at d0 remains on the **same cohort**, so it is not explained solely by missing horizon marks. The field's positive gross taker markout does not establish a positive net short-horizon trading edge.

There is also a clock limitation: markouts start at **settlement-ID creation**, not known matching time; outside BOOK, depth is measured against the book at that settlement time. BOOK uses the reconstructed lower endpoint for its spread/depth anchor. Consequently “behind the best” is a historical proxy, not an observed posting decision or proof that deeper quotes would improve our strategy. Sweeps are records with multiple maker prices (`pf_mt_econ.py:555`), not a matched-size causal comparison with single-price trades.

**Required correction:** supersede the interpretation and label horizon, clock, fee convention, cohort and denominator. Retain resolution totals. Present positive short-horizon capture and later inventory/resolution loss separately. Do not infer the value of a whole quoting policy from an individual filled leg's settlement attribution, or treat taker/maker role attribution as independent strategy profit.

### 2. P2 — BLOCKING for the claimed mechanism: “consistency comes from flat inventory” is not measured

Evidence: `report.md:94`, `:108`; `tools/claims.py:89–97`; `research/pf_mt_report.py:940–972`.

The code selects the three addresses positive on all four measurement days and reports their aggregate residual P&L. It does not test flat inventory, inventory turnover speed or a causal explanation for their consistency. A small residual **P&L** is not a small position path. The review's cumulative recorded fills show per-address/day maximum absolute positions of approximately **152–504 shares**, with terminal positions also nonzero. This is recorded trade flow, not independently audited wallet holdings; it is enough to show why the existing aggregate residual cannot certify flat inventory. The three addresses are selected after observing these four days, so their consistency is descriptive.

**Required correction:** say that these three pure makers had positive daily P&L and relatively small aggregate residual contributions. Keep the full ex-ante ten as the primary population in the next attribution. The Bernoulli(END mid) null tests only the residual **END-to-resolution** component under its declared assumptions, not all inventory losses or the absence of trading skill. This finding does not invalidate the ledger totals.

### 3. P2 — NONBLOCKING for R1/R2/R4 totals: reconcile the checker's conventions; do not repair valid data to satisfy false positives

The receipt's suggested diagnoses at `report.md:114–119` are not sufficiently specific. In particular, the capacity checker already uses the hard lower endpoint, and the queue failures do not generally mean useful coverage was understated.

| Failed check | Which side is wrong / evidence | Required disposition |
|---|---|---|
| `posting_capacity_bound`, raw `rp_capacity` | Checker uses **1e−9 share** tolerance (`mt_checks.py:23`, `:509–512`, `:982–989`, `:2015–2029`); lifecycle uses **0.01 share** for rounded displayed sizes (`pf_mt_lifecycle.py:87`, `:484–487`, `:599–608`). In sampled market 3697902, all 25 failing scenario rows / 11 hashes have a capacity deficit no larger than 0.01 share: e.g. 10 executed versus 9.99 displayed. A second example is 0.01 share versus zero. | Honor the display-precision tolerance in the independent and raw checks; retain exact fill accounting separately. Do not tighten inferred posting bounds on rounding noise. Both check paths already use `x_lo_hard_ms`; the receipt's contrary speculation should be removed. |
| `class_scenario_consistency` | The checker treats every numerical difference as informative (`mt_checks.py:1030–1044`). Lifecycle regards a lower bound of zero and no upper bound as trivial (`pf_mt_lifecycle.py:705–707`). Across **all 71 markets**, 18,274 unresolved hashes differ across scenarios: 18,271 are only `(0, NaN)` versus `(NaN, NaN)`; the other three have lower bounds 0.01, 0.0066666 and 0.0033333 shares, within the declared tolerance. | Compare information-bearing bounds, not sentinels or sub-display-tolerance quantities. These rows do not warrant a promotion to meaningful queue coverage. The receipt's “24,165 consistent” is also not the failure count: it is the checker's `sensitive_but_identical` diagnostic. |
| `priority_exhaustion` | Frozen lifecycle explicitly excludes ambiguous tied fills/groups (`pf_mt_lifecycle.py:45–55`, `:533–550`, `:591`). Checker `priority_evidence` ignores that exclusion (`mt_checks.py:520–556`, `:1053–1069`). Every sampled failing exhaustion hash in the four targeted markets has a tied group. | Apply the declared eligibility rule before requiring exhaustion; do not invent an exact original size from an excluded group. Preserve the tie counts and untied positive controls. |
| `priority_ahead_lo` | Same eligibility mismatch (`mt_checks.py:1083–1094` versus lifecycle `:543–550`). All sampled ahead-bound disagreements have tied groups. | Compare first-fill bounds on the same tie-eligible population; retain an explicit ambiguous status. This is not evidence that the builder forgot available certain priority. |
| `priority_tests_T1_T2` | The two extra checker T2 records, in **3734864** and **4353706**, contain tied fills and are excluded by the declared builder rule. Exact record/hash examples are in `probe.json`. The checker counts them, while dividing its displayed rate by the builder's different denominator (`tools/claims.py:33–36`). | Recompute numerator **and denominator** under one eligibility convention. Keep both clearly labelled if desired. The declared 3/15,900 result and low-contradiction R1 reading are not overturned by these two excluded records. |
| `H2_presence` | The three rows are the three scenarios of one **non-own off-grid hash** in GT market 4334454, status `leg_off_grid+subcent_price`. Lifecycle deliberately makes invalid/sub-cent orders unresolved (`pf_mt_lifecycle.py:27–30`, `:393`, `:401`, `:694–699`); checker skips only `identity_conflict` (`mt_checks.py:951–963`). | Check a claimed presence interval only for its declared eligible population, and assert unresolved/status handling for excluded orders. It is conservative withholding, not a false observed span. It does not change the 86-order own score. |
| `tob_from_levels` | For the four sampled BOOK/GT markets, every size discrepancy is an **empty side**: extractor writes size **0**, while checker expects **NaN** because no level exists (`pf_mt_extract.py:859–862`; `mt_checks.py:1113–1128`). There are 39, 34, 44 and 39 such comparisons respectively. Prices are missing consistently; nonempty quantities agree. | Normalize the empty-side size convention, without making the missing price valid or admitting a one-sided mark. No extract or economics rebuild is indicated. |

Scope of verification: all-market queue-label census; exact two T2 disagreement markets and the sole H2 market; four targeted markets for capacity, exhaustion, ahead and top-of-book. The 230-CPU-second full checker/raw reparse was not rerun inside the 180-second reserve. Thus retain formal R3 provisional status until the consolidated checker receipt, but do not call these sampled discrepancies demonstrated bad lifecycle or R4 data. Bundle semantic checker fixes, real counterexamples and positive/negative controls into the next useful packet; no separate repair/review cycle per check.

### 4. P3 — NONBLOCKING for this dataset: receipt-earlier-than-UUID fallback disagrees with the contract

Evidence: `pf_mt_lifecycle.py:264–269` versus `mt_checks.py:886–891` and contract amendment 2. The code uses `st > 0` to choose the bounded lower endpoint and `st <= 0` for the receipt-only flag, although `hi` is `min(st, receipt)`.

A synthetic positive-UUID record received 500ms before its UUID time gets `hi = receipt`, `lo_h = receipt − 2000`, `rcpt=False`. The checker/contract require an unbounded lower end for this receipt-hi case. `controls.json` records the reproduction. The saved `fill_hi_rule` passes, so this is **not demonstrated on the current measured rows**. Add the guard and its fixture when the lifecycle is next touched or before applying it to new inputs; do not rerun current outputs solely for this latent case.

## Checks, provenance and limitations

- `probe.py`: four purposively selected saved markets, covering both extra T2 records, the only H2 market, and the raw-reparse BOOK example; computes failure causes and horizon tables. `probe2.log` / `probe.json` additionally verify that each cited tie occurs at the specific disputed record's timestamp, not merely somewhere in the hash's history. The earlier `probe.log` is retained.
- `supplement.py`: all 71 queue-class populations, empty-side checks, full R2 score reproduction, source hashes with both amendments, and independent address/day sums. Successful run: `supplement3.log` / `supplement.json`.
- `economic_probe.py`: all 335 report input hashes; exact-row markout/fee/resolution comparison on Oct 24–27 and recorded inventory paths. `economic_probe.json` is explicitly post-hoc, not a predeclared experiment.
- `controls.py`: pinned lifecycle **145 checks**, pinned economics **1,010 assertions**, plus the positive timing case and known-bad receipt-hi reproduction. These are synthetic implementation checks, not evidence of profitability.
- The review's first supplement incorrectly omitted amendment-1 hashes; its next assertion treated all trivial queue lower bounds as exact zero. Both failed logs are retained. The final review instrument applies both committed amendments and distinguishes zero from the declared 0.01-share tolerance. No development source or outcome was changed.
- The memory amendment is bound to its new source hashes and preceded the outputs. Its reported **21/21 CSV parity on each of two synthetic worlds** and checker **39/39 / 76 falsifiers** establish instrument parity on those fixtures, not proof of the economic interpretation. I inspected the changed streaming boundaries and verified the final input/source identities; I did not repeat the full-sized report or checker.
- All review runs used `PF_SLOT=codex` and absolute paths through `run_light.sh`. Seven research launches including both failed supplement attempts used **64.23 logged CPU-user seconds**; their summed wall times are **70.56 seconds** (conservative charge **0.021 core-hours**, including the small staging/verification overhead, below the 0.05 reserve). Peak RSS was **915 MiB**. Do not count this charge again if the ledger later imports the `codex_mt_review_*` runs. No full training, new label build, raw-tape experiment, new day, push or live action occurred.

## Largest evidenced limitation

We now have a reconciled **execution/profit ledger**, but the proposed explanation of profitable **resting behavior** is still confounded by valuation horizon, role-coupled inventory and uncertain action timing. Roughly 97% of the top-ten filled orders have no nonempty proven persistence interval in this reconstruction; queue/removal estimates are mostly assumption-sensitive. Agreement of the three scenarios also shares the unverified maximum-2-second matching-to-ID lag premise: it is not an exchange-certified bound. R2 validates only 86 filled own orders under the stated clock assumptions. None of this says useful maker behavior is impossible to infer; it specifies which conclusions still need evidence.

## ONE next priority

**Complete the already authorized TOP-3 DECISION-PATHS packet, with horizon-aligned turnover attribution and honest decision-time identification.** Bundle findings 1–3 and their checker controls in that one development round. Finding 4 can be fixed if the lifecycle is already being touched; it does not require a separate replay. The first useful deliverable is an integrated receipt, not seven fix-only reviews.

Amend the decision-study contract before its new measurements:

1. Use the **Oct 23 ex-ante top three by maker shares**, as approved in `9da9c2e61`, across Oct 24–27. These are **not** the three pure makers selected for positive P&L on all four days. Keep the existing ex-ante top-ten tables as a benchmark; any winning-address subgroup stays explicitly post-hoc.
2. Reconcile chronological FIFO **completed closes**, preserving both entry and closing leg identities/roles and prorated fees. Attribute each matched dollar once. Report maker→maker, maker→taker, taker→maker and taker→taker closes, holding-time distributions and turnover per share beside residual drift, resolution and recorded inventory excursions. On identical executed cohorts, distinguish gross/net **1s / 10s / 60s / END / resolution**, native and common denominators. The primary economic question is whether good completed turnover is subsequently lost through residual exposure, or whether the completed turnover itself is poor.
3. **Do not label fill-relative features as known posting-decision inputs.** The new contract's `posting_window_hi` can be the first fill, long after submission. Taking features at `match_bound − {0,50,250}ms` can also be after the real decision when the match-to-ID lag is longer. Keep an association as assumption-sensitive unless it survives the feasible action-time/observation bounds; show unsupported intervals explicitly. Binance event time is a venue-state reference, not proof of what that maker had received. Preserve multiple compatible histories rather than treating one fitted path as the original policy.
4. The new take-risk grid is selected from first/last realized fills ±30s. Its rate is conditional on that outcome-selected activity span. Either use a predeclared market-clock population for an unconditional action-rate claim, or name the conditional estimand and keep the missing participation problem explicit. A descriptive logistic fit is not evidence that the rule predicts profits or reproduces the maker's latent policy. Include a known-bad synthetic history with the same public fills but different earlier commands, so the instrument must retain uncertainty instead of asserting identification.
5. Correct the seven check semantics together, preserve meaningful counterexamples/refusals, and compare v2 to v1 on unchanged economic rows. Do not alter valid bounds or the accepted ledger merely to satisfy an inapplicable old predicate. Any remaining changes must be tied to material output effects. Keep the v1 receipt immutable.

**No additional USER approval is needed for this authorized packet within the newly recorded 1.85-hour task / 5.25-hour line caps** and the study's declared run/test limits. Include this review's **0.021-hour** conservative charge in the ledger. Freeze material timing/population amendments before their results; use a bounded projection, and request a new decision only if the approved scope/resources cannot cover it. No new day, untouched validation or live use is authorized here. This review did not launch the follow-up or review its not-yet-submitted implementation.
