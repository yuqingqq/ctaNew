# ARRIVAL-QUEUE-DECOMPOSITION-v1 — combined independent review

2026-11-22 UTC, Codex. **Target: 75e9531d21610a3b16f5597153299bc92c210a3a**.
Range: `13a7049159ad482d879832dd3368deeaecb152b6..75e9531d21610a3b16f5597153299bc92c210a3a`.
Reviewed development freeze `29869269` and results at a detached checkout.
The development session owns implementation. This commit contains only review
evidence and review-owned state additions; no push, replay, fit or new day.

**Accept the saved decomposition with the corrections below. No blocking
label/accounting defect or required replay was found.** The four actionable
items are NONBLOCKING for proceeding to the next bounded diagnostic because the
current artifacts were independently verified, the wrong interpretations are
corrected here, and no candidate/skill claim is being accepted. Bundle their
corrections into one next receipt; do not run separate repair rounds.

## What the evidence supports

Population as of the frozen output at 2026-11-22: 4,840 neutral BASELINE live
states, one sampled review per generation, on 19 complete markets from consumed
Oct 24–27. Four days, no intervals or validation. Every state and its four
arrival intervals is present. The 6,992 future-window fills / 74,887.868346 shares
reconcile to the prior audit. There are 249 union windows without all horizons;
4,591 union-common states remain, of which 3,138 have no fill and 1,453 have fills.
The common markouts use **66,010.283129 shares**, not all 74,887.868346.

On that union-common cohort, execution edge is about +0.38 c/share, then
−0.89 at 50 ms, −2.23 at 1 s, −2.45 at 5 s and −2.89 at 30 s. The interval
readings reproduce: I0/I1 partially recover after 1 s, while I2/I3 worsen by
30 s. Later intervals account for about 64% of the union-common net loss at
30 s. These are descriptive marked values under the baseline continuation,
not per-order avoided losses, always-kept fill rates or profitable strategy
evidence. Survival to a later interval is itself policy/path-dependent.

The pooled queue-band ordering and the reported day/side ordering reproduce.
The queue × pressure question is still **unsupported: 0/60 evaluable contrasts**.
Correcting support to common-cohort filled states leaves that answer unchanged.
It also leaves the existing R2/R4 sparse classifications unchanged. This cannot
be read as proof that pressure has no information within queue states.

The client-versus-simulated-venue queue comparison reproduces: 2,903 within one
share, 1,535 client estimates at least one share below, 402 at least one above.
These are first-review diagnostics against the simulator, not exchange truth
or contemporaneous quality measurements at every later sampled live decision.

## Consolidated actionable checklist

1. **P2 — NONBLOCKING: correct the current-occupancy interpretation of Q=0.**
   Receipt `report.md:109–116` calls 343 behind-best front states empty levels
   and, together with 313 inside-best states, describes 656 as alone at their
   price. But `pf_ladder_rest.py:451–470` computes ahead as a running minimum;
   it separately computes the currently displayed size, exported as `fp_shown`
   at `:493`. Zero minimum does not imply current displayed size zero.

   Exact-key joins to the four hash-bound original feature parquets give:
   **458 front states currently have zero displayed size; 331 have positive
   displayed size. Of the 343 behind-best states, 198 have positive displayed
   size.** All 133 at-touch front states are also in the positive-display group.
   The depth counts 313/343/133 are correct; the inference from depth plus Q=0
   to occupancy is not. `evidence.json` and `occupancy_examples.csv` bind and
   illustrate the reproduction. “Empty” here refers only to the received
   external displayed level, not a claim about all venue liquidity.

   **Next-round correction:** use actual `fp_shown` alongside Q and depth;
   supersede the receipt's occupancy count in band. A zero queue estimate at
   an occupied level may reflect prior progress/new arrivals or estimator error;
   this review does not assign its cause. Never equate Q=0 with exact priority.

2. **P2 — NONBLOCKING: display the right cohort denominators and apply support
   screens to the cohort supplying the value.**
   `pf_ladder_arrival_queue.py:319–331` intentionally carries all-state base
   counts into both native/common rows, while horizon values use valid/common
   states. `tools/read_arrival_queue.py:79–87` copies the base counts and uses
   their `any_fill` for `sparse`; `:98–106` does the same for contrasts and
   day support. Receipt `:42–50` and `:72–88` labels these mixed tables as
   common-cohort tables without showing the common shares/filled-state counts.

   | Union queue group | All filled states | Common filled states | All shares | Common shares |
   |---|---:|---:|---:|---:|
   | front | 238 | 129 | 8,475.66 | 4,682.60 |
   | within_R | 705 | 609 | 33,206.73 | 29,435.52 |
   | beyond_R | 759 | 715 | 33,205.48 | 31,892.16 |

   Front's common markout therefore describes a notably narrower subset than
   its displayed fill count. Its I0 count is 75 common filled states rather
   than 128; I1 is 38 rather than 86. The published value formulas themselves
   are correct. `support.csv` provides all current queue/progress/pressure
   counts and verifies zero pooled sparse-classification flips at the declared
   threshold. All 60 pressure comparisons remain insufficient.

   **Next-round correction:** retain explicitly named all-state counts and add
   common/native valid counts, filled counts, shares and rates; screen each
   value on its own filled support. Show common-cohort censoring by queue/price
   state and time to END. Do not multiply common c/share by all shares to infer
   net value. In the same change, fix/rename `read_arrival_queue.py:142`:
   `days_evaluable > 0` plus agreement on evaluable days is not “negative all
   days.” Require all four declared days for that name, or name the supported
   subset explicitly. The current zero-evaluable result is unaffected.

3. **P2 — NONBLOCKING with this review's verified workaround: extend checker
   coverage to the derived values used by the readings.**
   `aq_checks.py:195–205` checks labels and selected state values but omits
   `client_minus_venue`, actual queue values, ratios and several progress
   diagnostics. `:240–246` checks aggregate counts/numerators but not reported
   per-share/per-opportunity rates, any-fill rate, filled-out/status counts or
   all derived reading predicates. Reading output files are not a check family.
   Family R is independently implemented and really reproduces the prior table;
   it does not close those unrelated gaps.

   In-memory probes demonstrate that **zeroing every client-minus-venue value**
   passes the affected F/S checks, and **negating the 1 s per-share column**
   passes C. These are derived-output holes, not failed checks on the original
   data. This review separately re-derived all those current state values,
   denominators, ratios, status counts and filled-out counts: no mismatch.
   It reproduced the saved reading tables and their principal predicates and
   verified the post-run day-order and one-share figures.

   **Next-round correction:** add these reused fields, cohort-specific support
   and reader predicates to the bounded checker, with positive and known-bad
   controls for the two demonstrated corruptions. Verify actual ledger bytes
   against their day-record gzip hash when reusing the checker; its load/A path
   at `:69–76`, `:180–184` currently checks the day-record identity but not those
   gzip bytes. The builder's parent loader does, and this review checked the
   four gzip hashes independently. No current artifact mismatch or rerun.

4. **P3 — NONBLOCKING: an arrival bin does not establish the latest actionable
   signal time.** Receipt `report.md:68` associates all I0 fills with a signal
   already visible at d. I0 is [d+37,d+287), so some of those fills can still
   be prevented by a later 50 ms review. For example, d=1,000, e=1,037 and a
   fill at 1,137 is I0; a request at 1,100 takes effect at 1,137 and excludes
   the fill under the engine's strict removal rule (`pf_branch_engine.py:155–160`).
   The latest feasible live request grid is floor((fill_time−37)/50)×50,
   subject to the actual order/action state. The decomposition measures neither
   when a condition first appears nor whether it is available at that grid.
   Its 22.5% is a share of **net** 1 s loss on the union-common cohort, not a
   percentage of time or a measured timing-unavoidable fraction.

   **Next-round correction:** remove that implication and keep arrival timing,
   signal lead time and cancel-in-flight outcomes separate. A wording correction
   is sufficient; do not launch a fresh timing replay for this item.

## Timing, causality and checks reviewed

- **Effect and arrival:** `effect_of` (`:106–108`) agrees with the accepted
  cancel/pending clocks. `interval_of` (`:131–135`) is left-closed/right-open;
  fills at effect enter I0 and boundary fills enter the next interval.
- **Generation end/follow-up:** `:111–119`, `:139–150`, `:258–273`, `:292–302`
  handle the completing-fill boundary, strict removal, full-fill/END/removal
  tie order and union end cause. The pre-freeze boundary-status repair is
  adopted. No fill after the allowed generation end appears in this dataset.
- **Values and censoring:** `:122–128`, `:264–273` keep missing horizons missing
  and retain fill-positive/negative terms. Common interval cohorts and the
  union-common cohort are different, as declared. Native totals reconcile to
  the audit rather than being confused with common totals.
- **Queue/progress:** zero/unknown and invalid remaining have distinct bands.
  `:186–210` rejects future history and uses the exact saved 250 ms state;
  the unavailable/not-yet-born statuses are retained. No venue value enters
  the condition views. No hidden full-history queue summary is used.
- **Venue/notice:** `:218–231` joins the exact first-review tick and same order;
  the open-lower/closed-upper notice gap matches all 4,840 cases. Recomputed
  live known remaining also matches every state. Actual per-gid queue values
  were verified independently, not inferred from the notice identity alone.
- **Identity/reconciliation:** four parent ledgers and original feature
  parquets match committed record hashes; raw output CSVs match the committed
  compressed copies and audit hashes. Every state, interval, 8,760 cell rows
  and the 1,431-row prior condition table reconcile. All 12 pinned checks and
  12 falsifiers pass. Seven-group synthetic selftest passes; all 37 saved
  mutation records were rejudged against the pinned source/harness/selftest
  hashes, with copied-record rejection. Passing those tests is not model skill.
- **Process:** pre-freeze rehearsals are disclosed and superseded. Day ordering,
  one-share queue differences and depth composition are labelled post-run in
  the receipt. Their numerical verification does not convert them to declared
  validation. The occupancy inference requires finding 1's correction.

The prior horizon/queue-plan acceptance items are implemented, subject to the
cohort and occupancy interpretations above. The residual/ablation deferred
diagnostic limitations are not needed for this decomposition; no claim of their
general repair is made. POSITIVE-PRESSURE-GUARD-v1 remains stopped.

## ONE next priority

**QUEUE-PRICE-STATE-AUDIT-v1:** one small, frozen, saved-data follow-up that
separates queue estimates from current price-level occupancy and distance.
Bundle the four corrections above with its single result receipt.

Use these same neutral states and exact-key, hash-bound original feature rows.
Keep all states. Declare current price-state groups before computing economics:
inside-best, at-touch, and behind-best; keep the exact depth/rung in the exported
rows. Within those groups keep actual `fp_shown == 0`, `>0`, unknown separately,
and the existing three Q/R bands plus unknown/invalid statuses. The principal
queue comparison is **within at-touch occupied states**, by side/day; behind
occupied states are a separate exact-depth diagnostic, not silently pooled.
Do not impose Q<=shown: estimated progress and later displayed arrivals differ.

Reuse the existing arrival intervals and marks, with both native and explicitly
matched common-cohort counts, fill rates, quantities, positive/negative/net
values at 1/5/30 s. Show censoring and remaining time to END, rather than calling
the positive front common markout an all-front result. Keep the declared support
minimum on the actual valued filled states; sparse cells remain sparse. Include
the existing pressure cross-tab as a support census only unless the previously
declared support criterion is met; do not merge bins to manufacture support.

Purpose: decide whether the apparent queue ordering persists at a comparable
occupied price level, or is mostly price-state/coverage composition. Report the
answer as descriptive and separate from model skill. No new thresholds, models,
days, venue-queue decision inputs or replay. If the comparable cells remain
thin, conclude that this saved population cannot answer the question; do not
automatically start another tuned variant. A later request for more data or a
counterfactual queue-value experiment is a separate USER decision.

Keep this whole development batch **<=0.020 additional core-hours**, including
its necessary fixtures/checks, and the line **<=2.0**. Reuse unaffected evidence;
do not rerun every old mutation suite. If projection cannot fit, stop. This
bounded saved-data step **does not need new USER approval**. Extra resources,
new days, untouched validation or live use do. The separate +0.02-ish amendment
for the stopped positive-pressure replay is still the USER's decision.

## Resources and review artifacts

Pinned family I reproduces round 0.076708 h and cumulative **1.942061 h** through
the development receipt. This review used 2.10 CPU s for the selftest, 69.79 for
the complete saved-evidence verification and 0.44 for the final summary, peak
263 MiB. Recorded cumulative use is approximately **1.962153 h**, leaving
**0.037847 h** before small commit verification. The proposed 0.020 h batch
reserves about 0.0178 h for a bounded subsequent review/overhead. All numerical
checks ran through PF_SLOT=codex and absolute run_light/script paths. No busy
slot bypass and no full training/tape experiment.

Evidence: `verify.py`, `evidence.json`, `support.csv`, `occupancy_examples.csv`,
`summary.py`, `summary.json`, the selftest/verification/summary logs, and
`SHA256SUMS`. Other-session changes and existing result artifacts are preserved.

**Largest evidenced limitation:** queue, current price state and horizon
coverage are mixed, and the queue × pressure cells lack support. The accounting
is reliable for the stated neutral population; it has not established whether
the available signals can distinguish good and bad exposure at comparable queue
positions. The single next priority is the bounded queue/price-state separation
above, not a larger model or strategy promotion.
