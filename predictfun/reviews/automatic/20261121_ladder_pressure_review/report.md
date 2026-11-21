# BINANCE-PRESSURE-GUARD-v1 — combined independent review

2026-11-21, Codex. **Target: 40a3ee8d4cac39e98ee065845df63f853544c855**.
Range: `2c5dbd533b3586e30db45220838266982670664b..40a3ee8d4cac39e98ee065845df63f853544c855`.
Reviewed freeze `793c706b7d01ca0fa697d0975ff8d2d79847ab2b` and the results,
using a detached checkout at the target. Development files were not changed.

**Accept as a consumed-day diagnostic. No blocking finding or required rerun.**
The improvement reconciles to the saved fills; it is not a CSV/accounting or
grid-alignment artifact. This is useful evidence for side-specific adverse
movement selection, especially on asks. It is not profitability, prospective
validation, or proof that the full replay gain will survive real execution.

Population as of the receipt at 2026-11-21 19:25Z: **19 complete markets**
(7/5/5/2 on Oct 24/25/26/27), from 96 selected; 54 missing endpoints and 23
stale future books remain excluded with their original statuses. These four
days are consumed adaptive development data. The feature and Oct 24–25 edges
were chosen using this research population. No intervals or new-day claims.

## Evidence and interpretation

| Metric | Original ladder | Pressure guard |
|---|---:|---:|
| END marked value, cents | −209,557.7 | −86,109.6 |
| Filled shares | 96,991.1 | 79,399.5 |
| Common-cutoff +1 s markout, cents | −203,083.9 | −85,164.3 |
| +1 s per valid filled share, cents | −2.102 | −1.077 |
| Mean exact both-side active time | 40.70% | 40.19% |

Paired END improves **+123,448.1 c**, with positive daily deltas
**+31,276.8 / +36,873.9 / +36,125.1 / +19,172.3 c**. Eighteen markets improve;
one worsens. The sum is still negative. +117,919.6 c of improvement is in the
eligible +1 s term, and +64,094 c is already visible at +50 ms. This is not
merely a favorable terminal-inventory mark. It also does not mean the entire
END change is caused by the individual removed fills: the strategy changes
future orders, fills and inventory restrictions.

The matched local screen is supported: 111 selected of 4,831 uncensored live
states; 59 movable strata containing 104 selected states; 500 draws. Selected
mean −97.7 c versus null p05 −58.8 c; no draw is as negative. Bids alone are at
the 13.2th percentile, asks at 0.0%. The frozen pooled `better_selection`
predicate is reproduced. This is a local association on the baseline's sampled
states, not a null distribution for the full feedback policy's P&L, nor a
demonstration of selection on both sides separately. Zero of 500 is an
empirical rank, not a zero probability or a validation p-value.

Reduced participation is real and uneven: Oct 27 both-side activity falls
**39.84% → 20.77%**; Oct 26 ask activity **66.21% → 48.69%**. These decreases
can explain part of the improvement when baseline exposure loses. They do
not create a bookkeeping gain: both-arm marks, quantities, cash and inventory
reconcile, +1 s value per share improves on both sides every day, and the
matched screen removes some obvious exposure confounding. Those observations
support useful selection without identifying how much of +123,448 c is
incremental predictive value. The three largest gains account for 45.17%.

Immediate pressure churn is small: **23 / 4,422** reviewed generations are
pressure-pulled at their first live review; 25 first-review pulls in total.
There are 433 pressure pulls and 1,289 withheld slot proposals, including 980
reducing-side suppressions across the interventions. Withholds are repeated
decision opportunities, not canceled orders. Structural removals remain the
dominant removal mechanism. Do not describe this result as the old 99% instant
model-pull behavior.

## Consolidated actionable findings

1. **P2 — NONBLOCKING for this receipt: the acceptance checker still does not
   cover all participation/inventory readings it claims to cover.**
   `pressure_checks.py:192–194` enumerates reconciled columns without exact
   uptime, inventory, or queue metrics. `:252–255` checks filled-quantity deltas
   but not the reported active-time or inventory deltas. The receipt's claim
   at `report.md:192` that every reading field is recomputed is too broad.
   In-memory reproduction adds **25 percentage points** to every pressure
   arm's `both_exact`, regenerates its matching predicates, and still passes
   **all 22 checks**. The original coordinated markout corruption is now caught.
   **Workaround verified here:** rebuild exact active intervals from each
   order's activation/removal/full fill, and exact inventory integrals from
   all 38 fill ledgers. Every current reported uptime and inventory number
   agrees. Bundle these fields, their reading comparisons, and the coordinated
   uptime refusal into the next checker; no standalone fix/replay cycle.

2. **P3 — NONBLOCKING: queue timestamps and the meaning of “re-entry” need
   precise labels before using them to price lost queue priority.**
   `pf_ladder_pressure.py:243–244` stores `ahead0` beside the submission time.
   However, `pf_branch_engine.py:126–134` initializes that number using the
   venue frame at **activation, submission +58 ms**. It is an activation queue
   snapshot, evaluated eagerly at submission, not the venue queue observable
   at submission. The guard never reads it, so this is not a policy leak.
   Also `pf_ladder_pressure.py:442–448` pairs each pull with the next
   same-side submission, regardless of price or birth rung. Of 430 such
   pairings, **274 change rung and only 57 keep the price**. This is a valid
   next-side-order diagnostic, not a matched replacement or a counterfactual
   queue cost. Keep receipt-known estimates separate; label the activation
   clock, retain both gids/prices/rungs and comparison eligibility, and name
   `q_surrendered_mean` as an ahead difference on these pairs. Same-price
   comparisons still do not establish the queue position an unpulled order
   would have retained. No economic rerun is needed for these label changes.

These are the actionable repair items found across the reviewed boundaries.
The economic and simulation limitations below are not additional implementation
blockers. Defer unrelated hardening and bundle cheap repairs with the next
experiment.

## Timing, implementation and artifact checks

- **Signal clock is coherent under the declared clock model.**
  `pf_step5_features_rl.py:25,78–81,111–119` forms bookTicker availability as
  `E + BN_IN + D_DEC`, then selects the last row no later than `tau`; the four
  clock artifacts specify 1.5 ms and 2 ms. Thus raw `E <= tau − 3.5 ms`, with
  further age recorded by `bn_age`. The past return endpoint uses the same
  availability clock at `tau − 250 ms`. `pf_stop_depth.py:134–143` supplies
  −return for bids and +return for asks; `pf_stop_model.py:1567–1576` leaves
  this feature unnormalized apart from float32. There is no extra `tau + 2`
  future read. Independently compared **226,480** cached side/tick values and
  ages bit-for-bit to filtered, existing step-5 rows at `start_abs_ms + 50*k`;
  all agree, all ages are finite and nonnegative.
- **Clock caveat:** Binance availability is the declared Tokyo median-delay
  model, not the research host's per-message receive timestamp. The builder
  does not read bookTicker `recv` at `:78`. The 64 perturbations work on the
  already-constructed world; they cannot establish real feed-tail latency or
  independently validate the raw collector. No delivered input is later than
  the decision under this model; actual wall-clock delivery is not proven.
- `pressure_cache:163–183` requires an exact grid match before use;
  `PressureGuard:122–153` uses the current side and tick, preserving inventory
  precedence. `pf_branch_ladder.py:162–172,179–227` advances trades/notices,
  applies structural restrictions, updates inventory mode, then alternates
  side commands. Reservations include cancel-pending generations until the
  proper notice. All 433 logged live pulls reconcile to request time and
  **effect = request +37 ms**; intervention positions reproduce fills known
  after +100 ms. All 1,722 guard logs match actual commands. The saved
  cache gives no missing values. Float32 inputs are compared against the
  original double edges; the bid threshold does not round down.
- Disabled parity was independently extended to **commands, decisions,
  venue/client positions, reservations, cash and mid arrays**, as well as
  the ledger/fingerprint/queue controls. All 19 equal the accepted baseline.
  Arm identities, tape ids and common END marks agree.
- Cache checks reproduce 14,968 saved snapshot rows and the side sign. The
  64 causal records have the declared minimum-hash membership, equal cached
  values, unchanged future-perturbed inputs and firing delivered controls.
  The perturbation code and synthetic controls were reviewed/rerun; the 64
  real-world feature computations were not rerun in this routine review.
- Selection strata, without-replacement draws, censor exclusions and all
  saved selection fields reproduce. Nine censored live states, none selected,
  stay outside the statistic; no-fill windows remain zero. No outcome enters
  the matching. This is one sampled live state per generation, not every
  guard intervention and not an entry-selection control.
- Cash, position, END, markouts and denominators reconcile per arm. Every
  fill has its corresponding mids row and lies in [0, END]. The common
  `END−1000 ms` cutoff is applied to both arms. The additive END identity and
  lost/added identity pass. The latter uses the declared multiset
  `(t, side, price, quantity)`: 3,564 lost and 1,753 added fill records.
  Different partial quantities count as unmatched whole records, so gross
  lost/added volume is not a unique quantity-level avoided-loss attribution;
  its net identity is correct.
- All receipt output hashes match raw copies, day artifact hashes match,
  and the frozen research source closure matches the pinned source. The
  pre-freeze probe is timing-only at the inspected code. Pilot rehearsal
  intentionally fails whole-population membership but exercises the other
  21 checks; its generated pilot economics are development consumption, not
  untouched validation. No post-freeze rule amendment was found. The two
  initial mutant escapes and final 29/29 receipt are retained; the full mutant
  queue was inspected, not redundantly rerun.

**Prior findings:** the markout-checker defect is closed for economic readings,
with the residual coverage gap in item 1; queue data are now present, with the
interpretation correction in item 2; isolated run-day refusal tests are fixed
and pass, including an all-pass path reaching the load sentinel. No prior
blocking item remains open.

`reproduce.py` runs only through PF_SLOT=codex and the absolute `run_light.sh`.
It passed the pinned selftest, 22 saved checks, 14 frozen falsifiers, all above
saved-data reconciliations, and the targeted checker-gap reproduction in
**19.48 user CPU seconds**, peak **503 MiB**. An initial review-script attempt
used the wrong identity key (`t_start` instead of `start_abs_ms`), stopped
before reconciliation, and is retained as `reproduce_attempt1.log` (11.26 CPU
seconds). No development output was changed. No real-tape replay, model fit,
full label build, new market or new day was run. Evidence and independent
38-arm metrics are beside this report.

## Largest limitation and ONE next priority

**The largest evidenced economic limitation is still adverse value on the
retained fills: −1.077 c/share at +1 s and −86,109.6 c at END.** The guard
removes a large part of the damage, but it does not yet make this ladder
profitable. Its apparent selection is confined to 19 already-consumed
markets under M50 trade timing and a fast modeled feed.

The unchanged-external-flow assumption can exaggerate the counterfactual
benefit: 100-share quotes, their removal and re-entry do not change public
volume, book behavior or competitors. Fewer quotes in the candidate can make
that approximation affect the two arms differently. Equal baseline parity
and correct accounting do not establish market realism or the sign of this
bias. The available saved evidence does not identify that reaction. No claim
that unchanged flow either caused or cannot cause the gain is justified.

**ONE experiment: PRESSURE-LEAD-SENSITIVITY-v1 — test one additional 50 ms of
signal-availability delay, with the pressure rule otherwise frozen.** This
checks whether the useful signal survives a modest timing margin before
tuning a stronger guard. It is a diagnostic sensitivity, not a fix for an
observed look-ahead bug, a cooldown, or a market-impact validation.

1. Freeze the experiment before output. Same 19 complete markets and all 96
   coverage statuses, paths, marks, M50 trade placement, two levels, nominal
   100 shares, 200/100/50 inventory settings, 50 ms action grid and 58/37/100
   ms execution/notice clocks. Same sides, q5 edges and float32 representation.
   Do not stack the reservation guard or change structural/sizing rules.
2. At action tick k the delayed candidate uses cached signal/status from
   **k−1**, on its own side. Tick 0 is explicit warm-up with no extra guard
   action. Log current decision time, source decision time, source age and
   effective age (+50 ms), value, action/reason, inventory, gid and effect.
   Keep inventory and structural decisions current. Both PULL and withhold
   use this same lag; clearing uses the delayed stream too. No peeking at k
   to choose whether to apply the lag, and no extra timer after clearance.
3. Reuse the hash-bound saved market caches. Rerun the zero-lag control and
   require per-market equality to this receipt's pressure ledger/commands;
   the original disabled baseline may be reused with its identity checks.
   Synthetic controls must exercise an ON crossing, an OFF crossing, initial
   warm-up and wrong-side/one-tick-future refusals. Keep source and output
   identity checks. Bundle findings 1–2 without editing older frozen receipts.
4. Compare original baseline, zero lag and +50 ms per market/day: paired END,
   +50/+100/+1 s markouts and valid quantities, END decomposition, side/rung,
   exact participation/inventory, generations, pull/withhold reasons and
   reducing-side interventions. Report retained gain relative to zero lag
   descriptively; do not pick a new lag or threshold from the outcome.
   On the same baseline live states, join each delayed value by exact market,
   side and decision tick. Repeat the frozen 500-draw matched screen with
   that lagged mask, identical strata/seed/minimum support/censoring; keep
   per-side diagnostics separate. Labels still start after the **current**
   action's cancel effect, not after the source tick's effect.
5. Declare the reading: `survives_50ms_screen` only if delayed-vs-baseline
   END and pooled +1 s/share improve on all four days and the delayed local
   selection screen passes. Otherwise report each failed/thin component and
   `not_demonstrated_at_50ms`; that does not establish that no shorter timing
   margin works. No new tuning, prospective validation or profitability claim.
6. At most **0.50 additional core-hours**, and in all cases within the original
   2-core-hour ladder allocation after checking cumulative logged use. Pilot
   the fixed first two complete markets per date; run the remainder only if
   projection fits. Existing slots and memory limits; explicit stop if not.
   Submit one combined repair/control/result receipt, not serial fix reviews.

**No additional USER permission is needed for this bounded consumed-day
experiment.** New days, untouched validation, increased resources, live
execution or a different market-impact model require a separate USER decision.
The latency sensitivity will leave coverage and external-flow realism open;
do not present it as resolving those limits.
