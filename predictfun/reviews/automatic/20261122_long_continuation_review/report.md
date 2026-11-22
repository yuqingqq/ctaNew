# LONG-CONTINUATION-PAIRED-VALUE-v1 — combined review

2026-11-22. Full pinned target: **6b806c192980de106da80fa97ce4993d6553ec66**.
Range: **4e07b3eae73c7a049dcebb89fd14b03d6da3c805..6b806c192980de106da80fa97ce4993d6553ec66**.
Reviewed freeze `7407bdf1` and results `6b806c19` in detached checkout
`/tmp/pf-lc-review-6b806c19`. Development owns implementation. No historical
replay, fit, new day, implementation change, push or external messaging.

## Decision

**Accept the 64 paired END labels and the declared `reliable_and_varied`
pilot gate. No result-blocking defect or required replay.** The implemented
continuation is common to both arms, and the paired full-book accounting
reconciles. The gate means the pilot produced valid, varied labels; it does
not establish predictability, a useful cancellation rule or profitability.

The report overinterprets its sequence diagnostic and contains a replacement
delay bug and an incorrect day-sign statement. Bundle the corrections and
the saved-data attribution below; do not repeat the pilot or start another
small-fix review cycle.

Population: **64 pairs, 128 arms, eight supported markets on consumed
Oct 24–27**, as of the pinned target. Thirty-two REVIEW and 32 ADMISSION
pairs were selected from the frozen 4,783-state census. These are repeated,
overlapping counterfactual continuations of eight markets, not 64 independent
markets or additive trading returns. Four consumed days: no validation or
confidence interval.

| Kind | Retain / alternative better / ties | Sum Y, c | Sum absolute Y, c |
|---|---:|---:|---:|
| REVIEW | 11 / 8 / 13 | -7,699.27 | 22,634.91 |
| ADMISSION | 4 / 7 / 21 | +4,239.14 | 12,395.77 |

REVIEW's aggregate Y is negative on **all four** sampled days. ADMISSION
is positive on Oct 24–25 and negative on Oct 26–27. The full-book comparison
is now measurable. Its economic mechanism still needs explanation before
using it to build a predictor.

## Consolidated actionable findings

1. **P2 — NONBLOCKING for Y; the zero sequence count cannot test the user's
   early-loss/queue/turnover hypothesis.**

   `predictfun/research/pf_ladder_long_continuation.py:207-213` values every
   fill at the same END mark and then tests for a negative first fill group
   followed by a positive sum of later same-side lineage fills.
   For one fixed-price generation, every contribution is
   `side_sign * quantity * (END_mark - order_price)`: its sign cannot reverse
   between fills. Every saved generation indeed has one side and price.
   A later replacement can change the lineage's price and sign, but this is
   not a measurement of good and bad flow reaching the *same retained order*.
   Opposite-side spread capture is outside this sequence by construction.

   Consequently `report.md:40-47,85` correctly counts zero for its declared
   predicate, but this provides no negative evidence about early adverse
   fills followed by profitable two-sided turnover. The positive fixture at
   `pf_ladder_long_continuation_selftest.py:177-179` even changes one gid's
   price from 51 to 47; that is not a possible order in this engine. Replace
   it with realizable fixed-price gids and an opposite-side close when the
   diagnostic is extended. The primary END target remains unchanged.

   **Review control:** two 10-share buys at 50 followed by a 20-share sell at
   51 have full-book END value +20 c even with END at 49. This same-side
   sequence labels the buys `early_loss_no_later_benefit`. This is a
   synthetic accounting example, not evidence of realized queue necessity.

   This limitation follows from the END-signed sequence requested in the
   earlier review; it is a correction to that diagnostic design, not a claim
   that development ignored the contract. Keep END Y as the economic target.
   Explain compensation using actual opposing fills and matched prices,
   while keeping short-horizon adverse markouts separately named. Do not
   promote either sequence to a causal queue-necessity label.

2. **P2 — NONBLOCKING for labels; first replacement delay uses a factual gid
   across different branches. The reported timing must be superseded.**

   `pf_ladder_long_continuation.py:256` excludes every replacement whose gid
   equals `st['gid']`. After ADMISSION WAIT, the first subsequent order can
   reuse that numeric id because the original POST never happened. It is
   still a new order in that branch. The filter skips it and reports a later
   replacement's delay.

   This affects **12 of 32 ADMISSION alternative arms**: their actual first
   lineage submission is at **+50 ms**, while the exported delay ranges from
   100 to 9,800 ms. Across the 63 alternative arms with a replacement, the
   mean is **869.84 ms**, rather than **1,361.11 ms** (`report.md:62`).
   Exact keys and corrections are in `replacement_delay_corrections.csv`.
   The engine's queue, fills, lineage membership and Y are unaffected.

   **Fix in the same follow-up receipt:** identify the branch's actual focal
   submission separately from subsequent lineage orders. In the WAIT arm,
   the first later submission is the first replacement regardless of its
   numeric gid. Distinguish submission attempt from accepted activation;
   the current metric includes refused submissions. Recompute the exported
   field and summary from saved replacement rows; no replay is necessary.

   Related interpretation: `:164-175,231` opens all carried inventory at the
   fork for FIFO timing. **126 of 128 arms carry inventory.** The reported
   7.7 s is the implemented mean holding duration with carried lots' ages
   reset, not their full acquisition-to-disposal age. Preserve original
   acquisition times from saved `pre_fills` in the turnover follow-up, and
   report carried versus newly acquired inventory separately.

3. **P2 — NONBLOCKING; pooled cancellation of components is being mistaken
   for economic equivalence and general dominance.**

   `report.md:38,81` says neither kind has one sign every day. Its own table
   and saved labels show REVIEW Y = **-544.06 / -1,896.84 / -2,562.82 /
   -2,695.55 c**. Only ADMISSION has mixed day signs.

   `:79-83` calls the comparison a “wash” and says most Y flows through other
   orders. Equal positive/negative counts, median zero and a pooled net
   do not establish equivalence; the sample mixes different actions and
   overlapping paths. The net REVIEW split (+656 c focal, -8,355 c other)
   is correct, but does not describe the size of per-pair contributions:

   | Kind | Sum abs(Y_focal), c | Sum abs(Y_other), c | Other component larger in absolute value |
   |---|---:|---:|---:|
   | REVIEW | 47,594.56 | 34,515.45 | 10 / 32 pairs |
   | ADMISSION | 51,578.82 | 53,254.79 | 6 / 32 pairs |

   Both components can be large and offset each other. Use “coupled,
   offsetting contributions; REVIEW favours the alternative in aggregate
   on these sampled days” rather than a general claim that the focal order
   hardly matters. These absolute-component figures are review diagnostics,
   not a new selection screen or a causal importance estimator.

   The reported positive fill-time edge versus negative inventory movement
   can describe the whole continuation, but the starting inventory's common
   movement term cancels in Y. Neither the negative absolute Q nor an
   offsetting component alone identifies the action's loss mechanism.
   Have the report derive its day-sign statements from the table; no new
   experiment is needed to correct them.

4. **P3 — NONBLOCKING with the completed checks below; narrow the claimed
   independent coverage and strengthen these checks when reused.**

   `lc_checks.py:164-165` checks callback count and matching context hashes,
   but not the expected action/time/gid/rung. A changed callback action or
   time still passes `G_gates`. `:193-210` does not recompute the arm metrics
   in the summary; a changed `retain_edge` passes `S_summary`. These are
   semantic-family escapes, not a claim that unchanged hashes accept edited
   files. The actual 128 callback tuples match the manifest and requested
   actions, and the actual summary reproduces fully from the exported arms.

   Saved paired ledgers at `pf_ladder_long_continuation.py:353-359` omit the
   complete arm orders, requests, commands, guard logs and fill-time mids.
   Thus the checker cannot independently reconstruct every participation,
   churn or markout metric claimed in its “everything” wording. Actual Y,
   cash/inventory, exact callback identity, replacement books and the
   inherited unchanged engine semantics have bounded checks; no evidence
   of wrong current Y was found. Full arm trace parity is executed by the
   runner, not independently reconstructed from a retained full trace here.
   Also, the future-only selftest at `selftest.py:154-163` compares fills,
   despite its header claiming commands too. The review's small synthetic
   supplement checks both commands and fills and passes, with a changed-
   future positive control. Same-action REVIEW/ADMISSION parity also passes.

   Prospective validity guard: `:202-204,269` accepts a missing fork mark as
   reconciled and falls back to post-fill Y with status `ok`. The review's
   synthetic missing-mark control demonstrates this. **All real fork and
   terminal marks are finite and common**, and all 5,270 accepted replacement
   records have activation-queue evidence; these permissive paths do not
   affect this pilot. Mark missing inputs explicitly before expanding the
   population. Preserve full arm command/order provenance and required
   marks when a new replay is already justified. Do not rerun these markets
   merely to improve ancillary logs.

## Prior findings, verification and limits

Review `32d102bd` findings 1–2 are implemented: the focal (side, birth-rung)
lineage gets the same pressure-pull exemption in both arms, while withholding,
structural and inventory rules remain active. New gids receive new queues;
other slots use their own branch's fills and reservations. The label axes
are separate and exhaustive, with the sequence-design limitation above.
The prior headroom wording was superseded in `4e07b3ea`; it does not need
another numerical run. Earlier ancillary checker gaps remain scoped rather
than triggering a cleanup cycle.

At `:332-341`, the original roll-in is kept separate from deep-copied arm
states, with read-only tape/cache shared. Both copies replay the tick's
common prefix; the forced callback is identified after trades, notices and
earlier commands. There are no same-tick new fills after a command under the
10 ms activation profile, so the `<= fork time` fill prefix is consistent.
All real pre-fork fills equal the factual record's prefix. Engine inventory
is used for marked accounting, client-known state for policy decisions.

Bounded verification, all through `PF_SLOT=codex` with absolute paths:

- Six-group synthetic selftest passes; 35 final mutant records re-judged
  against final source identities. No full mutant suite rerun. The earlier
  failed mutation attempts remain provenance, and the rehearsal is synthetic.
- Families A/B/Y/G/P/S/I and all 12 falsifiers pass on the raw saved outputs.
  Frozen files are unchanged since `7407bdf1`; all 96 imported research
  sources match the pinned target. Parent manifest, pilot gzip/npz files,
  data-day records, committed day/parity copies and measurement copies agree.
- Census, selection and exact membership reproduce. All 64 Y values equal
  both the independent post-fork decomposition and retain END minus
  alternative END. Both absolute Qs reconcile; canonical terminal marks,
  finite common starting marks and prefix fills are checked on all 128 arms.
- All actual callback tuples match their manifest action/time/order or slot;
  all accepted replacement queues agree with their saved activation books.
  The review reproduces inventory paths, FIFO quantities under the declared
  reset convention, replacement rows, and every summary field from its
  constituent rows. Independent raw-book reloads and real paired replay
  were not performed; fill-time markout/participation verification is limited
  by the retained evidence noted in finding 4.
- Additional synthetic same-action forks and future-only perturbations
  preserve commands and fills before the perturbation and change future
  fills, as expected. These are implementation controls, not model skill.

Measured review checks: **9.76 + 1.07 + 0.98 = 11.81 CPU seconds**, peak
282 MiB, within the **108 s / 0.03 h** reserve. Staging/commit verification
is also within that reserve. Development ledger reproduces **0.223522 h**
for the batch and **2.656311 h** cumulative before this review; full reserved
line **2.686311 h**, below the approved 2.85. The old 3.01 s classification
is not charged again.

## ONE next priority — saved paired turnover and residual-inventory attribution

**Complete this pilot with PAIRED-TURNOVER-ATTRIBUTION-v1 on the same saved
64 pairs. Bundle the above measurement/wording corrections with that one
declared accounting diagnostic. No new replay, label day, model or threshold
variant.** This is the missing explanation before designing a predictor:
does retaining a queue improve completed spread turnover, and how much of
its benefit or loss instead remains in inventory and later book changes?

Freeze the builder, predicates, output schema and small controls before
reading the new attribution. Use every existing pair and status; retain the
original primary Y, rule, sampling and END. The deliverable must:

1. Reconstruct FIFO lots from the complete saved pre/post fills, keeping
   each lot's actual side, price, quantity, acquisition time and arm-local
   gid. At the fork, separate carried lots from new inventory. Use the fork
   mark as the carried lots' incremental value basis, while preserving their
   acquisition times; do not charge pre-fork losses again.
2. Reconcile each Q exactly as **matched closing profit + marked remaining
   inventory value relative to its basis**. Then reconcile paired Y as the
   differences of those two components. Report matched quantities/prices,
   carried versus newly opened exposure, actual holding times, remaining
   inventory and focal-lineage versus other-order participation. FIFO is a
   declared allocation convention, not proof of causal necessity.
3. Link focal fills to their opposite-side offsets and show whether their
   completed round trips earn spread. Keep own-order END signs separate
   from round-trip value. A negative own-order END mark with a profitable
   opposite close must not be called an uncompensated loss. Do not invent
   unavailable fill-time or short-horizon marks; those diagnostics remain
   unknown where the retained evidence does not supply them.
4. Reconcile retained versus alternative fill differences by execution
   event/time, side, price and quantity, not shared gid numbers. Report the
   full distribution and day/side/kind counts beside sums of absolute and
   net components. Never add overlapping pair outcomes as portfolio P&L.
   Keep queue mechanism `unidentified` unless the evidence actually separates
   queue from price, absence, sizing and inventory feedback.
5. Include hand controls for profitable buy/sell turnover despite a negative
   buy END mark, a loss-making round trip, residual directional inventory,
   pre-fork carried inventory closed later, partial closes, and gid reuse
   after WAIT. Include a corrupted decomposition refusal and exact unchanged
   Y/control-output checks. Correct the twelve replacement delays from
   saved rows and derive the report's day signs automatically.

The resulting table should say how much paired value comes from completed
turnover versus residual exposure, per kind and day, with counts and
concentration. It should not declare a winner or learn a risk rule. The
present 15/15 split is enough for a pilot variation gate, not enough to
train or validate a high-dimensional value predictor.

**USER decision:** none is needed for this saved-evidence completion of the
approved pilot, provided its total follow-up costs **at most 0.06 core-hours
including a 0.02 h review reserve**, with the existing **0.40 h batch / 2.85 h
line ceilings unchanged**. Even charging this review's full reserve, that
would be at most 0.313522 h for the batch and 2.746311 h for the line.
The development session should stop if it cannot fit. New simulations,
data/label days, fitting, untouched validation or additional resources need
an explicit scope/resource decision; no such work is authorized here.

**Largest evidenced limitation:** the primary paired economics are usable,
but the current one-sided END-sign sequence cannot measure the user's
two-sided compensation mechanism. Large offsetting book contributions make
the net alone a poor explanation. Resolve the turnover-versus-residual
inventory accounting on these saved pairs before another model experiment.
